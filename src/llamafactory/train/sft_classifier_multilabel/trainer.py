import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import csv
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from datetime import datetime
if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)

from dataclasses import dataclass, field
import torchmetrics
import torch.nn as nn
import numpy as np
import time
import math
from transformers.trainer_utils import speed_metrics, EvalLoopOutput, EvalPrediction, denumpify_detensorize
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_numpify, IterableDatasetShard
from collections.abc import Mapping
from pathlib import Path
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer import has_length
from transformers.trainer import is_sagemaker_mp_enabled

import torch.nn.functional as F
# 定义 AsymmetricLoss
class AsymmetricLoss(nn.Module):
    def __init__(self, beta=0.8, gamma_pos=4.0, gamma_neg=1.0, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.beta = beta
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        log_probs = F.logsigmoid(inputs)
        log_neg_probs = F.logsigmoid(-inputs)
        pos_loss = -targets * log_probs
        neg_loss = -self.beta * (1 - targets) * log_neg_probs
        pt_pos = torch.exp(-pos_loss)
        pt_neg = torch.exp(-neg_loss)
        focal_pos = (1 - pt_pos) ** self.gamma_pos * pos_loss
        focal_neg = (1 - pt_neg) ** self.gamma_neg * neg_loss
        loss = focal_pos + focal_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.25, 0.1, 0.25, 0.75], gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha = self.alpha.to(inputs.device).unsqueeze(0).expand(inputs.size(0), -1)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CustomSeq2SeqRegressionTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics for multi-class classification.
    """
    
    num_labels = 5
    # 实例化时会根据 output_dim 重新创建 metrics
    ACCURACY = torchmetrics.Accuracy(task='multilabel', num_labels=num_labels)
    AUROC = torchmetrics.AUROC(task='multilabel', num_labels=num_labels)
    PRECISION = torchmetrics.Precision(task='multilabel', num_labels=num_labels, average='macro')
    RECALL = torchmetrics.Recall(task='multilabel', num_labels=num_labels, average='macro')
    F1Score = torchmetrics.F1Score(task='multilabel', num_labels=num_labels, average='macro')

    lm_loss_after_each_logging = []
    classification_loss_after_each_logging = []

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        for key, value in finetuning_args.__dict__.items():
            if key not in self.args.__dict__:
                setattr(self.args, key, value)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.classifier_config = {
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.finetuning_args.num_layers,
            "proj_dim": self.finetuning_args.proj_dim,
            "output_dim": self.finetuning_args.output_dim,  # Must be 3 for multi-class
            "input_dropout": self.finetuning_args.input_dropout,
            "dropout": self.finetuning_args.dropout
        }
        self.unfreeze_classifier()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # 根据输出维度动态设置标签数与名称
        self.num_labels = getattr(self.finetuning_args, "output_dim", 4) or 4
        if self.num_labels == 5:
            self.label_names_custom = ["misogynous", "shaming", "stereotype", "objectification", "violence"]
        else:
            self.label_names_custom = ["humour", "sarcasm", "offensive", "motivational"]
        # 重新实例化 metrics 以匹配标签数
        self.ACCURACY = torchmetrics.Accuracy(task='multilabel', num_labels=self.num_labels)
        self.AUROC = torchmetrics.AUROC(task='multilabel', num_labels=self.num_labels)
        self.PRECISION = torchmetrics.Precision(task='multilabel', num_labels=self.num_labels, average='macro')
        self.RECALL = torchmetrics.Recall(task='multilabel', num_labels=self.num_labels, average='macro')
        self.F1Score = torchmetrics.F1Score(task='multilabel', num_labels=self.num_labels, average='macro')



    def unfreeze_classifier(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            classifier_parameters = [
                name for name, _ in opt_model.named_parameters() if "classifier" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in classifier_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in classifier_parameters)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in classifier_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.classifier_lr
                },
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer
    

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, eval_mode=False):
        gen_labels = inputs.get('labels')



        classification_labels = inputs.pop('classification_labels')  
        


               # 验证并转换为多标签张量
        target_labels = []
        for label in classification_labels:
            if not isinstance(label, list) or len(label) != self.num_labels or not all(bit in [0, 1] for bit in label):
                logger.error(f"Invalid classification_label: {label}, expected a list of {self.num_labels} binary values")
                raise ValueError(f"Invalid classification_label: {label}")
            target_labels.append(label)
        target_labels = torch.tensor(target_labels, dtype=torch.float).to(model.device)

        # print("###2 target_labels:")
        # print(target_labels)


        outputs = model(**inputs, output_hidden_states=True)

        lm_loss = outputs["loss"]
        _, pred = self.get_embeds_from_last_layer(gen_labels, outputs)
        classification_loss = self.loss_fn(pred, target_labels)

        loss = lm_loss * self.args.loss_ratio[0] + classification_loss * self.args.loss_ratio[1]
        
        if not eval_mode:
            self.lm_loss_after_each_logging.append(lm_loss.detach().cpu().item())
            self.classification_loss_after_each_logging.append(classification_loss.detach().cpu().item())
        
        if not eval_mode:
            return (loss, outputs) if return_outputs else loss
        else:
            return (lm_loss, classification_loss), outputs, target_labels, pred

    def get_embeds_from_last_layer(self, labels, output, infer_mode=False, output_embeds=False):
        device = torch.cuda.current_device()
        if not infer_mode:
            last_negative_indices = (labels == -100).nonzero(as_tuple=False)
            batch_size = labels.size(0)
            last_negative_per_batch = [
                last_negative_indices[last_negative_indices[:, 0] == i, 1].max().item()
                for i in range(batch_size)
            ]
            last_negative_tensor = torch.tensor(last_negative_per_batch, device=device)
            
            if self.args.embed_layer == "last":
                hidden_state = output.hidden_states[-1]
            elif self.args.embed_layer == "penultimate":
                hidden_state = output.hidden_states[-2]

            if self.args.embed_mode == "last_token":
                x = hidden_state[torch.arange(batch_size, device=device), last_negative_tensor, :]
            elif self.args.embed_mode == "pool":
                seq_len = labels.size(1)
                mask = torch.arange(seq_len, device=device).unsqueeze(0) <= last_negative_tensor.unsqueeze(1)
                masked_hidden_states = hidden_state * mask.unsqueeze(-1)
                token_counts = mask.sum(dim=1).unsqueeze(1)
                x = masked_hidden_states.sum(dim=1) / token_counts
        else:
            if self.args.embed_layer == "last":
                hidden_state = output.hidden_states[-1]
            elif self.args.embed_layer == "penultimate":
                hidden_state = output.hidden_states[-2]
            if self.args.embed_mode == "last_token":
                x = hidden_state[:, -1, :]
            elif self.args.embed_mode == "pool":
                x = hidden_state.mean(dim=1)
        
        if not output_embeds:
            # Align input dtype with classifier weight dtype to avoid BF16/FP32 mismatch
            try:
                clf_dtype = next(self.model.classifier.parameters()).dtype
                if x.dtype != clf_dtype:
                    x = x.to(dtype=clf_dtype)
            except Exception:
                pass
            x = self.model.classifier(x)
            return output, x
        else:
            try:
                clf_dtype = next(self.model.classifier.parameters()).dtype
                if x.dtype != clf_dtype:
                    x = x.to(dtype=clf_dtype)
            except Exception:
                pass
            x, embed = self.model.classifier(x, return_embed=True)
            return output, x, embed

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor
        return padded_tensor.contiguous()

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        if not self.is_world_process_zero():
            return
        preds = predict_results.predictions
        if preds is None:
            logger.warning("No predictions to save.")
            return

        preds_array = np.asarray(preds)
        # Multi-label classifier path: predictions are float logits.
        if preds_array.dtype.kind == "f":
            pred_probs = torch.sigmoid(torch.tensor(preds_array, dtype=torch.float)).numpy()
            pred_labels = (pred_probs > 0.5).astype(int)

            labels = predict_results.label_ids
            if labels is None:
                labels_array = np.full_like(pred_labels, -1)
            else:
                labels_array = np.asarray(labels)

            input_texts: List[str] = []
            if hasattr(dataset, "text"):
                input_texts = dataset.text
            elif hasattr(dataset, "input_text"):
                input_texts = dataset.input_text
            else:
                input_ids = None
                if hasattr(dataset, "input_ids"):
                    input_ids = dataset.input_ids
                elif isinstance(dataset, Mapping) and "input_ids" in dataset:
                    input_ids = dataset["input_ids"]
                if input_ids is not None:
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.cpu().numpy()
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            label_names = getattr(
                self, "label_names_custom", [f"label_{i}" for i in range(pred_probs.shape[1])]
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.args.output_dir, f"predictions_{timestamp}.csv")
            with open(output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["input_text"]
                for name in label_names:
                    header.extend([f"label_{name}", f"pred_prob_{name}", f"pred_label_{name}"])
                header.extend(["all_labels", "all_preds"])
                writer.writerow(header)

                for i in range(len(pred_probs)):
                    row = [input_texts[i] if i < len(input_texts) else ""]
                    for j in range(pred_probs.shape[1]):
                        label_val = -1
                        if labels_array is not None and labels_array.shape[0] > i and labels_array.shape[1] > j:
                            label_val = labels_array[i][j]
                        row.extend([label_val, pred_probs[i][j], pred_labels[i][j]])
                    row.append(str(labels_array[i].tolist() if labels_array is not None else []))
                    row.append(str(pred_labels[i].tolist()))
                    writer.writerow(row)
            logger.info(f"预测结果已保存到 {output_file}")
            return

        # Generative path fallback (not typical for classifier runs).
        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )
        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)
        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(self.args, "ignore_keys", None)
        with torch.no_grad():
            if has_labels:
                (lm_loss, classification_loss), _, target_labels, preds = self.compute_loss(
                    model, inputs, return_outputs=True, eval_mode=True)
                lm_loss = lm_loss.mean().detach()
                classification_loss = classification_loss.mean().detach()
            else:
                outputs = model(**inputs, classification_mode=True, output_hidden_states=True)
                loss = None
        if has_labels:
            return (lm_loss, classification_loss), preds, target_labels
        else:
            return None, preds, target_labels

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    #     has_labels = all(inputs.get(k) is not None for k in self.label_names)
    #     inputs = self._prepare_inputs(inputs)
    #     if ignore_keys is None:
    #         ignore_keys = getattr(self.args, "ignore_keys", None)
    #     with torch.no_grad():
    #         if has_labels:
    #             (lm_loss, classification_loss), outputs, target_labels, preds = self.compute_loss(
    #                 model, inputs, return_outputs=True, eval_mode=True)
    #             lm_loss = lm_loss.mean().detach()
    #             classification_loss = classification_loss.mean().detach()
    
    #         else:
    #             outputs = model(**inputs, classification_mode=True, output_hidden_states=True)
    #             lm_logits = outputs.logits
    #             lm_preds = torch.argmax(lm_logits, dim=-1)
    #             decoded_lm_preds = self.tokenizer.batch_decode(lm_preds, skip_special_tokens=True)
    #             decoded_lm_labels = None
    #             loss = None
    #     if has_labels:
    #         return (lm_loss, classification_loss), preds, target_labels, decoded_lm_preds, decoded_lm_labels
    #     else:
    #         return None, preds, target_labels, decoded_lm_preds, decoded_lm_labels
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs
    ) -> Dict[str, float]:
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        # 保存预测结果到 CSV
        # if self.is_world_process_zero():
        #     self.save_predictions_to_csv(eval_dataset, output)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics
    
        
    def compute_metrics_custom(self, eval_pred):
        if self.args.task == "meme_classification":
            logits, labels = eval_pred
            # Ensure logits and labels are tensors
            logits = torch.tensor(logits, dtype=torch.float) if not isinstance(logits, torch.Tensor) else logits
            labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()
            preds = (torch.sigmoid(logits) > 0.5).float()  # 使用 sigmoid 和阈值 0.5
            acc = self.ACCURACY(preds, labels)
            roc = self.AUROC(logits, labels)
            pre = self.PRECISION(preds, labels)
            recall = self.RECALL(preds, labels)
            f1 = self.F1Score(preds, labels)
            print(
                f"Accuracy: {acc:.4f}, AUROC: {roc:.4f}, Precision: {pre:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            return {"accuracy": acc, "auroc": roc, "precision": pre, "recall": recall, "f1": f1}
        else:
            return super().compute_metrics(eval_pred)
        
    def save_predictions_to_csv(self, dataset, output: EvalLoopOutput):
            if not self.is_world_process_zero():
                return
            
            # 准备输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.args.output_dir, f"predictions_{timestamp}.csv")
            
            # 获取预测结果和标签
            predictions = output.predictions
            labels = output.label_ids
            
            # 确保我们有预测和标签
            if predictions is None or labels is None:
                logger.warning("无法保存预测结果到CSV，因为预测或标签为空")
                return
            
            # 转换预测结果为概率
            pred_probs = torch.sigmoid(torch.tensor(predictions)).numpy()
            pred_labels = (pred_probs > 0.5).astype(int)
            
            # 获取输入文本
            input_texts = []
            if hasattr(dataset, "text"):
                input_texts = dataset.text
            elif hasattr(dataset, "input_text"):
                input_texts = dataset.input_text
            else:
                # 如果没有直接的文本字段，尝试解码input_ids
                if hasattr(dataset, "input_ids"):
                    input_ids = dataset.input_ids
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.cpu().numpy()
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # 写入CSV文件
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                header = ["input_text"]
                for name in self.label_names_custom:
                    header.extend([
                        f"label_{name}", 
                        f"pred_prob_{name}", 
                        f"pred_label_{name}"
                    ])
                header.extend(["all_labels", "all_preds"])
                writer.writerow(header)
                
                # 写入每一行数据
                for i in range(len(pred_probs)):
                    row = [input_texts[i] if i < len(input_texts) else ""]
                    
                    for j in range(len(self.label_names_custom)):
                        row.extend([
                            labels[i][j] if len(labels) > i and len(labels[i]) > j else -1,
                            pred_probs[i][j],
                            pred_labels[i][j]
                        ])
                    
                    # 添加所有标签和预测的字符串表示
                    row.append(str(labels[i].tolist() if hasattr(labels[i], 'tolist') else labels[i]))
                    row.append(str(pred_labels[i].tolist()))
                    
                    writer.writerow(row)
            
            logger.info(f"预测结果已保存到 {output_file}")

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args
        prediction_loss_only = False
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            if model is not self.model:
                self.model_wrapped = model
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size
        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)
        if args.past_index >= 0:
            self._past = None
        losses_host = None
        lm_losses_host = None
        classification_losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        all_losses = None
        all_lm_losses = None
        all_ce_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        observed_num_examples = 0
        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size
            (lm_loss, classification_loss), logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(
                inputs["input_ids"]) if args.include_inputs_for_metrics else None
            loss = lm_loss + classification_loss
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(
                    losses_host, losses, padding_index=-100)
            if lm_loss is not None:
                lm_losses = self.accelerator.gather_for_metrics((lm_loss.repeat(batch_size)))
                lm_losses_host = lm_losses if lm_losses_host is None else nested_concat(
                    lm_losses_host, lm_losses, padding_index=-100)
            if classification_loss is not None:
                classification_losses = self.accelerator.gather_for_metrics(
                    (classification_loss.repeat(batch_size)))
                classification_losses_host = classification_losses if classification_losses_host is None else nested_concat(
                    classification_losses_host, classification_losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control)
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if lm_losses_host is not None:
                    lm_losses = nested_numpify(lm_losses_host)
                    all_lm_losses = lm_losses if all_lm_losses is None else np.concatenate((all_lm_losses, lm_losses), axis=0)
                if classification_losses_host is not None:
                    classification_losses = nested_numpify(classification_losses_host)
                    all_ce_losses = classification_losses if all_ce_losses is None else np.concatenate(
                        (all_ce_losses, classification_losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(
                        all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                lm_losses_host, classification_losses_host = None, None
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if lm_losses_host is not None:
            lm_losses = nested_numpify(lm_losses_host)
            all_lm_losses = lm_losses if all_lm_losses is None else np.concatenate((all_lm_losses, lm_losses), axis=0)
        if classification_losses_host is not None:
            classification_losses = nested_numpify(classification_losses_host)
            all_ce_losses = classification_losses if all_ce_losses is None else np.concatenate(
                (all_ce_losses, classification_losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        if self.compute_metrics_custom is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics_custom(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics_custom(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        metrics = denumpify_detensorize(metrics)
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if all_lm_losses is not None:
            metrics[f"{metric_key_prefix}_lm_loss"] = all_lm_losses.mean().item()
        if all_ce_losses is not None:
            metrics[f"{metric_key_prefix}_classification_loss"] = all_ce_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        output = EvalLoopOutput(
            predictions=all_preds, 
            label_ids=all_labels, 
            metrics=metrics, 
            num_samples=num_samples
        )
        # 保存预测结果到CSV
        if self.is_world_process_zero():
            self.save_predictions_to_csv(eval_dataset, output)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            lm_loss_after_each_logging = np.mean(self.lm_loss_after_each_logging)
            classification_loss_after_each_logging = np.mean(self.classification_loss_after_each_logging)
            self.lm_loss_after_each_logging = []
            self.classification_loss_after_each_logging = []
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["lm_loss"] = round(lm_loss_after_each_logging, 4)
            logs["classification_loss"] = round(classification_loss_after_each_logging, 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir)
        if hasattr(self.model, "classifier"):
            classifier_path = os.path.join(output_dir, "classifier.bin")
            torch.save(self.model.classifier.state_dict(), classifier_path)
        if hasattr(self, "classifier_config"):
            config_path = os.path.join(output_dir, "classifier_config.json")
            with open(config_path, "w") as f:
                json.dump(self.classifier_config, f, indent=4)
