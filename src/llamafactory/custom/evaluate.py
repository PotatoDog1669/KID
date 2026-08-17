import torch
import torch.nn as nn
import json
import os
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from PIL import Image
from llamafactory.custom.hm_inference import QWen2Classifier, Classifier, predict_and_eval, eval_metrics  # Replace with actual import path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset for .pt files
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_file):
        try:
            data = torch.load(feature_file, weights_only=False)
            assert 'ids' in data and 'feats' in data and 'labels' in data, "Feature file missing required keys"
            self.ids = data['ids']
            self.features = data['feats']  # Shape: [num_samples, hidden_size]
            self.labels = data['labels']   # Shape: [num_samples]
            assert len(self.features) == len(self.labels) == len(self.ids), "Inconsistent lengths"
            logger.info(f"Loaded dataset from {feature_file}: {len(self.labels)} samples")
        except Exception as e:
            raise RuntimeError(f"Failed to load feature file {feature_file}: {str(e)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (None, None, self.labels[idx], self.ids[idx], self.features[idx])  # Mimic split_dl format

# Dataset for .jsonl files
class JsonlDataset(Dataset):
    def __init__(self, jsonl_file, image_dir, prompt="Is this a hateful meme?"):
        self.data = []
        self.image_dir = image_dir
        self.prompt = prompt
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        logger.info(f"Loaded dataset from {jsonl_file}: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item['img'])
        image = Image.open(image_path).convert('RGB')
        text = item.get('text', '')
        label = item.get('label', 0)  # Default to 0 if no label
        return (image, text, label, item['id'], None)  # Mimic split_dl format

# Evaluation for .pt files using Classifier and eval_metrics
def evaluate_pt_file(classifier, data_file, split_name, device, batch_size=64, use_wandb=False):
    classifier.eval()
    dataset = PrecomputedFeatureDataset(data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_logits = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            _, _, labels, ids, features = batch
            features = features.to(device).float()
            labels = labels.to(device).float()
            logits = classifier(features).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_ids.extend(ids)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Call eval_metrics
    acc, auroc, precision, recall, f1 = eval_metrics(
        dataset="FB",
        labels=all_labels,
        predicted=all_logits,
        name=split_name,
        compute_loss=False,
        apply_sigmoid=True
    )

    metrics = {
        f"{split_name}_accuracy": acc,
        f"{split_name}_auroc": auroc,
        f"{split_name}_precision": precision,
        f"{split_name}_recall": recall,
        f"{split_name}_f1": f1
    }
    logger.info(f"Metrics for {split_name}: {metrics}")

    if use_wandb:
        logging_columns = ["id", "gt_label", "pred_label", "pred_logit"]
        logging_table = wandb.Table(columns=logging_columns)
        preds = (torch.sigmoid(all_logits) >= 0.5).long()
        for id_, label, pred, logit in zip(all_ids, all_labels, preds, all_logits):
            logging_table.add_data(id_, label.item(), pred.item(), logit.item())
        wandb.log({f"{split_name}_logging_table": logging_table})
        wandb.log(metrics)

    return metrics

# Evaluation for .jsonl files using QWen2Classifier and predict_and_eval
def evaluate_jsonl_file(classifier, data_file, image_dir, model_path, split_name, device, batch_size=64, use_wandb=False):
    dataset = JsonlDataset(data_file, image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize QWen2Classifier
    vlm_classifier = QWen2Classifier(
        model_path=model_path,
        prompt="Is this a hateful meme?",
        base_model_path=model_path,  # For LoRA or full model
        processor_path=model_path
    )
    
    # Manually load classifier to avoid _init_model
    classifier_config_path = os.path.join(os.path.dirname(args.classifier_path), 'classifier_config.json')
    classifier_bin_path = args.classifier_path
    vlm_classifier.classifier = vlm_classifier.init_classifier(classifier_config_path)
    vlm_classifier.model = nn.Module()
    vlm_classifier.model.add_module("classifier", vlm_classifier.classifier.to(device).eval())
    vlm_classifier.model.classifier.load_state_dict(torch.load(classifier_bin_path, weights_only=False))
    logger.info(f"Loaded classifier weights from {classifier_bin_path}")

    # Call predict_and_eval
    (gen_metrics, cls_metrics) = predict_and_eval(
        VLMClassifier=vlm_classifier,
        dataset="FB",
        split_dl=dataloader,
        split=split_name,
        use_wandb=use_wandb
    )

    metrics = {
        f"{split_name}_accuracy": cls_metrics[0],
        f"{split_name}_auroc": cls_metrics[1],
        f"{split_name}_precision": cls_metrics[2],
        f"{split_name}_recall": cls_metrics[3],
        f"{split_name}_f1": cls_metrics[4]
    }
    logger.info(f"Metrics for {split_name} (Classifier): {metrics}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP classifier using existing methods")
    parser.add_argument("--data_file", type=str, required=True, help="Path to .pt or .jsonl file")
    parser.add_argument("--image_dir", type=str, help="Directory containing images for .jsonl")
    parser.add_argument("--classifier_path", type=str, default="./output/qwen2_vl-2b_lora_sft/classifier.bin")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Path to Qwen2-VL model")
    parser.add_argument("--split_name", type=str, default="dev_seen")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="meme_classification", name=f"eval_{args.split_name}")

    # Load classifier
    classifier_config_path = os.path.join(os.path.dirname(args.classifier_path), 'classifier_config.json')
    with open(classifier_config_path, "r") as f:
        classifier_config = json.load(f)
    classifier = Classifier(
        input_shape=classifier_config["hidden_size"],
        num_layers=classifier_config["num_layers"],
        proj_dim=classifier_config["proj_dim"],
        output_dim=classifier_config["output_dim"],
        input_dropout=classifier_config["input_dropout"],
        dropout=classifier_config["dropout"]
    ).to(args.device, dtype=torch.bfloat16)
    checkpoint = torch.load(args.classifier_path, weights_only=False)
    classifier.load_state_dict(checkpoint)
    logger.info(f"Loaded classifier weights from {args.classifier_path}")

    # Evaluate based on file type
    if args.data_file.endswith('.pt'):
        metrics = evaluate_pt_file(
            classifier,
            args.data_file,
            args.split_name,
            args.device,
            args.batch_size,
            args.use_wandb
        )
    elif args.data_file.endswith('.jsonl'):
        if not args.image_dir:
            raise ValueError("Image directory required for .jsonl file")
        metrics = evaluate_jsonl_file(
            classifier,
            args.data_file,
            args.image_dir,
            args.model_path,
            args.split_name,
            args.device,
            args.batch_size,
            args.use_wandb
        )
    else:
        raise ValueError("Unsupported file format. Use .pt or .jsonl")

    # Save metrics
    output_dir = os.path.dirname(args.classifier_path)
    metrics_file = os.path.join(output_dir, f"eval_metrics_{args.split_name}.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()