# Meme-KID

Meme-KID 是面向多模态 meme 理解的 Knowledge-Injected Dual-Head Learning（KID）代码发布。

KID 使用教师多模态模型构建实体锚定的辅助知识，再对学生模型进行一次训练，同时优化生成头与判别分类头：

```text
L_total = L_gen + L_cls
```

本仓库不包含 RGCL、对比检索、二阶段训练、数据集、模型权重、API key、checkpoint 或实验日志。

## 目录结构

```text
configs/kid/          论文八个任务的最终 KID 配置
knowledge_pipeline/   教师知识构建和质量检查脚本
scripts/kid/          可复现的训练和烟测启动脚本
src/llamafactory/     必需的 LLaMA-Factory 训练基础设施和 KID trainer
docs/                 复现说明、数据来源和容器说明
```

支持的论文任务为：`hateful_memes`、`harmeme`、`mami_task_a`、`mami_task_b`、`toxicn_mm_task_a`、`toxicn_mm_task_b`、`bangla_abuse_task_a` 和 `bangla_abuse_task_b`。

## 快速开始

```bash
git clone https://github.com/PotatoDog1669/KID.git Meme-KID
cd Meme-KID
pip install -e .
cp .env.example .env
# 在 .env 中填写 TRAIN_DATASET、EVAL_DATASET、DATA_VERSION 和 RUN_NAME。
bash scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml
```

### Docker 快速开始

先拉取已发布的 CUDA 镜像，再运行一轮最小烟测：

```bash
docker pull ghcr.io/potatodog1669/kid:latest

docker run --rm --gpus all --entrypoint bash \
  -v /path/to/data:/app/data:ro \
  -v /path/to/model:/app/model:ro \
  -v /path/to/media:/app/media:ro \
  -v /path/to/output:/app/outputs \
  -e MODEL_NAME_OR_PATH=/app/model \
  -e DATASET_DIR=/app/data \
  -e MEDIA_DIR=/app/media \
  -e TRAIN_DATASET=your_train_dataset \
  -e EVAL_DATASET=your_eval_dataset \
  -e SMOKE_OUTPUT_DIR=/app/outputs/smoke \
  ghcr.io/potatodog1669/kid:latest \
  -lc 'bash scripts/kid/smoke_test.sh'
```

烟测只使用 2 条样本和 1 个优化步。正式训练时，将最后一行替换为 `bash scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml`，并额外设置 `DATA_VERSION` 和 `RUN_NAME`。数据集命名、Compose 和本地构建方式见 [docs/CONTAINER.md](docs/CONTAINER.md)。

训练前请阅读 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)，获取数据前请阅读 [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)。需要一致的 CUDA 依赖环境时，请使用 [docs/CONTAINER.md](docs/CONTAINER.md) 中的 Docker 方案。

## 许可证与引用

底层训练基础设施派生自 LLaMA-Factory。公开发布时请保留其许可证和引用声明。
