# Meme-KID

Meme-KID is the code release for Knowledge-Injected Dual-Head Learning (KID) for multimodal meme understanding.

KID first constructs entity-anchored auxiliary knowledge with a teacher multimodal model. A student model is then fine-tuned once with a generation head and a discriminative classification head. The training objective is:

```text
L_total = L_gen + L_cls
```

This release does not include RGCL, contrastive retrieval, a second training stage, datasets, model weights, API keys, checkpoints, or experiment logs.

## Layout

```text
configs/kid/          One final KID configuration for each paper task
knowledge_pipeline/   Teacher-knowledge construction and quality-check scripts
scripts/kid/          Reproducible training launcher
src/llamafactory/     Required LLaMA-Factory training infrastructure and KID trainers
docs/                 Reproducibility and data-acquisition notes
```

The eight paper tasks are `hateful_memes`, `harmeme`, `mami_task_a`, `mami_task_b`, `toxicn_mm_task_a`, `toxicn_mm_task_b`, `bangla_abuse_task_a`, and `bangla_abuse_task_b`.

## Quick Start

```bash
git clone https://github.com/PotatoDog1669/KID.git Meme-KID
cd Meme-KID
pip install -e .
cp .env.example .env
# Set TRAIN_DATASET, EVAL_DATASET, DATA_VERSION, and RUN_NAME in .env.
bash scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml
```

### Docker Quick Start

Pull the published CUDA image and run a one-step smoke test before a full experiment:

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

The smoke test uses two samples and one optimization step. For a full run, replace the final command with `bash scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml` and also set `DATA_VERSION` and `RUN_NAME`. See [docs/CONTAINER.md](docs/CONTAINER.md) for dataset naming, Compose, and local-build details.

Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before training, [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) before obtaining data, and [docs/CONTAINER.md](docs/CONTAINER.md) for the reproducible CUDA container. The base training framework is derived from LLaMA-Factory; retain its license and citation notices when publishing this repository.
