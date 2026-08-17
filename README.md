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

Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before training, [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) before obtaining data, and [docs/CONTAINER.md](docs/CONTAINER.md) for the reproducible CUDA container. The base training framework is derived from LLaMA-Factory; retain its license and citation notices when publishing this repository.
