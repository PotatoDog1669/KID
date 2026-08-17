# Reproducibility

## Method

KID has one student fine-tuning stage. The semantic generation head learns to produce the target response and the classification head predicts the task label from the same model representation. The loss is `L_gen + L_cls`, with the two weights configured as `loss_ratio: [1.0, 1.0]`.

Teacher knowledge is constructed before training and injected into the student input. The teacher should produce evidence and contextual knowledge, not a final label.

## Environment

Use a CUDA-capable Python environment compatible with the version pins in `pyproject.toml` and `requirements.txt`.

```bash
pip install -e .
cp .env.example .env
set -a
source .env
set +a
```

## Data Preparation

Download each original dataset under its own license, place it outside the repository or under the ignored `data/` directory, and register the processed ShareGPT-style datasets in `data/dataset_info.json` locally. Do not commit downloaded images, annotations, teacher outputs, or a dataset registry containing local paths.

Generate or validate teacher knowledge with the scripts in `knowledge_pipeline/scripts/`. The reusable prompt templates are in `knowledge_pipeline/prompts/`.

## Training

The launcher requires the dataset variables used by the YAML configurations.

```bash
export TRAIN_DATASET=your_train_dataset_name
export EVAL_DATASET=your_eval_dataset_name
export DATA_VERSION=your_data_version
export RUN_NAME=toxicn_task_b_kid
bash scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml
```

Set a unique `RUN_NAME` for each run. Output directories are ignored by Git.

## Validation Boundary

The release is intended to be validated with syntax, import, and configuration checks before launching a GPU job. No script in this repository starts a container or training run automatically.
