#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
template_path=${1:-"$repo_root/configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml"}

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a local model directory or Hugging Face model ID.}"
: "${DATASET_DIR:?Set DATASET_DIR to the directory containing dataset_info.json.}"
: "${MEDIA_DIR:?Set MEDIA_DIR to the image directory used by the selected dataset.}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to a locally registered dataset name.}"
: "${EVAL_DATASET:?Set EVAL_DATASET to a locally registered dataset name.}"

smoke_output_dir=${SMOKE_OUTPUT_DIR:-"$repo_root/outputs/smoke"}
tmp_config=$(mktemp "${TMPDIR:-/tmp}/meme-kid-smoke.XXXXXX.yaml")

cleanup() {
    rm -f "$tmp_config"
}
trap cleanup EXIT

cp "$template_path" "$tmp_config"
sed -i \
    -e "s|^model_name_or_path: .*|model_name_or_path: ${MODEL_NAME_OR_PATH}|" \
    -e "s|^dataset: .*|dataset: ${TRAIN_DATASET}|" \
    -e "s|^eval_dataset: .*|eval_dataset: ${EVAL_DATASET}|" \
    -e "s|^media_dir: .*|media_dir: ${MEDIA_DIR}|" \
    -e "s|^max_samples: .*|max_samples: 2|" \
    -e "s|^preprocessing_num_workers: .*|preprocessing_num_workers: 1|" \
    -e "s|^output_dir: .*|output_dir: ${smoke_output_dir}|" \
    -e "s|^logging_dir: .*|logging_dir: ${smoke_output_dir}/logs|" \
    -e "s|^per_device_train_batch_size: .*|per_device_train_batch_size: 1|" \
    -e "s|^gradient_accumulation_steps: .*|gradient_accumulation_steps: 1|" \
    -e "s|^eval_strategy: .*|eval_strategy: \"no\"|" \
    -e "s|^report_to: .*|report_to: none|" \
    -e "s|^flash_attn: .*|flash_attn: disabled|" \
    -e "s|^enable_liger_kernel: .*|enable_liger_kernel: false|" \
    "$tmp_config"

{
    printf '\ndataset_dir: %s\n' "$DATASET_DIR"
    printf 'max_steps: 1\n'
    printf 'save_strategy: "no"\n'
    printf 'overwrite_output_dir: true\n'
} >> "$tmp_config"

cd "$repo_root"
llamafactory-cli train "$tmp_config"
