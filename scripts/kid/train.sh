#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
config_path=${1:?"usage: bash scripts/kid/train.sh <config.yaml>"}

cd "$repo_root"

for required_var in TRAIN_DATASET EVAL_DATASET DATA_VERSION RUN_NAME; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "Missing required environment variable: $required_var" >&2
        exit 2
    fi
done

export DATE=${DATE:-$(date +%Y%m%d)}
tmp_config=$(mktemp "${TMPDIR:-/tmp}/meme-kid-train.XXXXXX.yaml")
cleanup() {
    rm -f "$tmp_config"
}
trap cleanup EXIT

envsubst < "$config_path" > "$tmp_config"

# Allow a local model/data mount to override the public config defaults.
if [[ -n "${MODEL_NAME_OR_PATH:-}" ]]; then
    sed -i "s|^model_name_or_path: .*|model_name_or_path: ${MODEL_NAME_OR_PATH}|" "$tmp_config"
fi
if [[ -n "${MEDIA_DIR:-}" ]]; then
    sed -i "s|^media_dir: .*|media_dir: ${MEDIA_DIR}|" "$tmp_config"
fi
if [[ -n "${DATASET_DIR:-}" ]]; then
    printf '\ndataset_dir: %s\n' "$DATASET_DIR" >> "$tmp_config"
fi

llamafactory-cli train "$tmp_config"
