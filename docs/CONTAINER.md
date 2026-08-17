# Container Environment

The KID release includes a CUDA Dockerfile at `docker/docker-cuda/Dockerfile.kid`. It installs the package versions required by this repository rather than relying on whatever versions happen to be present in a base image.

Build the image from the repository root:

```bash
docker build -f docker/docker-cuda/Dockerfile.kid -t meme-kid:latest .
```

Mount datasets read-only and write outputs outside the source tree:

```bash
docker run --rm --gpus all \
  -v /path/to/data:/app/data:ro \
  -v /path/to/model:/app/model:ro \
  -v /path/to/data/image/ToxiCN_MM:/app/media:ro \
  -v /path/to/outputs:/app/outputs \
  --entrypoint bash \
  -e MODEL_NAME_OR_PATH=/app/model \
  -e DATASET_DIR=/app/data \
  -e MEDIA_DIR=/app/media \
  -e TRAIN_DATASET=your_train_dataset \
  -e EVAL_DATASET=your_eval_dataset \
  -e DATA_VERSION=your_data_version \
  -e RUN_NAME=your_run_name \
  meme-kid:latest scripts/kid/train.sh configs/kid/toxicn_mm_task_b/kid_qwen2_5vl_7b.yaml
```

For the reusable one-step smoke test, also provide `MODEL_NAME_OR_PATH`, `DATASET_DIR`, `MEDIA_DIR`, and `TRAIN_DATASET`, then invoke `bash scripts/kid/smoke_test.sh` inside the container. The smoke test uses two samples, runs one optimization step, disables evaluation and checkpoint saving, and writes only to `outputs/smoke/`.

`docker/docker-cuda/docker-compose.kid.yml` provides the equivalent Compose entry point. Keep model files, datasets, Hugging Face caches, and outputs outside Git.
