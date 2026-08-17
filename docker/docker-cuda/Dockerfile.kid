ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

# The NGC base image can contain newer packages than this project supports.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gettext-base && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --upgrade -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir "bitsandbytes>=0.43.0" "torchmetrics>=1.3.0,<1.7.0"

COPY . /app
RUN python -m pip install --no-cache-dir --no-deps -e /app

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["llamafactory-cli"]
CMD ["--help"]
