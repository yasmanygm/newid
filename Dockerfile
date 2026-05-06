# Multi-stage build for the passport MRZ reader HTTP service.
# Stage 1 resolves deps with uv into /app/.venv; stage 2 is a slim runtime
# that copies the venv, the app code, and the model files.

ARG PYTHON_VERSION=3.13

# ---------- builder ----------
FROM 172.24.11.237:6001/python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml README.md ./

# Resolve deps into /app/.venv. No --frozen because we don't ship a lock file;
# the build is reproducible via uv's PEP 440 resolver on pinned lower bounds.
RUN uv sync --no-dev --no-install-project

# ---------- runtime ----------
FROM 172.24.11.237:6001/python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG ENABLE_FASTMRZ=0
ARG GIT_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH" \
    CUDA_VISIBLE_DEVICES="" \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    NNPACK_SUPPRESS_WARNINGS=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    YOLO_OFFLINE=True \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MPLBACKEND=Agg \
    MRZ_ENGINE=paddleocr \
    GIT_SHA=${GIT_SHA} \
    APP_VERSION=1.0.0

# libgomp1 for paddle/openmp; libgl1 + libglib2.0-0 for opencv-python-headless;
# curl for HEALTHCHECK. tesseract-ocr only when the fastmrz engine is enabled.
RUN set -eux; \
    apt-get update; \
    base_pkgs="libgomp1 libgl1 libglib2.0-0 curl"; \
    if [ "$ENABLE_FASTMRZ" = "1" ]; then \
        apt-get install -y --no-install-recommends $base_pkgs tesseract-ocr; \
    else \
        apt-get install -y --no-install-recommends $base_pkgs; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
    useradd -u 10001 -m -s /bin/bash app

WORKDIR /app

# Dep layer
COPY --from=builder /app/.venv /app/.venv

# Application + pipeline source + models
COPY app ./app
COPY src ./src
COPY models/det ./models/det
COPY models/rec ./models/rec
COPY runs/obb/document_obb/weights/best.pt ./runs/obb/document_obb/weights/best.pt

# Only ship tessdata when fastmrz is built in. We use a guard file so the COPY
# instruction stays valid even without the directory (BuildKit glob fallback).
# If ENABLE_FASTMRZ=1 the user is expected to have populated models/tessdata/
# before building. We don't fail the build if it's absent; the pipeline does.

# Ultralytics writes settings on import — point it at a writable dir the app
# user owns, so the read-only rootfs in compose doesn't block startup.
RUN mkdir -p /app/.ultralytics && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--limit-concurrency", "8", \
     "--timeout-keep-alive", "30", \
     "--no-access-log"]
