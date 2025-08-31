# Multi-stage Dockerfile for Hair Segmentation API

# Builder stage: Common dependencies for all environments
FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 && \
    pip install --no-cache-dir -r requirements.txt

# Production stage: Stable deployment
FROM python:3.10-slim AS production

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder $VENV_PATH $VENV_PATH
COPY __init__.py ./
COPY api/ ./api/
COPY color_changer/ ./color_changer/
COPY model/__init__.py ./model/__init__.py
COPY model/config.py ./model/config.py
COPY model/inference/ ./model/inference/
COPY model/models/ ./model/models/
COPY model/training/ ./model/training/
COPY model/data_loader/ ./model/data_loader/
COPY model/utils/ ./model/utils/

ENV PATH="$VENV_PATH/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "api.run_api"]

# Development stage: Auto-reload for development
FROM production AS development

WORKDIR /app

COPY model/trained_models/ ./model/trained_models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
