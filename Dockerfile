# Multi-arch CPU base with PyTorch preinstalled
FROM pytorch/pytorch:2.1.2-cpu-py3.10

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (if needed by OpenCV etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install python deps (torch provided by base image)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy project
COPY . .

# Expose API port
EXPOSE 8000

# Default envs (override in runtime)
ENV HOST=0.0.0.0 PORT=8000 RELOAD=false LOG_LEVEL=info DEVICE_PREFERENCE=cpu

# Start API
CMD ["python", "-m", "api.run_api"]

