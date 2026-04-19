# GPU-enabled Dockerfile — PyTorch + CUDA + cuDNN pre-installed
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System deps for OpenCV (libgl1 / libglib2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install other necessary packages and dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    apt-utils \
	vim \
	git \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY pyproject.toml ./
COPY perceptra_seg/ ./perceptra_seg/
COPY service/ ./service/
COPY scripts/ ./scripts/
COPY config.yaml ./

# Upgrade build tools, then install — torch already present in base image
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir .[server,torch,sam3]
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Pre-download SAM3 checkpoint at build time so no HF auth is needed at runtime.
# Pass --build-arg HF_TOKEN=<your_token> to enable; omit to skip (runtime download fallback).
ARG HF_TOKEN=""
RUN mkdir -p /opt/models && \
    HF_TOKEN=$HF_TOKEN python scripts/download_sam3.py

# Non-root user — give segmentor read access to the pre-downloaded model
RUN useradd -m -u 1000 segmentor \
    && chown -R segmentor:segmentor /app \
    && chown -R segmentor:segmentor /opt/models
USER segmentor

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:8080/v1/healthz')"

EXPOSE 8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
