# CPU-only Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./
COPY perceptra_seg/ ./perceptra_seg/
COPY service/ ./service/
COPY config.yaml ./

# Upgrade build tools first so setuptools supports PEP 660 (build_editable)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install local package
RUN pip3 install --no-cache-dir .[server,torch,sam3]
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git


# Create non-root user
RUN useradd -m -u 1000 segmentor && chown -R segmentor:segmentor /app
USER segmentor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:8080/v1/healthz')"

EXPOSE 8080

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]