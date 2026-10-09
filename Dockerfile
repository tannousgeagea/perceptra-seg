# syntax=docker/dockerfile:1
#
# GPU image for the perceptra-seg REST service (SAM v1 / v2 / v3).
#
#   docker compose build                         # SAM3 from the latest commit on main
#   docker compose build --build-arg SAM3_REF=<commit-sha>   # reproducible pin
#
# Model weights are NOT baked in: they are downloaded on first start into the
# /models volume (SAM3 needs HF_TOKEN at runtime, see .env.example).

ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
# Upstream git refs (branch, tag or commit SHA).
ARG SAM3_REF=main
ARG SAM2_REF=main
ARG SAM1_REF=main

# Upstream SAM3 source. ADD re-checks the remote on every build, so a new commit on
# SAM3_REF invalidates the cache and is picked up without --no-cache.
FROM scratch AS sam3-src
ARG SAM3_REF
ADD --keep-git-dir=true https://github.com/facebookresearch/sam3.git#${SAM3_REF} /

FROM ${BASE_IMAGE}
ARG SAM2_REF
ARG SAM1_REF

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    PERCEPTRA_SEG_CACHE_DIR=/models/perceptra-seg \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The base image's Python is OS-managed (PEP 668) and has no ensurepip, so install into
# a venv that still sees the base image's torch/triton via --system-site-packages.
RUN python3 -m venv --without-pip --system-site-packages /opt/venv

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY perceptra_seg/ ./perceptra_seg/
COPY service/ ./service/
COPY scripts/ ./scripts/
COPY config.yaml ./

# Single resolver pass so sam3's numpy<2 constraint is honoured by every package.
# --no-build-isolation reuses the base image's torch (sam2 lists torch as a build
# requirement, which would otherwise download a second multi-GB copy);
# SAM2_BUILD_CUDA=0 skips sam2's optional CUDA extension (no nvcc in a runtime image).
RUN --mount=type=bind,from=sam3-src,target=/tmp/sam3,rw \
    python -m pip install --upgrade pip "setuptools<81" wheel \
    && SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation \
        ".[server,sam3]" \
        /tmp/sam3 \
        "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}" \
        "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git@${SAM1_REF}" \
    && printf '{"sam3_commit": "%s"}\n' "$(git -c safe.directory='*' -C /tmp/sam3 rev-parse HEAD)" \
        > /app/build-info.json \
    && cat /app/build-info.json \
    && python -c "import sam3, sam2, segment_anything, perceptra_seg, service.main"

# Ubuntu 24.04 bases ship a default "ubuntu" user on UID 1000.
RUN (userdel -r ubuntu 2>/dev/null || true) \
    && useradd -m -u 1000 segmentor \
    && mkdir -p /models \
    && chown -R segmentor:segmentor /app /models
USER segmentor

VOLUME ["/models"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import json,sys,urllib.request; sys.exit(json.load(urllib.request.urlopen('http://localhost:8080/v1/healthz', timeout=5))['status'] != 'ok')"

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
