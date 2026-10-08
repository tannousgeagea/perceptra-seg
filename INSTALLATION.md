# Installation Guide

## 1. Install perceptra-seg

```bash
pip install "perceptra-seg[torch]"          # SDK
pip install "perceptra-seg[server,torch]"   # SDK + REST service
pip install perceptra-seg                   # REST client only (no torch)
```

## 2. Install the SAM model code you need

The model code lives on GitHub, not PyPI (PyPI packages cannot depend on git URLs), so install it separately:

```bash
# sam_v1
pip install "git+https://github.com/facebookresearch/segment-anything.git"

# sam_v2
pip install "git+https://github.com/facebookresearch/sam2.git"

# sam_v3 — always the latest upstream code; requires a CUDA GPU and PyTorch >= 2.7
pip install "perceptra-seg[sam3]" "git+https://github.com/facebookresearch/sam3.git"
```

Update SAM3 to the newest upstream commit later with:

```bash
pip install -U --force-reinstall --no-deps "git+https://github.com/facebookresearch/sam3.git"
```

## 3. Model weights

- **sam_v1 / sam_v2**: downloaded automatically on first use into `$PERCEPTRA_SEG_CACHE_DIR`
  (default `~/.cache/segmentor`), or pass `checkpoint_path`.
- **sam_v3**: gated on HuggingFace. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3),
  then `export HF_TOKEN=...` (cached in `$HF_HOME`), or pass `checkpoint_path` /
  `SEGMENTOR_SAM3_CHECKPOINT=/path/sam3.pt`.

## 4. Verify

```bash
python -m perceptra_seg.quickstart
```

## Requirements file

```txt
perceptra-seg[torch,sam3]>=0.3.0
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
SAM-2 @ git+https://github.com/facebookresearch/sam2.git
sam3 @ git+https://github.com/facebookresearch/sam3.git
```

Replace the branch with `@<commit-sha>` on each git URL for reproducible environments.

## Docker

See "Deploying the service" in the README: `cp .env.example .env && docker compose up -d --build`.
