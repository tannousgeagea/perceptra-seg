"""
Pre-download the SAM3 checkpoint from HuggingFace during Docker build.

Usage (called from Dockerfile RUN step):
    HF_TOKEN=hf_xxx python scripts/download_sam3.py

Exits 0 when:
  - No HF_TOKEN provided (pre-download skipped gracefully)
  - Checkpoint downloaded and saved to DEST successfully

Exits 1 when:
  - HF_TOKEN provided but download fails (causes build to fail loudly)
"""
import os
import sys
import shutil

DEST = "/opt/models/sam3.pt"
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if not HF_TOKEN:
    print("HF_TOKEN not set — skipping SAM3 pre-download. "
          "The checkpoint will be fetched at runtime (requires HF_TOKEN at that point).")
    sys.exit(0)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed — cannot pre-download SAM3.", file=sys.stderr)
    sys.exit(1)

print("Downloading facebook/sam3 checkpoint from HuggingFace …")
try:
    cached_path = hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        token=HF_TOKEN,
    )
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.copy2(cached_path, DEST)
    size_gb = os.path.getsize(DEST) / (1024 ** 3)
    print(f"SAM3 checkpoint saved to {DEST} ({size_gb:.2f} GB)")
except Exception as exc:
    print(f"ERROR: Failed to download SAM3 checkpoint: {exc}", file=sys.stderr)
    sys.exit(1)
