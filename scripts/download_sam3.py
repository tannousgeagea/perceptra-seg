"""
Pre-fetch the SAM3 checkpoint into the HuggingFace cache ($HF_HOME).

The service downloads it on first start anyway; run this to warm the cache ahead of
time (e.g. before a deploy) so the first start doesn't wait on a ~3.5 GB download:

    docker compose run --rm perceptra-seg python scripts/download_sam3.py

Requires HF_TOKEN with access to the gated facebook/sam3 repository.
Exits non-zero on failure.
"""
import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "facebook/sam3"

if not os.environ.get("HF_TOKEN", "").strip():
    print("HF_TOKEN is not set — facebook/sam3 is gated and needs an access token.", file=sys.stderr)
    sys.exit(1)

try:
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
    path = hf_hub_download(repo_id=REPO_ID, filename="sam3.pt")
except Exception as exc:
    print(f"ERROR: failed to download SAM3 checkpoint: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"SAM3 checkpoint cached at {path} ({os.path.getsize(path) / 1024**3:.2f} GB)")
