"""Checkpoint download helpers."""

import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def cache_dir() -> Path:
    """Checkpoint cache: $PERCEPTRA_SEG_CACHE_DIR or ~/.cache/segmentor."""
    path = Path(os.environ.get("PERCEPTRA_SEG_CACHE_DIR") or Path.home() / ".cache" / "segmentor")
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_checkpoint(url: str, filename: str) -> str:
    """Download ``url`` into the cache once; partial downloads are never left in place."""
    target = cache_dir() / filename
    if target.exists():
        return str(target)

    tmp = target.with_name(f".{target.name}.{os.getpid()}.part")
    logger.info("Downloading checkpoint %s -> %s", url, target)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("Saved to %s", target)
    return str(target)
