"""Helpers shared by the PyTorch backends."""

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@contextlib.contextmanager
def autocast(device: torch.device | None, precision: str) -> Iterator[None]:
    """Mixed precision for inference on CUDA; a no-op for fp32 or CPU.

    Weights stay fp32 and autocast casts per-op. Converting the model with ``.half()``
    instead breaks the SAM predictors, which feed it float32 images.
    """
    dtype = AUTOCAST_DTYPES.get(precision)
    if dtype is None or device is None or device.type != "cuda":
        yield
        return
    with torch.autocast("cuda", dtype=dtype):
        yield


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def split_masks(masks: Any, scores: Any) -> tuple[list[np.ndarray], list[float]]:
    """Flatten predictor output into per-object (HxW uint8 mask, score) pairs.

    Handles ``(H, W)``, ``(N, H, W)`` and ``(N, 1, H, W)`` masks with matching ``()``,
    ``(N,)`` or ``(N, 1)`` scores, as numpy arrays or tensors.
    """
    masks_np = to_numpy(masks)
    scores_np = to_numpy(scores).reshape(-1)
    if masks_np.ndim == 2:
        masks_np = masks_np[None]
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    return (
        [(m > 0).astype(np.uint8) for m in masks_np],
        [float(s) for s in scores_np[: len(masks_np)]],
    )
