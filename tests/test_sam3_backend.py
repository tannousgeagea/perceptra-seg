"""Unit tests for the SAM3 backend that need neither sam3 weights nor a GPU."""

import numpy as np
import pytest
import torch

from perceptra_seg.backends.torch_sam_v3 import (
    TorchSAMv3Backend,
    _split_masks,
    _xyxy_to_norm_cxcywh,
)
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import ModelLoadError


@pytest.mark.parametrize(
    "masks, scores",
    [
        (np.ones((3, 1, 8, 8)), np.array([[0.1], [0.2], [0.3]])),  # predict_inst, batched boxes
        (np.ones((1, 8, 8)), np.array([0.5])),  # predict_inst, single prompt
        (torch.ones(3, 1, 8, 8, dtype=torch.bool), torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16)),  # processor
        (np.ones((8, 8)), np.float32(0.5)),
    ],
)
def test_split_masks_shapes(masks, scores) -> None:
    out_masks, out_scores = _split_masks(masks, scores)
    assert len(out_masks) == len(out_scores)
    assert all(m.shape == (8, 8) and m.dtype == np.uint8 for m in out_masks)
    assert all(isinstance(s, float) for s in out_scores)


def test_split_masks_empty() -> None:
    assert _split_masks(torch.zeros(0, 1, 8, 8, dtype=torch.bool), torch.zeros(0)) == ([], [])


def test_box_normalization() -> None:
    assert _xyxy_to_norm_cxcywh((10, 20, 30, 60), w=100, h=200) == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_requires_cuda() -> None:
    cfg = SegmentorConfig()
    cfg.runtime.device = "cpu"
    with pytest.raises(ModelLoadError, match="CUDA"):
        TorchSAMv3Backend(cfg)._resolve_device()


def test_explicit_checkpoint_must_exist(tmp_path) -> None:
    cfg = SegmentorConfig()
    cfg.model.checkpoint_path = str(tmp_path / "missing.pt")
    with pytest.raises(ModelLoadError, match="not found"):
        TorchSAMv3Backend(cfg)._resolve_checkpoint()

    ckpt = tmp_path / "sam3.pt"
    ckpt.write_bytes(b"x")
    cfg.model.checkpoint_path = str(ckpt)
    assert TorchSAMv3Backend(cfg)._resolve_checkpoint() == str(ckpt)


@pytest.mark.parametrize("requested, expected", [("cuda", "cuda:0"), ("cuda:1", "cuda:1")])
def test_cuda_device_gets_explicit_index(monkeypatch: pytest.MonkeyPatch, requested, expected) -> None:
    # torch.cuda.set_device rejects an index-less "cuda" device.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cfg = SegmentorConfig()
    cfg.runtime.device = requested
    device = TorchSAMv3Backend(cfg)._resolve_device()
    assert str(device) == expected
    assert device.index is not None
