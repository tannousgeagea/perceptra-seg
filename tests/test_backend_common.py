"""Tests for shared backend helpers and v1/v2 precision/config handling (no weights needed)."""

from pathlib import Path

import pytest
import torch

from perceptra_seg.backends._common import autocast
from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend
from perceptra_seg.backends.torch_sam_v2 import TorchSAMv2Backend
from perceptra_seg.config import SegmentorConfig


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_autocast_is_noop_on_cpu(precision: str) -> None:
    with autocast(torch.device("cpu"), precision):
        assert not torch.is_autocast_enabled("cpu")
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


def test_autocast_noop_without_device() -> None:
    with autocast(None, "fp16"):
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


@pytest.mark.parametrize("precision, expected", [("fp16", "fp16"), ("bf16", "fp16"), ("fp32", "fp32")])
def test_sam_v1_maps_bf16_to_fp16(monkeypatch: pytest.MonkeyPatch, precision: str, expected: str) -> None:
    # segment_anything converts outputs to numpy, which has no bfloat16.
    seen = {}
    monkeypatch.setattr(
        "perceptra_seg.backends.torch_sam_v1.autocast",
        lambda device, p: seen.setdefault("precision", p) or autocast(None, p),
    )
    cfg = SegmentorConfig()
    cfg.runtime.precision = precision  # type: ignore[assignment]
    TorchSAMv1Backend(cfg)._autocast()
    assert seen["precision"] == expected


@pytest.mark.parametrize("variant", ["vit_h", "vit_l", "vit_b"])
def test_sam_v2_config_names_exist_upstream(variant: str) -> None:
    cfg = SegmentorConfig()
    cfg.model.encoder_variant = variant  # type: ignore[assignment]
    name = TorchSAMv2Backend(cfg)._get_model_config()
    sam2 = pytest.importorskip("sam2")
    assert (Path(sam2.__file__).parent / f"{name}.yaml").exists(), name


def test_backends_keep_fp32_weights() -> None:
    # Regression: .half() on the model broke inference ("Input type (float) and bias type (c10::Half)").
    for module in ("torch_sam_v1", "torch_sam_v2"):
        source = (Path(__file__).parent.parent / "perceptra_seg" / "backends" / f"{module}.py").read_text()
        assert ".half()" not in source, module
