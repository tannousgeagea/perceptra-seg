import numpy as np
import pytest

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend


def test_torch_sam_v1_backend_init() -> None:
    """Test TorchSAMv1Backend initialization."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    
    backend = TorchSAMv1Backend(config)
    assert backend.config == config
    assert backend.device is None  # Not loaded yet


def test_torch_sam_v1_checkpoint_path_resolution() -> None:
    """Test checkpoint path resolution."""
    config = SegmentorConfig()
    config.runtime.device = "cpu"
    config.model.encoder_variant = "vit_h"
    
    backend = TorchSAMv1Backend(config)
    
    # Should return URL or cached path
    path = backend._get_checkpoint_path()
    assert path is not None
    assert isinstance(path, str)


@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("torch"),
    reason="PyTorch not installed"
)
def test_backend_inference(config: SegmentorConfig) -> None:
    """Test backend inference (if PyTorch available)."""
    from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend
    
    backend = TorchSAMv1Backend(config)
    backend.load()
    
    # Create dummy image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (100, 100, 300, 300)
    
    mask, score = backend.infer_from_box(image, box)
    
    assert mask.shape == (480, 640)
    assert 0 <= score <= 1
    
    backend.close()

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="torch.compile check needs a CUDA GPU"
)
@pytest.mark.parametrize(
    "model, module, precision",
    [("sam_v1", "segment_anything", "fp32"), ("sam_v3", "sam3", "bf16")],
)
def test_compiled_encoder_matches_eager(model: str, module: str, precision: str) -> None:
    """runtime.compile wraps the image encoder and leaves the masks unchanged."""
    import importlib.util
    from pathlib import Path

    from PIL import Image

    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    from perceptra_seg import Segmentor

    image = np.array(Image.open(Path(__file__).parent.parent / "assets/images/truck.jpg").convert("RGB"))
    boxes = [(450, 300, 1350, 900), (100, 400, 700, 1100), (1000, 200, 1700, 800)]
    results = {}
    for compile_encoder in (False, True):
        config = SegmentorConfig()
        config.model.name = model  # type: ignore[assignment]
        config.model.encoder_variant = "vit_b"  # smallest SAM v1/v2 checkpoints
        config.runtime.device = "cuda"
        config.runtime.precision = precision  # type: ignore[assignment]
        config.runtime.compile = compile_encoder
        backend = Segmentor(config=config).backend
        try:
            results[compile_encoder] = [backend.infer_from_box(image, box) for box in boxes]
        finally:
            backend.close()
    for (eager_mask, eager_score), (compiled_mask, compiled_score) in zip(results[False], results[True]):
        iou = (eager_mask & compiled_mask).sum() / max((eager_mask | compiled_mask).sum(), 1)
        assert iou > 0.98 and abs(eager_score - compiled_score) < 0.02


@pytest.mark.parametrize("model, precision", [("sam_v2", "fp16"), ("sam_v1", "fp16")])
def test_compile_is_skipped_where_it_hurts(model: str, precision: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """sam_v2 (slower) and sam_v1 under mixed precision (masks drift) never call torch.compile."""
    import importlib.util

    import torch

    module = "sam2" if model == "sam_v2" else "segment_anything"
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend
    from perceptra_seg.backends.torch_sam_v2 import TorchSAMv2Backend

    def fail(*args, **kwargs):
        raise AssertionError("torch.compile should not be called")

    monkeypatch.setattr(torch, "compile", fail)
    config = SegmentorConfig()
    config.model.name = model  # type: ignore[assignment]
    config.model.encoder_variant = "vit_b"
    config.runtime.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.runtime.precision = precision  # type: ignore[assignment]
    config.runtime.compile = True
    backend = (TorchSAMv2Backend if model == "sam_v2" else TorchSAMv1Backend)(config)
    backend.load()
    backend.close()
