"""Data models for Segmentor."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Result from a segmentation operation.

    Attributes:
        mask: Binary mask as numpy array (HxW) if requested
        polygons: List of polygon contours, each as list of (x,y) tuples
        rle: COCO RLE dictionary with 'size' and 'counts'
        png_bytes: PNG-encoded mask bytes
        score: Confidence score from model (0-1)
        area: Number of pixels in mask
        bbox: Bounding box as (x1, y1, x2, y2) or None
        latency_ms: Total processing time in milliseconds
        model_info: Dict with model name, backend, device info
        request_id: Unique identifier for this request
    """

    mask: np.ndarray | None = None
    polygons: list[list[tuple[float, float]]] | None = None
    rle: dict[str, Any] | None = None
    png_bytes: bytes | None = None
    score: float = 0.0
    area: int = 0
    bbox: tuple[int, int, int, int] | None = None
    latency_ms: float = 0.0
    model_info: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    text_label: Optional[str] = None  # Add this field
    metadata: dict[str, Any] = field(default_factory=dict) 

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values and numpy arrays."""
        result = {
            "score": self.score,
            "area": self.area,
            "bbox": self.bbox,
            "latency_ms": self.latency_ms,
            "model_info": self.model_info,
            "request_id": self.request_id,
        }
        if self.rle is not None:
            result["rle"] = self.rle
        if self.polygons is not None:
            result["polygons"] = self.polygons
        if self.png_bytes is not None:
            import base64

            result["png_base64"] = base64.b64encode(self.png_bytes).decode()
        return result