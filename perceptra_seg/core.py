"""Core Segmentor class."""

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from perceptra_seg.backends.base import BaseSAMBackend
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, InvalidPromptError, ModelLoadError
from perceptra_seg.models import SegmentationResult
from perceptra_seg.utils.cache import EmbeddingCache
from perceptra_seg.utils.image_io import load_image
from perceptra_seg.utils.dependency_check import ensure_dependency
from perceptra_seg.utils.mask_utils import (
    apply_morphology,
    mask_to_png_bytes,
    mask_to_polygons,
    mask_to_rle,
    remove_small_components,
)

logger = logging.getLogger(__name__)


class Segmentor:
    """Main segmentation interface supporting SAM v1 and v2.

    Args:
        config: SegmentorConfig instance or None to use defaults
        **kwargs: Override specific config values
    """

    def __init__(
        self,
        config: SegmentorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize configuration
        if config is None:
            config = SegmentorConfig()

        # Apply kwargs overrides with smart mapping
        # Common shortcuts for user convenience
        shortcuts = {
            "model": ("model", "name"),
            "backend": ("runtime", "backend"),
            "device": ("runtime", "device"),
            "precision": ("runtime", "precision"),
            "batch_size": ("runtime", "batch_size"),
            "checkpoint_path": ("model", "checkpoint_path"),
            "encoder_variant": ("model", "encoder_variant"),
        }

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if "." in key:
                section, field = key.split(".", 1)
                if hasattr(config, section):
                    setattr(getattr(config, section), field, value)
            elif key in shortcuts:
                # Use shortcut mapping
                section_name, field_name = shortcuts[key]
                section = getattr(config, section_name)
                setattr(section, field_name, value)
            else:
                # Try to find which config section this key belongs to
                applied = False
                for section_name in ["model", "runtime", "tiling", "outputs", "thresholds", 
                                     "postprocess", "cache", "server", "logging"]:
                    section = getattr(config, section_name, None)
                    if section and hasattr(section, key):
                        setattr(section, key, value)
                        applied = True
                        break
                
                if not applied:
                    logger.warning(f"Unknown config parameter: {key}")

        self.config = config
        self.backend: BaseSAMBackend | None = None
        self.cache: EmbeddingCache | None = None

        if self.config.cache.enabled:
            self.cache = EmbeddingCache(max_size=self.config.cache.max_items)

        self._load_backend()

    def _load_backend(self) -> None:
        """Load the appropriate backend based on configuration."""
        backend_key = f"{self.config.runtime.backend}_{self.config.model.name}"

        try:
            if backend_key == "torch_sam_v1":
                # Check if SAM v1 is installed
                try:
                    ensure_dependency(
                        package_name="segment_anything",
                        git_url="https://github.com/facebookresearch/segment-anything.git",
                        optional=False,
                    )
                except ImportError:
                    raise ModelLoadError(
                        "SAM v1 not found. Install it with:\n"
                        "  pip install git+https://github.com/facebookresearch/segment-anything.git"
                    )
                
                from perceptra_seg.backends.torch_sam_v1 import TorchSAMv1Backend
                self.backend = TorchSAMv1Backend(self.config)
                
            elif backend_key == "torch_sam_v2":
                # Check if SAM v2 is installed
                try:
                    ensure_dependency(
                        package_name="sam2",
                        git_url="https://github.com/facebookresearch/segment-anything-2.git",
                        optional=False,
                    )
                except ImportError:
                    raise ModelLoadError(
                        "SAM v2 not found. Install it with:\n"
                        "  pip install git+https://github.com/facebookresearch/segment-anything-2.git"
                    )
                
                from perceptra_seg.backends.torch_sam_v2 import TorchSAMv2Backend
                self.backend = TorchSAMv2Backend(self.config)
            elif backend_key == "torch_sam_v3":
                # Check if SAM v2 is installed
                try:
                    ensure_dependency(
                        package_name="sam3",
                        git_url="https://github.com/facebookresearch/sam3.git",
                        optional=True,
                    )
                except ImportError:
                    raise ModelLoadError(
                        "SAM v3 not found. Install it with:\n"
                        "  pip install git+https://github.com/facebookresearch/sam3.git"
                    )
                
                from perceptra_seg.backends.torch_sam_v3 import TorchSAMv3Backend
                self.backend = TorchSAMv3Backend(self.config)
            elif backend_key == "onnx_sam_v1":
                from perceptra_seg.backends.onnx_sam_v1 import ONNXSAMv1Backend

                self.backend = ONNXSAMv1Backend(self.config)
            elif backend_key == "onnx_sam_v2":
                from perceptra_seg.backends.onnx_sam_v2 import ONNXSAMv2Backend

                self.backend = ONNXSAMv2Backend(self.config)
            else:
                raise BackendError(f"Unknown backend: {backend_key}")

            self.backend.load()
            logger.info(f"Loaded backend: {backend_key}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load backend {backend_key}: {e}") from e

    def set_backend(self, backend_name: str) -> None:
        """Switch to a different backend.

        Args:
            backend_name: Name of backend ('torch' or 'onnx')
        """
        if self.backend is not None:
            self.backend.close()

        self.config.runtime.backend = backend_name  # type: ignore
        self._load_backend()

    def segment_from_box(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> SegmentationResult:
        """Segment object from bounding box.

        Args:
            image: Input image (numpy array, PIL Image, bytes, or path)
            box: Bounding box as (x1, y1, x2, y2) in absolute pixels
            output_formats: List of output formats ['rle', 'png', 'polygons', 'numpy']
            return_overlay: Whether to return overlay visualization

        Returns:
            SegmentationResult with requested outputs
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Load and validate image
        img_array = load_image(image)
        self._validate_box(box, img_array.shape)      #type: ignore

        if output_formats is None:
            output_formats = self.config.outputs.default_formats       #type: ignore

        # Perform segmentation
        if self.backend is None:
            raise BackendError("Backend not loaded")

        mask, score = self.backend.infer_from_box(img_array, box)

        # Postprocess
        mask = self._postprocess_mask(mask, img_array.shape)             #type: ignore

        # Generate outputs
        result = self._create_result(
            mask=mask,
            score=score,
            output_formats=output_formats,                   #type: ignore
            latency_ms=(time.time() - start_time) * 1000,
            request_id=request_id,
        )

        return result

    def segment_from_points(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        points: list[tuple[int, int, int]],
        *,
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> SegmentationResult:
        """Segment object from point prompts.

        Args:
            image: Input image
            points: List of (x, y, label) where label is 1 (positive) or 0 (negative)
            output_formats: List of output formats
            return_overlay: Whether to return overlay

        Returns:
            SegmentationResult with requested outputs
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        img_array = load_image(image)
        self._validate_points(points, img_array.shape)            #type: ignore

        if output_formats is None:
            output_formats = self.config.outputs.default_formats          #type: ignore

        if self.backend is None:
            raise BackendError("Backend not loaded")

        mask, score = self.backend.infer_from_points(img_array, points)
        mask = self._postprocess_mask(mask, img_array.shape)          #type: ignore
 
        result = self._create_result(
            mask=mask,
            score=score,
            output_formats=output_formats,         #type: ignore
            latency_ms=(time.time() - start_time) * 1000,
            request_id=request_id,
        )

        return result

    def segment_from_text(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        text: str,
        *,
        output_formats: list[str] | None = None,
    ) -> list[SegmentationResult]:
        """Segment using SAM3 text prompt."""
        start_time = time.time()

        if self.backend is None:
            raise BackendError("Backend not loaded")

        img = load_image(image)

        masks, scores = self.backend.infer_from_text(img, text)
        total_latency = (time.time() - start_time) * 1000
        per_item_latency = total_latency / len(masks) if masks else total_latency

        results = []
        for mask, score in zip(masks, scores):
            mask = self._postprocess_mask(mask, img.shape)   # type: ignore
            result = self._create_result(
                mask=mask,
                score=score,
                output_formats=output_formats or self.config.outputs.default_formats,  # type: ignore
                latency_ms=per_item_latency,
                request_id=str(uuid.uuid4()),
            )
            results.append(result)

        return results

    def segment_from_exemplar_box(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
    ) -> list[SegmentationResult]:
        """Segment using SAM3 exemplar visual prompt."""
        start_time = time.time()

        if self.backend is None:
            raise BackendError("Backend not loaded")

        img = load_image(image)
        self._validate_box(box, img.shape)    #type: ignore

        masks, scores = self.backend.infer_from_exemplar_box(img, box)
        total_latency = (time.time() - start_time) * 1000
        per_item_latency = total_latency / len(masks) if masks else total_latency

        results = []
        for mask, score in zip(masks, scores):
            mask = self._postprocess_mask(mask, img.shape)      # type: ignore
            results.append(
                self._create_result(
                    mask=mask,
                    score=score,
                    output_formats=output_formats or self.config.outputs.default_formats,   # type: ignore
                    latency_ms=per_item_latency,
                    request_id=str(uuid.uuid4()),
                )
            )

        return results

    def segment_from_text_and_box(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        text: str,
        box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
    ) -> list[SegmentationResult]:
        """Segment using combined text + box prompts."""
        start_time = time.time()
        if self.backend is None:
            raise BackendError("Backend not loaded")

        img = load_image(image)
        self._validate_box(box, img.shape)  # type: ignore

        masks, scores = self.backend.infer_from_text_and_box(img, text, box)
        total_latency = (time.time() - start_time) * 1000
        per_item_latency = total_latency / len(masks) if masks else total_latency

        results = []
        for mask, score in zip(masks, scores):
            mask = self._postprocess_mask(mask, img.shape) # type: ignore
            results.append(
                self._create_result(
                    mask=mask,
                    score=score,
                    output_formats=output_formats or self.config.outputs.default_formats,  # type: ignore
                    latency_ms=per_item_latency,
                    request_id=str(uuid.uuid4()),
                )
            )

        return results

    def segment(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: list[tuple[int, int, int]] | None = None,
        text: str | None = None,
        exemplar_box: tuple[int, int, int, int] | None = None,
        *,
        strategy: Literal["merge", "largest", "all"] = "largest",
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> list[SegmentationResult]:
        """Segment with multiple prompts.

        Args:
            image: Input image
            boxes: List of bounding boxes
            points: List of point prompts
            strategy: How to handle multiple masks ('merge', 'largest', 'all')
            output_formats: List of output formats
            return_overlay: Whether to return overlay

        Returns:
            List of SegmentationResult instances
        """
        if not any([boxes, points, text, exemplar_box]):
            raise InvalidPromptError("Must provide either boxes or points")

        if boxes and not points:
            results = self.segment_batch(
                image, boxes=boxes, output_formats=output_formats, return_overlay=return_overlay
            )

        elif points and not boxes:
            results = self.segment_batch(
                image, points=[points], output_formats=output_formats, return_overlay=return_overlay
            )
        else:
            # Mixed prompts - combine batch results
            results = []
            if boxes:
                results.extend(
                    self.segment_batch(image, boxes=boxes, output_formats=output_formats)
                )
            if points:
                results.extend(
                    self.segment_batch(image, points=[points], output_formats=output_formats)
                )
            # Semantic prompts (SAM3)
            if text:
                results.extend(
                    self.segment_from_text(image, text, output_formats=output_formats)
                )
            
            if exemplar_box:
                results.extend(
                    self.segment_from_exemplar_box(image, exemplar_box, output_formats=output_formats)
                )

        # Apply strategy
        if strategy == "largest" and len(results) > 1:
            largest = max(results, key=lambda r: r.area)
            return [largest]
        elif strategy == "merge" and len(results) > 1:
            # Validate first result has mask
            if results[0].mask is None:
                raise BackendError("Cannot merge: numpy format required. Add 'numpy' to output_formats")
            
            merged_mask = results[0].mask.copy()
            for r in results[1:]:
                if r.mask is not None:
                    merged_mask = np.logical_or(merged_mask, r.mask)

                # Create merged result
                merged_result = self._create_result(
                    mask=merged_mask.astype(np.uint8),
                    score=np.mean([r.score for r in results]),              #type: ignore
                    output_formats=output_formats or self.config.outputs.default_formats,          #type: ignore
                    latency_ms=sum(r.latency_ms for r in results),
                    request_id=str(uuid.uuid4()),
                )
                return [merged_result]

        return results

    def segment_batch(
        self,
        image: np.ndarray | Image.Image | bytes | str | Path,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: list[list[tuple[int, int, int]]] | None = None,
        *,
        output_formats: list[str] | None = None,
        return_overlay: bool = False,
    ) -> list[SegmentationResult]:
        """Segment multiple objects in the same image efficiently.
        
        Args:
            image: Input image
            boxes: List of bounding boxes [(x1,y1,x2,y2), ...]
            points: List of point prompts [[(x,y,label), ...], ...]
            output_formats: List of output formats
            return_overlay: Whether to return overlay
            
        Returns:
            List of SegmentationResult, one per prompt
            
        Example:
            >>> boxes = [(100,100,300,300), (400,200,600,500)]
            >>> results = seg.segment_batch(image, boxes=boxes)
            >>> print(f"Segmented {len(results)} objects")
        """
        start_time = time.time()
        
        # Load image once
        img_array = load_image(image)
        
        if output_formats is None:
            output_formats = self.config.outputs.default_formats               #type: ignore
        
        if self.backend is None:
            raise BackendError("Backend not loaded")
        
        results = []
        
        # Process boxes in batch
        if boxes:
            # Validate all boxes first
            for box in boxes:
                self._validate_box(box, img_array.shape)               #type: ignore
            
            # Backend batch inference
            masks, scores = self.backend.infer_from_boxes_batch(img_array, boxes)
            total_latency = (time.time() - start_time) * 1000
            per_item_latency = total_latency / len(masks) if masks else 0

            for mask, score in zip(masks, scores):
                mask = self._postprocess_mask(mask, img_array.shape)          #type: ignore
                result = self._create_result(
                    mask=mask,
                    score=score,
                    output_formats=output_formats,               #type: ignore
                    latency_ms=per_item_latency,
                    request_id=str(uuid.uuid4()),
                )
                results.append(result)
        
        # Process points in batch
        if points:
            points_start = time.time()

            # Validate all points first
            for point_list in points:
                self._validate_points(point_list, img_array.shape)          #type: ignore
            
            # Backend batch inference
            masks, scores = self.backend.infer_from_points_batch(img_array, points)
            total_latency = (time.time() - points_start) * 1000
            per_item_latency = total_latency / len(masks) if masks else 0
            
            for mask, score in zip(masks, scores):
                mask = self._postprocess_mask(mask, img_array.shape)          #type: ignore
                result = self._create_result(
                    mask=mask,
                    score=score,
                    output_formats=output_formats,                   #type: ignore
                    latency_ms=per_item_latency,
                    request_id=str(uuid.uuid4()),
                )
                results.append(result)
        
        return results

    def warmup(self, image_size: tuple[int, int] | None = None) -> None:
        """Warm up the model with a dummy forward pass.

        Args:
            image_size: Optional image size (height, width)
        """
        if image_size is None:
            image_size = (1024, 1024)

        dummy_image = np.zeros((*image_size, 3), dtype=np.uint8)
        dummy_box = (100, 100, 200, 200)

        logger.info("Warming up model...")
        try:
            self.segment_from_box(dummy_image, dummy_box)
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def close(self) -> None:
        """Clean up resources."""
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def _validate_box(
        self, box: tuple[int, int, int, int], image_shape: tuple[int, int, int]
    ) -> None:
        """Validate bounding box coordinates."""
        x1, y1, x2, y2 = box
        h, w = image_shape[:2]

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise InvalidPromptError(f"Box {box} out of image bounds ({w}x{h})")

        if x2 <= x1 or y2 <= y1:
            raise InvalidPromptError(f"Invalid box dimensions: {box}")

    def _validate_points(
        self, points: list[tuple[int, int, int]], image_shape: tuple[int, int, int]
    ) -> None:
        """Validate point coordinates."""
        h, w = image_shape[:2]
        for x, y, label in points:
            if x < 0 or y < 0 or x >= w or y >= h:
                raise InvalidPromptError(f"Point ({x}, {y}) out of image bounds ({w}x{h})")
            if label not in (0, 1):
                raise InvalidPromptError(f"Point label must be 0 or 1, got {label}")

    def _postprocess_mask(self, mask: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
        """Apply postprocessing to mask."""
        if self.config.postprocess.remove_small_components:
            min_area = int(self.config.outputs.min_area_ratio * mask.size)
            mask = remove_small_components(mask, min_area=min_area)

        if self.config.postprocess.morphological_closing:
            kernel_size = self.config.postprocess.closing_kernel_size
            mask = apply_morphology(mask, operation="closing", kernel_size=kernel_size)

        return mask

    def _create_result(
        self,
        mask: np.ndarray,
        score: float,
        output_formats: list[str],
        latency_ms: float,
        request_id: str,
    ) -> SegmentationResult:
        """Create SegmentationResult from mask and metadata."""
        result = SegmentationResult(
            score=score,
            area=int(np.sum(mask)),
            latency_ms=latency_ms,
            model_info={
                "name": self.config.model.name,
                "backend": self.config.runtime.backend,
                "device": self.config.runtime.device,
            },
            request_id=request_id,
        )

        # Compute bbox
        if np.any(mask):
            coords = np.argwhere(mask)
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            result.bbox = (int(x1), int(y1), int(x2), int(y2))

        # Generate requested outputs
        for fmt in output_formats:
            if fmt == "numpy":
                result.mask = mask
            elif fmt == "rle":
                result.rle = mask_to_rle(mask)
            elif fmt == "polygons":
                result.polygons = mask_to_polygons(mask)
            elif fmt == "png":
                result.png_bytes = mask_to_png_bytes(mask)

        return result

    def __enter__(self) -> "Segmentor":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()