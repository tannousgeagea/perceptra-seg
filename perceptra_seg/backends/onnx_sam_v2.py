"""ONNX Runtime backend for SAM v2."""

import logging

import numpy as np

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ONNXSAMv2Backend:
    """ONNX Runtime implementation for SAM v2."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config

    def load(self) -> None:
        """Load SAM v2 ONNX model."""
        raise ModelLoadError(
            "ONNX SAM v2 backend requires pre-exported ONNX models. "
            "This is a placeholder for future implementation."
        )

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        raise NotImplementedError

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        raise NotImplementedError
    
    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate mask from bounding box."""
        raise NotImplementedError

    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate mask from point prompts."""
        raise NotImplementedError

    def infer_from_text(
        self, image: np.ndarray, text: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Generate masks from a text prompt (Concept Segmentation).
        
        Args:
            image: RGB image (HxWx3)
            text: Natural language description (e.g., "person", "red car")

        Returns:
            Tuple of (list of binary_masks, list of confidence_scores). 
            Returns multiple masks as text prompts imply detection of all instances.
        """
        raise NotImplementedError

    def infer_from_exemplar_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Generate masks by visually searching for an exemplar object.

        Args:
            image: RGB image to search in (HxWx3)
            exemplar: RGB crop of the reference object (HxWx3)

        Returns:
            Tuple of (list of binary_masks, list of confidence_scores).
        """
        raise NotImplementedError

    def infer_from_text_and_box(
        self,
        image: np.ndarray,
        text: str,
        box: tuple[int, int, int, int],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Combined text + box prompt.
        Example: "find blue pipe objects within this box"
        """
        raise NotImplementedError


    def close(self) -> None:
        """Clean up resources."""
        pass