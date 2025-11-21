"""Base backend protocol."""

from typing import Protocol, runtime_checkable

import numpy as np

from perceptra_seg.config import SegmentorConfig


@runtime_checkable
class BaseSAMBackend(Protocol):
    """Protocol for SAM backend implementations."""

    def __init__(self, config: SegmentorConfig) -> None:
        """Initialize backend with configuration."""
        ...

    def load(self) -> None:
        """Load model weights and initialize."""
        ...

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box.

        Args:
            image: RGB image as numpy array (HxWx3)
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        ...

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts.

        Args:
            image: RGB image as numpy array (HxWx3)
            points: List of (x, y, label) tuples

        Returns:
            Tuple of (binary_mask, confidence_score)
        """
        ...

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple bounding boxes efficiently.
        
        Args:
            image: RGB image as numpy array (HxWx3)
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Tuple of (list of binary_masks, list of confidence_scores)
        """
        ...
    
    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple point prompts efficiently.
        
        Args:
            image: RGB image as numpy array (HxWx3)
            points_list: List of point prompt lists
            
        Returns:
            Tuple of (list of binary_masks, list of confidence_scores)
        """
        ...

# --- Semantic Prompts (New in SAM 3) ---

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
        ...

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
        ...

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
        ...

    def close(self) -> None:
        """Clean up resources."""
        ...