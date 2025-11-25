"""Mask processing utilities."""

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Convert binary mask to COCO RLE format.

    Args:
        mask: Binary mask (HxW)

    Returns:
        Dictionary with 'size' and 'counts'
    """
    # Simple RLE encoding
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return {
        "size": list(mask.shape),
        "counts": runs.tolist(),
    }


def mask_to_polygons(
    mask: np.ndarray, tolerance: float = 2.0
) -> list[list[tuple[float, float]]]:
    """Convert binary mask to polygon contours.

    Args:
        mask: Binary mask (HxW)
        tolerance: Simplification tolerance

    Returns:
        List of polygons, each as list of (x, y) tuples
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # Simplify polygon
        epsilon = tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 3:
            poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
            polygons.append(poly)

    return polygons


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert binary mask to PNG bytes.

    Args:
        mask: Binary mask (HxW)

    Returns:
        PNG-encoded bytes
    """
    # Convert to 0-255 range
    mask_img = (mask * 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_img, mode="L")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components from mask.

    Args:
        mask: Binary mask (HxW)
        min_area: Minimum component area in pixels

    Returns:
        Filtered mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    # Keep only large components
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1

    return filtered_mask


def apply_morphology(
    mask: np.ndarray, operation: str = "closing", kernel_size: int = 3
) -> np.ndarray:
    """Apply morphological operation to mask.

    Args:
        mask: Binary mask (HxW)
        operation: 'closing', 'opening', 'dilation', or 'erosion'
        kernel_size: Size of structuring element

    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    operations = {
        "closing": cv2.MORPH_CLOSE,
        "opening": cv2.MORPH_OPEN,
        "dilation": cv2.MORPH_DILATE,
        "erosion": cv2.MORPH_ERODE,
    }

    op = operations.get(operation, cv2.MORPH_CLOSE)
    result = cv2.morphologyEx(mask.astype(np.uint8), op, kernel)

    return result.astype(mask.dtype)

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0