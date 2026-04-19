"""Mask processing utilities."""

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Convert binary mask to COCO RLE format."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"size": list(mask.shape), "counts": runs.tolist()}


def mask_to_polygons(
    mask: np.ndarray,
    tolerance: float = 1.0,
    smooth_tolerance: float = 3.0,
) -> list[list[tuple[float, float]]]:
    """Convert binary mask to smooth polygon contours.

    Pipeline:
      1. Gaussian blur + re-threshold  — rounds pixel-level staircase edges
      2. findContours (CHAIN_APPROX_NONE) — full point set for smoothing
      3. Shapely buffer(r).buffer(-r)   — rounds concave corners
      4. Shapely simplify               — reduces point count while preserving shape

    Args:
        mask: Binary mask (H×W), values 0 or 1.
        tolerance: Simplification tolerance in pixels (Shapely simplify).
        smooth_tolerance: Buffer radius in pixels for corner rounding.

    Returns:
        List of polygons, each as list of (x, y) float tuples.
    """
    if not np.any(mask):
        return []

    # 1. Smooth mask edges with a Gaussian blur before contour extraction.
    #    This converts the pixel-level staircase boundary into a smooth gradient
    #    that, after re-thresholding, yields a rounded mask edge.
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigma=1.5)
    smooth_mask = (blurred > 0.5).astype(np.uint8)

    # 2. Extract contours with all points (CHAIN_APPROX_NONE) so Shapely has
    #    enough data to smooth, rather than just segment endpoints.
    contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygons = []
    for contour in contours:
        if len(contour) < 6:
            continue

        points = [(float(p[0][0]), float(p[0][1])) for p in contour]

        try:
            poly = ShapelyPolygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)

            if poly.is_empty or poly.area < 4:
                continue

            # 3. Buffer trick: dilate then erode rounds both convex and concave
            #    corners, giving the characteristic smooth SAM mask outline.
            if smooth_tolerance > 0:
                poly = (
                    poly
                    .buffer(smooth_tolerance, join_style=1)   # round joins
                    .buffer(-smooth_tolerance, join_style=1)
                )

            if poly.is_empty:
                continue

            # Handle MultiPolygon (buffer can split the shape at thin necks)
            if poly.geom_type == "MultiPolygon":
                parts = list(poly.geoms)
            else:
                parts = [poly]

            for part in parts:
                # 4. Simplify reduces vertex count while preserving topology.
                part = part.simplify(tolerance, preserve_topology=True)
                if part.is_empty or part.area < 4:
                    continue
                coords = list(part.exterior.coords[:-1])  # drop closing duplicate
                if len(coords) >= 3:
                    polygons.append([(float(x), float(y)) for x, y in coords])

        except Exception:
            # Shapely failed — fall back to simple cv2 approximation.
            logger.debug("Shapely smoothing failed for contour, using cv2 fallback")
            approx = cv2.approxPolyDP(contour, max(tolerance, 1.0), True)
            if len(approx) >= 3:
                polygons.append([(float(p[0][0]), float(p[0][1])) for p in approx])

    return polygons


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert binary mask to PNG bytes."""
    mask_img = (mask * 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_img, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def remove_small_components(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components from mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 1
    return filtered_mask


def apply_morphology(
    mask: np.ndarray, operation: str = "closing", kernel_size: int = 5
) -> np.ndarray:
    """Apply morphological operation to mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    operations = {
        "closing": cv2.MORPH_CLOSE,
        "opening": cv2.MORPH_OPEN,
        "dilation": cv2.MORPH_DILATE,
        "erosion": cv2.MORPH_ERODE,
    }
    op = operations.get(operation, cv2.MORPH_CLOSE)
    return cv2.morphologyEx(mask.astype(np.uint8), op, kernel).astype(mask.dtype)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0
