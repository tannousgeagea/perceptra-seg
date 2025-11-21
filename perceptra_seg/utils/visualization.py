"""Visualization utilities for segmentation results."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Union, Optional
import colorsys

from perceptra_seg.models import SegmentationResult


def generate_colors(n: int, saturation: float = 0.8, value: float = 0.9) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def overlay_mask(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Overlay single mask on image with transparency.
    
    Args:
        image: RGB image
        mask: Binary mask (H, W)
        color: RGB color tuple
        alpha: Transparency (0=transparent, 1=opaque)
        
    Returns:
        PIL Image with overlay
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create colored overlay
    overlay = Image.new("RGB", image.size, color)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    
    # Blend with alpha
    result = Image.composite(overlay, image, mask_img)
    return Image.blend(image, result, alpha)


def draw_mask_boundary(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Image.Image:
    """Draw mask boundary contour on image.
    
    Args:
        image: RGB image
        mask: Binary mask
        color: RGB color
        thickness: Line thickness
        
    Returns:
        PIL Image with boundary
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).copy()
    else:
        image = image.copy()
    
    # Find contours using simple boundary detection
    from scipy import ndimage
    boundary = mask.astype(np.uint8) - ndimage.binary_erosion(mask).astype(np.uint8)
    
    # Draw boundary pixels
    draw = ImageDraw.Draw(image)
    y_coords, x_coords = np.where(boundary)
    
    for y, x in zip(y_coords, x_coords):
        draw.ellipse(
            [x - thickness//2, y - thickness//2, x + thickness//2, y + thickness//2],
            fill=color
        )
    
    return image


def visualize_results(
    image: Union[np.ndarray, Image.Image],
    results: List[SegmentationResult],
    mode: str = "overlay",
    alpha: float = 0.5,
    show_boxes: bool = True,
    show_scores: bool = True,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
) -> Image.Image:
    """Visualize multiple segmentation results on image.
    
    Args:
        image: Input image
        results: List of SegmentationResult objects
        mode: "overlay", "boundary", or "both"
        alpha: Transparency for overlay mode
        show_boxes: Draw bounding boxes
        show_scores: Show confidence scores
        colors: Custom color list (auto-generated if None)
        thickness: Line/boundary thickness
        
    Returns:
        PIL Image with visualizations
        
    Example:
        >>> results = seg.segment_batch(image, boxes=boxes)
        >>> vis = visualize_results(image, results, mode="both")
        >>> vis.save("output.jpg")
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()
    
    if not results:
        return image
    
    # Generate colors
    if colors is None:
        colors = generate_colors(len(results))
    
    # Process each result
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        
        # Draw mask
        if result.mask is not None:
            if mode in ["overlay", "both"]:
                image = overlay_mask(image, result.mask, color, alpha)
            
            if mode in ["boundary", "both"]:
                image = draw_mask_boundary(image, result.mask, color, thickness)
        
        # Draw bounding box
        if show_boxes and result.bbox:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = result.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
            
            # Draw score
            if show_scores:
                text = f"{result.score:.2f}"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Text background
                bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)
    
    return image


def create_side_by_side(
    image: Union[np.ndarray, Image.Image],
    results: List[SegmentationResult],
    **kwargs,
) -> Image.Image:
    """Create side-by-side comparison (original | visualization).
    
    Args:
        image: Input image
        results: Segmentation results
        **kwargs: Arguments for visualize_results()
        
    Returns:
        Combined PIL Image
    """
    if isinstance(image, np.ndarray):
        original = Image.fromarray(image)
    else:
        original = image.copy()
    
    visualized = visualize_results(image, results, **kwargs)
    
    # Combine horizontally
    total_width = original.width + visualized.width
    combined = Image.new("RGB", (total_width, original.height))
    combined.paste(original, (0, 0))
    combined.paste(visualized, (original.width, 0))
    
    return combined


def save_visualization(
    image: Union[np.ndarray, Image.Image],
    results: List[SegmentationResult],
    output_path: str,
    **kwargs,
) -> None:
    """Convenience function to visualize and save.
    
    Args:
        image: Input image
        results: Segmentation results
        output_path: Save path
        **kwargs: Arguments for visualize_results()
    """
    vis = visualize_results(image, results, **kwargs)
    vis.save(output_path)