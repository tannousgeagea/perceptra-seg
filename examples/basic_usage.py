"""Basic usage example for Segmentor.

This example shows how to segment objects using bounding boxes and points.
"""

import numpy as np
from PIL import Image

from perceptra_seg import Segmentor


def main() -> None:
    """Run basic segmentation examples."""
    
    # Create a simple test image (white square on black background)
    print("Creating test image...")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:300, 150:350] = 255  # White square
    
    # Initialize Segmentor
    print("\nInitializing Segmentor...")
    seg = Segmentor(
        backend="torch",
        model="sam_v3",
        device="cuda"  # Change to "cpu" if no GPU
    )
    
    # Example 1: Segment from bounding box
    print("\n--- Example 1: Bounding Box ---")
    result = seg.segment_from_box(
        image,
        box=(150, 100, 350, 300),  # (x1, y1, x2, y2)
        output_formats=["numpy", "rle", "png"]
    )
    
    print(f"Confidence Score: {result.score:.3f}")
    print(f"Mask Area: {result.area} pixels")
    print(f"Processing Time: {result.latency_ms:.1f}ms")
    print(f"Bounding Box: {result.bbox}")
    
    # Save mask
    mask_img = Image.fromarray(result.mask * 255)
    mask_img.save("example_box_mask.png")
    print("✓ Saved mask to example_box_mask.png")
    
    # Example 2: Segment from points
    print("\n--- Example 2: Point Prompts ---")
    result = seg.segment_from_points(
        image,
        points=[
            (250, 200, 1),  # Positive point (inside object)
            (260, 210, 1),  # Another positive point
            (50, 50, 0),    # Negative point (background)
        ],
        output_formats=["numpy", "polygons"]
    )
    
    print(f"Confidence Score: {result.score:.3f}")
    print(f"Number of Polygons: {len(result.polygons) if result.polygons else 0}")
    
    # Example 3: Multiple objects
    print("\n--- Example 3: Multiple Objects ---")
    # Add another square to the image
    image[350:400, 400:500] = 255
    
    results = seg.segment(
        image,
        boxes=[
            (150, 100, 350, 300),  # First square
            (400, 350, 500, 400),  # Second square
        ],
        strategy="all",  # Return all masks
        output_formats=["numpy"]
    )
    
    print(f"Found {len(results)} objects")
    for i, res in enumerate(results, 1):
        print(f"  Object {i}: area={res.area}, score={res.score:.3f}")
    
    # Cleanup
    seg.close()
    print("\n✓ All examples completed!")


if __name__ == "__main__":
    main()