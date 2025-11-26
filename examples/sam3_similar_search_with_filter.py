from perceptra_seg import Segmentor
from perceptra_seg.utils.visualization import visualize_results
from PIL import Image
from pathlib import Path
import numpy as np


def main():

    """Find objects similar to exemplar with filtering."""
    exemplar_box = (480, 290, 590, 650)  # (x1, y1, x2, y2) around reference bolt
    min_score = 0.9
    max_results = 50

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")

    seg = Segmentor(model="sam_v3", device="cuda")
    
    results = seg.segment_from_exemplar_box(
        image,
        box=exemplar_box,
        output_formats=["numpy", "polygons"]
    )
    
    # Filter by confidence
    filtered = [r for r in results if r.score >= min_score]
    
    # Sort by score and limit
    filtered.sort(key=lambda r: r.score, reverse=True)
    filtered = filtered[:max_results]
    
    vis = visualize_results(image, filtered, mode="both", show_scores=True)
    vis.save("exemplar_filters_detections.jpg")

# Usage
if __name__ == "__main__":
    main()