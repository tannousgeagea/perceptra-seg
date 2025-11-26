
from perceptra_seg import Segmentor, SegmentationResult, SegmentorConfig
from perceptra_seg.utils.visualization import visualize_results
from PIL import Image
from pathlib import Path
import numpy as np


def segment_from_multiple_exemplars(
    seg: Segmentor,
    image,
    exemplar_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.8,
) -> list[SegmentationResult]:
    """Search using multiple exemplars and deduplicate.
    
    Args:
        seg: Segmentor instance
        image: Input image
        exemplar_boxes: List of reference boxes for different object types
        iou_threshold: Merge threshold for overlapping detections
        
    Returns:
        Deduplicated results from all exemplars
        
    Example:
        >>> # Search for two types of components
        >>> exemplar_boxes = [
        ...     (50, 60, 100, 110),   # Type A component
        ...     (300, 400, 350, 450)  # Type B component
        ... ]
        >>> results = segment_from_multiple_exemplars(seg, img, exemplar_boxes)
    """
    from perceptra_seg.utils.mask_utils import compute_iou
    
    all_results = []
    
    # Get results for each exemplar
    for idx, box in enumerate(exemplar_boxes):
        results = seg.segment_from_exemplar_box(
            image, box, output_formats=["numpy"]
        )
        
        # Tag with exemplar ID
        for r in results:
            r.metadata["exemplar_id"] = idx
            all_results.append(r)
    
    # Sort by score
    # all_results.sort(key=lambda r: r.score, reverse=True)
    
    # Deduplicate overlaps
    kept = []
    for result in all_results:
        should_keep = True
        for kept_result in kept:
            if result.mask is not None and kept_result.mask is not None:
                iou = compute_iou(result.mask, kept_result.mask)
                if iou > iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            kept.append(result)
    
    return kept


def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")

    config = SegmentorConfig()
    config.postprocess.remove_small_components = False
    seg = Segmentor(model="sam_v3", device="cuda", config=config)

    # Define exemplars for different component types
    exemplars = [
        (480, 290, 590, 650),  # person
        (500, 596, 549, 623),  # shoe
    ]

    results = segment_from_multiple_exemplars(seg, image, exemplars)

    # Group by exemplar type
    from collections import defaultdict
    by_type = defaultdict(list)
    for r in results:
        exemplar_id = r.metadata.get("exemplar_id", -1)
        by_type[exemplar_id].append(r)

    print(f"Type 0 (person): {len(by_type[0])}")
    print(f"Type 1 (show): {len(by_type[1])}")

    # Visualize all detected instances
    vis = visualize_results(image, results, mode="boundary", show_scores=True)
    vis.save("exemplar_multiple_detections.jpg")

if __name__ == "__main__":
    main()