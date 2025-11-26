from perceptra_seg import Segmentor, SegmentationResult, SegmentorConfig
from perceptra_seg.utils.visualization import visualize_results
from PIL import Image
from pathlib import Path
import numpy as np

def quality_control_inspection(
    seg: Segmentor,
    image,
    good_exemplar_box: tuple[int, int, int, int],
    expected_count: int,
    tolerance: float = 0.1,
):
    """Automated quality control using exemplar matching.
    
    Args:
        image_path: Path to product image
        good_exemplar_box: Box around a known good component
        expected_count: Expected number of components
        tolerance: Allowed deviation ratio
    """
    
    # Find all components matching the good exemplar
    results = seg.segment_from_exemplar_box(
        image,
        box=good_exemplar_box,
        output_formats=["numpy"]
    )
    
    # Filter high-confidence detections
    high_conf = [r for r in results if r.score > 0.7]
    detected_count = len(high_conf)
    
    # Check count
    min_count = int(expected_count * (1 - tolerance))
    max_count = int(expected_count * (1 + tolerance))
    
    status = "PASS" if min_count <= detected_count <= max_count else "FAIL"
    
    print(f"Quality Check: {status}")
    print(f"  Expected: {expected_count}")
    print(f"  Detected: {detected_count}")
    print(f"  Range: [{min_count}, {max_count}]")
    
    # Visualize
    vis = visualize_results(image, high_conf, mode="both")
    vis.save(f"qc_result_{status}.jpg")
    
    return status, detected_count, high_conf


def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")


    config = SegmentorConfig()
    config.postprocess.remove_small_components = False
    seg = Segmentor(model="sam_v3", device="cuda", config=config)

    exemplar_box = (480, 290, 590, 650) 


    # Usage
    status, count, results = quality_control_inspection(
        seg=seg,
        image=image,
        good_exemplar_box=exemplar_box,  # Reference component
        expected_count=6,
        tolerance=0.1
    )

if __name__ == "__main__":
    main()