from perceptra_seg import Segmentor
from perceptra_seg.utils.visualization import visualize_results
from PIL import Image
from pathlib import Path
import numpy as np



def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")

    # Initialize SAM3 segmentor
    seg = Segmentor(model="sam_v3", device="cuda")

    # Define exemplar box around a reference object (e.g., a specific bolt)
    exemplar_box = (480, 290, 590, 650)  # (x1, y1, x2, y2) around reference bolt

    # Find all similar objects in the image
    results = seg.segment_from_exemplar_box(
        image,
        box=exemplar_box,
        output_formats=["numpy", "rle"]
    )

    print(f"Found {len(results)} similar objects")
    for i, result in enumerate(results):
        print(f"  Object {i+1}: score={result.score:.3f}, area={result.area}")

    # Visualize all detected instances
    vis = visualize_results(image, results, mode="both", show_scores=True)
    vis.save("exemplar_detections.jpg")

if __name__ == "__main__":
    main()