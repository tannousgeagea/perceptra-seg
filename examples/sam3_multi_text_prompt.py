
from pathlib import Path
from PIL import Image, ImageDraw
from perceptra_seg import Segmentor
from perceptra_seg.utils.visualization import visualize_results


def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")

    # Initialize segmentor (SAM v3)
    print("\nInitializing Segmentor...")
    seg = Segmentor(
        backend="torch",
        model="sam_v3",
        device="cuda"  # change to CPU if needed
    )

    # Separate results per concept
    results_dict = seg.segment_from_text_batch(
        image, 
        ["apple", "orange", "banana"],
        min_score=0.5
    )

    print(f"Apples: {len(results_dict['apple'])}")
    print(f"Oranges: {len(results_dict['orange'])}")

    # Visualize by category
    for text, results in results_dict.items():
        vis = visualize_results(image, results)
        vis.save(f"{text}_detection.jpg")

    # Merged with deduplication (handles overlapping concepts)
    all_results = seg.segment_from_text_batch_merged(
        image,
        ["fruit", "apple", "food"],  # Overlapping concepts
        iou_threshold=0.7
    )

    for r in all_results:
        print(f"{r.text_label}: {r.score:.2f}")