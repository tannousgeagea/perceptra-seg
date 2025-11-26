
from pathlib import Path
from PIL import Image, ImageDraw
from perceptra_seg import Segmentor, SegmentorConfig
from perceptra_seg.utils.visualization import visualize_results


def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")


    config = SegmentorConfig()
    config.model.name = "sam_v3"
    config.runtime.backend = "torch"
    config.runtime.device = "cuda"
    config.outputs.min_area_ratio = 0.0001
    
    # Initialize segmentor (SAM v3)
    print("\nInitializing Segmentor...")
    # seg = Segmentor(
    #     backend="torch",
    #     model="sam_v3",
    #     device="cuda"  # change to CPU if needed
    # )

    seg = Segmentor(
        config=config
    )
    # Separate results per concept
    results_dict = seg.segment_from_text_batch(
        image, 
        ["shoe", "girl", "boy"],
        min_score=0.5
    )

    print(f"shoe: {len(results_dict['shoe'])}")
    print(f"girl: {len(results_dict['girl'])}")

    # Visualize by category
    for text, results in results_dict.items():
        vis = visualize_results(image, results)
        vis.save(f"{text}_detection.jpg")

    # Merged with deduplication (handles overlapping concepts)
    all_results = seg.segment_from_text_batch_merged(
        image,
        ["shoe", "girl", "boy"],  # Overlapping concepts
        iou_threshold=0.7
    )

    for r in all_results:
        print(f"{r.text_label}: {r.score:.2f} - {r.bbox}")

if __name__ == "__main__":
    main()