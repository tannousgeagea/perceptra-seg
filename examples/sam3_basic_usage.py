from pathlib import Path
from PIL import Image, ImageDraw
from perceptra_seg import Segmentor, SegmentorConfig
from perceptra_seg.utils.visualization import visualize_results


def main():

    print("Loading test image...")
    root = Path(__file__).resolve().parent.parent
    image = Image.open(f"{root}/assets/images/test_image.jpg")

    # Example prompts
    box = (480, 290, 590, 650)
    text_prompt = "shoe"
    exemplar_box = (480, 290, 590, 650)  # Example region to use as exemplar

    # Initialize segmentor (SAM v3)
    print("\nInitializing Segmentor...")
    config = SegmentorConfig()
    config.postprocess.remove_small_components = False
    seg = Segmentor(
        backend="torch",
        model="sam_v3",
        config=config,
        device="cuda"  # change to CPU if needed
    )

    # -------------------------------------------------------------------------
    # EXAMPLE 1: Bounding Box Segmentation
    # -------------------------------------------------------------------------
    print("\n--- Example 1: Bounding Box Segmentation ---")
    box_result = seg.segment_from_box(
        image,
        box=box,
        output_formats=["numpy", "rle", "png"]
    )

    print(f"  -> Score: {box_result.score:.3f}")
    print(f"  -> Area: {box_result.area}")
    print(f"  -> BBox: {box_result.bbox}")
    print(f"  -> Latency: {box_result.latency_ms:.1f}ms")

    vis = visualize_results(image, [box_result], mode="both", show_scores=True)
    vis.save("example_box.jpg")

    # -------------------------------------------------------------------------
    # EXAMPLE 2: Text Prompt (Concept Segmentation)
    # -------------------------------------------------------------------------
    print("\n--- Example 2: Text Prompt Segmentation ---")
    text_results = seg.segment_from_text(
        image,
        text=text_prompt,
        output_formats=["numpy"]
    )

    if len(text_results):
        print(f"  -> Score: {text_results[0].score:.3f}")
        print(f"  -> Area: {text_results[0].area}")
        print(f"  -> BBox: {text_results[0].bbox}")
        print(f"  -> Latency: {text_results[0].latency_ms:.1f}ms")

    vis = visualize_results(image, text_results, mode="both", show_scores=True)
    vis.save("example_text.jpg")

    # -------------------------------------------------------------------------
    # EXAMPLE 3: Exemplar Box (Find Similar Objects)
    # -------------------------------------------------------------------------
    print("\n--- Example 3: Exemplar Box (Visual Prompt) ---")
    exemplar_results = seg.segment_from_exemplar_box(
        image,
        exemplar_box,
        output_formats=["numpy", "png"]
    )

    vis = visualize_results(image, exemplar_results, mode="both", show_scores=True)
    vis.save("example_exemplar.jpg")

    # -------------------------------------------------------------------------
    # EXAMPLE 4: Text + Box Combined Prompt
    # -------------------------------------------------------------------------
    print("\n--- Example 4: Combined Text + Box Prompt ---")
    combined_results = seg.segment_from_text_and_box(
        image,
        text=text_prompt,
        box=box,
        output_formats=["numpy", "png"]
    )

    vis = visualize_results(image, combined_results, mode="both", show_scores=True)
    vis.save("example_text_box.jpg")


if __name__ == "__main__":
    main()
