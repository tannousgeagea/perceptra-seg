from pathlib import Path
from PIL import Image, ImageDraw
from perceptra_seg import Segmentor


def save_result_visuals(image, result, prefix):
    """Helper to save mask + bounding box visualizations."""
    # Save binary mask
    if result.mask is not None:
        mask_img = Image.fromarray(result.mask * 255)
        mask_img.save(f"{prefix}_mask.png")
        print(f"✓ Saved mask to {prefix}_mask.png")

    # Save image with bbox
    if result.bbox:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(result.bbox, outline="red", width=3)
        img_copy.save(f"{prefix}_image.png")
        print(f"✓ Saved image to {prefix}_image.png")


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
    seg = Segmentor(
        backend="torch",
        model="sam_v3",
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

    print(f"Score: {box_result.score:.3f}")
    print(f"Area: {box_result.area}")
    print(f"BBox: {box_result.bbox}")
    print(f"Latency: {box_result.latency_ms:.1f}ms")

    save_result_visuals(image, box_result, "example_box")

    # -------------------------------------------------------------------------
    # EXAMPLE 2: Text Prompt (Concept Segmentation)
    # -------------------------------------------------------------------------
    print("\n--- Example 2: Text Prompt Segmentation ---")
    text_results = seg.segment_from_text(
        image,
        text=text_prompt,
        output_formats=["numpy"]
    )

    for i, r in enumerate(text_results):
        print(f"[{i}] Score: {r.score:.3f}, Area: {r.area}, BBox: {r.bbox}")
        save_result_visuals(image, r, f"example_text_{i}")

    # -------------------------------------------------------------------------
    # EXAMPLE 3: Exemplar Box (Find Similar Objects)
    # -------------------------------------------------------------------------
    print("\n--- Example 3: Exemplar Box (Visual Prompt) ---")
    exemplar_results = seg.segment_from_exemplar_box(
        image,
        exemplar_box,
        output_formats=["numpy", "png"]
    )

    for i, r in enumerate(exemplar_results):
        print(f"[{i}] Score: {r.score:.3f}, Area: {r.area}, BBox: {r.bbox}")
        save_result_visuals(image, r, f"example_exemplar_{i}")

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

    for i, r in enumerate(combined_results):
        print(f"[{i}] Score: {r.score:.3f}, Area: {r.area}, BBox: {r.bbox}")
        save_result_visuals(image, r, f"example_text_box_{i}")


if __name__ == "__main__":
    main()
