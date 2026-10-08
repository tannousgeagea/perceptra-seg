"""Example: calling a deployed perceptra-seg service with the packaged REST client.

    docker compose up -d          # or: uvicorn service.main:app --port 8080
    python examples/api_client.py --url http://localhost:29086 --api-key <key>
"""

import argparse

from PIL import Image, ImageDraw

from perceptra_seg.client import SegmentorAPIError, SegmentorClient


def _make_test_image() -> Image.Image:
    """Two rectangles on a dark background."""
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([150, 100, 350, 300], fill=(220, 220, 220))
    draw.rectangle([400, 150, 580, 350], fill=(180, 100, 100))
    return img


def _print_result(label: str, result: dict | list[dict]) -> None:
    items = result if isinstance(result, list) else [result]
    print(f"  {label}: {len(items)} mask(s)")
    for i, r in enumerate(items):
        print(f"    [{i}] score={r.get('score', 0):.3f}  area={r.get('area', '?')}px  "
              f"bbox={r.get('bbox')}  latency={r.get('latency_ms', 0):.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:29086")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    client = SegmentorClient(args.url, api_key=args.api_key)

    print("── Health check ──────────────────────────────────────")
    info = client.health()
    print(f"  status        : {info['status']}")
    print(f"  primary_model : {info['primary_model']}")
    print(f"  build         : {info.get('build')}")
    for name, minfo in info["models"].items():
        print(f"  {name}: device={minfo['device']} precision={minfo['precision']}")
    if not info["models"]:
        print("  No models loaded")
        return

    image = _make_test_image()

    for model in info["models"]:
        print(f"\n── {model} ─────────────────────────────────────────")
        _print_result("box", client.segment_box(image, (150, 100, 350, 300), model=model))
        _print_result("points", client.segment_points(image, [(250, 200, 1), (500, 50, 0)], model=model))

        try:
            _print_result("text", client.segment_text(image, "rectangle", model=model))
            _print_result("exemplar", client.segment_exemplar(image, (150, 100, 350, 300), model=model))
            batch = client.segment_text_batch(image, ["rectangle", "red rectangle"], model=model)
            for text, results in batch.items():
                _print_result(f"text batch [{text}]", results)
        except SegmentorAPIError as e:
            print(f"  concept prompts skipped ({e.status_code}): {e.detail}")

        try:
            _print_result("auto", client.segment_auto(image, points_per_side=16, model=model))
        except SegmentorAPIError as e:
            print(f"  auto skipped ({e.status_code}): {e.detail}")


if __name__ == "__main__":
    main()
