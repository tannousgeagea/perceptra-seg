"""Example API client for Segmentor service.

This demonstrates how to use Segmentor as a REST API.
"""

import base64
import io
from pathlib import Path

import requests
from PIL import Image, ImageDraw


class SegmentorClient:
    """Client for Segmentor REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _encode(self, image_path: str | Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return full health payload (status, model_loaded, device, etc.)."""
        response = requests.get(f"{self.base_url}/v1/healthz", timeout=5)
        response.raise_for_status()
        return response.json()

    # ── Interactive prompts ───────────────────────────────────────────────────

    def segment_from_box(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
        output_formats: list[str] | None = None,
    ) -> dict:
        """Segment object from bounding box (x1, y1, x2, y2) in pixel coords."""
        response = requests.post(
            f"{self.base_url}/v1/segment/box",
            headers=self.headers,
            json={
                "image": self._encode(image_path),
                "box": list(box),
                "output_formats": output_formats or ["rle", "polygons"],
            },
        )
        response.raise_for_status()
        return response.json()

    def segment_from_points(
        self,
        image_path: str | Path,
        points: list[tuple[int, int, int]],
        output_formats: list[str] | None = None,
    ) -> dict:
        """Segment object from point prompts.

        Args:
            points: List of (x, y, label) tuples — label 1 = positive, 0 = negative.
        """
        api_points = [{"x": x, "y": y, "label": label} for x, y, label in points]
        response = requests.post(
            f"{self.base_url}/v1/segment/points",
            headers=self.headers,
            json={
                "image": self._encode(image_path),
                "points": api_points,
                "output_formats": output_formats or ["rle", "polygons"],
            },
        )
        response.raise_for_status()
        return response.json()

    # ── SAM3-only endpoints ───────────────────────────────────────────────────

    def segment_from_text(
        self,
        image_path: str | Path,
        text: str,
        output_formats: list[str] | None = None,
    ) -> list[dict]:
        """Find all objects matching a natural-language description (SAM3 only).

        Args:
            text: Natural language prompt, e.g. "red car" or "person wearing hat".

        Returns:
            List of segmentation results, one per matched object.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/text",
            headers=self.headers,
            json={
                "image": self._encode(image_path),
                "text": text,
                "output_formats": output_formats or ["rle", "polygons"],
            },
        )
        response.raise_for_status()
        return response.json()

    def segment_from_exemplar(
        self,
        image_path: str | Path,
        exemplar_box: tuple[int, int, int, int],
        output_formats: list[str] | None = None,
    ) -> list[dict]:
        """Find all objects visually similar to the region inside exemplar_box (SAM3 only).

        Args:
            exemplar_box: (x1, y1, x2, y2) pixel coords of the reference object.

        Returns:
            List of segmentation results for all similar objects found.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/exemplar",
            headers=self.headers,
            json={
                "image": self._encode(image_path),
                "exemplar_box": list(exemplar_box),
                "output_formats": output_formats or ["rle", "polygons"],
            },
        )
        response.raise_for_status()
        return response.json()

    # ── Auto-segment (no prompt) ──────────────────────────────────────────────

    def segment_auto(
        self,
        image_path: str | Path,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        output_formats: list[str] | None = None,
    ) -> list[dict]:
        """Auto-segment entire image with no prompts (SAM v1/v2 only).

        Args:
            points_per_side: Grid density for automatic point sampling.
            pred_iou_thresh: Minimum predicted IoU score to keep a mask.
            stability_score_thresh: Minimum stability score to keep a mask.

        Returns:
            List of segmentation results for every detected object.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/auto",
            headers=self.headers,
            timeout=120,
            json={
                "image": self._encode(image_path),
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
                "output_formats": output_formats or ["rle", "polygons"],
            },
        )
        response.raise_for_status()
        return response.json()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_mask_image(self, result: dict) -> Image.Image | None:
        """Decode the PNG mask from a result dict, if present."""
        if "png_base64" not in result:
            return None
        return Image.open(io.BytesIO(base64.b64decode(result["png_base64"])))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_test_image(path: str = "test_api_image.png") -> str:
    """Create a simple test image with two white rectangles on a black background."""
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([150, 100, 350, 300], fill=(220, 220, 220))  # left object
    draw.rectangle([400, 150, 580, 350], fill=(180, 100, 100))  # right object (reddish)
    img.save(path)
    return path


def _print_result(label: str, result: dict | list[dict]) -> None:
    items = result if isinstance(result, list) else [result]
    print(f"  {label}: {len(items)} mask(s)")
    for i, r in enumerate(items):
        print(f"    [{i}] score={r.get('score', 0):.3f}  area={r.get('area', '?')}px  "
              f"bbox={r.get('bbox')}  latency={r.get('latency_ms', 0):.1f}ms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = SegmentorClient(base_url="http://localhost:29086")

    # Health
    print("── Health check ──────────────────────────────────────")
    try:
        info = client.health_check()
        print(f"  status       : {info.get('status')}")
        print(f"  model_loaded : {info.get('model_loaded')}")
        print(f"  model_name   : {info.get('model_name')}")
        print(f"  device       : {info.get('device')}")
        print(f"  precision    : {info.get('precision')}")
        print(f"  gpu_mem_mb   : {info.get('gpu_memory_used_mb')}")
    except Exception as e:
        print(f"  Service unavailable: {e}")
        print("  Start with: docker-compose up perceptra-seg")
        return

    if not info.get("model_loaded"):
        print("  Model not loaded yet — wait for start_period to elapse")
        return

    # Build test image
    print("\n── Creating test image ───────────────────────────────")
    img_path = _make_test_image()
    print(f"  Saved: {img_path}")

    # Box prompt
    print("\n── segment_from_box ──────────────────────────────────")
    result = client.segment_from_box(img_path, box=(150, 100, 350, 300))
    _print_result("box", result)

    # Point prompt
    print("\n── segment_from_points ───────────────────────────────")
    result = client.segment_from_points(
        img_path,
        points=[(250, 200, 1), (500, 50, 0)],  # positive inside, negative outside
    )
    _print_result("points", result)

    # Text prompt (SAM3 only — will return 501 on v1/v2)
    print("\n── segment_from_text (SAM3) ──────────────────────────")
    try:
        results = client.segment_from_text(img_path, text="white rectangle")
        _print_result("text", results)
    except requests.HTTPError as e:
        print(f"  Skipped ({e.response.status_code}): {e.response.json().get('detail')}")

    # Exemplar prompt (SAM3 only — will return 501 on v1/v2)
    print("\n── segment_from_exemplar (SAM3) ──────────────────────")
    try:
        results = client.segment_from_exemplar(img_path, exemplar_box=(150, 100, 350, 300))
        _print_result("exemplar", results)
    except requests.HTTPError as e:
        print(f"  Skipped ({e.response.status_code}): {e.response.json().get('detail')}")

    # Auto-segment (SAM v1/v2 — will return 501 on SAM3)
    print("\n── segment_auto (SAM v1/v2) ──────────────────────────")
    try:
        results = client.segment_auto(
            img_path,
            points_per_side=16,        # lower for speed in this demo
            pred_iou_thresh=0.85,
            stability_score_thresh=0.90,
        )
        _print_result("auto", results)
    except requests.HTTPError as e:
        print(f"  Skipped ({e.response.status_code}): {e.response.json().get('detail')}")

    # Cleanup
    Path(img_path).unlink(missing_ok=True)
    print("\n✓ Done")


if __name__ == "__main__":
    main()
