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

    def _params(self, model: str | None) -> dict:
        """Build query params — omit `model` when None (server uses its primary)."""
        return {"model": model} if model else {}

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Return full health payload.

        Response includes:
          - status: "ok" | "degraded"
          - primary_model: name of the default model
          - models: dict of {model_name: {loaded, device, precision}}
          - gpu_memory_used_mb: total GPU memory currently allocated
        """
        response = requests.get(f"{self.base_url}/v1/healthz", timeout=5)
        response.raise_for_status()
        return response.json()

    # ── Interactive prompts ───────────────────────────────────────────────────

    def segment_from_box(
        self,
        image_path: str | Path,
        box: tuple[int, int, int, int],
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> dict:
        """Segment object from bounding box (x1, y1, x2, y2) in pixel coords.

        Args:
            model: Model to use (e.g. "sam_v2"). Omit for the server default.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/box",
            params=self._params(model),
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
        model: str | None = None,
    ) -> dict:
        """Segment object from point prompts.

        Args:
            points: List of (x, y, label) tuples — label 1 = positive, 0 = negative.
            model: Model to use. Omit for the server default.
        """
        api_points = [{"x": x, "y": y, "label": label} for x, y, label in points]
        response = requests.post(
            f"{self.base_url}/v1/segment/points",
            params=self._params(model),
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
        model: str | None = None,
    ) -> list[dict]:
        """Find all objects matching a natural-language description (SAM3 only).

        Args:
            text: Natural language prompt, e.g. "red car" or "person wearing hat".
            model: Must be "sam_v3" or a model that supports text prompts.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/text",
            params=self._params(model),
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
        model: str | None = None,
    ) -> list[dict]:
        """Find all objects visually similar to exemplar_box (SAM3 only).

        Args:
            exemplar_box: (x1, y1, x2, y2) pixel coords of the reference object.
            model: Must be "sam_v3" or a model that supports exemplar prompts.
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/exemplar",
            params=self._params(model),
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
        model: str | None = None,
    ) -> list[dict]:
        """Auto-segment entire image with no prompts (SAM v1/v2 only).

        Args:
            points_per_side: Grid density for automatic point sampling.
            pred_iou_thresh: Minimum predicted IoU score to keep a mask.
            stability_score_thresh: Minimum stability score to keep a mask.
            model: Must be "sam_v1" or "sam_v2".
        """
        response = requests.post(
            f"{self.base_url}/v1/segment/auto",
            params=self._params(model),
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

    # Health — lists all loaded models
    print("── Health check ──────────────────────────────────────")
    try:
        info = client.health_check()
        print(f"  status        : {info.get('status')}")
        print(f"  primary_model : {info.get('primary_model')}")
        print(f"  gpu_mem_mb    : {info.get('gpu_memory_used_mb')}")
        print(f"  loaded models :")
        for name, minfo in (info.get("models") or {}).items():
            print(f"    {name}: device={minfo['device']} precision={minfo['precision']}")
    except Exception as e:
        print(f"  Service unavailable: {e}")
        print("  Start with: docker-compose up perceptra-seg")
        return

    if not info.get("model_loaded"):
        print("  No models loaded yet — wait for start_period to elapse")
        return

    loaded_models: list[str] = list((info.get("models") or {}).keys())
    primary = info.get("primary_model") or loaded_models[0]

    # Build test image
    print("\n── Creating test image ───────────────────────────────")
    img_path = _make_test_image()
    print(f"  Saved: {img_path}")

    # Box + point prompts — exercise every loaded model
    for model_name in loaded_models:
        print(f"\n── segment_from_box  [{model_name}] ─────────────────────")
        try:
            result = client.segment_from_box(
                img_path, box=(150, 100, 350, 300), model=model_name
            )
            _print_result("box", result)
        except requests.HTTPError as e:
            print(f"  Error: {e.response.status_code} — {e.response.json().get('detail')}")

        print(f"\n── segment_from_points  [{model_name}] ──────────────────")
        try:
            result = client.segment_from_points(
                img_path,
                points=[(250, 200, 1), (500, 50, 0)],
                model=model_name,
            )
            _print_result("points", result)
        except requests.HTTPError as e:
            print(f"  Error: {e.response.status_code} — {e.response.json().get('detail')}")

    # Text prompt — only works on sam_v3
    print("\n── segment_from_text (sam_v3 only) ──────────────────")
    try:
        results = client.segment_from_text(
            img_path, text="white rectangle", model="sam_v3"
        )
        _print_result("text", results)
    except requests.HTTPError as e:
        print(f"  Skipped ({e.response.status_code}): {e.response.json().get('detail')}")

    # Exemplar prompt — only works on sam_v3
    print("\n── segment_from_exemplar (sam_v3 only) ──────────────")
    try:
        results = client.segment_from_exemplar(
            img_path, exemplar_box=(150, 100, 350, 300), model="sam_v3"
        )
        _print_result("exemplar", results)
    except requests.HTTPError as e:
        print(f"  Skipped ({e.response.status_code}): {e.response.json().get('detail')}")

    # Auto-segment — only works on sam_v1/v2
    print("\n── segment_auto (sam_v1/v2 only) ────────────────────")
    auto_model = next((m for m in loaded_models if m in ("sam_v1", "sam_v2")), None)
    if auto_model:
        try:
            results = client.segment_auto(
                img_path,
                points_per_side=16,
                pred_iou_thresh=0.85,
                stability_score_thresh=0.90,
                model=auto_model,
            )
            _print_result(f"auto [{auto_model}]", results)
        except requests.HTTPError as e:
            print(f"  Error: {e.response.status_code} — {e.response.json().get('detail')}")
    else:
        print("  No sam_v1/sam_v2 loaded — skipping")

    # Cleanup
    Path(img_path).unlink(missing_ok=True)
    print("\n✓ Done")


if __name__ == "__main__":
    main()
