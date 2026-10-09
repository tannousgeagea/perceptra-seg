"""HTTP client for a running perceptra-seg service.

Needs only the core dependencies (no torch), so it can be used from any service that
talks to a deployed instance:

    >>> from perceptra_seg.client import SegmentorClient
    >>> client = SegmentorClient("http://seg-server:29086", api_key="...")
    >>> masks = client.segment_text("truck.jpg", "truck", model="sam_v3")
"""

import base64
import io
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from perceptra_seg.exceptions import SegmentorError

ImageInput = str | Path | bytes | np.ndarray | Image.Image


class SegmentorAPIError(SegmentorError):
    """Raised when the service returns a non-2xx response."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def encode_image(image: ImageInput) -> str:
    """Encode an image for the API. ``http(s)://`` strings are passed through as URLs."""
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        return image
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    elif isinstance(image, bytes):
        data = image
    else:
        pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
    return base64.b64encode(data).decode()


class SegmentorClient:
    """Client for the perceptra-seg REST API (``/v1``).

    Args:
        base_url: Service root, e.g. ``http://localhost:29086``.
        api_key: Bearer token if the server has ``SEGMENTOR_SERVER_API_KEYS`` set.
        timeout: Per-request timeout in seconds.
        default_model: Model used when a call does not pass ``model``; ``None`` lets the
            server use its primary model.
        max_retries: Retries when the server is busy (429 queue full, 503 wait timeout),
            honouring ``Retry-After`` with jitter. ``0`` disables retrying.
    """

    _RETRY_STATUSES = (429, 503)

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 120.0,
        default_model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.default_model = default_model
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, model: str | None = None, **kwargs: Any) -> Any:
        model = model or self.default_model
        for attempt in range(self.max_retries + 1):
            response = self.session.request(
                method,
                f"{self.base_url}/v1{path}",
                params={"model": model} if model else None,
                timeout=self.timeout,
                **kwargs,
            )
            if response.status_code not in self._RETRY_STATUSES or attempt == self.max_retries:
                break
            time.sleep(self._retry_delay(response, attempt))
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise SegmentorAPIError(response.status_code, detail)
        return response.json()

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        try:
            base = float(response.headers.get("Retry-After", ""))
        except ValueError:
            base = 0.5 * 2**attempt
        return base * random.uniform(0.5, 1.5)

    def _post(self, path: str, payload: dict[str, Any], model: str | None) -> Any:
        return self._request("POST", path, model=model, json=payload)

    # --- Health ---

    def health(self) -> dict[str, Any]:
        """Service status and loaded models."""
        return self._request("GET", "/healthz")

    # --- Geometric prompts (all models) ---

    def segment_box(
        self,
        image: ImageInput,
        box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Segment the object inside ``box`` (x1, y1, x2, y2)."""
        payload = {"image": encode_image(image), "box": list(box), "output_formats": output_formats or ["rle"]}
        return self._post("/segment/box", payload, model)

    def segment_points(
        self,
        image: ImageInput,
        points: list[tuple[int, int, int]],
        *,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Segment from (x, y, label) points; label 1 = foreground, 0 = background."""
        payload = {
            "image": encode_image(image),
            "points": [{"x": x, "y": y, "label": label} for x, y, label in points],
            "output_formats": output_formats or ["rle"],
        }
        return self._post("/segment/points", payload, model)

    def segment(
        self,
        image: ImageInput,
        *,
        boxes: list[tuple[int, int, int, int]] | None = None,
        points: list[tuple[int, int, int]] | None = None,
        strategy: str = "largest",
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Segment with several boxes and/or one point set; ``strategy`` is largest|merge|all."""
        payload: dict[str, Any] = {
            "image": encode_image(image),
            "strategy": strategy,
            "output_formats": output_formats or ["rle"],
        }
        if boxes:
            payload["boxes"] = [list(b) for b in boxes]
        if points:
            payload["points"] = [{"x": x, "y": y, "label": label} for x, y, label in points]
        return self._post("/segment", payload, model)

    # --- Concept prompts (sam_v3) ---

    def segment_text(
        self,
        image: ImageInput,
        text: str,
        *,
        box: tuple[int, int, int, int] | None = None,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Every instance matching ``text``; optional ``box`` adds a positive exemplar."""
        payload: dict[str, Any] = {
            "image": encode_image(image),
            "text": text,
            "output_formats": output_formats or ["rle", "polygons"],
        }
        if box is not None:
            payload["box"] = list(box)
        return self._post("/segment/text", payload, model)

    def segment_text_batch(
        self,
        image: ImageInput,
        texts: list[str],
        *,
        min_score: float = 0.0,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Several concepts in one call (image encoded once); returns ``{text: [results]}``."""
        payload = {
            "image": encode_image(image),
            "texts": texts,
            "min_score": min_score,
            "output_formats": output_formats or ["rle", "polygons"],
        }
        return self._post("/segment/text/batch", payload, model)

    def segment_exemplar(
        self,
        image: ImageInput,
        exemplar_box: tuple[int, int, int, int],
        *,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Every object visually similar to the one inside ``exemplar_box``."""
        payload = {
            "image": encode_image(image),
            "exemplar_box": list(exemplar_box),
            "output_formats": output_formats or ["rle", "polygons"],
        }
        return self._post("/segment/exemplar", payload, model)

    # --- Automatic (sam_v1 / sam_v2) ---

    def segment_auto(
        self,
        image: ImageInput,
        *,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        output_formats: list[str] | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Segment everything in the image with no prompts."""
        payload = {
            "image": encode_image(image),
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "output_formats": output_formats or ["rle", "polygons"],
        }
        return self._post("/segment/auto", payload, model)

    @staticmethod
    def decode_png_mask(result: dict[str, Any]) -> np.ndarray | None:
        """Binary HxW mask from a result requested with ``output_formats=["png"]``."""
        if not result.get("png_base64"):
            return None
        return np.array(Image.open(io.BytesIO(base64.b64decode(result["png_base64"])))) > 0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "SegmentorClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
