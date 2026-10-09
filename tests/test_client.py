"""End-to-end test of the packaged REST client against a live (fake-model) server."""

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

from perceptra_seg.client import SegmentorAPIError, SegmentorClient
from perceptra_seg.config import SegmentorConfig
from service.main import create_app
from tests.fakes import make_segmentor


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = SegmentorConfig()
    config.server.api_keys = ["secret"]
    app = create_app(config, models={"sam_v3": make_segmentor("sam_v3"), "sam_v2": make_segmentor("sam_v2")})
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def image() -> np.ndarray:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    return img


def test_client_roundtrip(base_url: str, image: np.ndarray, tmp_path) -> None:
    with SegmentorClient(base_url, api_key="secret") as client:
        assert client.health()["status"] == "ok"

        result = client.segment_box(image, (20, 20, 80, 80), output_formats=["rle", "png"])
        assert result["area"] == 3600
        assert client.decode_png_mask(result).sum() == 3600

        path = tmp_path / "img.png"
        from PIL import Image

        Image.fromarray(image).save(path)
        assert client.segment_points(path, [(50, 50, 1)], model="sam_v2")["model_info"]["name"] == "sam_v2"
        assert len(client.segment(image, boxes=[(0, 0, 10, 10), (50, 50, 60, 60)], strategy="all")) == 2
        assert len(client.segment_text(image, "square")) == 2
        assert len(client.segment_text(image, "square", box=(20, 20, 80, 80))) == 1
        assert set(client.segment_text_batch(image, ["a", "b"])) == {"a", "b"}
        assert len(client.segment_exemplar(image, (20, 20, 80, 80))) == 1


def test_client_errors(base_url: str, image: np.ndarray) -> None:
    with pytest.raises(SegmentorAPIError) as exc:
        SegmentorClient(base_url).segment_box(image, (1, 1, 5, 5))
    assert exc.value.status_code == 401

    client = SegmentorClient(base_url, api_key="secret", default_model="sam_v2")
    with pytest.raises(SegmentorAPIError) as exc:
        client.segment_text(image, "square")
    assert exc.value.status_code == 400
    assert "sam_v3" in str(exc.value.detail)


def _response(status: int, body: str = '{"ok": true}', retry_after: str | None = None):
    import requests

    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return response


@pytest.mark.parametrize("busy_status", [429, 503])
def test_client_retries_when_busy(monkeypatch: pytest.MonkeyPatch, busy_status: int) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("perceptra_seg.client.time.sleep", sleeps.append)
    client = SegmentorClient("http://seg")
    replies = iter([_response(busy_status, '{"detail": "busy"}', retry_after="1"), _response(200)])
    monkeypatch.setattr(client.session, "request", lambda *a, **kw: next(replies))
    assert client.health() == {"ok": True}
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.5


def test_client_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("perceptra_seg.client.time.sleep", lambda _: None)
    calls = []

    def busy(*args, **kwargs):
        calls.append(1)
        return _response(429, '{"detail": "busy"}')

    client = SegmentorClient("http://seg", max_retries=0)
    monkeypatch.setattr(client.session, "request", busy)
    with pytest.raises(SegmentorAPIError) as exc:
        client.health()
    assert exc.value.status_code == 429 and len(calls) == 1

    client.max_retries = 2
    calls.clear()
    with pytest.raises(SegmentorAPIError):
        client.health()
    assert len(calls) == 3
