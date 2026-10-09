"""Tests for the FastAPI service (no model weights needed)."""

import base64
import io
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from perceptra_seg.config import SegmentorConfig
from service.main import create_app
from tests.fakes import FakeConceptBackend, make_segmentor


def _b64_png(width: int = 100, height: int = 100) -> str:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def config() -> SegmentorConfig:
    return SegmentorConfig()


@pytest.fixture
def client(config: SegmentorConfig):
    models = {"sam_v3": make_segmentor("sam_v3"), "sam_v2": make_segmentor("sam_v2")}
    with TestClient(create_app(config, models=models)) as c:
        yield c


@pytest.fixture
def image_b64() -> str:
    return _b64_png()


def test_health_check(client: TestClient) -> None:
    data = client.get("/v1/healthz").json()
    assert data["status"] == "ok"
    assert data["primary_model"] == "sam_v3"
    assert set(data["models"]) == {"sam_v3", "sam_v2"}
    assert "perceptra_seg" in data["build"]


def test_health_degraded_without_models(config: SegmentorConfig, image_b64: str) -> None:
    with TestClient(create_app(config, models={})) as c:
        assert c.get("/v1/healthz").json()["status"] == "degraded"
        assert c.post("/v1/segment/box", json={"image": image_b64, "box": [1, 1, 5, 5]}).status_code == 503


def test_segment_box(client: TestClient, image_b64: str) -> None:
    response = client.post(
        "/v1/segment/box",
        json={"image": image_b64, "box": [20, 20, 80, 80], "output_formats": ["rle", "polygons"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["area"] == 60 * 60
    assert data["rle"]["size"] == [100, 100]
    assert data["polygons"]
    assert data["model_info"]["name"] == "sam_v3"


def test_segment_points_on_selected_model(client: TestClient, image_b64: str) -> None:
    response = client.post(
        "/v1/segment/points?model=sam_v2",
        json={"image": image_b64, "points": [{"x": 50, "y": 50, "label": 1}]},
    )
    assert response.status_code == 200
    assert response.json()["model_info"]["name"] == "sam_v2"


def test_unknown_model_is_404(client: TestClient, image_b64: str) -> None:
    response = client.post("/v1/segment/box?model=sam_v9", json={"image": image_b64, "box": [1, 1, 5, 5]})
    assert response.status_code == 404


def test_invalid_box_is_400(client: TestClient, image_b64: str) -> None:
    response = client.post("/v1/segment/box", json={"image": image_b64, "box": [0, 0, 10000, 10000]})
    assert response.status_code == 400


def test_invalid_image_is_400(client: TestClient) -> None:
    response = client.post("/v1/segment/box", json={"image": "not-base64!!", "box": [0, 0, 10, 10]})
    assert response.status_code == 400


def test_image_too_large_is_413(config: SegmentorConfig) -> None:
    config.server.max_image_dimension = 50
    with TestClient(create_app(config, models={"sam_v3": make_segmentor()})) as c:
        response = c.post("/v1/segment/box", json={"image": _b64_png(), "box": [1, 1, 5, 5]})
    assert response.status_code == 413


def test_segment_text(client: TestClient, image_b64: str) -> None:
    response = client.post("/v1/segment/text", json={"image": image_b64, "text": "square"})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_segment_text_with_box(client: TestClient, image_b64: str) -> None:
    response = client.post(
        "/v1/segment/text", json={"image": image_b64, "text": "square", "box": [20, 20, 80, 80]}
    )
    assert response.status_code == 200
    assert [r["area"] for r in response.json()] == [3600]


def test_segment_text_batch(client: TestClient, image_b64: str) -> None:
    response = client.post(
        "/v1/segment/text/batch",
        json={"image": image_b64, "texts": ["square", "corner"], "min_score": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"square", "corner"}
    assert all(len(v) == 1 for v in data.values())  # the 0.4-score hit is filtered


def test_segment_exemplar(client: TestClient, image_b64: str) -> None:
    response = client.post("/v1/segment/exemplar", json={"image": image_b64, "exemplar_box": [20, 20, 80, 80]})
    assert response.status_code == 200
    assert len(response.json()) == 1


@pytest.mark.parametrize(
    "path, payload, model",
    [
        ("/v1/segment/text", {"text": "square"}, "sam_v2"),
        ("/v1/segment/text", {"text": "square", "box": [20, 20, 80, 80]}, "sam_v2"),
        ("/v1/segment/text/batch", {"texts": ["square"]}, "sam_v2"),
        ("/v1/segment/exemplar", {"exemplar_box": [20, 20, 80, 80]}, "sam_v2"),
        ("/v1/segment/auto", {}, "sam_v3"),
    ],
)
def test_unsupported_prompt_for_model_is_400(
    client: TestClient, image_b64: str, path: str, payload: dict, model: str
) -> None:
    response = client.post(f"{path}?model={model}", json={"image": image_b64, **payload})
    assert response.status_code == 400
    assert "does not support" in response.json()["detail"]


def test_segment_general_merge(client: TestClient, image_b64: str) -> None:
    response = client.post(
        "/v1/segment",
        json={
            "image": image_b64,
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15], [50, 50, 60, 60]],
            "strategy": "merge",
            "output_formats": ["numpy", "rle"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["area"] == 100 + 100 - 25 + 100  # union of all three boxes


class TestAuth:
    @pytest.fixture
    def config(self) -> SegmentorConfig:
        cfg = SegmentorConfig()
        cfg.server.api_keys = ["key-one", "key-two"]
        return cfg

    def test_missing_token_is_401(self, client: TestClient, image_b64: str) -> None:
        assert client.post("/v1/segment/box", json={"image": image_b64, "box": [1, 1, 5, 5]}).status_code == 401

    def test_wrong_token_is_403(self, client: TestClient, image_b64: str) -> None:
        headers = {"Authorization": "Bearer key"}  # substring of a real key
        response = client.post("/v1/segment/box", json={"image": image_b64, "box": [1, 1, 5, 5]}, headers=headers)
        assert response.status_code == 403

    def test_valid_token(self, client: TestClient, image_b64: str) -> None:
        headers = {"Authorization": "Bearer key-two"}
        response = client.post("/v1/segment/box", json={"image": image_b64, "box": [1, 1, 5, 5]}, headers=headers)
        assert response.status_code == 200

    def test_health_is_public(self, client: TestClient) -> None:
        assert client.get("/v1/healthz").status_code == 200


def test_api_keys_from_env(monkeypatch: pytest.MonkeyPatch, image_b64: str) -> None:
    monkeypatch.setenv("SEGMENTOR_SERVER_API_KEYS", "alpha, beta")
    with TestClient(create_app(models={"sam_v3": make_segmentor()})) as c:
        payload = {"image": image_b64, "box": [1, 1, 5, 5]}
        assert c.post("/v1/segment/box", json=payload).status_code == 401
        assert c.post("/v1/segment/box", json=payload, headers={"Authorization": "Bearer beta"}).status_code == 200


def test_requests_to_one_model_are_serialized(config: SegmentorConfig, image_b64: str) -> None:
    backend = FakeConceptBackend(delay=0.05)
    with TestClient(create_app(config, models={"sam_v3": make_segmentor(backend=backend)})) as c:
        payload = {"image": image_b64, "box": [20, 20, 80, 80]}
        threads = [threading.Thread(target=c.post, args=("/v1/segment/box",), kwargs={"json": payload}) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert backend.max_active == 1
