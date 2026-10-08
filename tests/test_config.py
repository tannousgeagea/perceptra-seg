"""Tests for configuration loading and env overrides."""

import pytest

from perceptra_seg.config import SegmentorConfig


def test_env_overrides_are_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTOR_RUNTIME_DEVICE", "cuda:1")
    monkeypatch.setenv("SEGMENTOR_SERVER_PORT", "9000")
    monkeypatch.setenv("SEGMENTOR_SERVER_API_KEYS", "a, b,,c")
    monkeypatch.setenv("SEGMENTOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("SEGMENTOR_THRESHOLDS_CONCEPT_CONFIDENCE_THRESHOLD", "0.3")
    cfg = SegmentorConfig()
    cfg.apply_env_overrides()
    assert cfg.runtime.device == "cuda:1"
    assert cfg.server.port == 9000
    assert cfg.server.api_keys == ["a", "b", "c"]
    assert cfg.cache.enabled is False
    assert cfg.thresholds.concept_confidence_threshold == 0.3


def test_env_overrides_ignore_unknown_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTOR_MODEL_NAMES", "sam_v2,sam_v3")  # service-level, not a config field
    monkeypatch.setenv("SEGMENTOR_SAM3_CHECKPOINT", "/x.pt")
    cfg = SegmentorConfig()
    cfg.apply_env_overrides()
    assert cfg.model.name == "sam_v1"


def test_env_overrides_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEGMENTOR_MODEL_NAME", "sam_v9")
    with pytest.raises(ValueError):
        SegmentorConfig().apply_env_overrides()
