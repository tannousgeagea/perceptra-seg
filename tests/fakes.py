"""Weight-free test doubles: a real Segmentor wired to a fake backend."""

import threading
import time

import numpy as np

from perceptra_seg import Segmentor, SegmentorConfig


def _box_mask(shape: tuple[int, ...], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    return mask


class FakeGeometricBackend:
    """Box/point prompts only, like the SAM v1/v2 backends."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def _enter(self) -> None:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._counter_lock:
            self.active -= 1

    def infer_from_box(self, image, box):
        self._enter()
        return _box_mask(image.shape, box), 0.9

    def infer_from_points(self, image, points):
        self._enter()
        x, y, _ = points[0]
        return _box_mask(image.shape, (max(x - 10, 0), max(y - 10, 0), x + 10, y + 10)), 0.8

    def infer_from_boxes_batch(self, image, boxes):
        self._enter()
        return [_box_mask(image.shape, b) for b in boxes], [0.9] * len(boxes)

    def infer_from_points_batch(self, image, points_list):
        results = [self.infer_from_points(image, pts) for pts in points_list]
        return [m for m, _ in results], [s for _, s in results]

    def generate_all(self, image, **kwargs):
        self._enter()
        h, w = image.shape[:2]
        return [
            {"segmentation": _box_mask(image.shape, (0, 0, w // 2, h // 2)).astype(bool), "predicted_iou": 0.9},
            {"segmentation": _box_mask(image.shape, (w // 2, h // 2, w, h)).astype(bool), "predicted_iou": 0.8},
        ]

    # Mirrors the real SAM v1/v2 backends, which stub concept prompts out.
    def infer_from_text(self, image, text):
        raise NotImplementedError

    def infer_from_exemplar_box(self, image, box):
        raise NotImplementedError

    def infer_from_text_and_box(self, image, text, box):
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeConceptBackend(FakeGeometricBackend):
    """Adds the SAM3 concept-prompt surface (and, like SAM3, has no auto-segmentation)."""

    generate_all = None

    def infer_from_text(self, image, text):
        self._enter()
        return [_box_mask(image.shape, (0, 0, 20, 20)), _box_mask(image.shape, (40, 40, 60, 60))], [0.95, 0.4]

    def infer_from_text_batch(self, image, text_prompts):
        out = {t: self.infer_from_text(image, t) for t in text_prompts}
        return {t: m for t, (m, _) in out.items()}, {t: s for t, (_, s) in out.items()}

    def infer_from_exemplar_box(self, image, box):
        self._enter()
        return [_box_mask(image.shape, box)], [0.85]

    def infer_from_text_and_box(self, image, text, box):
        return self.infer_from_exemplar_box(image, box)


def make_segmentor(model: str = "sam_v3", backend=None) -> Segmentor:
    """Real Segmentor (validation, post-processing, output formats) over a fake backend."""
    config = SegmentorConfig()
    config.model.name = model  # type: ignore[assignment]
    config.runtime.device = "cpu"
    config.postprocess.morphological_closing = False
    seg = Segmentor.__new__(Segmentor)
    seg.config = config
    seg.cache = None
    seg.backend = backend or (FakeConceptBackend() if model == "sam_v3" else FakeGeometricBackend())
    return seg
