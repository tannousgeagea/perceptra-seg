"""Exclusive, deadline-bounded access to one model's backend, plus the queue metrics."""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

QUEUE_DEPTH = Gauge(
    "perceptra_queue_depth", "Requests admitted for a model (running + waiting)", ["model"]
)
GPU_WAIT = Histogram(
    "perceptra_gpu_wait_seconds",
    "Time a backend call waited for the model",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
INFERENCE = Histogram(
    "perceptra_inference_seconds",
    "Time a backend call held the model",
    ["model", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
REJECTED = Counter(
    "perceptra_rejected_total", "Requests rejected by admission control", ["model", "reason"]
)
IMAGE_REPEATS = Counter(
    "perceptra_image_repeat_total",
    "Backend calls on an image seen in the recent past (embedding-cache potential)",
    ["model"],
)


class OverloadedError(Exception):
    """The model's queue is full; the request was not admitted (HTTP 429)."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' is at capacity, retry later")
        self.model = model


class QueueTimeoutError(Exception):
    """A backend call waited longer than the deadline for the model (HTTP 503)."""

    def __init__(self, model: str, timeout: float) -> None:
        super().__init__(f"Model '{model}' did not become available within {timeout:g}s, retry later")
        self.model = model


class GpuGate:
    """Proxy for a backend that serialises its calls; the rest of the Segmentor runs concurrently.

    Backends keep per-image state (predictor features, SAM3 inference state), so only one
    call may run at a time. Image loading, validation and post-processing happen in the
    Segmentor outside these calls and therefore overlap across requests.
    """

    _RECENT_IMAGES = 256

    def __init__(self, name: str, backend: Any, wait_timeout: float) -> None:
        self._name = name
        self._backend = backend
        self._wait_timeout = wait_timeout
        self._lock = threading.Lock()
        self._recent: OrderedDict[bytes, None] = OrderedDict()
        self._recent_lock = threading.Lock()

    def __getattr__(self, attr: str) -> Any:
        value = getattr(self._backend, attr)
        if not callable(value):  # e.g. `generate_all = None` marks an unsupported capability
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], np.ndarray):
                self._track_image(args[0])
            start = time.perf_counter()
            if not self._lock.acquire(timeout=self._wait_timeout):
                REJECTED.labels(self._name, "timeout").inc()
                raise QueueTimeoutError(self._name, self._wait_timeout)
            GPU_WAIT.labels(self._name).observe(time.perf_counter() - start)
            try:
                with INFERENCE.labels(self._name, attr).time():
                    return value(*args, **kwargs)
            finally:
                self._lock.release()

        return call

    def _track_image(self, image: np.ndarray) -> None:
        digest = hashlib.blake2b(str(image.shape).encode(), digest_size=16)
        digest.update(np.ascontiguousarray(image).data)  # no copy for contiguous arrays
        key = digest.digest()
        with self._recent_lock:
            if key in self._recent:
                self._recent.move_to_end(key)
                IMAGE_REPEATS.labels(self._name).inc()
                return
            self._recent[key] = None
            if len(self._recent) > self._RECENT_IMAGES:
                self._recent.popitem(last=False)

    def close(self) -> None:
        self._backend.close()
