"""PyTorch backend for SAM v3, backed by the upstream ``sam3`` package.

Install the model code from the official repository (always tracks latest):

    pip install "git+https://github.com/facebookresearch/sam3.git"

Checkpoints are gated on HuggingFace (``facebook/sam3``); either provide a
local checkpoint or set ``HF_TOKEN`` so the weights can be downloaded.
"""

import contextlib
import hashlib
import logging
import os
from collections.abc import Iterator
from importlib import resources
from typing import Any

import numpy as np
import torch
from PIL import Image

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError

logger = logging.getLogger(__name__)

# Legacy location used by older images that baked the checkpoint in at build time.
_LEGACY_CHECKPOINT = "/opt/models/sam3.pt"


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _split_masks(masks: Any, scores: Any) -> tuple[list[np.ndarray], list[float]]:
    """Flatten predictor/processor output into per-object (HxW uint8 mask, score) pairs.

    Handles the shapes produced by upstream sam3: ``(H, W)``, ``(N, H, W)`` and
    ``(N, 1, H, W)`` masks with matching ``()``, ``(N,)`` or ``(N, 1)`` scores.
    """
    masks_np = _to_numpy(masks)
    scores_np = _to_numpy(scores).reshape(-1)
    if masks_np.ndim == 2:
        masks_np = masks_np[None]
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    return (
        [(m > 0).astype(np.uint8) for m in masks_np],
        [float(s) for s in scores_np[: len(masks_np)]],
    )


def _xyxy_to_norm_cxcywh(box: tuple[int, int, int, int], w: int, h: int) -> list[float]:
    """Pixel xyxy box -> normalized [cx, cy, w, h] as expected by Sam3Processor."""
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]


class TorchSAMv3Backend:
    """SAM v3 backend using the official ``build_sam3_image_model`` + ``Sam3Processor`` API.

    Supports geometric prompts (box / points, via the SAM1-style interactive head) and
    concept prompts (text, visual exemplars, text + exemplar).
    """

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.inference_state: dict[str, Any] | None = None
        self.device: torch.device | None = None
        self._cached_image_hash: str | None = None

    # --- Lifecycle ---

    def load(self) -> None:
        """Build the SAM3 image model and processor."""
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise ModelLoadError(
                "SAM3 package not found. Install it with:\n"
                '  pip install "git+https://github.com/facebookresearch/sam3.git"'
            ) from e

        try:
            self.device = self._resolve_device()
            checkpoint_path = self._resolve_checkpoint()
            if self.config.runtime.precision != "bf16":
                logger.info("SAM3 always runs under bf16 autocast (runtime.precision=%s ignored)",
                            self.config.runtime.precision)
            logger.info(
                "Building SAM3 model on %s — checkpoint: %s",
                self.device,
                checkpoint_path or "HuggingFace (facebook/sam3)",
            )
            if self.device.type == "cuda":
                # Upstream allocates some buffers on the bare "cuda" device; make that
                # resolve to the configured GPU (e.g. cuda:1).
                torch.cuda.set_device(self.device)
            self.model = build_sam3_image_model(
                bpe_path=self._bpe_path(),
                device=self.device.type,
                checkpoint_path=checkpoint_path,
                load_from_HF=checkpoint_path is None,
                enable_inst_interactivity=True,
            ).to(self.device)
            self.processor = Sam3Processor(
                self.model,
                device=str(self.device),
                confidence_threshold=self.config.thresholds.concept_confidence_threshold,
            )
            logger.info("SAM3 processor initialized")
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v3: {e}") from e

    def _resolve_device(self) -> torch.device:
        # Upstream sam3 hard-codes CUDA in several places (position-encoding and decoder
        # caches), so there is no CPU fallback.
        requested = torch.device(self.config.runtime.device)
        if requested.type != "cuda" or not torch.cuda.is_available():
            raise ModelLoadError(
                f"SAM3 requires a CUDA GPU (runtime.device={self.config.runtime.device!r}, "
                f"cuda available={torch.cuda.is_available()})"
            )
        return requested

    @staticmethod
    def _bpe_path() -> str:
        # Resolved explicitly: upstream's fallback relies on pkg_resources, which
        # recent setuptools releases no longer ship.
        return str(resources.files("sam3") / "assets" / "bpe_simple_vocab_16e6.txt.gz")

    def _resolve_checkpoint(self) -> str | None:
        """Return a local checkpoint path, or None to download from HuggingFace.

        Resolution order:
          1. config.model.checkpoint_path (explicit override)
          2. SEGMENTOR_SAM3_CHECKPOINT env var
          3. /opt/models/sam3.pt (legacy images that baked the checkpoint in)
        """
        explicit = self.config.model.checkpoint_path or os.environ.get("SEGMENTOR_SAM3_CHECKPOINT")
        if explicit:
            if not os.path.isfile(explicit):
                raise ModelLoadError(f"SAM3 checkpoint not found: {explicit}")
            return explicit
        if os.path.isfile(_LEGACY_CHECKPOINT):
            return _LEGACY_CHECKPOINT
        return None

    @contextlib.contextmanager
    def _autocast(self) -> Iterator[None]:
        # Upstream sam3 is written for bf16 autocast (all official examples enable it) and
        # casts some activations to bf16 internally, so it cannot run in plain fp32/fp16.
        assert self.device is not None
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            yield

    # --- Image state ---

    def _update_state(self, image: np.ndarray) -> None:
        """Encode the image (skipped if it is the same as the last one) and clear prompts."""
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        try:
            if img_hash != self._cached_image_hash or self.inference_state is None:
                with self._autocast():
                    self.inference_state = self.processor.set_image(Image.fromarray(image))
                self._cached_image_hash = img_hash
            self.processor.reset_all_prompts(self.inference_state)
        except Exception as e:
            self._cached_image_hash = None
            raise BackendError(f"Failed to set image in SAM3 processor: {e}") from e

    def _concept_outputs(self) -> tuple[list[np.ndarray], list[float]]:
        assert self.inference_state is not None
        return _split_masks(self.inference_state["masks"], self.inference_state["scores"])

    # --- Geometric prompts (SAM1-style interactive head) ---

    def _predict_inst(self, **kwargs: Any) -> tuple[list[np.ndarray], list[float]]:
        with self._autocast():
            masks, scores, _ = self.model.predict_inst(
                self.inference_state, multimask_output=False, **kwargs
            )
        return _split_masks(masks, scores)

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        try:
            self._update_state(image)
            masks, scores = self._predict_inst(box=np.array(box)[None, :])
            return masks[0], scores[0]
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM v3 box inference failed: {e}") from e

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        try:
            self._update_state(image)
            masks, scores = self._predict_inst(
                point_coords=np.array([[x, y] for x, y, _ in points]),
                point_labels=np.array([label for _, _, label in points]),
            )
            return masks[0], scores[0]
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM v3 point inference failed: {e}") from e

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        try:
            self._update_state(image)
            return self._predict_inst(box=np.array(boxes))
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM v3 batch box inference failed: {e}") from e

    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        try:
            self._update_state(image)
            mask_list, score_list = [], []
            for points in points_list:
                masks, scores = self._predict_inst(
                    point_coords=np.array([[x, y] for x, y, _ in points]),
                    point_labels=np.array([label for _, _, label in points]),
                )
                mask_list.append(masks[0])
                score_list.append(scores[0])
            return mask_list, score_list
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM v3 batch point inference failed: {e}") from e

    # --- Concept prompts (text / exemplar) ---

    def infer_from_text(self, image: np.ndarray, text: str) -> tuple[list[np.ndarray], list[float]]:
        """Find every instance of the concept described by ``text``."""
        try:
            self._update_state(image)
            with self._autocast():
                self.processor.set_text_prompt(state=self.inference_state, prompt=text)
            return self._concept_outputs()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM3 text inference failed: {e}") from e

    def infer_from_text_batch(
        self, image: np.ndarray, text_prompts: list[str]
    ) -> tuple[dict[str, list[np.ndarray]], dict[str, list[float]]]:
        """Run several text prompts against one shared image encoding."""
        try:
            self._update_state(image)
            masks_dict: dict[str, list[np.ndarray]] = {}
            scores_dict: dict[str, list[float]] = {}
            for text in text_prompts:
                self.processor.reset_all_prompts(self.inference_state)
                with self._autocast():
                    self.processor.set_text_prompt(state=self.inference_state, prompt=text)
                masks_dict[text], scores_dict[text] = self._concept_outputs()
            return masks_dict, scores_dict
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM3 multi-text inference failed: {e}") from e

    def infer_from_exemplar_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Find every object visually similar to the exemplar inside ``box``."""
        try:
            self._update_state(image)
            h, w = image.shape[:2]
            with self._autocast():
                self.processor.add_geometric_prompt(
                    state=self.inference_state, box=_xyxy_to_norm_cxcywh(box, w, h), label=True
                )
            return self._concept_outputs()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM3 exemplar inference failed: {e}") from e

    def infer_from_text_and_box(
        self, image: np.ndarray, text: str, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Concept prompt combining text with a positive visual exemplar box."""
        try:
            self._update_state(image)
            h, w = image.shape[:2]
            with self._autocast():
                self.processor.set_text_prompt(state=self.inference_state, prompt=text)
                self.processor.add_geometric_prompt(
                    state=self.inference_state, box=_xyxy_to_norm_cxcywh(box, w, h), label=True
                )
            return self._concept_outputs()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"SAM3 text+box inference failed: {e}") from e

    def close(self) -> None:
        self.inference_state = None
        self._cached_image_hash = None
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
