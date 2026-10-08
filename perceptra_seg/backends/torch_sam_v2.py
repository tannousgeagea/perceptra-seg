"""PyTorch backend for SAM v2."""

import logging
from typing import Any

import numpy as np
import torch

from perceptra_seg.backends._common import autocast, split_masks
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError
from perceptra_seg.utils.checkpoints import download_checkpoint

logger = logging.getLogger(__name__)


class TorchSAMv2Backend:
    """PyTorch implementation for SAM v2."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.predictor: Any = None
        self.device: torch.device | None = None

    def load(self) -> None:
        """Load SAM v2 model."""
        try:
            from sam2.build_sam import build_sam2             # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor       # type: ignore

            device_str = self.config.runtime.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

            checkpoint_path = self._get_checkpoint_path()
            model_cfg = self._get_model_config()

            # Weights stay fp32; fp16/bf16 is applied per call via autocast.
            sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info(f"Loaded SAM v2 on {self.device} (precision={self.config.runtime.precision})")

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v2: {e}") from e

    def _get_model_config(self) -> str:
        """Get SAM v2 config name."""
        variant_map = {
            "vit_h": "sam2_hiera_l",
            "vit_l": "sam2_hiera_l",
            "vit_b": "sam2_hiera_b+",
        }
        return variant_map.get(self.config.model.encoder_variant, "sam2_hiera_l")

    def _get_checkpoint_path(self) -> str:
        """Get or download checkpoint."""
        if self.config.model.checkpoint_path:
            return self.config.model.checkpoint_path

        checkpoint_urls = {
            "sam2_hiera_l": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "sam2_hiera_b+": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        }

        model_cfg = self._get_model_config()
        url = checkpoint_urls.get(model_cfg)
        if not url:
            raise ModelLoadError(f"Unknown model config: {model_cfg}")

        return download_checkpoint(url, f"{model_cfg}.pt")

    def _autocast(self) -> Any:
        return autocast(self.device, self.config.runtime.precision)

    def _predict(self, image: np.ndarray, **prompt: Any) -> tuple[list[np.ndarray], list[float]]:
        with self._autocast():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(multimask_output=False, **prompt)
        return split_masks(masks, scores)

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        try:
            masks, scores = self._predict(image, box=np.array(box))
            return masks[0], scores[0]
        except Exception as e:
            raise BackendError(f"SAM v2 inference failed: {e}") from e

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        try:
            masks, scores = self._predict(
                image,
                point_coords=np.array([[x, y] for x, y, _ in points]),
                point_labels=np.array([label for _, _, label in points]),
            )
            return masks[0], scores[0]
        except Exception as e:
            raise BackendError(f"SAM v2 inference failed: {e}") from e

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Masks for several boxes in one pass (SAM2's predict() takes an (N, 4) box array)."""
        try:
            return self._predict(image, box=np.array(boxes))
        except Exception as e:
            raise BackendError(f"SAM v2 batch inference failed: {e}") from e

    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """One mask per point set; the image is encoded once."""
        try:
            mask_list, score_list = [], []
            with self._autocast():
                self.predictor.set_image(image)
                for points in points_list:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=np.array([[x, y] for x, y, _ in points]),
                        point_labels=np.array([label for _, _, label in points]),
                        multimask_output=False,
                    )
                    m, s = split_masks(masks, scores)
                    mask_list.append(m[0])
                    score_list.append(s[0])
            return mask_list, score_list
        except Exception as e:
            raise BackendError(f"SAM v2 batch inference failed: {e}") from e

    # Concept prompts are SAM3-only; the Segmentor reports these as unsupported.
    def infer_from_text(self, image: np.ndarray, text: str) -> tuple[list[np.ndarray], list[float]]:
        raise NotImplementedError

    def infer_from_exemplar_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        raise NotImplementedError

    def infer_from_text_and_box(
        self, image: np.ndarray, text: str, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        raise NotImplementedError

    def generate_all(
        self,
        image: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> list[dict]:
        """Auto-segment entire image without explicit prompts."""
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore

            mask_generator = SAM2AutomaticMaskGenerator(
                self.predictor.model,  # type: ignore
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            with self._autocast():
                return mask_generator.generate(image)
        except Exception as e:
            raise BackendError(f"SAM v2 auto-segmentation failed: {e}") from e

    def close(self) -> None:
        """Clean up resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
