"""PyTorch backend for SAM v1."""

import logging
from typing import Any

import numpy as np
import torch

from perceptra_seg.backends._common import autocast, split_masks
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError
from perceptra_seg.utils.checkpoints import download_checkpoint

logger = logging.getLogger(__name__)


class TorchSAMv1Backend:
    """PyTorch implementation for SAM v1."""

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.predictor: Any = None
        self.device: torch.device | None = None

    def load(self) -> None:
        """Load SAM v1 model."""
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore

            device_str = self.config.runtime.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

            checkpoint_path = self._get_checkpoint_path()

            # Weights stay fp32; fp16/bf16 is applied per call via autocast.
            model_type = self.config.model.encoder_variant
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)

            self.predictor = SamPredictor(sam)
            if self.config.runtime.precision == "bf16":
                logger.info("SAM v1 runs bf16 precision as fp16 autocast")
            logger.info(
                f"Loaded SAM v1 ({model_type}) on {self.device} (precision={self.config.runtime.precision})"
            )

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v1: {e}") from e

    def _get_checkpoint_path(self) -> str:
        """Get or download checkpoint."""
        if self.config.model.checkpoint_path:
            return self.config.model.checkpoint_path

        # Auto-download logic
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }

        variant = self.config.model.encoder_variant
        url = checkpoint_urls.get(variant)
        if not url:
            raise ModelLoadError(f"Unknown encoder variant: {variant}")

        return download_checkpoint(url, f"sam_v1_{variant}.pth")

    def _autocast(self) -> Any:
        # segment_anything converts outputs to numpy internally, which has no bfloat16.
        precision = self.config.runtime.precision
        return autocast(self.device, "fp16" if precision == "bf16" else precision)

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
            raise BackendError(f"SAM v1 inference failed: {e}") from e

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
            raise BackendError(f"SAM v1 inference failed: {e}") from e

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Masks for several boxes in one pass via SamPredictor.predict_torch."""
        try:
            with self._autocast():
                self.predictor.set_image(image)
                input_boxes = torch.tensor(boxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, scores, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            return split_masks(masks, scores)
        except Exception as e:
            raise BackendError(f"SAM v1 batch inference failed: {e}") from e

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
            raise BackendError(f"SAM v1 batch inference failed: {e}") from e

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
            from segment_anything import SamAutomaticMaskGenerator  # type: ignore

            mask_generator = SamAutomaticMaskGenerator(
                self.predictor.model,  # type: ignore
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
            )
            with self._autocast():
                return mask_generator.generate(image)
        except Exception as e:
            raise BackendError(f"SAM v1 auto-segmentation failed: {e}") from e

    def close(self) -> None:
        """Clean up resources."""
        self.predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
