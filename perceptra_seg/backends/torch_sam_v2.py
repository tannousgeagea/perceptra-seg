"""PyTorch backend for SAM v2."""

import logging
from pathlib import Path

import numpy as np
import torch

from typing import Any
from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError

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

            sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)

            if self.config.runtime.precision == "fp16" and self.device.type == "cuda":
                sam2_model = sam2_model.half()

            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info(f"Loaded SAM v2 on {self.device}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v2: {e}") from e

    def _get_model_config(self) -> str:
        """Get SAM v2 config name."""
        variant_map = {
            "vit_h": "sam2_hiera_l",
            "vit_l": "sam2_hiera_l",
            "vit_b": "sam2_hiera_b_plus",
        }
        return variant_map.get(self.config.model.encoder_variant, "sam2_hiera_l")

    def _get_checkpoint_path(self) -> str:
        """Get or download checkpoint."""
        if self.config.model.checkpoint_path:
            return self.config.model.checkpoint_path

        checkpoint_urls = {
            "sam2_hiera_l": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "sam2_hiera_b_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        }

        model_cfg = self._get_model_config()
        url = checkpoint_urls.get(model_cfg)
        if not url:
            raise ModelLoadError(f"Unknown model config: {model_cfg}")

        cache_dir = Path.home() / ".cache" / "segmentor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / f"{model_cfg}.pt"

        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM v2 checkpoint from {url}")
            import urllib.request

            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Saved to {checkpoint_path}")

        return str(checkpoint_path)

    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        try:
            self.predictor.set_image(image)

            box_np = np.array(box)

            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False,
            )

            return masks[0].astype(np.uint8), float(scores[0])

        except Exception as e:
            raise BackendError(f"SAM v2 inference failed: {e}") from e

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from point prompts."""
        try:
            self.predictor.set_image(image)

            point_coords = np.array([[x, y] for x, y, _ in points])
            point_labels = np.array([label for _, _, label in points])

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8)
            score = float(scores[0])

            return mask, score

        except Exception as e:
            raise BackendError(f"SAM v2 inference failed: {e}") from e
        
    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple boxes efficiently using predict_torch.
        
        Args:
            image: RGB image as numpy array (HxWx3)
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Tuple of (list of binary_masks, list of confidence_scores)
        """
        try:
            import torch
            
            # Set image once (computes embedding once)
            self.predictor.set_image(image)
            
            # Convert boxes to torch tensor
            input_boxes = torch.tensor(boxes, device=self.predictor.device)
            
            # Transform boxes to the input frame
            transformed_boxes = self.predictor.transform.apply_boxes_torch(
                input_boxes, 
                image.shape[:2]
            )
            
            # Batch prediction using predict_torch
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # masks shape: (batch_size, 1, H, W)
            # scores shape: (batch_size, 1)
            
            # Convert to list of individual masks and scores
            mask_list = []
            score_list = []
            
            for i in range(masks.shape[0]):
                mask = masks[i, 0].cpu().numpy().astype(np.uint8)
                mask_list.append(mask)
                
                score = float(scores[i, 0].cpu())
                score_list.append(score)
            
            return mask_list, score_list
            
        except Exception as e:
            raise BackendError(f"SAM v2 batch inference failed: {e}") from e


    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Generate masks from multiple point prompts efficiently."""
        try:
            self.predictor.set_image(image)
            
            mask_list = []
            score_list = []
            
            for points in points_list:
                point_coords = np.array([[x, y] for x, y, _ in points])
                point_labels = np.array([label for _, _, label in points])
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
                
                mask_list.append(masks[0].astype(np.uint8))
                score_list.append(float(scores[0]))
            
            return mask_list, score_list
            
        except Exception as e:
            raise BackendError(f"SAM v2 batch inference failed: {e}") from e

    def infer_from_text(
        self, image: np.ndarray, text: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Generate masks from a text prompt (Concept Segmentation).
        
        Args:
            image: RGB image (HxWx3)
            text: Natural language description (e.g., "person", "red car")

        Returns:
            Tuple of (list of binary_masks, list of confidence_scores). 
            Returns multiple masks as text prompts imply detection of all instances.
        """
        raise NotImplementedError

    def infer_from_exemplar_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Generate masks by visually searching for an exemplar object.

        Args:
            image: RGB image to search in (HxWx3)
            exemplar: RGB crop of the reference object (HxWx3)

        Returns:
            Tuple of (list of binary_masks, list of confidence_scores).
        """
        raise NotImplementedError

    def infer_from_text_and_box(
        self,
        image: np.ndarray,
        text: str,
        box: tuple[int, int, int, int],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Combined text + box prompt.
        Example: "find blue pipe objects within this box"
        """
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()