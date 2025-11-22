"""PyTorch backend for SAM v3 (Official Processor Pattern)."""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import os
import hashlib
import numpy as np
import torch
from PIL import Image

from perceptra_seg.config import SegmentorConfig
from perceptra_seg.exceptions import BackendError, ModelLoadError

logger = logging.getLogger(__name__)

class TorchSAMv3Backend:
    """
    Official PyTorch implementation for SAM v3 using the Processor/State pattern.
    Matches API structure: build_sam3_image_model -> Sam3Processor -> set_image -> inference_state
    """

    def __init__(self, config: SegmentorConfig) -> None:
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.inference_state: Any = None  # Holds the embeddings/metadata for the current image
        self.device: torch.device | None = None
        self._cached_image_hash: str | None = None

    def load(self) -> None:
        """Load SAM v3 model and processor."""
        try:
            # Official imports based on your snippet
            import perceptra_seg.vendor.sam3 as sam3
            from perceptra_seg.vendor.sam3.model_builder import build_sam3_image_model
            from perceptra_seg.vendor.sam3.model.sam3_image_processor import Sam3Processor

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            device_str = self.config.runtime.device
            self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

            # We assume build_sam3_image_model accepts a checkpoint path or config
            # If the official repo allows argument-less build, it likely downloads default or uses internal config
            checkpoint_path = self._get_checkpoint_path(sam3_root)
            

            print(f"Device: {self.device}")
            logger.info(f"Building SAM 3 Image Model from {checkpoint_path}...")
            self.model = build_sam3_image_model(
                bpe_path=checkpoint_path,
                enable_inst_interactivity=True,
                # device=self.device
            )

            # Initialize the Processor
            self.processor = Sam3Processor(self.model)
            logger.info("SAM 3 Processor initialized.")

        except ImportError:
            raise ModelLoadError(
                "SAM 3 package not found. Ensure 'sam3' is installed from the official repo."
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load SAM v3: {e}") from e

    def _get_checkpoint_path(self, sam3_root:str) -> Optional[str]:
        """Get checkpoint path from config."""
        return self.config.model.checkpoint_path or f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

    # --- Image State Management ---

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute hash for image caching."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def _update_state(self, image: Union[np.ndarray, Image.Image]) -> None:
        """
        Updates the inference state for a new image.
        API: inference_state = processor.set_image(image)
        """
        try:
            # Convert numpy array to PIL as the official example uses PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                img_hash = self._compute_image_hash(image)
            else:
                pil_image = image
                img_hash = self._compute_image_hash(np.array(image))

            # Skip if same image
            if img_hash == self._cached_image_hash and self.inference_state is not None:
                return

            # The processor returns a state object containing embeddings
            self.inference_state = self.processor.set_image(pil_image)
            self.processor.reset_all_prompts(self.inference_state)
            self._cached_image_hash = img_hash

        except Exception as e:
            raise BackendError(f"Failed to set image in SAM 3 processor: {e}") from e

    # --- Geometric Prompts (Inferred from processor pattern) ---
    def infer_from_box(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float]:
        """Generate mask from bounding box."""
        try:
            self._update_state(image)
            box_np = np.array(box)

            masks, scores, _ = self.model.predict_inst(
                self.inference_state,
                point_coords=None,
                point_labels=None,
                box=box_np[None, :],
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8)
            score = float(scores[0])

            return mask, score

        except Exception as e:
            raise BackendError(f"SAM v3 inference failed: {e}") from e

    def infer_from_points(
        self, image: np.ndarray, points: list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, float]:
        """
        Geometric: Point Prompt.
        Assumes processor.set_point_prompt(state=..., points=..., labels=...) exists.
        """
        try:
            self._update_state(image)

            # Separate coords and labels
            coords = np.array([[p[0], p[1]] for p in points])
            labels = np.array([p[2] for p in points])

            masks, scores, _ = self.model.predict_inst(
                self.inference_state,
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
            )

            mask = masks[0].astype(np.uint8)
            score = float(scores[0])

            return mask, score

        except Exception as e:
            raise BackendError(f"SAM v3 inference failed: {e}") from e

    # --- Semantic Prompts (Explicit in Snippet) ---

    def infer_from_text(
        self,
        image: np.ndarray,
        text: Union[str, List[str]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Semantic: pure text prompt (Concept Segmentation).
        """
        self._update_state(image)

        output = self.processor.set_text_prompt(
            state=self.inference_state,
            prompt=text,
        )

        masks = output["masks"]
        scores = output["scores"]

        mask_list, score_list = [], []

        for m, s in zip(masks, scores):
            if hasattr(m, "cpu"): m = m.cpu().squeeze(0).numpy()
            if hasattr(s, "cpu"): s = float(s.cpu())

            mask_list.append(m.astype(np.uint8))
            score_list.append(float(s))

        return mask_list, score_list

    def infer_from_exemplar_box(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Visual exemplar box prompt — finds objects similar to the exemplar.
        Uses SAM3 processor.add_visual_exemplar_prompt() if available.
        """
        try:
            from perceptra_seg.vendor.sam3.visualization_utils import normalize_bbox
            from perceptra_seg.vendor.sam3.model.box_ops import box_xywh_to_cxcywh

            # Update image state
            pil = Image.fromarray(image)
            self._update_state(pil)

            w, h = pil.size

            box_tensor = torch.tensor(box).view(1, 4)    # xyxy
            box_xywh = torch.tensor([
                box[0],
                box[1],
                box[2] - box[0],
                box[3] - box[1],
            ]).view(1, 4)

            cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_cxcywh = normalize_bbox(cxcywh, w, h)[0].tolist()

            # Official SAM3 exemplar call
            output = self.processor.add_geometric_prompt(
                state=self.inference_state,
                box=norm_cxcywh,
                label=True,
            )

            masks = output["masks"]
            scores = output["scores"]

            out_masks, out_scores = [], []
            for m, s in zip(masks, scores):
                if hasattr(m, "cpu"): m = m.squeeze(0).cpu().numpy()
                if hasattr(s, "cpu"): s = float(s.cpu())
                out_masks.append(m.astype(np.uint8))
                out_scores.append(float(s))

            return out_masks, out_scores

        except Exception as e:
            raise BackendError(f"SAM3 exemplar inference failed: {e}") from e

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
        try:
            self._update_state(image)

            # 1. Add geometric box
            box_np = np.array(box)
            _ = self.model.predict_inst(
                self.inference_state,
                point_coords=None,
                point_labels=None,
                box=box_np[None, :],
                multimask_output=False,
            )

            # 2. Add text prompt
            output = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=text,
            )

            masks, scores = output["masks"], output["scores"]
            mask_list, score_list = [], []

            for m, s in zip(masks, scores):
                if hasattr(m, "cpu"): m = m.squeeze(0).cpu().numpy()
                if hasattr(s, "cpu"): s = float(s.cpu())
                mask_list.append(m.astype(np.uint8))
                score_list.append(float(s))

            return mask_list, score_list

        except Exception as e:
            raise BackendError(f"SAM3 text+box inference failed: {e}") from e


    # --- Batch Optimization ---

    def infer_from_boxes_batch(
        self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Batch Box Inference."""

        try:
            self._update_state(image)
            input_boxes = np.array(boxes)

            masks, scores, _ = self.model.predict_inst(
                self.inference_state,
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            # Convert to list of individual masks and scores
            mask_list = []
            score_list = []
            
            for i in range(masks.shape[0]):
                mask = masks[i, 0].cpu().numpy().astype(np.uint8)
                mask_list.append(mask)
                score_list.append(float(scores[i, 0].cpu()))
            
            return mask_list, score_list
        except Exception as e:
            raise BackendError(f"SAM v3 batch inference failed: {e}") from e

    def infer_from_points_batch(
        self, image: np.ndarray, points_list: list[list[tuple[int, int, int]]]
    ) -> tuple[list[np.ndarray], list[float]]:
        """Batch Point Inference."""
        try:
            self._update_state(image)  # Once
            
            mask_list = []
            score_list = []
            
            # Iterate over each point set (SAM3 may not support multi-prompt batching)
            for points in points_list:
                coords = np.array([[p[0], p[1]] for p in points])
                labels = np.array([p[2] for p in points])
                
                masks, scores, _ = self.model.predict_inst(
                    self.inference_state,
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
                
                mask_list.append(masks[0].astype(np.uint8))
                score_list.append(float(scores[0]))
                
            return mask_list, score_list
            
        except Exception as e:
            raise BackendError(f"SAM v3 batch inference failed: {e}") from e

    def close(self) -> None:
        """Clean up resources."""
        self.inference_state = None
        self._cached_image_hash = None  # Clear cache

        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()