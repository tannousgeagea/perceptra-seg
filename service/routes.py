"""API route handlers."""

import base64
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from perceptra_seg.exceptions import InvalidPromptError, SegmentorError

logger = logging.getLogger(__name__)

router = APIRouter()


# Request models
class PointPrompt(BaseModel):
    """Point prompt with coordinates and label."""

    x: int
    y: int
    label: int = Field(..., ge=0, le=1, description="1 for positive, 0 for negative")


class SegmentBoxRequest(BaseModel):
    """Request for box-based segmentation."""

    image: str = Field(..., description="Base64-encoded image or URL")
    box: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    output_formats: list[str] = Field(default=["rle"], description="Output formats")
    strategy: str = Field(default="largest", description="Strategy for multiple masks")


class SegmentPointsRequest(BaseModel):
    """Request for point-based segmentation."""

    image: str
    points: list[PointPrompt]
    output_formats: list[str] = Field(default=["rle"])


class SegmentRequest(BaseModel):
    """General segmentation request."""

    image: str
    boxes: list[list[int]] | None = None
    points: list[PointPrompt] | None = None
    strategy: str = Field(default="largest")
    output_formats: list[str] = Field(default=["rle"])


# Response models
class SegmentationResponse(BaseModel):
    """Segmentation response."""

    rle: dict[str, Any] | None = None
    png_base64: str | None = None
    polygons: list[list[list[float]]] | None = None
    score: float
    area: int
    bbox: list[int] | None = None
    latency_ms: float
    model_info: dict[str, Any]
    request_id: str


# Dependency functions
async def get_segmentor(request: Request) -> Any:
    """Get Segmentor instance from app state."""
    if not request.app.state.segmentor:
        raise HTTPException(status_code=503, detail="Segmentor not initialized")
    return request.app.state.segmentor


async def verify_api_key(request: Request) -> None:
    """Verify API key if configured."""
    config = request.app.state.config
    if not config.server.api_keys:
        return  # No auth required

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    token = auth_header[7:]  # Remove "Bearer "
    if token not in config.server.api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


def decode_image(image_str: str) -> bytes:
    """Decode base64 image or fetch from URL."""
    if image_str.startswith(("http://", "https://")):
        # Return URL string to be handled by load_image
        return image_str  # type: ignore

    # Decode base64
    try:
        # Remove data URL prefix if present
        if "base64," in image_str:
            image_str = image_str.split("base64,")[1]
        return base64.b64decode(image_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


class SegmentTextRequest(BaseModel):
    """Request for text-prompt segmentation (SAM3 only)."""

    image: str = Field(..., description="Base64-encoded image or URL")
    text: str = Field(..., min_length=1, max_length=500, description="Natural language prompt")
    output_formats: list[str] = Field(default=["rle", "polygons"])


class SegmentExemplarRequest(BaseModel):
    """Request for exemplar-based similarity search (SAM3 only)."""

    image: str = Field(..., description="Base64-encoded image or URL")
    exemplar_box: list[int] = Field(..., min_length=4, max_length=4, description="[x1, y1, x2, y2] pixel coords")
    output_formats: list[str] = Field(default=["rle", "polygons"])


class SegmentAutoRequest(BaseModel):
    """Request for automatic full-image segmentation."""

    image: str = Field(..., description="Base64-encoded image or URL")
    points_per_side: int = Field(default=32, ge=8, le=64)
    pred_iou_thresh: float = Field(default=0.88, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(default=0.95, ge=0.0, le=1.0)
    output_formats: list[str] = Field(default=["rle", "polygons"])


# Routes
@router.get("/healthz")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint with model info."""
    segmentor = request.app.state.segmentor
    model_loaded = segmentor is not None

    gpu_memory_mb = None
    try:
        import torch
        if torch.cuda.is_available() and model_loaded:
            gpu_memory_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass

    config = request.app.state.config
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name": config.model.name if config else None,
        "device": config.runtime.device if config else None,
        "precision": config.runtime.precision if config else None,
        "gpu_memory_used_mb": gpu_memory_mb,
    }


@router.post("/segment/box", response_model=SegmentationResponse)
async def segment_box(
    request: SegmentBoxRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Segment object from bounding box."""
    try:
        start_time = time.time()

        # Decode image
        image_data = decode_image(request.image)

        # Segment
        result = segmentor.segment_from_box(
            image=image_data,
            box=tuple(request.box),
            output_formats=request.output_formats,
        )

        return result.to_dict()

    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment_box")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/segment/points", response_model=SegmentationResponse)
async def segment_points(
    request: SegmentPointsRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Segment object from point prompts."""
    try:
        image_data = decode_image(request.image)

        # Convert points
        points = [(p.x, p.y, p.label) for p in request.points]

        result = segmentor.segment_from_points(
            image=image_data,
            points=points,
            output_formats=request.output_formats,
        )

        return result.to_dict()

    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment_points")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/segment/text", response_model=list[SegmentationResponse])
async def segment_text(
    request: SegmentTextRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Segment objects matching a text prompt (SAM3 only)."""
    try:
        image_data = decode_image(request.image)
        results = segmentor.segment_from_text(
            image=image_data,
            text=request.text,
            output_formats=request.output_formats,
        )
        return [r.to_dict() for r in results]
    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment_text")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/segment/exemplar", response_model=list[SegmentationResponse])
async def segment_exemplar(
    request: SegmentExemplarRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Find all objects visually similar to the exemplar box (SAM3 only)."""
    try:
        image_data = decode_image(request.image)
        results = segmentor.segment_from_exemplar_box(
            image=image_data,
            box=tuple(request.exemplar_box),
            output_formats=request.output_formats,
        )
        return [r.to_dict() for r in results]
    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment_exemplar")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/segment/auto", response_model=list[SegmentationResponse])
async def segment_auto(
    request: SegmentAutoRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Auto-segment entire image with no prompts (SAM v1/v2 only)."""
    try:
        image_data = decode_image(request.image)
        results = segmentor.segment_auto(
            image=image_data,
            points_per_side=request.points_per_side,
            pred_iou_thresh=request.pred_iou_thresh,
            stability_score_thresh=request.stability_score_thresh,
            output_formats=request.output_formats,
        )
        return [r.to_dict() for r in results]
    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment_auto")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/segment", response_model=list[SegmentationResponse])
async def segment(
    request: SegmentRequest,
    segmentor: Any = Depends(get_segmentor),
    _auth: None = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """General segmentation endpoint supporting boxes and/or points."""
    try:
        image_data = decode_image(request.image)

        # Convert inputs
        boxes = [tuple(box) for box in request.boxes] if request.boxes else None
        points = [(p.x, p.y, p.label) for p in request.points] if request.points else None

        results = segmentor.segment(
            image=image_data,
            boxes=boxes,
            points=points,
            strategy=request.strategy,
            output_formats=request.output_formats,
        )

        return [r.to_dict() for r in results]

    except InvalidPromptError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SegmentorError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in segment")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")