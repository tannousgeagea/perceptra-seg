"""API route handlers."""

import base64
import binascii
import io
import logging
import secrets
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from service.gate import QUEUE_DEPTH, REJECTED, OverloadedError

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class LoadedModel:
    """A loaded Segmentor plus admission control for it.

    The segmentor's backend is wrapped in a ``GpuGate``, so only the model call itself is
    exclusive; image loading and post-processing of concurrent requests overlap. At most
    ``max_pending`` requests are admitted per model (running + waiting); beyond that the
    request is rejected with 429 before it takes a worker thread.
    """

    name: str
    segmentor: Any
    max_pending: int = 16
    pending: int = 0  # only touched on the event loop

    async def run(self, method: str, **kwargs: Any) -> Any:
        if self.pending >= self.max_pending:
            REJECTED.labels(self.name, "queue_full").inc()
            raise OverloadedError(self.name)
        self.pending += 1
        QUEUE_DEPTH.labels(self.name).inc()
        try:
            return await run_in_threadpool(getattr(self.segmentor, method), **kwargs)
        finally:
            self.pending -= 1
            QUEUE_DEPTH.labels(self.name).dec()


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
    points: list[PointPrompt] = Field(..., min_length=1)
    output_formats: list[str] = Field(default=["rle"])


class SegmentRequest(BaseModel):
    """General segmentation request."""

    image: str
    boxes: list[list[int]] | None = None
    points: list[PointPrompt] | None = None
    strategy: str = Field(default="largest")
    output_formats: list[str] = Field(default=["rle"])


class SegmentTextRequest(BaseModel):
    """Request for text-prompt segmentation (SAM3 only)."""

    image: str = Field(..., description="Base64-encoded image or URL")
    text: str = Field(..., min_length=1, max_length=500, description="Natural language prompt")
    box: list[int] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional [x1, y1, x2, y2] positive exemplar refining the text prompt",
    )
    output_formats: list[str] = Field(default=["rle", "polygons"])


class SegmentTextBatchRequest(BaseModel):
    """Several text prompts against one image, sharing a single encoding (SAM3 only)."""

    image: str = Field(..., description="Base64-encoded image or URL")
    texts: list[str] = Field(..., min_length=1, max_length=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
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
async def verify_api_key(request: Request) -> None:
    """Verify the bearer token if API keys are configured."""
    config = request.app.state.config
    if not config.server.api_keys:
        return  # No auth required

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )

    token = auth_header[7:].encode()
    if not any(secrets.compare_digest(token, key.encode()) for key in config.server.api_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


async def get_model(
    request: Request,
    model: str | None = Query(
        None,
        description="Model to use (e.g. 'sam_v2', 'sam_v3'). Omit to use the primary model.",
    ),
) -> LoadedModel:
    """Resolve the requested model."""
    models: dict[str, LoadedModel] = request.app.state.models
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_name = model or request.app.state.primary_model
    entry = models.get(model_name)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not loaded. Available: {sorted(models.keys())}",
        )
    return entry


def decode_image(image_str: str, request: Request) -> bytes | str:
    """Decode a base64 image (enforcing size limits) or pass a URL through."""
    if image_str.startswith(("http://", "https://")):
        return image_str  # fetched by load_image

    server = request.app.state.config.server
    if "base64," in image_str:
        image_str = image_str.split("base64,", 1)[1]
    if len(image_str) * 3 / 4 > server.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {server.max_image_size_mb} MB")
    try:
        data = base64.b64decode(image_str, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    try:
        width, height = Image.open(io.BytesIO(data)).size  # header only, no full decode
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {e}")
    if max(width, height) > server.max_image_dimension:
        raise HTTPException(
            status_code=413,
            detail=f"Image {width}x{height} exceeds max dimension {server.max_image_dimension}",
        )
    return data


# Routes
@router.get("/healthz")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint — reports all loaded models."""
    models: dict[str, LoadedModel] = request.app.state.models
    primary: str | None = request.app.state.primary_model

    try:
        import torch
        gpu_total_mb = round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else None
    except Exception:
        gpu_total_mb = None

    models_info = {
        name: {
            "loaded": True,
            "device": entry.segmentor.config.runtime.device,
            "precision": entry.segmentor.config.runtime.precision,
            "pending": entry.pending,
            "max_pending": entry.max_pending,
        }
        for name, entry in models.items()
    }

    # Backward-compat flat fields for the primary model
    primary_seg = models[primary].segmentor if primary in models else None
    return {
        "status": "ok" if models else "degraded",
        "primary_model": primary,
        "models": models_info,
        "gpu_memory_used_mb": gpu_total_mb,
        "build": request.app.state.build_info,
        # legacy fields
        "model_loaded": bool(models),
        "model_name": primary,
        "device": primary_seg.config.runtime.device if primary_seg else None,
        "precision": primary_seg.config.runtime.precision if primary_seg else None,
    }


@router.post("/segment/box", response_model=SegmentationResponse)
async def segment_box(
    body: SegmentBoxRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> dict[str, Any]:
    """Segment object from bounding box."""
    result = await model.run(
        "segment_from_box",
        image=decode_image(body.image, request),
        box=tuple(body.box),
        output_formats=body.output_formats,
    )
    return result.to_dict()


@router.post("/segment/points", response_model=SegmentationResponse)
async def segment_points(
    body: SegmentPointsRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> dict[str, Any]:
    """Segment object from point prompts."""
    result = await model.run(
        "segment_from_points",
        image=decode_image(body.image, request),
        points=[(p.x, p.y, p.label) for p in body.points],
        output_formats=body.output_formats,
    )
    return result.to_dict()


@router.post("/segment/text", response_model=list[SegmentationResponse])
async def segment_text(
    body: SegmentTextRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> list[dict[str, Any]]:
    """Segment all objects matching a text prompt, optionally refined by an exemplar box (SAM3 only)."""
    image = decode_image(body.image, request)
    if body.box is not None:
        results = await model.run(
            "segment_from_text_and_box",
            image=image,
            text=body.text,
            box=tuple(body.box),
            output_formats=body.output_formats,
        )
    else:
        results = await model.run(
            "segment_from_text", image=image, text=body.text, output_formats=body.output_formats
        )
    return [r.to_dict() for r in results]


@router.post("/segment/text/batch", response_model=dict[str, list[SegmentationResponse]])
async def segment_text_batch(
    body: SegmentTextBatchRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> dict[str, list[dict[str, Any]]]:
    """Segment several concepts in one image with a shared encoding (SAM3 only)."""
    results = await model.run(
        "segment_from_text_batch",
        image=decode_image(body.image, request),
        text_prompts=body.texts,
        min_score=body.min_score,
        output_formats=body.output_formats,
    )
    return {text: [r.to_dict() for r in items] for text, items in results.items()}


@router.post("/segment/exemplar", response_model=list[SegmentationResponse])
async def segment_exemplar(
    body: SegmentExemplarRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> list[dict[str, Any]]:
    """Find all objects visually similar to the exemplar box (SAM3 only)."""
    results = await model.run(
        "segment_from_exemplar_box",
        image=decode_image(body.image, request),
        box=tuple(body.exemplar_box),
        output_formats=body.output_formats,
    )
    return [r.to_dict() for r in results]


@router.post("/segment/auto", response_model=list[SegmentationResponse])
async def segment_auto(
    body: SegmentAutoRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> list[dict[str, Any]]:
    """Auto-segment entire image with no prompts (SAM v1/v2 only)."""
    results = await model.run(
        "segment_auto",
        image=decode_image(body.image, request),
        points_per_side=body.points_per_side,
        pred_iou_thresh=body.pred_iou_thresh,
        stability_score_thresh=body.stability_score_thresh,
        output_formats=body.output_formats,
    )
    return [r.to_dict() for r in results]


@router.post("/segment", response_model=list[SegmentationResponse])
async def segment(
    body: SegmentRequest,
    request: Request,
    _auth: None = Depends(verify_api_key),
    model: LoadedModel = Depends(get_model),
) -> list[dict[str, Any]]:
    """General segmentation endpoint supporting boxes and/or points."""
    results = await model.run(
        "segment",
        image=decode_image(body.image, request),
        boxes=[tuple(box) for box in body.boxes] if body.boxes else None,
        points=[(p.x, p.y, p.label) for p in body.points] if body.points else None,
        strategy=body.strategy,
        output_formats=body.output_formats,
    )
    return [r.to_dict() for r in results]
