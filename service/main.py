"""FastAPI application factory."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from perceptra_seg.__version__ import __version__
from perceptra_seg.config import SegmentorConfig, ServerConfig
from perceptra_seg.exceptions import (
    ImageLoadError,
    InvalidPromptError,
    SegmentorError,
    UnsupportedOperationError,
)
from service.gate import GpuGate, OverloadedError, QueueTimeoutError
from service.middleware import LoggingMiddleware
from service.routes import LoadedModel, router

logger = logging.getLogger(__name__)


# Written by the Dockerfile: the upstream sam3 commit the image was built from.
_BUILD_INFO_PATH = Path(__file__).resolve().parent.parent / "build-info.json"


def _build_info() -> dict[str, Any]:
    info: dict[str, Any] = {"perceptra_seg": __version__}
    try:
        info.update(json.loads(_BUILD_INFO_PATH.read_text()))
    except (OSError, ValueError):
        pass
    return info


def _parse_model_names() -> list[str]:
    """
    Read SEGMENTOR_MODEL_NAMES (comma-separated) or fall back to
    SEGMENTOR_MODEL_NAME (legacy singular) then config default.
    """
    multi = os.getenv("SEGMENTOR_MODEL_NAMES", "").strip()
    if multi:
        return [n.strip() for n in multi.split(",") if n.strip()]
    single = os.getenv("SEGMENTOR_MODEL_NAME", "").strip()
    return [single] if single else ["sam_v2"]


def load_config() -> SegmentorConfig:
    """Base config: optional YAML file (SEGMENTOR_CONFIG) + SEGMENTOR_* env overrides."""
    path = os.getenv("SEGMENTOR_CONFIG", "").strip()
    cfg = SegmentorConfig.from_yaml(path) if path else SegmentorConfig()
    cfg.apply_env_overrides()
    return cfg


def _build_config_for(base: SegmentorConfig, model_name: str) -> SegmentorConfig:
    """Copy of the shared config pinned to one model."""
    cfg = base.model_copy(deep=True)
    cfg.model.name = model_name  # type: ignore[assignment]
    return cfg


def _serve(name: str, segmentor: Any, server: ServerConfig) -> LoadedModel:
    """Gate the segmentor's backend and wrap it with admission control."""
    if segmentor.backend is not None and not isinstance(segmentor.backend, GpuGate):
        segmentor.backend = GpuGate(name, segmentor.backend, server.request_timeout)
    return LoadedModel(name=name, segmentor=segmentor, max_pending=server.max_queue_per_model)


def _load_models(base: SegmentorConfig, names: list[str]) -> dict[str, LoadedModel]:
    from perceptra_seg import Segmentor

    models: dict[str, LoadedModel] = {}
    for name in names:
        try:
            cfg = _build_config_for(base, name)
            models[name] = _serve(name, Segmentor(config=cfg), base.server)
            logger.info("Model loaded: %s (device=%s precision=%s)",
                        name, cfg.runtime.device, cfg.runtime.precision)
        except Exception:
            logger.exception("Failed to load model '%s' — skipping", name)
    return models


def create_app(
    config: SegmentorConfig | None = None,
    models: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Service config. Defaults to ``load_config()`` (YAML + env overrides).
        models: Pre-built ``{name: Segmentor}`` mapping; skips loading at startup (tests,
            embedding). When omitted, models listed in SEGMENTOR_MODEL_NAMES are loaded.
    """
    base_config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if models is not None:
            loaded = {name: _serve(name, seg, base_config.server) for name, seg in models.items()}
            primary = next(iter(loaded), None)
        else:
            names = _parse_model_names()
            logger.info("Loading models: %s", names)
            loaded = _load_models(base_config, names)
            primary = next((n for n in names if n in loaded), None)
            if not loaded:
                logger.error("No models loaded — service will return 503 on inference requests")

        # Every admitted request holds a worker thread while it runs or waits for its model;
        # size the pool so admitted requests never queue for a thread.
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(limiter.total_tokens, sum(m.max_pending for m in loaded.values()) + 8)

        app.state.models = loaded
        app.state.primary_model = primary
        try:
            yield
        finally:
            for name, entry in loaded.items():
                try:
                    entry.segmentor.close()
                    logger.info("Model closed: %s", name)
                except Exception:
                    logger.exception("Error closing model '%s'", name)

    app = FastAPI(
        title="Perceptra-Seg API",
        description="Production segmentation service with SAM models",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=base_config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    app.state.config = base_config
    app.state.build_info = _build_info()
    app.state.models = {}
    app.state.primary_model = None

    @app.exception_handler(InvalidPromptError)
    @app.exception_handler(UnsupportedOperationError)
    @app.exception_handler(ImageLoadError)
    async def _bad_request(_: Request, exc: SegmentorError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OverloadedError)
    async def _overloaded(_: Request, exc: OverloadedError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)}, headers={"Retry-After": "1"})

    @app.exception_handler(QueueTimeoutError)
    async def _queue_timeout(_: Request, exc: QueueTimeoutError) -> JSONResponse:
        logger.warning("%s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)}, headers={"Retry-After": "1"})

    @app.exception_handler(SegmentorError)
    async def _segmentor_error(_: Request, exc: SegmentorError) -> JSONResponse:
        logger.error("Segmentation failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(router, prefix="/v1")
    app.mount("/metrics", make_asgi_app())

    return app


# For uvicorn: uvicorn service.main:app
app = create_app()
