"""FastAPI application factory."""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from perceptra_seg.config import SegmentorConfig
from service.middleware import LoggingMiddleware
from service.routes import router

logger = logging.getLogger(__name__)


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


def _build_config_for(model_name: str) -> SegmentorConfig:
    """
    Create a SegmentorConfig for one model: apply shared env overrides
    (device, precision, api_keys, …) then pin the model name explicitly
    so it is not overridden by SEGMENTOR_MODEL_NAME in the env.
    """
    cfg = SegmentorConfig()
    cfg.apply_env_overrides()
    cfg.model.name = model_name  # type: ignore[assignment]
    return cfg


def create_app(config: SegmentorConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application."""
    # `config` param kept for backward compat (e.g. tests); multi-model
    # loading always uses env vars at startup.
    base_config = config or SegmentorConfig()

    app = FastAPI(
        title="Segmentor API",
        description="Production segmentation service with SAM models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
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
    app.state.models: dict = {}
    app.state.primary_model: str | None = None
    # Legacy single-model attr — always points to the primary.
    app.state.segmentor = None

    app.include_router(router, prefix="/v1")

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.on_event("startup")
    async def startup_event() -> None:
        from perceptra_seg import Segmentor

        model_names = _parse_model_names()
        logger.info("Loading models: %s", model_names)

        models: dict = {}
        for name in model_names:
            try:
                cfg = _build_config_for(name)
                seg = Segmentor(config=cfg)
                models[name] = seg
                logger.info("Model loaded: %s (device=%s precision=%s)",
                            name, cfg.runtime.device, cfg.runtime.precision)
            except Exception:
                logger.exception("Failed to load model '%s' — skipping", name)

        if not models:
            logger.error("No models loaded — service will return 503 on inference requests")

        app.state.models = models
        app.state.primary_model = model_names[0] if models else None
        app.state.segmentor = models.get(model_names[0]) if models else None

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        for name, seg in app.state.models.items():
            try:
                seg.close()
                logger.info("Model closed: %s", name)
            except Exception:
                logger.exception("Error closing model '%s'", name)

    return app


# For uvicorn: uvicorn service.main:app
app = create_app()
