"""Configuration management for Segmentor."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    name: Literal["sam_v1", "sam_v2", "sam_v3"] = "sam_v1"
    encoder_variant: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    checkpoint_path: str | None = None


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    backend: Literal["torch", "onnx"] = "torch"
    device: str = "cuda"
    precision: Literal["fp16", "bf16", "fp32"] = "fp32"
    batch_size: int = 1
    enable_batch_inference: bool = True
    deterministic: bool = True
    seed: int = 42


class TilingConfig(BaseModel):
    """Tiling configuration for large images."""

    enabled: bool = False
    tile_size: int = 1024
    stride: int = 256
    blend_mode: Literal["linear", "average"] = "linear"


class OutputsConfig(BaseModel):
    """Output configuration."""

    default_formats: list[Literal["rle", "png", "polygons", "numpy"]] = ["rle"]
    include_overlay: bool = False
    min_area_ratio: float = 0.001
    smooth_tolerance: float = 3.0   # Shapely buffer radius for polygon corner rounding
    simplify_tolerance: float = 1.0  # Shapely simplify tolerance (vertex reduction)


class ThresholdsConfig(BaseModel):
    """Thresholds configuration."""

    mask_threshold: float = 0.5
    iou_threshold: float = 0.88
    concept_confidence_threshold: float = 0.5  # SAM3 text/exemplar detections below this are dropped


class PostprocessConfig(BaseModel):
    """Postprocessing configuration."""

    remove_small_components: bool = True
    morphological_closing: bool = True
    closing_kernel_size: int = 5


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    max_items: int = 100
    ttl_seconds: int = 3600


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    cors_origins: list[str] = ["*"]
    api_keys: list[str] = Field(default_factory=list)
    max_image_size_mb: int = 20
    max_image_dimension: int = 8000
    request_timeout: float = 30  # seconds a request may wait for its model before a 503
    max_queue_per_model: int = 16  # admitted requests per model (running + waiting); beyond -> 429


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    log_file: str | None = None


class SegmentorConfig(BaseModel):
    """Complete Segmentor configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SegmentorConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            SegmentorConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentorConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            SegmentorConfig instance
        """
        return cls(**data)

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        ``SEGMENTOR_<SECTION>_<FIELD>=value``, e.g. ``SEGMENTOR_RUNTIME_DEVICE=cpu``.
        Values are validated/coerced by pydantic; list fields take comma-separated values
        (``SEGMENTOR_SERVER_API_KEYS=key1,key2``). Unknown keys are ignored.
        """
        prefix = "SEGMENTOR_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            section, _, field = key[len(prefix) :].lower().partition("_")
            section_obj = getattr(self, section, None)
            if not isinstance(section_obj, BaseModel) or field not in type(section_obj).model_fields:
                continue

            parsed: Any = value
            if isinstance(getattr(section_obj, field), list):
                parsed = [v.strip() for v in value.split(",") if v.strip()]
            data = section_obj.model_dump()
            data[field] = parsed
            setattr(self, section, type(section_obj).model_validate(data))
