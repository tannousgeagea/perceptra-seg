"""Configuration management for Segmentor."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


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


class ThresholdsConfig(BaseModel):
    """Thresholds configuration."""

    mask_threshold: float = 0.5
    iou_threshold: float = 0.88


class PostprocessConfig(BaseModel):
    """Postprocessing configuration."""

    remove_small_components: bool = True
    morphological_closing: bool = False
    closing_kernel_size: int = 3


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
    request_timeout: int = 30


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
        """Apply environment variable overrides."""
        # Example: SEGMENTOR_RUNTIME_DEVICE=cpu
        prefix = "SEGMENTOR_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix) :].lower().split("_")
            if len(parts) < 2:
                continue

            section = parts[0]
            field = "_".join(parts[1:])

            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, field):
                    setattr(section_obj, field, value)