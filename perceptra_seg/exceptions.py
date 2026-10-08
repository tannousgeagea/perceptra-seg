"""Custom exceptions for Segmentor."""


class SegmentorError(Exception):
    """Base exception for all Segmentor errors."""

    pass


class ModelLoadError(SegmentorError):
    """Raised when model fails to load."""

    pass


class InvalidPromptError(SegmentorError):
    """Raised when prompt coordinates are invalid."""

    pass


class BackendError(SegmentorError):
    """Raised when backend operation fails."""

    pass


class ConfigError(SegmentorError):
    """Raised when configuration is invalid."""

    pass


class ImageLoadError(SegmentorError):
    """Raised when image cannot be loaded."""

    pass

class UnsupportedOperationError(SegmentorError):
    """Raised when the loaded model does not support the requested prompt type."""

    pass
