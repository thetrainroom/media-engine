"""Configuration settings for Polybos Media Engine."""

import platform
from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings


class DeviceType(StrEnum):
    """Compute device types."""

    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API settings
    api_version: str = "1.0"
    log_level: str = "INFO"

    # Whisper settings
    whisper_model: str = "large-v3"
    fallback_language: str = "en"

    # Speaker diarization settings
    hf_token: str | None = None  # HuggingFace token for pyannote models
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Processing settings
    face_sample_fps: float = 1.0
    object_sample_fps: float = 2.0
    min_face_size: int = 80

    # Temp directory for processing
    temp_dir: str = "/tmp/polybos"

    model_config = {"env_prefix": "POLYBOS_"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> DeviceType:
    """Get the best available compute device."""
    if is_apple_silicon():
        return DeviceType.MPS
    elif has_cuda():
        return DeviceType.CUDA
    else:
        return DeviceType.CPU
