"""Configuration settings for Polybos Media Engine."""

import json
import logging
import platform
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# =============================================================================
# Default Constants
# =============================================================================

# Config file location
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "polybos" / "config.json"

# API
DEFAULT_API_VERSION = "1.0"
DEFAULT_LOG_LEVEL = "INFO"

# Whisper speech-to-text
DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_FALLBACK_LANGUAGE = "en"

# Speaker diarization
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Processing
DEFAULT_FACE_SAMPLE_FPS = 1.0
DEFAULT_OBJECT_SAMPLE_FPS = 2.0
DEFAULT_MIN_FACE_SIZE = 80

# Object detection (Qwen VLM)
DEFAULT_QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_QWEN_FRAMES_PER_SCENE = 1

# OCR - Latin script languages (see https://www.jaided.ai/easyocr/)
# For CJK: ch_sim, ch_tra, ja, ko
# Note: Finnish (fi) not supported by EasyOCR
DEFAULT_OCR_LANGUAGES: list[str] = [
    "en",  # English
    "no",  # Norwegian
    "de",  # German
    "fr",  # French
    "es",  # Spanish
    "it",  # Italian
    "pt",  # Portuguese
    "nl",  # Dutch
    "sv",  # Swedish
    "da",  # Danish
    "pl",  # Polish
]

# Temp directory
DEFAULT_TEMP_DIR = "/tmp/polybos"


# =============================================================================
# Enums
# =============================================================================


class DeviceType(StrEnum):
    """Compute device types."""

    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class ObjectDetector(StrEnum):
    """Object detection backend."""

    YOLO = "yolo"
    QWEN = "qwen"


# =============================================================================
# Settings (loaded from JSON config file)
# =============================================================================


class Settings(BaseModel):
    """Application settings loaded from JSON config file.

    Config file location: ~/.config/polybos/config.json
    """

    # API settings
    api_version: str = DEFAULT_API_VERSION
    log_level: str = DEFAULT_LOG_LEVEL

    # Whisper settings
    whisper_model: str = DEFAULT_WHISPER_MODEL
    fallback_language: str = DEFAULT_FALLBACK_LANGUAGE

    # Speaker diarization settings
    hf_token: str | None = None  # HuggingFace token for pyannote models
    diarization_model: str = DEFAULT_DIARIZATION_MODEL

    # Processing settings
    face_sample_fps: float = DEFAULT_FACE_SAMPLE_FPS
    object_sample_fps: float = DEFAULT_OBJECT_SAMPLE_FPS
    min_face_size: int = DEFAULT_MIN_FACE_SIZE

    # Object detection settings
    object_detector: ObjectDetector = ObjectDetector.YOLO
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_frames_per_scene: int = DEFAULT_QWEN_FRAMES_PER_SCENE

    # OCR settings
    ocr_languages: list[str] = DEFAULT_OCR_LANGUAGES.copy()

    # Temp directory for processing
    temp_dir: str = DEFAULT_TEMP_DIR


def get_config_path() -> Path:
    """Get the config file path."""
    return DEFAULT_CONFIG_PATH


def load_config_from_file(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Optional path to config file. Defaults to ~/.config/polybos/config.json

    Returns:
        Dictionary of settings (empty if file doesn't exist)
    """
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        return {}

    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return {}


def save_config_to_file(settings: Settings, config_path: Path | None = None) -> None:
    """Save configuration to JSON file.

    Args:
        settings: Settings instance to save
        config_path: Optional path to config file. Defaults to ~/.config/polybos/config.json
    """
    path = config_path or DEFAULT_CONFIG_PATH

    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)

    logger.info(f"Saved config to {path}")


# Cached settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings instance (loaded from config file on first call)."""
    global _settings

    if _settings is None:
        config_data = load_config_from_file()
        _settings = Settings(**config_data)
        if config_data:
            logger.info(f"Loaded settings from {DEFAULT_CONFIG_PATH}")
        else:
            logger.info("Using default settings")

    return _settings


def reload_settings() -> Settings:
    """Reload settings from config file."""
    global _settings
    _settings = None
    return get_settings()


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
