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
DEFAULT_WHISPER_MODEL = "auto"  # Auto-select based on VRAM
DEFAULT_FALLBACK_LANGUAGE = "en"

# Speaker diarization
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Processing
DEFAULT_FACE_SAMPLE_FPS = 1.0
DEFAULT_OBJECT_SAMPLE_FPS = 2.0
DEFAULT_MIN_FACE_SIZE = 80

# Object detection (Qwen VLM)
DEFAULT_QWEN_MODEL = "auto"  # Auto-select based on VRAM
DEFAULT_QWEN_FRAMES_PER_SCENE = 1
DEFAULT_OBJECT_DETECTOR = "auto"  # Auto-select based on VRAM

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

    Model settings support "auto" to automatically select based on VRAM:
    - whisper_model: "auto" | "tiny" | "small" | "medium" | "large-v3"
    - qwen_model: "auto" | "Qwen/Qwen2-VL-2B-Instruct" | "Qwen/Qwen2-VL-7B-Instruct"
    - object_detector: "auto" | "yolo" | "qwen"
    """

    # API settings
    api_version: str = DEFAULT_API_VERSION
    log_level: str = DEFAULT_LOG_LEVEL

    # Whisper settings ("auto" = select based on VRAM)
    whisper_model: str = DEFAULT_WHISPER_MODEL
    fallback_language: str = DEFAULT_FALLBACK_LANGUAGE

    # Speaker diarization settings
    hf_token: str | None = None  # HuggingFace token for pyannote models
    diarization_model: str = DEFAULT_DIARIZATION_MODEL

    # Processing settings
    face_sample_fps: float = DEFAULT_FACE_SAMPLE_FPS
    object_sample_fps: float = DEFAULT_OBJECT_SAMPLE_FPS
    min_face_size: int = DEFAULT_MIN_FACE_SIZE

    # Object detection settings ("auto" = select based on VRAM)
    object_detector: str = DEFAULT_OBJECT_DETECTOR  # "auto", "yolo", or "qwen"
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_frames_per_scene: int = DEFAULT_QWEN_FRAMES_PER_SCENE

    # OCR settings
    ocr_languages: list[str] = DEFAULT_OCR_LANGUAGES.copy()

    # Temp directory for processing
    temp_dir: str = DEFAULT_TEMP_DIR

    def get_whisper_model(self) -> str:
        """Get resolved Whisper model (handles 'auto')."""
        if self.whisper_model == "auto":
            return get_auto_whisper_model()
        return self.whisper_model

    def get_qwen_model(self) -> str:
        """Get resolved Qwen model (handles 'auto')."""
        if self.qwen_model == "auto":
            return get_auto_qwen_model()
        return self.qwen_model

    def get_object_detector(self) -> ObjectDetector:
        """Get resolved object detector (handles 'auto')."""
        if self.object_detector == "auto":
            return get_auto_object_detector()
        return ObjectDetector(self.object_detector)


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


# =============================================================================
# VRAM Detection and Auto Model Selection
# =============================================================================


@lru_cache(maxsize=1)
def get_available_vram_gb() -> float:
    """Get available GPU memory in GB.

    Returns:
        Available VRAM in GB. For Apple Silicon, returns unified memory.
        Returns 0 if no GPU is available.
    """
    if has_cuda():
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            logger.info(f"CUDA GPU: {props.name}, {total_gb:.1f}GB VRAM")
            return total_gb
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory: {e}")
            return 0

    elif is_apple_silicon():
        try:
            import subprocess
            # Get total system memory (unified memory on Apple Silicon)
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / (1024 ** 3)
            # Apple Silicon can use ~75% of unified memory for GPU
            available_gb = total_gb * 0.75
            logger.info(f"Apple Silicon: {total_gb:.0f}GB unified, ~{available_gb:.0f}GB for GPU")
            return available_gb
        except Exception as e:
            logger.warning(f"Failed to get Apple Silicon memory: {e}")
            return 8.0  # Conservative default for M1/M2

    return 0


def get_auto_whisper_model() -> str:
    """Select Whisper model based on available VRAM.

    | VRAM     | Model    | Size   | Quality |
    |----------|----------|--------|---------|
    | <4GB     | tiny     | 75MB   | Basic   |
    | 4-6GB    | small    | 488MB  | Good    |
    | 6-10GB   | medium   | 1.5GB  | Better  |
    | 10GB+    | large-v3 | 3GB    | Best    |
    """
    vram = get_available_vram_gb()

    if vram >= 10:
        model = "large-v3"
    elif vram >= 6:
        model = "medium"
    elif vram >= 4:
        model = "small"
    else:
        model = "tiny"

    logger.info(f"Auto-selected Whisper model: {model} (VRAM: {vram:.1f}GB)")
    return model


def get_auto_qwen_model() -> str:
    """Select Qwen2-VL model based on available VRAM.

    | VRAM     | Model          | Size  | Quality |
    |----------|----------------|-------|---------|
    | <8GB     | (use YOLO)     | -     | Basic   |
    | 8-16GB   | Qwen2-VL-2B    | ~5GB  | Good    |
    | 16GB+    | Qwen2-VL-7B    | ~15GB | Best    |
    """
    vram = get_available_vram_gb()

    if vram >= 16:
        model = "Qwen/Qwen2-VL-7B-Instruct"
    elif vram >= 8:
        model = "Qwen/Qwen2-VL-2B-Instruct"
    else:
        # Not enough VRAM for Qwen, should use YOLO instead
        model = "Qwen/Qwen2-VL-2B-Instruct"
        logger.warning(f"Low VRAM ({vram:.1f}GB) - consider using YOLO instead of Qwen")

    logger.info(f"Auto-selected Qwen model: {model} (VRAM: {vram:.1f}GB)")
    return model


def get_auto_object_detector() -> ObjectDetector:
    """Select object detector based on available VRAM.

    YOLO is faster and uses less memory.
    Qwen provides better scene understanding but needs more VRAM.
    """
    vram = get_available_vram_gb()

    if vram >= 8:
        detector = ObjectDetector.QWEN
    else:
        detector = ObjectDetector.YOLO

    logger.info(f"Auto-selected object detector: {detector} (VRAM: {vram:.1f}GB)")
    return detector


def get_vram_summary() -> dict:
    """Get a summary of VRAM and auto-selected models.

    Useful for frontend to display hardware capabilities.
    """
    vram = get_available_vram_gb()
    device = get_device()

    return {
        "device": str(device),
        "vram_gb": round(vram, 1),
        "auto_whisper_model": get_auto_whisper_model(),
        "auto_qwen_model": get_auto_qwen_model() if vram >= 8 else None,
        "auto_object_detector": str(get_auto_object_detector()),
        "recommendations": {
            "can_use_large_whisper": vram >= 10,
            "can_use_qwen": vram >= 8,
            "can_use_qwen_7b": vram >= 16,
        },
    }
