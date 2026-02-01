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
    - yolo_model: "auto" | "yolov8n.pt" | "yolov8s.pt" | "yolov8m.pt" | "yolov8l.pt" | "yolov8x.pt"
    - clip_model: "auto" | "ViT-B-16" | "ViT-B-32" | "ViT-L-14"
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

    # YOLO model ("auto" = select based on VRAM)
    yolo_model: str = "auto"

    # CLIP model ("auto" = select based on VRAM)
    clip_model: str = "auto"

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

    def get_yolo_model(self) -> str:
        """Get resolved YOLO model (handles 'auto')."""
        if self.yolo_model == "auto":
            return get_auto_yolo_model()
        return self.yolo_model

    def get_clip_model(self) -> str:
        """Get resolved CLIP model (handles 'auto')."""
        if self.clip_model == "auto":
            return get_auto_clip_model()
        return self.clip_model

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


def get_gpu_name() -> str | None:
    """Get the GPU name/model.

    Returns:
        GPU name string (e.g., "NVIDIA GeForce RTX 4090", "Apple M2 Max")
        or None if no GPU is available.
    """
    if has_cuda():
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:
            return None
    elif is_apple_silicon():
        try:
            import subprocess

            # Get chip name from system_profiler
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            cpu_brand = result.stdout.strip()
            # Extract Apple chip name if present
            if "Apple" in cpu_brand:
                return cpu_brand
            # Fallback: try to get from SPHardwareDataType
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "Chip:" in line:
                    return line.split(":")[-1].strip()
            return "Apple Silicon"
        except Exception:
            return "Apple Silicon"
    return None


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
            total_gb = props.total_memory / (1024**3)
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
            total_gb = total_bytes / (1024**3)
            logger.info(f"Apple Silicon: {total_gb:.0f}GB unified memory")
            return total_gb
        except Exception as e:
            logger.warning(f"Failed to get Apple Silicon memory: {e}")
            return 8.0  # Conservative default for M1/M2

    return 0


def get_free_memory_gb() -> float:
    """Get memory available for ML models without causing excessive swapping.

    Uses psutil for cross-platform memory detection, which correctly accounts
    for free, inactive, and purgeable memory on macOS.

    Returns:
        Estimated GB available for loading models.
    """
    try:
        import psutil

        mem = psutil.virtual_memory()
        # psutil.available is the memory that can be given to processes
        # without swapping - this is what we want
        available_gb = mem.available / (1024**3)

        # Leave a 1GB buffer for system processes
        available_for_models = max(0.0, available_gb - 1.0)

        logger.info(f"Memory: {mem.total / (1024**3):.0f}GB total, " f"{mem.available / (1024**3):.1f}GB available, " f"{available_for_models:.1f}GB for models")
        return available_for_models

    except ImportError:
        logger.warning("psutil not installed, using fallback memory detection")

    # Fallback without psutil
    try:
        import sys

        if sys.platform == "darwin":
            import subprocess

            # macOS fallback: use total memory * 0.5 as conservative estimate
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_bytes = int(result.stdout.strip())
            total_gb = total_bytes / (1024**3)
            return total_gb * 0.5  # Conservative: assume 50% available
        else:
            # Linux: try /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)

    except Exception as e:
        logger.warning(f"Failed to get free memory: {e}")

    return 8.0  # Conservative fallback


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


def get_auto_yolo_model() -> str:
    """Select YOLO model based on available VRAM.

    | VRAM     | Model     | Size   | Speed   |
    |----------|-----------|--------|---------|
    | <2GB     | yolov8n   | 6MB    | Fastest |
    | 2-4GB    | yolov8s   | 22MB   | Fast    |
    | 4-8GB    | yolov8m   | 52MB   | Medium  |
    | 8-16GB   | yolov8l   | 87MB   | Slow    |
    | 16GB+    | yolov8x   | 136MB  | Slowest |
    """
    vram = get_available_vram_gb()

    if vram >= 16:
        model = "yolov8x.pt"
    elif vram >= 8:
        model = "yolov8l.pt"
    elif vram >= 4:
        model = "yolov8m.pt"
    elif vram >= 2:
        model = "yolov8s.pt"
    else:
        model = "yolov8n.pt"

    logger.info(f"Auto-selected YOLO model: {model} (VRAM: {vram:.1f}GB)")
    return model


def get_auto_clip_model() -> str:
    """Select CLIP model based on available VRAM.

    | VRAM     | Model     | Size    | Quality |
    |----------|-----------|---------|---------|
    | <2GB     | ViT-B-16  | 335MB   | Good    |
    | 2-4GB    | ViT-B-32  | 338MB   | Good    |
    | 4GB+     | ViT-L-14  | 933MB   | Best    |

    Note: ViT-B-16 and ViT-B-32 are similar size but different patch sizes.
    ViT-B-32 is slightly faster, ViT-B-16 has better detail recognition.
    """
    vram = get_available_vram_gb()

    if vram >= 4:
        model = "ViT-L-14"
    elif vram >= 2:
        model = "ViT-B-32"
    else:
        model = "ViT-B-16"

    logger.info(f"Auto-selected CLIP model: {model} (VRAM: {vram:.1f}GB)")
    return model


def get_vram_summary() -> dict:
    """Get a summary of VRAM and auto-selected models.

    Useful for frontend to display hardware capabilities.
    """
    vram = get_available_vram_gb()  # Total VRAM (cached)
    free_mem = get_free_memory_gb()  # Currently available (not cached)
    device = get_device()

    # Model memory requirements
    qwen_2b_needs = 5.0
    qwen_7b_needs = 15.0
    whisper_large_needs = 6.0

    return {
        "device": str(device),
        "gpu_name": get_gpu_name(),
        "vram_gb": round(vram, 1),
        "free_memory_gb": round(free_mem, 1),
        "auto_whisper_model": get_auto_whisper_model(),
        "auto_qwen_model": get_auto_qwen_model() if vram >= 8 else None,
        "auto_yolo_model": get_auto_yolo_model(),
        "auto_clip_model": get_auto_clip_model(),
        "auto_object_detector": str(get_auto_object_detector()),
        # What the hardware CAN support (based on total VRAM)
        "recommendations": {
            "can_use_large_whisper": vram >= 10,
            "can_use_qwen": vram >= 8,
            "can_use_qwen_7b": vram >= 16,
            "can_use_clip_l14": vram >= 4,
            "can_use_yolo_xlarge": vram >= 16,
        },
        # What can load RIGHT NOW (based on free memory)
        "available_now": {
            "qwen_2b": free_mem >= qwen_2b_needs,
            "qwen_7b": free_mem >= qwen_7b_needs,
            "whisper_large": free_mem >= whisper_large_needs,
            "whisper_medium": free_mem >= 4.0,
            "whisper_small": free_mem >= 2.0,
            "yolo": True,  # YOLO is always available (~100MB)
            "clip": free_mem >= 1.0,
        },
    }


# =============================================================================
# Memory Monitoring
# =============================================================================

# Approximate memory requirements for models (in GB)
MODEL_MEMORY_REQUIREMENTS: dict[str, float] = {
    # Whisper models
    "tiny": 1.0,
    "small": 2.0,
    "medium": 4.0,
    "large-v3": 6.0,
    # YOLO models
    "yolov8n.pt": 0.2,
    "yolov8s.pt": 0.3,
    "yolov8m.pt": 0.5,
    "yolov8l.pt": 0.8,
    "yolov8x.pt": 1.2,
    # Qwen VLM
    "Qwen/Qwen2-VL-2B-Instruct": 6.0,
    "Qwen/Qwen2-VL-7B-Instruct": 16.0,
    # CLIP
    "ViT-B-16": 0.4,
    "ViT-B-32": 0.4,
    "ViT-L-14": 1.0,
    # OCR
    "easyocr": 3.0,
    # Face detection
    "deepface": 0.5,
}


def get_available_memory_gb() -> tuple[float, float]:
    """Get available system RAM and GPU memory in GB.

    Returns:
        (available_ram_gb, available_vram_gb)
    """
    try:
        import psutil  # type: ignore[import-not-found]

        mem = psutil.virtual_memory()
        available_ram = mem.available / (1024**3)
    except ImportError:
        logger.warning("psutil not installed, cannot monitor RAM")
        available_ram = 8.0  # Conservative default

    # GPU memory
    available_vram = 0.0
    if has_cuda():
        try:
            import torch

            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            available_vram = (total - allocated) / (1024**3)
        except Exception:
            available_vram = get_available_vram_gb()
    elif is_apple_silicon():
        # Unified memory - estimate from system available
        available_vram = available_ram * 0.75

    return available_ram, available_vram


def check_memory_before_load(model_name: str, clear_memory_func: Any | None = None) -> bool:
    """Check if enough memory is available before loading a model.

    If memory is low, attempts to free memory by calling the clear function.

    Args:
        model_name: Name of the model to load (must be in MODEL_MEMORY_REQUIREMENTS)
        clear_memory_func: Optional function to call to free memory (e.g., gc.collect)

    Returns:
        True if enough memory is available, False otherwise
    """
    required_gb = MODEL_MEMORY_REQUIREMENTS.get(model_name, 2.0)  # Default 2GB
    ram, vram = get_available_memory_gb()

    # Use VRAM for GPU models, RAM for CPU
    device = get_device()
    available = vram if device != DeviceType.CPU else ram

    if available < required_gb:
        logger.warning(f"Low memory ({available:.1f}GB available) for {model_name} " f"({required_gb:.1f}GB required)")

        # Try to free memory
        if clear_memory_func is not None:
            logger.info("Attempting to free memory...")
            clear_memory_func()

            # Re-check
            ram, vram = get_available_memory_gb()
            available = vram if device != DeviceType.CPU else ram

            if available < required_gb:
                logger.error(f"Still insufficient memory ({available:.1f}GB) for {model_name}")
                return False
            else:
                logger.info(f"Memory freed, now {available:.1f}GB available")

    return True
