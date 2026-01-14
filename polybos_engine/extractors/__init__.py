"""Video feature extractors."""

from .clip import extract_clip
from .faces import extract_faces
from .frames import FrameExtractor, extract_frames_batch, get_video_duration
from .metadata import (
    FFPROBE_WORKERS,
    extract_metadata,
    list_extractors,
    run_ffprobe_batch,
    shutdown_ffprobe_pool,
)
from .motion import MotionAnalysis, MotionType, analyze_motion, get_sample_timestamps
from .objects import extract_objects
from .objects_qwen import extract_objects_qwen, unload_qwen_model
from .ocr import extract_ocr
from .scenes import extract_scenes
from .telemetry import extract_telemetry
from .transcribe import extract_transcript, unload_whisper_model
from .vad import AudioContent, detect_voice_activity, unload_vad_model

__all__ = [
    "extract_metadata",
    "run_ffprobe_batch",
    "list_extractors",
    "FFPROBE_WORKERS",
    "shutdown_ffprobe_pool",
    "extract_transcript",
    "extract_faces",
    "extract_scenes",
    "extract_objects",
    "extract_objects_qwen",
    "extract_clip",
    "extract_ocr",
    "extract_telemetry",
    "analyze_motion",
    "get_sample_timestamps",
    "MotionAnalysis",
    "MotionType",
    "unload_qwen_model",
    "unload_whisper_model",
    "detect_voice_activity",
    "AudioContent",
    "unload_vad_model",
    # Frame extraction utilities
    "FrameExtractor",
    "extract_frames_batch",
    "get_video_duration",
]
