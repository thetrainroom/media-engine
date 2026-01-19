"""Video feature extractors."""

from .clip import extract_clip, extract_clip_image, unload_clip_model
from .faces import extract_faces, unload_face_model
from .frame_buffer import (
    SharedFrame,
    SharedFrameBuffer,
    decode_frames,
    get_extractor_timestamps,
)
from .frames import FrameExtractor, extract_frames_batch, get_video_duration
from .metadata import (
    FFPROBE_WORKERS,
    extract_metadata,
    list_extractors,
    run_ffprobe_batch,
    shutdown_ffprobe_pool,
)
from .motion import (
    MotionAnalysis,
    MotionType,
    analyze_motion,
    get_adaptive_timestamps,
    get_sample_timestamps,
)
from .objects import extract_objects, unload_yolo_model
from .objects_qwen import extract_objects_qwen, unload_qwen_model
from .ocr import extract_ocr, unload_ocr_model
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
    "extract_clip_image",
    "extract_ocr",
    "extract_telemetry",
    "analyze_motion",
    "get_sample_timestamps",
    "get_adaptive_timestamps",
    "MotionAnalysis",
    "MotionType",
    # Model unload functions
    "unload_qwen_model",
    "unload_whisper_model",
    "unload_yolo_model",
    "unload_clip_model",
    "unload_ocr_model",
    "unload_face_model",
    "unload_vad_model",
    # Voice activity detection
    "detect_voice_activity",
    "AudioContent",
    # Frame extraction utilities
    "FrameExtractor",
    "extract_frames_batch",
    "get_video_duration",
    # Shared frame buffer
    "SharedFrame",
    "SharedFrameBuffer",
    "decode_frames",
    "get_extractor_timestamps",
]
