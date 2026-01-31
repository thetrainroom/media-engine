"""Pydantic models for batch processing."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class JobProgress(BaseModel):
    """Progress within current extraction step."""

    message: str  # e.g., "Loading model...", "Processing frame 2/5"
    current: int | None = None
    total: int | None = None
    # ETA tracking
    stage_elapsed_seconds: float | None = None  # Time spent in current stage
    eta_seconds: float | None = None  # Estimated seconds remaining for current stage
    # Total ETA fields (for full batch/queue visibility)
    total_eta_seconds: float | None = None  # Total time remaining for entire batch
    queue_eta_seconds: float | None = None  # Total time remaining for all queued batches
    queued_batches: int | None = None  # Number of batches waiting in queue


# Job TTL for automatic cleanup (1 hour)
JOB_TTL_SECONDS = 3600


class BatchRequest(BaseModel):
    """Request for batch extraction.

    Model selection is configured via global settings (GET/PUT /settings).
    This keeps batch requests simple and hardware config in one place.
    """

    files: list[str]
    enable_metadata: bool = True
    enable_vad: bool = False  # Voice Activity Detection
    enable_scenes: bool = False
    enable_transcript: bool = False
    enable_faces: bool = False
    enable_objects: bool = False  # YOLO object detection (fast, bounding boxes)
    enable_visual: bool = False  # Qwen VLM scene descriptions (slower, richer)
    enable_clip: bool = False
    enable_ocr: bool = False
    enable_motion: bool = False

    # Context for Whisper
    language: str | None = (
        None  # Force specific language (ISO 639-1 code, e.g., "en", "no")
    )
    language_hints: list[str] | None = None  # Hints (currently unused by Whisper)
    context_hint: str | None = None
    # Context for Qwen VLM - per-file context mapping (file path -> context dict)
    # Example: {"/path/video1.mp4": {"location": "Oslo"}, "/path/video2.mp4": {"location": "Bergen"}}
    contexts: dict[str, dict[str, str]] | None = None
    # Per-file timestamps for visual/VLM analysis (file path -> list of timestamps)
    # Example: {"/path/video1.mp4": [10.0, 30.0], "/path/video2.mp4": [5.0, 15.0, 25.0]}
    visual_timestamps: dict[str, list[float]] | None = None
    # Optional LUT path for visual analysis (e.g., for log footage color correction)
    # Applied to extracted frames before sending to Qwen
    lut_path: str | None = None


class BatchFileStatus(BaseModel):
    """Status for a single file in a batch."""

    file: str
    filename: str
    status: str  # pending, running, completed, failed
    results: dict[str, Any] = {}
    error: str | None = None
    timings: dict[str, float] = {}  # extractor -> seconds
    extractor_status: dict[str, str] = {}  # extractor -> pending/active/completed/failed/skipped


class ExtractorTiming(BaseModel):
    """Timing for a single extractor stage."""

    extractor: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    files_processed: int = 0


class BatchJobStatus(BaseModel):
    """Status of a batch extraction job."""

    batch_id: str
    status: str  # queued, pending, running, completed, failed
    queue_position: int | None = (
        None  # Position in queue (1 = next to run), None if not queued
    )
    current_extractor: str | None = None
    progress: JobProgress | None = None
    files: list[BatchFileStatus] = []
    created_at: datetime
    completed_at: datetime | None = None
    # Timing and resource metrics
    extractor_timings: list[ExtractorTiming] = []
    elapsed_seconds: float | None = None
    memory_mb: int | None = None  # Current process memory
    peak_memory_mb: int | None = None  # Peak process memory during batch
