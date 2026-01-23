"""FastAPI application for Polybos Media Engine."""

# Prevent fork crashes on macOS with Hugging Face tokenizers library.
# The tokenizers library registers atfork handlers that panic when the process forks
# (e.g., to run ffmpeg via subprocess). This must be set BEFORE any imports.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# On macOS, use 'spawn' instead of 'fork' for multiprocessing to avoid crashes
# with libraries that aren't fork-safe (tokenizers, PyTorch, etc.)
import multiprocessing
import sys

if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

import asyncio
import atexit
import gc
import json
import logging
import logging.handlers
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# When running under a parent process that doesn't read our stdout/stderr,
# writes block when the pipe buffer fills up. Redirect to /dev/null to prevent this.
# This must happen BEFORE any logging or library initialization.
_is_interactive = sys.stdout.isatty() and sys.stderr.isatty()
if not _is_interactive:
    # Redirect stdout/stderr to /dev/null to prevent blocking writes
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull
    sys.stderr = _devnull

# Configure non-blocking logging using a queue
_log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)  # Unlimited size
_queue_handler = logging.handlers.QueueHandler(_log_queue)

# Always log to file (this is the only output when running non-interactively)
_file_handler = logging.FileHandler("/tmp/polybos_engine.log")
_log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_file_handler.setFormatter(_log_formatter)

# Build handler list - only include stderr if running interactively
_handlers: list[logging.Handler] = [_file_handler]
if _is_interactive:
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_log_formatter)
    _handlers.append(_stream_handler)

# QueueListener handles the actual I/O in a separate thread
_queue_listener = logging.handlers.QueueListener(
    _log_queue, *_handlers, respect_handler_level=True
)
_queue_listener.start()

# Register cleanup on exit
atexit.register(_queue_listener.stop)

# Configure root logger to use queue handler (non-blocking)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_queue_handler],
)

# ruff: noqa: E402 (imports after logging.basicConfig is intentional)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from polybos_engine import __version__
from polybos_engine.config import (
    get_free_memory_gb,
    get_settings,
    get_vram_summary,
    reload_settings,
    save_config_to_file,
)
from polybos_engine.extractors import (
    FFPROBE_WORKERS,
    SharedFrameBuffer,
    analyze_motion,
    decode_frames,
    detect_voice_activity,
    extract_clip,
    extract_faces,
    extract_metadata,
    extract_objects,
    extract_objects_qwen,
    extract_ocr,
    extract_scenes,
    extract_telemetry,
    extract_transcript,
    get_adaptive_timestamps,
    get_extractor_timestamps,
    get_sample_timestamps,
    run_ffprobe_batch,
    shutdown_ffprobe_pool,
    unload_clip_model,
    unload_face_model,
    unload_ocr_model,
    unload_qwen_model,
    unload_vad_model,
    unload_whisper_model,
    unload_yolo_model,
)
from polybos_engine.extractors.vad import AudioContent
from polybos_engine.schemas import (
    HealthResponse,
    MediaType,
    SettingsResponse,
    SettingsUpdate,
    get_media_type,
)

logger = logging.getLogger(__name__)


def _clear_memory() -> None:
    """Force garbage collection and clear GPU/MPS caches.

    Call before loading heavy AI models to free up memory.
    """
    # Multiple gc passes to handle circular references
    for _ in range(3):
        gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch, "mps"):
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except ImportError:
        pass

    # Also try mlx cleanup
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    # Final gc pass after GPU cleanup
    gc.collect()


def _get_memory_mb() -> int:
    """Get current process memory usage in MB."""
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process()
        return process.memory_info().rss // (1024 * 1024)
    except ImportError:
        return 0


# Create FastAPI app
app = FastAPI(
    title="Polybos Media Engine",
    description="AI-powered video extraction API",
    version=__version__,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Shutdown Handler
# ============================================================================


def _cleanup_resources():
    """Clean up all resources.

    Note: This runs during Python shutdown via atexit, so we must be careful
    not to import new modules or use logging (file handlers may be closed).
    """
    try:
        # Save timing history before shutdown
        _save_timing_history()
        shutdown_ffprobe_pool()
        unload_whisper_model()
        unload_qwen_model()
        unload_yolo_model()
        unload_clip_model()
        unload_ocr_model()
        unload_face_model()
        unload_vad_model()
    except Exception:
        pass  # Suppress errors during shutdown


atexit.register(_cleanup_resources)


@app.post("/shutdown")
async def shutdown_engine():
    """Gracefully shutdown the engine.

    Call this before killing the process to ensure clean resource cleanup.
    """
    logger.info("Shutdown requested via API")
    _cleanup_resources()

    # Schedule process exit after response is sent
    def delayed_exit():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=delayed_exit, daemon=True)
    thread.start()

    return {"status": "shutting_down"}


# ============================================================================
# Batch Job System (extractor-first processing for memory efficiency)
# ============================================================================


class JobProgress(BaseModel):
    """Progress within current extraction step."""

    message: str  # e.g., "Loading model...", "Processing frame 2/5"
    current: int | None = None
    total: int | None = None
    # ETA tracking
    stage_elapsed_seconds: float | None = None  # Time spent in current stage
    eta_seconds: float | None = None  # Estimated seconds remaining for stage


# Job TTL for automatic cleanup (1 hour)
JOB_TTL_SECONDS = 3600


# Historical timing data for ETA predictions
# Key: (extractor, resolution_bucket) -> list of processing times in seconds
_timing_history: dict[tuple[str, str], list[float]] = {}
_timing_history_lock = threading.Lock()
_timing_history_dirty = False  # Track if we need to save
_timing_history_last_save = 0.0  # Last save timestamp

# Keep last N samples per bucket for rolling average
_MAX_TIMING_SAMPLES = 20

# Timing history persistence
_TIMING_HISTORY_FILE = Path.home() / ".config" / "polybos" / "timing_history.json"
_TIMING_SAVE_INTERVAL = 30.0  # Save at most every 30 seconds


def _load_timing_history() -> None:
    """Load timing history from disk on startup."""
    global _timing_history
    if not _TIMING_HISTORY_FILE.exists():
        return
    try:
        with open(_TIMING_HISTORY_FILE) as f:
            data = json.load(f)
        # Convert string keys back to tuples
        with _timing_history_lock:
            for key_str, values in data.items():
                # Key format: "extractor|resolution"
                parts = key_str.split("|")
                if len(parts) == 2:
                    _timing_history[(parts[0], parts[1])] = values[-_MAX_TIMING_SAMPLES:]
        logger.info(f"Loaded timing history: {len(_timing_history)} buckets")
    except Exception as e:
        logger.warning(f"Failed to load timing history: {e}")


def _save_timing_history() -> None:
    """Save timing history to disk."""
    global _timing_history_dirty, _timing_history_last_save
    with _timing_history_lock:
        if not _timing_history:
            return
        # Convert tuple keys to strings for JSON
        data = {f"{k[0]}|{k[1]}": v for k, v in _timing_history.items()}
    try:
        _TIMING_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TIMING_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        _timing_history_dirty = False
        _timing_history_last_save = time.time()
        logger.debug(f"Saved timing history: {len(data)} buckets")
    except Exception as e:
        logger.warning(f"Failed to save timing history: {e}")


# Load timing history on module import
_load_timing_history()


def _get_resolution_bucket(width: int | None, height: int | None) -> str:
    """Get resolution bucket for timing predictions."""
    if width is None or height is None:
        return "unknown"
    pixels = width * height
    if pixels <= 921600:  # 1280x720
        return "720p"
    elif pixels <= 2073600:  # 1920x1080
        return "1080p"
    elif pixels <= 3686400:  # 2560x1440
        return "1440p"
    elif pixels <= 8294400:  # 3840x2160
        return "4k"
    elif pixels <= 14745600:  # 5120x2880
        return "5k"
    else:
        return "8k+"


def _record_timing(
    extractor: str,
    resolution_bucket: str,
    seconds: float,
    units: float | None = None,
) -> None:
    """Record processing rate for future ETA predictions.

    Args:
        extractor: Name of the extractor (transcript, visual, objects, etc.)
        resolution_bucket: Resolution category (720p, 1080p, 4k, etc.)
        seconds: Wall clock time to process
        units: Normalization units - depends on extractor:
            - transcript: duration in minutes (stores seconds per minute)
            - visual: number of timestamps (stores seconds per timestamp)
            - objects/faces/ocr/clip: number of frames (stores seconds per frame)
            - If None, stores raw seconds (for metadata, telemetry, etc.)
    """
    global _timing_history_dirty

    # Calculate rate (seconds per unit) or use raw seconds
    if units and units > 0:
        rate = seconds / units
    else:
        rate = seconds

    key = (extractor, resolution_bucket)
    with _timing_history_lock:
        if key not in _timing_history:
            _timing_history[key] = []
        _timing_history[key].append(rate)
        # Keep only recent samples
        if len(_timing_history[key]) > _MAX_TIMING_SAMPLES:
            _timing_history[key] = _timing_history[key][-_MAX_TIMING_SAMPLES:]
        sample_count = len(_timing_history[key])
        avg = sum(_timing_history[key]) / sample_count
        _timing_history_dirty = True

        unit_label = "/unit" if units else "s"
        logger.debug(
            f"Recorded timing: {extractor}@{resolution_bucket} = {rate:.2f}{unit_label} "
            f"(avg: {avg:.2f}{unit_label} from {sample_count} samples)"
        )
    # Save periodically (not on every update to reduce disk I/O)
    if _timing_history_dirty and time.time() - _timing_history_last_save > _TIMING_SAVE_INTERVAL:
        _save_timing_history()


def _get_predicted_rate(extractor: str, resolution_bucket: str) -> float | None:
    """Get predicted processing rate based on historical data.

    Returns the average rate (seconds per unit) for the given extractor and resolution.
    Multiply by the number of units to get predicted time.
    """
    key = (extractor, resolution_bucket)
    with _timing_history_lock:
        if key in _timing_history and _timing_history[key]:
            return sum(_timing_history[key]) / len(_timing_history[key])
    return None


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


# In-memory batch store
batch_jobs: dict[str, BatchJobStatus] = {}
batch_jobs_lock = threading.Lock()

# Batch queue - only one batch runs at a time, others wait in queue
batch_queue: list[tuple[str, BatchRequest]] = []  # (batch_id, request) tuples
batch_queue_lock = threading.Lock()
batch_running: bool = False  # True if a batch is currently running


def _cleanup_expired_batch_jobs() -> int:
    """Remove completed/failed batch jobs older than TTL.

    Returns:
        Number of batch jobs removed
    """
    now = datetime.now(timezone.utc)
    removed = 0

    with batch_jobs_lock:
        expired = [
            bid
            for bid, batch in batch_jobs.items()
            if batch.status in ("completed", "failed")
            and batch.completed_at is not None
            and (now - batch.completed_at).total_seconds() > JOB_TTL_SECONDS
        ]
        for bid in expired:
            del batch_jobs[bid]
            removed += 1

    if removed > 0:
        logger.info(f"Cleaned up {removed} expired batch jobs")

    return removed


def _update_queue_positions() -> None:
    """Update queue_position for all queued batches."""
    with batch_queue_lock:
        with batch_jobs_lock:
            for i, (bid, _) in enumerate(batch_queue):
                if bid in batch_jobs:
                    batch_jobs[bid].queue_position = i + 1  # 1-indexed


def _start_next_batch() -> None:
    """Start the next batch from the queue if one exists.

    Called when a batch completes or fails. Sets batch_running = False
    if no more batches in queue.
    """
    global batch_running

    with batch_queue_lock:
        if not batch_queue:
            batch_running = False
            logger.info("Batch queue empty, no more batches to run")
            return

        # Pop the next batch from queue
        next_batch_id, next_request = batch_queue.pop(0)
        logger.info(f"Starting next batch from queue: {next_batch_id}")

    # Update queue positions for remaining batches
    _update_queue_positions()

    # Update batch status from queued to pending
    with batch_jobs_lock:
        if next_batch_id in batch_jobs:
            batch_jobs[next_batch_id].status = "pending"
            batch_jobs[next_batch_id].queue_position = None

    # Start the batch in a new thread
    thread = threading.Thread(target=run_batch_job, args=(next_batch_id, next_request))
    thread.start()


def run_batch_job(batch_id: str, request: BatchRequest) -> None:
    """Run batch extraction - processes all files per extractor stage.

    This is more memory efficient as each model is loaded once,
    processes all files, then is unloaded before the next model.
    """
    settings = get_settings()

    # Resolve models from settings (handles "auto" -> actual model name)
    whisper_model = settings.get_whisper_model()
    qwen_model = settings.get_qwen_model()
    yolo_model = settings.get_yolo_model()
    clip_model = settings.get_clip_model()

    logger.info(
        f"Batch {batch_id} models: whisper={whisper_model}, qwen={qwen_model}, "
        f"yolo={yolo_model}, clip={clip_model}"
    )

    batch_start_time = time.time()
    peak_memory = _get_memory_mb()
    stage_start_times: dict[str, float] = {}  # extractor -> start time
    file_resolutions: dict[int, str] = (
        {}
    )  # file_idx -> resolution bucket (for timing predictions)

    def update_batch_progress(
        extractor: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        nonlocal peak_memory

        # Track stage start time
        if extractor not in stage_start_times:
            stage_start_times[extractor] = time.time()

        # Calculate ETA
        stage_elapsed: float | None = None
        eta: float | None = None
        if extractor in stage_start_times:
            stage_elapsed = round(time.time() - stage_start_times[extractor], 1)
            # Calculate ETA if we have progress info
            if current is not None and total is not None and current > 0:
                avg_time_per_item = stage_elapsed / current
                remaining_items = total - current
                eta = round(avg_time_per_item * remaining_items, 1)
            elif current == 0 and total is not None and total > 0:
                # No progress yet - try to use historical timing for prediction
                # Use the most common resolution in the batch, or "unknown" if none set
                common_res = "unknown"
                if file_resolutions:
                    res_counts: dict[str, int] = {}
                    for res in file_resolutions.values():
                        res_counts[res] = res_counts.get(res, 0) + 1
                    common_res = max(res_counts, key=lambda r: res_counts[r])
                predicted = _get_predicted_rate(extractor, common_res)
                if predicted is not None:
                    eta = round(predicted * total, 1)

        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].current_extractor = extractor
                batch_jobs[batch_id].progress = JobProgress(
                    message=message,
                    current=current,
                    total=total,
                    stage_elapsed_seconds=stage_elapsed,
                    eta_seconds=eta,
                )
                # Update memory and elapsed time
                current_mem = _get_memory_mb()
                peak_memory = max(peak_memory, current_mem)
                batch_jobs[batch_id].memory_mb = current_mem
                batch_jobs[batch_id].peak_memory_mb = peak_memory
                batch_jobs[batch_id].elapsed_seconds = round(
                    time.time() - batch_start_time, 1
                )

    def update_file_status(
        file_idx: int,
        status: str,
        result_key: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].status = status
                if result_key and result is not None:
                    batch_jobs[batch_id].files[file_idx].results[result_key] = result
                if error:
                    batch_jobs[batch_id].files[file_idx].error = error

    def update_extractor_status(
        file_idx: int, extractor: str, status: str
    ) -> None:
        """Update extractor status for a file.

        Args:
            file_idx: Index of the file in the batch
            extractor: Name of the extractor
            status: One of 'pending', 'active', 'completed', 'failed', 'skipped'
        """
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].extractor_status[extractor] = status

    def start_extractor_timing(extractor: str) -> datetime:
        """Start timing for an extractor stage."""
        started = datetime.now(timezone.utc)
        # Reset stage start time for ETA calculation
        stage_start_times[extractor] = time.time()
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].extractor_timings.append(
                    ExtractorTiming(extractor=extractor, started_at=started)
                )
        return started

    def end_extractor_timing(extractor: str, files_processed: int) -> None:
        """End timing for an extractor stage."""
        completed = datetime.now(timezone.utc)
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                for timing in batch_jobs[batch_id].extractor_timings:
                    if timing.extractor == extractor and timing.completed_at is None:
                        timing.completed_at = completed
                        timing.duration_seconds = round(
                            (completed - timing.started_at).total_seconds(), 2
                        )
                        timing.files_processed = files_processed
                        break

    def update_file_timing(
        file_idx: int, extractor: str, duration: float, units: float | None = None
    ) -> None:
        """Record per-file timing for an extractor.

        Args:
            file_idx: Index of the file in the batch
            extractor: Name of the extractor
            duration: Wall clock seconds to process
            units: Normalization units for rate calculation:
                - transcript: duration in minutes
                - visual: number of timestamps
                - objects/faces/ocr/clip: number of frames
                - None: store raw seconds (metadata, telemetry, etc.)
        """
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].timings[extractor] = round(
                    duration, 2
                )
        # Record to historical timing for future ETA predictions
        resolution = file_resolutions.get(file_idx, "unknown")
        _record_timing(extractor, resolution, duration, units)

    try:
        with batch_jobs_lock:
            batch_jobs[batch_id].status = "running"

        files = request.files
        total_files = len(files)

        # Track files that failed metadata extraction - skip them in all subsequent stages
        # If we can't read the file with ffprobe, there's no point trying other extractors
        failed_files: set[int] = set()

        # Stage 1: Metadata (parallel ffprobe for speed)
        if request.enable_metadata:
            start_extractor_timing("metadata")
            update_batch_progress(
                "metadata",
                f"Running ffprobe ({FFPROBE_WORKERS} parallel workers)...",
                0,
                total_files,
            )

            # Run all ffprobe calls in parallel
            probe_results = run_ffprobe_batch(files)

            # Extract metadata from each probe result
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress(
                    "metadata", f"Processing {Path(file_path).name}", i + 1, total_files
                )
                update_extractor_status(i, "metadata", "active")
                probe_data = probe_results.get(file_path)

                if isinstance(probe_data, Exception):
                    logger.warning(f"Metadata failed for {file_path}: {probe_data}")
                    logger.warning(
                        f"Skipping all extractors for {file_path} - file unreadable"
                    )
                    update_file_status(i, "failed", "metadata", None, str(probe_data))
                    update_extractor_status(i, "metadata", "failed")
                    update_file_timing(i, "metadata", time.time() - file_start)
                    failed_files.add(i)
                    continue

                try:
                    metadata = extract_metadata(file_path, probe_data)
                    update_file_status(i, "running", "metadata", metadata.model_dump())
                    update_extractor_status(i, "metadata", "completed")
                    # Store resolution bucket for timing predictions
                    file_resolutions[i] = _get_resolution_bucket(
                        metadata.resolution.width,
                        metadata.resolution.height,
                    )
                except Exception as e:
                    logger.warning(f"Metadata failed for {file_path}: {e}")
                    logger.warning(
                        f"Skipping all extractors for {file_path} - file unreadable"
                    )
                    update_file_status(i, "failed", "metadata", None, str(e))
                    update_extractor_status(i, "metadata", "failed")
                    failed_files.add(i)
                update_file_timing(i, "metadata", time.time() - file_start)
            end_extractor_timing("metadata", total_files)

        # Stage 2: Telemetry (always runs - lightweight, no models)
        start_extractor_timing("telemetry")
        update_batch_progress("telemetry", "Extracting telemetry...", 0, total_files)
        for i, file_path in enumerate(files):
            file_start = time.time()
            update_batch_progress(
                "telemetry", f"Processing {Path(file_path).name}", i + 1, total_files
            )
            update_extractor_status(i, "telemetry", "active")
            try:
                telemetry = extract_telemetry(file_path)
                update_file_status(
                    i,
                    "running",
                    "telemetry",
                    telemetry.model_dump() if telemetry else None,
                )
                update_extractor_status(i, "telemetry", "completed")
            except Exception as e:
                logger.warning(f"Telemetry failed for {file_path}: {e}")
                update_extractor_status(i, "telemetry", "failed")
            update_file_timing(i, "telemetry", time.time() - file_start)
        end_extractor_timing("telemetry", total_files)

        # Stage 3: Voice Activity Detection (WebRTC VAD - lightweight)
        # Skip for images and files without audio tracks
        if request.enable_vad:
            start_extractor_timing("vad")
            update_batch_progress("vad", "Analyzing audio...", 0, total_files)
            vad_ran = False  # Track if we actually ran VAD on any file
            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "vad", "skipped")
                    continue
                file_start = time.time()
                update_extractor_status(i, "vad", "active")

                # Check media type - skip VAD for images
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    logger.info(f"Skipping VAD for {file_path} - image file")
                    no_audio_result = {
                        "audio_content": str(AudioContent.NO_AUDIO),
                        "speech_ratio": 0.0,
                        "speech_segments": [],
                        "total_duration": 0.0,
                    }
                    update_file_status(i, "running", "vad", no_audio_result)
                    update_extractor_status(i, "vad", "completed")
                    update_file_timing(i, "vad", time.time() - file_start)
                    continue

                # Check if metadata shows no audio track
                has_audio_track = True
                with batch_jobs_lock:
                    file_results = batch_jobs[batch_id].files[i].results
                    if file_results and file_results.get("metadata"):
                        metadata = file_results["metadata"]
                        if metadata.get("audio") is None:
                            has_audio_track = False

                if not has_audio_track:
                    logger.info(f"Skipping VAD for {file_path} - no audio track")
                    no_audio_result = {
                        "audio_content": str(AudioContent.NO_AUDIO),
                        "speech_ratio": 0.0,
                        "speech_segments": [],
                        "total_duration": 0.0,
                    }
                    update_file_status(i, "running", "vad", no_audio_result)
                    update_extractor_status(i, "vad", "completed")
                    update_file_timing(i, "vad", time.time() - file_start)
                    continue

                # Run VAD for files with audio
                update_batch_progress(
                    "vad", f"Analyzing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    vad_result = detect_voice_activity(file_path)
                    update_file_status(i, "running", "vad", vad_result)
                    update_extractor_status(i, "vad", "completed")
                    vad_ran = True
                except Exception as e:
                    logger.warning(f"VAD failed for {file_path}: {e}")
                    update_extractor_status(i, "vad", "failed")
                update_file_timing(i, "vad", time.time() - file_start)

            # Only unload if we actually loaded the model
            if vad_ran:
                update_batch_progress("vad", "Unloading VAD model...", None, None)
                unload_vad_model()
            end_extractor_timing("vad", total_files)

        # Stage 4: Per-file visual processing
        # Process each file completely before moving to next (memory efficient)
        # Order: Motion → Scenes → Decode frames → Objects → Faces → OCR → CLIP → Release buffer
        #
        # This approach:
        # - Decodes frames once per file
        # - Runs all visual extractors on those frames
        # - Releases buffer before processing next file
        # - Keeps only one file's frames in memory at a time

        needs_visual_processing = any(
            [
                request.enable_motion,
                request.enable_scenes,
                request.enable_objects,
                request.enable_faces,
                request.enable_ocr,
                request.enable_clip,
            ]
        )

        # Track motion data for adaptive timestamps
        motion_data: dict[int, Any] = {}
        adaptive_timestamps: dict[int, list[float]] = {}

        # Track person timestamps for smart face detection
        person_timestamps: dict[int, list[float]] = {}

        # Skip motion analysis if timestamps are already provided
        has_precomputed_timestamps = bool(request.visual_timestamps)

        if needs_visual_processing:
            start_extractor_timing("visual_processing")
            update_batch_progress(
                "visual_processing",
                "Processing video frames...",
                0,
                total_files,
            )

            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue

                fname = Path(file_path).name
                media_type = get_media_type(file_path)
                file_start = time.time()

                update_batch_progress(
                    "visual_processing",
                    f"Processing {fname}",
                    i + 1,
                    total_files,
                )

                # --- Motion Analysis ---
                if request.enable_motion or (
                    (
                        request.enable_objects
                        or request.enable_faces
                        or request.enable_clip
                        or request.enable_ocr
                    )
                    and not has_precomputed_timestamps
                    and media_type != MediaType.IMAGE
                ):
                    motion_start = time.time()
                    update_extractor_status(i, "motion", "active")
                    try:
                        if media_type == MediaType.IMAGE:
                            motion_data[i] = None
                            adaptive_timestamps[i] = [0.0]
                            update_extractor_status(i, "motion", "completed")
                        else:
                            motion = analyze_motion(file_path)
                            motion_data[i] = motion
                            adaptive_timestamps[i] = get_adaptive_timestamps(motion)

                            # Always store motion data when computed (needed for Pass 2 timestamps)
                            motion_result = {
                                "duration": motion.duration,
                                "fps": motion.fps,
                                "primary_motion": motion.primary_motion.value,
                                "avg_intensity": float(motion.avg_intensity),
                                "is_stable": bool(motion.is_stable),
                                "segments": [
                                    {
                                        "start": seg.start,
                                        "end": seg.end,
                                        "motion_type": seg.motion_type.value,
                                        "intensity": float(seg.intensity),
                                    }
                                    for seg in motion.segments
                                ],
                            }
                            update_file_status(
                                i, "running", "motion", motion_result
                            )
                            update_extractor_status(i, "motion", "completed")
                            logger.info(
                                f"Motion for {fname}: stable={motion.is_stable}, "
                                f"timestamps={len(adaptive_timestamps[i])}"
                            )
                    except Exception as e:
                        logger.warning(f"Motion analysis failed for {file_path}: {e}")
                        update_extractor_status(i, "motion", "failed")
                        motion_data[i] = None
                        # Fallback: generate uniform timestamps from duration
                        # This ensures visual extractors still run even if motion fails
                        file_result = batch_jobs[batch_id].files[i]
                        meta = file_result.results.get("metadata")
                        if meta and meta.get("duration"):
                            duration = meta["duration"]
                            # Generate ~10 uniform timestamps
                            num_samples = min(10, max(3, int(duration / 10)))
                            step = duration / (num_samples + 1)
                            fallback_ts = [step * (j + 1) for j in range(num_samples)]
                            adaptive_timestamps[i] = fallback_ts
                            logger.info(
                                f"Using fallback timestamps for {fname}: {num_samples} uniform samples"
                            )
                        else:
                            adaptive_timestamps[i] = []
                    update_file_timing(i, "motion", time.time() - motion_start)

                # --- Scene Detection ---
                if request.enable_scenes and media_type != MediaType.IMAGE:
                    scenes_start = time.time()
                    update_extractor_status(i, "scenes", "active")
                    try:
                        scenes = extract_scenes(file_path)
                        update_file_status(
                            i,
                            "running",
                            "scenes",
                            scenes.model_dump() if scenes else None,
                        )
                        update_extractor_status(i, "scenes", "completed")
                    except Exception as e:
                        logger.warning(f"Scenes failed for {file_path}: {e}")
                        update_extractor_status(i, "scenes", "failed")
                    update_file_timing(i, "scenes", time.time() - scenes_start)

                # --- Decode Frames (for Objects, Faces, OCR, CLIP) ---
                buffer: SharedFrameBuffer | None = None
                visual_extractors_needed = any(
                    [
                        request.enable_objects,
                        request.enable_faces,
                        request.enable_ocr,
                        request.enable_clip,
                    ]
                )

                if visual_extractors_needed:
                    decode_start = time.time()
                    motion = motion_data.get(i)
                    timestamps = adaptive_timestamps.get(i, [])

                    # Use precomputed timestamps if provided for this file
                    if has_precomputed_timestamps and request.visual_timestamps:
                        file_timestamps = request.visual_timestamps.get(file_path)
                        if file_timestamps:
                            timestamps = file_timestamps

                    # Apply motion-based filtering for stable footage
                    if motion and motion.is_stable and timestamps:
                        timestamps = get_extractor_timestamps(
                            motion.is_stable, motion.avg_intensity, timestamps
                        )

                    # For images, use timestamp 0
                    if media_type == MediaType.IMAGE:
                        timestamps = [0.0]

                    if timestamps:
                        try:
                            buffer = decode_frames(
                                file_path,
                                timestamps=timestamps,
                                max_dimension=1920,
                            )
                            logger.info(
                                f"Decoded {len(buffer.frames)}/{len(timestamps)} frames for {fname}"
                            )
                        except Exception as e:
                            logger.warning(f"Frame decode failed for {file_path}: {e}")
                    update_file_timing(i, "frame_decode", time.time() - decode_start)

                # --- Objects (YOLO) ---
                if request.enable_objects and buffer is not None:
                    objects_start = time.time()
                    update_extractor_status(i, "objects", "active")
                    try:
                        objects = extract_objects(
                            file_path,
                            frame_buffer=buffer,
                            model_name=yolo_model,
                        )
                        if objects:
                            update_file_status(
                                i, "running", "objects", {"summary": objects.summary}
                            )
                            # Collect person timestamps for smart face sampling
                            person_ts = list(
                                set(
                                    d.timestamp
                                    for d in objects.detections
                                    if d.label == "person"
                                )
                            )
                            person_timestamps[i] = sorted(person_ts)
                            if person_ts:
                                logger.info(
                                    f"Found {len(person_ts)} person frames in {fname}"
                                )
                        else:
                            person_timestamps[i] = []
                        update_extractor_status(i, "objects", "completed")
                    except Exception as e:
                        logger.warning(f"Objects failed for {file_path}: {e}")
                        person_timestamps[i] = []
                        update_extractor_status(i, "objects", "failed")
                    # Use number of frames as units for rate calculation
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(
                        i, "objects", time.time() - objects_start, num_frames
                    )

                # --- Faces ---
                if request.enable_faces:
                    faces_start = time.time()
                    face_frame_count: int | None = None
                    update_extractor_status(i, "faces", "active")
                    try:
                        person_ts = person_timestamps.get(i, [])

                        if request.enable_objects and person_ts:
                            # YOLO found persons - decode those specific frames
                            person_buffer = decode_frames(
                                file_path, timestamps=person_ts
                            )
                            faces = extract_faces(file_path, frame_buffer=person_buffer)
                            face_frame_count = len(person_buffer.frames)
                            logger.info(
                                f"Face detection on {len(person_ts)} person frames for {fname}"
                            )
                        elif buffer is not None:
                            # No person timestamps from YOLO (or YOLO not enabled)
                            # Fall back to regular frame buffer
                            faces = extract_faces(file_path, frame_buffer=buffer)
                            face_frame_count = len(buffer.frames)
                            if request.enable_objects:
                                logger.info(
                                    f"Face detection on {len(buffer.frames)} frames for {fname} "
                                    "(YOLO found no persons, using all frames)"
                                )
                        else:
                            faces = None

                        if faces:
                            faces_data = {
                                "count": faces.count,
                                "unique_estimate": faces.unique_estimate,
                                "detections": [
                                    {
                                        "timestamp": d.timestamp,
                                        "bbox": d.bbox.model_dump(),
                                        "confidence": d.confidence,
                                        "embedding": d.embedding,
                                        "image_base64": d.image_base64,
                                        "needs_review": d.needs_review,
                                        "review_reason": d.review_reason,
                                    }
                                    for d in faces.detections
                                ],
                            }
                            update_file_status(i, "running", "faces", faces_data)
                        else:
                            update_file_status(
                                i,
                                "running",
                                "faces",
                                {"count": 0, "unique_estimate": 0, "detections": []},
                            )
                        update_extractor_status(i, "faces", "completed")
                    except Exception as e:
                        logger.warning(f"Faces failed for {file_path}: {e}")
                        update_extractor_status(i, "faces", "failed")
                    update_file_timing(
                        i, "faces", time.time() - faces_start, face_frame_count
                    )

                # --- OCR ---
                if request.enable_ocr and buffer is not None:
                    ocr_start = time.time()
                    update_extractor_status(i, "ocr", "active")
                    try:
                        ocr = extract_ocr(file_path, frame_buffer=buffer)
                        update_file_status(
                            i, "running", "ocr", ocr.model_dump() if ocr else None
                        )
                        update_extractor_status(i, "ocr", "completed")
                    except Exception as e:
                        logger.warning(f"OCR failed for {file_path}: {e}")
                        update_extractor_status(i, "ocr", "failed")
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "ocr", time.time() - ocr_start, num_frames)

                # --- CLIP ---
                if request.enable_clip and buffer is not None:
                    clip_start = time.time()
                    update_extractor_status(i, "clip", "active")
                    try:
                        clip = extract_clip(
                            file_path,
                            frame_buffer=buffer,
                            model_name=clip_model,
                        )
                        if clip:
                            update_file_status(i, "running", "clip", clip.model_dump())
                        else:
                            update_file_status(i, "running", "clip", None)
                        update_extractor_status(i, "clip", "completed")
                    except Exception as e:
                        logger.warning(f"CLIP failed for {file_path}: {e}")
                        update_extractor_status(i, "clip", "failed")
                    num_frames = len(buffer.frames) if buffer else None
                    update_file_timing(i, "clip", time.time() - clip_start, num_frames)

                # --- Release buffer for this file ---
                if buffer is not None:
                    logger.info(f"Releasing frame buffer for {fname}")
                    del buffer
                    gc.collect()

                # Update peak memory after each file
                peak_memory = max(peak_memory, _get_memory_mb())

            # Unload all visual models after processing all files
            update_batch_progress(
                "visual_processing", "Unloading models...", None, None
            )
            if request.enable_objects:
                unload_yolo_model()
            if request.enable_faces:
                unload_face_model()
            if request.enable_ocr:
                unload_ocr_model()
            if request.enable_clip:
                unload_clip_model()

            end_extractor_timing("visual_processing", total_files)

        # Stage 5: Visual (Qwen VLM - scene descriptions)
        # Separate stage because Qwen is very heavy and has its own frame handling
        if request.enable_visual:
            start_extractor_timing("visual")
            logger.info("Visual enabled (Qwen VLM)")
            _clear_memory()
            update_batch_progress("visual", "Loading Qwen model...", 0, total_files)
            logger.info(f"Qwen batch contexts: {request.contexts}")

            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "visual", "skipped")
                    continue
                file_start = time.time()
                fname = Path(file_path).name
                update_batch_progress(
                    "visual", f"Analyzing: {fname}", i + 1, total_files
                )
                update_extractor_status(i, "visual", "active")
                try:
                    motion = motion_data.get(i)
                    # Get per-file timestamps if provided
                    timestamps: list[float] | None = None
                    if request.visual_timestamps:
                        timestamps = request.visual_timestamps.get(file_path)
                    if timestamps is None and motion:
                        timestamps = get_sample_timestamps(motion, max_samples=5)

                    file_context = (
                        request.contexts.get(file_path) if request.contexts else None
                    )
                    logger.info(
                        f"Calling Qwen with context for {fname}: {file_context}"
                    )
                    visual_result = extract_objects_qwen(
                        file_path,
                        timestamps=timestamps,
                        model_name=qwen_model,
                        context=file_context,
                    )
                    visual_data: dict[str, Any] = {"summary": visual_result.summary}
                    if visual_result.descriptions:
                        visual_data["descriptions"] = visual_result.descriptions
                    update_file_status(i, "running", "visual", visual_data)
                    update_extractor_status(i, "visual", "completed")
                except Exception as e:
                    logger.warning(f"Visual failed for {file_path}: {e}", exc_info=True)
                    update_extractor_status(i, "visual", "failed")
                # Use number of timestamps as units for rate calculation
                num_timestamps = len(timestamps) if timestamps else None
                update_file_timing(i, "visual", time.time() - file_start, num_timestamps)

            update_batch_progress("visual", "Unloading Qwen model...", None, None)
            unload_qwen_model()
            end_extractor_timing("visual", total_files)

        # Stage 6: Transcript (Whisper - heavy model)
        # Skip for images and files without audio tracks
        if request.enable_transcript:
            start_extractor_timing("transcript")
            whisper_ran = False  # Track if we actually ran Whisper

            # Check if any files need transcription before loading model
            files_to_transcribe: list[int] = []
            for i, file_path in enumerate(files):
                if i in failed_files:
                    update_extractor_status(i, "transcript", "skipped")
                    continue
                # Skip images
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    update_extractor_status(i, "transcript", "skipped")
                    continue
                # Check for audio track
                has_audio = True
                with batch_jobs_lock:
                    file_results = batch_jobs[batch_id].files[i].results
                    if file_results and file_results.get("metadata"):
                        if file_results["metadata"].get("audio") is None:
                            has_audio = False
                if has_audio:
                    files_to_transcribe.append(i)
                else:
                    update_extractor_status(i, "transcript", "skipped")

            if files_to_transcribe:
                # Clear memory before loading heavy model
                logger.info("Clearing memory before Whisper...")
                _clear_memory()
                update_batch_progress(
                    "transcript",
                    "Loading Whisper model...",
                    0,
                    len(files_to_transcribe),
                )

                for idx, i in enumerate(files_to_transcribe):
                    file_path = files[i]
                    file_start = time.time()
                    update_batch_progress(
                        "transcript",
                        f"Transcribing {Path(file_path).name}",
                        idx + 1,
                        len(files_to_transcribe),
                    )
                    update_extractor_status(i, "transcript", "active")
                    try:
                        transcript = extract_transcript(
                            file_path,
                            model=whisper_model,
                            language=request.language,
                            fallback_language=settings.fallback_language,
                            language_hints=request.language_hints,
                            context_hint=request.context_hint,
                        )
                        update_file_status(
                            i,
                            "running",
                            "transcript",
                            transcript.model_dump() if transcript else None,
                        )
                        update_extractor_status(i, "transcript", "completed")
                        whisper_ran = True
                    except Exception as e:
                        logger.warning(f"Transcript failed for {file_path}: {e}")
                        update_extractor_status(i, "transcript", "failed")
                    # Get duration in minutes for rate calculation
                    duration_minutes: float | None = None
                    with batch_jobs_lock:
                        file_results = batch_jobs[batch_id].files[i].results
                        if file_results and file_results.get("metadata"):
                            duration_sec = file_results["metadata"].get("duration")
                            if duration_sec:
                                duration_minutes = duration_sec / 60.0
                    update_file_timing(
                        i, "transcript", time.time() - file_start, duration_minutes
                    )

                # Unload Whisper to free memory
                if whisper_ran:
                    update_batch_progress(
                        "transcript", "Unloading Whisper model...", None, None
                    )
                    unload_whisper_model()
            else:
                logger.info("Skipping Whisper - no files with audio tracks")

            end_extractor_timing("transcript", total_files)

        # Mark files as completed (skip failed files - they stay "failed")
        with batch_jobs_lock:
            for i in range(len(files)):
                if i in failed_files:
                    # File already marked as failed - don't overwrite
                    logger.info(f"Batch {batch_id} file {i} skipped (failed metadata)")
                    continue
                # Log results before marking complete
                result_keys = list(batch_jobs[batch_id].files[i].results.keys())
                logger.info(
                    f"Batch {batch_id} file {i} results before completion: keys={result_keys}"
                )
                batch_jobs[batch_id].files[i].status = "completed"
            batch_jobs[batch_id].status = "completed"
            batch_jobs[batch_id].current_extractor = None
            batch_jobs[batch_id].progress = None
            batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
            # Final metrics
            batch_jobs[batch_id].elapsed_seconds = round(
                time.time() - batch_start_time, 2
            )
            batch_jobs[batch_id].memory_mb = _get_memory_mb()
            batch_jobs[batch_id].peak_memory_mb = max(peak_memory, _get_memory_mb())

        # Log timing summary
        logger.info(
            f"Batch {batch_id} completed in {batch_jobs[batch_id].elapsed_seconds}s, peak memory: {batch_jobs[batch_id].peak_memory_mb}MB"
        )
        for timing in batch_jobs[batch_id].extractor_timings:
            logger.info(
                f"  {timing.extractor}: {timing.duration_seconds}s ({timing.files_processed} files)"
            )

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}")
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].status = "failed"
                batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
                batch_jobs[batch_id].elapsed_seconds = round(
                    time.time() - batch_start_time, 2
                )
                batch_jobs[batch_id].memory_mb = _get_memory_mb()
                batch_jobs[batch_id].peak_memory_mb = peak_memory

    finally:
        # Cleanup old batch jobs to free memory
        _cleanup_expired_batch_jobs()

        # Clear memory before starting next batch
        logger.info("Clearing memory after batch completion...")
        _clear_memory()

        # Always start the next batch from queue (or set batch_running = False)
        _start_next_batch()


def _create_batch_sync(
    batch_id: str, request: BatchRequest
) -> tuple[bool, int | None, str]:
    """Synchronous helper to create batch (runs in thread pool).

    Returns:
        (should_start, queue_position, status)
    """
    global batch_running

    # Cleanup expired batch jobs
    _cleanup_expired_batch_jobs()

    # Check if we should start immediately or queue
    with batch_queue_lock:
        should_start = not batch_running
        if should_start:
            batch_running = True
            queue_position = None
            status = "pending"
            logger.info(f"Starting batch {batch_id} immediately (no batch running)")
        else:
            # Add to queue
            batch_queue.append((batch_id, request))
            queue_position = len(batch_queue)
            status = "queued"
            logger.info(f"Queued batch {batch_id} at position {queue_position}")

    # Build initial extractor status for each file
    # Order matches processing order in run_batch_job
    extractor_flags = [
        ("metadata", request.enable_metadata),
        ("telemetry", True),  # Always runs
        ("vad", request.enable_vad),
        ("motion", request.enable_motion),
        ("scenes", request.enable_scenes),
        ("objects", request.enable_objects),
        ("faces", request.enable_faces),
        ("ocr", request.enable_ocr),
        ("clip", request.enable_clip),
        ("visual", request.enable_visual),
        ("transcript", request.enable_transcript),
    ]
    initial_extractor_status = {
        name: "pending" if enabled else "skipped" for name, enabled in extractor_flags
    }

    batch = BatchJobStatus(
        batch_id=batch_id,
        status=status,
        queue_position=queue_position,
        files=[
            BatchFileStatus(
                file=f,
                filename=Path(f).name,
                status="pending",
                extractor_status=initial_extractor_status.copy(),
            )
            for f in request.files
        ],
        created_at=datetime.now(timezone.utc),
    )

    with batch_jobs_lock:
        batch_jobs[batch_id] = batch

    return should_start, queue_position, status


@app.post("/batch")
async def create_batch(request: BatchRequest) -> dict[str, str]:
    """Create a new batch extraction job (memory-efficient extractor-first processing).

    Only one batch runs at a time. If a batch is already running, new batches
    are queued and will start automatically when the current batch finishes.
    """
    # Validate all files exist
    for file_path in request.files:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    batch_id = str(uuid.uuid4())[:8]

    # Run lock operations in thread pool to avoid blocking event loop
    should_start, _, _ = await asyncio.to_thread(_create_batch_sync, batch_id, request)

    # Start immediately if no batch running
    if should_start:
        thread = threading.Thread(target=run_batch_job, args=(batch_id, request))
        thread.start()

    return {"batch_id": batch_id}


def _get_batch_sync(batch_id: str, status_only: bool = False) -> BatchJobStatus | None:
    """Synchronous helper to get batch status (runs in thread pool).

    Args:
        batch_id: The batch ID to look up
        status_only: If True, return status/progress without large result data
    """
    with batch_jobs_lock:
        batch = batch_jobs.get(batch_id)
        if batch is None:
            return None

        if status_only:
            # Return a copy with results stripped out (keep status, progress, timings)
            return BatchJobStatus(
                batch_id=batch.batch_id,
                status=batch.status,
                queue_position=batch.queue_position,
                current_extractor=batch.current_extractor,
                progress=batch.progress,
                files=[
                    BatchFileStatus(
                        file=f.file,
                        filename=f.filename,
                        status=f.status,
                        results={},  # Empty - no large data
                        error=f.error,
                        timings=f.timings,
                        extractor_status=f.extractor_status,
                    )
                    for f in batch.files
                ],
                created_at=batch.created_at,
                completed_at=batch.completed_at,
                extractor_timings=batch.extractor_timings,
                elapsed_seconds=batch.elapsed_seconds,
                memory_mb=batch.memory_mb,
                peak_memory_mb=batch.peak_memory_mb,
            )

        return batch


@app.get("/batch/{batch_id}")
async def get_batch(batch_id: str, status_only: bool = False) -> BatchJobStatus:
    """Get batch job status and results.

    Args:
        batch_id: The batch ID to look up
        status_only: If True, return only status/progress without large result data.
            Use this for polling progress to avoid transferring large embeddings/transcripts.
    """
    # Run lock acquisition in thread pool to avoid blocking event loop
    result = await asyncio.to_thread(_get_batch_sync, batch_id, status_only)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    return result


def _delete_batch_sync(batch_id: str) -> tuple[bool, bool]:
    """Synchronous helper to delete batch (runs in thread pool).

    Returns:
        (found, was_queued) - whether batch was found and if it was queued
    """
    with batch_jobs_lock:
        if batch_id not in batch_jobs:
            return False, False
        was_queued = batch_jobs[batch_id].status == "queued"
        del batch_jobs[batch_id]

    # If it was queued, remove from queue and update positions
    if was_queued:
        with batch_queue_lock:
            batch_queue[:] = [(bid, req) for bid, req in batch_queue if bid != batch_id]
        _update_queue_positions()

    return True, was_queued


@app.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str) -> dict[str, str]:
    """Delete a batch job and free its memory.

    Jobs can be deleted at any time. If the batch is queued, it will be
    removed from the queue. If running, deletion will not stop processing
    - it will just remove the status tracking.
    """
    # Run lock acquisition in thread pool to avoid blocking event loop
    found, _ = await asyncio.to_thread(_delete_batch_sync, batch_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    logger.info(f"Deleted batch job {batch_id}")
    return {"status": "deleted", "batch_id": batch_id}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        api_version=settings.api_version,
    )


LOG_FILE = "/tmp/polybos_engine.log"


@app.get("/logs")
async def get_logs(
    lines: int = 100,
    level: str | None = None,
) -> dict[str, Any]:
    """Get recent log entries for debugging.

    Args:
        lines: Number of lines to return (default 100, max 1000)
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Dict with log lines and metadata
    """
    lines = min(lines, 1000)  # Cap at 1000 lines

    if not os.path.exists(LOG_FILE):
        return {"lines": [], "total": 0, "returned": 0, "file": LOG_FILE}

    try:
        # Use tail to efficiently read last N lines without loading entire file
        # Read more lines if filtering by level (we'll filter down after)
        read_lines = lines * 10 if level else lines

        result = subprocess.run(
            ["tail", "-n", str(read_lines), LOG_FILE],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"tail failed: {result.stderr}")

        all_lines = result.stdout.splitlines()

        # Filter by level if specified
        if level:
            level_upper = level.upper()
            all_lines = [line for line in all_lines if f" {level_upper} " in line]
            # Take only requested number after filtering
            all_lines = all_lines[-lines:]

        return {
            "lines": all_lines,
            "total": len(all_lines),  # Note: this is approximate when using tail
            "returned": len(all_lines),
            "file": LOG_FILE,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Timeout reading logs")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")


@app.get("/settings", response_model=SettingsResponse)
async def get_settings_endpoint():
    """Get current settings.

    Returns all settings with sensitive values (like hf_token) masked.
    """
    settings = get_settings()
    return SettingsResponse(
        api_version=settings.api_version,
        log_level=settings.log_level,
        whisper_model=settings.whisper_model,
        fallback_language=settings.fallback_language,
        hf_token_set=bool(settings.hf_token),
        diarization_model=settings.diarization_model,
        face_sample_fps=settings.face_sample_fps,
        object_sample_fps=settings.object_sample_fps,
        min_face_size=settings.min_face_size,
        object_detector=settings.object_detector,
        qwen_model=settings.qwen_model,
        qwen_frames_per_scene=settings.qwen_frames_per_scene,
        yolo_model=settings.yolo_model,
        clip_model=settings.clip_model,
        ocr_languages=settings.ocr_languages,
        temp_dir=settings.temp_dir,
    )


@app.put("/settings", response_model=SettingsResponse)
async def update_settings(update: SettingsUpdate):
    """Update settings.

    Only provided fields are updated. Changes are persisted to config file.
    Set hf_token to empty string to clear it.
    """
    settings = get_settings()

    # Update only provided fields
    update_data = update.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "hf_token":
            # Allow clearing token with empty string
            if value == "":
                value = None
            setattr(settings, field, value)
        else:
            setattr(settings, field, value)

    # Save to config file
    save_config_to_file(settings)

    # Reload to ensure consistency
    new_settings = reload_settings()

    logger.info(f"Settings updated: {list(update_data.keys())}")

    return SettingsResponse(
        api_version=new_settings.api_version,
        log_level=new_settings.log_level,
        whisper_model=new_settings.whisper_model,
        fallback_language=new_settings.fallback_language,
        hf_token_set=bool(new_settings.hf_token),
        diarization_model=new_settings.diarization_model,
        face_sample_fps=new_settings.face_sample_fps,
        object_sample_fps=new_settings.object_sample_fps,
        min_face_size=new_settings.min_face_size,
        object_detector=new_settings.object_detector,
        qwen_model=new_settings.qwen_model,
        qwen_frames_per_scene=new_settings.qwen_frames_per_scene,
        yolo_model=new_settings.yolo_model,
        clip_model=new_settings.clip_model,
        ocr_languages=new_settings.ocr_languages,
        temp_dir=new_settings.temp_dir,
    )


@app.get("/hardware")
async def hardware():
    """Get hardware capabilities and auto-selected models.

    Returns information about available GPU/VRAM and which models
    will be used with the current "auto" settings.
    """
    return get_vram_summary()


# Store for model check results
_model_check_results: dict[str, dict] = {}
_model_check_status: dict[str, str] = {}  # "running", "complete", "error"


def _run_model_checks(check_id: str) -> None:
    """Background task to check which models can load."""
    import time

    from polybos_engine.extractors.clip import get_clip_backend
    from polybos_engine.extractors.objects_qwen import _get_qwen_model

    results: dict[str, dict] = {}
    _model_check_status[check_id] = "running"

    try:
        # Test Qwen 2B
        logger.info("Testing Qwen 2B model...")
        start = time.time()
        try:
            _get_qwen_model("Qwen/Qwen2-VL-2B-Instruct")
            results["qwen_2b"] = {
                "canLoad": True,
                "error": None,
                "loadTimeSeconds": round(time.time() - start, 1),
            }
            unload_qwen_model()
        except Exception as e:
            results["qwen_2b"] = {
                "canLoad": False,
                "error": str(e),
                "loadTimeSeconds": round(time.time() - start, 1),
            }

        # Test Whisper large-v3
        logger.info("Testing Whisper large-v3 model...")
        start = time.time()
        try:
            from polybos_engine.config import has_cuda, is_apple_silicon

            if is_apple_silicon():
                # Create a tiny silent audio file to test model loading
                import tempfile
                import wave

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    # Write a minimal valid WAV file (0.1 second of silence)
                    with wave.open(f.name, "w") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(16000)
                        wav.writeframes(b"\x00" * 3200)  # 0.1s of silence

                try:
                    import mlx_whisper  # type: ignore[import-not-found]

                    # This will load the model and transcribe the silent audio
                    mlx_whisper.transcribe(
                        temp_path,
                        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                    )
                finally:
                    import os

                    os.unlink(temp_path)
            elif has_cuda():
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]

                WhisperModel("large-v3", device="cuda")
            else:
                import whisper  # type: ignore[import-not-found]

                whisper.load_model("large-v3")
            results["whisper_large"] = {
                "canLoad": True,
                "error": None,
                "loadTimeSeconds": round(time.time() - start, 1),
            }
            unload_whisper_model()
        except Exception as e:
            results["whisper_large"] = {
                "canLoad": False,
                "error": str(e),
                "loadTimeSeconds": round(time.time() - start, 1),
            }

        # Test CLIP
        logger.info("Testing CLIP model...")
        start = time.time()
        try:
            get_clip_backend()
            results["clip"] = {
                "canLoad": True,
                "error": None,
                "loadTimeSeconds": round(time.time() - start, 1),
            }
            unload_clip_model()
        except Exception as e:
            results["clip"] = {
                "canLoad": False,
                "error": str(e),
                "loadTimeSeconds": round(time.time() - start, 1),
            }

        # Test YOLO
        logger.info("Testing YOLO model...")
        start = time.time()
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            YOLO("yolov8m.pt")
            results["yolo"] = {
                "canLoad": True,
                "error": None,
                "loadTimeSeconds": round(time.time() - start, 1),
            }
            unload_yolo_model()
        except Exception as e:
            results["yolo"] = {
                "canLoad": False,
                "error": str(e),
                "loadTimeSeconds": round(time.time() - start, 1),
            }

        # Test Face detection (DeepFace)
        logger.info("Testing Face detection model...")
        start = time.time()
        try:
            from deepface import DeepFace  # type: ignore[import-not-found]

            DeepFace.build_model("Facenet")
            results["faces"] = {
                "canLoad": True,
                "error": None,
                "loadTimeSeconds": round(time.time() - start, 1),
            }
            unload_face_model()
        except Exception as e:
            results["faces"] = {
                "canLoad": False,
                "error": str(e),
                "loadTimeSeconds": round(time.time() - start, 1),
            }

        _model_check_results[check_id] = {
            "results": results,
            "freeMemoryGb": get_free_memory_gb(),
        }
        _model_check_status[check_id] = "complete"
        logger.info(f"Model check {check_id} complete: {results}")

    except Exception as e:
        logger.error(f"Model check {check_id} failed: {e}")
        _model_check_status[check_id] = "error"
        _model_check_results[check_id] = {"error": str(e)}


@app.post("/check-models")
async def start_model_check():
    """Start checking which models can actually load.

    Returns immediately with a check_id. Poll GET /check-models/{check_id} for results.
    Takes 30-60 seconds to complete.
    """
    import threading
    import uuid

    check_id = str(uuid.uuid4())[:8]

    # Start background thread
    thread = threading.Thread(target=_run_model_checks, args=(check_id,), daemon=True)
    thread.start()

    return {"check_id": check_id, "status": "running"}


@app.get("/check-models/{check_id}")
async def get_model_check_result(check_id: str):
    """Get the result of a model check.

    Returns status: "running", "complete", or "error".
    When complete, includes models dict with load results.
    """
    status = _model_check_status.get(check_id, "not_found")

    if status == "not_found":
        raise HTTPException(status_code=404, detail=f"Check ID {check_id} not found")

    if status == "running":
        return {"check_id": check_id, "status": "running"}

    # Complete or error - return results
    result = _model_check_results.get(check_id, {})
    return {"check_id": check_id, "status": status, **result}


@app.get("/extractors")
async def list_extractors():
    """List available extractors and their descriptions."""
    return {
        "extractors": [
            {
                "name": "metadata",
                "description": "Video metadata (duration, resolution, codec, device, GPS)",
                "enable_flag": "enable_metadata",
            },
            {
                "name": "transcript",
                "description": "Audio transcription using Whisper",
                "enable_flag": "enable_transcript",
            },
            {
                "name": "scenes",
                "description": "Scene boundary detection",
                "enable_flag": "enable_scenes",
            },
            {
                "name": "faces",
                "description": "Face detection with embeddings",
                "enable_flag": "enable_faces",
            },
            {
                "name": "objects",
                "description": "Object detection with YOLO (fast, bounding boxes)",
                "enable_flag": "enable_objects",
            },
            {
                "name": "visual",
                "description": "Scene descriptions with Qwen2-VL (slower, richer)",
                "enable_flag": "enable_visual",
            },
            {
                "name": "clip",
                "description": "CLIP visual embeddings per scene",
                "enable_flag": "enable_clip",
            },
            {
                "name": "ocr",
                "description": "Text extraction from video frames",
                "enable_flag": "enable_ocr",
            },
            {
                "name": "telemetry",
                "description": "GPS/flight path (always extracted automatically)",
            },
        ]
    }


@app.post("/encode_text")
async def encode_text(request: dict):
    """Encode a text query to a CLIP embedding for text-to-image search.

    Request body:
        text: str - The text query to encode
        model_name: str (optional) - CLIP model name (e.g., "ViT-B-32")

    Returns:
        embedding: list[float] - The normalized CLIP embedding (512 or 768 dimensions)
        model: str - The model used for encoding
    """
    from polybos_engine.extractors.clip import encode_text_query, get_clip_backend

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text query is required")

    model_name = request.get("model_name")

    try:
        embedding = encode_text_query(text, model_name)
        backend = get_clip_backend(model_name)
        return {
            "embedding": embedding,
            "model": backend.get_model_name(),
        }
    except Exception as e:
        logger.error(f"Text encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
