"""FastAPI application for Polybos Media Engine."""

import atexit
import gc
import logging
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler("/tmp/polybos_engine.log"), logging.StreamHandler()],
)

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
from polybos_engine.schemas import (
    HealthResponse,
    MediaType,
    SettingsResponse,
    SettingsUpdate,
    get_media_type,
)
from polybos_engine.extractors.vad import AudioContent

logger = logging.getLogger(__name__)


def _clear_memory() -> None:
    """Force garbage collection and clear GPU/MPS caches.

    Call before loading heavy AI models to free up memory.
    """
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
    not to import new modules or use logging that may fail.
    """
    try:
        logger.info("Cleaning up resources...")
    except Exception:
        pass  # Logging may fail during shutdown

    try:
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

# Keep last N samples per bucket for rolling average
_MAX_TIMING_SAMPLES = 20


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


def _record_timing(extractor: str, resolution_bucket: str, seconds: float) -> None:
    """Record processing time for future ETA predictions."""
    key = (extractor, resolution_bucket)
    with _timing_history_lock:
        if key not in _timing_history:
            _timing_history[key] = []
        _timing_history[key].append(seconds)
        # Keep only recent samples
        if len(_timing_history[key]) > _MAX_TIMING_SAMPLES:
            _timing_history[key] = _timing_history[key][-_MAX_TIMING_SAMPLES:]
        sample_count = len(_timing_history[key])
        avg = sum(_timing_history[key]) / sample_count
        logger.debug(
            f"Recorded timing: {extractor}@{resolution_bucket} = {seconds:.2f}s "
            f"(avg: {avg:.2f}s from {sample_count} samples)"
        )


def _get_predicted_time(extractor: str, resolution_bucket: str) -> float | None:
    """Get predicted processing time based on historical data."""
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
    language: str | None = None  # Force specific language (ISO 639-1 code, e.g., "en", "no")
    language_hints: list[str] | None = None  # Hints (currently unused by Whisper)
    context_hint: str | None = None
    # Context for Qwen VLM - per-file context mapping (file path -> context dict)
    # Example: {"/path/video1.mp4": {"location": "Oslo"}, "/path/video2.mp4": {"location": "Bergen"}}
    contexts: dict[str, dict[str, str]] | None = None
    qwen_timestamps: list[float] | None = None


class BatchFileStatus(BaseModel):
    """Status for a single file in a batch."""

    file: str
    filename: str
    status: str  # pending, running, completed, failed
    results: dict[str, Any] = {}
    error: str | None = None
    timings: dict[str, float] = {}  # extractor -> seconds


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
    status: str  # pending, running, completed, failed
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
    file_resolutions: dict[int, str] = {}  # file_idx -> resolution bucket (for timing predictions)

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
                predicted = _get_predicted_time(extractor, common_res)
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

    def update_file_timing(file_idx: int, extractor: str, duration: float) -> None:
        """Record per-file timing for an extractor."""
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].timings[extractor] = round(
                    duration, 2
                )
        # Record to historical timing for future ETA predictions
        resolution = file_resolutions.get(file_idx, "unknown")
        _record_timing(extractor, resolution, duration)

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
                probe_data = probe_results.get(file_path)

                if isinstance(probe_data, Exception):
                    logger.warning(f"Metadata failed for {file_path}: {probe_data}")
                    logger.warning(f"Skipping all extractors for {file_path} - file unreadable")
                    update_file_status(i, "failed", "metadata", None, str(probe_data))
                    update_file_timing(i, "metadata", time.time() - file_start)
                    failed_files.add(i)
                    continue

                try:
                    metadata = extract_metadata(file_path, probe_data)
                    update_file_status(i, "running", "metadata", metadata.model_dump())
                    # Store resolution bucket for timing predictions
                    file_resolutions[i] = _get_resolution_bucket(
                        metadata.resolution.width,
                        metadata.resolution.height,
                    )
                except Exception as e:
                    logger.warning(f"Metadata failed for {file_path}: {e}")
                    logger.warning(f"Skipping all extractors for {file_path} - file unreadable")
                    update_file_status(i, "failed", "metadata", None, str(e))
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
            try:
                telemetry = extract_telemetry(file_path)
                update_file_status(
                    i,
                    "running",
                    "telemetry",
                    telemetry.model_dump() if telemetry else None,
                )
            except Exception as e:
                logger.warning(f"Telemetry failed for {file_path}: {e}")
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
                    continue
                file_start = time.time()

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
                    update_file_timing(i, "vad", time.time() - file_start)
                    continue

                # Run VAD for files with audio
                update_batch_progress(
                    "vad", f"Analyzing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    vad_result = detect_voice_activity(file_path)
                    update_file_status(i, "running", "vad", vad_result)
                    vad_ran = True
                except Exception as e:
                    logger.warning(f"VAD failed for {file_path}: {e}")
                update_file_timing(i, "vad", time.time() - file_start)

            # Only unload if we actually loaded the model
            if vad_ran:
                update_batch_progress("vad", "Unloading VAD model...", None, None)
                unload_vad_model()
            end_extractor_timing("vad", total_files)

        # Stage 4: Scenes (PySceneDetect - moderate memory)
        # Skip for images (scene detection is video-only)
        if request.enable_scenes:
            start_extractor_timing("scenes")
            update_batch_progress("scenes", "Detecting scenes...", 0, total_files)
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                # Skip images - scene detection only makes sense for videos
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    continue
                file_start = time.time()
                update_batch_progress(
                    "scenes", f"Processing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    scenes = extract_scenes(file_path)
                    update_file_status(
                        i, "running", "scenes", scenes.model_dump() if scenes else None
                    )
                except Exception as e:
                    logger.warning(f"Scenes failed for {file_path}: {e}")
                update_file_timing(i, "scenes", time.time() - file_start)
            end_extractor_timing("scenes", total_files)

        # Stage 5: Transcript (Whisper - heavy model)
        # Skip for images and files without audio tracks
        if request.enable_transcript:
            start_extractor_timing("transcript")
            whisper_ran = False  # Track if we actually ran Whisper

            # Check if any files need transcription before loading model
            files_to_transcribe: list[int] = []
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                # Skip images
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
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

            if files_to_transcribe:
                # Clear memory before loading heavy model
                logger.info("Clearing memory before Whisper...")
                _clear_memory()
                update_batch_progress(
                    "transcript", "Loading Whisper model...", 0, len(files_to_transcribe)
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
                        whisper_ran = True
                    except Exception as e:
                        logger.warning(f"Transcript failed for {file_path}: {e}")
                    update_file_timing(i, "transcript", time.time() - file_start)

                # Unload Whisper to free memory
                if whisper_ran:
                    update_batch_progress(
                        "transcript", "Unloading Whisper model...", None, None
                    )
                    unload_whisper_model()
            else:
                logger.info("Skipping Whisper - no files with audio tracks")

            end_extractor_timing("transcript", total_files)

        # Stage 6: Motion Analysis (for smart sampling of faces/objects/clip/ocr)
        # Store motion data per file for later use
        motion_data: dict[int, Any] = {}  # file_idx -> MotionAnalysis
        adaptive_timestamps: dict[int, list[float]] = {}  # file_idx -> timestamps

        # Skip motion analysis if timestamps are already provided (e.g., Pass 2 reusing Pass 1 data)
        has_precomputed_timestamps = bool(request.qwen_timestamps)
        needs_motion = request.enable_motion or (
            (
                request.enable_objects
                or request.enable_visual
                or request.enable_faces
                or request.enable_clip
                or request.enable_ocr
            )
            and not has_precomputed_timestamps
        )

        if has_precomputed_timestamps and request.qwen_timestamps:
            logger.info(
                f"Using {len(request.qwen_timestamps)} pre-computed timestamps, skipping motion analysis"
            )

        if needs_motion:
            start_extractor_timing("motion")
            update_batch_progress(
                "motion", "Analyzing camera motion...", 0, total_files
            )
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                # Skip images - motion analysis is video-only
                # Use timestamp 0 for subsequent stages (objects, faces, etc.)
                media_type = get_media_type(file_path)
                if media_type == MediaType.IMAGE:
                    motion_data[i] = None
                    adaptive_timestamps[i] = [0.0]  # Single timestamp for image
                    continue
                file_start = time.time()
                update_batch_progress(
                    "motion", f"Analyzing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    motion = analyze_motion(file_path)
                    motion_data[i] = motion
                    adaptive_timestamps[i] = get_adaptive_timestamps(motion)

                    # Store motion results for client use (editing tools)
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
                    update_file_status(i, "running", "motion", motion_result)
                    logger.info(
                        f"Motion for {Path(file_path).name}: "
                        f"avg_intensity={motion.avg_intensity:.1f}, "
                        f"stable={motion.is_stable}, "
                        f"adaptive_timestamps={len(adaptive_timestamps[i])}"
                    )
                except Exception as e:
                    logger.warning(f"Motion analysis failed for {file_path}: {e}")
                    # Fall back to None (will use default sampling)
                    motion_data[i] = None
                    adaptive_timestamps[i] = []
                update_file_timing(i, "motion", time.time() - file_start)
            end_extractor_timing("motion", total_files)

        # Stage 6b: Decode shared frames for visual extractors
        # Decode video frames once and share across Objects, Faces, OCR, CLIP
        frame_buffers: dict[int, SharedFrameBuffer] = {}

        visual_extractors_enabled = any(
            [
                request.enable_objects,
                request.enable_faces,
                request.enable_ocr,
                request.enable_clip,
            ]
        )

        if visual_extractors_enabled and needs_motion:
            start_extractor_timing("frame_decode")
            update_batch_progress(
                "frame_decode", "Decoding video frames...", 0, total_files
            )

            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue

                # Get timestamps for this file (use motion-adaptive if available)
                motion = motion_data.get(i)
                timestamps = adaptive_timestamps.get(i, [])

                # Apply motion-based filtering for more stable footage
                if motion and motion.is_stable:
                    timestamps = get_extractor_timestamps(
                        motion.is_stable, motion.avg_intensity, timestamps
                    )

                if not timestamps:
                    continue

                file_start = time.time()
                update_batch_progress(
                    "frame_decode",
                    f"Decoding {Path(file_path).name} ({len(timestamps)} frames)",
                    i + 1,
                    total_files,
                )

                try:
                    buffer = decode_frames(
                        file_path,
                        timestamps=timestamps,
                        max_dimension=1920,
                    )
                    frame_buffers[i] = buffer
                    logger.info(
                        f"Decoded {len(buffer.frames)}/{len(timestamps)} frames "
                        f"for {Path(file_path).name}"
                    )
                except Exception as e:
                    logger.warning(f"Frame decode failed for {file_path}: {e}")

                update_file_timing(i, "frame_decode", time.time() - file_start)

            end_extractor_timing("frame_decode", len(frame_buffers))

        # Stage 7: Objects (YOLO - fast object detection with bounding boxes)
        # Store person timestamps for face detection
        person_timestamps: dict[int, list[float]] = (
            {}
        )  # file_idx -> timestamps where persons detected

        if request.enable_objects:
            start_extractor_timing("objects")
            logger.info("Objects enabled (YOLO)")
            update_batch_progress(
                "objects", "Detecting objects with YOLO...", 0, total_files
            )
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                file_start = time.time()
                update_batch_progress(
                    "objects",
                    f"Processing {Path(file_path).name}",
                    i + 1,
                    total_files,
                )
                try:
                    # Use shared frame buffer (required)
                    buffer = frame_buffers.get(i)
                    if buffer is None:
                        logger.warning(f"No frame buffer for {file_path}, skipping objects")
                        person_timestamps[i] = []
                        continue

                    objects = extract_objects(
                        file_path,
                        frame_buffer=buffer,
                        model_name=yolo_model,
                    )

                    if objects:
                        update_file_status(
                            i, "running", "objects", {"summary": objects.summary}
                        )

                        # Collect timestamps where persons were detected (for smart face sampling)
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
                                f"Found {len(person_ts)} person frames in {Path(file_path).name}"
                            )
                    else:
                        person_timestamps[i] = []
                except Exception as e:
                    logger.warning(f"Objects failed for {file_path}: {e}")
                    person_timestamps[i] = []
                update_file_timing(i, "objects", time.time() - file_start)
            # Unload YOLO to free memory
            update_batch_progress("objects", "Unloading YOLO model...", None, None)
            unload_yolo_model()
            end_extractor_timing("objects", total_files)

        # Stage 7b: Visual (Qwen VLM - scene descriptions)
        if request.enable_visual:
            start_extractor_timing("visual")
            logger.info("Visual enabled (Qwen VLM)")
            # Clear memory before loading heavy model
            logger.info("Clearing memory before Qwen...")
            _clear_memory()
            update_batch_progress(
                "visual", "Loading Qwen model...", 0, total_files
            )
            # Log contexts for debugging
            logger.info(f"Qwen batch contexts: {request.contexts}")

            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                file_start = time.time()
                fname = Path(file_path).name
                update_batch_progress(
                    "visual", f"Analyzing: {fname}", i + 1, total_files
                )
                try:
                    # Use motion-based timestamps for Qwen (fewer samples for VLM)
                    motion = motion_data.get(i)
                    timestamps = (
                        request.qwen_timestamps
                    )  # Use provided timestamps first
                    if timestamps is None and motion:
                        timestamps = get_sample_timestamps(motion, max_samples=5)

                    # Get per-file context
                    file_context = (
                        request.contexts.get(file_path)
                        if request.contexts
                        else None
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
                    logger.info(
                        f"Qwen result for file {i}: summary={visual_result.summary}, descriptions={visual_result.descriptions}"
                    )
                    update_file_status(i, "running", "visual", visual_data)
                    logger.info(f"Stored visual_data for file {i}: {visual_data}")
                except Exception as e:
                    logger.warning(
                        f"Visual failed for {file_path}: {e}", exc_info=True
                    )
                update_file_timing(i, "visual", time.time() - file_start)
            # Unload Qwen to free memory
            update_batch_progress("visual", "Unloading Qwen model...", None, None)
            unload_qwen_model()
            end_extractor_timing("visual", total_files)

        # Stage 8: Faces (YOLO-triggered - only where persons detected)
        if request.enable_faces:
            start_extractor_timing("faces")
            update_batch_progress("faces", "Detecting faces...", 0, total_files)
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                file_start = time.time()
                update_batch_progress(
                    "faces", f"Processing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    # Smart sampling: use person timestamps from YOLO if available
                    person_ts = person_timestamps.get(i, [])
                    buffer = frame_buffers.get(i)

                    if request.enable_objects and person_ts:
                        # YOLO-triggered: decode frames specifically for person timestamps
                        person_buffer = decode_frames(file_path, timestamps=person_ts)
                        faces = extract_faces(file_path, frame_buffer=person_buffer)
                        logger.info(
                            f"Face detection on {len(person_ts)} person frames for {Path(file_path).name}"
                        )
                    elif request.enable_objects and not person_ts:
                        # YOLO ran but found no persons - skip face detection
                        logger.info(
                            f"Skipping face detection for {Path(file_path).name} - no persons detected by YOLO"
                        )
                        faces = None
                    elif buffer is None:
                        logger.warning(f"No frame buffer for {file_path}, skipping faces")
                        faces = None
                    else:
                        # Objects disabled - use shared frame buffer
                        faces = extract_faces(file_path, frame_buffer=buffer)

                    if faces:
                        faces_data = {
                            "count": faces.count,
                            "unique_estimate": faces.unique_estimate,
                            "detections": [
                                {
                                    "timestamp": d.timestamp,
                                    "bbox": d.bbox.model_dump(),
                                    "confidence": d.confidence,
                                    "embedding": d.embedding,  # Include for face matching
                                    "image_base64": d.image_base64,
                                    "needs_review": d.needs_review,
                                    "review_reason": d.review_reason,
                                }
                                for d in faces.detections
                            ],
                        }
                        update_file_status(i, "running", "faces", faces_data)
                    else:
                        # No faces - store empty result
                        update_file_status(
                            i,
                            "running",
                            "faces",
                            {"count": 0, "unique_estimate": 0, "detections": []},
                        )
                except Exception as e:
                    logger.warning(f"Faces failed for {file_path}: {e}")
                update_file_timing(i, "faces", time.time() - file_start)
            # Unload face detection models to free memory
            update_batch_progress("faces", "Unloading face models...", None, None)
            unload_face_model()
            end_extractor_timing("faces", total_files)

        # Stage 9: OCR (EasyOCR - moderate model) with motion-adaptive timestamps
        if request.enable_ocr:
            start_extractor_timing("ocr")
            update_batch_progress("ocr", "Extracting text...", 0, total_files)
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                file_start = time.time()
                update_batch_progress(
                    "ocr", f"Processing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    buffer = frame_buffers.get(i)
                    if buffer is None:
                        logger.warning(f"No frame buffer for {file_path}, skipping OCR")
                        ocr = None
                    else:
                        ocr = extract_ocr(file_path, frame_buffer=buffer)
                    update_file_status(
                        i, "running", "ocr", ocr.model_dump() if ocr else None
                    )
                except Exception as e:
                    logger.warning(f"OCR failed for {file_path}: {e}")
                update_file_timing(i, "ocr", time.time() - file_start)
            # Unload OCR model to free memory
            update_batch_progress("ocr", "Unloading OCR model...", None, None)
            unload_ocr_model()
            end_extractor_timing("ocr", total_files)

        # Stage 10: CLIP embeddings with motion-adaptive timestamps
        if request.enable_clip:
            start_extractor_timing("clip")
            update_batch_progress(
                "clip", "Extracting CLIP embeddings...", 0, total_files
            )
            for i, file_path in enumerate(files):
                if i in failed_files:
                    continue
                file_start = time.time()
                update_batch_progress(
                    "clip", f"Processing {Path(file_path).name}", i + 1, total_files
                )
                try:
                    buffer = frame_buffers.get(i)
                    if buffer is None:
                        logger.warning(f"No frame buffer for {file_path}, skipping CLIP")
                        clip = None
                    else:
                        clip = extract_clip(
                            file_path,
                            frame_buffer=buffer,
                            model_name=clip_model,
                        )
                    if clip:
                        update_file_status(
                            i,
                            "running",
                            "clip",
                            {"model": clip.model, "count": len(clip.segments)},
                        )
                    else:
                        update_file_status(i, "running", "clip", None)
                except Exception as e:
                    logger.warning(f"CLIP failed for {file_path}: {e}")
                update_file_timing(i, "clip", time.time() - file_start)
            # Unload CLIP model to free memory
            update_batch_progress("clip", "Unloading CLIP model...", None, None)
            unload_clip_model()
            end_extractor_timing("clip", total_files)

        # Release shared frame buffers to free memory
        if frame_buffers:
            logger.info(f"Releasing {len(frame_buffers)} shared frame buffers")
            for buf in frame_buffers.values():
                del buf
            frame_buffers.clear()
            gc.collect()

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


@app.post("/batch")
async def create_batch(request: BatchRequest) -> dict[str, str]:
    """Create a new batch extraction job (memory-efficient extractor-first processing)."""
    # Cleanup expired batch jobs on new batch creation (lazy cleanup)
    _cleanup_expired_batch_jobs()

    # Validate all files exist
    for file_path in request.files:
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    batch_id = str(uuid.uuid4())[:8]
    batch = BatchJobStatus(
        batch_id=batch_id,
        status="pending",
        files=[
            BatchFileStatus(file=f, filename=Path(f).name, status="pending")
            for f in request.files
        ],
        created_at=datetime.now(timezone.utc),
    )

    with batch_jobs_lock:
        batch_jobs[batch_id] = batch

    # Start background thread
    thread = threading.Thread(target=run_batch_job, args=(batch_id, request))
    thread.start()

    return {"batch_id": batch_id}


@app.get("/batch/{batch_id}")
async def get_batch(batch_id: str) -> BatchJobStatus:
    """Get batch job status and results."""
    with batch_jobs_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
        return batch_jobs[batch_id]


@app.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str) -> dict[str, str]:
    """Delete a batch job and free its memory.

    Jobs can be deleted at any time, but deleting a running batch
    will not stop its processing - it will just remove the status.
    """
    with batch_jobs_lock:
        if batch_id not in batch_jobs:
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
        del batch_jobs[batch_id]

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
            results["qwen_2b"] = {"canLoad": True, "error": None, "loadTimeSeconds": round(time.time() - start, 1)}
            unload_qwen_model()
        except Exception as e:
            results["qwen_2b"] = {"canLoad": False, "error": str(e), "loadTimeSeconds": round(time.time() - start, 1)}

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
            results["whisper_large"] = {"canLoad": True, "error": None, "loadTimeSeconds": round(time.time() - start, 1)}
            unload_whisper_model()
        except Exception as e:
            results["whisper_large"] = {"canLoad": False, "error": str(e), "loadTimeSeconds": round(time.time() - start, 1)}

        # Test CLIP
        logger.info("Testing CLIP model...")
        start = time.time()
        try:
            get_clip_backend()
            results["clip"] = {"canLoad": True, "error": None, "loadTimeSeconds": round(time.time() - start, 1)}
            unload_clip_model()
        except Exception as e:
            results["clip"] = {"canLoad": False, "error": str(e), "loadTimeSeconds": round(time.time() - start, 1)}

        # Test YOLO
        logger.info("Testing YOLO model...")
        start = time.time()
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            YOLO("yolov8m.pt")
            results["yolo"] = {"canLoad": True, "error": None, "loadTimeSeconds": round(time.time() - start, 1)}
            unload_yolo_model()
        except Exception as e:
            results["yolo"] = {"canLoad": False, "error": str(e), "loadTimeSeconds": round(time.time() - start, 1)}

        # Test Face detection (DeepFace)
        logger.info("Testing Face detection model...")
        start = time.time()
        try:
            from deepface import DeepFace  # type: ignore[import-not-found]

            DeepFace.build_model("Facenet")
            results["faces"] = {"canLoad": True, "error": None, "loadTimeSeconds": round(time.time() - start, 1)}
            unload_face_model()
        except Exception as e:
            results["faces"] = {"canLoad": False, "error": str(e), "loadTimeSeconds": round(time.time() - start, 1)}

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
