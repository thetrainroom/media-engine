"""FastAPI application for Polybos Media Engine."""

import atexit
import gc
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/polybos_engine.log'),
        logging.StreamHandler()
    ]
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from polybos_engine import __version__
from polybos_engine.config import ObjectDetector, get_settings
from polybos_engine.extractors import (
    FFPROBE_WORKERS,
    analyze_motion,
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
    get_sample_timestamps,
    run_ffprobe_batch,
    unload_clip_model,
    unload_face_model,
    unload_ocr_model,
    unload_qwen_model,
    unload_vad_model,
    unload_whisper_model,
    unload_yolo_model,
)
from polybos_engine.extractors.shot_type import detect_shot_type
from polybos_engine.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MotionResult,
    MotionSegment,
)

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
        import psutil
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
    """Clean up all resources."""
    logger.info("Cleaning up resources...")
    try:
        from polybos_engine.extractors import (
            shutdown_ffprobe_pool,
            unload_clip_model,
            unload_face_model,
            unload_ocr_model,
            unload_qwen_model,
            unload_vad_model,
            unload_whisper_model,
            unload_yolo_model,
        )
        shutdown_ffprobe_pool()
        unload_whisper_model()
        unload_qwen_model()
        unload_yolo_model()
        unload_clip_model()
        unload_ocr_model()
        unload_face_model()
        unload_vad_model()
        logger.info("All resources cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up resources: {e}")

atexit.register(_cleanup_resources)


@app.post("/shutdown")
async def shutdown_engine():
    """Gracefully shutdown the engine.

    Call this before killing the process to ensure clean resource cleanup.
    """
    import os
    import signal

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


# Job TTL for automatic cleanup (1 hour)
JOB_TTL_SECONDS = 3600

class BatchRequest(BaseModel):
    """Request for batch extraction.

    Model fields support:
    - "auto" - auto-select based on VRAM
    - Specific model name (e.g., "yolov8m.pt", "ViT-L-14")
    """
    files: list[str]
    enable_metadata: bool = True
    enable_vad: bool = False  # Voice Activity Detection
    enable_scenes: bool = False
    enable_transcript: bool = False
    enable_faces: bool = False
    enable_objects: bool = False
    enable_clip: bool = False
    enable_ocr: bool = False
    enable_motion: bool = False

    # Object detection settings
    object_detector: ObjectDetector | None = None

    # Model selection (per-request overrides, "auto" = VRAM-based selection)
    whisper_model: str = "auto"
    qwen_model: str = "auto"
    yolo_model: str = "auto"
    clip_model: str = "auto"

    # Context for Whisper
    language_hints: list[str] | None = None
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
            bid for bid, batch in batch_jobs.items()
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
    from polybos_engine.config import (
        get_auto_clip_model,
        get_auto_qwen_model,
        get_auto_whisper_model,
        get_auto_yolo_model,
    )

    settings = get_settings()

    # Resolve object detector (handles "auto")
    detector = request.object_detector or settings.get_object_detector()

    # Resolve model names (handles "auto" -> actual model name)
    whisper_model = (
        get_auto_whisper_model() if request.whisper_model == "auto"
        else request.whisper_model
    )
    qwen_model = (
        get_auto_qwen_model() if request.qwen_model == "auto"
        else request.qwen_model
    )
    yolo_model = (
        get_auto_yolo_model() if request.yolo_model == "auto"
        else request.yolo_model
    )
    clip_model = (
        get_auto_clip_model() if request.clip_model == "auto"
        else request.clip_model
    )

    logger.info(
        f"Batch {batch_id} models: whisper={whisper_model}, qwen={qwen_model}, "
        f"yolo={yolo_model}, clip={clip_model}, detector={detector}"
    )

    batch_start_time = time.time()
    peak_memory = _get_memory_mb()

    def update_batch_progress(extractor: str, message: str, current: int | None = None, total: int | None = None) -> None:
        nonlocal peak_memory
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].current_extractor = extractor
                batch_jobs[batch_id].progress = JobProgress(
                    message=message, current=current, total=total
                )
                # Update memory and elapsed time
                current_mem = _get_memory_mb()
                peak_memory = max(peak_memory, current_mem)
                batch_jobs[batch_id].memory_mb = current_mem
                batch_jobs[batch_id].peak_memory_mb = peak_memory
                batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 1)

    def update_file_status(file_idx: int, status: str, result_key: str | None = None, result: Any = None, error: str | None = None) -> None:
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
                        timing.duration_seconds = round((completed - timing.started_at).total_seconds(), 2)
                        timing.files_processed = files_processed
                        break

    def update_file_timing(file_idx: int, extractor: str, duration: float) -> None:
        """Record per-file timing for an extractor."""
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].timings[extractor] = round(duration, 2)

    try:
        with batch_jobs_lock:
            batch_jobs[batch_id].status = "running"

        files = request.files
        total_files = len(files)

        # Stage 1: Metadata (parallel ffprobe for speed)
        if request.enable_metadata:
            start_extractor_timing("metadata")
            update_batch_progress(
                "metadata",
                f"Running ffprobe ({FFPROBE_WORKERS} parallel workers)...",
                0,
                total_files
            )

            # Run all ffprobe calls in parallel
            probe_results = run_ffprobe_batch(files)

            # Extract metadata from each probe result
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("metadata", f"Processing {Path(file_path).name}", i + 1, total_files)
                probe_data = probe_results.get(file_path)

                if isinstance(probe_data, Exception):
                    logger.warning(f"Metadata failed for {file_path}: {probe_data}")
                    update_file_status(i, "running", "metadata", None, str(probe_data))
                    update_file_timing(i, "metadata", time.time() - file_start)
                    continue

                try:
                    metadata = extract_metadata(file_path, probe_data)
                    update_file_status(i, "running", "metadata", metadata.model_dump())
                except Exception as e:
                    logger.warning(f"Metadata failed for {file_path}: {e}")
                    update_file_status(i, "running", "metadata", None, str(e))
                update_file_timing(i, "metadata", time.time() - file_start)
            end_extractor_timing("metadata", total_files)

        # Stage 2: Telemetry (always runs - lightweight, no models)
        start_extractor_timing("telemetry")
        update_batch_progress("telemetry", "Extracting telemetry...", 0, total_files)
        for i, file_path in enumerate(files):
            file_start = time.time()
            update_batch_progress("telemetry", f"Processing {Path(file_path).name}", i + 1, total_files)
            try:
                telemetry = extract_telemetry(file_path)
                update_file_status(i, "running", "telemetry", telemetry.model_dump() if telemetry else None)
            except Exception as e:
                logger.warning(f"Telemetry failed for {file_path}: {e}")
            update_file_timing(i, "telemetry", time.time() - file_start)
        end_extractor_timing("telemetry", total_files)

        # Stage 3: Voice Activity Detection (Silero VAD - lightweight)
        if request.enable_vad:
            start_extractor_timing("vad")
            update_batch_progress("vad", "Loading VAD model...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("vad", f"Analyzing {Path(file_path).name}", i + 1, total_files)
                try:
                    vad_result = detect_voice_activity(file_path)
                    update_file_status(i, "running", "vad", vad_result)
                except Exception as e:
                    logger.warning(f"VAD failed for {file_path}: {e}")
                update_file_timing(i, "vad", time.time() - file_start)
            # Unload VAD model to free memory
            update_batch_progress("vad", "Unloading VAD model...", None, None)
            unload_vad_model()
            end_extractor_timing("vad", total_files)

        # Stage 4: Scenes (PySceneDetect - moderate memory)
        if request.enable_scenes:
            start_extractor_timing("scenes")
            update_batch_progress("scenes", "Detecting scenes...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("scenes", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    scenes = extract_scenes(file_path)
                    update_file_status(i, "running", "scenes", scenes.model_dump() if scenes else None)
                except Exception as e:
                    logger.warning(f"Scenes failed for {file_path}: {e}")
                update_file_timing(i, "scenes", time.time() - file_start)
            end_extractor_timing("scenes", total_files)

        # Stage 5: Transcript (Whisper - heavy model)
        if request.enable_transcript:
            start_extractor_timing("transcript")
            # Clear memory before loading heavy model
            logger.info("Clearing memory before Whisper...")
            _clear_memory()
            update_batch_progress("transcript", "Loading Whisper model...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("transcript", f"Transcribing {Path(file_path).name}", i + 1, total_files)
                try:
                    transcript = extract_transcript(
                        file_path,
                        model=whisper_model,
                        language_hints=request.language_hints,
                        context_hint=request.context_hint,
                    )
                    update_file_status(i, "running", "transcript", transcript.model_dump() if transcript else None)
                except Exception as e:
                    logger.warning(f"Transcript failed for {file_path}: {e}")
                update_file_timing(i, "transcript", time.time() - file_start)
            # Unload Whisper to free memory
            update_batch_progress("transcript", "Unloading Whisper model...", None, None)
            unload_whisper_model()
            end_extractor_timing("transcript", total_files)

        # Stage 6: Motion Analysis (for smart sampling of faces/objects/clip/ocr)
        # Store motion data per file for later use
        motion_data: dict[int, Any] = {}  # file_idx -> MotionAnalysis
        adaptive_timestamps: dict[int, list[float]] = {}  # file_idx -> timestamps

        # Skip motion analysis if timestamps are already provided (e.g., Pass 2 reusing Pass 1 data)
        has_precomputed_timestamps = bool(request.qwen_timestamps)
        needs_motion = request.enable_motion or (
            (request.enable_objects or request.enable_faces or request.enable_clip or request.enable_ocr)
            and not has_precomputed_timestamps
        )

        if has_precomputed_timestamps and request.qwen_timestamps:
            logger.info(f"Using {len(request.qwen_timestamps)} pre-computed timestamps, skipping motion analysis")

        if needs_motion:
            start_extractor_timing("motion")
            update_batch_progress("motion", "Analyzing camera motion...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("motion", f"Analyzing {Path(file_path).name}", i + 1, total_files)
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

        # Stage 7: Objects (YOLO with motion-adaptive timestamps, or Qwen)
        # Store person timestamps for face detection
        person_timestamps: dict[int, list[float]] = {}  # file_idx -> timestamps where persons detected

        if request.enable_objects:
            start_extractor_timing("objects")
            logger.info(f"Objects enabled. Detector type: {type(detector)}, value: {detector}, is QWEN: {detector == ObjectDetector.QWEN}")
            if detector == ObjectDetector.QWEN:
                # Clear memory before loading heavy model
                logger.info("Clearing memory before Qwen...")
                _clear_memory()
                update_batch_progress("objects", "Loading Qwen model...", 0, total_files)
                # Log contexts for debugging
                logger.info(f"Qwen batch contexts: {request.contexts}")

                for i, file_path in enumerate(files):
                    file_start = time.time()
                    fname = Path(file_path).name
                    update_batch_progress("objects", f"Analyzing: {fname}", i + 1, total_files)
                    try:
                        # Use motion-based timestamps for Qwen (fewer samples for VLM)
                        motion = motion_data.get(i)
                        timestamps = request.qwen_timestamps  # Use provided timestamps first
                        if timestamps is None and motion:
                            timestamps = get_sample_timestamps(motion, max_samples=5)

                        # Get per-file context
                        file_context = request.contexts.get(file_path) if request.contexts else None
                        logger.info(f"Calling Qwen with context for {fname}: {file_context}")
                        objects = extract_objects_qwen(
                            file_path,
                            timestamps=timestamps,
                            model_name=qwen_model,
                            context=file_context,
                        )
                        objects_data: dict[str, Any] = {"summary": objects.summary}
                        if objects.descriptions:
                            objects_data["descriptions"] = objects.descriptions
                        logger.info(f"Qwen result for file {i}: summary={objects.summary}, descriptions={objects.descriptions}")
                        update_file_status(i, "running", "objects", objects_data)
                        logger.info(f"Stored objects_data for file {i}: {objects_data}")

                        # Qwen doesn't return per-detection timestamps, so no person_timestamps
                        person_timestamps[i] = []
                    except Exception as e:
                        logger.warning(f"Objects failed for {file_path}: {e}", exc_info=True)
                        person_timestamps[i] = []
                    update_file_timing(i, "objects", time.time() - file_start)
                # Unload Qwen to free memory
                update_batch_progress("objects", "Unloading Qwen model...", None, None)
                unload_qwen_model()
            else:
                # YOLO with motion-adaptive timestamps
                update_batch_progress("objects", "Detecting objects with YOLO...", 0, total_files)
                for i, file_path in enumerate(files):
                    file_start = time.time()
                    update_batch_progress("objects", f"Processing {Path(file_path).name}", i + 1, total_files)
                    try:
                        # Use motion-adaptive timestamps if available
                        timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                        objects = extract_objects(file_path, timestamps=timestamps, model_name=yolo_model)

                        if objects:
                            update_file_status(i, "running", "objects", {"summary": objects.summary})

                            # Collect timestamps where persons were detected (for smart face sampling)
                            person_ts = list(set(
                                d.timestamp for d in objects.detections if d.label == "person"
                            ))
                            person_timestamps[i] = sorted(person_ts)
                            if person_ts:
                                logger.info(f"Found {len(person_ts)} person frames in {Path(file_path).name}")
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

        # Stage 8: Faces (YOLO-triggered - only where persons detected)
        if request.enable_faces:
            start_extractor_timing("faces")
            update_batch_progress("faces", "Detecting faces...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("faces", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    # Smart sampling: use person timestamps from YOLO if available
                    person_ts = person_timestamps.get(i, [])

                    if request.enable_objects and person_ts:
                        # YOLO-triggered: only detect faces where persons were found
                        faces = extract_faces(file_path, timestamps=person_ts)
                        logger.info(f"Face detection on {len(person_ts)} person frames for {Path(file_path).name}")
                    elif request.enable_objects and not person_ts:
                        # YOLO ran but found no persons - skip face detection
                        logger.info(f"Skipping face detection for {Path(file_path).name} - no persons detected by YOLO")
                        faces = None
                    else:
                        # Objects disabled - fall back to motion-adaptive sampling
                        timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                        faces = extract_faces(file_path, timestamps=timestamps)

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
                        update_file_status(i, "running", "faces", {"count": 0, "unique_estimate": 0, "detections": []})
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
                file_start = time.time()
                update_batch_progress("ocr", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    # Use motion-adaptive timestamps if available
                    timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                    ocr = extract_ocr(file_path, timestamps=timestamps)
                    update_file_status(i, "running", "ocr", ocr.model_dump() if ocr else None)
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
            update_batch_progress("clip", "Extracting CLIP embeddings...", 0, total_files)
            for i, file_path in enumerate(files):
                file_start = time.time()
                update_batch_progress("clip", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    # Use motion-adaptive timestamps if available
                    timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                    clip = extract_clip(file_path, timestamps=timestamps, model_name=clip_model)
                    if clip:
                        update_file_status(i, "running", "clip", {"model": clip.model, "count": len(clip.segments)})
                    else:
                        update_file_status(i, "running", "clip", None)
                except Exception as e:
                    logger.warning(f"CLIP failed for {file_path}: {e}")
                update_file_timing(i, "clip", time.time() - file_start)
            # Unload CLIP model to free memory
            update_batch_progress("clip", "Unloading CLIP model...", None, None)
            unload_clip_model()
            end_extractor_timing("clip", total_files)

        # Mark all files as completed
        with batch_jobs_lock:
            for i in range(len(files)):
                # Log results before marking complete
                result_keys = list(batch_jobs[batch_id].files[i].results.keys())
                logger.info(f"Batch {batch_id} file {i} results before completion: keys={result_keys}")
                batch_jobs[batch_id].files[i].status = "completed"
            batch_jobs[batch_id].status = "completed"
            batch_jobs[batch_id].current_extractor = None
            batch_jobs[batch_id].progress = None
            batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
            # Final metrics
            batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 2)
            batch_jobs[batch_id].memory_mb = _get_memory_mb()
            batch_jobs[batch_id].peak_memory_mb = max(peak_memory, _get_memory_mb())

        # Log timing summary
        logger.info(f"Batch {batch_id} completed in {batch_jobs[batch_id].elapsed_seconds}s, peak memory: {batch_jobs[batch_id].peak_memory_mb}MB")
        for timing in batch_jobs[batch_id].extractor_timings:
            logger.info(f"  {timing.extractor}: {timing.duration_seconds}s ({timing.files_processed} files)")

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}")
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].status = "failed"
                batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)
                batch_jobs[batch_id].elapsed_seconds = round(time.time() - batch_start_time, 2)
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
    thread = threading.Thread(
        target=run_batch_job,
        args=(batch_id, request)
    )
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


@app.get("/hardware")
async def hardware():
    """Get hardware capabilities and auto-selected models.

    Returns information about available GPU/VRAM and which models
    will be used with the current "auto" settings.
    """
    from polybos_engine.config import get_vram_summary
    return get_vram_summary()


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract metadata and features from video file.

    This endpoint runs enabled extractors on the video file and returns
    the combined results. Use enable_* flags to select extractors.

    For RAW formats (BRAW, ARRIRAW) that ffmpeg can't decode, provide a
    proxy_file for frame-based analysis. Metadata is extracted from the
    main file, while transcript/faces/motion/etc. use the proxy.
    """
    start_time = time.time()

    # Validate file exists
    file_path = Path(request.file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {request.file}")

    # Validate proxy file if provided
    proxy_path: str | None = None
    if request.proxy_file:
        proxy = Path(request.proxy_file)
        if not proxy.exists():
            raise HTTPException(
                status_code=404, detail=f"Proxy file not found: {request.proxy_file}"
            )
        if not proxy.is_file():
            raise HTTPException(
                status_code=400, detail=f"Proxy is not a file: {request.proxy_file}"
            )
        proxy_path = request.proxy_file
        logger.info(f"Using proxy file for frame analysis: {proxy.name}")

    # Determine which file to use for frame-based analysis
    # Metadata always from original, frames from proxy if provided
    analysis_file = proxy_path or request.file

    settings = get_settings()
    logger.info(f"Starting extraction for: {request.file}")

    # Extract metadata
    metadata = None
    if request.enable_metadata:
        try:
            metadata = extract_metadata(request.file)
            res = metadata.resolution
            logger.info(f"Metadata extracted: {metadata.duration}s, {res.width}x{res.height}")

            # Detect shot type (part of metadata) - uses frames
            try:
                shot_type = detect_shot_type(analysis_file)
                if shot_type:
                    metadata.shot_type = shot_type
                    logger.info(f"Shot type detected: {shot_type.primary} ({shot_type.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Shot type detection failed: {e}")
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")

    # Extract scenes (frame-based)
    scenes = None
    if request.enable_scenes:
        try:
            scenes = extract_scenes(analysis_file)
            logger.info(f"Scene detection complete: {scenes.count} scenes")
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")

    # Extract transcript (audio-based, uses analysis file)
    transcript = None
    if request.enable_transcript:
        try:
            transcript = extract_transcript(
                analysis_file,
                model=request.whisper_model,
                language=request.language,
                fallback_language=request.fallback_language,
                language_hints=request.language_hints,
                context_hint=request.context_hint,
            )
            logger.info(f"Transcription complete: {len(transcript.segments)} segments")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")

    # Extract faces (frame-based)
    faces = None
    if request.enable_faces:
        try:
            faces = extract_faces(
                analysis_file,
                scenes=scenes.detections if scenes else None,
                sample_fps=request.face_sample_fps,
            )
            logger.info(f"Face detection: {faces.count} faces, ~{faces.unique_estimate} unique")
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")

    # Motion analysis (frame-based)
    motion_result: MotionResult | None = None
    motion_analysis = None
    if request.enable_motion:
        try:
            motion_analysis = analyze_motion(analysis_file)
            motion_result = MotionResult(
                duration=motion_analysis.duration,
                fps=motion_analysis.fps,
                primary_motion=str(motion_analysis.primary_motion),
                segments=[
                    MotionSegment(
                        start=s.start,
                        end=s.end,
                        motion_type=str(s.motion_type),
                        intensity=s.intensity,
                    )
                    for s in motion_analysis.segments
                ],
                avg_intensity=motion_analysis.avg_intensity,
                is_stable=motion_analysis.is_stable,
            )
            logger.info(f"Motion: {motion_result.primary_motion}, {len(motion_result.segments)} segments")
        except Exception as e:
            logger.warning(f"Motion analysis failed: {e}")

    # Extract objects (frame-based)
    objects = None
    if request.enable_objects:
        try:
            scene_detections = scenes.detections if scenes else None
            # Resolve object detector (handles "auto")
            detector = request.object_detector or settings.get_object_detector()
            if detector == ObjectDetector.QWEN:
                # Use timestamps from request (frontend decides)
                timestamps = request.qwen_timestamps
                if timestamps:
                    logger.info(f"Using {len(timestamps)} timestamps from request")
                else:
                    logger.info("No timestamps provided, Qwen will sample from middle")

                objects = extract_objects_qwen(
                    analysis_file,
                    timestamps=timestamps,
                    context=request.context,
                )
                logger.info(f"Qwen object detection: {len(objects.summary)} types")
            else:
                objects = extract_objects(
                    analysis_file,
                    scenes=scene_detections,
                    sample_fps=request.object_sample_fps,
                )
                logger.info(f"YOLO object detection: {len(objects.detections)} detections")
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")

    # Extract CLIP embeddings (frame-based)
    embeddings = None
    if request.enable_clip:
        try:
            embeddings = extract_clip(analysis_file, scenes=scenes)
            logger.info(f"CLIP extraction complete: {len(embeddings.segments)} segments")
        except Exception as e:
            logger.warning(f"CLIP extraction failed: {e}")

    # Extract OCR (frame-based)
    ocr = None
    if request.enable_ocr:
        try:
            ocr = extract_ocr(analysis_file, scenes=scenes)
            logger.info(f"OCR complete: {len(ocr.detections)} text regions")
        except Exception as e:
            logger.warning(f"OCR failed: {e}")

    # Extract telemetry (always - lightweight, no models)
    telemetry = None
    try:
        telemetry = extract_telemetry(request.file)
        if telemetry:
            logger.info(f"Telemetry: {len(telemetry.points)} points from {telemetry.source}")
    except Exception as e:
        logger.warning(f"Telemetry extraction failed: {e}")

    extraction_time = time.time() - start_time
    logger.info(f"Extraction complete in {extraction_time:.2f}s")

    return ExtractResponse(
        file=request.file,
        filename=file_path.name,
        extracted_at=datetime.now(timezone.utc),
        extraction_time_seconds=round(extraction_time, 2),
        api_version=settings.api_version,
        engine_version=__version__,
        metadata=metadata,
        transcript=transcript,
        faces=faces,
        scenes=scenes,
        objects=objects,
        embeddings=embeddings,
        ocr=ocr,
        motion=motion_result,
        telemetry=telemetry,
    )


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
                "description": "Object detection (YOLO or Qwen2-VL)",
                "enable_flag": "enable_objects",
                "backend": str(get_settings().get_object_detector()),
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
