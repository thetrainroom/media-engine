"""FastAPI application for Polybos Media Engine."""

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
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
    unload_qwen_model,
    unload_vad_model,
    unload_whisper_model,
)
from polybos_engine.extractors.shot_type import detect_shot_type
from polybos_engine.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MotionResult,
    MotionSegment,
    TelemetryResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Polybos Media Engine",
    description="AI-powered video extraction API",
    version=__version__,
)

# Add CORS middleware for demo frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve demo frontend
demo_path = Path(__file__).parent.parent / "demo"
if demo_path.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_path), html=True), name="demo")


# ============================================================================
# Job Queue System
# ============================================================================

class JobProgress(BaseModel):
    """Progress within current extraction step."""
    message: str  # e.g., "Loading model...", "Processing frame 2/5"
    current: int | None = None
    total: int | None = None


class JobStatus(BaseModel):
    """Status of an extraction job."""
    job_id: str
    status: str  # pending, running, completed, failed
    file: str
    filename: str
    current_step: str | None = None
    progress: JobProgress | None = None  # Progress within current step
    completed_steps: list[str] = []
    results: dict[str, Any] = {}
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# In-memory job store (use Redis in production)
jobs: dict[str, JobStatus] = {}
jobs_lock = threading.Lock()


def _build_qwen_context(results: dict[str, Any]) -> dict[str, str] | None:
    """Build context for Qwen from earlier extraction results.

    Extracts relevant information from metadata, transcript, faces, etc.
    to provide context for more accurate scene descriptions.
    """
    context: dict[str, str] = {}

    # From metadata
    metadata = results.get("metadata")
    if metadata:
        # Device info
        device = metadata.get("device")
        if device and device.get("make"):
            device_str = device.get("make", "")
            if device.get("model"):
                device_str += f" {device['model']}"
            if device_str:
                context["device"] = device_str

        # Shot type
        shot_type = metadata.get("shot_type")
        if shot_type and shot_type.get("primary"):
            context["shot_type"] = shot_type["primary"]

        # GPS location (could be reverse geocoded in future)
        gps = metadata.get("gps")
        if gps and gps.get("latitude"):
            context["coordinates"] = f"{gps['latitude']:.4f}, {gps['longitude']:.4f}"

    # From transcript
    transcript = results.get("transcript")
    if transcript:
        # Language
        lang = transcript.get("language")
        if lang and isinstance(lang, str):
            # Map language codes to readable names
            lang_names = {
                "en": "English", "de": "German", "fr": "French",
                "es": "Spanish", "it": "Italian", "no": "Norwegian",
                "sv": "Swedish", "da": "Danish", "nl": "Dutch",
                "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
            }
            context["language"] = lang_names.get(lang, lang)

        # Speaker count
        speaker_count = transcript.get("speaker_count")
        if speaker_count and speaker_count > 0:
            context["speakers"] = f"{speaker_count} speaker(s)"

    # From faces (could be matched to database in future)
    faces = results.get("faces")
    if faces:
        unique_count = faces.get("unique_estimate", 0)
        if unique_count > 0:
            context["people_visible"] = f"{unique_count} person(s)"

    # Return None if no context was found
    return context if context else None


def run_extraction_job(
    job_id: str,
    request: ExtractRequest,
) -> None:
    """Run extraction in background thread."""
    settings = get_settings()
    file_path = request.file
    # Use proxy file for frame-based analysis if provided
    analysis_file = request.proxy_file or request.file
    # Use request parameter or fall back to settings
    # Resolve object detector (handles "auto")
    detector = request.object_detector or settings.get_object_detector()
    context = request.context

    def update_step(step: str) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].current_step = step
                jobs[job_id].progress = None

    def update_progress(message: str, current: int | None = None, total: int | None = None) -> None:
        """Update progress within current step."""
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].progress = JobProgress(
                    message=message, current=current, total=total
                )

    def complete_step(step: str, result: Any) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].completed_steps.append(step)
                jobs[job_id].current_step = None
                jobs[job_id].progress = None
                if result is not None:
                    jobs[job_id].results[step] = result

    try:
        with jobs_lock:
            jobs[job_id].status = "running"

        # Metadata
        if request.enable_metadata:
            update_step("metadata")
            metadata = extract_metadata(file_path)
            complete_step("metadata", metadata.model_dump())

            # Shot type (part of metadata) - uses frames
            update_step("shot_type")
            try:
                from polybos_engine.extractors.shot_type import detect_shot_type
                shot_type = detect_shot_type(analysis_file)
                if shot_type:
                    complete_step("shot_type", shot_type.model_dump())
                else:
                    complete_step("shot_type", None)
            except Exception as e:
                logger.warning(f"Shot type failed: {e}")
                complete_step("shot_type", None)

        # Scenes (frame-based)
        if request.enable_scenes:
            update_step("scenes")
            try:
                scenes = extract_scenes(analysis_file)
                complete_step("scenes", scenes.model_dump() if scenes else None)
            except Exception as e:
                logger.warning(f"Scenes failed: {e}")
                complete_step("scenes", None)

        # Transcript (audio-based)
        if request.enable_transcript:
            update_step("transcript")
            try:
                transcript = extract_transcript(
                    analysis_file,
                    model=request.whisper_model,
                    language=request.language,
                    fallback_language=request.fallback_language,
                    language_hints=request.language_hints,
                    context_hint=request.context_hint,
                    progress_callback=update_progress,
                )
                complete_step("transcript", transcript.model_dump() if transcript else None)
            except Exception as e:
                logger.warning(f"Transcript failed: {e}")
                complete_step("transcript", None)

        # Faces (frame-based)
        if request.enable_faces:
            update_step("faces")
            try:
                # Get scene detections for better sampling
                scene_detections = None
                if "scenes" in jobs[job_id].results and jobs[job_id].results["scenes"]:
                    from polybos_engine.schemas import SceneDetection
                    scene_data = jobs[job_id].results["scenes"]
                    scene_detections = [
                        SceneDetection(**s) for s in scene_data.get("detections", [])
                    ]

                faces = extract_faces(
                    analysis_file,
                    scenes=scene_detections,
                    sample_fps=request.face_sample_fps,
                )
                # Exclude embeddings from job results (too large for JSON polling)
                if faces:
                    faces_data = {
                        "count": faces.count,
                        "unique_estimate": faces.unique_estimate,
                        "detections": [
                            {
                                "timestamp": d.timestamp,
                                "bbox": d.bbox.model_dump(),
                                "confidence": d.confidence,
                                "image_base64": d.image_base64,
                                "needs_review": d.needs_review,
                                "review_reason": d.review_reason,
                            }
                            for d in faces.detections
                        ],
                    }
                    complete_step("faces", faces_data)
                else:
                    complete_step("faces", None)
            except Exception as e:
                logger.warning(f"Faces failed: {e}")
                complete_step("faces", None)

        # Motion analysis (frame-based)
        if request.enable_motion:
            update_step("motion")
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
                complete_step("motion", motion_result.model_dump())
                logger.info(f"Motion: {motion_result.primary_motion}, {len(motion_result.segments)} segments")
            except Exception as e:
                logger.warning(f"Motion analysis failed: {e}")
                complete_step("motion", None)

        # Objects (frame-based, use configurable detector)
        if request.enable_objects:
            update_step("objects")
            try:
                # Get scene detections for scene-aware sampling
                scene_detections = None
                if "scenes" in jobs[job_id].results and jobs[job_id].results["scenes"]:
                    from polybos_engine.schemas import SceneDetection
                    scene_data = jobs[job_id].results["scenes"]
                    scene_detections = [
                        SceneDetection(**s) for s in scene_data.get("detections", [])
                    ]

                # Use configured detector (yolo or qwen)
                if detector == ObjectDetector.QWEN:
                    # Use passed context, or build from earlier extraction results
                    qwen_context = context or _build_qwen_context(jobs[job_id].results)

                    # Use timestamps from request (frontend decides)
                    timestamps = request.qwen_timestamps
                    if timestamps:
                        logger.info(f"Using {len(timestamps)} timestamps from request")
                    else:
                        logger.info("No timestamps provided, Qwen will sample from middle")

                    objects = extract_objects_qwen(
                        analysis_file,
                        timestamps=timestamps,
                        context=qwen_context,
                        progress_callback=update_progress,
                    )
                else:
                    objects = extract_objects(analysis_file, scenes=scene_detections)

                # Only include summary for job polling (full detections too large)
                if objects:
                    objects_data: dict[str, Any] = {"summary": objects.summary}
                    if objects.descriptions:
                        objects_data["descriptions"] = objects.descriptions
                    complete_step("objects", objects_data)
                else:
                    complete_step("objects", None)
            except Exception as e:
                logger.warning(f"Objects failed: {e}")
                complete_step("objects", None)

        # CLIP (frame-based)
        if request.enable_clip:
            update_step("clip")
            try:
                embeddings = extract_clip(analysis_file)
                complete_step("clip", {"count": len(embeddings.segments)} if embeddings else None)
            except Exception as e:
                logger.warning(f"CLIP failed: {e}")
                complete_step("clip", None)

        # OCR (frame-based)
        if request.enable_ocr:
            update_step("ocr")
            try:
                ocr = extract_ocr(analysis_file)
                complete_step("ocr", ocr.model_dump() if ocr else None)
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                complete_step("ocr", None)

        # Telemetry
        update_step("telemetry")
        try:
            telemetry = extract_telemetry(file_path)
            complete_step("telemetry", telemetry.model_dump() if telemetry else None)
        except Exception as e:
            logger.warning(f"Telemetry failed: {e}")
            complete_step("telemetry", None)

        with jobs_lock:
            jobs[job_id].status = "completed"
            jobs[job_id].completed_at = datetime.now(timezone.utc)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].status = "failed"
                jobs[job_id].error = str(e)
                jobs[job_id].completed_at = datetime.now(timezone.utc)


@app.post("/jobs")
async def create_job(request: ExtractRequest) -> dict[str, str]:
    """Create a new extraction job."""
    file_path = Path(request.file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file}")

    job_id = str(uuid.uuid4())[:8]
    job = JobStatus(
        job_id=job_id,
        status="pending",
        file=request.file,
        filename=file_path.name,
        created_at=datetime.now(timezone.utc),
    )

    with jobs_lock:
        jobs[job_id] = job

    # Start background thread
    thread = threading.Thread(
        target=run_extraction_job,
        args=(job_id, request)
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    """Get job status and partial results."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return jobs[job_id]


# ============================================================================
# Batch Job System (extractor-first processing for memory efficiency)
# ============================================================================

class BatchRequest(BaseModel):
    """Request for batch extraction."""
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
    object_detector: ObjectDetector | None = None
    whisper_model: str = "large-v3"


class BatchFileStatus(BaseModel):
    """Status for a single file in a batch."""
    file: str
    filename: str
    status: str  # pending, running, completed, failed
    results: dict[str, Any] = {}
    error: str | None = None


class BatchJobStatus(BaseModel):
    """Status of a batch extraction job."""
    batch_id: str
    status: str  # pending, running, completed, failed
    current_extractor: str | None = None
    progress: JobProgress | None = None
    files: list[BatchFileStatus] = []
    created_at: datetime
    completed_at: datetime | None = None


# In-memory batch store
batch_jobs: dict[str, BatchJobStatus] = {}
batch_jobs_lock = threading.Lock()


def run_batch_job(batch_id: str, request: BatchRequest) -> None:
    """Run batch extraction - processes all files per extractor stage.

    This is more memory efficient as each model is loaded once,
    processes all files, then is unloaded before the next model.
    """
    settings = get_settings()
    # Resolve object detector (handles "auto")
    detector = request.object_detector or settings.get_object_detector()

    def update_batch_progress(extractor: str, message: str, current: int | None = None, total: int | None = None) -> None:
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].current_extractor = extractor
                batch_jobs[batch_id].progress = JobProgress(
                    message=message, current=current, total=total
                )

    def update_file_status(file_idx: int, status: str, result_key: str | None = None, result: Any = None, error: str | None = None) -> None:
        with batch_jobs_lock:
            if batch_id in batch_jobs and file_idx < len(batch_jobs[batch_id].files):
                batch_jobs[batch_id].files[file_idx].status = status
                if result_key and result is not None:
                    batch_jobs[batch_id].files[file_idx].results[result_key] = result
                if error:
                    batch_jobs[batch_id].files[file_idx].error = error

    try:
        with batch_jobs_lock:
            batch_jobs[batch_id].status = "running"

        files = request.files
        total_files = len(files)

        # Stage 1: Metadata (parallel ffprobe for speed)
        if request.enable_metadata:
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
                update_batch_progress("metadata", f"Processing {Path(file_path).name}", i + 1, total_files)
                probe_data = probe_results.get(file_path)

                if isinstance(probe_data, Exception):
                    logger.warning(f"Metadata failed for {file_path}: {probe_data}")
                    update_file_status(i, "running", "metadata", None, str(probe_data))
                    continue

                try:
                    metadata = extract_metadata(file_path, probe_data)
                    update_file_status(i, "running", "metadata", metadata.model_dump())
                except Exception as e:
                    logger.warning(f"Metadata failed for {file_path}: {e}")
                    update_file_status(i, "running", "metadata", None, str(e))

        # Stage 2: Voice Activity Detection (Silero VAD - lightweight)
        if request.enable_vad:
            update_batch_progress("vad", "Loading VAD model...", 0, total_files)
            for i, file_path in enumerate(files):
                update_batch_progress("vad", f"Analyzing {Path(file_path).name}", i + 1, total_files)
                try:
                    vad_result = detect_voice_activity(file_path)
                    update_file_status(i, "running", "vad", vad_result)
                except Exception as e:
                    logger.warning(f"VAD failed for {file_path}: {e}")
            # Unload VAD model to free memory
            update_batch_progress("vad", "Unloading VAD model...", None, None)
            unload_vad_model()

        # Stage 3: Scenes (PySceneDetect - moderate memory)
        if request.enable_scenes:
            update_batch_progress("scenes", "Detecting scenes...", 0, total_files)
            for i, file_path in enumerate(files):
                update_batch_progress("scenes", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    scenes = extract_scenes(file_path)
                    update_file_status(i, "running", "scenes", scenes.model_dump() if scenes else None)
                except Exception as e:
                    logger.warning(f"Scenes failed for {file_path}: {e}")

        # Stage 4: Transcript (Whisper - heavy model)
        if request.enable_transcript:
            update_batch_progress("transcript", "Loading Whisper model...", 0, total_files)
            for i, file_path in enumerate(files):
                update_batch_progress("transcript", f"Transcribing {Path(file_path).name}", i + 1, total_files)
                try:
                    transcript = extract_transcript(file_path, model=request.whisper_model)
                    update_file_status(i, "running", "transcript", transcript.model_dump() if transcript else None)
                except Exception as e:
                    logger.warning(f"Transcript failed for {file_path}: {e}")
            # Unload Whisper to free memory
            update_batch_progress("transcript", "Unloading Whisper model...", None, None)
            unload_whisper_model()

        # Stage 5: Motion Analysis (for smart sampling of faces/objects/clip/ocr)
        # Store motion data per file for later use
        motion_data: dict[int, Any] = {}  # file_idx -> MotionAnalysis
        adaptive_timestamps: dict[int, list[float]] = {}  # file_idx -> timestamps

        needs_motion = request.enable_motion or request.enable_objects or request.enable_faces or request.enable_clip or request.enable_ocr
        if needs_motion:
            update_batch_progress("motion", "Analyzing camera motion...", 0, total_files)
            for i, file_path in enumerate(files):
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

        # Stage 6: Objects (YOLO with motion-adaptive timestamps, or Qwen)
        # Store person timestamps for face detection
        person_timestamps: dict[int, list[float]] = {}  # file_idx -> timestamps where persons detected

        if request.enable_objects:
            if detector == ObjectDetector.QWEN:
                update_batch_progress("objects", "Loading Qwen model...", 0, total_files)
                for i, file_path in enumerate(files):
                    fname = Path(file_path).name
                    update_batch_progress("objects", f"Analyzing: {fname}", i + 1, total_files)
                    try:
                        # Use motion-based timestamps for Qwen (fewer samples for VLM)
                        motion = motion_data.get(i)
                        timestamps: list[float] | None = None
                        if motion:
                            timestamps = get_sample_timestamps(motion, max_samples=5)

                        objects = extract_objects_qwen(file_path, timestamps=timestamps)
                        objects_data: dict[str, Any] = {"summary": objects.summary}
                        if objects.descriptions:
                            objects_data["descriptions"] = objects.descriptions
                        update_file_status(i, "running", "objects", objects_data)

                        # Qwen doesn't return per-detection timestamps, so no person_timestamps
                        person_timestamps[i] = []
                    except Exception as e:
                        logger.warning(f"Objects failed for {file_path}: {e}")
                        person_timestamps[i] = []
                # Unload Qwen to free memory
                update_batch_progress("objects", "Unloading Qwen model...", None, None)
                unload_qwen_model()
            else:
                # YOLO with motion-adaptive timestamps
                update_batch_progress("objects", "Detecting objects with YOLO...", 0, total_files)
                for i, file_path in enumerate(files):
                    update_batch_progress("objects", f"Processing {Path(file_path).name}", i + 1, total_files)
                    try:
                        # Use motion-adaptive timestamps if available
                        timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                        objects = extract_objects(file_path, timestamps=timestamps)

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

        # Stage 7: Faces (YOLO-triggered - only where persons detected)
        if request.enable_faces:
            update_batch_progress("faces", "Detecting faces...", 0, total_files)
            for i, file_path in enumerate(files):
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

        # Stage 8: OCR (EasyOCR - moderate model) with motion-adaptive timestamps
        if request.enable_ocr:
            update_batch_progress("ocr", "Extracting text...", 0, total_files)
            for i, file_path in enumerate(files):
                update_batch_progress("ocr", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    # Use motion-adaptive timestamps if available
                    timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                    ocr = extract_ocr(file_path, timestamps=timestamps)
                    update_file_status(i, "running", "ocr", ocr.model_dump() if ocr else None)
                except Exception as e:
                    logger.warning(f"OCR failed for {file_path}: {e}")

        # Stage 9: CLIP embeddings with motion-adaptive timestamps
        if request.enable_clip:
            update_batch_progress("clip", "Extracting CLIP embeddings...", 0, total_files)
            for i, file_path in enumerate(files):
                update_batch_progress("clip", f"Processing {Path(file_path).name}", i + 1, total_files)
                try:
                    # Use motion-adaptive timestamps if available
                    timestamps = adaptive_timestamps.get(i) if adaptive_timestamps.get(i) else None
                    clip = extract_clip(file_path, timestamps=timestamps)
                    if clip:
                        update_file_status(i, "running", "clip", {"model": clip.model, "count": len(clip.segments)})
                    else:
                        update_file_status(i, "running", "clip", None)
                except Exception as e:
                    logger.warning(f"CLIP failed for {file_path}: {e}")

        # Mark all files as completed
        with batch_jobs_lock:
            for i in range(len(files)):
                batch_jobs[batch_id].files[i].status = "completed"
            batch_jobs[batch_id].status = "completed"
            batch_jobs[batch_id].current_extractor = None
            batch_jobs[batch_id].progress = None
            batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {e}")
        with batch_jobs_lock:
            if batch_id in batch_jobs:
                batch_jobs[batch_id].status = "failed"
                batch_jobs[batch_id].completed_at = datetime.now(timezone.utc)


@app.post("/batch")
async def create_batch(request: BatchRequest) -> dict[str, str]:
    """Create a new batch extraction job (memory-efficient extractor-first processing)."""
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
    )


@app.get("/browse")
async def browse_directory(
    path: str = Query("/Volumes/Backup", description="Directory to browse"),
):
    """Browse a directory for video files."""
    dir_path = Path(path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    video_extensions = {".mp4", ".mov", ".mxf", ".avi", ".mkv", ".m4v", ".webm"}
    items = []

    try:
        for item in sorted(dir_path.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                items.append({"name": item.name, "path": str(item), "type": "directory"})
            elif item.suffix.lower() in video_extensions:
                size_mb = item.stat().st_size / (1024 * 1024)
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "video",
                    "size_mb": round(size_mb, 1),
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    return {
        "path": str(dir_path),
        "parent": str(dir_path.parent) if dir_path.parent != dir_path else None,
        "items": items,
    }


# MIME types for video files
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mxf": "video/mxf",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
    ".webm": "video/webm",
}


@app.get("/video")
async def stream_video(
    request: Request,
    file: str = Query(..., description="Path to video file"),
):
    """Stream a video file with range request support for seeking."""
    file_path = Path(file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file}")

    file_size = file_path.stat().st_size
    content_type = VIDEO_MIME_TYPES.get(file_path.suffix.lower(), "video/mp4")

    # Parse range header for seeking support
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end" format
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        # Clamp end to file size
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(remaining, 1024 * 1024)  # 1MB chunks
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,  # Partial Content
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )
    else:
        # No range header - stream entire file
        def iter_file():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
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
                "description": "GPS/flight path from drone sidecar files",
                "endpoint": "/telemetry",
            },
        ]
    }


@app.get("/telemetry", response_model=TelemetryResult | None)
async def telemetry(
    file: str = Query(..., description="Path to video file"),
    sample_interval: float = Query(1.0, description="Sample one point every N seconds"),
):
    """Extract telemetry/flight path from video sidecar files.

    Returns GPS track with timestamps for drones with SRT sidecars (DJI, etc.)
    or exposure-only data for cameras with embedded subtitles (DJI Pocket).
    """
    file_path = Path(file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file}")

    try:
        result = extract_telemetry(file, sample_interval=sample_interval)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="No telemetry data found (no SRT sidecar or embedded subtitles)",
            )
        logger.info(f"Telemetry extracted: {len(result.points)} points from {result.source}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Telemetry extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Telemetry extraction failed: {e}")


@app.get("/telemetry/gpx")
async def telemetry_gpx(
    file: str = Query(..., description="Path to video file"),
    sample_interval: float = Query(1.0, description="Sample one point every N seconds"),
):
    """Export telemetry as GPX track for use in mapping applications."""
    from fastapi.responses import Response

    file_path = Path(file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file}")

    result = extract_telemetry(file, sample_interval=sample_interval)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No telemetry data found",
        )

    gpx_content = result.to_gpx()
    return Response(
        content=gpx_content,
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f"attachment; filename={file_path.stem}.gpx"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
