"""FastAPI application for Polybos Media Engine."""

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from polybos_engine import __version__
from polybos_engine.config import get_settings
from polybos_engine.extractors import (
    extract_clip,
    extract_faces,
    extract_metadata,
    extract_objects,
    extract_ocr,
    extract_scenes,
    extract_telemetry,
    extract_transcript,
)
from polybos_engine.extractors.shot_type import detect_shot_type
from polybos_engine.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
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

class JobStatus(BaseModel):
    """Status of an extraction job."""
    job_id: str
    status: str  # pending, running, completed, failed
    file: str
    filename: str
    current_step: str | None = None
    completed_steps: list[str] = []
    results: dict[str, Any] = {}
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


# In-memory job store (use Redis in production)
jobs: dict[str, JobStatus] = {}
jobs_lock = threading.Lock()


def run_extraction_job(job_id: str, file_path: str) -> None:
    """Run extraction in background thread."""
    settings = get_settings()

    def update_step(step: str) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].current_step = step

    def complete_step(step: str, result: Any) -> None:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].completed_steps.append(step)
                jobs[job_id].current_step = None
                if result is not None:
                    jobs[job_id].results[step] = result

    try:
        with jobs_lock:
            jobs[job_id].status = "running"

        # Metadata
        update_step("metadata")
        metadata = extract_metadata(file_path)
        complete_step("metadata", metadata.model_dump())

        # Shot type
        update_step("shot_type")
        try:
            from polybos_engine.extractors.shot_type import detect_shot_type
            shot_type = detect_shot_type(file_path)
            if shot_type:
                complete_step("shot_type", shot_type.model_dump())
            else:
                complete_step("shot_type", None)
        except Exception as e:
            logger.warning(f"Shot type failed: {e}")
            complete_step("shot_type", None)

        # Scenes
        update_step("scenes")
        try:
            scenes = extract_scenes(file_path)
            complete_step("scenes", scenes.model_dump() if scenes else None)
        except Exception as e:
            logger.warning(f"Scenes failed: {e}")
            complete_step("scenes", None)

        # Transcript
        update_step("transcript")
        try:
            transcript = extract_transcript(file_path)
            complete_step("transcript", transcript.model_dump() if transcript else None)
        except Exception as e:
            logger.warning(f"Transcript failed: {e}")
            complete_step("transcript", None)

        # Faces
        update_step("faces")
        try:
            faces = extract_faces(file_path)
            complete_step("faces", faces.model_dump() if faces else None)
        except Exception as e:
            logger.warning(f"Faces failed: {e}")
            complete_step("faces", None)

        # Objects
        update_step("objects")
        try:
            objects = extract_objects(file_path)
            complete_step("objects", objects.model_dump() if objects else None)
        except Exception as e:
            logger.warning(f"Objects failed: {e}")
            complete_step("objects", None)

        # CLIP
        update_step("clip")
        try:
            embeddings = extract_clip(file_path)
            complete_step("clip", {"count": len(embeddings.segments)} if embeddings else None)
        except Exception as e:
            logger.warning(f"CLIP failed: {e}")
            complete_step("clip", None)

        # OCR
        update_step("ocr")
        try:
            ocr = extract_ocr(file_path)
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
    thread = threading.Thread(target=run_extraction_job, args=(job_id, request.file))
    thread.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    """Get job status and partial results."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return jobs[job_id]


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        api_version=settings.api_version,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract metadata and features from video file.

    This endpoint runs all enabled extractors on the video file and returns
    the combined results. Extractors can be disabled using skip_* flags.
    """
    start_time = time.time()

    # Validate file exists
    file_path = Path(request.file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file}")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {request.file}")

    settings = get_settings()
    logger.info(f"Starting extraction for: {request.file}")

    # Extract metadata (always required)
    try:
        metadata = extract_metadata(request.file)
        res = metadata.resolution
        logger.info(f"Metadata extracted: {metadata.duration}s, {res.width}x{res.height}")
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {e}")

    # Detect shot type using CLIP (if not skipped)
    if not request.skip_clip:
        try:
            shot_type = detect_shot_type(request.file)
            if shot_type:
                metadata.shot_type = shot_type
                logger.info(f"Shot type detected: {shot_type.primary} ({shot_type.confidence:.2f})")
        except Exception as e:
            logger.warning(f"Shot type detection failed: {e}")

    # Extract scenes (needed by CLIP and OCR)
    scenes = None
    if not request.skip_scenes:
        try:
            scenes = extract_scenes(request.file)
            logger.info(f"Scene detection complete: {scenes.count} scenes")
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")

    # Extract transcript
    transcript = None
    if not request.skip_transcript:
        try:
            transcript = extract_transcript(
                request.file,
                model=request.whisper_model,
                language=request.language,
                fallback_language=request.fallback_language,
                language_hints=request.language_hints,
                context_hint=request.context_hint,
            )
            logger.info(f"Transcription complete: {len(transcript.segments)} segments")
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")

    # Extract faces (use scene boundaries if available for better sampling)
    faces = None
    if not request.skip_faces:
        try:
            faces = extract_faces(
                request.file,
                scenes=scenes.detections if scenes else None,
                sample_fps=request.face_sample_fps,
            )
            logger.info(f"Face detection: {faces.count} faces, ~{faces.unique_estimate} unique")
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")

    # Extract objects
    objects = None
    if not request.skip_objects:
        try:
            objects = extract_objects(
                request.file,
                sample_fps=request.object_sample_fps,
            )
            logger.info(f"Object detection complete: {len(objects.detections)} detections")
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")

    # Extract CLIP embeddings
    embeddings = None
    if not request.skip_clip:
        try:
            embeddings = extract_clip(request.file, scenes=scenes)
            logger.info(f"CLIP extraction complete: {len(embeddings.segments)} segments")
        except Exception as e:
            logger.warning(f"CLIP extraction failed: {e}")

    # Extract OCR
    ocr = None
    if not request.skip_ocr:
        try:
            ocr = extract_ocr(request.file, scenes=scenes)
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


@app.get("/extractors")
async def list_extractors():
    """List available extractors and their descriptions."""
    return {
        "extractors": [
            {
                "name": "metadata",
                "description": "Video metadata (duration, resolution, codec, device, GPS)",
                "always_enabled": True,
            },
            {
                "name": "transcript",
                "description": "Audio transcription using Whisper",
                "skip_flag": "skip_transcript",
            },
            {
                "name": "scenes",
                "description": "Scene boundary detection",
                "skip_flag": "skip_scenes",
            },
            {
                "name": "faces",
                "description": "Face detection with embeddings",
                "skip_flag": "skip_faces",
            },
            {
                "name": "objects",
                "description": "Object detection using YOLO",
                "skip_flag": "skip_objects",
            },
            {
                "name": "clip",
                "description": "CLIP visual embeddings per scene",
                "skip_flag": "skip_clip",
            },
            {
                "name": "ocr",
                "description": "Text extraction from video frames",
                "skip_flag": "skip_ocr",
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
