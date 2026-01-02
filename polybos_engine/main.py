"""FastAPI application for Polybos Media Engine."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from polybos_engine import __version__
from polybos_engine.config import get_settings
from polybos_engine.extractors import (
    extract_clip,
    extract_faces,
    extract_metadata,
    extract_objects,
    extract_ocr,
    extract_scenes,
    extract_transcript,
)
from polybos_engine.extractors.shot_type import detect_shot_type
from polybos_engine.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
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
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
