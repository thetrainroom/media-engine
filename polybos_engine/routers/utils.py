"""Utility endpoints."""

import logging
import os
import signal
import threading
import time

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["utils"])
logger = logging.getLogger(__name__)


@router.post("/shutdown")
async def shutdown_engine():
    """Gracefully shutdown the engine.

    Call this before killing the process to ensure clean resource cleanup.
    """
    from polybos_engine.app import cleanup_resources

    logger.info("Shutdown requested via API")
    cleanup_resources()

    # Schedule process exit after response is sent
    def delayed_exit():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=delayed_exit, daemon=True)
    thread.start()

    return {"status": "shutting_down"}


@router.get("/extractors")
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


@router.post("/encode_text")
async def encode_text(request: dict):
    """Encode a text query to a CLIP embedding for text-to-image search.

    Request body:
        text: str - The text query to encode
        model_name: str (optional) - CLIP model name (e.g., "ViT-B-32")
        translate: bool (optional) - Whether to translate non-English queries to English (default: true)

    Returns:
        embedding: list[float] - The normalized CLIP embedding (512 or 768 dimensions)
        model: str - The model used for encoding
        original_text: str - The original query text
        translated_text: str - The text that was actually encoded (may be translated)
        detected_language: str | None - Detected language of the original text
        was_translated: bool - Whether the text was translated
    """
    from polybos_engine.extractors.clip import encode_text_query, get_clip_backend
    from polybos_engine.extractors.translate import translate_query_for_clip

    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text query is required")

    model_name = request.get("model_name")
    enable_translation = request.get("translate", True)

    try:
        # Translate query if needed
        translated_text, detected_lang, was_translated = translate_query_for_clip(
            text, enable_translation=enable_translation
        )

        # Encode the (possibly translated) text
        embedding = encode_text_query(translated_text, model_name)
        backend = get_clip_backend(model_name)

        return {
            "embedding": embedding,
            "model": backend.get_model_name(),
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "was_translated": was_translated,
        }
    except Exception as e:
        logger.error(f"Text encoding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
