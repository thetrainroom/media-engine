"""FastAPI app factory for Media Engine."""

import atexit
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from media_engine import __version__
from media_engine.batch.timing import save_timing_history
from media_engine.extractors import (
    shutdown_ffprobe_pool,
    unload_clip_model,
    unload_face_model,
    unload_ocr_model,
    unload_qwen_model,
    unload_vad_model,
    unload_whisper_model,
    unload_yolo_model,
)
from media_engine.routers import (
    batch_router,
    health_router,
    models_router,
    settings_router,
    utils_router,
)

logger = logging.getLogger(__name__)


def cleanup_resources() -> None:
    """Clean up all resources.

    Note: This runs during Python shutdown via atexit, so we must be careful
    not to import new modules or use logging (file handlers may be closed).
    """
    try:
        # Save timing history before shutdown
        save_timing_history()
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Media Engine",
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

    # Include routers
    app.include_router(batch_router)
    app.include_router(health_router)
    app.include_router(settings_router)
    app.include_router(models_router)
    app.include_router(utils_router)

    # Register cleanup on exit
    atexit.register(cleanup_resources)

    return app
