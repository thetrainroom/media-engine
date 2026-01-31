"""Model checking endpoints."""

import logging
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException

from polybos_engine.config import get_free_memory_gb
from polybos_engine.extractors import (
    unload_clip_model,
    unload_face_model,
    unload_qwen_model,
    unload_whisper_model,
    unload_yolo_model,
)

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)

# Store for model check results
_model_check_results: dict[str, dict] = {}
_model_check_status: dict[str, str] = {}  # "running", "complete", "error"


def _run_model_checks(check_id: str) -> None:
    """Background task to check which models can load."""
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


@router.post("/check-models")
async def start_model_check():
    """Start checking which models can actually load.

    Returns immediately with a check_id. Poll GET /check-models/{check_id} for results.
    Takes 30-60 seconds to complete.
    """
    check_id = str(uuid.uuid4())[:8]

    # Start background thread
    thread = threading.Thread(target=_run_model_checks, args=(check_id,), daemon=True)
    thread.start()

    return {"check_id": check_id, "status": "running"}


@router.get("/check-models/{check_id}")
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
