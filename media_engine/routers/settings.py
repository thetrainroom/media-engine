"""Settings endpoints."""

import logging

from fastapi import APIRouter

from media_engine.config import get_settings, reload_settings, save_config_to_file
from media_engine.schemas import SettingsResponse, SettingsUpdate

router = APIRouter(tags=["settings"])
logger = logging.getLogger(__name__)


@router.get("/settings", response_model=SettingsResponse)
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


@router.put("/settings", response_model=SettingsResponse)
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
