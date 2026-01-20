"""Shot type detection using CLIP classification."""

import logging
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Final

import cv2

from polybos_engine.config import has_cuda, is_apple_silicon
from polybos_engine.extractors.frames import FrameExtractor
from polybos_engine.schemas import ShotType

logger = logging.getLogger(__name__)


class ShotTypeLabel(StrEnum):
    """Shot type classification labels."""

    AERIAL = "aerial"
    INTERVIEW = "interview"
    B_ROLL = "b-roll"
    STUDIO = "studio"
    HANDHELD = "handheld"
    STATIC = "static"
    PHONE = "phone"
    DASHCAM = "dashcam"
    SECURITY = "security"
    BROADCAST = "broadcast"
    UNKNOWN = "unknown"


# Shot type labels for CLIP classification
SHOT_TYPE_LABELS: Final[list[str]] = [
    "aerial drone footage from above",
    "interview with person talking to camera",
    "b-roll footage of scenery or environment",
    "studio footage with controlled lighting",
    "handheld camera footage",
    "tripod static shot",
    "phone footage vertical video",
    "dashcam footage from car",
    "security camera footage",
    "news broadcast footage",
]

# Map CLIP labels to simplified shot types
LABEL_TO_TYPE: Final[dict[str, ShotTypeLabel]] = {
    "aerial drone footage from above": ShotTypeLabel.AERIAL,
    "interview with person talking to camera": ShotTypeLabel.INTERVIEW,
    "b-roll footage of scenery or environment": ShotTypeLabel.B_ROLL,
    "studio footage with controlled lighting": ShotTypeLabel.STUDIO,
    "handheld camera footage": ShotTypeLabel.HANDHELD,
    "tripod static shot": ShotTypeLabel.STATIC,
    "phone footage vertical video": ShotTypeLabel.PHONE,
    "dashcam footage from car": ShotTypeLabel.DASHCAM,
    "security camera footage": ShotTypeLabel.SECURITY,
    "news broadcast footage": ShotTypeLabel.BROADCAST,
}


def detect_shot_type(file_path: str, sample_count: int = 5) -> ShotType | None:
    """Detect shot type using CLIP classification.

    Args:
        file_path: Path to video file
        sample_count: Number of frames to sample for classification

    Returns:
        ShotType with primary type and confidence, or None if detection fails
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Get video duration (0 for images)
        duration = _get_video_duration(file_path)

        # Sample frames at regular intervals (or single frame for images)
        temp_dir = tempfile.mkdtemp(prefix="polybos_shot_")

        try:
            frames: list[str] = []
            if duration == 0:
                # Image or zero-duration: use single frame at timestamp 0
                frame_path = _extract_frame_at(file_path, temp_dir, 0.0)
                if frame_path:
                    frames.append(frame_path)
            else:
                for i in range(sample_count):
                    timestamp = (i + 0.5) * duration / sample_count
                    frame_path = _extract_frame_at(file_path, temp_dir, timestamp)
                    if frame_path:
                        frames.append(frame_path)

            if not frames:
                return None

            # Classify frames using CLIP
            votes = _classify_frames(frames)

            if not votes:
                return None

            # Get most common classification
            best_label = max(votes, key=lambda k: votes.get(k, 0))
            confidence = votes[best_label] / len(frames)

            return ShotType(
                primary=LABEL_TO_TYPE.get(best_label, ShotTypeLabel.UNKNOWN),
                confidence=round(confidence, 3),
                detection_method="clip",
            )

        finally:
            # Clean up
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"Shot type detection failed: {e}")
        return None


def _classify_frames(frame_paths: list[str]) -> dict[str, int]:
    """Classify frames using CLIP and return vote counts."""
    votes: dict[str, int] = {}

    if is_apple_silicon():
        votes = _classify_with_mlx(frame_paths)
    else:
        votes = _classify_with_openclip(frame_paths)

    return votes


def _classify_with_openclip(frame_paths: list[str]) -> dict[str, int]:
    """Classify frames using OpenCLIP."""
    import open_clip  # type: ignore[import-not-found]
    import torch
    from PIL import Image

    device: str = "cuda" if has_cuda() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_tokens = tokenizer(SHOT_TYPE_LABELS).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    votes: dict[str, int] = {}

    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                similarity = (image_features @ text_features.T).softmax(dim=-1)
                best_idx = similarity.argmax().item()
                best_label = SHOT_TYPE_LABELS[best_idx]

                votes[best_label] = votes.get(best_label, 0) + 1

        except Exception as e:
            logger.warning(f"Failed to classify frame {frame_path}: {e}")

    return votes


def _classify_with_mlx(frame_paths: list[str]) -> dict[str, int]:
    """Classify frames using MLX-CLIP (Apple Silicon)."""
    # Fall back to OpenCLIP for now as MLX-CLIP API may vary
    # TODO: Implement native MLX-CLIP classification
    return _classify_with_openclip(frame_paths)


def _extract_frame_at(file_path: str, output_dir: str, timestamp: float) -> str | None:
    """Extract a single frame at specified timestamp.

    Uses FrameExtractor which handles both videos (via OpenCV/ffmpeg)
    and images (via direct loading).
    """
    output_path = os.path.join(output_dir, f"frame_{timestamp:.3f}.jpg")

    with FrameExtractor(file_path) as extractor:
        frame = extractor.get_frame_at(timestamp)

        if frame is not None:
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                logger.warning(
                    f"Frame at {timestamp}s: could not save to {output_path}"
                )
        else:
            logger.warning(f"Frame at {timestamp}s: extraction failed")

    return None


def _get_video_duration(file_path: str) -> float:
    """Get video/image duration in seconds (0 for images)."""
    from polybos_engine.extractors.frames import get_video_duration

    return get_video_duration(file_path)
