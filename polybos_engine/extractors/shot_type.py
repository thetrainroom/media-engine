"""Shot type detection using CLIP classification."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from polybos_engine.config import has_cuda, is_apple_silicon
from polybos_engine.schemas import ShotType

logger = logging.getLogger(__name__)

# Shot type labels for CLIP classification
SHOT_TYPE_LABELS = [
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
LABEL_TO_TYPE = {
    "aerial drone footage from above": "aerial",
    "interview with person talking to camera": "interview",
    "b-roll footage of scenery or environment": "b-roll",
    "studio footage with controlled lighting": "studio",
    "handheld camera footage": "handheld",
    "tripod static shot": "static",
    "phone footage vertical video": "phone",
    "dashcam footage from car": "dashcam",
    "security camera footage": "security",
    "news broadcast footage": "broadcast",
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
        raise FileNotFoundError(f"Video file not found: {file_path}")

    try:
        # Get video duration
        duration = _get_video_duration(file_path)
        if duration <= 0:
            return None

        # Sample frames at regular intervals
        temp_dir = tempfile.mkdtemp(prefix="polybos_shot_")

        try:
            frames = []
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
            best_label = max(votes, key=votes.get)
            confidence = votes[best_label] / len(frames)

            return ShotType(
                primary=LABEL_TO_TYPE.get(best_label, "unknown"),
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
    votes = {}

    if is_apple_silicon():
        votes = _classify_with_mlx(frame_paths)
    else:
        votes = _classify_with_openclip(frame_paths)

    return votes


def _classify_with_openclip(frame_paths: list[str]) -> dict[str, int]:
    """Classify frames using OpenCLIP."""
    import open_clip
    import torch
    from PIL import Image

    device = "cuda" if has_cuda() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_tokens = tokenizer(SHOT_TYPE_LABELS).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    votes = {}

    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

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


def _extract_frame_at(video_path: str, output_dir: str, timestamp: float) -> str | None:
    """Extract a single frame at specified timestamp."""
    output_path = os.path.join(output_dir, f"frame_{timestamp:.3f}.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        video_path,
        "-vframes",
        "1",
        "-q:v",
        "2",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(output_path):
            return output_path
    except subprocess.CalledProcessError:
        pass

    return None


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
