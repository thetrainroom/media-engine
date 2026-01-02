"""OCR extraction using PaddleOCR."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from polybos_engine.schemas import BoundingBox, OcrDetection, OcrResult, ScenesResult

logger = logging.getLogger(__name__)


def extract_ocr(
    file_path: str,
    scenes: ScenesResult | None = None,
    min_confidence: float = 0.7,
    sample_fps: float = 0.5,
) -> OcrResult:
    """Extract text from video frames using OCR.

    Args:
        file_path: Path to video file
        scenes: Optional scene detection results (sample at scene changes)
        min_confidence: Minimum detection confidence
        sample_fps: Fallback sample rate if no scenes

    Returns:
        OcrResult with detected text
    """
    from paddleocr import PaddleOCR

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Initialize OCR (use_angle_cls for rotated text)
    logger.info("Initializing PaddleOCR")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_ocr_")

    try:
        # Determine which frames to sample
        if scenes and scenes.detections:
            # Sample at scene changes (start of each scene)
            logger.info(f"Sampling OCR at {len(scenes.detections)} scene boundaries")
            timestamps = [scene.start for scene in scenes.detections]
        else:
            # Fall back to fixed interval
            logger.info(f"Sampling OCR at {sample_fps} fps")
            duration = _get_video_duration(file_path)
            timestamps: list[float] = []
            t = 0.0
            while t < duration:
                timestamps.append(t)
                t += 1.0 / sample_fps

        detections: list[OcrDetection] = []
        seen_texts: set[str] = set()  # For deduplication

        for timestamp in timestamps:
            frame_path = _extract_frame_at(file_path, temp_dir, timestamp)

            if not frame_path:
                continue

            try:
                # Run OCR
                results = ocr.ocr(frame_path, cls=True)

                if not results or not results[0]:
                    continue

                for line in results[0]:
                    if not line:
                        continue

                    # line format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], (text, confidence)
                    bbox_points, (text, confidence) = line

                    if confidence < min_confidence:
                        continue

                    # Skip if we've seen this exact text recently
                    text_key = text.strip().lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)

                    # Convert polygon to bounding box
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)

                    detection = OcrDetection(
                        timestamp=timestamp,
                        text=text.strip(),
                        confidence=round(confidence, 3),
                        bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    )
                    detections.append(detection)

            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}s: {e}")
                continue

        logger.info(f"Detected {len(detections)} text regions")

        return OcrResult(detections=detections)

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to extract frame at {timestamp}s: {e.stderr}")

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
