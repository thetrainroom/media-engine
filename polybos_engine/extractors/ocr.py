"""OCR extraction using EasyOCR."""

import logging
from typing import Any

from polybos_engine.config import get_device, get_settings, DeviceType
from polybos_engine.extractors.frames import FrameExtractor, get_video_duration
from polybos_engine.schemas import BoundingBox, OcrDetection, OcrResult, ScenesResult

logger = logging.getLogger(__name__)

# Singleton reader instance (lazy loaded)
_ocr_reader: Any = None
_ocr_languages: list[str] | None = None


def _get_ocr_reader(languages: list[str] | None = None) -> Any:
    """Get or create the EasyOCR reader (singleton).

    Note: Once initialized, the reader keeps its languages.
    To change languages, restart the server.
    """
    global _ocr_reader, _ocr_languages

    if _ocr_reader is not None:
        return _ocr_reader

    import easyocr  # type: ignore[import-not-found]

    if languages is None:
        # Get from settings
        settings = get_settings()
        languages = settings.ocr_languages

    _ocr_languages = languages

    # Enable GPU if available (CUDA only - EasyOCR doesn't support MPS)
    device = get_device()
    use_gpu = device == DeviceType.CUDA
    device_name = "CUDA GPU" if use_gpu else "CPU"

    logger.info(f"Initializing EasyOCR with languages: {languages} on {device_name}")
    _ocr_reader = easyocr.Reader(languages, gpu=use_gpu)

    return _ocr_reader


def extract_ocr(
    file_path: str,
    scenes: ScenesResult | None = None,
    min_confidence: float = 0.5,
    sample_fps: float = 0.5,
    timestamps: list[float] | None = None,  # Direct timestamp list (e.g., from motion analysis)
    languages: list[str] | None = None,
) -> OcrResult:
    """Extract text from video frames using OCR.

    Sampling strategy (in priority order):
    1. If timestamps provided: use those directly (e.g., from motion-adaptive sampling)
    2. If scenes provided: sample at scene boundaries
    3. Otherwise: sample at fixed fps

    Args:
        file_path: Path to video file
        scenes: Optional scene detection results (sample at scene changes)
        min_confidence: Minimum detection confidence
        sample_fps: Fallback sample rate if no scenes
        timestamps: Optional list of specific timestamps to sample (overrides scenes/fps)
        languages: OCR languages (default: ["en", "no"])

    Returns:
        OcrResult with detected text
    """
    # Get OCR reader
    reader = _get_ocr_reader(languages)

    # Determine which frames to sample (priority: explicit > scenes > fixed fps)
    sample_timestamps: list[float]
    if timestamps is not None:
        sample_timestamps = sorted(set(timestamps))
        logger.info(f"Using {len(sample_timestamps)} provided timestamps for OCR")
    elif scenes and scenes.detections:
        # Sample at scene changes (start of each scene)
        logger.info(f"Sampling OCR at {len(scenes.detections)} scene boundaries")
        sample_timestamps = [scene.start for scene in scenes.detections]
    else:
        # Fall back to fixed interval
        logger.info(f"Sampling OCR at {sample_fps} fps")
        duration = get_video_duration(file_path)
        sample_timestamps = []
        t = 0.0
        while t < duration:
            sample_timestamps.append(t)
            t += 1.0 / sample_fps

    detections: list[OcrDetection] = []
    seen_texts: set[str] = set()  # For deduplication

    # Use OpenCV for fast frame extraction
    with FrameExtractor(file_path) as extractor:
        for timestamp in sample_timestamps:
            frame = extractor.get_frame_at(timestamp)
            if frame is None:
                continue

            try:
                # Run OCR directly on numpy array (no need to save to disk)
                results = reader.readtext(frame)

                for bbox_points, text, confidence in results:
                    if confidence < min_confidence:
                        continue

                    # Skip if we've seen this exact text recently
                    text_key = text.strip().lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)

                    # Convert polygon to bounding box
                    # bbox_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)

                    detection = OcrDetection(
                        timestamp=timestamp,
                        text=text.strip(),
                        confidence=round(float(confidence), 3),
                        bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    )
                    detections.append(detection)

            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}s: {e}")
                continue

    logger.info(f"Detected {len(detections)} text regions")

    return OcrResult(detections=detections)
