"""OCR extraction using EasyOCR with fast MSER pre-filtering."""

from __future__ import annotations

import gc
import logging
from typing import Any

import cv2
import numpy as np

from polybos_engine.config import DeviceType, get_device, get_settings
from polybos_engine.extractors.frame_buffer import SharedFrameBuffer
from polybos_engine.schemas import BoundingBox, OcrDetection, OcrResult

logger = logging.getLogger(__name__)

# Singleton reader instance (lazy loaded)
_ocr_reader: Any = None
_ocr_languages: list[str] | None = None

# MSER detector (reusable, no state)
_mser_detector: Any = None


def _get_mser_detector() -> Any:
    """Get or create MSER detector (singleton)."""
    global _mser_detector
    if _mser_detector is None:
        _mser_detector = cv2.MSER_create(  # type: ignore[attr-defined]
            delta=5,  # Stability threshold
            min_area=50,  # Min region size
            max_area=14400,  # Max region size (120x120)
            max_variation=0.25,
        )
    return _mser_detector


def has_text_regions(
    frame: np.ndarray,
    min_regions: int = 3,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 15.0,
) -> bool:
    """Fast detection of potential text regions using MSER.

    MSER (Maximally Stable Extremal Regions) is a classic computer vision
    algorithm that finds stable regions in images. Text characters are
    typically stable regions with specific aspect ratios.

    This is ~100x faster than deep learning OCR and can be used to skip
    frames that definitely don't contain text.

    Args:
        frame: BGR image as numpy array
        min_regions: Minimum text-like regions to consider "has text"
        min_aspect_ratio: Minimum width/height ratio for text-like regions
        max_aspect_ratio: Maximum width/height ratio for text-like regions

    Returns:
        True if frame likely contains text, False otherwise
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Detect MSER regions
    mser = _get_mser_detector()
    regions, _ = mser.detectRegions(gray)

    if len(regions) < min_regions:
        return False

    # Filter regions by text-like characteristics
    text_like_count = 0
    for region in regions:
        # Get bounding box
        _, _, w, h = cv2.boundingRect(region)  # type: ignore[call-overload]

        if h == 0:
            continue

        aspect_ratio = w / h

        # Text characters typically have aspect ratios between 0.2 and 15
        # (narrow letters like 'i' to wide text blocks)
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            # Additional filter: text regions tend to be small-medium sized
            area = w * h
            if 100 <= area <= 50000:
                text_like_count += 1

        # Early exit if we've found enough
        if text_like_count >= min_regions:
            return True

    return text_like_count >= min_regions


def unload_ocr_model() -> None:
    """Unload the EasyOCR model to free memory."""
    global _ocr_reader, _ocr_languages

    if _ocr_reader is None:
        return

    logger.info("Unloading EasyOCR model to free memory")

    try:
        import torch

        del _ocr_reader
        _ocr_reader = None
        _ocr_languages = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        gc.collect()
        logger.info("EasyOCR model unloaded")
    except Exception as e:
        logger.warning(f"Error unloading EasyOCR model: {e}")
        _ocr_reader = None
        _ocr_languages = None


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
    frame_buffer: SharedFrameBuffer,
    min_confidence: float = 0.5,
    languages: list[str] | None = None,
    skip_prefilter: bool = False,
) -> OcrResult:
    """Extract text from video frames using two-phase OCR.

    Phase 1: Fast MSER-based text detection (~5ms/frame)
    Phase 2: Deep learning OCR on frames with text (~500ms/frame)

    This typically skips 80-90% of frames, providing major speedup.

    Args:
        file_path: Path to video file (used for logging)
        frame_buffer: Pre-decoded frames from SharedFrameBuffer
        min_confidence: Minimum detection confidence
        languages: OCR languages (default from settings)
        skip_prefilter: If True, skip MSER pre-filter and run OCR on all frames

    Returns:
        OcrResult with detected text
    """
    detections: list[OcrDetection] = []
    seen_texts: set[str] = set()  # For deduplication

    # Stats for logging
    frames_checked = 0
    frames_with_text = 0
    frames_skipped = 0

    # Lazy-load OCR reader only if we find frames with text
    reader: Any = None

    def process_frame(frame: np.ndarray, timestamp: float) -> None:
        """Process a single frame for OCR."""
        nonlocal frames_checked, frames_with_text, frames_skipped, reader

        frames_checked += 1

        # Phase 1: Fast MSER pre-filter
        if not skip_prefilter:
            if not has_text_regions(frame):
                frames_skipped += 1
                return

        frames_with_text += 1

        # Phase 2: Deep learning OCR (only on frames that passed pre-filter)
        try:
            # Lazy load OCR reader on first use
            if reader is None:
                reader = _get_ocr_reader(languages)

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

    # Process frames from shared buffer
    logger.info(f"Processing {len(frame_buffer.frames)} frames for OCR")
    for ts in sorted(frame_buffer.frames.keys()):
        shared_frame = frame_buffer.frames[ts]
        process_frame(shared_frame.bgr, ts)

    # Log stats
    if frames_checked > 0:
        skip_pct = (frames_skipped / frames_checked) * 100
        logger.info(
            f"OCR: {frames_checked} frames checked, {frames_skipped} skipped ({skip_pct:.0f}%), "
            f"{frames_with_text} processed, {len(detections)} text regions found"
        )
    else:
        logger.info("OCR: no frames to process")

    return OcrResult(detections=detections)
