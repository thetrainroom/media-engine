"""Tests for OCR extractor."""

import pytest

from polybos_engine.extractors.frame_buffer import decode_frames
from polybos_engine.extractors.ocr import extract_ocr


@pytest.mark.slow
def test_extract_ocr(test_video_path):
    """Test OCR extraction."""
    # Decode frames first (new API requires frame_buffer)
    timestamps = [0.5, 1.0, 1.5, 2.0]
    frame_buffer = decode_frames(test_video_path, timestamps=timestamps)

    result = extract_ocr(test_video_path, frame_buffer=frame_buffer)

    # Detections list should exist (may be empty if no text)
    assert isinstance(result.detections, list)

    # If text detected, check structure
    for detection in result.detections:
        assert detection.timestamp >= 0
        assert detection.confidence >= 0
        assert detection.text is not None
        assert detection.bbox.width > 0
        assert detection.bbox.height > 0


def test_ocr_file_not_found():
    """Test that decode_frames raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        decode_frames("/nonexistent/video.mp4", timestamps=[1.0])
