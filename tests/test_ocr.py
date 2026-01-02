"""Tests for OCR extractor."""

import pytest

from polybos_engine.extractors.ocr import extract_ocr


@pytest.mark.slow
def test_extract_ocr(test_video_path):
    """Test OCR extraction."""
    result = extract_ocr(test_video_path, sample_fps=0.2)

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
    """Test OCR extraction with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_ocr("/nonexistent/video.mp4")
