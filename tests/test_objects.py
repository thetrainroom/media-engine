"""Tests for object detection extractor."""

import pytest

from polybos_engine.extractors.objects import extract_objects


@pytest.mark.slow
def test_extract_objects(test_video_path):
    """Test object detection."""
    objects = extract_objects(test_video_path, sample_fps=0.5)

    # Should have summary dict
    assert isinstance(objects.summary, dict)

    # If objects detected, check structure
    for detection in objects.detections:
        assert detection.timestamp >= 0
        assert detection.confidence >= 0
        assert detection.label is not None
        assert detection.bbox.width > 0
        assert detection.bbox.height > 0


@pytest.mark.slow
def test_objects_summary_matches_detections(test_video_path):
    """Test that summary counts match detections."""
    objects = extract_objects(test_video_path, sample_fps=0.5)

    # Count detections by label
    label_counts = {}
    for d in objects.detections:
        label_counts[d.label] = label_counts.get(d.label, 0) + 1

    # Summary should match
    assert objects.summary == label_counts


def test_objects_file_not_found():
    """Test object detection with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_objects("/nonexistent/video.mp4")
