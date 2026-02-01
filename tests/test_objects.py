"""Tests for object detection extractor."""

import pytest

from media_engine.extractors.frame_buffer import decode_frames
from media_engine.extractors.objects import extract_objects


@pytest.mark.slow
def test_extract_objects(test_video_path):
    """Test object detection."""
    # Decode frames first (new API requires frame_buffer)
    timestamps = [0.5, 1.0, 1.5, 2.0]
    frame_buffer = decode_frames(test_video_path, timestamps=timestamps)

    objects = extract_objects(test_video_path, frame_buffer=frame_buffer)

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
    timestamps = [0.5, 1.0, 1.5, 2.0]
    frame_buffer = decode_frames(test_video_path, timestamps=timestamps)

    objects = extract_objects(test_video_path, frame_buffer=frame_buffer)

    # Count detections by label
    label_counts: dict[str, int] = {}
    for d in objects.detections:
        label_counts[d.label] = label_counts.get(d.label, 0) + 1

    # Summary should match
    assert objects.summary == label_counts


def test_objects_file_not_found():
    """Test that decode_frames raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        decode_frames("/nonexistent/video.mp4", timestamps=[1.0])
