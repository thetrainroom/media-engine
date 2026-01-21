"""Tests for face detection extractor."""

import pytest

from polybos_engine.extractors.faces import extract_faces


@pytest.mark.slow
def test_extract_faces(test_video_path):
    """Test face detection."""
    faces = extract_faces(test_video_path, sample_fps=0.5)

    # Count and unique_estimate should be non-negative
    assert faces.count >= 0
    assert faces.unique_estimate >= 0

    # If faces detected, check structure
    for detection in faces.detections:
        assert detection.timestamp >= 0
        assert detection.confidence >= 0
        assert detection.bbox.width > 0
        assert detection.bbox.height > 0


@pytest.mark.slow
def test_faces_with_min_size(test_video_path):
    """Test face detection with minimum size filter."""
    faces = extract_faces(test_video_path, sample_fps=0.5, min_face_size=100)

    # All detected faces should be at least 100px
    for detection in faces.detections:
        assert detection.bbox.width >= 100 or detection.bbox.height >= 100


def test_faces_file_not_found():
    """Test that decode_frames raises FileNotFoundError for non-existent files."""
    from polybos_engine.extractors.frame_buffer import decode_frames

    with pytest.raises(FileNotFoundError):
        decode_frames("/nonexistent/video.mp4", timestamps=[1.0])
