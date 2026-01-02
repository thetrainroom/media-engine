"""Tests for scene detection extractor."""

import pytest

from polybos_engine.extractors.scenes import extract_scenes


def test_extract_scenes(test_video_path):
    """Test scene detection."""
    scenes = extract_scenes(test_video_path)

    # Should detect at least one scene
    assert scenes.count >= 1
    assert len(scenes.detections) == scenes.count

    # First scene should start at 0
    if scenes.detections:
        assert scenes.detections[0].start == 0.0
        assert scenes.detections[0].index == 0


def test_scene_continuity(test_video_path):
    """Test that scenes are continuous (no gaps)."""
    scenes = extract_scenes(test_video_path)

    for i in range(1, len(scenes.detections)):
        prev = scenes.detections[i - 1]
        curr = scenes.detections[i]

        # Current scene should start where previous ended
        assert abs(curr.start - prev.end) < 0.1  # Allow small tolerance


def test_scenes_file_not_found():
    """Test scene detection with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_scenes("/nonexistent/video.mp4")
