"""Tests for scene detection extractor."""

import pytest

from polybos_engine.extractors.scenes import extract_scenes


def test_extract_scenes(test_video_path):
    """Test scene detection."""
    scenes = extract_scenes(test_video_path)

    # Scene count should be non-negative (may be 0 if no scene changes)
    assert scenes.count >= 0
    assert len(scenes.detections) == scenes.count

    # If scenes detected, check structure
    if scenes.detections:
        # First scene should start at 0
        assert scenes.detections[0].start == 0.0
        assert scenes.detections[0].index == 0


def test_scene_continuity(test_video_path):
    """Test that scenes are continuous (no gaps)."""
    scenes = extract_scenes(test_video_path)

    # Skip if no scenes detected
    if len(scenes.detections) < 2:
        pytest.skip("Not enough scenes to test continuity")

    for i in range(1, len(scenes.detections)):
        prev = scenes.detections[i - 1]
        curr = scenes.detections[i]

        # Current scene should start where previous ended
        assert abs(curr.start - prev.end) < 0.1  # Allow small tolerance


def test_scenes_file_not_found():
    """Test scene detection with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_scenes("/nonexistent/video.mp4")
