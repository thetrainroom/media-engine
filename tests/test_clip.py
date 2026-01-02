"""Tests for CLIP embedding extractor."""

import pytest

from polybos_engine.extractors.clip import extract_clip
from polybos_engine.extractors.scenes import extract_scenes


@pytest.mark.slow
def test_extract_clip(test_video_path):
    """Test CLIP embedding extraction."""
    result = extract_clip(test_video_path, fallback_interval=5.0)

    assert result.model is not None
    assert len(result.segments) > 0

    # Check embedding structure
    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end > segment.start
        assert len(segment.embedding) > 0  # Should have embedding vector


@pytest.mark.slow
def test_clip_with_scenes(test_video_path):
    """Test CLIP extraction with scene boundaries."""
    scenes = extract_scenes(test_video_path)
    result = extract_clip(test_video_path, scenes=scenes)

    # Should have same number of segments as scenes
    assert len(result.segments) == scenes.count

    # Scene indices should match
    for i, segment in enumerate(result.segments):
        assert segment.scene_index == i


def test_clip_file_not_found():
    """Test CLIP extraction with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_clip("/nonexistent/video.mp4")
