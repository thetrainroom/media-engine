"""Tests for metadata extractor."""

import pytest

from media_engine.extractors.metadata import extract_metadata


def test_extract_metadata(test_video_path):
    """Test metadata extraction."""
    metadata = extract_metadata(test_video_path)

    assert metadata.duration > 0
    assert metadata.resolution.width > 0
    assert metadata.resolution.height > 0
    assert metadata.file_size > 0


def test_extract_metadata_file_not_found():
    """Test metadata extraction with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_metadata("/nonexistent/video.mp4")


def test_metadata_codec(test_video_path):
    """Test codec extraction."""
    metadata = extract_metadata(test_video_path)

    # Most videos have at least video codec
    assert metadata.codec.video is not None


def test_metadata_fps(test_video_path):
    """Test FPS extraction."""
    metadata = extract_metadata(test_video_path)

    # FPS should be reasonable (1-120)
    if metadata.fps is not None:
        assert 1 <= metadata.fps <= 120
