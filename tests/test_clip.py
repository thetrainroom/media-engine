"""Tests for CLIP embedding extractor."""

import pytest

from polybos_engine.extractors.clip import extract_clip
from polybos_engine.extractors.frame_buffer import decode_frames
from polybos_engine.extractors.scenes import extract_scenes


@pytest.mark.slow
def test_extract_clip(test_video_path):
    """Test CLIP embedding extraction."""
    # Decode frames first (new API requires frame_buffer)
    timestamps = [0.5, 1.0, 1.5, 2.0]
    frame_buffer = decode_frames(test_video_path, timestamps=timestamps)

    result = extract_clip(test_video_path, frame_buffer=frame_buffer)

    assert result.model is not None
    assert len(result.segments) > 0

    # Check embedding structure
    for segment in result.segments:
        assert segment.start >= 0
        assert segment.end >= segment.start  # end can equal start for single-frame segments
        assert len(segment.embedding) > 0  # Should have embedding vector


@pytest.mark.slow
def test_clip_with_scenes(test_video_path):
    """Test CLIP extraction with scene boundaries."""
    scenes = extract_scenes(test_video_path)

    # Skip if no scenes detected (video may have no scene changes)
    if scenes.count == 0:
        pytest.skip("No scenes detected in test video")

    # Get timestamps from scene midpoints
    timestamps = [(s.start + s.end) / 2 for s in scenes.detections]
    frame_buffer = decode_frames(test_video_path, timestamps=timestamps)

    result = extract_clip(test_video_path, frame_buffer=frame_buffer)

    # Should have embeddings for each frame
    assert len(result.segments) == len(frame_buffer.frames)


def test_clip_file_not_found():
    """Test that decode_frames raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        decode_frames("/nonexistent/video.mp4", timestamps=[1.0])
