"""Tests for CLIP embedding extractor."""

import pytest

from media_engine.extractors.clip import SIGLIP2_MODELS, _scene_index_for, extract_clip, is_siglip_model
from media_engine.extractors.frame_buffer import decode_frames
from media_engine.extractors.frames import get_video_duration
from media_engine.extractors.scenes import extract_scenes


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


# --- Fixed-fps sampling mode (api_version 1.1) ---


def test_extract_clip_requires_buffer_in_default_mode():
    """Default mode without a frame buffer is a usage error."""
    with pytest.raises(ValueError):
        extract_clip("/some/video.mp4")


def test_scene_index_for():
    """Timestamps map to their containing scene; outside = None."""
    boundaries = [(0.0, 10.0), (10.0, 25.0), (25.0, 30.0)]

    assert _scene_index_for(0.5, boundaries) == 0
    assert _scene_index_for(10.0, boundaries) == 1  # boundary belongs to next scene
    assert _scene_index_for(24.9, boundaries) == 1
    assert _scene_index_for(30.5, boundaries) == 2  # past the end clamps to last scene
    assert _scene_index_for(5.0, None) is None
    assert _scene_index_for(5.0, []) is None


@pytest.mark.slow
def test_extract_clip_default_mode_fields(test_video_path):
    """Default mode reports per_scene sample_mode and per-segment timestamps."""
    frame_buffer = decode_frames(test_video_path, timestamps=[0.5, 1.5])
    result = extract_clip(test_video_path, frame_buffer=frame_buffer)

    assert result.sample_mode == "per_scene"
    assert result.sample_fps is None
    for segment in result.segments:
        assert segment.timestamp == segment.start


@pytest.mark.slow
def test_extract_clip_fixed_fps(test_video_path):
    """Fixed 1.0 fps produces ~duration samples at window centers."""
    duration = get_video_duration(test_video_path)
    result = extract_clip(test_video_path, sample_fps=1.0)

    assert result.sample_mode == "fixed_fps"
    assert result.sample_fps == 1.0

    expected = int(duration * 1.0)
    # ffmpeg's fps filter can produce one frame fewer near the tail
    assert expected - 1 <= len(result.segments) <= expected + 1

    for k, segment in enumerate(result.segments):
        assert segment.timestamp == pytest.approx((k + 0.5) / 1.0)
        assert segment.start == pytest.approx(max(0.0, segment.timestamp - 0.5))
        assert segment.end <= duration + 1e-6
        assert segment.scene_index is None  # no boundaries provided
        assert len(segment.embedding) > 0


@pytest.mark.slow
def test_extract_clip_fixed_fps_half_rate(test_video_path):
    """0.5 fps produces half as many samples as 1.0 fps."""
    duration = get_video_duration(test_video_path)
    result = extract_clip(test_video_path, sample_fps=0.5)

    expected = int(duration * 0.5)
    assert expected - 1 <= len(result.segments) <= expected + 1
    if result.segments:
        assert result.segments[0].timestamp == pytest.approx(1.0)  # (0 + 0.5) / 0.5


@pytest.mark.slow
def test_extract_clip_fixed_fps_scene_index(test_video_path):
    """Samples map to their containing scene when boundaries are provided."""
    duration = get_video_duration(test_video_path)
    boundaries = [(0.0, duration / 2), (duration / 2, duration)]

    result = extract_clip(test_video_path, sample_fps=1.0, scene_boundaries=boundaries)

    for segment in result.segments:
        assert segment.timestamp is not None
        if segment.timestamp < duration / 2:
            assert segment.scene_index == 0
        else:
            assert segment.scene_index == 1


# --- SigLIP 2 backend ---


def test_siglip_model_name_detection():
    """SigLIP names route to the transformers backend; CLIP names do not."""
    assert is_siglip_model("SigLIP2-B-16")
    assert is_siglip_model("SigLIP2-SO400M")
    assert is_siglip_model("google/siglip2-so400m-patch16-384")  # raw HF id
    assert not is_siglip_model("ViT-B-32")
    assert not is_siglip_model("ViT-L-14")
    assert not is_siglip_model(None)

    # Short names resolve to google/ HF ids
    for short, hf in SIGLIP2_MODELS.items():
        assert hf.startswith("google/siglip2-"), (short, hf)


@pytest.mark.slow
def test_siglip2_encode_image_and_text(test_video_path):
    """SigLIP2 produces matching-dim, normalized image and text embeddings."""
    import numpy as np

    from media_engine.extractors.clip import encode_text_query, get_clip_backend, unload_clip_model

    backend = get_clip_backend("SigLIP2-B-16")
    assert backend.get_model_name() == "SigLIP2-B-16"

    frame_buffer = decode_frames(test_video_path, timestamps=[1.0])
    frame = next(iter(frame_buffer.frames.values()))
    img = np.array(backend.encode_image_from_array(frame.rgb), dtype=np.float64)
    txt = np.array(encode_text_query("a photo", "SigLIP2-B-16"), dtype=np.float64)

    assert len(img) == len(txt) == 768  # B-16 dim
    assert np.linalg.norm(img) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(txt) == pytest.approx(1.0, abs=1e-3)
    unload_clip_model()
