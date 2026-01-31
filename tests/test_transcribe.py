"""Tests for transcription extractor."""

import os

import pytest

from polybos_engine.extractors.transcribe import extract_audio, extract_transcript

# Use a video with speech for transcription tests
SPEECH_VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "test_data", "video", "sample_with_speech.MP4"
)


@pytest.fixture
def video_with_audio():
    """Get path to video file with audio track."""
    if os.path.exists(SPEECH_VIDEO):
        return SPEECH_VIDEO
    # Fallback to TEST_VIDEO_PATH if it has audio
    return os.environ.get("TEST_VIDEO_PATH", SPEECH_VIDEO)


def test_extract_audio(video_with_audio, tmp_path):
    """Test audio extraction."""
    if not os.path.exists(video_with_audio):
        pytest.skip(f"Test video not found: {video_with_audio}")

    output_path = str(tmp_path / "audio.wav")
    try:
        result = extract_audio(video_with_audio, output_path)
        assert result == output_path
        assert (tmp_path / "audio.wav").exists()
    except RuntimeError as e:
        if "no audio" in str(e).lower() or "does not contain any stream" in str(e):
            pytest.skip("Test video has no audio track")
        raise


@pytest.mark.slow
def test_extract_transcript(video_with_audio):
    """Test transcript extraction."""
    if not os.path.exists(video_with_audio):
        pytest.skip(f"Test video not found: {video_with_audio}")

    try:
        transcript = extract_transcript(video_with_audio)
        assert transcript.duration > 0
        assert transcript.language is not None
        # Segments may be empty if no speech
    except RuntimeError as e:
        if "no audio" in str(e).lower() or "does not contain any stream" in str(e):
            pytest.skip("Test video has no audio track")
        raise


@pytest.mark.slow
def test_extract_transcript_with_language(video_with_audio):
    """Test transcript extraction with forced language."""
    if not os.path.exists(video_with_audio):
        pytest.skip(f"Test video not found: {video_with_audio}")

    try:
        transcript = extract_transcript(video_with_audio, language="en")
        assert transcript.language == "en"
    except RuntimeError as e:
        if "no audio" in str(e).lower() or "does not contain any stream" in str(e):
            pytest.skip("Test video has no audio track")
        raise


def test_transcript_file_not_found():
    """Test transcript extraction with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_transcript("/nonexistent/video.mp4")
