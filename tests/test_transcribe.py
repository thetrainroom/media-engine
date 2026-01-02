"""Tests for transcription extractor."""

import pytest

from polybos_engine.extractors.transcribe import extract_audio, extract_transcript


def test_extract_audio(test_video_path, tmp_path):
    """Test audio extraction."""
    output_path = str(tmp_path / "audio.wav")
    result = extract_audio(test_video_path, output_path)

    assert result == output_path
    assert (tmp_path / "audio.wav").exists()


@pytest.mark.slow
def test_extract_transcript(test_video_path):
    """Test transcript extraction."""
    transcript = extract_transcript(test_video_path)

    assert transcript.duration > 0
    assert transcript.language is not None
    # Segments may be empty if no speech


@pytest.mark.slow
def test_extract_transcript_with_language(test_video_path):
    """Test transcript extraction with forced language."""
    transcript = extract_transcript(test_video_path, language="en")

    assert transcript.language == "en"


def test_transcript_file_not_found():
    """Test transcript extraction with non-existent file."""
    with pytest.raises(FileNotFoundError):
        extract_transcript("/nonexistent/video.mp4")
