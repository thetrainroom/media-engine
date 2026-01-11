"""Voice Activity Detection using Silero VAD.

Fast detection of speech presence in audio files.
Used to skip Whisper transcription for silent/ambient clips.
"""

import logging
import subprocess
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Lazy-loaded model
_vad_model: Any = None
_vad_utils: tuple[Any, ...] | None = None


class AudioContent(StrEnum):
    """Classification of audio content."""

    SPEECH = "speech"
    AMBIENT = "ambient"
    MUSIC = "music"
    SILENT = "silent"
    UNKNOWN = "unknown"


def _load_vad_model() -> tuple[Any, tuple[Any, ...]]:
    """Load Silero VAD model (cached)."""
    global _vad_model, _vad_utils

    if _vad_model is not None and _vad_utils is not None:
        return _vad_model, _vad_utils

    logger.info("Loading Silero VAD model...")

    # Load from torch hub - returns (model, utils) tuple
    # Type ignore needed as torch.hub.load has dynamic return type
    model, utils = torch.hub.load(  # type: ignore[misc]
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )

    _vad_model = model
    _vad_utils = utils

    logger.info("Silero VAD model loaded")
    return model, utils


def _extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        output_path: Path for extracted audio (WAV)
        sample_rate: Target sample rate (16000 for VAD)

    Returns:
        True if extraction succeeded
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Audio extraction timed out for {video_path}")
        return False
    except Exception as e:
        logger.warning(f"Audio extraction failed for {video_path}: {e}")
        return False


def detect_voice_activity(
    file_path: str,
    threshold: float = 0.5,
    min_speech_duration: float = 0.5,
    sample_limit_seconds: float = 120.0,
) -> dict:
    """Detect voice activity in a video/audio file.

    Args:
        file_path: Path to video or audio file
        threshold: VAD confidence threshold (0.0-1.0)
        min_speech_duration: Minimum seconds of speech to classify as "speech"
        sample_limit_seconds: Maximum seconds to analyze (for long files)

    Returns:
        Dict with:
        - audio_content: AudioContent classification
        - speech_ratio: Percentage of audio that is speech (0.0-1.0)
        - speech_segments: List of (start, end) tuples for speech
        - total_duration: Total audio duration analyzed
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return {
            "audio_content": AudioContent.UNKNOWN,
            "speech_ratio": 0.0,
            "speech_segments": [],
            "total_duration": 0.0,
        }

    # Load model - utils is (get_speech_timestamps, save_audio, read_audio, ...)
    model, utils = _load_vad_model()
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    # Extract or read audio
    with tempfile.TemporaryDirectory() as tmpdir:
        # Check if it's a video file that needs audio extraction
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mxf"}
        if path.suffix.lower() in video_extensions:
            audio_path = Path(tmpdir) / "audio.wav"
            if not _extract_audio(file_path, str(audio_path)):
                logger.warning(f"Could not extract audio from {file_path}")
                return {
                    "audio_content": AudioContent.UNKNOWN,
                    "speech_ratio": 0.0,
                    "speech_segments": [],
                    "total_duration": 0.0,
                }
            wav_path = str(audio_path)
        else:
            # Assume it's already an audio file
            wav_path = file_path

        # Read audio
        try:
            wav = read_audio(wav_path, sampling_rate=16000)
        except Exception as e:
            logger.warning(f"Could not read audio from {wav_path}: {e}")
            return {
                "audio_content": AudioContent.UNKNOWN,
                "speech_ratio": 0.0,
                "speech_segments": [],
                "total_duration": 0.0,
            }

        # Limit sample length for performance
        sample_rate = 16000
        max_samples = int(sample_limit_seconds * sample_rate)
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        total_duration = len(wav) / sample_rate

        # Detect speech timestamps
        try:
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                threshold=threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=250,  # Minimum speech segment
                min_silence_duration_ms=100,  # Minimum silence between segments
            )
        except Exception as e:
            logger.warning(f"VAD failed for {file_path}: {e}")
            return {
                "audio_content": AudioContent.UNKNOWN,
                "speech_ratio": 0.0,
                "speech_segments": [],
                "total_duration": total_duration,
            }

    # Calculate speech statistics
    speech_segments = []
    total_speech_samples = 0

    for ts in speech_timestamps:
        start_sec = ts["start"] / sample_rate
        end_sec = ts["end"] / sample_rate
        speech_segments.append((start_sec, end_sec))
        total_speech_samples += ts["end"] - ts["start"]

    speech_ratio = total_speech_samples / len(wav) if len(wav) > 0 else 0.0
    total_speech_duration = total_speech_samples / sample_rate

    # Classify audio content
    if total_speech_duration >= min_speech_duration and speech_ratio > 0.1:
        audio_content = AudioContent.SPEECH
    elif speech_ratio < 0.01:
        # Very little audio activity - could be silent or very quiet ambient
        audio_content = AudioContent.SILENT
    else:
        # Some audio but not speech - ambient or music
        # TODO: Could add music detection in the future
        audio_content = AudioContent.AMBIENT

    logger.info(
        f"VAD result for {path.name}: {audio_content} "
        f"(speech_ratio={speech_ratio:.2%}, duration={total_speech_duration:.1f}s)"
    )

    return {
        "audio_content": str(audio_content),
        "speech_ratio": round(speech_ratio, 3),
        "speech_segments": speech_segments,
        "total_duration": round(total_duration, 2),
    }


def unload_vad_model():
    """Unload VAD model to free memory."""
    global _vad_model, _vad_utils

    if _vad_model is not None:
        del _vad_model
        _vad_model = None
        _vad_utils = None
        logger.info("VAD model unloaded")
