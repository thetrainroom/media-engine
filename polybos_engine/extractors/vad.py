"""Voice Activity Detection using WebRTC VAD.

Fast detection of speech presence in audio files.
Used to skip Whisper transcription for silent/ambient clips.
"""

import logging
import subprocess
import tempfile
import wave
from enum import StrEnum
from pathlib import Path

import webrtcvad  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class AudioContent(StrEnum):
    """Classification of audio content.

    Simplified categories for UI display:
    - NO_AUDIO: File has no audio track (images, some video files)
    - SPEECH: Audio with speech detected (should run Whisper)
    - AUDIO: Audio present but no speech (ambient/music/silent - skip Whisper)
    - UNKNOWN: Could not determine (extraction failed)
    """

    NO_AUDIO = "no_audio"
    SPEECH = "speech"
    AUDIO = "audio"  # Has audio but no speech (ambient, music, or silent)
    UNKNOWN = "unknown"


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
        "-i",
        video_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        "1",  # Mono
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


def _read_wav_frames(
    wav_path: str,
    frame_duration_ms: int = 30,
    max_duration_seconds: float = 120.0,
) -> tuple[list[bytes], int, float]:
    """Read WAV file and split into frames for VAD.

    Args:
        wav_path: Path to WAV file
        frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
        max_duration_seconds: Maximum audio duration to analyze

    Returns:
        Tuple of (frames, sample_rate, total_duration)
    """
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"Unsupported sample rate: {sample_rate}")
        if n_channels != 1:
            raise ValueError(f"Expected mono audio, got {n_channels} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit audio, got {sample_width * 8}-bit")

        # Calculate frame size
        frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width
        max_frames = int(max_duration_seconds * 1000 / frame_duration_ms)

        frames = []
        total_samples = 0

        while len(frames) < max_frames:
            frame = wf.readframes(int(sample_rate * frame_duration_ms / 1000))
            if len(frame) < frame_size:
                break
            frames.append(frame)
            total_samples += int(sample_rate * frame_duration_ms / 1000)

        total_duration = total_samples / sample_rate
        return frames, sample_rate, total_duration


def detect_voice_activity(
    file_path: str,
    aggressiveness: int = 2,
    min_speech_duration: float = 0.5,
    sample_limit_seconds: float = 120.0,
) -> dict:
    """Detect voice activity in a video/audio file using WebRTC VAD.

    Args:
        file_path: Path to video or audio file
        aggressiveness: VAD aggressiveness (0-3, higher = less sensitive to speech)
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
            "audio_content": str(AudioContent.UNKNOWN),
            "speech_ratio": 0.0,
            "speech_segments": [],
            "total_duration": 0.0,
        }

    # Create VAD instance
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration_ms = 30  # 30ms frames

    with tempfile.TemporaryDirectory() as tmpdir:
        # Check if it's a video file that needs audio extraction
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mxf"}
        audio_extensions = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}

        if path.suffix.lower() in video_extensions:
            audio_path = Path(tmpdir) / "audio.wav"
            if not _extract_audio(file_path, str(audio_path)):
                logger.warning(f"Could not extract audio from {file_path}")
                return {
                    "audio_content": str(AudioContent.UNKNOWN),
                    "speech_ratio": 0.0,
                    "speech_segments": [],
                    "total_duration": 0.0,
                }
            wav_path = str(audio_path)
        elif path.suffix.lower() in audio_extensions:
            # Convert to WAV format for webrtcvad
            audio_path = Path(tmpdir) / "audio.wav"
            if not _extract_audio(file_path, str(audio_path)):
                logger.warning(f"Could not convert audio from {file_path}")
                return {
                    "audio_content": str(AudioContent.UNKNOWN),
                    "speech_ratio": 0.0,
                    "speech_segments": [],
                    "total_duration": 0.0,
                }
            wav_path = str(audio_path)
        else:
            # Assume it's already a WAV file
            wav_path = file_path

        # Read audio frames
        try:
            frames, sample_rate, total_duration = _read_wav_frames(
                wav_path,
                frame_duration_ms=frame_duration_ms,
                max_duration_seconds=sample_limit_seconds,
            )
        except Exception as e:
            logger.warning(f"Could not read audio from {wav_path}: {e}")
            return {
                "audio_content": str(AudioContent.UNKNOWN),
                "speech_ratio": 0.0,
                "speech_segments": [],
                "total_duration": 0.0,
            }

        if not frames:
            logger.warning(f"No audio frames extracted from {file_path}")
            return {
                "audio_content": str(
                    AudioContent.AUDIO
                ),  # Has audio track but empty/silent
                "speech_ratio": 0.0,
                "speech_segments": [],
                "total_duration": 0.0,
            }

        # Analyze each frame
        speech_frames = []
        for frame in frames:
            try:
                is_speech = vad.is_speech(frame, sample_rate)
                speech_frames.append(is_speech)
            except Exception:
                speech_frames.append(False)

    # Calculate speech statistics
    speech_count = sum(speech_frames)
    total_frames = len(speech_frames)
    speech_ratio = speech_count / total_frames if total_frames > 0 else 0.0
    total_speech_duration = speech_count * frame_duration_ms / 1000

    # Build speech segments (consecutive speech frames)
    speech_segments = []
    segment_start = None

    for i, is_speech in enumerate(speech_frames):
        time_sec = i * frame_duration_ms / 1000
        if is_speech and segment_start is None:
            segment_start = time_sec
        elif not is_speech and segment_start is not None:
            speech_segments.append((segment_start, time_sec))
            segment_start = None

    # Close final segment if needed
    if segment_start is not None:
        speech_segments.append((segment_start, total_duration))

    # Classify audio content
    if total_speech_duration >= min_speech_duration and speech_ratio > 0.1:
        audio_content = AudioContent.SPEECH
    else:
        # Audio present but no speech detected (silent, ambient, or music)
        audio_content = AudioContent.AUDIO

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
    """No-op for WebRTC VAD (no model to unload)."""
    pass
