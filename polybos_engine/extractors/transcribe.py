"""Audio transcription using Whisper with platform-specific backends."""

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polybos_engine.config import has_cuda, is_apple_silicon
from polybos_engine.schemas import Transcript, TranscriptHints, TranscriptSegment


@dataclass
class TranscriptionSegment:
    """A single segment from transcription."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result from a transcription backend."""

    language: str
    language_probability: float
    segments: list[TranscriptionSegment] = field(default_factory=list)

logger = logging.getLogger(__name__)


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        model: str = "large-v3",
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            model: Whisper model name
            language: Force language (None for auto-detect)
            initial_prompt: Context prompt for better accuracy

        Returns:
            TranscriptionResult with language, segments, and confidence
        """
        pass


class WhisperMLX(TranscriptionBackend):
    """Apple Silicon backend using mlx-whisper."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def _load_model(self, model: str) -> None:
        if self._model is None or self._model_name != model:
            import mlx_whisper  # type: ignore[import-not-found]

            self._model = mlx_whisper
            self._model_name = model
            logger.info(f"Loaded mlx-whisper model: {model}")

    def transcribe(
        self,
        audio_path: str,
        model: str = "large-v3",
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        self._load_model(model)

        result: dict[str, Any] = self._model.transcribe(
            audio_path,
            path_or_hf_repo=f"mlx-community/whisper-{model}-mlx",
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,
        )

        return TranscriptionResult(
            language=result.get("language", "unknown"),
            language_probability=result.get("language_probability", 0.0),
            segments=[
                TranscriptionSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                for s in result.get("segments", [])
            ],
        )


class WhisperCUDA(TranscriptionBackend):
    """NVIDIA GPU backend using faster-whisper."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def _load_model(self, model: str) -> None:
        if self._model is None or self._model_name != model:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            self._model = WhisperModel(model, device="cuda", compute_type="float16")
            self._model_name = model
            logger.info(f"Loaded faster-whisper model: {model}")

    def transcribe(
        self,
        audio_path: str,
        model: str = "large-v3",
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        self._load_model(model)

        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,
        )

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=[
                TranscriptionSegment(start=s.start, end=s.end, text=s.text.strip())
                for s in segments
            ],
        )


class WhisperCPU(TranscriptionBackend):
    """CPU fallback using openai-whisper."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str | None = None

    def _load_model(self, model: str) -> None:
        # Use smaller model for CPU
        actual_model = "medium" if model == "large-v3" else model

        if self._model is None or self._model_name != actual_model:
            import whisper  # type: ignore[import-not-found]

            self._model = whisper.load_model(actual_model)
            self._model_name = actual_model
            logger.info(f"Loaded openai-whisper model: {actual_model}")

    def transcribe(
        self,
        audio_path: str,
        model: str = "large-v3",
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        self._load_model(model)

        result: dict[str, Any] = self._model.transcribe(
            audio_path,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,
        )

        return TranscriptionResult(
            language=result.get("language", "unknown"),
            language_probability=0.0,  # Not provided by openai-whisper
            segments=[
                TranscriptionSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                for s in result.get("segments", [])
            ],
        )


# Singleton backend instance
_backend: TranscriptionBackend | None = None


def get_transcription_backend() -> TranscriptionBackend:
    """Get the appropriate transcription backend for the current platform."""
    global _backend

    if _backend is not None:
        return _backend

    if is_apple_silicon():
        try:
            import mlx_whisper  # type: ignore[import-not-found]  # noqa: F401

            _backend = WhisperMLX()
            logger.info("Using mlx-whisper backend (Apple Silicon)")
            return _backend
        except ImportError:
            logger.warning("mlx-whisper not available, falling back")

    if has_cuda():
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]  # noqa: F401

            _backend = WhisperCUDA()
            logger.info("Using faster-whisper backend (CUDA)")
            return _backend
        except ImportError:
            logger.warning("faster-whisper not available, falling back")

    try:
        import whisper  # type: ignore[import-not-found]  # noqa: F401

        _backend = WhisperCPU()
        logger.info("Using openai-whisper backend (CPU)")
        return _backend
    except ImportError:
        raise RuntimeError(
            "No Whisper backend available. Install one of: "
            "mlx-whisper, faster-whisper, openai-whisper"
        )


def extract_audio(video_path: str, output_path: str | None = None) -> str:
    """Extract audio from video file as 16kHz mono WAV.

    Args:
        video_path: Path to video file
        output_path: Output path for audio file (optional)

    Returns:
        Path to extracted audio file
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e.stderr}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr}")


def extract_transcript(
    file_path: str,
    model: str = "large-v3",
    language: str | None = None,
    fallback_language: str = "en",
    language_hints: list[str] | None = None,
    context_hint: str | None = None,
) -> Transcript:
    """Extract transcript from video file.

    Args:
        file_path: Path to video file
        model: Whisper model name
        language: Force language (skip detection)
        fallback_language: Fallback for short clips with low confidence
        language_hints: Language hints (not directly used by Whisper, but logged)
        context_hint: Context hint used as initial_prompt

    Returns:
        Transcript object with segments and metadata
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Extract audio
    logger.info(f"Extracting audio from {file_path}")
    audio_path = extract_audio(file_path)

    try:
        # Get audio duration
        audio_duration = _get_audio_duration(audio_path)

        # Get backend
        backend = get_transcription_backend()

        # Determine language
        use_language = language
        fallback_applied = False

        if use_language is None:
            # First pass: detect language
            logger.info("Detecting language...")
            detect_result = backend.transcribe(
                audio_path, model=model, language=None, initial_prompt=None
            )

            detected_lang = detect_result.language
            confidence = detect_result.language_probability

            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")

            # Apply fallback for short clips with low confidence
            if confidence < 0.7 and audio_duration < 15:
                use_language = fallback_language
                fallback_applied = True
                logger.info(f"Low confidence on short clip, using fallback: {use_language}")
            else:
                use_language = detected_lang

        # Main transcription
        logger.info(f"Transcribing with language={use_language}, model={model}")
        result = backend.transcribe(
            audio_path,
            model=model,
            language=use_language,
            initial_prompt=context_hint,
        )

        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=s.text)
            for s in result.segments
        ]

        return Transcript(
            language=result.language,
            confidence=result.language_probability,
            duration=audio_duration,
            hints_used=TranscriptHints(
                language_hints=language_hints or [],
                context_hint=context_hint,
                fallback_applied=fallback_applied,
            ),
            segments=segments,
        )

    finally:
        # Clean up temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
