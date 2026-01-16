"""CLIP embedding extraction with platform-specific backends."""

import gc
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import cv2  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

from polybos_engine.config import has_cuda, is_apple_silicon
from polybos_engine.schemas import ClipResult, ClipSegment, ScenesResult

Embedding: TypeAlias = list[float]

logger = logging.getLogger(__name__)

# Map OpenCLIP-style model names to HuggingFace model names for MLX-CLIP
OPENCLIP_TO_HF_MODEL_MAP: dict[str, str] = {
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-L-14": "openai/clip-vit-large-patch14",
}


def _load_image_rgb(image_path: str) -> NDArray[np.uint8]:
    """Load image using OpenCV and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore[return-value]


class CLIPBackend(ABC):
    """Abstract base class for CLIP backends."""

    @abstractmethod
    def encode_image(self, image_path: str) -> Embedding:
        """Encode image to embedding vector.

        Args:
            image_path: Path to image file

        Returns:
            List of floats representing the embedding
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class OpenCLIPBackend(CLIPBackend):
    """OpenCLIP backend for CUDA/CPU."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip  # type: ignore[import-not-found]

        self.device = "cuda" if has_cuda() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._model_name = model_name
        logger.info(f"Loaded OpenCLIP model: {model_name} on {self.device}")

    def encode_image(self, image_path: str) -> Embedding:
        import torch
        from PIL import Image

        # Load with OpenCV and convert to PIL for preprocessing
        rgb_array = _load_image_rgb(image_path)
        image = Image.fromarray(rgb_array)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten().tolist()

    def get_model_name(self) -> str:
        return self._model_name


class MLXCLIPBackend(CLIPBackend):
    """MLX-CLIP backend for Apple Silicon."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        # Lazy import for Apple Silicon only
        self._model: Any = None
        self._processor: Any = None
        self._model_name = model_name

    def _load_model(self) -> None:
        if self._model is None:
            try:
                # Try mlx-clip first
                import mlx_clip  # type: ignore[import-not-found]

                self._model = mlx_clip.load(self._model_name)
                self._processor = None  # MLX-CLIP handles preprocessing
                logger.info(f"Loaded MLX-CLIP model: {self._model_name}")
            except ImportError:
                # Fall back to transformers + mlx
                from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-not-found]

                self._processor = CLIPProcessor.from_pretrained(self._model_name)
                self._model = CLIPModel.from_pretrained(self._model_name)
                logger.info(f"Loaded transformers CLIP model: {self._model_name}")

    def encode_image(self, image_path: str) -> Embedding:
        from PIL import Image

        self._load_model()

        # Load with OpenCV and convert to PIL for preprocessing
        rgb_array = _load_image_rgb(image_path)
        image = Image.fromarray(rgb_array)

        if self._processor is not None:
            # Using transformers
            inputs = self._processor(images=image, return_tensors="pt")
            outputs = self._model.get_image_features(**inputs)
            embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding.detach().numpy().flatten().tolist()
        else:
            # Using mlx-clip
            embedding = self._model.encode_image(image)
            return embedding.tolist()

    def get_model_name(self) -> str:
        return self._model_name


# Singleton backend instance
_backend: CLIPBackend | None = None
_backend_model_name: str | None = None


def unload_clip_model() -> None:
    """Unload the CLIP model to free memory."""
    global _backend, _backend_model_name

    if _backend is None:
        return

    logger.info("Unloading CLIP model to free memory")

    try:
        import torch

        # Clear internal model references
        if isinstance(_backend, MLXCLIPBackend):
            if _backend._model is not None:
                del _backend._model
                _backend._model = None
            if _backend._processor is not None:
                del _backend._processor
                _backend._processor = None
        elif isinstance(_backend, OpenCLIPBackend):
            if hasattr(_backend, "model"):
                del _backend.model

        del _backend
        _backend = None
        _backend_model_name = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        gc.collect()
        logger.info("CLIP model unloaded")
    except Exception as e:
        logger.warning(f"Error unloading CLIP model: {e}")
        _backend = None
        _backend_model_name = None


def get_clip_backend(model_name: str | None = None) -> CLIPBackend:
    """Get the appropriate CLIP backend for the current platform.

    Args:
        model_name: CLIP model name. For OpenCLIP: "ViT-B-16", "ViT-B-32", "ViT-L-14".
                   For MLX-CLIP: "openai/clip-vit-base-patch32", etc.
                   If None, uses defaults.
    """
    global _backend, _backend_model_name

    # If model name changed, unload the old model
    if (
        _backend is not None
        and model_name is not None
        and _backend_model_name != model_name
    ):
        logger.info(f"Switching CLIP model from {_backend_model_name} to {model_name}")
        unload_clip_model()

    if _backend is not None:
        return _backend

    if is_apple_silicon():
        try:
            # MLX-CLIP uses HuggingFace model names - translate if needed
            if model_name and model_name in OPENCLIP_TO_HF_MODEL_MAP:
                mlx_model = OPENCLIP_TO_HF_MODEL_MAP[model_name]
                logger.info(f"Translated CLIP model name: {model_name} -> {mlx_model}")
            else:
                mlx_model = model_name or "openai/clip-vit-base-patch32"
            _backend = MLXCLIPBackend(model_name=mlx_model)
            _backend_model_name = (
                model_name or mlx_model
            )  # Store original name for comparison
            logger.info(f"Using MLX-CLIP backend (Apple Silicon): {mlx_model}")
            return _backend
        except Exception as e:
            logger.warning(f"MLX-CLIP not available: {e}, falling back to OpenCLIP")

    # OpenCLIP uses short model names like "ViT-B-32"
    openclip_model = model_name or "ViT-B-32"
    _backend = OpenCLIPBackend(model_name=openclip_model)
    _backend_model_name = openclip_model
    logger.info(f"Using OpenCLIP backend: {openclip_model}")
    return _backend


def extract_clip(
    file_path: str,
    scenes: ScenesResult | None = None,
    fallback_interval: float = 10.0,
    timestamps: (
        list[float] | None
    ) = None,  # Direct timestamp list (e.g., from motion analysis)
    model_name: str | None = None,  # CLIP model name (e.g., "ViT-B-32", "ViT-L-14")
) -> ClipResult:
    """Extract CLIP embeddings from video.

    Sampling strategy (in priority order):
    1. If timestamps provided: use those directly (e.g., from motion-adaptive sampling)
    2. If scenes provided: one embedding per scene (middle of scene)
    3. Otherwise: sample at fixed interval

    Args:
        file_path: Path to video file
        scenes: Optional scene detection results (one embedding per scene)
        fallback_interval: Interval in seconds if no scenes provided
        timestamps: Optional list of specific timestamps to sample (overrides scenes/interval)
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-L-14"). If None, uses default.

    Returns:
        ClipResult with embeddings per segment
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    backend = get_clip_backend(model_name)

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="polybos_clip_")

    try:
        segments: list[ClipSegment] = []

        if timestamps is not None:
            # Use provided timestamps directly
            logger.info(
                f"Extracting CLIP embeddings at {len(timestamps)} provided timestamps"
            )

            for i, ts in enumerate(sorted(set(timestamps))):
                frame_path = _extract_frame_at(file_path, temp_dir, ts)

                if frame_path:
                    try:
                        embedding = backend.encode_image(frame_path)
                        segments.append(
                            ClipSegment(
                                start=ts,
                                end=ts,  # Single point in time
                                scene_index=i,
                                embedding=embedding,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to encode frame at {ts}s: {e}")
        elif scenes and scenes.detections:
            # Extract one frame per scene (middle of scene)
            logger.info(f"Extracting CLIP embeddings for {scenes.count} scenes")

            for scene in scenes.detections:
                mid_time = (scene.start + scene.end) / 2
                frame_path = _extract_frame_at(file_path, temp_dir, mid_time)

                if frame_path:
                    try:
                        embedding = backend.encode_image(frame_path)
                        segments.append(
                            ClipSegment(
                                start=scene.start,
                                end=scene.end,
                                scene_index=scene.index,
                                embedding=embedding,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to encode frame at {mid_time}s: {e}")
        else:
            # Fall back to fixed interval
            logger.info(f"Extracting CLIP embeddings at {fallback_interval}s intervals")
            duration = _get_video_duration(file_path)

            current_time = 0.0
            index = 0

            while current_time < duration:
                end_time = min(current_time + fallback_interval, duration)
                mid_time = (current_time + end_time) / 2

                frame_path = _extract_frame_at(file_path, temp_dir, mid_time)

                if frame_path:
                    try:
                        embedding = backend.encode_image(frame_path)
                        segments.append(
                            ClipSegment(
                                start=current_time,
                                end=end_time,
                                scene_index=index,
                                embedding=embedding,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to encode frame at {mid_time}s: {e}")

                current_time = end_time
                index += 1

        logger.info(f"Generated {len(segments)} CLIP embeddings")

        return ClipResult(
            model=backend.get_model_name(),
            segments=segments,
        )

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_frame_at(video_path: str, output_dir: str, timestamp: float) -> str | None:
    """Extract a single frame at specified timestamp."""
    output_path = os.path.join(output_dir, f"frame_{timestamp:.3f}.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        video_path,
        "-vframes",
        "1",
        "-update",
        "1",  # Required for ffmpeg 8.x single-image output
        "-q:v",
        "2",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        if os.path.exists(output_path):
            return output_path
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to extract frame at {timestamp}s: {e.stderr}")

    return None


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
