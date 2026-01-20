"""CLIP embedding extraction with platform-specific backends."""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import cv2  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

from polybos_engine.config import has_cuda, is_apple_silicon
from polybos_engine.extractors.frame_buffer import SharedFrameBuffer
from polybos_engine.schemas import ClipResult, ClipSegment

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
    def encode_image_from_array(self, rgb_array: NDArray[np.uint8]) -> Embedding:
        """Encode image from RGB numpy array to embedding vector.

        Args:
            rgb_array: RGB image as numpy array (H, W, 3)

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
        rgb_array = _load_image_rgb(image_path)
        return self.encode_image_from_array(rgb_array)

    def encode_image_from_array(self, rgb_array: NDArray[np.uint8]) -> Embedding:
        import torch
        from PIL import Image

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
        rgb_array = _load_image_rgb(image_path)
        return self.encode_image_from_array(rgb_array)

    def encode_image_from_array(self, rgb_array: NDArray[np.uint8]) -> Embedding:
        from PIL import Image

        self._load_model()

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


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
}


def extract_clip(
    file_path: str,
    frame_buffer: SharedFrameBuffer,
    model_name: str | None = None,  # CLIP model name (e.g., "ViT-B-32", "ViT-L-14")
) -> ClipResult:
    """Extract CLIP embeddings from video frames.

    For images, use extract_clip_image() instead.

    Args:
        file_path: Path to video file (used for logging)
        frame_buffer: Pre-decoded frames from SharedFrameBuffer
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-L-14"). If None, uses default.

    Returns:
        ClipResult with embeddings per segment
    """
    backend = get_clip_backend(model_name)

    # Process frames from shared buffer
    logger.info(f"Extracting CLIP embeddings from {len(frame_buffer.frames)} frames")
    segments: list[ClipSegment] = []

    for i, ts in enumerate(sorted(frame_buffer.frames.keys())):
        shared_frame = frame_buffer.frames[ts]
        try:
            embedding = backend.encode_image_from_array(shared_frame.rgb)
            segments.append(
                ClipSegment(
                    start=ts,
                    end=ts,
                    scene_index=i,
                    embedding=embedding,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to encode frame at {ts}s: {e}")

    logger.info(f"Generated {len(segments)} CLIP embeddings")

    return ClipResult(
        model=backend.get_model_name(),
        segments=segments,
    )


def extract_clip_image(
    file_path: str,
    model_name: str | None = None,
) -> ClipResult:
    """Extract CLIP embedding from a single image file.

    Args:
        file_path: Path to image file
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-L-14"). If None, uses default.

    Returns:
        ClipResult with single embedding
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    backend = get_clip_backend(model_name)
    logger.info(f"Encoding image directly: {file_path}")

    embedding = backend.encode_image(file_path)
    return ClipResult(
        model=backend.get_model_name(),
        segments=[
            ClipSegment(
                start=0.0,
                end=0.0,
                scene_index=0,
                embedding=embedding,
            )
        ],
    )
