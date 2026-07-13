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

from media_engine.config import get_device, has_cuda, is_apple_silicon
from media_engine.extractors.frame_buffer import SharedFrameBuffer
from media_engine.schemas import ClipResult, ClipSegment

Embedding: TypeAlias = list[float]

logger = logging.getLogger(__name__)

# Map OpenCLIP-style model names to HuggingFace model names for MLX-CLIP
OPENCLIP_TO_HF_MODEL_MAP: dict[str, str] = {
    "ViT-B-16": "openai/clip-vit-base-patch16",
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-L-14": "openai/clip-vit-large-patch14",
}

# SigLIP 2 short names -> HuggingFace model IDs (transformers backend).
# SigLIP 2 (Feb 2025): sigmoid-loss image-text encoders with a multilingual
# text tower (109 languages) - stronger retrieval than the 2021 CLIP models.
# NOTE: embeddings live in a different space than CLIP - not comparable;
# ClipResult.model identifies which model produced an embedding.
SIGLIP2_MODELS: dict[str, str] = {
    "SigLIP2-B-16": "google/siglip2-base-patch16-224",  # 768-dim
    "SigLIP2-SO400M": "google/siglip2-so400m-patch16-384",  # 1152-dim
}


def is_siglip_model(model_name: str | None) -> bool:
    """Whether a model name refers to a SigLIP-family model."""
    if not model_name:
        return False
    return model_name in SIGLIP2_MODELS or "siglip" in model_name.lower()


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

    @abstractmethod
    def encode_text(self, text: str) -> Embedding:
        """Encode text to embedding vector.

        Args:
            text: Text string to encode

        Returns:
            List of floats representing the embedding (normalized)
        """
        pass


class OpenCLIPBackend(CLIPBackend):
    """OpenCLIP backend for CUDA/CPU."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip  # type: ignore[import-not-found]

        self.device = "cuda" if has_cuda() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
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

    def encode_text(self, text: str) -> Embedding:
        import open_clip  # type: ignore[import-not-found]
        import torch

        # Tokenize the text
        tokenized = open_clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokenized)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten().tolist()


class MLXCLIPBackend(CLIPBackend):
    """MLX-CLIP backend for Apple Silicon.

    Uses mlx_clip library which requires converting HuggingFace weights
    to MLX format first. Falls back to transformers if mlx_clip is not available.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._model: Any = None
        self._processor: Any = None  # Image processor (mlx_clip) or CLIPProcessor (transformers)
        self._tokenizer: Any = None  # Tokenizer for mlx_clip
        self._model_name = model_name
        self._use_mlx_clip = False

    def _get_mlx_cache_path(self) -> Path:
        """Get the local cache path for converted MLX weights."""
        cache_dir = Path.home() / ".cache" / "mlx_clip"
        safe_name = self._model_name.replace("/", "_")
        return cache_dir / safe_name

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from mlx_clip import (  # type: ignore[import-not-found]
                CLIPImageProcessor,
                CLIPModel,
                CLIPTokenizer,
                convert_weights,  # type: ignore[import-not-found]
            )

            mlx_path = self._get_mlx_cache_path()
            if not mlx_path.exists():
                logger.info(f"Converting {self._model_name} to MLX format at {mlx_path}")
                convert_weights(hf_repo=self._model_name, mlx_path=str(mlx_path))

            self._model = CLIPModel.from_pretrained(str(mlx_path))
            self._processor = CLIPImageProcessor.from_pretrained(str(mlx_path))
            self._tokenizer = CLIPTokenizer.from_pretrained(str(mlx_path))
            self._use_mlx_clip = True
            logger.info(f"Loaded MLX-CLIP model: {self._model_name}")
        except (ImportError, Exception) as e:
            logger.warning(f"MLX-CLIP not available ({e}), falling back to transformers")
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-not-found]

            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._use_mlx_clip = False
            logger.info(f"Loaded transformers CLIP model: {self._model_name}")

    def encode_image(self, image_path: str) -> Embedding:
        rgb_array = _load_image_rgb(image_path)
        return self.encode_image_from_array(rgb_array)

    def encode_image_from_array(self, rgb_array: NDArray[np.uint8]) -> Embedding:
        from PIL import Image

        self._load_model()
        image = Image.fromarray(rgb_array)

        if self._use_mlx_clip:
            import mlx.core as mx  # type: ignore[import-not-found]

            pixel_values = self._processor([image])  # Expects list of images
            features = self._model.get_image_features(pixel_values)
            # Normalize
            features = features / mx.linalg.norm(features, axis=-1, keepdims=True)
            return np.array(features).flatten().tolist()
        else:
            import torch

            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.get_image_features(pixel_values=inputs["pixel_values"])
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    outputs = outputs.last_hidden_state[:, 0]
                embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding.numpy().flatten().tolist()

    def get_model_name(self) -> str:
        return self._model_name

    def encode_text(self, text: str) -> Embedding:
        self._load_model()

        if self._use_mlx_clip:
            import mlx.core as mx  # type: ignore[import-not-found]

            tokens = self._tokenizer.tokenize(text)
            tokens = mx.expand_dims(mx.array(tokens), axis=0)  # Add batch dim
            features = self._model.get_text_features(tokens)
            features = features / mx.linalg.norm(features, axis=-1, keepdims=True)
            return np.array(features).flatten().tolist()
        else:
            import torch

            inputs = self._processor(text=text, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self._model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    outputs = outputs.last_hidden_state[:, 0]
                embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            return embedding.numpy().flatten().tolist()


def _to_feature_tensor(features: Any) -> Any:
    """Unwrap transformers feature outputs to a plain tensor.

    transformers 5 returns output objects (BaseModelOutputWithPooling) from
    get_image_features/get_text_features; older versions return the tensor.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0]
    return features


class SigLIPTransformersBackend(CLIPBackend):
    """SigLIP 2 backend via transformers, for all platforms (MPS/CUDA/CPU).

    SigLIP's architecture is not supported by mlx_clip or the CLIPModel
    classes, so a single transformers-based backend serves every platform;
    torch picks the accelerator (MPS on Apple Silicon, CUDA on NVIDIA).
    """

    def __init__(self, model_name: str) -> None:
        # Keep the requested (short) name for get_model_name so ClipResult.model
        # and /encode_text round-trip with the same identifier
        self._model_name = model_name
        self._hf_model = SIGLIP2_MODELS.get(model_name, model_name)
        self._model: Any = None
        self._processor: Any = None
        self._device = str(get_device())

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]

        logger.info(f"Loading SigLIP model: {self._hf_model} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(self._hf_model)
        self._model = AutoModel.from_pretrained(self._hf_model)
        self._model = self._model.to(self._device)
        self._model.eval()

    def encode_image(self, image_path: str) -> Embedding:
        rgb_array = _load_image_rgb(image_path)
        return self.encode_image_from_array(rgb_array)

    def encode_image_from_array(self, rgb_array: NDArray[np.uint8]) -> Embedding:
        import torch
        from PIL import Image

        self._load_model()
        image = Image.fromarray(rgb_array)

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = _to_feature_tensor(features)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()

    def get_model_name(self) -> str:
        return self._model_name

    def encode_text(self, text: str) -> Embedding:
        import torch

        self._load_model()

        # SigLIP text towers are trained with fixed-length padding
        inputs = self._processor(text=[text], padding="max_length", truncation=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = _to_feature_tensor(features)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()


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
        if isinstance(_backend, (MLXCLIPBackend, SigLIPTransformersBackend)):
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
    if _backend is not None and model_name is not None and _backend_model_name != model_name:
        logger.info(f"Switching CLIP model from {_backend_model_name} to {model_name}")
        unload_clip_model()

    if _backend is not None:
        return _backend

    # SigLIP models use the transformers backend on every platform - the
    # architecture is unsupported by mlx_clip and the CLIPModel classes
    if is_siglip_model(model_name):
        assert model_name is not None
        _backend = SigLIPTransformersBackend(model_name)
        _backend_model_name = model_name
        logger.info(f"Using SigLIP transformers backend: {model_name}")
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
            _backend_model_name = model_name or mlx_model  # Store original name for comparison
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


# Soft warning threshold for dense sampling (memory pressure territory)
MAX_RECOMMENDED_CLIP_SAMPLES = 5000


def _scene_index_for(timestamp: float, scene_boundaries: list[tuple[float, float]] | None) -> int | None:
    """Find the index of the scene containing a timestamp, or None."""
    if not scene_boundaries:
        return None
    for idx, (start, end) in enumerate(scene_boundaries):
        if start <= timestamp < end:
            return idx
    # Timestamp past the last scene boundary (e.g. rounding at clip end)
    if timestamp >= scene_boundaries[-1][1]:
        return len(scene_boundaries) - 1
    return None


def extract_clip(
    file_path: str,
    frame_buffer: SharedFrameBuffer | None = None,
    model_name: str | None = None,  # CLIP model name (e.g., "ViT-B-32", "ViT-L-14")
    sample_fps: float | None = None,
    scene_boundaries: list[tuple[float, float]] | None = None,
) -> ClipResult:
    """Extract CLIP embeddings from video frames.

    Two sampling modes:
    - Default (sample_fps=None): embeds the pre-decoded frames from the shared
      frame buffer (sample_mode "per_scene", the pre-1.1 behavior).
    - Fixed rate (sample_fps set): streams frames at the given rate via its own
      ffmpeg pipe, independent of the shared buffer, with bounded memory
      (sample_mode "fixed_fps").

    For images, use extract_clip_image() instead.

    Args:
        file_path: Path to video file
        frame_buffer: Pre-decoded frames (required when sample_fps is None)
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-L-14"). If None, uses default.
        sample_fps: Fixed sampling rate in Hz; None = default per-scene mode
        scene_boundaries: (start, end) per detected scene, used to assign
            scene_index in fixed_fps mode (ignored in default mode)

    Returns:
        ClipResult with embeddings per segment
    """
    if sample_fps is not None:
        return _extract_clip_fixed_fps(file_path, sample_fps, model_name, scene_boundaries)

    if frame_buffer is None:
        raise ValueError("frame_buffer is required when sample_fps is not set")

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
                    timestamp=ts,
                    scene_index=i,
                    embedding=embedding,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to encode frame at {ts}s: {e}")

    logger.info(f"Generated {len(segments)} CLIP embeddings")

    return ClipResult(
        model=backend.get_model_name(),
        sample_mode="per_scene",
        sample_fps=None,
        segments=segments,
    )


def _extract_clip_fixed_fps(
    file_path: str,
    sample_fps: float,
    model_name: str | None,
    scene_boundaries: list[tuple[float, float]] | None,
) -> ClipResult:
    """Extract CLIP embeddings at a fixed sample rate (fixed_fps mode).

    Streams frames in chunks so memory stays bounded regardless of clip length.
    """
    from media_engine.extractors.frame_buffer import iter_frames_fixed_fps
    from media_engine.extractors.frames import get_video_duration

    backend = get_clip_backend(model_name)

    duration = get_video_duration(file_path)
    expected = int(duration * sample_fps)
    if expected > MAX_RECOMMENDED_CLIP_SAMPLES:
        logger.warning(f"Dense CLIP sampling: {expected} samples requested for {file_path} ({duration:.0f}s at {sample_fps} fps) - expect memory pressure and long inference time")

    logger.info(f"Extracting CLIP embeddings at fixed {sample_fps} fps (~{expected} samples)")
    half_window = 0.5 / sample_fps
    segments: list[ClipSegment] = []

    try:
        for chunk in iter_frames_fixed_fps(file_path, fps=sample_fps):
            for frame in chunk:
                try:
                    embedding = backend.encode_image_from_array(frame.rgb)
                except Exception as e:
                    logger.warning(f"Failed to encode frame at {frame.timestamp}s: {e}")
                    continue
                ts = frame.timestamp
                segments.append(
                    ClipSegment(
                        start=max(0.0, ts - half_window),
                        end=min(duration, ts + half_window) if duration > 0 else ts + half_window,
                        timestamp=ts,
                        scene_index=_scene_index_for(ts, scene_boundaries),
                        embedding=embedding,
                    )
                )
            # Chunk (and its pixel data) is released before the next one is decoded
    except Exception as e:
        # Broken containers (e.g. missing duration metadata) can fail the
        # streaming path entirely — fall through to the single-frame fallback
        logger.warning(f"Fixed-fps streaming failed for {file_path}: {e} — trying single frame")

    if not segments:
        # Clips shorter than one sample window (duration < 1/fps) yield no
        # frames from the fps-filter stream, and broken containers may fail
        # streaming — embed a single frame so the clip still gets an
        # embedding. Try mid-clip first, then the very first frame (seeking
        # can silently return nothing on damaged files).
        from media_engine.extractors.frame_buffer import decode_frames

        mid = duration / 2 if duration > 0 else 0.0
        candidates = [mid, 0.0] if mid > 0 else [0.0]
        for candidate in candidates:
            logger.info(f"No streamed frames, embedding single frame at {candidate:.2f}s")
            try:
                buffer = decode_frames(file_path, timestamps=[candidate], max_dimension=1920)
            except Exception as e:
                logger.warning(f"Single-frame decode at {candidate:.2f}s failed for {file_path}: {e}")
                continue
            for ts in sorted(buffer.frames.keys()):
                try:
                    embedding = backend.encode_image_from_array(buffer.frames[ts].rgb)
                except Exception as e:
                    logger.warning(f"Failed to encode fallback frame at {ts}s: {e}")
                    continue
                segments.append(
                    ClipSegment(
                        start=0.0,
                        end=duration if duration > 0 else ts,
                        timestamp=ts,
                        scene_index=_scene_index_for(ts, scene_boundaries),
                        embedding=embedding,
                    )
                )
            if segments:
                break

        if not segments:
            # Last resort: read the first decodable frame WITHOUT seeking.
            # Damaged containers (missing index/duration) can refuse seeks
            # while still decoding sequentially from the start.
            logger.info(f"Seeked decodes failed, reading first frame without seek for {file_path}")
            try:
                rgb = _first_frame_no_seek(file_path)
                embedding = backend.encode_image_from_array(rgb)
                segments.append(
                    ClipSegment(
                        start=0.0,
                        end=duration,
                        timestamp=0.0,
                        scene_index=_scene_index_for(0.0, scene_boundaries),
                        embedding=embedding,
                    )
                )
            except Exception as e:
                logger.warning(f"Seek-free first-frame fallback failed for {file_path}: {e}")

    logger.info(f"Generated {len(segments)} CLIP embeddings (fixed_fps mode)")

    return ClipResult(
        model=backend.get_model_name(),
        sample_mode="fixed_fps",
        sample_fps=sample_fps,
        segments=segments,
    )


def _first_frame_no_seek(file_path: str, max_dimension: int = 1920) -> "np.ndarray":
    """Decode the very first frame of a video sequentially (no -ss seek).

    Streams one PNG through an image2pipe so no output dimensions need to be
    known up front. Used as a last resort for damaged containers that refuse
    seeking but still decode from the start.
    """
    import io
    import subprocess

    from PIL import Image

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        file_path,
        "-vf",
        f"scale='min({max_dimension},iw)':-2",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg first-frame decode failed: {result.stderr.decode(errors='replace')[:200]}")
    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    return np.asarray(image)


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
                timestamp=0.0,
                scene_index=0,
                embedding=embedding,
            )
        ],
    )


def encode_text_query(
    text: str,
    model_name: str | None = None,
) -> Embedding:
    """Encode a text query to a CLIP embedding.

    The resulting embedding can be used for text-to-image search by comparing
    against image embeddings using cosine similarity.

    Args:
        text: Text query to encode
        model_name: CLIP model name (e.g., "ViT-B-32", "ViT-L-14"). If None, uses default.

    Returns:
        List of floats representing the normalized embedding
    """
    backend = get_clip_backend(model_name)
    logger.info(f"Encoding text query: {text[:50]}...")
    return backend.encode_text(text)
