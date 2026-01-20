"""Tests for model loading/unloading lifecycle using images.

These tests verify that models can be loaded, used, and unloaded multiple times
without memory leaks or errors.

Requires images in /Volumes/Backup/drone/ or TEST_IMAGES_DIR environment variable.
"""

import gc
import glob
import os

import pytest

# Image directories to search (in order of preference)
IMAGE_SEARCH_PATHS = [
    "/Volumes/Backup/drone/20251015_vindhella_borgund",
    "/Volumes/Backup/drone",
    "/Volumes/Backup",
]


def get_test_images(count: int = 10) -> list[str]:
    """Find test images from backup directory.

    Args:
        count: Number of images to return

    Returns:
        List of image paths, or empty list if none found
    """
    # Check environment variable first
    custom_dir = os.environ.get("TEST_IMAGES_DIR")
    if custom_dir and os.path.isdir(custom_dir):
        search_paths = [custom_dir]
    else:
        search_paths = IMAGE_SEARCH_PATHS

    images: list[str] = []
    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue

        # Find JPG/JPEG files
        patterns = [
            os.path.join(search_path, "*.jpg"),
            os.path.join(search_path, "*.JPG"),
            os.path.join(search_path, "*.jpeg"),
            os.path.join(search_path, "*.JPEG"),
        ]

        for pattern in patterns:
            images.extend(glob.glob(pattern))
            if len(images) >= count:
                break

        if len(images) >= count:
            break

    return images[:count]


@pytest.fixture
def test_images():
    """Fixture providing test images."""
    images = get_test_images(10)
    if not images:
        pytest.skip(
            "No test images found. Set TEST_IMAGES_DIR or mount /Volumes/Backup"
        )
    return images


class TestYOLOModelLifecycle:
    """Test YOLO model loading/unloading cycles."""

    def test_yolo_load_unload_cycle(self, test_images):
        """Test YOLO can be loaded, used, and unloaded multiple times."""
        from polybos_engine.extractors.objects import _get_yolo_model, unload_yolo_model

        for _ in range(3):
            # Load model
            model = _get_yolo_model("yolov8n.pt")  # Use nano for speed
            assert model is not None

            # Run inference on images
            for img_path in test_images[:5]:
                results = model(img_path, verbose=False)
                assert results is not None

            # Unload model
            unload_yolo_model()
            gc.collect()

    def test_yolo_model_switching(self, test_images):
        """Test switching between YOLO models."""
        from polybos_engine.extractors.objects import _get_yolo_model, unload_yolo_model

        # Load nano model
        model1 = _get_yolo_model("yolov8n.pt")
        assert model1 is not None

        # Switch to small model (should unload nano first)
        model2 = _get_yolo_model("yolov8s.pt")
        assert model2 is not None

        # Back to nano
        model3 = _get_yolo_model("yolov8n.pt")
        assert model3 is not None

        # Cleanup
        unload_yolo_model()


class TestCLIPModelLifecycle:
    """Test CLIP model loading/unloading cycles."""

    def test_clip_load_unload_cycle(self, test_images):
        """Test CLIP can be loaded, used, and unloaded multiple times."""
        from polybos_engine.extractors.clip import get_clip_backend, unload_clip_model

        for _ in range(3):
            # Load model
            backend = get_clip_backend()
            assert backend is not None

            # Encode images
            for img_path in test_images[:5]:
                embedding = backend.encode_image(img_path)
                assert embedding is not None
                assert len(embedding) > 0

            # Unload model
            unload_clip_model()
            gc.collect()

    def test_clip_embedding_consistency(self, test_images):
        """Test CLIP produces consistent embeddings across load cycles."""
        from polybos_engine.extractors.clip import get_clip_backend, unload_clip_model

        # First load - get embeddings
        backend1 = get_clip_backend()
        embedding1 = backend1.encode_image(test_images[0])
        unload_clip_model()
        gc.collect()

        # Second load - same image should produce similar embedding
        backend2 = get_clip_backend()
        embedding2 = backend2.encode_image(test_images[0])
        unload_clip_model()

        # Embeddings should be very similar (not exact due to potential float differences)
        similarity = sum(a * b for a, b in zip(embedding1, embedding2))
        assert (
            similarity > 0.99
        ), f"Embeddings should be consistent, got similarity {similarity}"


class TestOCRModelLifecycle:
    """Test OCR model loading/unloading cycles."""

    def test_ocr_load_unload_cycle(self, test_images):
        """Test OCR can be loaded, used, and unloaded multiple times."""
        from polybos_engine.extractors.ocr import _get_ocr_reader, unload_ocr_model

        for _ in range(3):
            # Load model
            reader = _get_ocr_reader(["en"])
            assert reader is not None

            # Run OCR on images
            for img_path in test_images[:5]:
                results = reader.readtext(img_path)
                # Results may be empty if no text, but should not error
                assert isinstance(results, list)

            # Unload model
            unload_ocr_model()
            gc.collect()


class TestFaceModelLifecycle:
    """Test face detection model loading/unloading cycles."""

    def test_face_unload_multiple_times(self):
        """Test face model unload can be called multiple times safely."""
        from polybos_engine.extractors.faces import unload_face_model

        # Should not error even without loading
        for _ in range(5):
            unload_face_model()
            gc.collect()


class TestMultiModelBatchCycle:
    """Test multiple models in batch processing pattern."""

    def test_sequential_model_loading(self, test_images):
        """Test loading models sequentially like batch processing does."""
        from polybos_engine.extractors.clip import get_clip_backend, unload_clip_model
        from polybos_engine.extractors.objects import _get_yolo_model, unload_yolo_model
        from polybos_engine.extractors.ocr import _get_ocr_reader, unload_ocr_model

        for _ in range(2):
            # Stage 1: YOLO
            model = _get_yolo_model("yolov8n.pt")
            for img_path in test_images:
                model(img_path, verbose=False)
            unload_yolo_model()
            gc.collect()

            # Stage 2: OCR
            reader = _get_ocr_reader(["en"])
            for img_path in test_images:
                reader.readtext(img_path)
            unload_ocr_model()
            gc.collect()

            # Stage 3: CLIP
            backend = get_clip_backend()
            for img_path in test_images:
                backend.encode_image(img_path)
            unload_clip_model()
            gc.collect()

    def test_batch_memory_stability(self, test_images):
        """Test that memory is released between batch cycles."""
        from polybos_engine.config import get_available_memory_gb
        from polybos_engine.extractors.clip import get_clip_backend, unload_clip_model
        from polybos_engine.extractors.objects import _get_yolo_model, unload_yolo_model

        # Get baseline memory
        gc.collect()
        baseline_ram, _ = get_available_memory_gb()

        memory_readings = []

        for _ in range(3):
            # Load and use YOLO
            model = _get_yolo_model("yolov8n.pt")
            for img_path in test_images:
                model(img_path, verbose=False)
            unload_yolo_model()

            # Load and use CLIP
            backend = get_clip_backend()
            for img_path in test_images:
                backend.encode_image(img_path)
            unload_clip_model()

            # Measure memory after cycle
            gc.collect()
            ram, vram = get_available_memory_gb()
            memory_readings.append((ram, vram))

        # Memory should not decrease significantly across cycles
        # (allowing some variance for system activity)
        final_ram, _ = memory_readings[-1]

        # Allow 20% variance
        assert (
            final_ram > baseline_ram * 0.8
        ), f"RAM decreased too much: {baseline_ram:.1f}GB -> {final_ram:.1f}GB"


class TestConcurrentUnloads:
    """Test that unload functions are safe to call concurrently."""

    def test_all_unloads_safe(self):
        """Test calling all unload functions in sequence."""
        from polybos_engine.extractors import (
            unload_clip_model,
            unload_face_model,
            unload_ocr_model,
            unload_qwen_model,
            unload_vad_model,
            unload_whisper_model,
            unload_yolo_model,
        )

        # Call all unloads multiple times
        for _ in range(3):
            unload_yolo_model()
            unload_clip_model()
            unload_ocr_model()
            unload_face_model()
            unload_whisper_model()
            unload_qwen_model()
            unload_vad_model()
            gc.collect()
