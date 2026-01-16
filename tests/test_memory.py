"""Tests for memory management functionality."""

import time

import pytest


class TestUnloadFunctions:
    """Test model unload functions."""

    def test_unload_yolo_model(self):
        """Test YOLO unload function can be called safely."""
        from polybos_engine.extractors.objects import unload_yolo_model

        # Should not raise even if model not loaded
        unload_yolo_model()
        unload_yolo_model()  # Call twice to ensure idempotent

    def test_unload_clip_model(self):
        """Test CLIP unload function can be called safely."""
        from polybos_engine.extractors.clip import unload_clip_model

        unload_clip_model()
        unload_clip_model()

    def test_unload_ocr_model(self):
        """Test OCR unload function can be called safely."""
        from polybos_engine.extractors.ocr import unload_ocr_model

        unload_ocr_model()
        unload_ocr_model()

    def test_unload_face_model(self):
        """Test face detection unload function can be called safely."""
        from polybos_engine.extractors.faces import unload_face_model

        unload_face_model()
        unload_face_model()

    def test_unload_whisper_model(self):
        """Test Whisper unload function can be called safely."""
        from polybos_engine.extractors.transcribe import unload_whisper_model

        unload_whisper_model()
        unload_whisper_model()

    def test_unload_qwen_model(self):
        """Test Qwen unload function can be called safely."""
        from polybos_engine.extractors.objects_qwen import unload_qwen_model

        unload_qwen_model()
        unload_qwen_model()

    def test_unload_vad_model(self):
        """Test VAD unload function can be called safely."""
        from polybos_engine.extractors.vad import unload_vad_model

        unload_vad_model()
        unload_vad_model()

    def test_all_unload_functions_exported(self):
        """Test all unload functions are exported from extractors package."""
        from polybos_engine.extractors import (
            unload_clip_model,
            unload_face_model,
            unload_ocr_model,
            unload_qwen_model,
            unload_vad_model,
            unload_whisper_model,
            unload_yolo_model,
        )

        # All should be callable
        assert callable(unload_yolo_model)
        assert callable(unload_clip_model)
        assert callable(unload_ocr_model)
        assert callable(unload_face_model)
        assert callable(unload_whisper_model)
        assert callable(unload_qwen_model)
        assert callable(unload_vad_model)


class TestMemoryMonitoring:
    """Test memory monitoring functions."""

    def test_model_memory_requirements_defined(self):
        """Test MODEL_MEMORY_REQUIREMENTS has expected models."""
        from polybos_engine.config import MODEL_MEMORY_REQUIREMENTS

        # Whisper models
        assert "tiny" in MODEL_MEMORY_REQUIREMENTS
        assert "small" in MODEL_MEMORY_REQUIREMENTS
        assert "medium" in MODEL_MEMORY_REQUIREMENTS
        assert "large-v3" in MODEL_MEMORY_REQUIREMENTS

        # YOLO models
        assert "yolov8m.pt" in MODEL_MEMORY_REQUIREMENTS

        # Qwen models
        assert "Qwen/Qwen2-VL-2B-Instruct" in MODEL_MEMORY_REQUIREMENTS

        # All values should be positive floats
        for memory in MODEL_MEMORY_REQUIREMENTS.values():
            assert isinstance(memory, float)
            assert memory > 0

    def test_get_available_memory_gb(self):
        """Test get_available_memory_gb returns valid values."""
        from polybos_engine.config import get_available_memory_gb

        ram, vram = get_available_memory_gb()

        # RAM should be positive
        assert isinstance(ram, float)
        assert ram > 0

        # VRAM can be 0 if no GPU
        assert isinstance(vram, float)
        assert vram >= 0

    def test_check_memory_before_load_returns_bool(self):
        """Test check_memory_before_load returns boolean."""
        from polybos_engine.config import check_memory_before_load

        # Should return True for small model (likely to fit)
        result = check_memory_before_load("tiny")
        assert isinstance(result, bool)

    def test_check_memory_with_clear_function(self):
        """Test check_memory_before_load can accept clear function."""
        from polybos_engine.config import check_memory_before_load

        clear_called = []

        def mock_clear():
            clear_called.append(True)

        # With a small model, clear shouldn't be needed
        check_memory_before_load("tiny", clear_memory_func=mock_clear)

        # Clear function may or may not be called depending on available memory

    def test_get_available_vram_gb(self):
        """Test VRAM detection function."""
        from polybos_engine.config import get_available_vram_gb

        vram = get_available_vram_gb()
        assert isinstance(vram, float)
        assert vram >= 0


class TestJobCleanup:
    """Test job queue cleanup functionality."""

    def test_job_ttl_constant_defined(self):
        """Test JOB_TTL_SECONDS is defined."""
        from polybos_engine.main import JOB_TTL_SECONDS

        assert isinstance(JOB_TTL_SECONDS, int)
        assert JOB_TTL_SECONDS > 0

    def test_cleanup_expired_jobs_function_exists(self):
        """Test cleanup function exists."""
        from polybos_engine.main import _cleanup_expired_jobs

        assert callable(_cleanup_expired_jobs)

    def test_cleanup_expired_jobs_returns_count(self):
        """Test cleanup function returns removed count."""
        from polybos_engine.main import _cleanup_expired_jobs

        result = _cleanup_expired_jobs()
        assert isinstance(result, int)
        assert result >= 0

    def test_cleanup_expired_batch_jobs_function_exists(self):
        """Test batch cleanup function exists."""
        from polybos_engine.main import _cleanup_expired_batch_jobs

        assert callable(_cleanup_expired_batch_jobs)

    def test_delete_job_endpoint(self, client):
        """Test DELETE /jobs/{job_id} endpoint."""
        # Try to delete non-existent job
        response = client.delete("/jobs/nonexistent")
        assert response.status_code == 404

    def test_delete_batch_endpoint(self, client):
        """Test DELETE /batch/{batch_id} endpoint."""
        # Try to delete non-existent batch
        response = client.delete("/batch/nonexistent")
        assert response.status_code == 404


class TestJobCleanupIntegration:
    """Integration tests for job cleanup."""

    def test_job_has_completed_at_field(self, client, test_video_path):
        """Test that completed jobs have completed_at timestamp."""
        # Create a job
        response = client.post(
            "/jobs",
            json={
                "file": test_video_path,
                "skip_transcript": True,
                "skip_faces": True,
                "skip_scenes": True,
                "skip_objects": True,
                "skip_clip": True,
                "skip_ocr": True,
            },
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for completion (metadata only should be fast)
        for _ in range(30):
            response = client.get(f"/jobs/{job_id}")
            assert response.status_code == 200
            job = response.json()
            if job["status"] == "completed":
                assert job["completed_at"] is not None
                break
            time.sleep(0.5)
        else:
            pytest.fail("Job did not complete in time")

        # Clean up
        client.delete(f"/jobs/{job_id}")

    def test_can_delete_completed_job(self, client, test_video_path):
        """Test that completed jobs can be deleted."""
        # Create and wait for job
        response = client.post(
            "/jobs",
            json={
                "file": test_video_path,
                "skip_transcript": True,
                "skip_faces": True,
                "skip_scenes": True,
                "skip_objects": True,
                "skip_clip": True,
                "skip_ocr": True,
            },
        )
        job_id = response.json()["job_id"]

        # Wait for completion
        for _ in range(30):
            response = client.get(f"/jobs/{job_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(0.5)

        # Delete job
        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's gone
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 404


class TestHardwareEndpoint:
    """Test hardware info endpoint."""

    def test_hardware_endpoint(self, client):
        """Test GET /hardware returns VRAM info."""
        response = client.get("/hardware")
        assert response.status_code == 200

        data = response.json()
        assert "device" in data
        assert "vram_gb" in data
        assert "auto_whisper_model" in data
        assert "auto_object_detector" in data
        assert "recommendations" in data

        # VRAM should be non-negative
        assert data["vram_gb"] >= 0


class TestAutoModelSelection:
    """Test automatic model selection based on VRAM."""

    def test_auto_whisper_model(self):
        """Test Whisper auto-selection returns valid model."""
        from polybos_engine.config import get_auto_whisper_model

        model = get_auto_whisper_model()
        assert model in ["tiny", "small", "medium", "large-v3"]

    def test_auto_qwen_model(self):
        """Test Qwen auto-selection returns valid model."""
        from polybos_engine.config import get_auto_qwen_model

        model = get_auto_qwen_model()
        assert "Qwen" in model

    def test_auto_object_detector(self):
        """Test object detector auto-selection returns valid option."""
        from polybos_engine.config import ObjectDetector, get_auto_object_detector

        detector = get_auto_object_detector()
        assert detector in [ObjectDetector.YOLO, ObjectDetector.QWEN]

    def test_auto_yolo_model(self):
        """Test YOLO auto-selection returns valid model."""
        from polybos_engine.config import get_auto_yolo_model

        model = get_auto_yolo_model()
        assert model in [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ]

    def test_auto_clip_model(self):
        """Test CLIP auto-selection returns valid model."""
        from polybos_engine.config import get_auto_clip_model

        model = get_auto_clip_model()
        assert model in ["ViT-B-16", "ViT-B-32", "ViT-L-14"]
