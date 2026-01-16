"""Tests for batch API with real video files.

These tests verify the full batch processing flow through the API:
- Creating batch jobs
- Polling for completion
- Verifying results
- Model loading/unloading between stages

Requires videos in /Volumes/Backup/drone/ or TEST_VIDEO_PATH environment variable.
"""

import glob
import os
import time

import pytest
from fastapi.testclient import TestClient

from polybos_engine.main import app


# Video directories to search (in order of preference)
VIDEO_SEARCH_PATHS = [
    "/Volumes/Backup/drone/20251015_vindhella_borgund",
    "/Volumes/Backup/drone",
    "/Volumes/Backup",
]


def get_test_videos(count: int = 3) -> list[str]:
    """Find test videos from backup directory.

    Args:
        count: Number of videos to return

    Returns:
        List of video paths, or empty list if none found
    """
    # Check environment variable first
    custom_path = os.environ.get("TEST_VIDEO_PATH")
    if custom_path and os.path.isfile(custom_path):
        return [custom_path]

    search_paths = VIDEO_SEARCH_PATHS

    videos: list[str] = []
    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue

        # Find video files (MP4, MOV)
        patterns = [
            os.path.join(search_path, "*.mp4"),
            os.path.join(search_path, "*.MP4"),
            os.path.join(search_path, "*.mov"),
            os.path.join(search_path, "*.MOV"),
        ]

        for pattern in patterns:
            videos.extend(glob.glob(pattern))
            if len(videos) >= count:
                break

        if len(videos) >= count:
            break

    return videos[:count]


@pytest.fixture
def api_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_videos():
    """Fixture providing test videos."""
    videos = get_test_videos(3)
    if not videos:
        pytest.skip("No test videos found. Set TEST_VIDEO_PATH or mount /Volumes/Backup")
    return videos


def wait_for_batch(client: TestClient, batch_id: str, timeout: int = 300) -> dict:
    """Poll batch status until completion or timeout.

    Args:
        client: Test client
        batch_id: Batch job ID
        timeout: Maximum wait time in seconds

    Returns:
        Final batch status dict
    """
    start = time.time()
    while time.time() - start < timeout:
        response = client.get(f"/batch/{batch_id}")
        assert response.status_code == 200

        status = response.json()
        if status["status"] in ("completed", "failed"):
            return status

        time.sleep(1)

    pytest.fail(f"Batch {batch_id} did not complete within {timeout}s")


class TestBatchCreation:
    """Test batch job creation."""

    def test_create_batch_metadata_only(self, api_client, test_videos):
        """Test creating batch with metadata only (fast)."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "batch_id" in data

        # Wait for completion
        status = wait_for_batch(api_client, data["batch_id"], timeout=30)
        assert status["status"] == "completed"
        assert len(status["files"]) == 1
        assert "metadata" in status["files"][0]["results"]

    def test_batch_returns_timing_metrics(self, api_client, test_videos):
        """Test that batch status includes timing and memory metrics."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_scenes": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=120)

        # Check batch-level metrics
        assert status["status"] == "completed"
        assert "elapsed_seconds" in status
        assert status["elapsed_seconds"] > 0
        assert "memory_mb" in status
        assert "peak_memory_mb" in status
        assert status["peak_memory_mb"] >= status["memory_mb"]

        # Check extractor timings
        assert "extractor_timings" in status
        assert len(status["extractor_timings"]) >= 1  # At least metadata

        for timing in status["extractor_timings"]:
            assert "extractor" in timing
            assert "duration_seconds" in timing
            assert "files_processed" in timing
            assert timing["duration_seconds"] >= 0
            assert timing["files_processed"] == 1

        # Check per-file timings
        assert len(status["files"]) == 1
        file_status = status["files"][0]
        assert "timings" in file_status
        assert "metadata" in file_status["timings"]
        assert file_status["timings"]["metadata"] >= 0

    def test_create_batch_file_not_found(self, api_client):
        """Test batch with non-existent file."""
        response = api_client.post(
            "/batch",
            json={
                "files": ["/nonexistent/video.mp4"],
                "enable_metadata": True,
            },
        )
        assert response.status_code == 404

    def test_create_batch_multiple_files(self, api_client, test_videos):
        """Test batch with multiple files."""
        if len(test_videos) < 2:
            pytest.skip("Need at least 2 test videos")

        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:2],
                "enable_metadata": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=60)

        assert status["status"] == "completed"
        assert len(status["files"]) == 2

        # All files should have metadata
        for file_status in status["files"]:
            assert file_status["status"] == "completed"
            assert "metadata" in file_status["results"]


class TestBatchExtractors:
    """Test batch with different extractors."""

    def test_batch_with_scenes(self, api_client, test_videos):
        """Test batch with scene detection enabled."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_scenes": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=120)

        assert status["status"] == "completed"
        assert "scenes" in status["files"][0]["results"]

    def test_batch_with_vad(self, api_client, test_videos):
        """Test batch with voice activity detection."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_vad": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=60)

        assert status["status"] == "completed"
        assert "vad" in status["files"][0]["results"]

    @pytest.mark.slow
    def test_batch_with_objects_yolo(self, api_client, test_videos):
        """Test batch with YOLO object detection."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_objects": True,
                "object_detector": "yolo",
                "yolo_model": "yolov8n.pt",  # Use nano for speed
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=180)

        assert status["status"] == "completed"
        assert "objects" in status["files"][0]["results"]

    @pytest.mark.slow
    def test_batch_with_ocr(self, api_client, test_videos):
        """Test batch with OCR."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_ocr": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=180)

        assert status["status"] == "completed"
        assert "ocr" in status["files"][0]["results"]

    @pytest.mark.slow
    def test_batch_with_clip(self, api_client, test_videos):
        """Test batch with CLIP embeddings."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_clip": True,
                "clip_model": "ViT-B-32",  # Smaller model for speed
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=180)

        assert status["status"] == "completed"
        assert "clip" in status["files"][0]["results"]


class TestBatchModelConfiguration:
    """Test per-request model configuration."""

    def test_batch_auto_model_selection(self, api_client, test_videos):
        """Test batch with 'auto' model selection."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_objects": True,
                "object_detector": "yolo",
                "yolo_model": "auto",
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=180)
        assert status["status"] == "completed"

    def test_batch_explicit_model_selection(self, api_client, test_videos):
        """Test batch with explicit model names."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_objects": True,
                "object_detector": "yolo",
                "yolo_model": "yolov8s.pt",  # Explicit small model
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=180)
        assert status["status"] == "completed"


class TestBatchLifecycle:
    """Test batch job lifecycle and cleanup."""

    def test_batch_status_polling(self, api_client, test_videos):
        """Test that batch status can be polled during processing."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_scenes": True,
            },
        )
        data = response.json()
        batch_id = data["batch_id"]

        # Poll a few times
        for _ in range(3):
            response = api_client.get(f"/batch/{batch_id}")
            assert response.status_code == 200

            status = response.json()
            assert status["batch_id"] == batch_id
            assert status["status"] in ("pending", "running", "completed", "failed")

            if status["status"] == "completed":
                break
            time.sleep(1)

    def test_batch_delete(self, api_client, test_videos):
        """Test deleting a completed batch."""
        # Create and wait for batch
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
            },
        )
        batch_id = response.json()["batch_id"]
        wait_for_batch(api_client, batch_id, timeout=30)

        # Delete batch
        response = api_client.delete(f"/batch/{batch_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        # Verify it's gone
        response = api_client.get(f"/batch/{batch_id}")
        assert response.status_code == 404

    def test_batch_not_found(self, api_client):
        """Test getting non-existent batch."""
        response = api_client.get("/batch/nonexistent")
        assert response.status_code == 404


class TestBatchMultipleExtractors:
    """Test batch with multiple extractors (model loading/unloading)."""

    @pytest.mark.slow
    def test_batch_multiple_extractors_sequential(self, api_client, test_videos):
        """Test batch with multiple extractors - verifies model unloading between stages."""
        response = api_client.post(
            "/batch",
            json={
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_scenes": True,
                "enable_objects": True,
                "enable_ocr": True,
                "object_detector": "yolo",
                "yolo_model": "yolov8n.pt",
            },
        )
        assert response.status_code == 200

        data = response.json()
        status = wait_for_batch(api_client, data["batch_id"], timeout=300)

        assert status["status"] == "completed"
        results = status["files"][0]["results"]

        assert "metadata" in results
        assert "scenes" in results
        assert "objects" in results
        assert "ocr" in results

    def test_batch_repeated_runs_lightweight(self, api_client, test_videos):
        """Fast stress test with lightweight extractors (metadata + scenes only).

        Runs 3 batches to verify memory cleanup between runs without heavy models.
        """
        num_runs = 3
        memory_history = []

        for i in range(num_runs):
            request = {
                "files": test_videos[:1],
                "enable_metadata": True,
                "enable_scenes": True,
            }

            response = api_client.post("/batch", json=request)
            assert response.status_code == 200, f"Batch {i+1}/{num_runs} creation failed"

            data = response.json()
            batch_id = data["batch_id"]

            status = wait_for_batch(api_client, batch_id, timeout=120)
            assert status["status"] == "completed", f"Batch {i+1}/{num_runs} failed: {status}"

            # Track memory and timing
            memory_history.append({
                "batch": i + 1,
                "elapsed_seconds": status.get("elapsed_seconds"),
                "memory_mb": status.get("memory_mb"),
                "peak_memory_mb": status.get("peak_memory_mb"),
                "extractor_timings": status.get("extractor_timings", []),
            })

            # Print timing report
            print(f"\nBatch {i+1}/{num_runs}:")
            print(f"  Elapsed: {status.get('elapsed_seconds')}s")
            print(f"  Memory: {status.get('memory_mb')}MB (peak: {status.get('peak_memory_mb')}MB)")
            for timing in status.get("extractor_timings", []):
                print(f"  {timing['extractor']}: {timing['duration_seconds']}s")

            # Delete to free memory
            api_client.delete(f"/batch/{batch_id}")

        # Verify memory doesn't grow excessively (peak should be within 50% of first run)
        if memory_history[0]["peak_memory_mb"] and memory_history[-1]["peak_memory_mb"]:
            first_peak = memory_history[0]["peak_memory_mb"]
            last_peak = memory_history[-1]["peak_memory_mb"]
            assert last_peak < first_peak * 1.5, f"Memory grew too much: {first_peak}MB -> {last_peak}MB"

    @pytest.mark.slow
    def test_batch_repeated_runs_stress(self, api_client, test_videos):
        """Test running multiple batches sequentially - stress test for memory cleanup.

        This is the key test for verifying that models are properly unloaded
        between batch runs and memory doesn't accumulate.
        """
        num_runs = 5
        memory_history = []

        for i in range(num_runs):
            # Vary the extractors to test different model loading patterns
            if i % 3 == 0:
                # YOLO + OCR
                request = {
                    "files": test_videos[:1],
                    "enable_metadata": True,
                    "enable_objects": True,
                    "enable_ocr": True,
                    "object_detector": "yolo",
                    "yolo_model": "yolov8n.pt",
                }
            elif i % 3 == 1:
                # CLIP + scenes
                request = {
                    "files": test_videos[:1],
                    "enable_metadata": True,
                    "enable_scenes": True,
                    "enable_clip": True,
                    "clip_model": "ViT-B-32",
                }
            else:
                # YOLO + CLIP + OCR (heavy)
                request = {
                    "files": test_videos[:1],
                    "enable_metadata": True,
                    "enable_objects": True,
                    "enable_clip": True,
                    "enable_ocr": True,
                    "object_detector": "yolo",
                    "yolo_model": "yolov8n.pt",
                    "clip_model": "ViT-B-32",
                }

            response = api_client.post("/batch", json=request)
            assert response.status_code == 200, f"Batch {i+1}/{num_runs} creation failed"

            data = response.json()
            batch_id = data["batch_id"]

            status = wait_for_batch(api_client, batch_id, timeout=300)
            assert status["status"] == "completed", f"Batch {i+1}/{num_runs} failed: {status}"

            # Track memory and timing
            memory_history.append({
                "batch": i + 1,
                "elapsed_seconds": status.get("elapsed_seconds"),
                "memory_mb": status.get("memory_mb"),
                "peak_memory_mb": status.get("peak_memory_mb"),
            })

            # Print timing report
            print(f"\nBatch {i+1}/{num_runs}:")
            print(f"  Elapsed: {status.get('elapsed_seconds')}s")
            print(f"  Memory: {status.get('memory_mb')}MB (peak: {status.get('peak_memory_mb')}MB)")
            for timing in status.get("extractor_timings", []):
                print(f"  {timing['extractor']}: {timing['duration_seconds']}s")

            # Delete to free memory
            api_client.delete(f"/batch/{batch_id}")

    @pytest.mark.slow
    def test_batch_concurrent_style_sequential(self, api_client, test_videos):
        """Test multiple batches as if submitted rapidly (but processed sequentially).

        Simulates the frontend submitting multiple files for processing.
        """
        if len(test_videos) < 3:
            pytest.skip("Need at least 3 test videos")

        batch_ids = []

        # Submit 3 batches rapidly
        for video in test_videos[:3]:
            response = api_client.post(
                "/batch",
                json={
                    "files": [video],
                    "enable_metadata": True,
                    "enable_objects": True,
                    "object_detector": "yolo",
                    "yolo_model": "yolov8n.pt",
                },
            )
            assert response.status_code == 200
            batch_ids.append(response.json()["batch_id"])

        # Wait for all to complete
        for i, batch_id in enumerate(batch_ids):
            status = wait_for_batch(api_client, batch_id, timeout=300)
            assert status["status"] == "completed", f"Batch {i+1} failed"

        # Cleanup
        for batch_id in batch_ids:
            api_client.delete(f"/batch/{batch_id}")


class TestHardwareEndpoint:
    """Test hardware info endpoint."""

    def test_hardware_info(self, api_client):
        """Test hardware endpoint returns model info."""
        response = api_client.get("/hardware")
        assert response.status_code == 200

        data = response.json()
        assert "device" in data
        assert "vram_gb" in data
        assert "auto_whisper_model" in data
        assert "auto_yolo_model" in data
        assert "auto_clip_model" in data
        assert "auto_object_detector" in data
        assert "recommendations" in data
