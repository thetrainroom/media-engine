"""Tests for API endpoints."""

import pytest


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "api_version" in data


def test_extractors_list(client):
    """Test extractors list endpoint."""
    response = client.get("/extractors")
    assert response.status_code == 200

    data = response.json()
    assert "extractors" in data

    names = [e["name"] for e in data["extractors"]]
    assert "metadata" in names
    assert "transcript" in names
    assert "faces" in names
    assert "scenes" in names
    assert "objects" in names
    assert "clip" in names
    assert "ocr" in names


def test_extract_file_not_found(client):
    """Test extract with non-existent file."""
    response = client.post("/extract", json={"file": "/nonexistent/video.mp4"})
    assert response.status_code == 404


def test_extract_metadata_only(client, test_video_path):
    """Test extract with only metadata (all extractors skipped)."""
    response = client.post(
        "/extract",
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

    data = response.json()
    assert data["file"] == test_video_path
    assert "metadata" in data
    assert data["metadata"]["duration"] > 0
    assert data["transcript"] is None
    assert data["faces"] is None


@pytest.mark.slow
def test_extract_full(client, test_video_path):
    """Test full extraction (all extractors enabled)."""
    response = client.post(
        "/extract",
        json={"file": test_video_path},
        timeout=300,  # 5 minutes for full extraction
    )
    assert response.status_code == 200

    data = response.json()
    assert "metadata" in data
    assert "extraction_time_seconds" in data
