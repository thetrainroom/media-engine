"""Tests for API endpoints."""


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


def test_settings_get(client):
    """Test GET /settings endpoint."""
    response = client.get("/settings")
    assert response.status_code == 200

    data = response.json()
    assert "whisper_model" in data
    assert "hf_token_set" in data
    assert isinstance(data["hf_token_set"], bool)
    assert "face_sample_fps" in data
    assert "object_detector" in data


def test_settings_update(client):
    """Test PUT /settings endpoint."""
    # Get current settings
    original = client.get("/settings").json()

    # Update a setting
    response = client.put("/settings", json={"face_sample_fps": 2.5})
    assert response.status_code == 200

    data = response.json()
    assert data["face_sample_fps"] == 2.5

    # Restore original
    client.put("/settings", json={"face_sample_fps": original["face_sample_fps"]})
