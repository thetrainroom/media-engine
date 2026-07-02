"""Tests for API endpoints."""


def test_api_version_is_code_owned(tmp_path):
    """api_version is never persisted and a stale value on disk is ignored."""
    import json

    from media_engine.config import DEFAULT_API_VERSION, Settings, load_config_from_file, save_config_to_file

    path = tmp_path / "config.json"
    save_config_to_file(Settings(), config_path=path)
    assert "api_version" not in load_config_from_file(path)

    # A config written by an older engine must not pin the reported version
    stale = load_config_from_file(path)
    stale["api_version"] = "1.0"
    path.write_text(json.dumps(stale))
    loaded = load_config_from_file(path)
    loaded.pop("api_version", None)  # mirrors get_settings()
    assert Settings(**loaded).api_version == DEFAULT_API_VERSION


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["api_version"] == "1.1"


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
    assert data["api_version"] == "1.1"
    assert "whisper_model" in data
    assert "hf_token_set" in data
    assert isinstance(data["hf_token_set"], bool)
    assert "face_sample_fps" in data
    assert "object_detector" in data
    # api 1.1 settings
    assert "clip_default_sample_fps" in data
    assert "motion_features_enabled" in data


def test_batch_clip_sample_fps_validation(client):
    """clip_sample_fps outside [0.1, 10.0] is rejected before any processing."""
    # Below minimum
    response = client.post("/batch", json={"files": ["/nonexistent.mp4"], "clip_sample_fps": 0.05})
    assert response.status_code == 422

    # Above maximum
    response = client.post("/batch", json={"files": ["/nonexistent.mp4"], "clip_sample_fps": 20.0})
    assert response.status_code == 422

    # Valid value passes validation (fails later on file existence, not schema)
    response = client.post("/batch", json={"files": ["/nonexistent.mp4"], "clip_sample_fps": 1.0})
    assert response.status_code == 404


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


def test_settings_v11_roundtrip(client):
    """The api-1.1 settings update, persist, and clear correctly."""
    original = client.get("/settings").json()

    response = client.put("/settings", json={"clip_default_sample_fps": 1.5, "motion_features_enabled": False})
    assert response.status_code == 200
    data = response.json()
    assert data["clip_default_sample_fps"] == 1.5
    assert data["motion_features_enabled"] is False

    # Out-of-range default rate is rejected
    response = client.put("/settings", json={"clip_default_sample_fps": 50.0})
    assert response.status_code == 422

    # Explicit null clears back to per-scene mode
    response = client.put("/settings", json={"clip_default_sample_fps": None})
    assert response.status_code == 200
    assert response.json()["clip_default_sample_fps"] is None

    # Restore original
    client.put(
        "/settings",
        json={
            "clip_default_sample_fps": original["clip_default_sample_fps"],
            "motion_features_enabled": original["motion_features_enabled"],
        },
    )
