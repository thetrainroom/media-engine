"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from media_engine.main import app


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_video_path():
    """Path to test video file.

    Set TEST_VIDEO_PATH environment variable to use a custom test video.
    """
    path = os.environ.get("TEST_VIDEO_PATH")
    if path and Path(path).exists():
        return path
    pytest.skip("TEST_VIDEO_PATH not set or file not found")


@pytest.fixture
def short_video_path():
    """Path to short test video (for quick tests).

    Set SHORT_VIDEO_PATH environment variable.
    """
    path = os.environ.get("SHORT_VIDEO_PATH")
    if path and Path(path).exists():
        return path
    pytest.skip("SHORT_VIDEO_PATH not set or file not found")
