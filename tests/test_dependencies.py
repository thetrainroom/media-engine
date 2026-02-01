"""Test that all required dependencies are importable."""

import importlib
import importlib.util

import pytest

# Modules that crash on import (e.g., paddle on macOS) - check with find_spec only
SPEC_CHECK_ONLY = {"paddle"}


class TestBaseDependencies:
    """Test base dependencies that are always required."""

    @pytest.mark.parametrize(
        "module",
        [
            "fastapi",
            "uvicorn",
            "pydantic",
            "pydantic_settings",
            "ffmpeg",
            "scenedetect",
            "deepface",
            "tf_keras",
            "transformers",
            "paddleocr",
            "paddle",
            "webrtcvad",
            "psutil",
            "langdetect",
            "httpx",
        ],
    )
    def test_base_dependency_importable(self, module: str):
        """Verify base dependency can be imported."""
        if module in SPEC_CHECK_ONLY:
            # Some modules crash on import but are installed - just check spec
            if importlib.util.find_spec(module) is None:
                pytest.fail(f"Base dependency '{module}' not installed")
            return

        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Base dependency '{module}' not importable: {e}")


class TestMLXDependencies:
    """Test MLX dependencies (Apple Silicon)."""

    @pytest.mark.parametrize(
        "module",
        [
            "mlx",
            "mlx_whisper",
            "open_clip",
            "torch",
            "ultralytics",
        ],
    )
    def test_mlx_dependency_importable(self, module: str):
        """Verify MLX dependency can be imported if MLX extras installed."""
        if importlib.util.find_spec("mlx") is None:
            pytest.skip("MLX extras not installed")
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"MLX dependency '{module}' not importable: {e}")


class TestCUDADependencies:
    """Test CUDA dependencies (NVIDIA GPU)."""

    @pytest.mark.parametrize(
        "module",
        [
            "faster_whisper",
            "open_clip",
            "torch",
            "ultralytics",
        ],
    )
    def test_cuda_dependency_importable(self, module: str):
        """Verify CUDA dependency can be imported if CUDA extras installed."""
        if importlib.util.find_spec("faster_whisper") is None:
            pytest.skip("CUDA extras not installed")
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"CUDA dependency '{module}' not importable: {e}")


class TestCPUDependencies:
    """Test CPU dependencies (fallback)."""

    @pytest.mark.parametrize(
        "module",
        [
            "whisper",
            "open_clip",
            "torch",
            "ultralytics",
        ],
    )
    def test_cpu_dependency_importable(self, module: str):
        """Verify CPU dependency can be imported if CPU extras installed."""
        if importlib.util.find_spec("whisper") is None:
            pytest.skip("CPU extras not installed")
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"CPU dependency '{module}' not importable: {e}")


class TestQwenDependencies:
    """Test Qwen dependencies (VLM)."""

    @pytest.mark.parametrize(
        "module",
        [
            "accelerate",
            "qwen_vl_utils",
        ],
    )
    def test_qwen_dependency_importable(self, module: str):
        """Verify Qwen dependency can be imported if Qwen extras installed."""
        if importlib.util.find_spec("qwen_vl_utils") is None:
            pytest.skip("Qwen extras not installed")
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Qwen dependency '{module}' not importable: {e}")


class TestMediaEngineImports:
    """Test that media_engine modules are importable."""

    @pytest.mark.parametrize(
        "module",
        [
            "media_engine",
            "media_engine.main",
            "media_engine.config",
            "media_engine.schemas",
            "media_engine.extractors",
            "media_engine.extractors.metadata",
            "media_engine.extractors.scenes",
            "media_engine.extractors.ocr",
            "media_engine.extractors.telemetry",
        ],
    )
    def test_media_engine_module_importable(self, module: str):
        """Verify media_engine module can be imported."""
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Module '{module}' not importable: {e}")

    def test_extractors_with_torch(self):
        """Verify extractors requiring torch are importable when torch is available."""
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch not installed")

        modules = [
            "media_engine.extractors.faces",
            "media_engine.extractors.objects",
            "media_engine.extractors.objects_qwen",
            "media_engine.extractors.clip",
            "media_engine.extractors.transcribe",
        ]
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Module '{module}' not importable: {e}")
