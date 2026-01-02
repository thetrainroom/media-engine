# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polybos Media Engine is an open-source (MIT), AI-powered video extraction API designed for small TV stations and content creators. It provides a "file in → JSON out" API that extracts metadata, transcripts, faces, scenes, objects, CLIP embeddings, OCR text, and device/shot type from video files.

**Business model**: Open-source backend with a commercial closed-source frontend (SvelteKit).

## Development Commands

```bash
# Install dependencies (choose one based on platform)
pip install -e ".[mlx]"     # Mac Apple Silicon
pip install -e ".[cuda]"    # NVIDIA GPU
pip install -e ".[cpu]"     # CPU fallback
pip install -e ".[dev]"     # Development tools

# Run development server
uvicorn polybos_engine.main:app --reload

# Run tests (set TEST_VIDEO_PATH first)
export TEST_VIDEO_PATH=/path/to/test.mp4
pytest tests/
pytest tests/test_metadata.py -v          # Single test file
pytest -m "not slow"                       # Skip slow tests

# Build and run with Docker
docker build -t polybos/media-engine .
docker-compose up
```

## API Endpoints

```
POST /extract      # Main extraction endpoint
GET  /health       # Health check
GET  /extractors   # List available extractors
```

### Extract Request Example

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/path/to/video.mp4",
    "skip_transcript": false,
    "skip_faces": false,
    "skip_scenes": false,
    "skip_objects": false,
    "skip_clip": false,
    "skip_ocr": false,
    "whisper_model": "large-v3",
    "face_sample_fps": 1.0,
    "object_sample_fps": 2.0
  }'
```

## Architecture

### File Structure

```
polybos_engine/
├── __init__.py          # Version
├── main.py              # FastAPI app with /extract and /health
├── config.py            # Settings and platform detection
├── schemas.py           # Pydantic request/response models
└── extractors/
    ├── __init__.py      # Exports all extractors
    ├── metadata.py      # ffprobe wrapper (duration, resolution, codec, GPS, device)
    ├── transcribe.py    # Whisper with MLX/CUDA/CPU backends
    ├── faces.py         # DeepFace + Facenet for detection and embeddings
    ├── scenes.py        # PySceneDetect ContentDetector
    ├── objects.py       # YOLO object detection
    ├── clip.py          # OpenCLIP/MLX-CLIP embeddings
    ├── ocr.py           # PaddleOCR text extraction
    └── shot_type.py     # CLIP-based shot classification
```

### Platform Detection Pattern

All AI modules use a backend abstraction for cross-platform support:

```python
# In config.py
def is_apple_silicon() -> bool
def has_cuda() -> bool
def get_device() -> str  # Returns "mps", "cuda", or "cpu"

# Each extractor implements backend selection:
if is_apple_silicon():
    # Use MLX-optimized model
elif has_cuda():
    # Use CUDA-optimized model
else:
    # Use CPU fallback
```

## Extractors

| Extractor | Skip Flag | Output |
|-----------|-----------|--------|
| metadata | (always runs) | duration, resolution, codec, fps, device, GPS |
| transcript | skip_transcript | segments with timestamps, language detection |
| scenes | skip_scenes | scene boundaries with start/end times |
| faces | skip_faces | bounding boxes, embeddings, unique count estimate |
| objects | skip_objects | labels, bounding boxes, summary counts |
| clip | skip_clip | per-scene embeddings for similarity search |
| ocr | skip_ocr | detected text with bounding boxes |
| shot_type | (uses skip_clip) | aerial, interview, b-roll, studio, etc. |

## Key Implementation Details

- **Whisper backends**: mlx-whisper (Mac), faster-whisper (CUDA), openai-whisper (CPU)
- **Language fallback**: If detection confidence <0.7 on clips <15s, uses fallback_language
- **Face filtering**: Skips faces <80px or low confidence; clusters embeddings to estimate unique count
- **Device detection**: Checks metadata for known drone manufacturers (DJI, Parrot, etc.)
- **Shot type**: CLIP zero-shot classification against predefined labels
- **Scene-aware sampling**: CLIP and OCR use scene boundaries when available

## Testing

Tests require video files set via environment variables:

```bash
export TEST_VIDEO_PATH=/path/to/any_video.mp4
export SHORT_VIDEO_PATH=/path/to/short_clip.mp4  # Optional, for quick tests

pytest tests/                    # All tests
pytest tests/ -m "not slow"      # Skip slow AI tests
pytest tests/test_api.py -v      # API tests only
```
