# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Media Engine is an open-source (MIT), AI-powered video extraction API designed for small TV stations and content creators. It provides a "file in → JSON out" API that extracts metadata, transcripts, faces, scenes, objects, CLIP embeddings, OCR text, and device/shot type from video files.

**Business model**: Open-source backend with a commercial closed-source frontend (SvelteKit).

## Requirements

- **Python 3.12+** (uses modern typing features including `StrEnum`, `type` aliases, union syntax `X | None`)
- ffmpeg/ffprobe for video processing
- Platform-specific ML backends (MLX for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- Optional: pyannote-audio for speaker diarization (requires HuggingFace token)

## Development Commands

```bash
# Install dependencies (choose one based on platform)
pip install -e ".[mlx]"     # Mac Apple Silicon
pip install -e ".[cuda]"    # NVIDIA GPU
pip install -e ".[cpu]"     # CPU fallback
pip install -e ".[dev]"     # Development tools

# Run development server
uvicorn media_engine.main:app --reload --port 8001

# Linting and type checking
ruff check media_engine/              # Lint
ruff check media_engine/ --fix        # Lint and auto-fix
pyright media_engine/                 # Type check (strict)

# Run tests (set TEST_VIDEO_PATH first)
export TEST_VIDEO_PATH=/path/to/test.mp4
pytest tests/
pytest tests/test_metadata.py -v          # Single test file
pytest -m "not slow"                       # Skip slow tests

# Run demo (starts both engine and demo server)
./demo/run.sh start       # Start both servers
./demo/run.sh stop        # Stop both servers
./demo/run.sh status      # Check status
```

## Demo

The demo frontend requires two servers:
- **Engine** (port 8001): The main extraction API
- **Demo server** (port 8002): File browsing and video streaming

```bash
# Start both servers
./demo/run.sh start

# Or manually:
python3.12 -m uvicorn media_engine.main:app --port 8001
python3.12 demo/server.py
```

Then open http://localhost:8002 in your browser.

## API Endpoints

```
POST /batch        # Batch extraction (recommended, memory-efficient)
GET  /batch/{id}   # Get batch status and results
POST /extract      # Synchronous extraction (single file)
GET  /health       # Health check
GET  /extractors   # List available extractors
GET  /hardware     # Get hardware capabilities and auto-selected models
```

### Batch Request Example (Recommended)

```bash
curl -X POST http://localhost:8001/batch \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["/path/to/video.mp4"],
    "enable_metadata": true,
    "enable_transcript": false,
    "enable_scenes": false,
    "enable_faces": false,
    "enable_objects": false,
    "enable_clip": false,
    "enable_ocr": false,
    "enable_motion": false
  }'
```

**Note:** Telemetry (GPS/flight path) is always extracted automatically when available.

## Architecture

### File Structure

```
media_engine/
├── __init__.py          # Version
├── main.py              # FastAPI app with /batch, /extract, /health
├── config.py            # Settings and platform detection
├── schemas.py           # Pydantic request/response models
└── extractors/
    ├── __init__.py      # Exports all extractors
    ├── metadata/        # Modular per-manufacturer metadata extraction
    │   ├── __init__.py  # Main entry point
    │   ├── base.py      # Common utilities (ffprobe, GPS parsing)
    │   ├── registry.py  # Extractor registration and detection
    │   ├── dji.py       # DJI drones and cameras
    │   ├── sony.py      # Sony cameras with XML sidecar
    │   ├── canon.py     # Canon cameras
    │   ├── apple.py     # iPhone/iPad
    │   ├── blackmagic.py # Blackmagic cameras
    │   ├── ffmpeg.py    # FFmpeg-encoded files (OBS, etc.)
    │   └── generic.py   # Fallback for unknown devices
    ├── telemetry.py     # GPS/flight path from SRT sidecars
    ├── transcribe.py    # Whisper with MLX/CUDA/CPU backends
    ├── faces.py         # DeepFace + Facenet for detection and embeddings
    ├── scenes.py        # PySceneDetect ContentDetector
    ├── objects.py       # YOLO object detection
    ├── objects_qwen.py  # Qwen VLM for scene descriptions
    ├── clip.py          # OpenCLIP/MLX-CLIP embeddings
    ├── ocr.py           # PaddleOCR text extraction
    ├── motion.py        # Camera motion analysis (optical flow)
    └── shot_type.py     # CLIP-based shot classification

demo/
├── index.html           # Demo frontend (single-page app)
├── server.py            # Demo server (file browsing, video streaming)
└── run.sh               # Start/stop script for both servers
```

### Platform Detection Pattern

All AI modules use a backend abstraction for cross-platform support:

```python
# In config.py
class DeviceType(StrEnum):
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"

def is_apple_silicon() -> bool
def has_cuda() -> bool
def get_device() -> DeviceType

# Each extractor implements backend selection:
if is_apple_silicon():
    # Use MLX-optimized model
elif has_cuda():
    # Use CUDA-optimized model
else:
    # Use CPU fallback
```

## Extractors

| Extractor | Enable Flag | Output |
|-----------|-------------|--------|
| metadata | enable_metadata | duration, resolution, codec, fps, device, GPS |
| telemetry | (always runs) | GPS/flight path from drone sidecar files |
| transcript | enable_transcript | segments with timestamps, language detection, speaker diarization |
| scenes | enable_scenes | scene boundaries with start/end times |
| faces | enable_faces | bounding boxes, embeddings, unique count estimate |
| objects | enable_objects | labels, bounding boxes, summary counts (YOLO or Qwen) |
| clip | enable_clip | per-scene embeddings for similarity search |
| ocr | enable_ocr | detected text with bounding boxes |
| motion | enable_motion | camera motion analysis (pan, tilt, zoom, handheld) |
| shot_type | (part of metadata) | aerial, interview, b-roll, studio, etc. |

## Configuration

Settings are stored in `~/.config/polybos/config.json`. The frontend can read/write this file.

```json
{
  "api_version": "1.0",
  "log_level": "INFO",
  "whisper_model": "large-v3",
  "fallback_language": "en",
  "hf_token": null,
  "diarization_model": "pyannote/speaker-diarization-3.1",
  "face_sample_fps": 1.0,
  "object_sample_fps": 2.0,
  "min_face_size": 80,
  "ocr_languages": ["en", "no", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "fi", "pl"],
  "temp_dir": "/tmp/polybos"
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `hf_token` | HuggingFace token for pyannote speaker diarization | null (diarization skipped) |
| `whisper_model` | Whisper model for transcription | large-v3 |
| `diarization_model` | Pyannote model for speaker diarization | pyannote/speaker-diarization-3.1 |
| `ocr_languages` | OCR languages (see https://www.jaided.ai/easyocr/) | Latin languages |

**Notes**:
- Pyannote models are gated. Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1 before using.
- For CJK OCR, add `ch_sim`, `ja`, `ko` to `ocr_languages`.

## Key Implementation Details

- **Whisper backends**: mlx-whisper (Mac), faster-whisper (CUDA), openai-whisper (CPU)
- **Speaker diarization**: pyannote-audio assigns speaker IDs to transcript segments (requires HF token)
- **Language fallback**: If detection confidence <0.7 on clips <15s, uses fallback_language
- **Face filtering**: Skips faces <80px or low confidence; clusters embeddings to estimate unique count
- **Device detection**: Checks metadata tags and XML sidecars for device info (DJI drones, Sony cameras, etc.)
- **Shot type**: CLIP zero-shot classification against predefined labels
- **Scene-aware sampling**: CLIP and OCR use scene boundaries when available

## Type System

The codebase uses strict typing with `pyright`. Key enums in `schemas.py`:

```python
class MediaDeviceType(StrEnum):
    DRONE = "drone"
    CAMERA = "camera"
    PHONE = "phone"
    ACTION_CAMERA = "action_camera"
    UNKNOWN = "unknown"

class DetectionMethod(StrEnum):
    METADATA = "metadata"
    XML_SIDECAR = "xml_sidecar"
    CLIP = "clip"
```

Shot types in `extractors/shot_type.py`:

```python
class ShotTypeLabel(StrEnum):
    AERIAL = "aerial"
    INTERVIEW = "interview"
    B_ROLL = "b-roll"
    STUDIO = "studio"
    # ... etc
```

Using `StrEnum` ensures JSON serialization works seamlessly with Pydantic while providing type safety.

## Testing

Tests require video files set via environment variables:

```bash
export TEST_VIDEO_PATH=/path/to/any_video.mp4
export SHORT_VIDEO_PATH=/path/to/short_clip.mp4  # Optional, for quick tests

pytest tests/                    # All tests
pytest tests/ -m "not slow"      # Skip slow AI tests
pytest tests/test_api.py -v      # API tests only
```

### Stress Test

The stress test (`tests/stress_test.py`) runs the engine repeatedly with various extractor combinations to verify stability under sustained load. It monitors memory usage and validates results.

```bash
# Requires test videos in test_data/video/
# Engine must be running on localhost:8001

# Run with defaults (10 iterations, random extractor configs)
python tests/stress_test.py

# Run for 50 iterations
python tests/stress_test.py --iterations 50

# Run for 1 hour
python tests/stress_test.py --duration 3600

# Thorough mode: test every file with every config
python tests/stress_test.py --thorough

# Heavy mode: larger files, all extractors including Qwen
python tests/stress_test.py --heavy

# Combine modes
python tests/stress_test.py --heavy --duration 7200
```

The stress test validates:
- All enabled extractors produce output
- Metadata has required fields (duration, resolution, fps)
- No memory leaks (compares first/second half memory averages)
- Files without audio don't trigger transcript warnings

## CI/CD

### Pull Requests

GitHub Actions automatically runs on every PR:
- **Lint**: ruff check, ruff format, pyright
- **Test**: pytest (unit tests without video files)
- **Build**: Verify package builds correctly

### Releasing to PyPI

Releases are triggered by git tags. The version is automatically derived from the tag using `hatch-vcs`.

```bash
# Create and push a tag
git tag v0.2.0
git push origin v0.2.0
```

This will:
1. Run lint checks
2. Build the package with version `0.2.0`
3. Publish to PyPI (requires trusted publishing configured)
4. Create a GitHub release with auto-generated notes

### PyPI Trusted Publishing Setup

Before the first release, configure trusted publishing at PyPI:
1. Go to https://pypi.org/manage/project/media-engine/settings/publishing/
2. Add a new publisher:
   - Owner: `thetrainroom`
   - Repository: `media-engine`
   - Workflow: `release.yml`
   - Environment: `pypi`
