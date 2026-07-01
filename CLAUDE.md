# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Media Engine is an open-source (MIT), AI-powered video extraction API designed for small TV stations and content creators. It provides a "file in → JSON out" API that extracts metadata, transcripts (with speaker diarization), faces, scenes, objects, VLM scene descriptions, CLIP embeddings, OCR text, camera motion, and GPS telemetry from video, image, and audio files.

**Business model**: Open-source backend with a commercial closed-source frontend (SvelteKit).

See **API.md** for the complete API reference including the exact JSON schema of every extractor result.

## Requirements

- **Python 3.12+** (uses modern typing features including `StrEnum`, `type` aliases, union syntax `X | None`)
- ffmpeg/ffprobe for video processing
- Platform-specific ML backends (MLX for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- Optional: pyannote-audio for speaker diarization (requires HuggingFace token, `make install-diarization`)
- Optional: Qwen VLM for scene descriptions (`pip install -e ".[qwen]"`)

## Development Commands

```bash
# Install dependencies (choose one based on platform)
pip install -e ".[mlx]"     # Mac Apple Silicon
pip install -e ".[cuda]"    # NVIDIA GPU
pip install -e ".[cpu]"     # CPU fallback
pip install -e ".[qwen]"    # Qwen VLM extras (visual descriptions)
pip install -e ".[dev]"     # Development tools
# Or via Makefile: make install-mlx / install-cuda / install-cpu / install-qwen / install-dev / install-diarization

# Run development server
uvicorn media_engine.main:app --reload --port 8001

# Linting, formatting, and type checking (all run in CI)
ruff check media_engine/              # Lint
ruff check media_engine/ --fix        # Lint and auto-fix
black media_engine/                   # Format (line length 180)
pyright media_engine/                 # Type check (basic mode)

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

### CLI Tools

Each extractor also has a standalone CLI entry point (defined in `pyproject.toml`, implemented in `cli/`): `meng-server`, `meng-metadata`, `meng-transcript`, `meng-faces`, `meng-scenes`, `meng-objects`, `meng-ocr`, `meng-clip`, `meng-motion`, `meng-telemetry`.

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
POST   /batch                    # Create batch extraction job (queued if one is running)
GET    /batch/{id}               # Get batch status and results (?status_only=true for polling)
DELETE /batch/{id}               # Delete a batch job and free memory
GET    /health                   # Health check
GET    /logs                     # Recent log entries (from /tmp/media_engine.log)
GET    /hardware                 # Hardware capabilities and auto-selected models
GET    /settings                 # Get settings (hf_token masked)
PUT    /settings                 # Update settings (persists to config file)
POST   /check-models             # Start background check of which models can load
GET    /check-models/{check_id}  # Poll model check results
GET    /extractors               # List available extractors
POST   /encode_text              # Encode text query to CLIP embedding (for search)
POST   /shutdown                 # Graceful shutdown (unloads all models)
```

There is no synchronous `/extract` endpoint anymore — all extraction goes through `/batch`.

### Batch Request Example

```bash
curl -X POST http://localhost:8001/batch \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["/path/to/video.mp4"],
    "enable_metadata": true,
    "enable_vad": false,
    "enable_scenes": false,
    "enable_transcript": false,
    "enable_faces": false,
    "enable_objects": false,
    "enable_visual": false,
    "enable_clip": false,
    "enable_ocr": false,
    "enable_motion": false
  }'
```

**Note:** Telemetry (GPS/flight path) is always extracted automatically when available. Model selection is configured via `PUT /settings`, not per-request. See API.md for all request fields (language hints, Qwen contexts/strategy, LUT path, etc.) and response schemas.

## Architecture

### File Structure

```
media_engine/
├── __init__.py          # Version
├── main.py              # Entry point: env setup, logging, creates app (uvicorn target)
├── app.py               # FastAPI app factory (routers, CORS, cleanup)
├── cli.py               # meng-server entry point
├── config.py            # Settings, platform detection, VRAM-based auto model selection
├── schemas.py           # Pydantic response models, media type detection
├── routers/
│   ├── batch.py         # POST/GET/DELETE /batch
│   ├── health.py        # /health, /logs, /hardware
│   ├── settings.py      # GET/PUT /settings
│   ├── models.py        # /check-models (verify models can load)
│   └── utils.py         # /extractors, /encode_text, /shutdown
├── batch/
│   ├── models.py        # BatchRequest, BatchJobStatus, JobProgress (with ETA fields)
│   ├── processor.py     # run_batch_job: the extractor-first pipeline
│   ├── queue.py         # Single-batch queue (one batch runs at a time)
│   ├── state.py         # Shared job state and locks
│   └── timing.py        # Historical timing data for ETA predictions
├── extractors/
│   ├── __init__.py      # Exports all extractors
│   ├── metadata/        # Modular per-manufacturer metadata extraction
│   │   ├── __init__.py  # Main entry point
│   │   ├── base.py      # Common utilities (ffprobe pool, GPS parsing, SEI timecodes)
│   │   ├── registry.py  # Extractor registration and detection
│   │   ├── dji.py       # DJI drones and cameras
│   │   ├── sony.py      # Sony cameras with XML sidecar
│   │   ├── canon.py     # Canon cameras
│   │   ├── apple.py     # iPhone/iPad
│   │   ├── blackmagic.py # Blackmagic cameras
│   │   ├── arri.py      # ARRI cinema cameras
│   │   ├── red.py       # RED cinema cameras
│   │   ├── gopro.py     # GoPro action cameras
│   │   ├── camera_360.py # 360 cameras (Insta360, etc.)
│   │   ├── tesla.py     # Tesla dashcam
│   │   ├── dv.py        # DV/HDV tape formats
│   │   ├── avchd.py     # AVCHD spanned recordings
│   │   ├── avchd_gps.py # AVCHD GPS + Sony MDPM SEI timecodes
│   │   ├── ffmpeg.py    # FFmpeg-encoded files (OBS, etc.)
│   │   └── generic.py   # Fallback for unknown devices
│   ├── telemetry.py     # GPS/flight path from SRT sidecars and embedded subtitles
│   ├── vad.py           # Voice activity detection (pure-Python energy+ZCR, no C deps)
│   ├── transcribe.py    # Whisper (MLX/CUDA/CPU) + pyannote speaker diarization
│   ├── translate.py     # Query translation for CLIP text search
│   ├── faces.py         # DeepFace + Facenet512 for detection and embeddings
│   ├── scenes.py        # PySceneDetect ContentDetector
│   ├── objects.py       # YOLO object detection
│   ├── objects_qwen.py  # Qwen VLM for scene descriptions (Qwen3-VL / Qwen3.5)
│   ├── clip.py          # OpenCLIP/MLX-CLIP embeddings + text query encoding
│   ├── ocr.py           # EasyOCR text extraction
│   ├── motion.py        # Camera motion analysis (optical flow)
│   ├── frames.py        # Frame decoding (shared, single decode per file)
│   ├── frame_buffer.py  # SharedFrameBuffer (reused across visual extractors)
│   └── shot_type.py     # CLIP-based shot classification (currently not wired into pipeline)
└── utils/
    ├── logging.py       # Log setup (/tmp/media_engine.log)
    └── memory.py        # Memory monitoring and clearing

cli/                     # Standalone per-extractor CLI tools (meng-*)
demo/
├── index.html           # Demo frontend (single-page app)
├── server.py            # Demo server (file browsing, video streaming)
└── run.sh               # Start/stop script for both servers
```

### Batch Pipeline (processor.py)

Batches process **extractor-first** for memory efficiency: each model is loaded once, runs over all files, then is unloaded before the next model. Stages in order:

1. **metadata** (parallel ffprobe) — if it fails, the file is skipped by all later stages
2. **telemetry** (always runs, lightweight)
3. **vad** (skipped for images and files without audio)
4. **visual_processing** (per file: motion → scenes → decode frames once into a `SharedFrameBuffer` → objects → faces → OCR → CLIP → release buffer)
5. **visual** (Qwen VLM, separate stage — heavy model with its own frame handling)
6. **transcript** (Whisper, heaviest — only loads if at least one file has audio)

Motion analysis also runs implicitly (when objects/faces/ocr/clip are enabled) to pick adaptive sample timestamps; its result is stored either way. Only one batch runs at a time; additional batches queue (`batch/queue.py`). Timing history (`batch/timing.py`) feeds ETA predictions in progress updates.

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

`config.py` also auto-selects models based on VRAM/free memory when settings are `"auto"` (see "Model Selection" in API.md).

## Extractors

| Extractor | Enable Flag | Result Key | Output |
|-----------|-------------|------------|--------|
| metadata | enable_metadata | `metadata` | duration, resolution, codecs, fps, device, GPS, color space, lens, timecode, keyframes, spanned recording, stereo 3D |
| telemetry | (always runs) | `telemetry` | GPS/flight path points from DJI SRT sidecars or embedded subtitle streams |
| vad | enable_vad | `vad` | audio content classification (no_audio/speech/audio), speech ratio and segments |
| transcript | enable_transcript | `transcript` | segments with timestamps, language detection, speaker diarization + voice embeddings |
| scenes | enable_scenes | `scenes` | scene boundaries with start/end times |
| faces | enable_faces | `faces` | bounding boxes, 512-dim Facenet512 embeddings, face crops (base64), unique count estimate |
| objects | enable_objects | `objects` | YOLO label counts (`summary` only — detections are internal) |
| visual | enable_visual | `visual` | Qwen VLM object summary + natural-language scene descriptions |
| clip | enable_clip | `clip` | embeddings for similarity search (512/768-dim); per-scene by default, fixed-rate via `clip_sample_fps` |
| ocr | enable_ocr | `ocr` | detected text with bounding boxes (EasyOCR) |
| motion | enable_motion | `motion` | camera motion segments (pan, tilt, push/pull, handheld), stability, plus per-segment feature stats (magnitude distribution, direction consistency, jerk, HF/LF energy) versioned via `features_version` |

The exact JSON shape of each result key is documented in **API.md → Extractor Results**.

## Configuration

Settings are stored in `~/.config/polybos/config.json`. The frontend can read/write this file (or use GET/PUT `/settings`).

```json
{
  "api_version": "1.0",
  "log_level": "INFO",
  "whisper_model": "auto",
  "fallback_language": "en",
  "hf_token": null,
  "diarization_model": "pyannote/speaker-diarization-community-1",
  "face_sample_fps": 1.0,
  "object_sample_fps": 2.0,
  "min_face_size": 80,
  "object_detector": "auto",
  "qwen_model": "auto",
  "qwen_strategy": "auto",
  "qwen_frames_per_scene": 1,
  "yolo_model": "auto",
  "clip_model": "auto",
  "ocr_languages": ["en", "no", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "pl"],
  "temp_dir": "/tmp/polybos"
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `hf_token` | HuggingFace token for pyannote speaker diarization | null (diarization skipped) |
| `whisper_model` | "auto", "tiny", "small", "medium", or "large-v3" | auto |
| `diarization_model` | Pyannote model for speaker diarization | pyannote/speaker-diarization-community-1 |
| `object_detector` | "auto", "yolo", or "qwen" | auto |
| `qwen_model` | Qwen VLM model or "auto" (Qwen3-VL-2B/8B, Qwen3.5-27B) | auto |
| `qwen_strategy` | "auto", "single", "context", "batch", "batch_context" | auto |
| `yolo_model` | "auto" or yolov8n/s/m/l/x.pt | auto |
| `clip_model` | "auto", "ViT-B-16", "ViT-B-32", or "ViT-L-14" | auto |
| `ocr_languages` | OCR languages (EasyOCR codes, see https://www.jaided.ai/easyocr/) | Latin languages |

**Notes**:
- `"auto"` model settings resolve at batch time from VRAM/free memory (see `config.py`).
- Pyannote models are gated. Accept the license at https://huggingface.co/pyannote/speaker-diarization-community-1 before using.
- For CJK OCR, add `ch_sim`, `ja`, `ko` to `ocr_languages`. Finnish (`fi`) is not supported by EasyOCR.

## Key Implementation Details

- **Whisper backends**: mlx-whisper (Mac), faster-whisper (CUDA), openai-whisper (CPU)
- **Speaker diarization**: pyannote-audio assigns speaker IDs to transcript segments and extracts 256-dim voice embeddings per speaker (requires HF token; MPS-accelerated on Apple Silicon)
- **VAD**: pure-Python energy + zero-crossing-rate heuristic (no C extensions) — used to classify audio and skip Whisper for silent/ambient clips
- **Language fallback**: If detection confidence <0.7 on clips <15s, uses fallback_language (reported via `hints_used.fallback_applied`)
- **Face filtering**: Skips faces <80px or low confidence; clusters embeddings to estimate unique count; long videos use adaptive batching with early exit once faces stabilize
- **Frame decoding**: One decode pass per file into a `SharedFrameBuffer`, reused by objects/faces/OCR/CLIP
- **Dense CLIP sampling**: opt-in via `clip_sample_fps` (request) or `clip_default_sample_fps` (setting); streams frames through its own ffmpeg fps-filter pipe in bounded chunks — never through the shared buffer. Default per-scene sampling is preserved byte-identically for backward compatibility.
- **Motion features**: extended per-segment/clip-level statistics (api 1.1) are post-processing on the flow series the classifier already computes — classification logic unchanged; `motion_features_enabled=false` omits them; bump `MOTION_FEATURES_VERSION` in `motion.py` when formulas change
- **api_version**: code-owned (`DEFAULT_API_VERSION` in config.py); never persisted to the config file and stale on-disk values are ignored on load
- **Device detection**: Checks metadata tags and XML sidecars for device info (DJI, Sony, Canon, Apple, Blackmagic, ARRI, RED, GoPro, 360 cameras, Tesla dashcam, DV, AVCHD)
- **Media types**: video, image, and audio files supported (by extension, `schemas.py`); images are analyzed as a single frame at t=0 and skip VAD/scenes/motion/transcript
- **Scene-aware sampling**: CLIP uses scene boundaries when available; sampling timestamps adapt to camera motion
- **YOLO weights path**: resolved to `$TORCH_HOME/ultralytics/` (never CWD — read-only inside macOS .app bundles)

## Type System

The codebase uses typing checked with `pyright` (basic mode). Key enums in `schemas.py`:

```python
class MediaDeviceType(StrEnum):
    DRONE = "drone"
    CAMERA = "camera"
    CINEMA_CAMERA = "cinema_camera"
    PHONE = "phone"
    ACTION_CAMERA = "action_camera"
    CAMERA_360 = "360_camera"
    DASHCAM = "dashcam"
    UNKNOWN = "unknown"

class DetectionMethod(StrEnum):
    METADATA = "metadata"
    XML_SIDECAR = "xml_sidecar"
    CLIP = "clip"

class MediaType(StrEnum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    UNKNOWN = "unknown"
```

Motion types in `extractors/motion.py`: `static`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `push_in`, `pull_out`, `handheld`, `complex`.

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
- **Lint**: ruff check, black --check, pyright
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
