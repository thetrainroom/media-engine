# Media Engine

AI-powered video metadata extraction API for small TV stations and content creators. Provides a "file in → JSON out" API that extracts metadata, transcripts, faces, scenes, objects, CLIP embeddings, OCR text, and camera motion from video files.

## Installation

```bash
# Apple Silicon Mac
pip install media-engine[mlx]

# NVIDIA GPU
pip install media-engine[cuda]

# CPU only
pip install media-engine[cpu]

# Start the server
meng-server
```

**Requirements**: Python 3.12+, ffmpeg

## Features

- **Metadata extraction** - Duration, resolution, codec, GPS, device info (modular per-manufacturer)
- **Transcription** - Whisper with speaker diarization
- **Face detection** - DeepFace with embedding clustering
- **Scene detection** - PySceneDetect content-aware boundaries
- **Object detection** - YOLO or Qwen VLM
- **CLIP embeddings** - Per-scene similarity search
- **OCR** - PaddleOCR text extraction
- **Motion analysis** - Camera pan/tilt/zoom detection
- **Shot type** - Aerial, interview, b-roll classification

## Quick Start

### Docker (Recommended)

```bash
# Clone and run
git clone https://github.com/thetrainroom/media-engine.git
cd media-engine

# Start the server
docker compose up -d

# Test
curl http://localhost:8001/health
```

Mount your media folder:
```bash
MEDIA_PATH=/path/to/videos docker compose up -d
```

For NVIDIA GPU support (uses `Dockerfile.cuda`):
```bash
docker compose --profile gpu up -d
```

Or build manually:
```bash
docker build -f Dockerfile.cuda -t media-engine-gpu .
docker run -p 8001:8001 --gpus all -v /path/to/media:/media media-engine-gpu
```

### Apple Silicon (Recommended: Native)

Docker on macOS runs in a Linux VM without Metal/MPS access. For GPU acceleration on Apple Silicon, run natively:

```bash
pip install media-engine[mlx]
meng-server
```

A `Dockerfile.mlx` is provided for consistency, but will use CPU in Docker:
```bash
docker compose --profile mlx up -d
```

### Development Installation

```bash
# Mac Apple Silicon
pip install -e ".[mlx]"

# NVIDIA GPU
pip install -e ".[cuda]"

# CPU only
pip install -e ".[cpu]"

# Run server with hot reload
uvicorn media_engine.main:app --reload --port 8001
```

## API Usage

### Extract metadata from a video

```bash
curl -X POST http://localhost:8001/extract \
  -H "Content-Type: application/json" \
  -d '{
    "file": "/media/video.mp4",
    "enable_metadata": true,
    "enable_transcript": true,
    "enable_faces": false,
    "enable_scenes": true,
    "enable_objects": false,
    "enable_clip": false,
    "enable_ocr": false,
    "enable_motion": false
  }'
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/extract` | POST | Extract features from video |
| `/extractors` | GET | List available extractors |

## Supported Devices

The metadata extractor automatically detects camera/device type:

| Manufacturer | Models | Features |
|--------------|--------|----------|
| **DJI** | Mavic, Air, Mini, Pocket, Osmo, Action | GPS from SRT, color profiles |
| **Sony** | PXW, FX, Alpha, ZV series | XML sidecar, S-Log, GPS |
| **Canon** | Cinema EOS, EOS R | XML sidecar |
| **Apple** | iPhone, iPad | QuickTime metadata, GPS |
| **Blackmagic** | Pocket, URSA, BRAW | ProApps metadata, BRAW detection |
| **RED** | DSMC2, V-RAPTOR, KOMODO | R3D native support |
| **ARRI** | ALEXA, ALEXA Mini, AMIRA | ARRIRAW detection |
| **Insta360** | X3, X4, ONE RS, GO 3 | 360 video detection |
| **FFmpeg** | OBS, Handbrake, etc. | Encoder detection |

Adding new manufacturers is easy - create a module in `media_engine/extractors/metadata/`.

## Architecture

```
media_engine/
├── main.py              # FastAPI app
├── config.py            # Settings and platform detection
├── schemas.py           # Pydantic models
└── extractors/
    ├── metadata/        # Modular per-manufacturer
    │   ├── dji.py
    │   ├── sony.py
    │   ├── apple.py
    │   └── ...
    ├── transcribe.py    # Whisper (MLX/CUDA/CPU)
    ├── faces.py         # DeepFace + embeddings
    ├── scenes.py        # PySceneDetect
    ├── objects.py       # YOLO
    ├── objects_qwen.py  # Qwen VLM
    ├── clip.py          # CLIP embeddings
    ├── ocr.py           # PaddleOCR
    └── motion.py        # Optical flow analysis
```

## Configuration

Settings are stored in `~/.config/polybos/config.json`:

```json
{
  "whisper_model": "large-v3",
  "fallback_language": "en",
  "hf_token": null,
  "face_sample_fps": 1.0,
  "object_sample_fps": 2.0,
  "ocr_languages": ["en", "no", "de", "fr", "es"]
}
```

Set `hf_token` to enable speaker diarization (requires accepting license at [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1)).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check media_engine/

# Type check
pyright media_engine/

# Test
export TEST_VIDEO_PATH=/path/to/test.mp4
pytest tests/
```

## Contributing

Contributions welcome! To add support for a new camera manufacturer:

1. Create `media_engine/extractors/metadata/yourmanufacturer.py`
2. Implement `detect()` and `extract()` methods
3. Register with `register_extractor("name", YourExtractor())`
4. Import in `metadata/__init__.py`

See existing modules for examples.

## License

MIT
