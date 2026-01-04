# Polybos Media Engine

AI-powered video metadata extraction API for small TV stations and content creators. Provides a "file in → JSON out" API that extracts metadata, transcripts, faces, scenes, objects, CLIP embeddings, OCR text, and camera motion from video files.

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
git clone https://github.com/your-org/polybos-media-engine.git
cd polybos-media-engine

# Start the server
docker compose up -d

# Test
curl http://localhost:8000/health
```

Mount your media folder:
```bash
MEDIA_PATH=/path/to/videos docker compose up -d
```

For NVIDIA GPU support:
```bash
docker compose --profile gpu up -d
```

### Local Installation

```bash
# Mac Apple Silicon
pip install -e ".[mlx]"

# NVIDIA GPU
pip install -e ".[cuda]"

# CPU only
pip install -e ".[cpu]"

# Run server
uvicorn polybos_engine.main:app --reload --port 8000
```

**Requirements**: Python 3.12+, ffmpeg

## API Usage

### Extract metadata from a video

```bash
curl -X POST http://localhost:8000/extract \
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
| **DJI** | Mavic, Air, Mini, Pocket, Osmo | GPS from SRT, color profiles |
| **Sony** | PXW, FX, Alpha, ZV series | XML sidecar, S-Log, GPS |
| **Canon** | Cinema EOS, EOS R | XML sidecar |
| **Apple** | iPhone, iPad | QuickTime metadata, GPS |
| **Blackmagic** | Pocket Cinema Camera | ProApps metadata |
| **FFmpeg** | OBS, Handbrake, etc. | Encoder detection |

Adding new manufacturers is easy - create a module in `polybos_engine/extractors/metadata/`.

## Architecture

```
polybos_engine/
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
ruff check polybos_engine/

# Type check
pyright polybos_engine/

# Test
export TEST_VIDEO_PATH=/path/to/test.mp4
pytest tests/
```

## Contributing

Contributions welcome! To add support for a new camera manufacturer:

1. Create `polybos_engine/extractors/metadata/yourmanufacturer.py`
2. Implement `detect()` and `extract()` methods
3. Register with `register_extractor("name", YourExtractor())`
4. Import in `metadata/__init__.py`

See existing modules for examples.

## License

MIT
