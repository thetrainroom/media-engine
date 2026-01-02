# Polybos Media Engine

AI-powered video extraction API for small TV stations and content creators.

## Quick Start

```bash
# Install (Mac Apple Silicon)
pip install -e ".[mlx]"

# Run server
uvicorn polybos_engine.main:app --reload

# Test
curl http://localhost:8000/health
```

## Features

- Metadata extraction (ffprobe)
- Transcription (Whisper)
- Face detection (DeepFace)
- Scene detection (PySceneDetect)
- Object detection (YOLO)
- CLIP embeddings
- OCR (PaddleOCR)
- Shot type classification

## License

MIT
