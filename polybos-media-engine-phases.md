# Polybos Media Engine — Development Phases

## Overview

This document outlines the phased development approach for `polybos-media-engine`, the open-source video extraction API.

**Repository:** `github.com/polybos/media-engine` (public, MIT license)

**Philosophy:** Each phase adds one extractor. Keep it simple, no intelligence — just "file in → JSON out". The commercial product (separate repo) handles sorting, combining, searching, and UI.

---

## Phase 1: Core Setup + Metadata + Transcription

### Goal
Minimal working HTTP API that extracts metadata and transcript from a video file.

### Endpoints

```
POST /extract
  Body: { "file": "/path/to/video.mp4" }
  Response: JSON with metadata + transcript

GET /health
  Response: { "status": "ok", "version": "0.1.0" }
```

### Output Schema

```json
{
  "file": "/path/to/video.mp4",
  "filename": "video.mp4",
  "extracted_at": "2025-01-02T14:30:00Z",
  "extraction_time_seconds": 45.2,
  "version": "0.1.0",
  
  "metadata": {
    "duration": 5765.4,
    "resolution": {
      "width": 1920,
      "height": 1080
    },
    "codec": {
      "video": "h264",
      "audio": "aac"
    },
    "fps": 25.0,
    "bitrate": 8500000,
    "file_size": 4823451234,
    "created_at": "2024-03-15T10:00:00Z",
    "device": {
      "make": "Sony",
      "model": "PXW-Z150",
      "software": null
    },
    "gps": {
      "latitude": 59.7441,
      "longitude": 10.2045,
      "altitude": 125.0
    }
  },
  
  "transcript": {
    "language": "de",
    "confidence": 0.91,
    "duration": 5765.4,
    "hints_used": {
      "language_hints": ["de", "gsw"],
      "context_hint": "Swiss German dialect, informal conversation",
      "fallback_applied": false
    },
    "segments": [
      {
        "start": 0.0,
        "end": 4.5,
        "text": "Grüezi mitenand, willkomme zur Sitzung."
      },
      {
        "start": 4.5,
        "end": 12.3,
        "text": "Hüt wämer über s'Budget für s'nächst Johr rede."
      }
    ]
  }
}
```

### File Structure

```
polybos-media-engine/
├── README.md
├── LICENSE                  # MIT
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml       # For easy local testing
│
├── polybos_engine/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings (model paths, etc.)
│   ├── schemas.py           # Pydantic models for request/response
│   │
│   └── extractors/
│       ├── __init__.py
│       ├── metadata.py      # ffprobe wrapper
│       └── transcribe.py    # Whisper wrapper
│
└── tests/
    ├── __init__.py
    ├── test_metadata.py
    └── test_transcribe.py
```

### Implementation Details

#### metadata.py
- Use `ffmpeg-python` or subprocess to call `ffprobe`
- Extract: duration, resolution, codec, fps, bitrate, file_size
- Extract device info from metadata tags (make, model, software)
- Extract GPS from metadata if present
- Return structured dict

#### transcribe.py
- Detect platform (Apple Silicon vs CUDA vs CPU)
- Use appropriate Whisper backend:
  - Mac: `mlx-whisper`
  - NVIDIA: `faster-whisper`
  - CPU fallback: `openai-whisper`
- Extract audio first: `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 temp.wav`
- Return segments with timestamps

**Language handling:**
```python
def transcribe(audio_path: str, options: dict) -> dict:
    # 1. If language forced, use it
    if options.get("language"):
        language = options["language"]
    
    # 2. Otherwise auto-detect
    else:
        detected = model.detect_language(audio_path)
        
        # 3. If low confidence + short clip, use fallback
        if detected.confidence < 0.7 and audio_duration < 15:
            language = options.get("fallback_language", "de")
            fallback_applied = True
        else:
            language = detected.language
    
    # 4. Use context_hint as initial_prompt (helps with dialect/domain)
    result = model.transcribe(
        audio_path,
        language=language,
        initial_prompt=options.get("context_hint")  # "Swiss German dialect..."
    )
    
    return result
```

**Example usage:**
```bash
# Swiss German content with hints
curl -X POST http://localhost:8000/extract \
  -d '{
    "file": "/videos/basel_interview.mp4",
    "language_hints": ["de", "gsw"],
    "context_hint": "Swiss German dialect from Basel region"
  }'

# Multilingual content
curl -X POST http://localhost:8000/extract \
  -d '{
    "file": "/videos/conference.mp4",
    "language_hints": ["en", "de", "fr"],
    "context_hint": "International conference, speakers switch languages"
  }'
```

#### main.py
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Polybos Media Engine")

class ExtractRequest(BaseModel):
    file: str
    skip_transcript: bool = False
    whisper_model: str = "large-v3"
    
    # Language options
    language: str | None = None              # Force language (skip detection)
    fallback_language: str = "de"            # For short clips with low confidence
    language_hints: list[str] = []           # e.g. ["de", "en", "fr", "gsw"]
    context_hint: str | None = None          # e.g. "Swiss German dialect, Basel region"

@app.post("/extract")
async def extract(request: ExtractRequest):
    # 1. Extract metadata
    # 2. Extract transcript (unless skipped)
    # 3. Return combined result
    pass

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
```

### Dependencies

```toml
[project]
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.0.0",
    "ffmpeg-python>=0.2.0",
]

[project.optional-dependencies]
cuda = ["faster-whisper>=0.10.0"]
mlx = ["mlx-whisper>=0.1.0"]
cpu = ["openai-whisper>=20231117"]
```

### Docker

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY . .
RUN pip install -e ".[cpu]"

EXPOSE 8000
CMD ["uvicorn", "polybos_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Test

```bash
# Run server
uvicorn polybos_engine.main:app --reload

# Test extraction
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"file": "/path/to/test_video.mp4"}'
```

### Done When
- [ ] `POST /extract` returns metadata for any video file
- [ ] `POST /extract` returns transcript with timestamps
- [ ] Works on Mac (MLX) and Linux (CUDA/CPU)
- [ ] Docker image builds and runs
- [ ] Basic tests pass

---

## Phase 1b: Face Detection

### Goal
Add face detection to extraction output. Detection only — no recognition, no naming.

### New Output Fields

```json
{
  "...existing fields...",
  
  "faces": {
    "count": 47,
    "unique_estimate": 3,
    "detections": [
      {
        "timestamp": 12.5,
        "bbox": {
          "x": 120,
          "y": 45,
          "width": 80,
          "height": 135
        },
        "confidence": 0.97,
        "embedding": [0.23, -0.12, 0.45, ...]  # 512-dim vector
      }
    ]
  }
}
```

### New Files

```
polybos_engine/
└── extractors/
    └── faces.py         # DeepFace/Facenet wrapper
```

### Implementation Details

#### faces.py
- Use DeepFace with Facenet model (MIT licensed)
- Sample frames at 1-2 fps (not every frame)
- Skip blurry or small faces
- Return bounding boxes + embeddings
- Estimate unique faces by clustering embeddings

#### Frame Sampling
```python
def extract_faces(video_path: str, fps: float = 1.0):
    # Extract frames at specified fps
    # ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg
    
    # For each frame:
    #   - Detect faces
    #   - Filter by size (min 80px) and confidence
    #   - Generate embedding
    #   - Add to results with timestamp
```

### API Change

```python
class ExtractRequest(BaseModel):
    file: str
    skip_transcript: bool = False
    skip_faces: bool = False
    whisper_model: str = "large-v3"
    face_sample_fps: float = 1.0
```

### Done When
- [ ] Face detections included in output
- [ ] Embeddings generated for each face
- [ ] Configurable sample rate
- [ ] Small/blurry faces filtered out

---

## Phase 1c: Scene Detection

### Goal
Detect scene changes/cuts to segment video into logical chunks.

### New Output Fields

```json
{
  "...existing fields...",
  
  "scenes": {
    "count": 47,
    "detections": [
      {
        "index": 0,
        "start": 0.0,
        "end": 15.3,
        "duration": 15.3
      },
      {
        "index": 1,
        "start": 15.3,
        "end": 28.7,
        "duration": 13.4
      }
    ]
  }
}
```

### New Files

```
polybos_engine/
└── extractors/
    └── scenes.py        # PySceneDetect wrapper
```

### Implementation Details

#### scenes.py
- Use PySceneDetect (BSD license)
- ContentDetector for most videos
- Return list of scene boundaries with timestamps

```python
from scenedetect import detect, ContentDetector

def extract_scenes(video_path: str):
    scenes = detect(video_path, ContentDetector())
    return [
        {
            "index": i,
            "start": scene[0].get_seconds(),
            "end": scene[1].get_seconds(),
            "duration": scene[1].get_seconds() - scene[0].get_seconds()
        }
        for i, scene in enumerate(scenes)
    ]
```

### Done When
- [ ] Scene boundaries detected
- [ ] Works with various video types (cuts, fades)
- [ ] Output includes timestamps for each scene

---

## Phase 1d: Object Detection

### Goal
Detect common objects in video frames.

### New Output Fields

```json
{
  "...existing fields...",
  
  "objects": {
    "summary": {
      "person": 234,
      "car": 12,
      "microphone": 45,
      "chair": 89
    },
    "detections": [
      {
        "timestamp": 12.5,
        "label": "person",
        "confidence": 0.92,
        "bbox": { "x": 100, "y": 50, "width": 200, "height": 400 }
      }
    ]
  }
}
```

### New Files

```
polybos_engine/
└── extractors/
    └── objects.py       # RT-DETR or YOLO wrapper
```

### Implementation Details

#### objects.py
- Use RT-DETR (Apache 2.0) or Grounding DINO (Apache 2.0)
- Sample frames (1-2 fps)
- Return detections with labels and bounding boxes
- Aggregate summary counts

### Done When
- [ ] Object detections included in output
- [ ] Summary counts per label
- [ ] Configurable sample rate

---

## Phase 1e: CLIP Embeddings

### Goal
Generate semantic embeddings for video frames, enabling visual similarity search.

### New Output Fields

```json
{
  "...existing fields...",
  
  "embeddings": {
    "model": "ViT-B/32",
    "segments": [
      {
        "start": 0.0,
        "end": 15.3,
        "scene_index": 0,
        "embedding": [0.12, -0.34, 0.56, ...]  # 512-dim vector
      }
    ]
  }
}
```

### New Files

```
polybos_engine/
└── extractors/
    └── clip.py          # CLIP/OpenCLIP wrapper
```

### Implementation Details

#### clip.py
- Use OpenCLIP (MIT license)
- Generate one embedding per scene (from scene detection)
- Or fixed interval if scenes not detected
- Platform detection: MLX-CLIP for Mac, OpenCLIP for NVIDIA

### Done When
- [ ] CLIP embeddings generated per scene
- [ ] Works on Mac (MLX) and Linux (CUDA)
- [ ] Embeddings suitable for similarity search

---

## Phase 1f: OCR (Text Detection)

### Goal
Extract visible text from video frames (lower thirds, signs, graphics).

### New Output Fields

```json
{
  "...existing fields...",
  
  "ocr": {
    "detections": [
      {
        "timestamp": 12.5,
        "text": "BREAKING NEWS",
        "confidence": 0.95,
        "bbox": { "x": 50, "y": 680, "width": 400, "height": 40 }
      },
      {
        "timestamp": 12.5,
        "text": "Mayor Jensen",
        "confidence": 0.91,
        "bbox": { "x": 50, "y": 720, "width": 200, "height": 30 }
      }
    ]
  }
}
```

### New Files

```
polybos_engine/
└── extractors/
    └── ocr.py           # PaddleOCR wrapper
```

### Implementation Details

#### ocr.py
- Use PaddleOCR or docTR (Apache 2.0)
- Sample at scene changes (text usually appears at cuts)
- Filter low-confidence detections
- Deduplicate repeated text across frames

### Done When
- [ ] Text extracted from video frames
- [ ] Bounding boxes for each text region
- [ ] Deduplication of repeated text

---

## Phase 1g: Device & Shot Type Detection

### Goal
Detect what device recorded the video and classify shot type.

### New Output Fields

```json
{
  "...existing fields...",
  
  "metadata": {
    "...existing fields...",
    
    "device": {
      "make": "DJI",
      "model": "Mavic 3",
      "type": "drone",
      "detection_method": "metadata",
      "confidence": 1.0
    },
    
    "shot_type": {
      "primary": "aerial",
      "confidence": 0.94,
      "detection_method": "clip"
    }
  }
}
```

### Implementation Details

#### Enhanced metadata.py
- Check make/model against known drone manufacturers
- If no metadata, use CLIP to classify:
  - "aerial drone footage"
  - "handheld camera"
  - "tripod static shot"
  - "interview setup"
  - "studio footage"
  - etc.

### Done When
- [ ] Device type detected from metadata
- [ ] Shot type classified via CLIP
- [ ] Drone footage correctly identified

---

## Phase 2: Selective Extraction

### Goal
Allow client to request only specific extractors, and support async processing for long videos.

### API Changes

```python
class ExtractRequest(BaseModel):
    file: str
    extractors: list[str] = ["metadata", "transcript"]  # Select which to run
    
    # Extractor-specific options
    whisper_model: str = "large-v3"
    face_sample_fps: float = 1.0
    object_sample_fps: float = 2.0
    
    # Processing options
    async_mode: bool = False  # Return job ID, poll for result
```

### New Endpoints

```
POST /extract
  Body: { "file": "...", "extractors": ["metadata", "transcript", "faces"] }
  Response: JSON result (sync) or { "job_id": "..." } (async)

GET /extract/{job_id}
  Response: { "status": "processing", "progress": 0.45 } or result

GET /extractors
  Response: List of available extractors with descriptions
```

### Done When
- [ ] Client can select specific extractors
- [ ] Async mode for long-running extractions
- [ ] Progress reporting

---

## Phase 3: Thumbnails & Previews

### Goal
Generate visual assets alongside data extraction.

### New Output Fields

```json
{
  "...existing fields...",
  
  "thumbnails": {
    "grid": "/output/video_thumb.jpg",
    "sprite": "/output/video_sprite.jpg",
    "scenes": [
      "/output/scene_000.jpg",
      "/output/scene_001.jpg"
    ]
  }
}
```

### API Changes

```python
class ExtractRequest(BaseModel):
    # ... existing fields ...
    generate_thumbnails: bool = False
    thumbnail_output_dir: str = "/output"
```

### Implementation
- Grid thumbnail (single image, first frame or middle)
- Sprite sheet (10x10 grid, 1 frame per second)
- Scene thumbnails (one per detected scene)

### Done When
- [ ] Thumbnails generated on request
- [ ] Sprite sheet for hover scrubbing
- [ ] Scene thumbnails

---

## Testing Strategy

### Test Files
Create/collect a set of test videos:
- Short clip (10 sec) — fast iteration
- Interview (static, talking head)
- B-roll (many scene changes)
- Drone footage (DJI metadata)
- Phone footage (vertical)
- Old footage (no metadata)

### Test Cases Per Extractor
- Happy path with good input
- Missing/corrupt metadata
- No speech (silent video)
- No faces in video
- Very long video (performance)
- Unsupported codec

---

## API Versioning

```
/v1/extract     # Current
/v1/health

# Future
/v2/extract     # Breaking changes
```

Include version in response:
```json
{
  "api_version": "1.0",
  "engine_version": "0.3.0",
  ...
}
```

---

## Summary: Phase Order

| Phase | Feature | Complexity | Time Estimate |
|-------|---------|------------|---------------|
| 1 | Metadata + Transcript | Low | 1-2 days |
| 1b | Face Detection | Medium | 1-2 days |
| 1c | Scene Detection | Low | 0.5-1 day |
| 1d | Object Detection | Medium | 1-2 days |
| 1e | CLIP Embeddings | Medium | 1 day |
| 1f | OCR | Low | 1 day |
| 1g | Device/Shot Type | Low | 0.5 day |
| 2 | Selective + Async | Medium | 1-2 days |
| 3 | Thumbnails | Low | 1 day |

**Total: ~10-14 days for complete extraction API**

Each phase is independently useful and deployable.
