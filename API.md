# Polybos Media Engine API Reference

Base URL: `http://localhost:8000`

Interactive documentation available at `/docs` (Swagger UI) when server is running.

---

## Batch Processing (Recommended)

The batch API is memory-efficient - it loads each model once, processes all files, then unloads before the next model.

### POST /batch

Create a new batch extraction job.

**Request Body:**
```json
{
  "files": ["/path/to/video1.mp4", "/path/to/video2.mp4"],
  "enable_metadata": true,
  "enable_vad": false,
  "enable_scenes": false,
  "enable_transcript": false,
  "enable_faces": false,
  "enable_objects": false,
  "enable_visual": false,
  "enable_clip": false,
  "enable_ocr": false,
  "enable_motion": false,
  "language_hints": ["en", "no"],
  "context_hint": "Interview about technology",
  "contexts": {
    "/path/to/video1.mp4": {"location": "Oslo", "person": "John Smith"},
    "/path/to/video2.mp4": {"location": "Bergen", "person": "Jane Doe"}
  },
  "qwen_timestamps": [10.0, 30.0, 60.0]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `files` | string[] | required | List of video file paths |
| `enable_metadata` | bool | true | Extract metadata (duration, resolution, GPS, device) |
| `enable_vad` | bool | false | Voice activity detection |
| `enable_scenes` | bool | false | Scene boundary detection |
| `enable_transcript` | bool | false | Whisper transcription |
| `enable_faces` | bool | false | Face detection with embeddings |
| `enable_objects` | bool | false | Object detection with YOLO (fast, bounding boxes) |
| `enable_visual` | bool | false | Scene descriptions with Qwen2-VL (slower, richer) |
| `enable_clip` | bool | false | CLIP embeddings for similarity search |
| `enable_ocr` | bool | false | Text extraction from frames |
| `enable_motion` | bool | false | Camera motion analysis |
| `contexts` | object | null | Per-file context for Qwen (file path -> context dict) |
| `language_hints` | string[] | null | Hint languages for transcription |
| `context_hint` | string | null | Context hint for Whisper |
| `qwen_timestamps` | float[] | null | Specific timestamps for Qwen analysis |

**Note:** Model selection (whisper_model, yolo_model, qwen_model, clip_model) is configured via `PUT /settings`. This keeps hardware-dependent configuration in one place.

**Note:** Telemetry (GPS/flight path) is always extracted automatically when available. No flag needed - it's lightweight and included in results.

**Response:**
```json
{
  "batch_id": "abc12345"
}
```

---

### GET /batch/{batch_id}

Get batch job status and results.

**Response:**
```json
{
  "batch_id": "abc12345",
  "status": "running",
  "current_extractor": "scenes",
  "progress": {
    "message": "Processing video1.mp4",
    "current": 1,
    "total": 2
  },
  "elapsed_seconds": 45.2,
  "memory_mb": 1024,
  "peak_memory_mb": 2048,
  "extractor_timings": [
    {
      "extractor": "metadata",
      "started_at": "2024-01-15T10:00:00Z",
      "completed_at": "2024-01-15T10:00:05Z",
      "duration_seconds": 5.0,
      "files_processed": 2
    }
  ],
  "files": [
    {
      "file": "/path/to/video1.mp4",
      "filename": "video1.mp4",
      "status": "completed",
      "timings": {"metadata": 2.5, "telemetry": 0.1, "scenes": 30.0},
      "results": {
        "metadata": {...},
        "telemetry": {...},
        "scenes": {...}
      },
      "error": null
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "completed_at": null
}
```

| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `running` | Processing in progress |
| `completed` | All files processed successfully |
| `failed` | Job failed with error |

---

### DELETE /batch/{batch_id}

Delete a batch job and free memory.

**Response:**
```json
{
  "status": "deleted"
}
```

---

## Utility Endpoints

### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "api_version": "1.0"
}
```

---

### GET /hardware

Get hardware capabilities and auto-selected models.

**Response:**
```json
{
  "device": "mps",
  "vram_gb": 24.0,
  "auto_whisper_model": "large-v3",
  "auto_qwen_model": "Qwen/Qwen2-VL-2B-Instruct",
  "auto_yolo_model": "yolov8m.pt",
  "auto_clip_model": "ViT-L-14",
  "auto_object_detector": "qwen",
  "recommendations": {
    "can_use_large_whisper": true,
    "can_use_qwen": true,
    "can_use_qwen_7b": false,
    "can_use_clip_l14": true,
    "can_use_yolo_xlarge": false
  }
}
```

---

### GET /settings

Get current settings. Sensitive values (like HuggingFace token) are masked.

**Response:**
```json
{
  "api_version": "1.0",
  "log_level": "INFO",
  "whisper_model": "auto",
  "fallback_language": "en",
  "hf_token_set": false,
  "diarization_model": "pyannote/speaker-diarization-3.1",
  "face_sample_fps": 1.0,
  "object_sample_fps": 2.0,
  "min_face_size": 80,
  "object_detector": "auto",
  "qwen_model": "auto",
  "qwen_frames_per_scene": 3,
  "yolo_model": "auto",
  "clip_model": "auto",
  "ocr_languages": ["en", "no", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "fi", "pl"],
  "temp_dir": "/tmp/polybos"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hf_token_set` | bool | Whether HuggingFace token is configured (actual value is never exposed) |
| `whisper_model` | string | "auto" or specific model name |
| `object_detector` | string | "auto", "yolo", or "qwen" |
| Other fields | various | See Settings below |

---

### PUT /settings

Update settings. Only provided fields are updated. Changes persist to `~/.config/polybos/config.json`.

**Request Body:**
```json
{
  "hf_token": "hf_xxxxxxxxxxxx",
  "whisper_model": "large-v3",
  "face_sample_fps": 2.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hf_token` | string | HuggingFace token for speaker diarization. Set to empty string `""` to clear. |
| `whisper_model` | string | "auto", "tiny", "small", "medium", or "large-v3" |
| `fallback_language` | string | Fallback language code for short clips |
| `diarization_model` | string | Pyannote model for speaker diarization |
| `face_sample_fps` | float | Face detection sampling rate |
| `object_sample_fps` | float | Object detection sampling rate |
| `min_face_size` | int | Minimum face size in pixels |
| `object_detector` | string | "auto", "yolo", or "qwen" |
| `qwen_model` | string | Qwen model name or "auto" |
| `qwen_frames_per_scene` | int | Frames per scene for Qwen |
| `yolo_model` | string | YOLO model name or "auto" |
| `clip_model` | string | CLIP model name or "auto" |
| `ocr_languages` | string[] | OCR language codes |
| `temp_dir` | string | Temporary directory for processing |

**Response:** Same as GET /settings (returns updated settings)

**Notes:**
- To enable speaker diarization, you need a HuggingFace token and must accept the [pyannote model license](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Get a token at https://huggingface.co/settings/tokens

---

### GET /extractors

List available extractors.

**Response:**
```json
{
  "extractors": [
    {"name": "metadata", "description": "Video metadata and device info"},
    {"name": "transcript", "description": "Speech-to-text with Whisper"},
    ...
  ]
}
```

---

### POST /shutdown

Gracefully shutdown the engine (unloads all models).

---

## Model Selection

All model fields support `"auto"` for automatic selection based on available VRAM:

| VRAM | Whisper | YOLO | CLIP | Object Detector |
|------|---------|------|------|-----------------|
| <4GB | tiny | yolov8n | ViT-B-16 | yolo |
| 4-8GB | small | yolov8s | ViT-B-32 | yolo |
| 8-16GB | medium | yolov8m | ViT-L-14 | qwen |
| 16GB+ | large-v3 | yolov8l | ViT-L-14 | qwen |

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid parameters) |
| 404 | File or job not found |
| 500 | Internal server error |

Error response body:
```json
{
  "detail": "Error message"
}
```
