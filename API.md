# Polybos Media Engine API Reference

Base URL: `http://localhost:8001`

Interactive documentation available at `/docs` (Swagger UI) when server is running.

Input files can be **video**, **image**, or **audio** (detected by extension). Extractors that don't apply to a media type are skipped automatically (e.g., no VAD/scenes/transcript for images; images are analyzed as a single frame at t=0).

---

## Batch Processing

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
  "language": "no",
  "language_hints": ["en", "no"],
  "context_hint": "Interview about technology",
  "contexts": {
    "/path/to/video1.mp4": {"location": "Oslo", "person": "John Smith"},
    "/path/to/video2.mp4": {"location": "Bergen", "person": "Jane Doe"}
  },
  "visual_timestamps": {
    "/path/to/video1.mp4": [10.0, 30.0, 60.0],
    "/path/to/video2.mp4": [5.0, 15.0]
  },
  "visual_strategy": {
    "/path/to/video2.mp4": "batch_context"
  },
  "visual_batch_overlap": {
    "/path/to/video2.mp4": true
  },
  "lut_path": "/path/to/color.cube"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `files` | string[] | required | List of media file paths (all must exist, else 404) |
| `enable_metadata` | bool | true | Extract metadata (duration, resolution, GPS, device) |
| `enable_vad` | bool | false | Voice activity detection |
| `enable_scenes` | bool | false | Scene boundary detection |
| `enable_transcript` | bool | false | Whisper transcription (+ speaker diarization if HF token set) |
| `enable_faces` | bool | false | Face detection with embeddings |
| `enable_objects` | bool | false | Object detection with YOLO (fast, label counts) |
| `enable_visual` | bool | false | Scene descriptions with Qwen VLM (slower, richer) |
| `enable_clip` | bool | false | CLIP embeddings for similarity search |
| `enable_ocr` | bool | false | Text extraction from frames |
| `enable_motion` | bool | false | Camera motion analysis |
| `clip_sample_fps` | float | null | Fixed CLIP sampling rate in Hz, range [0.1, 10.0]. When set, CLIP embeds frames at this rate regardless of scene boundaries; when null, per-scene sampling (pre-1.1 behavior). Falls back to the `clip_default_sample_fps` setting. |
| `language` | string | null | Force language for Whisper (ISO 639-1 code, e.g., "en", "no") |
| `language_hints` | string[] | null | Language hints (currently unused by Whisper) |
| `context_hint` | string | null | Context hint for Whisper initial prompt |
| `contexts` | object | null | Per-file context for Qwen (file path -> context dict) |
| `visual_timestamps` | object | null | Per-file timestamps for visual/VLM analysis (file path -> float[]) |
| `visual_strategy` | object | null | Per-file Qwen strategy override (file path -> "single", "context", "batch", or "batch_context") |
| `visual_batch_overlap` | object | null | Per-file batch overlap setting (file path -> bool). Enable for unstable camera. |
| `lut_path` | string | null | Path to LUT file (.cube) for log footage color correction (applied to frames before Qwen) |

**Note:** Model selection (whisper_model, yolo_model, qwen_model, clip_model) is configured via `PUT /settings`. This keeps hardware-dependent configuration in one place.

**Note:** Telemetry (GPS/flight path) is always extracted automatically when available. No flag needed - it's lightweight and included in results.

**Note:** Only one batch runs at a time. If a batch is already running, new batches are queued and start automatically when the current batch finishes.

**Response:**
```json
{
  "batch_id": "abc12345"
}
```

---

### GET /batch/{batch_id}

Get batch job status and results.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status_only` | bool | false | If true, return only status/progress without large result data. Use for polling progress. |

**Examples:**
```bash
# Poll status only (lightweight, no embeddings/transcripts)
curl "http://localhost:8001/batch/abc123?status_only=true"

# Get full results when done
curl "http://localhost:8001/batch/abc123"
```

**Response:**
```json
{
  "batch_id": "abc12345",
  "status": "running",
  "queue_position": null,
  "current_extractor": "visual_processing",
  "progress": {
    "message": "Processing video1.mp4",
    "current": 1,
    "total": 2,
    "stage_elapsed_seconds": 12.3,
    "eta_seconds": 30.5,
    "total_eta_seconds": 120.0,
    "queue_eta_seconds": null,
    "queued_batches": null
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
      "extractor_status": {
        "metadata": "completed",
        "telemetry": "completed",
        "vad": "skipped",
        "motion": "completed",
        "scenes": "completed",
        "frame_decode": "completed",
        "objects": "skipped",
        "faces": "skipped",
        "ocr": "skipped",
        "clip": "skipped",
        "visual": "skipped",
        "transcript": "skipped"
      },
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

**Batch status values:**

| Status | Description |
|--------|-------------|
| `queued` | Waiting in queue (another batch is running) |
| `pending` | Job created, about to start processing |
| `running` | Processing in progress |
| `completed` | All files processed successfully |
| `failed` | Job failed with error |

| Field | Description |
|-------|-------------|
| `queue_position` | Position in queue (1 = next to run). `null` if not queued. |

**Progress fields (ETA tracking):**

| Field | Description |
|-------|-------------|
| `stage_elapsed_seconds` | Time spent in the current extractor stage |
| `eta_seconds` | Estimated seconds remaining for the current stage |
| `total_eta_seconds` | Estimated seconds remaining for the entire batch |
| `queue_eta_seconds` | Estimated seconds for all queued batches (null if queue empty) |
| `queued_batches` | Number of batches waiting in queue (null if none) |

**Per-file `results` keys** (present only if the extractor was enabled and produced output): `metadata`, `telemetry`, `vad`, `motion`, `scenes`, `objects`, `visual`, `faces`, `ocr`, `clip`, `transcript`. See [Extractor Results](#extractor-results) for the exact schema of each.

**`status_only=true` response:**

When polling for progress, use `?status_only=true` to avoid transferring large result data (embeddings, transcripts, detections). The response includes:
- All status fields (`status`, `current_extractor`, `progress`, `queue_position`)
- Per-file `status`, `error`, `timings`, `extractor_status`
- Batch metrics (`elapsed_seconds`, `memory_mb`, `peak_memory_mb`, `extractor_timings`)

But excludes:
- `files[].results` (empty object instead of metadata, transcript, embeddings, etc.)

**Extractor Status Values:**

| Status | Description |
|--------|-------------|
| `pending` | Not yet started |
| `active` | Currently processing this file |
| `completed` | Finished successfully |
| `failed` | Failed with error |
| `skipped` | Extractor not enabled, or file skipped (e.g., no audio for transcript, image file for scenes) |

`extractor_status` also contains `frame_decode` (the shared frame decoding step used by objects/faces/ocr/clip); it has no entry in `results`.

**Note:** If metadata extraction fails for a file (unreadable by ffprobe), all subsequent extractors are skipped for that file and the file is marked `failed`.

---

### DELETE /batch/{batch_id}

Delete a batch job and free memory. If the batch is queued, it will be removed from the queue. If the batch is running, deletion removes the status tracking but does not stop processing.

**Response:**
```json
{
  "status": "deleted",
  "batch_id": "abc12345"
}
```

---

## Utility Endpoints

### GET /health

Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "0.3.0",
  "api_version": "1.1"
}
```

`api_version >= "1.1"` signals the fixed-rate CLIP sampling and extended motion feature fields are available.

---

### GET /logs

Get recent log entries for debugging. Reads from `/tmp/media_engine.log`.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lines` | int | 100 | Number of lines to return (max 1000) |
| `level` | string | null | Filter by log level (DEBUG, INFO, WARNING, ERROR) |

**Response:**
```json
{
  "lines": [
    "2024-01-15 10:00:00,123 INFO media_engine.main: Starting batch abc123",
    "2024-01-15 10:00:01,456 INFO media_engine.extractors.metadata: Extracted metadata"
  ],
  "total": 100,
  "returned": 100,
  "file": "/tmp/media_engine.log"
}
```

**Examples:**
```bash
# Get last 100 lines
curl http://localhost:8001/logs

# Get last 50 error lines
curl "http://localhost:8001/logs?lines=50&level=ERROR"
```

---

### GET /hardware

Get hardware capabilities and auto-selected models.

**Response:**
```json
{
  "device": "mps",
  "gpu_name": "Apple M2 Max",
  "vram_gb": 32.0,
  "free_memory_gb": 18.5,
  "auto_whisper_model": "large-v3",
  "auto_qwen_model": "Qwen/Qwen3-VL-8B-Instruct",
  "auto_qwen_strategy": "batch_context",
  "auto_yolo_model": "yolov8x.pt",
  "auto_clip_model": "ViT-L-14",
  "auto_object_detector": "qwen",
  "recommendations": {
    "can_use_large_whisper": true,
    "can_use_qwen": true,
    "can_use_qwen_8b": true,
    "can_use_qwen_27b": true,
    "can_use_clip_l14": true,
    "can_use_yolo_xlarge": true
  },
  "available_now": {
    "qwen_2b": true,
    "qwen_8b": true,
    "qwen_27b": true,
    "whisper_large": true,
    "whisper_medium": true,
    "whisper_small": true,
    "yolo": true,
    "clip": true
  }
}
```

- `vram_gb` is total VRAM (unified memory on Apple Silicon); `free_memory_gb` is memory currently available for models.
- `auto_qwen_model` is `null` when VRAM < 8GB (YOLO will be used instead).
- `recommendations` reflects what the hardware *can* support (total VRAM); `available_now` reflects what can load *right now* (free memory).

---

### POST /check-models

Start a background check of which models can actually load on this machine. Returns immediately; poll `GET /check-models/{check_id}` for results. Takes 30-60 seconds to complete.

**Response:**
```json
{
  "check_id": "a1b2c3d4",
  "status": "running"
}
```

### GET /check-models/{check_id}

Get the result of a model check. `status` is `running`, `complete`, or `error` (404 if check_id unknown).

**Response (complete):**
```json
{
  "check_id": "a1b2c3d4",
  "status": "complete",
  "freeMemoryGb": 18.5,
  "results": {
    "qwen_2b": {"canLoad": true, "error": null, "loadTimeSeconds": 12.3},
    "whisper_large": {"canLoad": true, "error": null, "loadTimeSeconds": 8.1},
    "clip": {"canLoad": true, "error": null, "loadTimeSeconds": 2.4},
    "yolo": {"canLoad": true, "error": null, "loadTimeSeconds": 1.1},
    "faces": {"canLoad": true, "error": null, "loadTimeSeconds": 3.0}
  }
}
```

---

### GET /settings

Get current settings. Sensitive values (like HuggingFace token) are masked.

**Response:**
```json
{
  "api_version": "1.1",
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
  "qwen_strategy": "auto",
  "qwen_frames_per_scene": 1,
  "yolo_model": "auto",
  "clip_model": "auto",
  "clip_default_sample_fps": null,
  "motion_features_enabled": true,
  "ocr_languages": ["en", "no", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "pl"],
  "temp_dir": "/tmp/polybos"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hf_token_set` | bool | Whether HuggingFace token is configured (actual value is never exposed) |
| `whisper_model` | string | "auto" or specific model name |
| `object_detector` | string | "auto", "yolo", or "qwen" |
| Other fields | various | See PUT /settings below |

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
| `log_level` | string | Log level (DEBUG, INFO, WARNING, ERROR) |
| `hf_token` | string | HuggingFace token for speaker diarization. Set to empty string `""` to clear. |
| `whisper_model` | string | "auto", "tiny", "small", "medium", or "large-v3" |
| `fallback_language` | string | Fallback language code for short clips |
| `diarization_model` | string | Pyannote model for speaker diarization |
| `face_sample_fps` | float | Face detection sampling rate |
| `object_sample_fps` | float | Object detection sampling rate |
| `min_face_size` | int | Minimum face size in pixels |
| `object_detector` | string | "auto", "yolo", or "qwen" |
| `qwen_model` | string | Qwen model name or "auto" |
| `qwen_strategy` | string | "auto", "single", "context", "batch", or "batch_context" |
| `qwen_frames_per_scene` | int | Frames per scene for Qwen |
| `yolo_model` | string | YOLO model name or "auto" |
| `clip_model` | string | CLIP model name or "auto" |
| `clip_default_sample_fps` | float or null | Default `clip_sample_fps` when a batch request doesn't specify one, range [0.1, 10.0]. Null = per-scene mode. |
| `motion_features_enabled` | bool | If false, the extended motion feature fields are omitted from responses (default true) |
| `ocr_languages` | string[] | OCR language codes (EasyOCR codes) |
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
    {"name": "metadata", "description": "Video metadata (duration, resolution, codec, device, GPS)", "enable_flag": "enable_metadata"},
    {"name": "transcript", "description": "Audio transcription using Whisper", "enable_flag": "enable_transcript"},
    {"name": "scenes", "description": "Scene boundary detection", "enable_flag": "enable_scenes"},
    {"name": "faces", "description": "Face detection with embeddings", "enable_flag": "enable_faces"},
    {"name": "objects", "description": "Object detection with YOLO (fast, bounding boxes)", "enable_flag": "enable_objects"},
    {"name": "visual", "description": "Scene descriptions with Qwen VLM (slower, richer)", "enable_flag": "enable_visual"},
    {"name": "clip", "description": "CLIP visual embeddings per scene", "enable_flag": "enable_clip"},
    {"name": "ocr", "description": "Text extraction from video frames", "enable_flag": "enable_ocr"},
    {"name": "telemetry", "description": "GPS/flight path (always extracted automatically)"}
  ]
}
```

---

### POST /encode_text

Encode a text query to a CLIP embedding for text-to-image similarity search. Non-English queries are translated to English before encoding (CLIP models are trained on English text).

**Request Body:**
```json
{
  "text": "solnedgang over fjorden",
  "model_name": "ViT-B-32",
  "translate": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | The text query to encode |
| `model_name` | string | settings `clip_model` | CLIP model name (must match the model used for video embeddings) |
| `translate` | bool | true | Translate non-English queries to English before encoding |

**Response:**
```json
{
  "embedding": [0.012, -0.034, ...],
  "model": "ViT-B-32",
  "original_text": "solnedgang over fjorden",
  "translated_text": "sunset over the fjord",
  "detected_language": "no",
  "was_translated": true
}
```

The embedding is normalized; 512 dimensions for ViT-B models, 768 for ViT-L-14.

---

### POST /shutdown

Gracefully shutdown the engine (unloads all models, then exits the process).

**Response:**
```json
{
  "status": "shutting_down"
}
```

---

## Model Selection

All model fields support `"auto"` for automatic selection. Whisper/YOLO/CLIP are selected from total VRAM (unified memory on Apple Silicon); the object detector, Qwen model, and Qwen strategy are selected from *free* memory at the time of the check.

| VRAM | Whisper | YOLO | CLIP |
|------|---------|------|------|
| <2GB | tiny | yolov8n | ViT-B-16 |
| 2-4GB | tiny/small | yolov8s | ViT-B-32 |
| 4-6GB | small | yolov8m | ViT-L-14 |
| 6-8GB | medium | yolov8m | ViT-L-14 |
| 8-10GB | medium | yolov8l | ViT-L-14 |
| 10-16GB | large-v3 | yolov8l | ViT-L-14 |
| 16GB+ | large-v3 | yolov8x | ViT-L-14 |

| Free Memory | Object Detector | Qwen Model |
|-------------|-----------------|------------|
| <8GB | yolo | - |
| 8-16GB | qwen | Qwen/Qwen3-VL-2B-Instruct |
| 16-24GB | qwen | Qwen/Qwen3-VL-8B-Instruct |
| 24GB+ | qwen | Qwen/Qwen3.5-27B (4-bit) |

### Qwen Strategy Selection

The `qwen_strategy` setting controls how Qwen analyzes multiple frames for temporal context:

| Strategy | Description | Memory | Use Case |
|----------|-------------|--------|----------|
| `single` | Each frame analyzed independently | Lowest | Fast processing, no temporal context needed |
| `context` | Previous frame's description passed as text | Low | Basic temporal awareness |
| `batch` | Multiple frames analyzed together | Medium | Action detection, scene understanding |
| `batch_context` | Batches with text context between groups | Higher | Richest temporal understanding |

**Auto-selection based on free memory:**

| Free Memory | Strategy |
|-------------|----------|
| <8GB | context |
| 8-12GB | batch |
| 12GB+ | batch_context |

**Auto batch size** (frames per Qwen call, for batch strategies):

| Free Memory | Batch Size |
|-------------|------------|
| <10GB | 2 |
| 10-15GB | 3 |
| 15-25GB | 4 |
| 25-40GB | 5 |
| 40GB+ | 6 |

**Batch overlap:** When `visual_batch_overlap` is enabled for a file, batches overlap by 1 frame for visual continuity. Useful for videos with unstable camera or rapid scene changes.

### CLIP sampling cost

Fixed-rate CLIP sampling multiplies inference cost proportionally: a 30-second clip at `clip_sample_fps: 1.0` produces 30 embeddings vs ~3 in per-scene mode (~10× the inference). Per-embedding cost is small (~15-20 ms for ViT-L-14 on Apple Silicon) and frames are streamed with bounded memory, but response payload grows linearly with the rate. Typical useful values: 0.5, 1.0, 2.0 — higher rates mostly produce near-duplicate embeddings.

---

## Extractor Results

The exact schema of each key in `files[].results`. Fields that could not be determined are `null` (or omitted where noted).

### `metadata`

```json
{
  "duration": 120.5,
  "resolution": {"width": 1920, "height": 1080},
  "codec": {"video": "hevc", "audio": "aac"},
  "video_codec": {
    "name": "hevc",
    "profile": "Main 10",
    "bit_depth": 10,
    "pixel_format": "yuv420p10le"
  },
  "audio": {
    "codec": "aac",
    "sample_rate": 48000,
    "channels": 2,
    "bit_depth": null,
    "bitrate": 256000
  },
  "fps": 25.0,
  "bitrate": 50000000,
  "file_size": 750000000,
  "timecode": "01:15:07:17",
  "created_at": "2024-01-15T10:00:00Z",
  "device": {
    "make": "Sony",
    "model": "PXW-FS7",
    "serial_number": "0034075",
    "software": null,
    "type": "camera",
    "detection_method": "xml_sidecar",
    "confidence": 1.0
  },
  "gps": {
    "latitude": 59.9139,
    "longitude": 10.7522,
    "altitude": 100.5
  },
  "gps_track": {
    "points": [
      {"latitude": 59.9139, "longitude": 10.7522, "altitude": 100.5, "timestamp": 0.0},
      {"latitude": 59.9140, "longitude": 10.7523, "altitude": 101.0, "timestamp": 1.0}
    ],
    "source": "srt_sidecar"
  },
  "color_space": {
    "transfer": "slog3",
    "primaries": "sgamut3",
    "matrix": "bt709",
    "lut_file": null,
    "detection_method": "xml_sidecar"
  },
  "lens": {
    "model": "XT14X5.8",
    "focal_length": 14.0,
    "focal_length_35mm": 28.0,
    "aperture": 2.8,
    "focus_distance": null,
    "iris": "F2.8",
    "detection_method": "metadata"
  },
  "shot_type": null,
  "keyframes": {
    "timestamps": [0.0, 2.0, 4.0],
    "count": 60,
    "is_fixed_interval": true,
    "avg_interval": 2.0
  },
  "spanned_recording": {
    "is_continuation": false,
    "sibling_files": ["00001.MTS"],
    "total_duration": 969.8,
    "file_index": 0
  },
  "stereo_3d": {
    "mode": "mvc",
    "eye_count": 2,
    "has_left_eye": true,
    "has_right_eye": true,
    "detection_method": "metadata"
  }
}
```

- `audio` is `null` for files without an audio track (used to skip VAD/transcription).
- `shot_type` (`{primary, confidence, detection_method}`) is defined in the schema but currently not populated by the pipeline — always `null`.
- `gps_track.source` is e.g. `"srt_sidecar"` or `"avchd_sei"`.
- `keyframes`: irregular keyframe intervals often indicate actual cuts, while fixed intervals (e.g., every 2s) indicate standard GOP compression.

#### Device Types

| Type | Description |
|------|-------------|
| `camera` | Professional or consumer camera |
| `cinema_camera` | Cinema camera (ARRI, RED, etc.) |
| `phone` | Smartphone (iPhone, Android) |
| `drone` | Aerial drone (DJI, etc.) |
| `action_camera` | Action camera (GoPro, Insta360) |
| `360_camera` | 360-degree camera |
| `dashcam` | Vehicle dashcam (Tesla, etc.) |
| `unknown` | Unknown device type |

#### Stereo 3D Modes

| Mode | Description |
|------|-------------|
| `mvc` | H.264 Multiview Video Coding (3D Blu-ray, consumer 3D camcorders) |
| `side_by_side` | Left/right frames side by side (half width each) |
| `side_by_side_full` | Full width side-by-side (doubled width) |
| `top_bottom` | Left/right frames stacked (half height each) |
| `top_bottom_full` | Full height top-bottom (doubled height) |
| `frame_sequential` | Alternating L/R frames |
| `dual_stream` | Separate files for each eye |

#### Spanned Recording

AVCHD cameras split long recordings at ~2GB boundaries (FAT32 limit). When detected:

- `is_continuation`: `true` if this file is NOT the first of the recording
- `sibling_files`: Other files belonging to the same recording (filenames only)
- `total_duration`: Combined duration of all files in the recording
- `file_index`: Position of this file (0-based)

#### Detection Methods

| Method | Description |
|--------|-------------|
| `metadata` | Extracted from embedded metadata tags |
| `xml_sidecar` | Parsed from XML sidecar file (Sony M01.XML, etc.) |
| `clip` | Detected using CLIP model |

### `telemetry`

GPS/flight path from drone SRT sidecars or embedded subtitle streams. `null` if no telemetry found.

```json
{
  "source": "dji_srt",
  "sample_rate": 1.0,
  "duration": 120.0,
  "points": [
    {
      "timestamp": 0.0,
      "recorded_at": "2024-01-15T10:00:00Z",
      "latitude": 59.9139,
      "longitude": 10.7522,
      "altitude": 100.5,
      "relative_altitude": 50.0,
      "iso": 100,
      "shutter": 0.01,
      "aperture": 2.8,
      "focal_length": 24.0,
      "color_mode": "d_log"
    }
  ]
}
```

- `source`: `"dji_srt"` (SRT sidecar file) or `"embedded_subtitle"` (subtitle stream inside the video).
- `timestamp` is seconds from start of video; `recorded_at` is the wall-clock time from telemetry.
- Camera settings (`iso`, `shutter`, `aperture`, `focal_length`, `color_mode`) are `null` when not present.

### `vad`

Voice activity detection (energy + zero-crossing-rate heuristic on PCM frames; analyzes up to the first 120s).

```json
{
  "audio_content": "speech",
  "speech_ratio": 0.42,
  "speech_segments": [[1.2, 5.7], [8.0, 12.3]],
  "total_duration": 120.0
}
```

| `audio_content` | Description |
|-----------------|-------------|
| `no_audio` | File has no audio track (also returned for images) |
| `speech` | Speech detected (worth running Whisper) |
| `audio` | Audio present but no speech (ambient/music/silent) |
| `unknown` | Could not determine (extraction failed) |

`speech_segments` is a list of `[start, end]` pairs in seconds.

### `motion`

Camera motion analysis via optical flow. Also computed implicitly (but stored in results) when objects/faces/ocr/clip are enabled without precomputed `visual_timestamps`, to pick adaptive sample timestamps.

```json
{
  "duration": 120.0,
  "fps": 25.0,
  "primary_motion": "pan_left",
  "avg_intensity": 3.2,
  "is_stable": false,
  "magnitude_p90_overall": 5.4,
  "jerk_max_overall": 12.1,
  "hf_lf_ratio_overall": 0.42,
  "features_version": "v1",
  "segments": [
    {
      "start": 10.5,
      "end": 25.0,
      "motion_type": "handheld",
      "intensity": 3.2,
      "magnitude_mean": 3.2,
      "magnitude_std": 1.1,
      "magnitude_p90": 4.8,
      "magnitude_max": 6.2,
      "direction_consistency": 0.72,
      "direction_reversals_per_sec": 0.8,
      "acceleration_mean": 0.9,
      "jerk_max": 8.4,
      "hf_energy": 0.35,
      "lf_energy": 0.65,
      "hf_lf_ratio": 0.35,
      "features_version": "v1"
    }
  ]
}
```

`motion_type` / `primary_motion` values: `static`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `push_in`, `pull_out`, `handheld`, `complex`. `push_in`/`pull_out` describe the optical flow pattern (radial expansion/contraction) — could be optical zoom or physical camera movement.

#### Extended motion features (api_version 1.1)

Everything beyond `start`/`end`/`motion_type`/`intensity` is additive (api_version 1.1). The fields are pure post-processing of the optical-flow series the classifier already computes — classification itself is unchanged. Set the `motion_features_enabled` setting to `false` to omit them entirely (escape hatch for consumers that reject unknown fields). The engine reports what the camera did; interpretation (good/bad footage) stays with the consumer.

| Field | Type | Meaning |
|-------|------|---------|
| `magnitude_mean` | float | Mean optical flow magnitude across segment (same as existing `intensity`; kept for clarity) |
| `magnitude_std` | float | Std dev of magnitude — how variable is the motion speed |
| `magnitude_p90` | float | 90th percentile of magnitude — near-peak speed, robust to single-frame outliers |
| `magnitude_max` | float | Max magnitude in segment (useful for detecting spikes / near-yanks) |
| `direction_consistency` | float in [0, 1] | 1.0 = perfectly monotonic direction; 0.0 = random/reversing (circular statistics over flow direction) |
| `direction_reversals_per_sec` | float | Count of direction changes >90° per second |
| `acceleration_mean` | float | Mean absolute d(magnitude)/dt (units: magnitude per second) |
| `jerk_max` | float | Peak d²(magnitude)/dt² in segment — the mathematical signature of "yanking" |
| `hf_energy` | float in [0, 1] | Normalized energy in >2 Hz band (shake, tremor, focus hunt) |
| `lf_energy` | float in [0, 1] | Normalized energy in <2 Hz band (intentional pan/tilt/dolly) |
| `hf_lf_ratio` | float | `hf_energy / (hf_energy + lf_energy)`, in [0, 1]. High = shaky. |
| `features_version` | string | e.g. `"v1"`. Bumped when any feature formula, the 2 Hz split, or smoothing parameters change — invalidate caches of derived scores when it changes. |

Clip-level summaries (`magnitude_p90_overall`, `jerk_max_overall`, `hf_lf_ratio_overall`) are computed over the full clip's flow series — useful for library-level filtering (e.g., "clips with `jerk_max_overall` > 20").

Very short segments (fewer than 5 analysis samples, i.e. <1s at the 5 fps analysis rate) get deterministic degenerate values: `magnitude_std`=0, `magnitude_p90`=`magnitude_max`=`magnitude_mean`, `direction_consistency`=1.0, reversals/acceleration/jerk=0, `hf_energy`=0, `lf_energy`=1.

### `scenes`

Scene boundary detection (PySceneDetect ContentDetector). Skipped for images.

```json
{
  "count": 3,
  "detections": [
    {"index": 0, "start": 0.0, "end": 12.5, "duration": 12.5},
    {"index": 1, "start": 12.5, "end": 30.0, "duration": 17.5}
  ]
}
```

### `objects`

YOLO object detection. **Only the summary (label -> count) is returned in results**; individual detections are used internally (e.g., person timestamps guide face sampling) but not included in the response.

```json
{
  "summary": {"person": 12, "car": 3, "dog": 1}
}
```

### `visual`

Qwen VLM scene descriptions. `descriptions` is omitted when empty.

```json
{
  "summary": {"person": 2, "laptop": 1},
  "descriptions": [
    "A person sitting at a desk working on a laptop in a bright office.",
    "Close-up of hands typing on a keyboard."
  ]
}
```

### `faces`

Face detection (DeepFace/Facenet) with per-detection embeddings. `{"count": 0, "unique_estimate": 0, "detections": []}` when no faces found.

```json
{
  "count": 5,
  "unique_estimate": 2,
  "detections": [
    {
      "timestamp": 10.5,
      "bbox": {"x": 100, "y": 50, "width": 120, "height": 150},
      "confidence": 0.98,
      "embedding": [0.01, -0.02, ...],
      "image_base64": "/9j/4AAQ...",
      "needs_review": false,
      "review_reason": null
    }
  ]
}
```

- `embedding`: 512-dim Facenet512 face embedding for recognition/clustering.
- `image_base64`: base64-encoded JPEG crop of the face (may be `null`).
- `needs_review`/`review_reason`: flags uncertain detections for manual review.
- `unique_estimate`: estimated number of distinct people (embedding clustering).

### `ocr`

Text extraction from frames (EasyOCR).

```json
{
  "detections": [
    {
      "timestamp": 10.0,
      "text": "BREAKING NEWS",
      "confidence": 0.95,
      "bbox": {"x": 20, "y": 900, "width": 400, "height": 60}
    }
  ]
}
```

### `clip`

CLIP embeddings for similarity search. Use `POST /encode_text` to encode queries with the same model. Two sampling modes, selected by the `clip_sample_fps` request field (or the `clip_default_sample_fps` setting):

**`per_scene` mode** (default, `clip_sample_fps` unset) — one embedding per motion-adaptive sample point:

```json
{
  "model": "ViT-B-32",
  "sample_mode": "per_scene",
  "sample_fps": null,
  "segments": [
    {"start": 0.5, "end": 0.5, "timestamp": 0.5, "scene_index": 0, "embedding": [0.01, ...]},
    {"start": 12.5, "end": 12.5, "timestamp": 12.5, "scene_index": 1, "embedding": [0.02, ...]}
  ]
}
```

Note: despite the mode name, sampling in this mode is motion-adaptive rather than strictly scene-based, and `scene_index` is a **running sample index** (0, 1, 2, …), not a reference to the `scenes` result. Both are preserved as-is for backward compatibility.

**`fixed_fps` mode** (`clip_sample_fps` set, api_version 1.1) — embeddings at a fixed rate across the whole clip, independent of scene boundaries. Lets downstream rankers locate visual peaks inside long, stable scenes:

```json
{
  "model": "ViT-L-14",
  "sample_mode": "fixed_fps",
  "sample_fps": 1.0,
  "segments": [
    {"start": 0.0, "end": 1.0, "timestamp": 0.5, "scene_index": 0, "embedding": [...]},
    {"start": 1.0, "end": 2.0, "timestamp": 1.5, "scene_index": 0, "embedding": [...]}
  ]
}
```

- `timestamp` is the center-of-sample time; `start`/`end` bracket the sample window (at 1.0 fps: `timestamp ∓ 0.5`, clamped to the clip).
- `scene_index` points to the containing scene in the `scenes` result when `enable_scenes` was set for the batch, else `null`.
- Frames are streamed in bounded chunks (own ffmpeg pass at reduced resolution), so memory stays flat regardless of clip length. A warning is logged when a request would produce >5000 samples for one file.
- Images fall back to the single-frame per_scene path.

Embedding dimensions: 512 for ViT-B models, 768 for ViT-L-14.

### `transcript`

Whisper transcription with optional speaker diarization (requires HuggingFace token, see PUT /settings). Skipped for images and files without an audio track.

```json
{
  "language": "en",
  "confidence": 0.98,
  "duration": 120.5,
  "speaker_count": 2,
  "speakers": [
    {
      "label": "SPEAKER_00",
      "embedding": [0.01, ...],
      "total_duration": 75.2
    }
  ],
  "hints_used": {
    "language_hints": [],
    "context_hint": null,
    "fallback_applied": false
  },
  "segments": [
    {"start": 0.0, "end": 4.2, "text": "Welcome to the show.", "speaker": "SPEAKER_00"}
  ]
}
```

- `speaker_count`/`speakers` are `null` when diarization is disabled (no HF token). Speaker `embedding` is a 256-dim voice embedding centroid from pyannote.
- `segments[].speaker` is `null` when diarization is disabled.
- `hints_used.fallback_applied` is `true` when language detection confidence was <0.7 on a clip shorter than 15s and the configured `fallback_language` was used instead.

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

Per-extractor failures inside a batch do not fail the whole batch — the extractor is marked `failed` in `extractor_status` and processing continues. Exceptions: metadata failure (file unreadable) and visual/transcript failures mark the *file* as failed.
