# Polybos Media Archive — Product Specification v2

## Overview

Polybos Media Archive is a self-hosted, AI-powered video archive and search system designed for small TV stations and content creators. Users can search their entire video library using natural language queries to find people, places, objects, or spoken words — then export relevant clips directly to their editing software.

### Business Model

- **Open-source backend** (`polybos-media-engine`) — MIT licensed
- **Closed-source frontend** — Commercial product, sold to customers
- Backend is functional but deliberately minimal UI (API + basic CLI)
- Revenue from frontend licenses, support, and customization

### Target Users

- Small/community TV stations with years of archived footage
- Content creators managing large video libraries
- Production companies needing searchable archives

---

## Core Features

### Search

Universal search box accepting natural language queries:

- **People**: "mayor Jensen", "the reporter with glasses"
- **Places**: "town hall", "Bragernes Church Drammen"
- **Objects**: "red car", "microphone", "Norwegian flag"
- **Speech**: "said something about budget", "mentioned climate"
- **Shot types**: "drone footage", "interviews", "b-roll"
- **Combinations**: "mayor Jensen town hall 2019"
- **Natural language**: "clips where people complained about parking"

Results ranked by relevance across all AI layers (faces, transcripts, visual content, objects, OCR, GPS).

### LLM-Enhanced Search (Optional)

When enabled, an LLM (local or cloud) enhances search capabilities:

- **Query expansion**: "parking complaint" → also searches "frustrated", "no spaces", "traffic"
- **Natural language understanding**: "angry citizens" → sentiment analysis on transcripts
- **Answer synthesis**: "What did the mayor say about the budget?" → summarized answer with clip references
- **Conversational search**: Follow-up questions with context

---

## AI Analysis Pipeline

### Processing Models

| Layer | AI Tool | Output | License |
|-------|---------|--------|---------|
| Transcription | Whisper (large-v3) | Full transcript with word-level timestamps | MIT ✅ |
| Face recognition | DeepFace + Facenet | Named persons with timestamps | MIT ✅ |
| Visual search | CLIP / OpenCLIP | Scene embeddings for semantic search | MIT ✅ |
| Object detection | RT-DETR or Grounding DINO | Detected objects with timestamps | Apache 2.0 ✅ |
| Text on screen | PaddleOCR or docTR | Lower thirds, signs, graphics | Apache 2.0 ✅ |
| Scene detection | PySceneDetect | Segment boundaries | BSD ✅ |
| LLM (optional) | Local (Ollama) or Cloud (Claude/OpenAI) | Summaries, topics, query expansion | Varies |

All core components use permissive licenses (MIT, Apache 2.0) for commercial use.

### Adaptive Frame Sampling

Not every frame needs processing — the system samples intelligently based on content:

| Task | Sample Rate | Rationale |
|------|-------------|-----------|
| Whisper | Audio only | No frames needed |
| Scene detection | All frames | Lightweight pixel comparison |
| Face detection | 1-2 fps | Faces don't change fast |
| Face recognition | Best face per scene | One good embedding enough |
| CLIP embeddings | 1 per scene or 5-10 sec | Semantic content changes slowly |
| Object detection | 1-2 fps | Objects persist across frames |
| OCR | Scene changes only | Graphics appear at cuts |

### Content-Aware Adaptive Sampling

For static content (interviews, talking heads), the system reduces sampling automatically:

```
30-minute interview analysis:

Without adaptive:  54,000 frames (30fps) → ~2 hours processing
With adaptive:     20-50 frames          → ~1 minute processing

Algorithm:
1. Analyze first frame of scene
2. Compare subsequent frames (CLIP similarity)
3. If >90% similar → increase sample interval (2→4→8→16→30 sec)
4. If content changes → reset to high frequency
5. Log any detected changes
```

Factors checked:
- Face embedding similarity (same person?)
- CLIP embedding similarity (same scene?)
- Face count changes (someone entered/left?)
- Significant object changes

---

## Face Recognition

### Training Workflow

Users train the system to recognize people:

1. System shows grid of detected unknown faces
2. Similar faces auto-clustered ("these 12 look like the same person")
3. User clicks and names a face
4. Name propagates to all clustered instances
5. Future uploads auto-tagged with known faces

### Face Quality Filtering

Not all detected faces are processed for recognition:

- Minimum size: 80px
- Blur detection: Skip blurry frames
- Angle: Prefer frontal faces
- One embedding per scene for same person

### Matching

```
New video uploaded
    │
    ▼
Face detected in frame
    │
    ▼
Generate embedding (DeepFace/Facenet)
    │
    ▼
Compare to all known person embeddings
    │
    ├── >90% match → Auto-tag (high confidence)
    ├── 75-90% match → Auto-tag (suggest verification)
    └── <75% match → "Unknown face" (user can name)
```

---

## Place Recognition

### Multi-Signal Location Identification

Places identified using combination of signals:

| Signal | Source | Reliability |
|--------|--------|-------------|
| GPS coordinates | Video metadata (EXIF) | ⭐⭐⭐⭐⭐ |
| Visual match | CLIP embedding | ⭐⭐⭐⭐ |
| OCR | Signs, text in frame | ⭐⭐⭐⭐ |
| Known place database | User-trained references | ⭐⭐⭐⭐ |
| OpenStreetMap lookup | GPS + building type | ⭐⭐⭐⭐ |

### GPS + Visual + OSM Integration

```
Frame shows a church
    │
    ├── CLIP: "church" (94%)
    ├── GPS: 59.7441°N, 10.2045°E (from metadata)
    │
    ▼
Query OpenStreetMap:
  "place_of_worship within 500m of coordinates"
    │
    ▼
Result: Only "Bragernes kirke" nearby
    │
    ▼
Auto-tag: "Bragernes Church, Drammen" (high confidence)
```

### Confidence Matrix

| GPS | Visual Match | OSM Lookup | Result |
|-----|--------------|------------|--------|
| ✅ Present | ✅ "church" | 1 church nearby | ⭐⭐⭐⭐⭐ Auto-tag |
| ✅ Present | ✅ "church" | 3 churches nearby | ⭐⭐⭐ Suggest options |
| ✅ Present | ❌ Generic | — | ⭐⭐ Tag area only |
| ❌ None | ✅ Known place match | — | ⭐⭐⭐⭐ Use CLIP match |
| ❌ None | ❌ Unknown | — | ⭐ "Unknown location" |

### Place Training (Like Face Training)

```
Known Places
├── Reference embeddings (multiple angles, seasons, lighting)
├── GPS coordinates (optional)
├── Building type (church, town hall, school, etc.)
└── User can add reference photos to improve matching
```

### External Data Sources

| Source | Data | Usage |
|--------|------|-------|
| OpenStreetMap | Buildings, landmarks, POIs | GPS → place name lookup |
| Wikidata | Named places with coordinates | Enrichment |
| GeoNames | Basic place names | Fallback |
| Custom database | Station's known locations | Primary for local coverage |

---

## Device & Shot Type Detection

### Source Device Detection

Automatically detect what device recorded the footage:

| Method | How | Reliability |
|--------|-----|-------------|
| EXIF/Metadata | Camera make/model in file | ⭐⭐⭐⭐⭐ |
| Visual analysis | CLIP classification | ⭐⭐⭐⭐ |
| Motion patterns | Movement analysis | ⭐⭐⭐ |
| Audio | Drone propeller noise | ⭐⭐⭐ |

### Drone Detection

```python
# Priority 1: Check metadata
DRONE_MANUFACTURERS = ['DJI', 'Parrot', 'Autel', 'Skydio', 'Yuneec']
if metadata.make in DRONE_MANUFACTURERS:
    return {"type": "drone", "confidence": 1.0}

# Priority 2: Visual classification
result = clip.classify(frame, ["aerial drone footage", "ground camera", ...])
if result == "aerial drone footage" and confidence > 0.85:
    return {"type": "drone", "confidence": confidence}
```

### Shot Type Classification

Auto-detected shot types:

| Shot Type | Detection Method |
|-----------|------------------|
| Drone/aerial | Metadata + CLIP "aerial view" |
| Studio | CLIP + controlled lighting |
| Interview | Static camera + 1-2 faces |
| B-roll | Scene variety + few/no faces |
| Live broadcast | Metadata + graphics overlays |
| Phone footage | Metadata + vertical aspect ratio |
| Dashcam | CLIP + motion pattern |
| Security cam | Wide static shot + timestamp overlay |

### Search Integration

| Query | Matches |
|-------|---------|
| "drone" | All aerial footage |
| "drone Beitostølen" | Aerial shots of that location |
| "interview mayor" | Interview-style shots with that face |
| "b-roll winter" | Non-interview outdoor winter footage |
| "phone footage" | Vertical/smartphone recordings |

---

## LLM Integration (Optional)

### Capabilities

| Feature | Description |
|---------|-------------|
| Transcript summarization | Generate summary, topics, key moments on ingest |
| Query expansion | "parking complaint" → related terms |
| Natural language search | "clips where people were angry" |
| Answer synthesis | "What did X say about Y?" → answer with sources |
| Conversational search | Follow-up questions with context |

### Transcript Analysis (On Ingest)

```
Asset: Council Meeting 2024-03-15
Duration: 2h 34min
Raw transcript: 47,000 words

LLM generates:
├── Summary: "Budget discussion focused on school funding..."
├── Topics: [budget, schools, roads, parking, taxes]
├── Key moments:
│   • 00:14:22 - Mayor presents budget
│   • 00:45:10 - Debate on school funding
│   • 01:22:05 - Parking complaint from citizen
├── Speakers detected: [Mayor Jensen, Councillor Hansen, ...]
└── Sentiment markers: [neutral, heated debate @01:22:00]
```

### Query Processing

```
User: "that interview where someone complained about parking"

Without LLM:
  Search: "complained" AND "parking"
  Result: Limited matches

With LLM:
  Expanded: parking, spaces, traffic, frustrated, annoyed, 
            terrible, no room, full, congestion
  Result: Much better recall
```

### Question Answering (RAG)

```
User: "What did the mayor say about the budget last year?"

System:
1. Search: mayor + budget + 2024
2. Retrieve relevant transcript segments
3. LLM synthesizes answer from multiple clips
4. Return answer WITH clip references

Response:
"In the March 2024 council meeting, Mayor Jensen proposed 
a 3% budget increase focused on schools. He stated that 
'education must be our priority for the coming years.'
Sources: [▶ Council Meeting 2024-03-15 @ 00:14:22]"
```

### Provider Options

```
Settings → AI Assistant

LLM Provider:
  ○ None (basic search only)
  ○ Local - Ollama (private, requires GPU)
      Model: [llama3 ▼] [mistral ▼] [custom]
  ○ Claude API (best quality, cloud)
      API Key: [sk-ant-................................]
  ○ OpenAI API (alternative cloud)
      API Key: [sk-...................................]

Privacy note: Cloud providers process transcript data externally.
```

### Provider Comparison

| Aspect | Local (Ollama) | Cloud (Claude/OpenAI) |
|--------|----------------|----------------------|
| Quality | Good | Best |
| Privacy | ✅ Data stays local | ⚠️ Sent to provider |
| Cost | Hardware only | Per-token |
| Offline | ✅ Works | ❌ Requires internet |
| Setup | More complex | Simple API key |

### Processing Costs

| Task | When | Frequency |
|------|------|-----------|
| Transcript summary | On ingest | Once per asset |
| Topic extraction | On ingest | Once per asset |
| Query expansion | On search | Every query |
| Answer synthesis | On demand | User-triggered |

---

## NLE Export

Selected clips exportable to editing software:

- **EDL** (Edit Decision List) — universal format
- **XML** (Premiere Pro, DaVinci Resolve compatible)
- **Folder export** — clips + sidecar metadata
- **Markers** — at relevant timecodes

---

## Ingest Pipeline

### Ingest Methods

| Method | Use Case |
|--------|----------|
| Watch folder | Daily workflow — auto-ingest dropped files |
| Web upload | Drag & drop one-off clips |
| Bulk import | Initial archive migration |

### Pipeline Flow

```
New file detected
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: Copy & Catalog                  │
│ • Copy to /originals                    │
│ • Extract metadata (ffprobe)            │
│ • Extract GPS if present                │
│ • Detect source device                  │
│ • Asset now searchable by filename/date │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Step 2: Generate Previews               │
│ • Thumbnails (always)                   │
│ • Sprite sheets for scrubbing           │
│ • Proxy file (optional)                 │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Step 3: AI Analysis (Background Queue)  │
│ • Scene detection                       │
│ • Whisper transcription                 │
│ • Face detection → recognition          │
│ • CLIP embeddings                       │
│ • Object detection                      │
│ • OCR                                   │
│ • Place recognition (GPS + visual)      │
│ • LLM summarization (if enabled)        │
└─────────────────────────────────────────┘
```

Asset is searchable immediately after Step 1. AI enrichment added progressively.

### FFmpeg Integration

| Task | Command | When |
|------|---------|------|
| Extract metadata | `ffprobe -print_format json -show_format -show_streams` | Ingest |
| Extract audio | `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav` | Before Whisper |
| Generate proxy | `ffmpeg -i video.mp4 -c:v libx264 -crf 23 -preset fast proxy.mp4` | Ingest (optional) |
| Generate HLS | `ffmpeg -i proxy.mp4 -hls_time 10 -hls_list_size 0 stream.m3u8` | Optional |
| Grid thumbnail | `ffmpeg -i video.mp4 -ss 10 -vframes 1 -vf scale=360:-1 thumb.jpg` | Ingest |
| Sprite sheet | `ffmpeg -i video.mp4 -vf "fps=1,scale=160:-1,tile=10x10" sprite.jpg` | Ingest |
| Extract frames | `ffmpeg -i video.mp4 -vf fps=2 frames/frame_%04d.jpg` | For AI analysis |

### Python FFmpeg Library

Use `ffmpeg-python` for clean integration:

```python
import ffmpeg

# Extract audio for Whisper
ffmpeg.input('video.mp4').output('audio.wav', ar=16000, ac=1).run()

# Generate sprite sheet
ffmpeg.input('video.mp4').output(
    'sprite.jpg',
    vf='fps=1,scale=160:-1,tile=10x10'
).run()

# Get metadata
probe = ffmpeg.probe('video.mp4')
duration = float(probe['format']['duration'])
gps = probe['format'].get('tags', {}).get('location')
```

---

## Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser                                │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              SvelteKit Frontend                       │  │
│  │                                                       │  │
│  │  • Search bar with autocomplete                       │  │
│  │  • Results grid (virtual scroll)                      │  │
│  │  • Video preview with transcript                      │  │
│  │  • Face training interface                            │  │
│  │  • Place training interface                           │  │
│  │  • Conversational search (LLM)                        │  │
│  │  • Export workflow                                    │  │
│  │  • Admin: users, settings, ingest status              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ REST + WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              polybos-media-engine (Python)                  │
│                      Open Source                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ FastAPI      │  │ AI Workers   │  │ Ingest Pipeline  │   │
│  │ REST + WS    │  │ (Dramatiq)   │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ PostgreSQL   │  │ Qdrant       │  │ Redis            │   │
│  │ (metadata)   │  │ (vectors)    │  │ (queue)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Ollama       │  │ OSM/Geo      │                         │
│  │ (local LLM)  │  │ Services     │                         │
│  │ (optional)   │  │              │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ NFS/SMB mount
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    TrueNAS / NAS Storage                    │
│                                                             │
│  /originals    — source files, read-only after ingest       │
│  /proxies      — H.264 streaming proxies (optional)         │
│  /thumbnails   — sprite sheets, grid thumbnails             │
│  /exports      — EDL/XML outputs                            │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| Frontend | SvelteKit | Fast, compiled, browser-based |
| UI components | Skeleton UI or DaisyUI | Tailwind-based |
| Video player | Vidstack | Modern, customizable |
| Backend framework | FastAPI | Async, WebSocket support |
| Task queue | Redis + Dramatiq | Background AI processing |
| Database | PostgreSQL 16 | Metadata, users, faces, places |
| Vector search | Qdrant | Semantic similarity search |
| Video processing | FFmpeg + ffmpeg-python | Proxy generation, thumbnails, frame extraction |
| Geo services | OpenStreetMap / Nominatim | GPS → place name lookup |
| Local LLM | Ollama (optional) | Private LLM inference |
| Deployment | Docker Compose | Single command setup |

### Docker Compose Services

```yaml
services:
  ui:
    image: polybos/media-ui
    ports: ["8080:8080"]
    depends_on: [api]
    
  api:
    image: polybos/media-engine
    environment:
      - DATABASE_URL=postgresql://...
      - QDRANT_URL=http://vectordb:6333
      - REDIS_URL=redis://redis:6379
      - LLM_PROVIDER=ollama  # or 'claude', 'openai', 'none'
      - OLLAMA_URL=http://ollama:11434
    volumes:
      - /mnt/nas/media:/media:ro
      - /mnt/nas/proxies:/proxies
      - /mnt/nas/thumbnails:/thumbnails
      - /mnt/nas/exports:/exports
    
  worker:
    image: polybos/media-engine
    command: worker
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - /mnt/nas/media:/media:ro
      - /mnt/nas/proxies:/proxies
      - /mnt/nas/thumbnails:/thumbnails
    
  db:
    image: postgres:16
    volumes:
      - db_data:/var/lib/postgresql/data
    
  vectordb:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    
  redis:
    image: redis:alpine
    
  ollama:  # Optional local LLM
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

volumes:
  db_data:
  qdrant_data:
  ollama_data:
```

---

## Data Model

### Core Entities

```
Asset
  ├── id (UUID)
  ├── file_path (original location)
  ├── proxy_path (nullable)
  ├── filename, duration, codec, resolution
  ├── created_at, ingested_at
  ├── status (pending, processing, ready, error)
  │
  ├── source_device
  │     ├── make ("DJI", "Sony", "Apple")
  │     ├── model ("Mavic 3", "iPhone 15")
  │     ├── type (drone, camera, phone, unknown)
  │     └── detection_confidence
  │
  ├── shot_type (aerial, interview, b-roll, studio, etc.)
  │
  ├── gps_location
  │     ├── latitude
  │     ├── longitude
  │     └── altitude
  │
  ├── llm_analysis (nullable, if LLM enabled)
  │     ├── summary
  │     ├── topics[]
  │     ├── key_moments[]
  │     └── sentiment_markers[]
  │
  ├── Segments[] (scene-based chunks)
  │     ├── start_tc, end_tc
  │     ├── thumbnail_path
  │     ├── clip_embedding (CLIP vector)
  │     ├── transcript_text
  │     └── is_static (for adaptive sampling)
  │
  ├── FaceAppearance[]
  │     ├── person_id → Person
  │     ├── start_tc, end_tc
  │     ├── confidence
  │     └── bounding_box
  │
  ├── PlaceAppearance[]
  │     ├── place_id → Place
  │     ├── start_tc, end_tc
  │     ├── confidence
  │     └── detection_method (gps, visual, ocr, manual)
  │
  ├── DetectedObject[]
  │     ├── label
  │     ├── confidence
  │     └── start_tc, end_tc
  │
  └── OCRText[]
        ├── text
        ├── bounding_box
        └── start_tc, end_tc

Person
  ├── id (UUID)
  ├── name
  ├── reference_embeddings[] (face vectors)
  ├── reference_images[] (for UI display)
  └── created_by → User

Place
  ├── id (UUID)
  ├── name ("Bragernes Church")
  ├── type (church, town_hall, school, hospital, etc.)
  ├── gps_location (optional)
  │     ├── latitude
  │     ├── longitude
  │     └── radius (for matching)
  ├── reference_embeddings[] (CLIP vectors)
  ├── reference_images[] (multiple angles/seasons)
  ├── osm_id (optional, link to OpenStreetMap)
  └── created_by → User

User
  ├── id (UUID)
  ├── username, email
  ├── password_hash (for local auth)
  ├── role (admin, editor, viewer)
  └── auth_provider (local, ldap, sso)
```

### Search Index

Hybrid search combining:

1. **Full-text** (PostgreSQL) — transcripts, filenames, OCR text, summaries
2. **Vector similarity** (Qdrant) — CLIP embeddings, face embeddings, place embeddings
3. **Structured filters** — date range, duration, resolution, shot type, device type
4. **LLM expansion** (optional) — query term expansion for better recall

Results fused and ranked by relevance across all sources.

---

## User Roles & Permissions

| Role | Permissions |
|------|-------------|
| Admin | Full access: users, settings, ingest, search, export, training |
| Editor | Search, export, face/place training, ingest |
| Viewer | Search, preview only |

### Authentication

- **Local users** — username/password, managed in app
- **LDAP/Active Directory** — optional integration
- **SSO (SAML/OIDC)** — optional for larger organizations

---

## Storage & Proxy Strategy

### Proxy Generation

Proxies are **optional**:

```
Settings → Storage → Proxy Generation

☑ Generate proxies for new assets
  Resolution: [1080p / 720p / 480p]
  
☐ Generate proxies for existing assets (background)

[Delete all proxies] — frees storage, preview uses originals
```

Without proxies:
- Thumbnails still work (always generated)
- Preview plays from original (slower seek)
- Search fully functional

### Storage Layout

```
/media (NAS mount)
  ├── originals/       # Source files, never modified
  │   ├── 2005/
  │   ├── 2006/
  │   └── ...
  ├── proxies/         # Optional H.264 streaming copies
  ├── thumbnails/      # Sprite sheets, grid thumbnails
  └── exports/         # EDL/XML outputs for NLE
```

### Offline Resilience

When NAS is unavailable:

| Feature | Status |
|---------|--------|
| Search | ✅ Works (metadata cached in PostgreSQL) |
| Thumbnails | ✅ Works (cached locally) |
| Preview | ❌ Unavailable |
| Export | ❌ Queued for when storage returns |
| Ingest | ❌ Paused |

---

## Backup

### Options

```
Settings → Backup

Database backup:
  ○ Manual export only
  ○ Scheduled to: [/mnt/backup/polybos]
  ○ S3-compatible: [endpoint] [bucket] [credentials]
  
  Frequency: [Daily / Weekly]
  Retain: [7 / 30 / 90 days]

[Backup now]  [Restore from backup...]
```

### What Gets Backed Up

- PostgreSQL dump (metadata, users, faces, places, all structured data)
- Qdrant vectors (can regenerate, but slow)
- Thumbnails (optional — can regenerate from originals)

Originals are the customer's responsibility (their NAS, their tape backup).

---

## Frontend UI Specifications

### Search Experience

```
┌────────────────────────────────────────────────────────────┐
│  🔍 [Search: mayor jensen town hall.....................]  │
│     Autocomplete: mayor jensen, mayor smith, town hall...  │
│                                                            │
│  Filters: [Date ▼] [Shot type ▼] [Location ▼] [Person ▼]  │
└────────────────────────────────────────────────────────────┘

Results stream in via WebSocket as they're found.

┌────────────────────────────────────────────────────────────┐
│  Results (47)                                    [Export ▼]│
│                                                            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ ▶ ░░ │ │ ▶ ░░ │ │ ▶ ░░ │ │ ▶ ░░ │ │ ▶ ░░ │ │ ▶ ░░ │    │
│  │      │ │      │ │      │ │      │ │      │ │      │    │
│  │ 2019 │ │ 2019 │ │ 2018 │ │ 2020 │ │ 2017 │ │ 2021 │    │
│  │ 🎤   │ │ 🚁   │ │ 🎤   │ │ 📱   │ │ 🎤   │ │ 🚁   │    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘    │
│   interview drone  interview phone  interview drone        │
│                                                            │
│  Virtual scroll — only renders visible thumbnails          │
└────────────────────────────────────────────────────────────┘
```

### Conversational Search (LLM Enabled)

```
┌────────────────────────────────────────────────────────────┐
│  💬 Ask a question about your archive                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ What did the mayor say about parking last year?      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  Claude: In the March 2024 council meeting, Mayor Jensen   │
│  addressed parking concerns, stating "we need to find      │
│  solutions for the town center congestion."                │
│                                                            │
│  Sources:                                                  │
│  ┌──────┐ Council Meeting 2024-03-15 @ 01:22:05           │
│  │ ▶ ░░ │ "...parking situation in the center..."         │
│  └──────┘                                                  │
│                                                            │
│  ┌──────┐ Town Hall Q&A 2024-06-20 @ 00:34:12             │
│  │ ▶ ░░ │ "...working on new parking solutions..."        │
│  └──────┘                                                  │
│                                                            │
│  [Follow-up: ___________________________________]          │
└────────────────────────────────────────────────────────────┘
```

### Video Preview

```
┌────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                      │  │
│  │                   Video Player                       │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│  [▶] advancement━━━━━━━━○━━━━━━━━━━━━━ 00:14:22 / 01:02:00│
│       │    │         │        │                           │
│       Face  Object   Speech   Scene change (AI markers)   │
│                                                           │
│  Transcript (synced):                                     │
│  ... and the mayor stated that [the budget] for next     │
│  year would include provisions for...                     │
│                                                           │
│  Detected:                                                │
│  👤 Mayor Jensen (94%)                                    │
│  📍 Bragernes Church, Drammen (91%)                       │
│  🚁 Drone footage                                         │
│                                                           │
│  [Add to export] [Open in folder] [Copy timecode]         │
└────────────────────────────────────────────────────────────┘
```

### Face Training

```
┌────────────────────────────────────────────────────────────┐
│  Unknown Faces (147)                          [Auto-group] │
│                                                            │
│  Group A (23 similar faces):                               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ...              │
│  │ 😐  │ │ 😐  │ │ 😐  │ │ 😐  │ │ 😐  │                  │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                  │
│  Name: [Mayor Jensen_______] [Apply to all 23]            │
│                                                            │
│  Group B (8 similar faces):                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ...                              │
│  │ 😐  │ │ 😐  │ │ 😐  │                                  │
│  └─────┘ └─────┘ └─────┘                                  │
│  Name: [________________] [Apply to all 8]                │
└────────────────────────────────────────────────────────────┘
```

### Place Training

```
┌────────────────────────────────────────────────────────────┐
│  Unknown Locations (52)                       [Auto-group] │
│                                                            │
│  Group A (15 similar scenes) — GPS: Drammen area           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ...                  │
│  │ ⛪       │ │ ⛪       │ │ ⛪       │                      │
│  │         │ │         │ │         │                      │
│  └─────────┘ └─────────┘ └─────────┘                      │
│  CLIP detected: "church"                                   │
│  OSM suggestion: "Bragernes kirke" (120m from GPS)         │
│                                                            │
│  Name: [Bragernes Church____]                              │
│  Type: [Church ▼]                                          │
│  [Accept OSM suggestion] [Apply to all 15]                 │
│                                                            │
│  Group B (7 similar scenes) — No GPS                       │
│  ┌─────────┐ ┌─────────┐ ...                              │
│  │ 🏛️       │ │ 🏛️       │                                  │
│  └─────────┘ └─────────┘                                  │
│  Name: [________________] Type: [Town Hall ▼]             │
└────────────────────────────────────────────────────────────┘
```

### Settings

```
┌────────────────────────────────────────────────────────────┐
│  Settings                                                  │
│                                                            │
│  ┌─ Storage ─────────────────────────────────────────────┐ │
│  │ Proxy generation: ☑ Enabled   Resolution: [1080p ▼]  │ │
│  │ [Delete all proxies] (frees 234 GB)                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ AI Processing ───────────────────────────────────────┐ │
│  │ Face detection rate:    [1 fps ▼]                    │ │
│  │ Object detection rate:  [2 fps ▼]                    │ │
│  │ CLIP embedding rate:    [Per scene ▼]                │ │
│  │ Min face size:          [80px ▼]                     │ │
│  │ ☑ Adaptive sampling (reduce for static content)      │ │
│  │ ☑ Skip blurry frames                                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ AI Assistant (LLM) ──────────────────────────────────┐ │
│  │ Provider: ○ None  ○ Ollama  ● Claude  ○ OpenAI       │ │
│  │ API Key: [sk-ant-••••••••••••••••••••]               │ │
│  │ ☑ Generate summaries on ingest                       │ │
│  │ ☑ Enable conversational search                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌─ Backup ──────────────────────────────────────────────┐ │
│  │ Target: ○ Manual  ● Scheduled  ○ S3                  │ │
│  │ Path: [/mnt/backup/polybos]                          │ │
│  │ Frequency: [Daily ▼]  Retain: [30 days ▼]            │ │
│  │ [Backup now] [Restore...]                            │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Action | Target |
|--------|--------|
| Keystroke to autocomplete | <50ms |
| First search result visible | <200ms |
| Thumbnail load | <100ms |
| Video playback start | <500ms |
| Hover scrub (sprite sheet) | Instant |
| LLM query response | <3 seconds |

---

## Hardware Requirements

### Minimum (Small Archive)

- CPU: 8 cores
- RAM: 32 GB
- GPU: RTX 3060 12GB (or CPU-only, slower)
- Storage: As needed for archive

### Recommended (TV Station)

- CPU: 16+ cores (Ryzen 9 / Xeon)
- RAM: 64-128 GB
- GPU: RTX 4090 24GB
- Storage: TrueNAS with 50+ TB

### Mac Development / Small Production

- Mac Studio M2 Max/Ultra 64GB
- External Thunderbolt storage or NAS
- All AI models run via MLX (Apple Silicon optimized)

### Processing Estimates (RTX 4090)

Per hour of footage:

| Task | Time |
|------|------|
| Proxy generation | ~5 min |
| Whisper transcription | ~6-10 min |
| Face detection/recognition | ~15-20 min |
| CLIP embeddings | ~10 min |
| Object detection | ~15 min |
| LLM summarization | ~2-3 min |
| **Total (without adaptive)** | **~50-65 min** |
| **Total (with adaptive sampling)** | **~20-40 min** |

---

## API Overview (Open Source Backend)

### REST Endpoints

```
POST   /api/auth/login
POST   /api/auth/logout
GET    /api/auth/me

GET    /api/assets
GET    /api/assets/{id}
POST   /api/assets/ingest
DELETE /api/assets/{id}

GET    /api/search?q={query}
WS     /api/search/stream              # Streaming results
POST   /api/search/ask                 # LLM question answering

GET    /api/persons
POST   /api/persons
PUT    /api/persons/{id}
POST   /api/persons/{id}/faces         # Add face to person

GET    /api/places
POST   /api/places
PUT    /api/places/{id}
POST   /api/places/{id}/references     # Add reference image
GET    /api/places/suggest?lat=&lon=   # OSM lookup

POST   /api/export/edl
POST   /api/export/xml

GET    /api/admin/users
POST   /api/admin/users
PUT    /api/admin/users/{id}
DELETE /api/admin/users/{id}

GET    /api/admin/settings
PUT    /api/admin/settings

POST   /api/admin/backup
GET    /api/admin/backup/status
POST   /api/admin/restore
```

### WebSocket Events

```javascript
// Search streaming
ws.send({ type: 'search', query: 'mayor jensen' })
ws.onmessage = { type: 'result', data: { asset_id, timecode, ... } }
ws.onmessage = { type: 'complete', total: 47 }

// Ingest progress
ws.onmessage = { type: 'ingest_progress', asset_id, stage, percent }
ws.onmessage = { type: 'ingest_complete', asset_id }

// AI processing
ws.onmessage = { type: 'ai_progress', asset_id, task, percent }
ws.onmessage = { type: 'ai_complete', asset_id, task }

// LLM response streaming
ws.send({ type: 'ask', question: 'What did the mayor say?' })
ws.onmessage = { type: 'llm_chunk', text: 'In the March...' }
ws.onmessage = { type: 'llm_sources', clips: [...] }
ws.onmessage = { type: 'llm_complete' }
```

---

## Development Phases

### Phase 1: PoC (1 week)

- [ ] Project setup, Docker Compose
- [ ] Basic ingest pipeline (watch folder → FFmpeg → storage)
- [ ] Metadata extraction (ffprobe, GPS, device info)
- [ ] Whisper transcription integration
- [ ] PostgreSQL schema, basic search on transcripts
- [ ] Minimal SvelteKit UI: search bar, results list, video preview
- [ ] Test with small dataset (~10 clips)

### Phase 2: Core AI Features (2-3 weeks)

- [ ] Scene detection (PySceneDetect)
- [ ] Adaptive frame sampling
- [ ] Face detection + recognition (DeepFace)
- [ ] Face training UI (naming workflow)
- [ ] CLIP visual search
- [ ] Object detection (RT-DETR)
- [ ] Vector search (Qdrant)
- [ ] Hybrid search ranking

### Phase 3: Location Intelligence (1-2 weeks)

- [ ] GPS extraction from metadata
- [ ] Place recognition (CLIP + known places)
- [ ] OpenStreetMap integration
- [ ] Place training UI
- [ ] Device/shot type detection

### Phase 4: LLM Integration (1-2 weeks)

- [ ] Ollama integration (local LLM)
- [ ] Cloud LLM support (Claude, OpenAI)
- [ ] Transcript summarization on ingest
- [ ] Query expansion
- [ ] Conversational search UI
- [ ] Question answering (RAG)

### Phase 5: Production Ready (2-3 weeks)

- [ ] Multi-user authentication (local)
- [ ] Role-based permissions
- [ ] LDAP/SSO integration
- [ ] Bulk import tool
- [ ] EDL/XML export
- [ ] Backup/restore
- [ ] Settings UI
- [ ] Performance optimization
- [ ] Error handling, logging

### Phase 6: Polish (1-2 weeks)

- [ ] UI/UX refinement
- [ ] Documentation
- [ ] Installer / setup wizard
- [ ] Demo video
- [ ] Pilot deployment at TV station

---

## File Structure

```
polybos-media-archive/
├── README.md
├── docker-compose.yml
├── docker-compose.dev.yml
│
├── backend/                    # Open source (MIT)
│   ├── polybos_engine/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app
│   │   ├── config.py
│   │   │
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── assets.py
│   │   │   ├── search.py
│   │   │   ├── persons.py
│   │   │   ├── places.py
│   │   │   ├── export.py
│   │   │   └── admin.py
│   │   │
│   │   ├── models/
│   │   │   ├── asset.py
│   │   │   ├── person.py
│   │   │   ├── place.py
│   │   │   ├── user.py
│   │   │   └── segment.py
│   │   │
│   │   ├── services/
│   │   │   ├── ingest.py
│   │   │   ├── search.py
│   │   │   ├── export.py
│   │   │   ├── geo.py          # OSM integration
│   │   │   ├── llm.py          # LLM abstraction
│   │   │   └── backup.py
│   │   │
│   │   ├── ai/
│   │   │   ├── whisper.py
│   │   │   ├── faces.py
│   │   │   ├── clip.py
│   │   │   ├── objects.py
│   │   │   ├── ocr.py
│   │   │   ├── scenes.py
│   │   │   └── device.py       # Device/shot type detection
│   │   │
│   │   ├── workers/
│   │   │   ├── ingest_worker.py
│   │   │   └── ai_worker.py
│   │   │
│   │   └── db/
│   │       ├── database.py
│   │       ├── migrations/
│   │       └── vector.py
│   │
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/                   # Closed source (commercial)
│   ├── src/
│   │   ├── routes/
│   │   │   ├── +page.svelte           # Search home
│   │   │   ├── +layout.svelte
│   │   │   ├── asset/
│   │   │   │   └── [id]/+page.svelte  # Asset detail
│   │   │   ├── faces/
│   │   │   │   └── +page.svelte       # Face training
│   │   │   ├── places/
│   │   │   │   └── +page.svelte       # Place training
│   │   │   ├── ask/
│   │   │   │   └── +page.svelte       # Conversational search
│   │   │   ├── admin/
│   │   │   │   ├── +page.svelte
│   │   │   │   ├── users/+page.svelte
│   │   │   │   └── settings/+page.svelte
│   │   │   └── login/+page.svelte
│   │   │
│   │   ├── lib/
│   │   │   ├── components/
│   │   │   │   ├── SearchBar.svelte
│   │   │   │   ├── ResultGrid.svelte
│   │   │   │   ├── VideoPlayer.svelte
│   │   │   │   ├── FaceTrainer.svelte
│   │   │   │   ├── PlaceTrainer.svelte
│   │   │   │   ├── ConversationalSearch.svelte
│   │   │   │   └── ExportDialog.svelte
│   │   │   │
│   │   │   ├── stores/
│   │   │   │   ├── auth.ts
│   │   │   │   ├── search.ts
│   │   │   │   └── assets.ts
│   │   │   │
│   │   │   └── api.ts
│   │   │
│   │   └── app.html
│   │
│   ├── static/
│   ├── package.json
│   ├── svelte.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
│
└── docs/
    ├── installation.md
    ├── configuration.md
    ├── api.md
    └── deployment.md
```

---

## AI Backend Abstraction

Design AI modules with swappable backends for Mac/Linux compatibility:

```python
# Example: Transcription backend abstraction

from abc import ABC, abstractmethod
from pathlib import Path

class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript:
        pass

class WhisperMLX(TranscriptionBackend):
    """Mac Apple Silicon via MLX"""
    def transcribe(self, audio_path: Path) -> Transcript:
        import mlx_whisper
        return mlx_whisper.transcribe(audio_path, model="large-v3")

class WhisperCUDA(TranscriptionBackend):
    """NVIDIA GPU via faster-whisper"""
    def transcribe(self, audio_path: Path) -> Transcript:
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="cuda")
        return model.transcribe(audio_path)

class WhisperCPU(TranscriptionBackend):
    """Fallback CPU implementation"""
    def transcribe(self, audio_path: Path) -> Transcript:
        import whisper
        model = whisper.load_model("medium")  # Smaller for CPU
        return model.transcribe(audio_path)

# Factory
def get_transcription_backend() -> TranscriptionBackend:
    if is_apple_silicon():
        return WhisperMLX()
    elif has_cuda():
        return WhisperCUDA()
    else:
        return WhisperCPU()
```

Same pattern for:
- Face recognition (DeepFace with different backends)
- CLIP (MLX-CLIP vs OpenCLIP)
- LLM (Ollama vs Claude API vs OpenAI)

---

## Licensing Summary

| Component | License | Commercial OK |
|-----------|---------|---------------|
| Whisper | MIT | ✅ |
| DeepFace + Facenet | MIT | ✅ |
| CLIP / OpenCLIP | MIT | ✅ |
| RT-DETR / Grounding DINO | Apache 2.0 | ✅ |
| PaddleOCR / docTR | Apache 2.0 | ✅ |
| PySceneDetect | BSD | ✅ |
| PostgreSQL | PostgreSQL | ✅ |
| Qdrant | Apache 2.0 | ✅ |
| FFmpeg | LGPL | ✅ (dynamic linking) |
| FastAPI | MIT | ✅ |
| SvelteKit | MIT | ✅ |
| Ollama | MIT | ✅ |
| OpenStreetMap data | ODbL | ✅ (with attribution) |

All clear for commercial use.

---

## Notes

### OpenStreetMap Attribution

If using OSM data, must display attribution:
"© OpenStreetMap contributors"

### LLM Privacy

When using cloud LLMs (Claude/OpenAI):
- Transcripts are sent to external servers
- Consider data sensitivity
- Offer local LLM option for privacy-conscious users

### GPS Privacy

Some footage may have sensitive location data:
- Option to strip GPS on ingest
- Access controls for location data
