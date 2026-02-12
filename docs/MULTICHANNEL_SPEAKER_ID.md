# Multi-Channel Audio Speaker Identification

Design document for using multi-channel audio from dual-mic and multi-mic camera setups to improve speaker identification accuracy.

## Problem

The current pipeline forces all audio to mono before any analysis:

```
# vad.py:56
"-ac", "1"

# transcribe.py:433
"-ac", "1"
```

This discards channel separation before VAD, transcription, and diarization. When a stereo lavalier or dual-mic setup is connected to a camera, each channel carries a different microphone's signal. The speaker closest to a given mic will be significantly louder on that channel. This is a strong identification signal that the current pipeline throws away.

**Use cases:**
- **2-channel stereo lav** (common): one lav per channel, plugged into camera's stereo input
- **Professional cameras** (Sony PXW, Canon C-series, Blackmagic, ARRI): up to 4-8 XLR/mini-XLR inputs, each recorded as a separate channel
- **Field recorders** (Zoom, Sound Devices): multi-track recordings with dedicated channels per mic

## Current Pipeline

### Audio extraction

Both VAD and transcription extract audio with ffmpeg, forcing mono:

```
ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

- `vad.py` `_extract_audio()` (line 35): extracts 16kHz mono WAV for WebRTC VAD
- `transcribe.py` `extract_audio()` (line 410): extracts 16kHz mono WAV for Whisper

### Speaker diarization

`run_diarization()` in `transcribe.py:333` runs pyannote on the mono audio. It returns speaker labels (`SPEAKER_00`, `SPEAKER_01`) and time segments, but no embeddings.

`assign_speakers_to_segments()` in `transcribe.py:375` maps speaker labels to transcript segments by maximum temporal overlap.

### Channel count detection

ffprobe already detects channel count in `metadata/base.py:652`:

```python
audio_info = AudioInfo(
    channels=audio_stream.get("channels"),  # 1, 2, 4, 6, 8, etc.
)
```

This value is available in `Metadata.audio.channels` but is never used by the transcription pipeline.

### Speaker embeddings gap

The Rust backend defines `SpeakerInfo` (`ai_results.rs:148`) expecting `label`, `embedding`, and `total_duration`. The `TranscriptResult` struct has a `speakers: Option<Vec<SpeakerInfo>>` field. However, the Python engine's `Transcript` schema does not include a `speakers` field, so `tr.speakers` is always `None` on the Rust side. The `process_speaker_detections()` function in the Rust backend is fully implemented and ready to consume speaker embeddings, but never receives them.

## Proposed Architecture

### Overview

```
Input file (N channels)
    |
    v
[Metadata stage] --- detects audio.channels = N
    |
    v
[Transcript stage] --- receives channel count
    |
    +-- channels >= 2?
    |       |
    |       v
    |   Extract per-channel audio
    |   Compute per-segment energy per channel
    |   Determine if channels are meaningfully different
    |       |
    |       +-- Channels differ? --> Channel-aware speaker assignment
    |       |                        (dominant channel = speaker identity)
    |       |
    |       +-- Channels similar? --> Fall back to pyannote diarization
    |
    +-- channels == 1?
            |
            v
        Existing mono pipeline (pyannote diarization)
    |
    v
Transcript with speaker labels
```

### Step 1: Per-channel audio extraction

When `audio.channels >= 2`, extract each channel as a separate mono file:

```
ffmpeg -i input.mp4 -vn -ar 16000 -af "pan=mono|c0=c0" -c:a pcm_s16le ch0.wav
ffmpeg -i input.mp4 -vn -ar 16000 -af "pan=mono|c0=c1" -c:a pcm_s16le ch1.wav
...
ffmpeg -i input.mp4 -vn -ar 16000 -af "pan=mono|c0=c{N-1}" -c:a pcm_s16le ch{N-1}.wav
```

Also extract the standard mono mix for Whisper transcription (as today).

### Step 2: Detect active channels

Not all channels carry useful signal. An 8-channel camera might only have 2 mics connected.

For each channel file, compute overall RMS energy and speech ratio (reuse WebRTC VAD). Classify channels as:
- **Active**: has speech (speech_ratio > 0.1, energy above noise floor)
- **Inactive**: silence or noise only

Discard inactive channels from further analysis.

### Step 3: Per-segment energy analysis

For each transcript segment (from Whisper on the mono mix):

1. Read the corresponding time range from each active channel's WAV
2. Compute RMS energy for that time range per channel
3. Determine the **dominant channel** (highest energy)
4. Compute a **channel confidence** score: ratio of dominant channel energy to second-highest

```python
def get_dominant_channel(
    channel_wavs: list[np.ndarray],  # per-channel audio arrays
    start_sample: int,
    end_sample: int,
) -> tuple[int, float]:
    """Returns (channel_index, confidence) for the dominant channel."""
    energies = []
    for wav in channel_wavs:
        segment = wav[start_sample:end_sample]
        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
        energies.append(rms)

    sorted_e = sorted(energies, reverse=True)
    dominant = np.argmax(energies)

    # Confidence: how much louder is the dominant channel?
    if sorted_e[1] > 0:
        confidence = sorted_e[0] / sorted_e[1]  # ratio > 1.0
    else:
        confidence = float('inf')

    return int(dominant), confidence
```

### Step 4: Channel-aware vs pyannote decision

Before assigning speakers, evaluate whether channels are meaningfully separated:

```python
# Across all speech segments, compute average channel confidence
avg_confidence = mean(confidences)

if avg_confidence > CHANNEL_CONFIDENCE_THRESHOLD:  # e.g., 1.5 = dominant is 50% louder
    # Channels carry distinct signals -> use channel-aware assignment
    use_channel_mode = True
else:
    # Channels are similar (shared mic, ambient) -> fall back to pyannote
    use_channel_mode = False
```

**Threshold tuning**: A ratio of 1.5 means the dominant channel is 50% louder (about 3.5 dB). This is conservative; real dual-lav setups typically show 6-20 dB difference. Start at 1.5 and adjust based on testing.

### Step 5: Speaker assignment

**Channel-aware mode**: Each active channel maps to one speaker.

```python
# Group segments by dominant channel
channel_groups: dict[int, list[TranscriptSegment]] = defaultdict(list)
for segment, dominant_ch, confidence in segment_analysis:
    channel_groups[dominant_ch].append(segment)

# Assign speaker labels: CHANNEL_00, CHANNEL_01, etc.
for ch_idx, segments in channel_groups.items():
    for seg in segments:
        seg.speaker = f"CHANNEL_{ch_idx:02d}"
```

**Pyannote fallback**: Use existing `run_diarization()` + `assign_speakers_to_segments()` on mono mix.

### Step 6: Return results

Extend the Python `Transcript` schema to include speaker info:

```python
class SpeakerInfo(BaseModel):
    label: str             # "CHANNEL_00" or "SPEAKER_00"
    embedding: list[float] # voice embedding (from pyannote embedding model)
    total_duration: float  # total speaking time in seconds

class Transcript(BaseModel):
    language: str
    confidence: float
    duration: float
    speaker_count: int | None = None
    speakers: list[SpeakerInfo] | None = None  # NEW
    hints_used: TranscriptHints
    segments: list[TranscriptSegment]
```

The Rust backend already deserializes this via `TranscriptResult.speakers` and processes it in `process_speaker_detections()`.

## Integration Points

### Files to modify

| File | Change |
|------|--------|
| `engine/media_engine/extractors/transcribe.py` | Add `extract_audio_per_channel()`, `analyze_channel_energy()`, modify `extract_transcript()` to accept `audio_channels` param |
| `engine/media_engine/schemas.py` | Add `SpeakerInfo` model, add `speakers` field to `Transcript` |
| `engine/media_engine/batch/processor.py` | Pass `audio_channels` from metadata to `extract_transcript()` |
| `engine/media_engine/extractors/vad.py` | No changes needed (VAD only needs mono for speech detection) |

### Rust backend (no changes needed)

The Rust side is already wired:
- `TranscriptResult.speakers: Option<Vec<SpeakerInfo>>` — deserializes from Python engine JSON
- `process_speaker_detections()` — stores speakers in `media_speakers` table with embeddings
- `cross_reference_faces_and_speakers()` — matches speakers to faces by time overlap
- Voice matching against known `person_voices` — uses cosine similarity on embeddings

## Edge Cases

### Unused channels
An 8-channel camera with only 2 mics connected will have 6 silent channels. The active channel detection (Step 2) filters these out automatically using VAD speech ratio.

### Bleed-through
A lav mic will pick up both speakers, but the wearer's voice will be significantly louder. The dominant channel analysis handles this — the per-segment energy comparison identifies the loudest channel regardless of bleed.

### Mid-recording changes
If mics are swapped or a new speaker joins, the per-segment analysis handles this naturally since each segment is analyzed independently.

### Mono sources
When `audio.channels == 1` (or channel count is unknown), skip channel analysis entirely and use the existing pyannote diarization pipeline.

### Stereo mix (not dual-mono)
Some cameras record a stereo mix from a single mic (left/right pickup). In this case, both channels will have similar energy for all speakers. The confidence threshold in Step 4 will detect this and fall back to pyannote.

### Single speaker on multiple channels
If the same person's mic feeds multiple channels (common in broadcast), the energy analysis will show similar levels across those channels. This could be detected by correlating the channels' waveforms — highly correlated channels likely carry the same source.

## Speaker Embedding Extraction

Independent of the multi-channel feature, the pipeline should extract speaker embeddings to populate the `SpeakerInfo.embedding` field. Two approaches:

### Option A: Pyannote embedding model
Pyannote provides a speaker embedding model (`pyannote/embedding`) that produces fixed-dimensional vectors per audio segment. For each detected speaker, extract their segments, compute an average embedding, and include it in `SpeakerInfo`.

### Option B: Channel-derived embeddings
For channel-aware mode, extract the embedding from the dominant channel's audio for each speaker. This produces cleaner embeddings since the audio is less contaminated by other speakers.

### Recommendation
Implement Option A as the baseline (works for all files), then use Option B when channel-aware mode is active (produces better embeddings from cleaner audio).

## Processing Cost

| Step | Cost | Notes |
|------|------|-------|
| Per-channel extraction | ~1s per channel | ffmpeg, I/O bound |
| Per-channel VAD | ~0.5s per channel | WebRTC VAD, CPU only |
| Energy analysis | negligible | numpy RMS on loaded audio |
| Embedding extraction | ~2-5s total | pyannote embedding model |

Total overhead for a 2-channel file: ~3-5 seconds. For 8 channels: ~10-15 seconds. This is small relative to Whisper transcription (30-120s for a typical video).

## Future Enhancements

- **Channel labels from metadata**: Some cameras embed channel names/roles in metadata (e.g., "Lav 1", "Boom", "Ambient"). Could be parsed and used as speaker hints.
- **Channel-aware Whisper**: Run Whisper separately per channel for better transcription of overlapping speech. Merge results using alignment.
- **Automatic mic type detection**: Classify channels as lav, boom, ambient, or unused based on spectral characteristics and energy patterns.
