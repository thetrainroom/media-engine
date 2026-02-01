"""ETA prediction system for batch processing."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from media_engine.batch.models import BatchRequest

logger = logging.getLogger(__name__)

# Historical timing data for ETA predictions
# Key: (extractor, resolution_bucket) -> list of processing times in seconds
_timing_history: dict[tuple[str, str], list[float]] = {}
_timing_history_lock = threading.Lock()
_timing_history_dirty = False  # Track if we need to save
_timing_history_last_save = 0.0  # Last save timestamp

# Keep last N samples per bucket for rolling average
_MAX_TIMING_SAMPLES = 20

# Timing history persistence
_TIMING_HISTORY_FILE = Path.home() / ".config" / "polybos" / "timing_history.json"
_TIMING_SAVE_INTERVAL = 30.0  # Save at most every 30 seconds


def load_timing_history() -> None:
    """Load timing history from disk on startup."""
    global _timing_history
    if not _TIMING_HISTORY_FILE.exists():
        return
    try:
        with open(_TIMING_HISTORY_FILE) as f:
            data = json.load(f)
        # Convert string keys back to tuples
        with _timing_history_lock:
            for key_str, values in data.items():
                # Key format: "extractor|resolution"
                parts = key_str.split("|")
                if len(parts) == 2:
                    _timing_history[(parts[0], parts[1])] = values[-_MAX_TIMING_SAMPLES:]
        logger.info(f"Loaded timing history: {len(_timing_history)} buckets")
    except Exception as e:
        logger.warning(f"Failed to load timing history: {e}")


def save_timing_history() -> None:
    """Save timing history to disk."""
    global _timing_history_dirty, _timing_history_last_save
    with _timing_history_lock:
        if not _timing_history:
            return
        # Convert tuple keys to strings for JSON
        data = {f"{k[0]}|{k[1]}": v for k, v in _timing_history.items()}
    try:
        _TIMING_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_TIMING_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        _timing_history_dirty = False
        _timing_history_last_save = time.time()
        logger.debug(f"Saved timing history: {len(data)} buckets")
    except Exception as e:
        logger.warning(f"Failed to save timing history: {e}")


def get_resolution_bucket(width: int | None, height: int | None) -> str:
    """Get resolution bucket for timing predictions."""
    if width is None or height is None:
        return "unknown"
    pixels = width * height
    if pixels <= 921600:  # 1280x720
        return "720p"
    elif pixels <= 2073600:  # 1920x1080
        return "1080p"
    elif pixels <= 3686400:  # 2560x1440
        return "1440p"
    elif pixels <= 8294400:  # 3840x2160
        return "4k"
    elif pixels <= 14745600:  # 5120x2880
        return "5k"
    else:
        return "8k+"


def record_timing(
    extractor: str,
    resolution_bucket: str,
    seconds: float,
    units: float | None = None,
) -> None:
    """Record processing rate for future ETA predictions.

    Args:
        extractor: Name of the extractor (transcript, visual, objects, etc.)
        resolution_bucket: Resolution category (720p, 1080p, 4k, etc.)
        seconds: Wall clock time to process
        units: Normalization units - depends on extractor:
            - transcript: duration in minutes (stores seconds per minute)
            - visual: number of timestamps (stores seconds per timestamp)
            - objects/faces/ocr/clip: number of frames (stores seconds per frame)
            - If None, stores raw seconds (for metadata, telemetry, etc.)
    """
    global _timing_history_dirty

    # Calculate rate (seconds per unit) or use raw seconds
    if units and units > 0:
        rate = seconds / units
    else:
        rate = seconds

    key = (extractor, resolution_bucket)
    with _timing_history_lock:
        if key not in _timing_history:
            _timing_history[key] = []
        _timing_history[key].append(rate)
        # Keep only recent samples
        if len(_timing_history[key]) > _MAX_TIMING_SAMPLES:
            _timing_history[key] = _timing_history[key][-_MAX_TIMING_SAMPLES:]
        sample_count = len(_timing_history[key])
        avg = sum(_timing_history[key]) / sample_count
        _timing_history_dirty = True

        unit_label = "/unit" if units else "s"
        logger.debug(f"Recorded timing: {extractor}@{resolution_bucket} = {rate:.2f}{unit_label} " f"(avg: {avg:.2f}{unit_label} from {sample_count} samples)")
    # Save periodically (not on every update to reduce disk I/O)
    if _timing_history_dirty and time.time() - _timing_history_last_save > _TIMING_SAVE_INTERVAL:
        save_timing_history()


def get_predicted_rate(extractor: str, resolution_bucket: str) -> float | None:
    """Get predicted processing rate based on historical data.

    Returns the average rate (seconds per unit) for the given extractor and resolution.
    Multiply by the number of units to get predicted time.
    """
    key = (extractor, resolution_bucket)
    with _timing_history_lock:
        if key in _timing_history and _timing_history[key]:
            return sum(_timing_history[key]) / len(_timing_history[key])
    return None


# Default processing rates (seconds per unit) when no historical data
# Used as fallback for ETA predictions
DEFAULT_RATES: dict[str, float] = {
    "metadata": 1.0,  # ~1 second per file
    "telemetry": 0.5,  # ~0.5 seconds per file
    "vad": 0.5,  # ~0.5 seconds per minute of video
    # Sub-extractors within visual_processing (per frame rates)
    "motion": 0.5,  # ~0.5 seconds per file (analyzes whole video)
    "scenes": 0.3,  # ~0.3 seconds per file
    "frame_decode": 0.05,  # ~0.05 seconds per frame
    "objects": 0.3,  # ~0.3 seconds per frame (YOLO)
    "faces": 0.2,  # ~0.2 seconds per frame
    "ocr": 0.3,  # ~0.3 seconds per frame
    "clip": 0.15,  # ~0.15 seconds per frame
    # Separate stages
    "visual": 5.0,  # ~5 seconds per timestamp (Qwen VLM is slow)
    "transcript": 3.0,  # ~3 seconds per minute of video
}

# Extractor processing order - must match run_batch_job()
EXTRACTOR_ORDER = [
    "metadata",
    "telemetry",
    "vad",
    "visual_processing",  # Combined: motion, scenes, frame_decode, objects, faces, ocr, clip
    "visual",  # Qwen VLM
    "transcript",
]


def predict_extractor_time(
    extractor: str,
    resolution_bucket: str,
    duration_seconds: float,
    num_frames: int | None = None,
    num_timestamps: int | None = None,
    enabled_sub_extractors: set[str] | None = None,
) -> float:
    """Predict processing time for a single extractor on a single file.

    Args:
        extractor: Name of the extractor
        resolution_bucket: Resolution category (720p, 1080p, 4k, etc.)
        duration_seconds: Video duration in seconds
        num_frames: Number of frames to process (for frame-based extractors)
        num_timestamps: Number of timestamps for visual/VLM analysis
        enabled_sub_extractors: For visual_processing, which sub-extractors are enabled

    Returns:
        Predicted processing time in seconds
    """
    # Duration in minutes for duration-based extractors
    duration_minutes = duration_seconds / 60.0

    # Extractors that scale with duration
    if extractor in ("vad", "transcript"):
        rate = get_predicted_rate(extractor, resolution_bucket)
        if rate is None:
            rate = DEFAULT_RATES.get(extractor, 1.0)
        return rate * duration_minutes

    # visual_processing: sum up time for each enabled sub-extractor
    if extractor == "visual_processing":
        total_time = 0.0
        sub_extractors = enabled_sub_extractors or {"motion", "scenes", "frame_decode", "objects", "faces", "ocr", "clip"}

        # Smart sampling typically uses ~20-50 frames, not duration*2
        # Use a more conservative estimate
        estimated_frames = num_frames if num_frames else min(50, max(10, int(duration_seconds / 2)))

        for sub in sub_extractors:
            rate = get_predicted_rate(sub, resolution_bucket)
            if rate is None:
                rate = DEFAULT_RATES.get(sub, 0.1)

            # motion, scenes store raw seconds per file
            # frame_decode, objects, faces, ocr, clip store seconds per frame
            if sub in ("motion", "scenes"):
                total_time += rate  # raw seconds per file
            else:
                total_time += rate * estimated_frames  # per-frame rate × frame count

        return total_time

    # Visual/Qwen scales with timestamps
    if extractor == "visual":
        rate = get_predicted_rate(extractor, resolution_bucket)
        if rate is None:
            rate = DEFAULT_RATES.get(extractor, 5.0)
        timestamps = num_timestamps if num_timestamps else 5
        return rate * timestamps

    # Fixed-time extractors (metadata, telemetry)
    rate = get_predicted_rate(extractor, resolution_bucket)
    if rate is None:
        rate = DEFAULT_RATES.get(extractor, 1.0)
    return rate


def get_enabled_extractors_from_request(
    request: BatchRequest,
) -> tuple[set[str], set[str]]:
    """Get the set of enabled extractors from a batch request.

    Returns:
        Tuple of (main_extractors, sub_extractors within visual_processing)
    """
    enabled = {"metadata", "telemetry"}  # Always enabled
    sub_extractors: set[str] = set()

    if request.enable_vad:
        enabled.add("vad")

    # Track which sub-extractors are enabled within visual_processing
    if request.enable_motion:
        sub_extractors.add("motion")
    if request.enable_scenes:
        sub_extractors.add("scenes")
    if request.enable_objects:
        sub_extractors.update({"frame_decode", "objects"})
    if request.enable_faces:
        sub_extractors.update({"frame_decode", "faces"})
    if request.enable_ocr:
        sub_extractors.update({"frame_decode", "ocr"})
    if request.enable_clip:
        sub_extractors.update({"frame_decode", "clip"})

    # visual_processing runs if any sub-extractor is enabled
    if sub_extractors:
        enabled.add("visual_processing")

    if request.enable_visual:
        enabled.add("visual")
    if request.enable_transcript:
        enabled.add("transcript")

    return enabled, sub_extractors


def calculate_queue_eta() -> tuple[float, int]:
    """Calculate total ETA for all queued batches.

    Returns: (total_seconds, batch_count)
    """
    from media_engine.batch.state import batch_queue, batch_queue_lock

    total_eta = 0.0
    batch_count = 0

    with batch_queue_lock:
        for _, request in batch_queue:
            batch_count += 1
            enabled, sub_extractors = get_enabled_extractors_from_request(request)

            # Estimate time for each file in queued batch
            for _file_path in request.files:
                # Use average duration estimate if metadata not yet available
                duration = 60.0  # Default: 1 minute estimate
                resolution = "1080p"  # Default resolution

                for ext in enabled:
                    if ext in EXTRACTOR_ORDER:
                        total_eta += predict_extractor_time(
                            ext,
                            resolution,
                            duration,
                            enabled_sub_extractors=sub_extractors if ext == "visual_processing" else None,
                        )

    return total_eta, batch_count


# Load timing history on module import
load_timing_history()
