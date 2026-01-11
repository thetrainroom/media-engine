"""Base utilities for metadata extraction."""

import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from polybos_engine.schemas import (
    GPS,
    AudioInfo,
    Codec,
    ColorSpace,
    DetectionMethod,
    LensInfo,
    Metadata,
    Resolution,
    VideoCodec,
)

logger = logging.getLogger(__name__)

# Pool for parallel ffprobe calls
_ffprobe_pool: ThreadPoolExecutor | None = None

# Number of workers based on CPU cores (leave 2 cores free, minimum 2 workers)
FFPROBE_WORKERS = max(2, (os.cpu_count() or 4) - 2)

# Timeout for ffprobe calls (seconds)
FFPROBE_TIMEOUT = 30


def get_ffprobe_pool() -> ThreadPoolExecutor:
    """Get or create the ffprobe thread pool."""
    global _ffprobe_pool
    if _ffprobe_pool is None:
        _ffprobe_pool = ThreadPoolExecutor(
            max_workers=FFPROBE_WORKERS,
            thread_name_prefix="ffprobe"
        )
        logger.info(f"Created ffprobe pool with {FFPROBE_WORKERS} workers")
    return _ffprobe_pool


def shutdown_ffprobe_pool() -> None:
    """Shutdown the ffprobe pool (call on app shutdown)."""
    global _ffprobe_pool
    if _ffprobe_pool is not None:
        _ffprobe_pool.shutdown(wait=False)
        _ffprobe_pool = None


@dataclass
class GPSCoordinates:
    """Parsed GPS coordinates from ISO 6709 format."""

    latitude: float
    longitude: float
    altitude: float | None = None


@dataclass
class SidecarMetadata:
    """Metadata extracted from sidecar files."""

    device: Any | None = None  # DeviceInfo
    gps: GPS | None = None
    color_space: ColorSpace | None = None
    lens: LensInfo | None = None


def run_ffprobe(file_path: str) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output.

    Args:
        file_path: Path to the media file
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
    ]

    # Note: -select_streams with comma syntax (v:0,a:0) doesn't work reliably
    # across ffprobe versions, so we get all streams and filter in code

    cmd.append(file_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=FFPROBE_TIMEOUT
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out after {FFPROBE_TIMEOUT}s for {file_path}")
        raise RuntimeError(f"ffprobe timed out for {file_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr}")
        raise RuntimeError(f"ffprobe failed for {file_path}: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")


def run_ffprobe_batch(
    file_paths: list[str],
) -> dict[str, dict[str, Any] | Exception]:
    """Run ffprobe on multiple files in parallel.

    Args:
        file_paths: List of file paths to probe

    Returns:
        Dict mapping file path to probe result or Exception if failed
    """
    if not file_paths:
        return {}

    pool = get_ffprobe_pool()
    futures = {
        pool.submit(run_ffprobe, path): path
        for path in file_paths
    }

    results: dict[str, dict[str, Any] | Exception] = {}
    for future in as_completed(futures):
        path = futures[future]
        try:
            results[path] = future.result()
        except Exception as e:
            logger.warning(f"ffprobe failed for {path}: {e}")
            results[path] = e

    return results


def parse_fps(video_stream: dict[str, Any]) -> float | None:
    """Parse frame rate from video stream."""
    # Try avg_frame_rate first
    fps_str = video_stream.get("avg_frame_rate", "")
    if fps_str and "/" in fps_str:
        num, den = fps_str.split("/")
        if int(den) != 0:
            return round(int(num) / int(den), 2)

    # Fall back to r_frame_rate
    fps_str = video_stream.get("r_frame_rate", "")
    if fps_str and "/" in fps_str:
        num, den = fps_str.split("/")
        if int(den) != 0:
            return round(int(num) / int(den), 2)

    return None


def parse_bit_depth(video_stream: dict[str, Any]) -> int | None:
    """Parse bit depth from video stream."""
    # Try bits_per_raw_sample first
    bits = video_stream.get("bits_per_raw_sample")
    if bits:
        try:
            return int(bits)
        except ValueError:
            pass

    # Parse from pixel format (e.g., yuv420p10le, yuv422p10be)
    pix_fmt = video_stream.get("pix_fmt", "")
    if pix_fmt:
        match = re.search(r"(\d+)(le|be)?$", pix_fmt)
        if match:
            depth = int(match.group(1))
            if depth in (10, 12, 16):
                return depth
        if pix_fmt in ("yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p"):
            return 8

    return None


def extract_timecode(
    tags: dict[str, str], video_stream: dict[str, Any] | None
) -> str | None:
    """Extract start timecode from metadata."""
    tags_lower = {k.lower(): v for k, v in tags.items()}

    tc = tags_lower.get("timecode")
    if tc:
        return tc

    if video_stream:
        stream_tags = video_stream.get("tags", {})
        stream_tags_lower = {k.lower(): v for k, v in stream_tags.items()}
        tc = stream_tags_lower.get("timecode")
        if tc:
            return tc

    return None


def parse_creation_time(
    tags: dict[str, str], stream_tags: dict[str, str] | None = None
) -> datetime | None:
    """Parse creation time from metadata tags.

    Checks format-level tags first, then stream tags as fallback.
    Normalizes keys to lowercase for case-insensitive lookup.
    """
    # Normalize keys to lowercase for case-insensitive lookup
    tags_lower = {k.lower(): v for k, v in tags.items()}

    time_str = (
        tags_lower.get("creation_time")
        or tags_lower.get("date")
        or tags_lower.get("com.apple.quicktime.creationdate")
        or tags_lower.get("date_recorded")
        or tags_lower.get("date-eng")  # Some MKV files
    )

    # Fallback to stream tags if format tags don't have the date
    if not time_str and stream_tags:
        stream_tags_lower = {k.lower(): v for k, v in stream_tags.items()}
        time_str = (
            stream_tags_lower.get("creation_time")
            or stream_tags_lower.get("date")
        )

    if not time_str:
        return None

    # Handle timezone suffixes by stripping them for parsing
    # ffprobe can return: "2024-06-15T10:30:00.000000Z"
    #                  or "2024-06-15 10:30:00+0200"
    #                  or "2024-06-15T10:30:00+02:00"
    time_str_clean = time_str.strip()

    # Remove timezone offset for parsing (we'll treat as UTC if present)
    # Patterns like +0200, +02:00, -0500, -05:00
    tz_pattern = r"[+-]\d{2}:?\d{2}$"
    time_str_no_tz = re.sub(tz_pattern, "", time_str_clean)

    formats = [
        ("%Y-%m-%dT%H:%M:%S.%f", None),  # Variable microseconds
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y:%m:%d %H:%M:%S", 19),  # EXIF format
        ("%Y/%m/%d %H:%M:%S", 19),
        ("%d/%m/%Y %H:%M:%S", 19),  # European format
        ("%Y-%m-%d", 10),  # Date only
    ]

    for fmt, length in formats:
        try:
            if length:
                return datetime.strptime(time_str_no_tz[:length], fmt)
            else:
                # Variable length (for microseconds)
                # Find the 'T' and parse accordingly
                if "T" in time_str_no_tz:
                    return datetime.strptime(time_str_no_tz.rstrip("Z"), fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse creation time: {time_str}")
    return None


def parse_iso6709(location: str) -> GPSCoordinates | None:
    """Parse ISO 6709 format GPS coordinates."""
    pattern = r"([+-]\d+\.?\d*)"
    matches = re.findall(pattern, location)

    if len(matches) >= 2:
        try:
            return GPSCoordinates(
                latitude=float(matches[0]),
                longitude=float(matches[1]),
                altitude=float(matches[2]) if len(matches) >= 3 else None,
            )
        except ValueError:
            pass

    return None


def parse_dms_coordinate(dms: str, ref: str | None) -> float | None:
    """Parse DMS (degrees;minutes;seconds) format to decimal degrees.

    Handles multiple formats:
    - 63;6;38.880 (all semicolons)
    - 63;6:38.880 (mixed semicolon and colon)
    """
    try:
        # Normalize: replace colons with semicolons
        normalized = dms.replace(":", ";")
        parts = normalized.split(";")
        if len(parts) != 3:
            return float(dms)

        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])

        decimal = degrees + minutes / 60 + seconds / 3600

        if ref in ("S", "W"):
            decimal = -decimal

        return decimal
    except (ValueError, IndexError):
        return None


def extract_gps_from_tags(tags: dict[str, str]) -> GPS | None:
    """Extract GPS coordinates from metadata tags."""
    tags_lower = {k.lower(): v for k, v in tags.items()}

    location = (
        tags_lower.get("location")
        or tags_lower.get("com.apple.quicktime.location.iso6709")
        or tags_lower.get("gps")
    )

    if location:
        coords = parse_iso6709(location)
        if coords:
            return GPS(
                latitude=coords.latitude,
                longitude=coords.longitude,
                altitude=coords.altitude,
            )

    lat = tags_lower.get("gps_latitude") or tags_lower.get("location-latitude")
    lon = tags_lower.get("gps_longitude") or tags_lower.get("location-longitude")

    if lat and lon:
        try:
            return GPS(
                latitude=float(lat),
                longitude=float(lon),
                altitude=float(tags_lower.get("gps_altitude", 0)) or None,
            )
        except ValueError:
            pass

    return None


def extract_color_space_from_stream(
    video_stream: dict[str, Any] | None, tags: dict[str, str]
) -> ColorSpace | None:
    """Extract color space information from video stream and format tags."""
    transfer: str | None = None
    primaries: str | None = None
    matrix: str | None = None

    if video_stream:
        transfer = video_stream.get("color_transfer")
        primaries = video_stream.get("color_primaries")
        matrix = video_stream.get("color_space")

    tags_lower = {k.lower(): v for k, v in tags.items()}
    custom_gamma = tags_lower.get("com.apple.proapps.customgamma", "")
    if custom_gamma:
        parts = custom_gamma.split(".")
        if parts:
            transfer = parts[-1]

    if not (transfer or primaries or matrix):
        return None

    return ColorSpace(
        transfer=transfer,
        primaries=primaries,
        matrix=matrix,
        detection_method=DetectionMethod.METADATA,
    )


def build_base_metadata(
    probe_data: dict[str, Any],
    file_path: str,
) -> Metadata:
    """Build base metadata from ffprobe data without device-specific processing."""
    format_info = probe_data.get("format", {})
    tags = format_info.get("tags", {})

    video_stream = None
    audio_stream = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    resolution = Resolution(
        width=video_stream.get("width", 0) if video_stream else 0,
        height=video_stream.get("height", 0) if video_stream else 0,
    )

    codec = Codec(
        video=video_stream.get("codec_name") if video_stream else None,
        audio=audio_stream.get("codec_name") if audio_stream else None,
    )

    video_codec: VideoCodec | None = None
    if video_stream:
        video_codec = VideoCodec(
            name=video_stream.get("codec_name", "unknown"),
            profile=video_stream.get("profile"),
            bit_depth=parse_bit_depth(video_stream),
            pixel_format=video_stream.get("pix_fmt"),
        )

    audio_info: AudioInfo | None = None
    if audio_stream:
        audio_info = AudioInfo(
            codec=audio_stream.get("codec_name"),
            sample_rate=int(audio_stream.get("sample_rate", 0)) or None,
            channels=audio_stream.get("channels"),
            bit_depth=audio_stream.get("bits_per_sample")
            or audio_stream.get("bits_per_raw_sample"),
            bitrate=int(audio_stream.get("bit_rate", 0)) or None,
        )

    fps = parse_fps(video_stream) if video_stream else None
    duration = float(format_info.get("duration", 0))
    bitrate = int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None
    file_size = os.path.getsize(file_path)

    # Get stream tags for fallback date extraction
    video_stream_tags = video_stream.get("tags", {}) if video_stream else None
    created_at = parse_creation_time(tags, video_stream_tags)
    timecode = extract_timecode(tags, video_stream)
    gps = extract_gps_from_tags(tags)
    color_space = extract_color_space_from_stream(video_stream, tags)

    return Metadata(
        duration=duration,
        resolution=resolution,
        codec=codec,
        video_codec=video_codec,
        audio=audio_info,
        fps=fps,
        bitrate=bitrate,
        file_size=file_size,
        timecode=timecode,
        created_at=created_at,
        device=None,
        gps=gps,
        color_space=color_space,
        lens=None,
    )
