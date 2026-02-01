"""Telemetry extraction from drone/camera sidecar files."""

import logging
import re
from datetime import datetime
from pathlib import Path

from media_engine.schemas import TelemetryPoint, TelemetryResult

logger = logging.getLogger(__name__)


def extract_telemetry(
    file_path: str,
    sample_interval: float = 1.0,
) -> TelemetryResult | None:
    """Extract telemetry/flight path from video sidecar files.

    Args:
        file_path: Path to video file
        sample_interval: Sample one point every N seconds (default: 1.0)

    Returns:
        TelemetryResult with GPS track and camera settings, or None if no telemetry
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Try DJI SRT sidecar first (has GPS for drones)
    result = _parse_dji_srt_telemetry(file_path, sample_interval)
    if result:
        return result

    # Try embedded subtitle stream (DJI Pocket, some other cameras)
    result = _parse_embedded_subtitle_telemetry(file_path, sample_interval)
    if result:
        return result

    # TODO: Add Sony NX5 GPS parsing
    # TODO: Add GoPro telemetry parsing

    return None


def _parse_dji_srt_telemetry(
    video_path: str,
    sample_interval: float = 1.0,
) -> TelemetryResult | None:
    """Parse DJI SRT file for full telemetry track.

    DJI SRT format (per frame):
    1
    00:00:00,000 --> 00:00:00,020
    <font size="28">FrameCnt: 1, DiffTime: 20ms
    2025-10-15 11:36:32.281
    [iso: 400] [shutter: 1/100.0] [fnum: 2.8] [ev: 0] [ct: 5790]
    [color_md : d_log] [focal_len: 24.00]
    [latitude: 61.05121] [longitude: 7.81233]
    [rel_alt: 47.100 abs_alt: 380.003] </font>
    """
    path = Path(video_path)

    # Find SRT file
    srt_patterns = [
        path.with_suffix(".SRT"),
        path.with_suffix(".srt"),
    ]

    srt_path = None
    for pattern in srt_patterns:
        if pattern.exists():
            srt_path = pattern
            break

    if not srt_path:
        return None

    try:
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        # Split into subtitle blocks
        blocks = re.split(r"\n\n+", content)

        points: list[TelemetryPoint] = []
        last_timestamp = -sample_interval  # Ensure first point is captured

        for block in blocks:
            # Parse timestamp from SRT format: 00:00:00,000 --> 00:00:01,000
            time_match = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->", block)
            if not time_match:
                continue

            hours = int(time_match.group(1))
            minutes = int(time_match.group(2))
            seconds = int(time_match.group(3))
            millis = int(time_match.group(4))
            timestamp = hours * 3600 + minutes * 60 + seconds + millis / 1000

            # Sample at specified interval
            if timestamp - last_timestamp < sample_interval:
                continue
            last_timestamp = timestamp

            # Parse GPS
            lat_match = re.search(r"\[latitude:\s*([-\d.]+)\]", block)
            lon_match = re.search(r"\[longitude:\s*([-\d.]+)\]", block)

            if not (lat_match and lon_match):
                continue

            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))

            # Skip invalid coordinates
            if lat == 0 and lon == 0:
                continue

            # Parse altitudes
            abs_alt_match = re.search(r"abs_alt:\s*([-\d.]+)", block)
            rel_alt_match = re.search(r"rel_alt:\s*([-\d.]+)", block)

            # Parse datetime: 2025-10-15 11:36:32.281
            recorded_at = None
            dt_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", block)
            if dt_match:
                try:
                    recorded_at = datetime.strptime(dt_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    pass

            # Parse camera settings
            iso_match = re.search(r"\[iso:\s*(\d+)\]", block)
            shutter_match = re.search(r"\[shutter:\s*1/([\d.]+)\]", block)
            fnum_match = re.search(r"\[fnum:\s*([\d.]+)\]", block)
            focal_match = re.search(r"\[focal_len:\s*([\d.]+)\]", block)
            color_match = re.search(r"\[color_md\s*:\s*(\w+)\]", block)

            point = TelemetryPoint(
                timestamp=timestamp,
                recorded_at=recorded_at,
                latitude=lat,
                longitude=lon,
                altitude=float(abs_alt_match.group(1)) if abs_alt_match else None,
                relative_altitude=(float(rel_alt_match.group(1)) if rel_alt_match else None),
                iso=int(iso_match.group(1)) if iso_match else None,
                shutter=1 / float(shutter_match.group(1)) if shutter_match else None,
                aperture=float(fnum_match.group(1)) if fnum_match else None,
                focal_length=float(focal_match.group(1)) if focal_match else None,
                color_mode=color_match.group(1) if color_match else None,
            )
            points.append(point)

        if not points:
            return None

        # Calculate duration from last timestamp
        duration = points[-1].timestamp if points else 0

        return TelemetryResult(
            source="dji_srt",
            sample_rate=1.0 / sample_interval,
            duration=duration,
            points=points,
        )

    except Exception as e:
        logger.warning(f"Error parsing DJI SRT telemetry {srt_path}: {e}")
        return None


def _parse_embedded_subtitle_telemetry(
    video_path: str,
    sample_interval: float = 1.0,
) -> TelemetryResult | None:
    """Parse embedded subtitle stream for telemetry (DJI Pocket, etc.).

    Some cameras embed telemetry as subtitle tracks instead of external SRT files.
    Format is similar to SRT: F/1.8, SS 293.11, ISO 110, EV -0.3
    """
    import subprocess

    try:
        # Extract subtitle stream using ffmpeg
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-map",
            "0:s:0",
            "-f",
            "srt",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not result.stdout:
            return None

        content = result.stdout

        # Parse subtitle blocks (simpler format than full SRT)
        # Format: F/1.8, SS 293.11, ISO 110, EV -0.3,
        blocks = re.split(r"\n\n+", content)

        points: list[TelemetryPoint] = []
        last_timestamp = -sample_interval

        for block in blocks:
            # Parse timestamp
            time_match = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->", block)
            if not time_match:
                continue

            hours = int(time_match.group(1))
            minutes = int(time_match.group(2))
            seconds = int(time_match.group(3))
            millis = int(time_match.group(4))
            timestamp = hours * 3600 + minutes * 60 + seconds + millis / 1000

            if timestamp - last_timestamp < sample_interval:
                continue
            last_timestamp = timestamp

            # Parse DJI Pocket format: F/1.8, SS 293.11, ISO 110, EV -0.3
            aperture_match = re.search(r"F/([\d.]+)", block)
            shutter_match = re.search(r"SS\s+([\d.]+)", block)
            iso_match = re.search(r"ISO\s+(\d+)", block)

            # This format doesn't have GPS, but has camera settings
            if aperture_match or iso_match:
                point = TelemetryPoint(
                    timestamp=timestamp,
                    latitude=0.0,  # No GPS in this format
                    longitude=0.0,
                    aperture=float(aperture_match.group(1)) if aperture_match else None,
                    shutter=(1 / float(shutter_match.group(1)) if shutter_match else None),
                    iso=int(iso_match.group(1)) if iso_match else None,
                )
                points.append(point)

        if not points:
            return None

        # Filter out points with no GPS (0,0)
        # For embedded subtitles, we keep exposure-only data
        duration = points[-1].timestamp if points else 0

        return TelemetryResult(
            source="embedded_subtitle",
            sample_rate=1.0 / sample_interval,
            duration=duration,
            points=points,
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting embedded subtitles from {video_path}")
        return None
    except Exception as e:
        logger.warning(f"Error parsing embedded subtitle telemetry: {e}")
        return None
