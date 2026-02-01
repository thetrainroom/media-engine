"""DJI metadata extraction.

Handles DJI drones and cameras:
- Mavic series (Air, Pro, Mini, etc.)
- Phantom series
- Inspire series
- Osmo/Pocket series (Pocket, Pocket 2, Action, etc.)
- FPV drones

Detection methods:
- encoder tag: "DJIMavic3", "DJI Pocket2", etc.
- make tag: "DJI"
- filename prefix: "DJI_"
- SRT sidecar file presence

SRT files contain per-frame telemetry:
[iso: 400] [shutter: 1/100.0] [fnum: 2.8] [latitude: 61.05121] ...
"""

import logging
import re
from pathlib import Path
from typing import Any

from media_engine.schemas import (
    GPS,
    ColorSpace,
    DetectionMethod,
    DeviceInfo,
    GPSTrack,
    GPSTrackPoint,
    LensInfo,
    MediaDeviceType,
    Metadata,
)

from .base import SidecarMetadata
from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)

# DJI device type mapping based on model name
DRONE_MODELS = {
    "mavic",
    "phantom",
    "inspire",
    "mini",
    "air",
    "fpv",
    "avata",
    "matrice",
    "agras",
}

CAMERA_MODELS = {
    "pocket",
    "osmo",
    "action",
    "ronin",
}


def _get_device_type(model: str | None, has_gps: bool = False) -> MediaDeviceType:
    """Determine device type from model name.

    Args:
        model: Model name string
        has_gps: Whether GPS data was found (indicates drone)
    """
    if not model:
        # No model info - use GPS as hint (drones have GPS, Pocket doesn't)
        return MediaDeviceType.DRONE if has_gps else MediaDeviceType.ACTION_CAMERA

    model_lower = model.lower()

    # Check for handheld cameras/gimbals FIRST
    for camera_model in CAMERA_MODELS:
        if camera_model in model_lower:
            return MediaDeviceType.ACTION_CAMERA

    # Check for drones
    for drone_model in DRONE_MODELS:
        if drone_model in model_lower:
            return MediaDeviceType.DRONE

    # No match - use GPS as hint
    return MediaDeviceType.DRONE if has_gps else MediaDeviceType.ACTION_CAMERA


def _parse_encoder_model(encoder: str) -> str | None:
    """Parse model name from encoder string.

    Examples:
        "DJIMavic3" -> "Mavic 3"
        "DJI Pocket2" -> "Pocket 2"
        "DJIMini3Pro" -> "Mini 3 Pro"
        "DJIFPV" -> "FPV"
    """
    if not encoder:
        return None

    # Remove DJI prefix (case insensitive)
    model = encoder
    if model.upper().startswith("DJI"):
        model = model[3:].strip()

    if not model:
        return None

    # Add spaces before numbers (Mavic3 -> Mavic 3)
    model = re.sub(r"(\D)(\d)", r"\1 \2", model)

    # Add spaces before uppercase letters (Mini3Pro -> Mini 3 Pro)
    model = re.sub(r"([a-z])([A-Z])", r"\1 \2", model)

    return model.strip()


def _parse_color_from_comment(comment: str) -> str | None:
    """Parse color mode from DJI Pocket/Osmo comment tag.

    The comment tag format is: "DE=D-CLike, Type=Normal, HQ=Normal, Mode=P"
    DE values: D-CLike (D-Cinelike), Normal, D-Log, etc.
    """
    if not comment:
        return None

    # Look for DE= pattern
    match = re.search(r"DE=([^,]+)", comment)
    if match:
        color_mode = match.group(1).strip()
        # Normalize common names
        if color_mode.lower() == "d-clike":
            return "D-Cinelike"
        elif color_mode.lower() == "d-log":
            return "D-Log"
        return color_mode

    return None


def _parse_srt_sidecar(video_path: str) -> SidecarMetadata | None:
    """Parse DJI SRT sidecar file for GPS and telemetry.

    DJI drones create SRT files with per-frame telemetry:
    - Video: DJI_0987.MP4
    - SRT:   DJI_0987.SRT

    Format: [iso: 400] [shutter: 1/100.0] [fnum: 2.8] [latitude: 61.05121] ...

    Returns SidecarMetadata with first GPS point and full GPS track.
    """
    path = Path(video_path)

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

        gps: GPS | None = None
        gps_track: GPSTrack | None = None
        color_space: ColorSpace | None = None
        lens: LensInfo | None = None

        # Extract ALL GPS coordinates for track
        lat_matches = re.findall(r"\[latitude:\s*([-\d.]+)\]", content)
        lon_matches = re.findall(r"\[longitude:\s*([-\d.]+)\]", content)
        abs_alt_matches = re.findall(r"abs_alt:\s*([-\d.]+)", content)

        if lat_matches and lon_matches and len(lat_matches) == len(lon_matches):
            gps_points: list[GPSTrackPoint] = []
            last_lat: float | None = None
            last_lon: float | None = None

            for i, (lat_str, lon_str) in enumerate(zip(lat_matches, lon_matches)):
                lat = float(lat_str)
                lon = float(lon_str)

                # Skip invalid 0,0 coordinates
                if lat == 0 and lon == 0:
                    continue

                # Get altitude if available
                alt: float | None = None
                if i < len(abs_alt_matches):
                    alt = float(abs_alt_matches[i])

                # Dedupe consecutive identical points
                if lat != last_lat or lon != last_lon:
                    gps_points.append(
                        GPSTrackPoint(
                            latitude=round(lat, 6),
                            longitude=round(lon, 6),
                            altitude=round(alt, 1) if alt is not None else None,
                        )
                    )
                    last_lat = lat
                    last_lon = lon

            # First valid point becomes the GPS location
            if gps_points:
                gps = GPS(
                    latitude=gps_points[0].latitude,
                    longitude=gps_points[0].longitude,
                    altitude=gps_points[0].altitude,
                )

                # Create track if we have multiple unique points
                if len(gps_points) > 1:
                    gps_track = GPSTrack(points=gps_points, source="srt_sidecar")
                    logger.info(f"Extracted {len(gps_points)} GPS points from SRT")

        # Color mode (d_log, d_cinelike, etc.)
        color_match = re.search(r"\[color_md\s*:\s*(\w+)\]", content)
        if color_match:
            color_mode = color_match.group(1)
            color_space = ColorSpace(
                transfer=color_mode,
                detection_method=DetectionMethod.METADATA,
            )

        # Focal length and aperture
        focal_match = re.search(r"\[focal_len:\s*([\d.]+)\]", content)
        fnum_match = re.search(r"\[fnum:\s*([\d.]+)\]", content)

        if focal_match or fnum_match:
            lens = LensInfo(
                focal_length=float(focal_match.group(1)) if focal_match else None,
                aperture=float(fnum_match.group(1)) if fnum_match else None,
                detection_method=DetectionMethod.METADATA,
            )

        if gps or gps_track or color_space or lens:
            return SidecarMetadata(gps=gps, gps_track=gps_track, color_space=color_space, lens=lens)
        return None

    except Exception as e:
        logger.warning(f"Error reading DJI SRT sidecar {srt_path}: {e}")
        return None


class DJIExtractor:
    """Metadata extractor for DJI devices."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a DJI device."""
        tags = get_tags_lower(probe_data)

        # Check make tag
        make = tags.get("make") or tags.get("manufacturer")
        if make and "DJI" in make.upper():
            return True

        # Check encoder tag (DJIMavic3, DJI Pocket2, etc.)
        encoder = tags.get("encoder", "")
        if encoder.upper().startswith("DJI"):
            return True

        # Check video stream handler_name (DJI Pocket uses "DJI.AVC")
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                stream_tags = stream.get("tags", {})
                handler = stream_tags.get("handler_name", "")
                if "DJI" in handler.upper():
                    return True

        # Check filename prefix
        filename = Path(file_path).name
        if filename.upper().startswith("DJI_"):
            return True

        # Check for SRT sidecar (DJI signature)
        path = Path(file_path)
        if path.with_suffix(".SRT").exists() or path.with_suffix(".srt").exists():
            # Read first line of SRT to confirm DJI format
            try:
                srt_upper = path.with_suffix(".SRT")
                srt_path = srt_upper if srt_upper.exists() else path.with_suffix(".srt")
                with open(srt_path, encoding="utf-8") as f:
                    content = f.read(500)
                    # DJI SRT has [iso:, [shutter:, etc.
                    if "[iso:" in content.lower() or "[shutter:" in content.lower():
                        return True
            except Exception:
                pass

        return False

    def extract(self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata) -> Metadata:
        """Extract DJI-specific metadata."""
        tags = get_tags_lower(probe_data)

        # Get make and model
        make = tags.get("make") or tags.get("manufacturer") or "DJI"
        model = tags.get("model") or tags.get("model_name")

        # Try to get model from encoder tag
        encoder = tags.get("encoder", "")
        if not model and encoder.upper().startswith("DJI"):
            model = _parse_encoder_model(encoder)

        # Parse SRT sidecar for additional metadata (drones have these)
        sidecar = _parse_srt_sidecar(file_path)

        # Get GPS and track - from sidecar (drone) or base metadata
        gps = sidecar.gps if sidecar and sidecar.gps else base_metadata.gps
        gps_track = sidecar.gps_track if sidecar and sidecar.gps_track else base_metadata.gps_track

        # Determine device type using model and GPS presence as hints
        has_gps = gps is not None
        device_type = _get_device_type(model, has_gps)

        device = DeviceInfo(
            make=make if make else "DJI",
            model=model,
            software=tags.get("software"),
            type=device_type,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Get color space - prefer SRT, then comment tag, then base
        color_space = base_metadata.color_space
        if sidecar and sidecar.color_space:
            color_space = sidecar.color_space
        else:
            # Try parsing from comment tag (DJI Pocket/Osmo)
            comment = tags.get("comment", "")
            color_mode = _parse_color_from_comment(comment)
            if color_mode:
                color_space = ColorSpace(
                    transfer=color_mode,
                    detection_method=DetectionMethod.METADATA,
                )

        lens = sidecar.lens if sidecar and sidecar.lens else base_metadata.lens

        return Metadata(
            duration=base_metadata.duration,
            resolution=base_metadata.resolution,
            codec=base_metadata.codec,
            video_codec=base_metadata.video_codec,
            audio=base_metadata.audio,
            fps=base_metadata.fps,
            bitrate=base_metadata.bitrate,
            file_size=base_metadata.file_size,
            timecode=base_metadata.timecode,
            created_at=base_metadata.created_at,
            device=device,
            gps=gps,
            gps_track=gps_track,
            color_space=color_space,
            lens=lens,
        )


# Register this extractor
register_extractor("dji", DJIExtractor())
