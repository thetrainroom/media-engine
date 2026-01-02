"""Metadata extraction using ffprobe and sidecar files."""

import json
import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    GPS,
    Codec,
    ColorSpace,
    DetectionMethod,
    DeviceInfo,
    LensInfo,
    MediaDeviceType,
    Metadata,
    Resolution,
)


@dataclass
class GPSCoordinates:
    """Parsed GPS coordinates from ISO 6709 format."""

    latitude: float
    longitude: float
    altitude: float | None = None


@dataclass
class SidecarMetadata:
    """Metadata extracted from sidecar files."""

    device: DeviceInfo | None = None
    gps: GPS | None = None
    color_space: ColorSpace | None = None
    lens: LensInfo | None = None

logger = logging.getLogger(__name__)

# Known drone manufacturers for device type detection
DRONE_MANUFACTURERS = {"DJI", "Parrot", "Autel", "Skydio", "Yuneec", "GoPro Karma"}


def extract_metadata(file_path: str) -> Metadata:
    """Extract metadata from video file using ffprobe.

    Args:
        file_path: Path to video file

    Returns:
        Metadata object with video information
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Run ffprobe
    probe_data = _run_ffprobe(file_path)

    # Parse format info
    format_info = probe_data.get("format", {})
    tags = format_info.get("tags", {})

    # Find video and audio streams
    video_stream = None
    audio_stream = None
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    # Extract resolution
    resolution = Resolution(
        width=video_stream.get("width", 0) if video_stream else 0,
        height=video_stream.get("height", 0) if video_stream else 0,
    )

    # Extract codec info
    codec = Codec(
        video=video_stream.get("codec_name") if video_stream else None,
        audio=audio_stream.get("codec_name") if audio_stream else None,
    )

    # Extract fps
    fps = _parse_fps(video_stream) if video_stream else None

    # Extract duration
    duration = float(format_info.get("duration", 0))

    # Extract bitrate
    bitrate = int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None

    # Extract file size
    file_size = os.path.getsize(file_path)

    # Extract creation time
    created_at = _parse_creation_time(tags)

    # Extract device info
    device = _extract_device_info(tags)

    # Extract GPS
    gps = _extract_gps(tags)

    # Extract color space from ffprobe
    color_space = _extract_color_space(video_stream)

    # Lens info (only available from sidecar)
    lens: LensInfo | None = None

    # Try Sony XML sidecar for additional metadata (preferred over generic detection)
    sidecar_info = _parse_sony_xml_sidecar(file_path)
    if sidecar_info:
        # Prefer XML sidecar device info (more accurate model name)
        if sidecar_info.device:
            device = sidecar_info.device
        if sidecar_info.gps:
            gps = sidecar_info.gps
        # Prefer XML sidecar color space (has LOG profile info)
        if sidecar_info.color_space:
            color_space = sidecar_info.color_space
        if sidecar_info.lens:
            lens = sidecar_info.lens

    return Metadata(
        duration=duration,
        resolution=resolution,
        codec=codec,
        fps=fps,
        bitrate=bitrate,
        file_size=file_size,
        created_at=created_at,
        device=device,
        gps=gps,
        color_space=color_space,
        lens=lens,
    )


def _run_ffprobe(file_path: str) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr}")
        raise RuntimeError(f"ffprobe failed for {file_path}: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        raise RuntimeError(f"Failed to parse ffprobe output: {e}")


def _parse_fps(video_stream: dict[str, Any]) -> float | None:
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


def _parse_creation_time(tags: dict[str, str]) -> datetime | None:
    """Parse creation time from metadata tags."""
    # Try various tag names
    time_str = (
        tags.get("creation_time")
        or tags.get("date")
        or tags.get("com.apple.quicktime.creationdate")
    )

    if not time_str:
        return None

    # Common formats with their expected string lengths
    formats = [
        ("%Y-%m-%dT%H:%M:%S.%fZ", 27),  # 2025-10-15T09:38:48.000000Z
        ("%Y-%m-%dT%H:%M:%SZ", 20),      # 2025-10-15T09:38:48Z
        ("%Y-%m-%dT%H:%M:%S", 19),       # 2025-10-15T09:38:48
        ("%Y-%m-%d %H:%M:%S", 19),       # 2025-10-15 09:38:48
        ("%Y:%m:%d %H:%M:%S", 19),       # 2025:10:15 09:38:48
    ]

    for fmt, length in formats:
        try:
            return datetime.strptime(time_str[:length], fmt)
        except ValueError:
            continue

    logger.warning(f"Could not parse creation time: {time_str}")
    return None


def _extract_device_info(tags: dict[str, str]) -> DeviceInfo | None:
    """Extract device information from metadata tags."""
    # Case-insensitive tag lookup
    tags_lower = {k.lower(): v for k, v in tags.items()}

    make = (
        tags_lower.get("make")
        or tags_lower.get("com.apple.quicktime.make")
        or tags_lower.get("manufacturer")
    )
    model = (
        tags_lower.get("model")
        or tags_lower.get("com.apple.quicktime.model")
        or tags_lower.get("model_name")
    )
    software = tags_lower.get("software") or tags_lower.get("com.apple.quicktime.software")

    # Check encoder tag for DJI drones (they use "DJIMavic3" format)
    encoder = tags_lower.get("encoder", "")
    if not make and not model and encoder:
        if encoder.upper().startswith("DJI"):
            make = "DJI"
            # Extract model from encoder string (e.g., "DJIMavic3" -> "Mavic3")
            model = encoder[3:] if len(encoder) > 3 else encoder

    # Check major_brand for Sony XAVC cameras
    major_brand = tags_lower.get("major_brand", "")
    if not make and not model and major_brand.upper() == "XAVC":
        make = "Sony"
        model = "XAVC Camera"  # Generic, actual model not in metadata

    if not make and not model:
        return None

    # Determine device type
    device_type = MediaDeviceType.UNKNOWN
    if make:
        if make.upper() in {m.upper() for m in DRONE_MANUFACTURERS}:
            device_type = MediaDeviceType.DRONE
        elif "IPHONE" in (model or "").upper() or "IPAD" in (model or "").upper():
            device_type = MediaDeviceType.PHONE
        elif "GOPRO" in make.upper():
            device_type = MediaDeviceType.ACTION_CAMERA
        else:
            device_type = MediaDeviceType.CAMERA

    return DeviceInfo(
        make=make,
        model=model,
        software=software,
        type=device_type,
        detection_method=DetectionMethod.METADATA,
        confidence=1.0,
    )


def _extract_gps(tags: dict[str, str]) -> GPS | None:
    """Extract GPS coordinates from metadata tags."""
    # Case-insensitive tag lookup
    tags_lower = {k.lower(): v for k, v in tags.items()}

    # Try various GPS tag formats
    location = (
        tags_lower.get("location")
        or tags_lower.get("com.apple.quicktime.location.iso6709")
        or tags_lower.get("gps")
    )

    if location:
        coords = _parse_iso6709(location)
        if coords:
            return GPS(
                latitude=coords.latitude,
                longitude=coords.longitude,
                altitude=coords.altitude,
            )

    # Try separate lat/lon tags
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


def _parse_iso6709(location: str) -> GPSCoordinates | None:
    """Parse ISO 6709 format GPS coordinates.

    Examples:
        +59.7441+010.2045/
        +59.7441+010.2045+125.0/
    """
    # Pattern for ISO 6709: +/-DD.DDDD+/-DDD.DDDD+/-AAAA/
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


def _extract_color_space(video_stream: dict[str, Any] | None) -> ColorSpace | None:
    """Extract color space information from video stream."""
    if not video_stream:
        return None

    transfer = video_stream.get("color_transfer")
    primaries = video_stream.get("color_primaries")
    matrix = video_stream.get("color_space")

    # Only return if we have some color info
    if not (transfer or primaries or matrix):
        return None

    return ColorSpace(
        transfer=transfer,
        primaries=primaries,
        matrix=matrix,
        detection_method=DetectionMethod.METADATA,
    )


def _parse_sony_xml_sidecar(video_path: str) -> SidecarMetadata | None:
    """Parse Sony XML sidecar file for additional metadata.

    Sony cameras create XML sidecar files with naming pattern:
    - Video: 20251014_C0476.MP4
    - XML:   20251014_C0476M01.XML
    """
    path = Path(video_path)

    # Try common Sony XML sidecar naming patterns
    xml_patterns = [
        path.with_suffix(".XML"),  # Same name
        path.parent / f"{path.stem}M01.XML",  # Sony pattern: filename + M01.XML
        path.parent / f"{path.stem}M01.xml",
    ]

    xml_path = None
    for pattern in xml_patterns:
        if pattern.exists():
            xml_path = pattern
            break

    if not xml_path:
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle XML namespace
        ns = {"nrt": "urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20"}

        device: DeviceInfo | None = None
        gps: GPS | None = None
        color_space: ColorSpace | None = None
        lens: LensInfo | None = None

        # Extract device info
        device_elem = root.find(".//nrt:Device", ns) or root.find(".//{*}Device")
        if device_elem is not None:
            manufacturer = device_elem.get("manufacturer")
            model_name = device_elem.get("modelName")

            if manufacturer or model_name:
                device_type = MediaDeviceType.CAMERA
                if manufacturer and manufacturer.upper() == "DJI":
                    device_type = MediaDeviceType.DRONE

                device = DeviceInfo(
                    make=manufacturer,
                    model=model_name,
                    software=None,
                    type=device_type,
                    detection_method=DetectionMethod.XML_SIDECAR,
                    confidence=1.0,
                )

        # Extract GPS from ExifGPS group
        gps_group = root.find(".//{*}Group[@name='ExifGPS']")
        if gps_group is not None:
            gps_items: dict[str, str | None] = {}
            for item in gps_group.findall(".//{*}Item"):
                name = item.get("name")
                if name is not None:
                    gps_items[name] = item.get("value")

            # Check if GPS has valid fix (Status != "V" means no fix)
            if gps_items.get("Status") != "V":
                lat = gps_items.get("Latitude")
                lon = gps_items.get("Longitude")
                alt = gps_items.get("Altitude")

                if lat and lon:
                    try:
                        gps = GPS(
                            latitude=float(lat),
                            longitude=float(lon),
                            altitude=float(alt) if alt else None,
                        )
                    except ValueError:
                        pass

        # Extract color space from VideoLayout or CameraUnitMetadata groups
        # Look for CaptureGammaEquation (s-log3), CaptureColorPrimaries (s-gamut3)
        color_items: dict[str, str | None] = {}
        for group_name in ["CameraUnitMetadata", "VideoLayout", "AcquisitionRecord"]:
            group = root.find(f".//*[@name='{group_name}']")
            if group is not None:
                for item in group.findall(".//{*}Item"):
                    name = item.get("name")
                    if name:
                        color_items[name] = item.get("value")

        # Also check top-level items
        for item in root.findall(".//{*}Item"):
            name = item.get("name")
            if name and name in [
                "CaptureGammaEquation",
                "CaptureColorPrimaries",
                "CodingEquations",
            ]:
                color_items[name] = item.get("value")

        # Look for LUT file reference
        lut_file: str | None = None
        for related in root.findall(".//{*}RelatedTo"):
            if related.get("rel") == "LUT":
                lut_file = related.get("file")
                break

        gamma = color_items.get("CaptureGammaEquation")
        primaries = color_items.get("CaptureColorPrimaries")
        coding = color_items.get("CodingEquations")

        if gamma or primaries or coding or lut_file:
            color_space = ColorSpace(
                transfer=gamma,
                primaries=primaries,
                matrix=coding,
                lut_file=lut_file,
                detection_method=DetectionMethod.XML_SIDECAR,
            )

        # Extract lens info from Camera or Lens groups
        lens_items: dict[str, str | None] = {}
        for group_name in ["Camera", "Lens", "CameraUnitMetadata"]:
            group = root.find(f".//*[@name='{group_name}']")
            if group is not None:
                for item in group.findall(".//{*}Item"):
                    name = item.get("name")
                    if name:
                        lens_items[name] = item.get("value")

        # Also check top-level items for lens data
        for item in root.findall(".//{*}Item"):
            name = item.get("name")
            if name and name in [
                "FocalLength",
                "FocalLength35mm",
                "FocalLengthIn35mmFilm",
                "FNumber",
                "Iris",
                "FocusDistance",
            ]:
                lens_items[name] = item.get("value")

        focal_length = lens_items.get("FocalLength")
        focal_35mm = lens_items.get("FocalLength35mm") or lens_items.get(
            "FocalLengthIn35mmFilm"
        )
        f_number = lens_items.get("FNumber")
        iris = lens_items.get("Iris")
        focus_dist = lens_items.get("FocusDistance")

        if focal_length or focal_35mm or f_number or iris:
            lens = LensInfo(
                focal_length=float(focal_length) if focal_length else None,
                focal_length_35mm=float(focal_35mm) if focal_35mm else None,
                aperture=float(f_number) if f_number else None,
                focus_distance=float(focus_dist) if focus_dist else None,
                iris=iris,
                detection_method=DetectionMethod.XML_SIDECAR,
            )

        if device or gps or color_space or lens:
            return SidecarMetadata(
                device=device, gps=gps, color_space=color_space, lens=lens
            )
        return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse Sony XML sidecar {xml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading Sony XML sidecar {xml_path}: {e}")
        return None
