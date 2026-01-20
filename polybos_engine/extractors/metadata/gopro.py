"""GoPro metadata extractor.

Handles GoPro cameras:
- HERO5, HERO6, HERO7, HERO8, HERO9, HERO10, HERO11, HERO12, HERO13
- MAX (360 camera)
- Session series

Detection methods:
- handler_name containing "GoPro"
- firmware tag pattern (e.g., "HD7.01.01.90.00" for HERO7)
- Filename patterns (GH*, GX*, GOPR*)

GoPro files contain:
- gpmd stream: GPS, accelerometer, gyroscope data
- Timecode
- Color space (usually BT.709)
"""

import logging
import struct
import subprocess
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    GPS,
    DetectionMethod,
    DeviceInfo,
    GPSTrack,
    GPSTrackPoint,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)

# GoPro model mapping from firmware prefix
GOPRO_MODELS = {
    "HD5": "HERO5 Black",
    "HD6": "HERO6 Black",
    "HD7": "HERO7 Black",
    "HD8": "HERO8 Black",
    "HD9": "HERO9 Black",
    "H10": "HERO10 Black",
    "H11": "HERO11 Black",
    "H12": "HERO12 Black",
    "H13": "HERO13 Black",
    "H21": "HERO Session",
    "H22": "HERO5 Session",
    "HX": "MAX",
    "H19": "MAX",  # Another MAX identifier
    "FS": "Fusion",
}


def _parse_firmware_model(firmware: str) -> str | None:
    """Parse GoPro model from firmware string.

    Examples:
    - "HD7.01.01.90.00" -> HERO7 Black
    - "H10.01.01.40.00" -> HERO10 Black
    """
    if not firmware:
        return None

    # Try direct prefix match
    for prefix, model in GOPRO_MODELS.items():
        if firmware.upper().startswith(prefix):
            return model

    return None


def _extract_gpmd_gps(file_path: str) -> tuple[GPS | None, GPSTrack | None]:
    """Extract GPS from GoPro GPMD stream.

    GoPro stores telemetry in a binary stream with FourCC tags.
    GPS data is under DEVC -> STRM -> GPS5 (lat, lon, alt, speed2d, speed3d).

    Returns tuple of (first GPS point, full GPS track).
    """
    try:
        # Extract gpmd stream using ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-codec",
            "copy",
            "-map",
            "0:d:0",  # First data stream (gpmd)
            "-f",
            "rawvideo",
            "pipe:1",
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=30, check=False)

        if result.returncode != 0 or not result.stdout:
            return None, None

        data = result.stdout
        gps_points: list[GPSTrackPoint] = []

        # Parse GPMD binary format
        # Looking for GPS5 tag which contains: lat, lon, alt, speed2d, speed3d
        # Each value is a signed 32-bit int, scaled by SCAL value

        i = 0
        current_scale = 1.0

        while i < len(data) - 8:
            # Read FourCC tag
            tag = data[i : i + 4]
            if len(tag) < 4:
                break

            # Check for SCAL (scale factor)
            if tag == b"SCAL":
                type_byte = data[i + 4] if i + 4 < len(data) else 0
                size = data[i + 5] if i + 5 < len(data) else 0
                count = (data[i + 6] << 8 | data[i + 7]) if i + 7 < len(data) else 0

                if type_byte == ord("l") and size == 4 and count >= 1:
                    # 32-bit signed int scale
                    scale_offset = i + 8
                    if scale_offset + 4 <= len(data):
                        current_scale = struct.unpack(
                            ">i", data[scale_offset : scale_offset + 4]
                        )[0]

            # Check for GPS5 (GPS data)
            elif tag == b"GPS5":
                type_byte = data[i + 4] if i + 4 < len(data) else 0
                size = data[i + 5] if i + 5 < len(data) else 0
                count = (data[i + 6] << 8 | data[i + 7]) if i + 7 < len(data) else 0

                if type_byte == ord("l") and size == 20:  # 5 x 4-byte ints
                    gps_offset = i + 8
                    for j in range(count):
                        sample_offset = gps_offset + j * 20
                        if sample_offset + 20 <= len(data):
                            values = struct.unpack(
                                ">iiiii", data[sample_offset : sample_offset + 20]
                            )
                            lat = values[0] / current_scale
                            lon = values[1] / current_scale
                            alt = values[2] / current_scale

                            # Validate coordinates
                            if -90 <= lat <= 90 and -180 <= lon <= 180 and lat != 0:
                                point = GPSTrackPoint(
                                    latitude=round(lat, 6),
                                    longitude=round(lon, 6),
                                    altitude=round(alt, 1) if alt != 0 else None,
                                )
                                # Dedupe consecutive identical points
                                if not gps_points or (
                                    point.latitude != gps_points[-1].latitude
                                    or point.longitude != gps_points[-1].longitude
                                ):
                                    gps_points.append(point)

            i += 1

        if gps_points:
            first_gps = GPS(
                latitude=gps_points[0].latitude,
                longitude=gps_points[0].longitude,
                altitude=gps_points[0].altitude,
            )
            track = (
                GPSTrack(points=gps_points, source="gpmd")
                if len(gps_points) > 1
                else None
            )
            logger.info(f"Extracted {len(gps_points)} GPS points from GoPro GPMD")
            return first_gps, track

        return None, None

    except Exception as e:
        logger.debug(f"Failed to extract GPS from GPMD: {e}")
        return None, None


class GoProExtractor:
    """Metadata extractor for GoPro cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a GoPro camera."""
        path = Path(file_path)
        tags = get_tags_lower(probe_data)

        # Check firmware tag
        firmware = tags.get("firmware", "")
        if firmware and any(
            firmware.upper().startswith(prefix) for prefix in GOPRO_MODELS
        ):
            return True

        # Check handler_name for "GoPro"
        for stream in probe_data.get("streams", []):
            handler = stream.get("tags", {}).get("handler_name", "")
            if "GoPro" in handler:
                return True

        # Check encoder tag
        for stream in probe_data.get("streams", []):
            encoder = stream.get("tags", {}).get("encoder", "")
            if "GoPro" in encoder:
                return True

        # Check filename pattern (GH*, GX*, GOPR*)
        name = path.stem.upper()
        if name.startswith(("GH", "GX", "GOPR")):
            return True

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract GoPro-specific metadata."""
        tags = get_tags_lower(probe_data)

        # Get firmware and parse model
        firmware = tags.get("firmware", "")
        model = _parse_firmware_model(firmware)

        # Determine device type (MAX is 360 camera)
        device_type = MediaDeviceType.ACTION_CAMERA
        if model and "MAX" in model:
            device_type = MediaDeviceType.CAMERA_360

        device = DeviceInfo(
            make="GoPro",
            model=model,
            software=firmware if firmware else None,
            type=device_type,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Extract GPS from GPMD stream
        gps, gps_track = _extract_gpmd_gps(file_path)

        # Use extracted GPS or keep base
        if gps is None:
            gps = base_metadata.gps
        if gps_track is None:
            gps_track = base_metadata.gps_track

        # Update metadata
        base_metadata.device = device
        base_metadata.gps = gps
        base_metadata.gps_track = gps_track

        return base_metadata


# Register the extractor
register_extractor("gopro", GoProExtractor())
