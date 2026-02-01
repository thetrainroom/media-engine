"""Tesla dashcam metadata extractor.

Tesla vehicles record dashcam footage from 4 cameras:
- front: Main forward-facing camera
- back: Rear camera
- left_repeater: Left side mirror camera
- right_repeater: Right side mirror camera

Files are saved in 1-minute segments with naming pattern:
YYYY-MM-DD_HH-MM-SS-camera.mp4

Sentry mode and dashcam events include:
- event.json: Contains timestamp, GPS (est_lat, est_lon), city, reason
- thumb.png: Thumbnail preview

Detection methods:
- Filename pattern matching
- event.json sidecar presence
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from media_engine.schemas import (
    GPS,
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import register_extractor

logger = logging.getLogger(__name__)

# Tesla filename pattern: YYYY-MM-DD_HH-MM-SS-camera.mp4
TESLA_FILENAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-(front|back|left_repeater|right_repeater)\.mp4$")


def _parse_event_json(video_path: str) -> GPS | None:
    """Parse Tesla event.json sidecar for GPS coordinates.

    The event.json file is in the parent folder of the video files
    and contains estimated GPS coordinates.
    """
    path = Path(video_path)

    # event.json is in the same directory as the video
    event_json = path.parent / "event.json"

    if not event_json.exists():
        return None

    try:
        with open(event_json, encoding="utf-8") as f:
            data = json.load(f)

        lat_str = data.get("est_lat")
        lon_str = data.get("est_lon")

        if lat_str and lon_str:
            lat = float(lat_str)
            lon = float(lon_str)

            if lat != 0 and lon != 0:
                logger.info(f"Extracted GPS from Tesla event.json: {lat}, {lon}")
                return GPS(latitude=lat, longitude=lon)

        return None

    except Exception as e:
        logger.warning(f"Error reading Tesla event.json: {e}")
        return None


def _detect_camera_position(filename: str) -> str | None:
    """Detect which camera the file is from based on filename."""
    name_lower = filename.lower()

    if "-front" in name_lower:
        return "front"
    elif "-back" in name_lower:
        return "rear"
    elif "-left_repeater" in name_lower:
        return "left"
    elif "-right_repeater" in name_lower:
        return "right"

    return None


class TeslaExtractor:
    """Metadata extractor for Tesla dashcam footage."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a Tesla dashcam."""
        path = Path(file_path)

        # Check filename pattern
        if TESLA_FILENAME_PATTERN.match(path.name):
            return True

        # Check for event.json in same directory (Tesla sentry/dashcam event)
        event_json = path.parent / "event.json"
        if event_json.exists():
            try:
                with open(event_json, encoding="utf-8") as f:
                    data = json.load(f)
                # Tesla event.json has specific keys
                if "est_lat" in data or "reason" in data:
                    return True
            except Exception:
                pass

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract Tesla dashcam metadata."""
        path = Path(file_path)

        # Detect camera position
        camera = _detect_camera_position(path.name)

        # Build model string with camera position
        model = "Dashcam"
        if camera:
            model = f"Dashcam ({camera})"

        device = DeviceInfo(
            make="Tesla",
            model=model,
            type=MediaDeviceType.DASHCAM,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Extract GPS from event.json
        gps = _parse_event_json(file_path)
        if gps is None:
            gps = base_metadata.gps

        base_metadata.device = device
        base_metadata.gps = gps

        return base_metadata


# Register the extractor
register_extractor("tesla", TeslaExtractor())
