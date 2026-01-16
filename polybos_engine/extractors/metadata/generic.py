"""Generic metadata extraction fallback.

This module handles files that don't match any specific manufacturer.
It extracts basic device info from standard metadata tags.

Detection:
- Always matches as fallback (registered last in __init__.py)
"""

import logging
from typing import Any

from polybos_engine.schemas import (
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower

logger = logging.getLogger(__name__)

# Known drone manufacturers for device type detection
DRONE_MANUFACTURERS = {"DJI", "Parrot", "Autel", "Skydio", "Yuneec", "GoPro Karma"}


def _determine_device_type(make: str | None, model: str | None) -> MediaDeviceType:
    """Determine device type from make and model strings."""
    if make:
        make_upper = make.upper()

        # Check for drones
        if make_upper in {m.upper() for m in DRONE_MANUFACTURERS}:
            return MediaDeviceType.DRONE

        # Check for action cameras
        if "GOPRO" in make_upper:
            return MediaDeviceType.ACTION_CAMERA

    if model:
        model_upper = model.upper()

        # Check for phones
        if "IPHONE" in model_upper or "IPAD" in model_upper:
            return MediaDeviceType.PHONE
        if "PIXEL" in model_upper or "GALAXY" in model_upper:
            return MediaDeviceType.PHONE

        # Check for action cameras
        if "GOPRO" in model_upper or "HERO" in model_upper:
            return MediaDeviceType.ACTION_CAMERA
        if "OSMO" in model_upper or "ACTION" in model_upper:
            return MediaDeviceType.ACTION_CAMERA

    # Default to camera for professional/unknown devices
    return MediaDeviceType.CAMERA if make or model else MediaDeviceType.UNKNOWN


class GenericExtractor:
    """Fallback metadata extractor for unknown devices."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Always match as fallback."""
        return True

    def extract(
        self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata
    ) -> Metadata:
        """Extract basic device info from metadata tags."""
        tags = get_tags_lower(probe_data)

        # Try various tag locations for make/model
        make = (
            tags.get("make")
            or tags.get("manufacturer")
            or tags.get("com.apple.quicktime.make")
            or tags.get("com.apple.proapps.manufacturer")
        )
        model = (
            tags.get("model")
            or tags.get("model_name")
            or tags.get("com.apple.quicktime.model")
            or tags.get("com.apple.proapps.cameraname")
        )
        software = tags.get("software") or tags.get("com.apple.quicktime.software")

        # Check encoder tag for additional info
        encoder = tags.get("encoder", "")
        if not make and not model and encoder:
            # Some cameras put info in encoder tag
            if encoder.upper().startswith("DJI"):
                make = "DJI"
                model = encoder[3:] if len(encoder) > 3 else encoder

        # If we still have no info, return base metadata unchanged
        if not make and not model:
            return base_metadata

        # Determine device type
        device_type = _determine_device_type(make, model)

        device = DeviceInfo(
            make=make,
            model=model,
            software=software,
            type=device_type,
            detection_method=DetectionMethod.METADATA,
            confidence=0.8,  # Lower confidence for generic detection
        )

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
            gps=base_metadata.gps,
            color_space=base_metadata.color_space,
            lens=base_metadata.lens,
        )


# Register this extractor LAST (it's the fallback)
# This is done in __init__.py to ensure proper order
