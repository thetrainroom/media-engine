"""Apple metadata extraction.

Handles Apple devices:
- iPhone (all models)
- iPad (all models)
- Mac (FaceTime camera, etc.)

Detection methods:
- make tag: "Apple"
- com.apple.quicktime.make tag
- Model contains "iPhone" or "iPad"

Apple QuickTime metadata tags:
- com.apple.quicktime.make
- com.apple.quicktime.model
- com.apple.quicktime.software
- com.apple.quicktime.creationdate
- com.apple.quicktime.location.iso6709
"""

import logging
import re
from typing import Any

from polybos_engine.schemas import (
    GPS,
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


def _parse_apple_location(location: str) -> GPS | None:
    """Parse Apple's ISO 6709 location string.

    Format: +59.7441+010.2045+125.0/
    """
    pattern = r"([+-]\d+\.?\d*)"
    matches = re.findall(pattern, location)

    if len(matches) >= 2:
        try:
            return GPS(
                latitude=float(matches[0]),
                longitude=float(matches[1]),
                altitude=float(matches[2]) if len(matches) >= 3 else None,
            )
        except ValueError:
            pass

    return None


def _determine_device_type(model: str | None) -> MediaDeviceType:
    """Determine Apple device type from model string."""
    if not model:
        return MediaDeviceType.PHONE

    model_upper = model.upper()

    if "IPHONE" in model_upper:
        return MediaDeviceType.PHONE
    elif "IPAD" in model_upper:
        return MediaDeviceType.PHONE  # Tablets are close to phones
    elif "MAC" in model_upper or "IMAC" in model_upper or "MACBOOK" in model_upper:
        return MediaDeviceType.CAMERA  # Mac webcams are cameras
    else:
        return MediaDeviceType.PHONE


def _clean_model_name(model: str | None) -> str | None:
    """Clean up Apple model name for display.

    Examples:
        "iPhone 15 Pro Max" -> "iPhone 15 Pro Max"
        "iPhone15,3" -> "iPhone 15 Pro Max" (if we had a lookup table)
    """
    if not model:
        return None

    # Apple sometimes uses internal model identifiers like "iPhone15,3"
    # For now, just return as-is. A future enhancement could add a lookup table.
    return model


class AppleExtractor:
    """Metadata extractor for Apple devices."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from an Apple device."""
        tags = get_tags_lower(probe_data)

        # Check make tag
        make = tags.get("make") or tags.get("com.apple.quicktime.make")
        if make and "APPLE" in make.upper():
            return True

        # Check model for iPhone/iPad
        model = tags.get("model") or tags.get("com.apple.quicktime.model")
        if model:
            model_upper = model.upper()
            if "IPHONE" in model_upper or "IPAD" in model_upper:
                return True

        # Check for Apple QuickTime-specific tags
        if tags.get("com.apple.quicktime.creationdate"):
            # This is a strong indicator of Apple origin
            if tags.get("com.apple.quicktime.make") or tags.get("com.apple.quicktime.model"):
                return True

        return False

    def extract(
        self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata
    ) -> Metadata:
        """Extract Apple-specific metadata."""
        tags = get_tags_lower(probe_data)

        # Get device info from QuickTime tags (preferred) or standard tags
        make = (
            tags.get("com.apple.quicktime.make")
            or tags.get("make")
            or "Apple"
        )
        model = (
            tags.get("com.apple.quicktime.model")
            or tags.get("model")
        )
        software = (
            tags.get("com.apple.quicktime.software")
            or tags.get("software")
        )

        # Clean up model name
        model = _clean_model_name(model)

        # Determine device type
        device_type = _determine_device_type(model)

        device = DeviceInfo(
            make=make,
            model=model,
            software=software,
            type=device_type,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Extract GPS from Apple-specific location tag
        gps = base_metadata.gps
        apple_location = tags.get("com.apple.quicktime.location.iso6709")
        if apple_location:
            parsed_gps = _parse_apple_location(apple_location)
            if parsed_gps:
                gps = parsed_gps

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
            color_space=base_metadata.color_space,
            lens=base_metadata.lens,
        )


# Register this extractor
register_extractor("apple", AppleExtractor())
