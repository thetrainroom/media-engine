"""Blackmagic Design metadata extraction.

Handles Blackmagic cameras:
- Pocket Cinema Camera (4K, 6K, 6K Pro, 6K G2)
- URSA Mini Pro (4.6K, 12K)
- Micro Cinema Camera
- Production Camera 4K

Detection methods:
- .braw extension (Blackmagic RAW)
- com.apple.proapps.manufacturer: "Blackmagic Design"
- com.apple.proapps.cameraname: camera model
- com.apple.proapps.customgamma: LOG profile

Note: Full BRAW metadata requires Blackmagic RAW SDK (free download).
Without it, we detect the format but limited metadata from ffprobe.
"""

import logging
from pathlib import Path
from typing import Any

from media_engine.schemas import (
    ColorSpace,
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


def _parse_custom_gamma(gamma_string: str) -> str | None:
    """Parse Blackmagic custom gamma string to get LOG profile name.

    Examples:
        "com.blackmagic-design.productioncamera4k.filmlog" -> "filmlog"
        "com.blackmagic-design.ursa.film" -> "film"
    """
    if not gamma_string:
        return None

    parts = gamma_string.split(".")
    if parts:
        return parts[-1]  # Return the last part as the profile name

    return None


class BlackmagicExtractor:
    """Metadata extractor for Blackmagic Design cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a Blackmagic camera."""
        path = Path(file_path)

        # Check for BRAW extension
        if path.suffix.lower() == ".braw":
            return True

        tags = get_tags_lower(probe_data)

        # Check ProApps manufacturer tag
        manufacturer = tags.get("com.apple.proapps.manufacturer", "")
        if "BLACKMAGIC" in manufacturer.upper():
            return True

        # Check make tag
        make = tags.get("make") or tags.get("manufacturer")
        if make and "BLACKMAGIC" in make.upper():
            return True

        # Check custom gamma for Blackmagic signature
        custom_gamma = tags.get("com.apple.proapps.customgamma", "")
        if "blackmagic" in custom_gamma.lower():
            return True

        return False

    def extract(self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata) -> Metadata:
        """Extract Blackmagic-specific metadata."""
        path = Path(file_path)
        tags = get_tags_lower(probe_data)

        # Get device info from ProApps tags (preferred)
        manufacturer = tags.get("com.apple.proapps.manufacturer") or tags.get("make") or "Blackmagic Design"
        camera_name = tags.get("com.apple.proapps.cameraname") or tags.get("model")

        # BRAW files are from cinema cameras
        is_braw = path.suffix.lower() == ".braw"
        if is_braw:
            logger.info("BRAW detected. For full metadata, install Blackmagic RAW SDK.")

        device = DeviceInfo(
            make=manufacturer,
            model=camera_name,
            software=tags.get("software"),
            type=MediaDeviceType.CINEMA_CAMERA,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Extract color space from custom gamma
        color_space = base_metadata.color_space
        custom_gamma = tags.get("com.apple.proapps.customgamma", "")
        if custom_gamma:
            profile_name = _parse_custom_gamma(custom_gamma)
            if profile_name:
                base_cs = base_metadata.color_space
                color_space = ColorSpace(
                    transfer=profile_name,
                    primaries=base_cs.primaries if base_cs else None,
                    matrix=base_cs.matrix if base_cs else None,
                    detection_method=DetectionMethod.METADATA,
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
            color_space=color_space,
            lens=base_metadata.lens,
        )


# Register this extractor
register_extractor("blackmagic", BlackmagicExtractor())
