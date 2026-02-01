"""Generic 360 camera metadata extractor.

Detects 360 cameras from various manufacturers:
- Insta360 (X3, X4, ONE RS, GO 3, etc.)
- Kandao QooCam (QooCam 8K, QooCam 3, etc.)
- GoPro MAX
- Ricoh Theta
- Samsung Gear 360

Detection methods:
- File extension (.insv, .insp for Insta360)
- Filename patterns (Q360_* for QooCam)
- Dual video streams with square resolution (unstitched fisheye)
- 2:1 aspect ratio (stitched equirectangular)
- Spherical metadata tags
- Handler names and make/model tags
"""

import logging
import re
from pathlib import Path
from typing import Any

from media_engine.schemas import (
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


# Known 360 camera identifiers
CAMERA_360_BRANDS = {
    "insta360": {
        "make": "Insta360",
        "patterns": [r"INS", r"Insta360"],
        "extensions": [".insv", ".insp"],
    },
    "kandao": {
        "make": "Kandao",
        "patterns": [r"Q360_", r"QooCam", r"Kandao"],
        "extensions": [],
    },
    "gopro_max": {
        "make": "GoPro",
        "model": "MAX",
        "patterns": [r"GoPro MAX", r"GPMAX"],
        "extensions": [".360"],
    },
    "ricoh": {
        "make": "Ricoh",
        "patterns": [r"RICOH THETA", r"THETA"],
        "extensions": [],
    },
    "samsung": {
        "make": "Samsung",
        "patterns": [r"Gear 360", r"SM-R210", r"SM-C200"],
        "extensions": [],
    },
}


class Camera360Extractor:
    """Extract metadata from 360 cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this is a 360 camera file."""
        path = Path(file_path)
        tags = get_tags_lower(probe_data)

        # Check file extensions
        suffix_lower = path.suffix.lower()
        for brand_info in CAMERA_360_BRANDS.values():
            if suffix_lower in brand_info.get("extensions", []):
                return True

        # Check filename patterns
        filename = path.name
        for brand_info in CAMERA_360_BRANDS.values():
            for pattern in brand_info.get("patterns", []):
                if re.search(pattern, filename, re.IGNORECASE):
                    return True

        # Check make/model tags
        make = tags.get("make", "") or tags.get("manufacturer", "")
        model = tags.get("model", "")
        make_model = f"{make} {model}".strip()

        for brand_info in CAMERA_360_BRANDS.values():
            for pattern in brand_info.get("patterns", []):
                if re.search(pattern, make_model, re.IGNORECASE):
                    return True

        # Check handler_name for 360 camera identifiers
        for stream in probe_data.get("streams", []):
            handler = stream.get("tags", {}).get("handler_name", "")
            for brand_info in CAMERA_360_BRANDS.values():
                for pattern in brand_info.get("patterns", []):
                    if re.search(pattern, handler, re.IGNORECASE):
                        return True

        # Check for spherical video metadata
        if self._has_spherical_metadata(probe_data, tags):
            return True

        # Check for dual square video streams (unstitched 360)
        if self._has_dual_fisheye_streams(probe_data):
            return True

        return False

    def _has_spherical_metadata(self, probe_data: dict[str, Any], tags: dict[str, Any]) -> bool:
        """Check for spherical/360 video metadata tags."""
        # Check format tags
        spherical_keys = [
            "spherical",
            "spherical-video",
            "projection_type",
            "stereo_mode",
            "stitching_software",
        ]
        for key in spherical_keys:
            if key in tags:
                return True

        # Check stream side_data for spherical projection
        for stream in probe_data.get("streams", []):
            side_data = stream.get("side_data_list", [])
            for data in side_data:
                if data.get("side_data_type") == "Spherical Mapping":
                    return True
                if "spherical" in str(data).lower():
                    return True

        return False

    def _has_dual_fisheye_streams(self, probe_data: dict[str, Any]) -> bool:
        """Check for dual video streams with square resolution (unstitched 360)."""
        video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]

        if len(video_streams) < 2:
            return False

        # Check if both streams are square (fisheye)
        square_streams = 0
        for stream in video_streams:
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            if width > 0 and width == height:
                square_streams += 1

        return square_streams >= 2

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract 360 camera metadata."""
        tags = get_tags_lower(probe_data)

        # Detect brand and model
        make, model = self._detect_brand_model(probe_data, file_path, tags)

        # Detect if it's unstitched (dual fisheye) or stitched (equirectangular)
        is_unstitched = self._has_dual_fisheye_streams(probe_data)

        device = DeviceInfo(
            make=make,
            model=model,
            type=MediaDeviceType.CAMERA_360,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Add note about stitching status in software field
        if is_unstitched:
            device.software = "unstitched dual-fisheye"

        base_metadata.device = device
        return base_metadata

    def _detect_brand_model(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        tags: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Detect 360 camera brand and model."""
        path = Path(file_path)
        filename = path.name
        suffix_lower = path.suffix.lower()

        # Check file extension first
        for brand_key, brand_info in CAMERA_360_BRANDS.items():
            if suffix_lower in brand_info.get("extensions", []):
                model = self._detect_model_from_resolution(probe_data, brand_key)
                return brand_info["make"], model

        # Check filename patterns
        if re.search(r"Q360_", filename):
            model = self._detect_qoocam_model(probe_data)
            return "Kandao", model

        # Check make/model tags
        make = tags.get("make", "") or tags.get("manufacturer", "")
        model = tags.get("model", "")

        if make:
            # Normalize known brands
            make_upper = make.upper()
            if "INSTA" in make_upper:
                return "Insta360", model or self._detect_model_from_resolution(probe_data, "insta360")
            if "GOPRO" in make_upper:
                return "GoPro", model or "MAX"
            if "RICOH" in make_upper or "THETA" in make_upper:
                return "Ricoh", model
            if "SAMSUNG" in make_upper:
                return "Samsung", model
            if "KANDAO" in make_upper or "QOOCAM" in make_upper:
                return "Kandao", model or self._detect_qoocam_model(probe_data)

            return make, model if model else None

        # Check handler for INS prefix (Insta360)
        for stream in probe_data.get("streams", []):
            handler = stream.get("tags", {}).get("handler_name", "")
            if "INS" in handler.upper():
                return "Insta360", self._detect_model_from_resolution(probe_data, "insta360")

        # Fallback for detected 360 video
        return "Unknown 360 Camera", None

    def _detect_model_from_resolution(self, probe_data: dict[str, Any], brand: str) -> str | None:
        """Detect model based on resolution."""
        video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]

        for stream in video_streams:
            width = stream.get("width", 0)
            height = stream.get("height", 0)

            if brand == "insta360":
                if width >= 3840 or height >= 3840:
                    return "X3/X4"
                elif width >= 2880 or height >= 2880:
                    return "ONE RS"
                elif width >= 1920 or height >= 1920:
                    return "ONE X/X2"

        return None

    def _detect_qoocam_model(self, probe_data: dict[str, Any]) -> str | None:
        """Detect QooCam model based on resolution and codec."""
        video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]

        for stream in video_streams:
            width = stream.get("width", 0)
            codec = stream.get("codec_name", "")

            if width >= 3840:
                if codec == "hevc":
                    return "8K"  # QooCam 8K uses HEVC
                return "8K/3"
            elif width >= 2880:
                return "3"

        return None


# Register the extractor
register_extractor("camera_360", Camera360Extractor())
