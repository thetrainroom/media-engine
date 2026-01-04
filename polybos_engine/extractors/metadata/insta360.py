"""Insta360 metadata extractor.

Detects Insta360 cameras (X3, X4, ONE RS, GO 3, etc.) via:
- .insv/.insp file extension
- "INS" prefix in handler_name
- Square resolution (360 video has two 1:1 streams)
"""

import logging
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import register_extractor

logger = logging.getLogger(__name__)


class Insta360Extractor:
    """Extract metadata from Insta360 cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this is an Insta360 file."""
        path = Path(file_path)

        # Check file extension (.insv for video, .insp for photo)
        if path.suffix.lower() in (".insv", ".insp"):
            return True

        # Check handler_name for INS prefix
        for stream in probe_data.get("streams", []):
            handler = stream.get("tags", {}).get("handler_name", "")
            if "INS" in handler.upper():
                return True

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract Insta360-specific metadata."""
        # Detect model and whether it's 360 format
        model, is_360 = self._detect_model(probe_data, file_path)

        # 360 cameras with unstitched footage get CAMERA_360 type
        device_type = MediaDeviceType.CAMERA_360 if is_360 else MediaDeviceType.ACTION_CAMERA

        device = DeviceInfo(
            make="Insta360",
            model=model,
            type=device_type,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        base_metadata.device = device

        return base_metadata

    def _detect_model(
        self, probe_data: dict[str, Any], file_path: str
    ) -> tuple[str | None, bool]:
        """Detect Insta360 model and whether it's 360 format.

        Returns:
            Tuple of (model_name, is_360_format)
        """
        path = Path(file_path)

        # .insv files are always unstitched 360
        is_360 = path.suffix.lower() == ".insv"

        # Count video streams - 360 cameras have 2 (front + back lens)
        video_streams = [
            s for s in probe_data.get("streams", [])
            if s.get("codec_type") == "video"
        ]

        if len(video_streams) >= 2:
            is_360 = True

        # Check resolution for model hints
        for stream in video_streams:
            width = stream.get("width", 0)
            height = stream.get("height", 0)

            # Square streams indicate unstitched 360 video
            if width == height:
                is_360 = True
                if width >= 3840:
                    return "X3/X4", is_360
                elif width >= 2880:
                    return "ONE RS", is_360

            # Equirectangular (stitched) 360: 2:1 aspect ratio
            if width > 0 and height > 0 and abs(width / height - 2.0) < 0.1:
                is_360 = True

        return None, is_360


# Register the extractor
register_extractor("insta360", Insta360Extractor())
