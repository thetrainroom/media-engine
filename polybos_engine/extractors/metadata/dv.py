"""DV and HDV format detection.

Detects tape-based camcorder formats:
- DV (SD): 720x480 NTSC or 720x576 PAL, dvvideo codec
- DVCAM: Professional DV variant
- DVCPRO: Panasonic professional DV
- HDV (HD): 1440x1080 or 1280x720, mpeg2video codec

These formats were used by consumer and prosumer camcorders
from the late 1990s through the 2010s.
"""

import logging
from typing import Any

from polybos_engine.schemas import (
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


class DVExtractor:
    """Metadata extractor for DV and HDV formats."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is DV or HDV format."""
        # Check video codec
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue

            codec = stream.get("codec_name", "").lower()

            # DV codec
            if codec == "dvvideo":
                return True

            # HDV uses mpeg2video with specific encoder tag
            if codec == "mpeg2video":
                tags = get_tags_lower(probe_data)
                encoder = tags.get("encoder", "").lower()
                if "hdv" in encoder:
                    return True

                # Also check stream tags
                stream_tags = stream.get("tags", {})
                for key, value in stream_tags.items():
                    if "hdv" in str(value).lower():
                        return True

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract DV/HDV format information."""
        format_name = "DV"
        model = "DV Camcorder"

        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue

            codec = stream.get("codec_name", "").lower()
            width = stream.get("width", 0)
            height = stream.get("height", 0)

            if codec == "dvvideo":
                # Detect DV variant
                if height == 576:
                    format_name = "DV PAL"
                elif height == 480:
                    format_name = "DV NTSC"
                else:
                    format_name = "DV"
                model = f"{format_name} Camcorder"

            elif codec == "mpeg2video":
                # HDV format
                tags = get_tags_lower(probe_data)
                encoder = tags.get("encoder", "")

                if "1080" in encoder:
                    format_name = "HDV 1080i"
                elif "720" in encoder:
                    format_name = "HDV 720p"
                elif width == 1440 and height == 1080:
                    format_name = "HDV 1080i"
                elif width == 1280 and height == 720:
                    format_name = "HDV 720p"
                else:
                    format_name = "HDV"

                model = f"{format_name} Camcorder"

            break

        device = DeviceInfo(
            make=None,
            model=model,
            type=MediaDeviceType.CAMERA,
            detection_method=DetectionMethod.METADATA,
            confidence=0.9,
        )

        base_metadata.device = device

        return base_metadata


# Register the extractor
register_extractor("dv", DVExtractor())
