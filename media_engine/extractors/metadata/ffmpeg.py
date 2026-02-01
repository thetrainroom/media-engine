"""FFmpeg metadata extraction.

Handles files encoded/processed with FFmpeg:
- OBS recordings
- Handbrake conversions
- Command-line FFmpeg output
- Other FFmpeg-based tools

Detection:
- encoder tag starts with "Lavf" (libavformat)
"""

import logging
from typing import Any

from media_engine.schemas import (
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


class FFmpegExtractor:
    """Metadata extractor for FFmpeg-encoded files."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file was encoded with FFmpeg."""
        tags = get_tags_lower(probe_data)

        # Check encoder tag for libavformat signature
        encoder = tags.get("encoder", "")
        if encoder.startswith("Lavf"):
            return True

        return False

    def extract(self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata) -> Metadata:
        """Extract metadata for FFmpeg-encoded files."""
        tags = get_tags_lower(probe_data)

        encoder = tags.get("encoder", "")

        device = DeviceInfo(
            make="FFmpeg",
            model=encoder if encoder else None,
            software=encoder if encoder else None,
            type=MediaDeviceType.UNKNOWN,
            detection_method=DetectionMethod.METADATA,
            confidence=0.8,
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


# Register this extractor
register_extractor("ffmpeg", FFmpegExtractor())
