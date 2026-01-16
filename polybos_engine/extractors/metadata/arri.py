"""ARRI metadata extractor.

Detects ARRI cameras (ALEXA, ALEXA Mini, ALEXA 35, AMIRA) via:
- .ari extension (ARRIRAW)
- .arx extension (ARRIRAW HDE)
- .mxf with ARRI metadata

Note: Full ARRIRAW metadata requires ARRI Meta Extract tool (free download).
Without it, we detect the format but can't read detailed metadata.
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

# ARRI camera models
ARRI_MODELS = {
    "alexa35": "ALEXA 35",
    "alexa 35": "ALEXA 35",
    "alexamini": "ALEXA Mini",
    "alexa mini": "ALEXA Mini",
    "minilf": "ALEXA Mini LF",
    "mini lf": "ALEXA Mini LF",
    "alexalf": "ALEXA LF",
    "alexa lf": "ALEXA LF",
    "alexa65": "ALEXA 65",
    "alexa 65": "ALEXA 65",
    "amira": "AMIRA",
    "alexa": "ALEXA",
}


class ArriExtractor:
    """Extract metadata from ARRI cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this is an ARRI file."""
        path = Path(file_path)

        # ARRIRAW extensions
        if path.suffix.lower() in (".ari", ".arx"):
            return True

        # Check for ARRI in metadata (for MXF/MOV files)
        tags = probe_data.get("format", {}).get("tags", {})
        for value in tags.values():
            if "arri" in str(value).lower():
                return True

        # Check stream metadata
        for stream in probe_data.get("streams", []):
            stream_tags = stream.get("tags", {})
            for value in stream_tags.values():
                if "arri" in str(value).lower():
                    return True

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract ARRI-specific metadata."""
        model = self._detect_model(probe_data, file_path)

        device = DeviceInfo(
            make="ARRI",
            model=model,
            type=MediaDeviceType.CINEMA_CAMERA,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        base_metadata.device = device

        # Note: For full ARRIRAW metadata, would need ARRI Meta Extract
        # Log a hint for users
        path = Path(file_path)
        if path.suffix.lower() in (".ari", ".arx"):
            logger.info(
                "ARRIRAW detected. For full metadata, install ARRI Meta Extract."
            )

        return base_metadata

    def _detect_model(self, probe_data: dict[str, Any], file_path: str) -> str | None:
        """Try to detect ARRI camera model."""
        # Check all metadata for model hints
        all_text = ""

        tags = probe_data.get("format", {}).get("tags", {})
        all_text += " ".join(str(v) for v in tags.values()).lower()

        for stream in probe_data.get("streams", []):
            stream_tags = stream.get("tags", {})
            all_text += " ".join(str(v) for v in stream_tags.values()).lower()

        # Search for known models
        for model_key, model_name in ARRI_MODELS.items():
            if model_key in all_text:
                return model_name

        return None


# Register the extractor
register_extractor("arri", ArriExtractor())
