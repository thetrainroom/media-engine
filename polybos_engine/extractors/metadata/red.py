"""RED Digital Cinema metadata extractor.

Detects RED cameras (DSMC, DSMC2, V-RAPTOR, KOMODO) via .r3d extension.
FFprobe can read R3D files natively.
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

# RED camera models by sensor/body
RED_MODELS = {
    "dragon": "RED Dragon",
    "helium": "RED Helium",
    "gemini": "RED Gemini",
    "monstro": "RED Monstro",
    "komodo": "KOMODO",
    "raptor": "V-RAPTOR",
    "ranger": "RANGER",
    "weapon": "WEAPON",
    "epic": "EPIC",
    "scarlet": "SCARLET",
    "raven": "RAVEN",
}


class RedExtractor:
    """Extract metadata from RED cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this is a RED R3D file."""
        path = Path(file_path)
        return path.suffix.lower() == ".r3d"

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract RED-specific metadata."""
        model = self._detect_model(probe_data)

        device = DeviceInfo(
            make="RED",
            model=model,
            type=MediaDeviceType.CINEMA_CAMERA,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        base_metadata.device = device

        return base_metadata

    def _detect_model(self, probe_data: dict[str, Any]) -> str | None:
        """Try to detect RED camera model from metadata."""
        # Check format tags
        tags = probe_data.get("format", {}).get("tags", {})

        # RED sometimes puts camera info in metadata
        for key, value in tags.items():
            key_lower = key.lower()
            value_lower = str(value).lower()

            for model_key, model_name in RED_MODELS.items():
                if model_key in key_lower or model_key in value_lower:
                    return model_name

        return None


# Register the extractor
register_extractor("red", RedExtractor())
