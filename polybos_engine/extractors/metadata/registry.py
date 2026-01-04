"""Manufacturer detection and extractor registry.

This module provides a registry pattern for metadata extractors.
Each manufacturer module registers itself with detect() and extract() functions.

To add a new manufacturer:
1. Create a new module (e.g., panasonic.py)
2. Implement detect(probe_data, file_path) -> bool
3. Implement extract(probe_data, file_path, base_metadata) -> Metadata
4. Import the module in __init__.py to trigger registration
"""

import logging
from typing import Any, Protocol

from polybos_engine.schemas import Metadata

logger = logging.getLogger(__name__)


class MetadataExtractor(Protocol):
    """Protocol for manufacturer-specific metadata extractors."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this extractor handles the given file.

        Args:
            probe_data: Parsed ffprobe JSON output
            file_path: Path to video file

        Returns:
            True if this extractor should handle the file
        """
        ...

    def extract(
        self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata
    ) -> Metadata:
        """Extract manufacturer-specific metadata.

        Args:
            probe_data: Parsed ffprobe JSON output
            file_path: Path to video file
            base_metadata: Base metadata from ffprobe (device-agnostic)

        Returns:
            Enhanced Metadata with device-specific fields
        """
        ...


# Global registry of extractors
_extractors: list[tuple[str, MetadataExtractor]] = []


def register_extractor(name: str, extractor: MetadataExtractor) -> None:
    """Register a metadata extractor.

    Args:
        name: Extractor name (e.g., "dji", "sony", "apple")
        extractor: Extractor instance implementing detect() and extract()
    """
    _extractors.append((name, extractor))
    logger.debug(f"Registered metadata extractor: {name}")


def get_extractor(
    probe_data: dict[str, Any], file_path: str
) -> tuple[str, MetadataExtractor] | None:
    """Find the appropriate extractor for a file.

    Iterates through registered extractors in order and returns the first
    one whose detect() method returns True.

    Args:
        probe_data: Parsed ffprobe JSON output
        file_path: Path to video file

    Returns:
        Tuple of (name, extractor) or None if no match
    """
    for name, extractor in _extractors:
        try:
            if extractor.detect(probe_data, file_path):
                logger.debug(f"Matched extractor: {name}")
                return name, extractor
        except Exception as e:
            logger.warning(f"Extractor {name} detect() failed: {e}")
            continue

    return None


def list_extractors() -> list[str]:
    """List all registered extractor names."""
    return [name for name, _ in _extractors]


# Helper to get common tag values for detection
def get_tags_lower(probe_data: dict[str, Any]) -> dict[str, str]:
    """Get format tags with lowercase keys."""
    format_info = probe_data.get("format", {})
    tags = format_info.get("tags", {})
    return {k.lower(): v for k, v in tags.items()}


def get_make_model(probe_data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract make and model from common metadata locations.

    Returns:
        Tuple of (make, model) - either may be None
    """
    tags = get_tags_lower(probe_data)

    make = (
        tags.get("make")
        or tags.get("com.apple.quicktime.make")
        or tags.get("manufacturer")
    )
    model = (
        tags.get("model")
        or tags.get("com.apple.quicktime.model")
        or tags.get("model_name")
    )

    return make, model
