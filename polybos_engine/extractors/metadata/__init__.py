"""Modular metadata extraction.

This package provides manufacturer-specific metadata extractors.
Each manufacturer module registers itself on import.

Usage:
    from polybos_engine.extractors.metadata import extract_metadata

    metadata = extract_metadata("/path/to/video.mp4")

To add a new manufacturer:
1. Create a new module (e.g., panasonic.py)
2. Implement a class with detect() and extract() methods
3. Register it using: register_extractor("panasonic", PanasonicExtractor())
4. Import the module below to trigger registration

The order of imports determines detection priority.
Specific manufacturers should be imported before generic.
"""

import logging
from pathlib import Path

from polybos_engine.schemas import Metadata

# Import manufacturer modules to trigger registration
# Order matters: more specific extractors first
from . import (
    apple,  # noqa: F401
    blackmagic,  # noqa: F401
    canon,  # noqa: F401
    dji,  # noqa: F401
    sony,  # noqa: F401
)
from .base import build_base_metadata, run_ffprobe

# Import and register generic fallback LAST
from .generic import GenericExtractor
from .registry import get_extractor, list_extractors, register_extractor

register_extractor("generic", GenericExtractor())

logger = logging.getLogger(__name__)

__all__ = [
    "extract_metadata",
    "list_extractors",
]


def extract_metadata(file_path: str) -> Metadata:
    """Extract metadata from video file.

    This function:
    1. Runs ffprobe to get basic metadata
    2. Detects the manufacturer/device
    3. Calls the appropriate extractor for enhanced metadata

    Args:
        file_path: Path to video file

    Returns:
        Metadata object with video information
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    # Run ffprobe
    probe_data = run_ffprobe(file_path)

    # Build base metadata (device-agnostic)
    base_metadata = build_base_metadata(probe_data, file_path)

    # Find and run the appropriate extractor
    match = get_extractor(probe_data, file_path)

    if match:
        name, extractor = match
        logger.info(f"Using {name} extractor for {path.name}")
        try:
            return extractor.extract(probe_data, file_path, base_metadata)
        except Exception as e:
            logger.warning(f"Extractor {name} failed: {e}, using base metadata")
            return base_metadata
    else:
        # This shouldn't happen since generic always matches
        logger.warning(f"No extractor matched for {path.name}")
        return base_metadata
