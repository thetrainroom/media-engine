"""Canon metadata extraction.

Handles Canon cameras:
- Cinema EOS: C70, C300, C500, etc.
- EOS R series: R5, R6, R3, etc.
- DSLRs: 5D, 1DX, etc.

Detection methods:
- make tag: "Canon"
- XML sidecar files (.XML)

Canon XML sidecar files contain:
- Device info (Manufacturer, ModelName)
- GPS coordinates (Location element)
- Creation date (CreationDate element)

Canon Cinema EOS MXF filename format:
- Example: A012C001_230515_BY9X.MXF or A012C001_230515BY9X.MXF
- The YYMMDD date is embedded after the clip number
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from media_engine.schemas import (
    GPS,
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .base import SidecarMetadata
from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)

# Pattern for Canon Cinema EOS MXF filenames with embedded date
# Format: A###C###H<YYMMDD><XX>_CANON.MXF
# Example: A012C001H200529BY_CANON.MXF -> date is 200529 (2020-05-29)
CANON_DATE_PATTERN = re.compile(r"H(\d{6})", re.IGNORECASE)


def _parse_date_from_filename(file_path: str) -> datetime | None:
    """Extract recording date from Canon MXF filename.

    Canon Cinema EOS cameras encode the date in the filename:
    - A012C001_230515_BY9X.MXF -> 2023-05-15
    - A012C001_230515BY9X.MXF -> 2023-05-15
    - CLIP_230515.MXF -> 2023-05-15

    The date format is YYMMDD (2-digit year, month, day).
    """
    filename = Path(file_path).stem

    match = CANON_DATE_PATTERN.search(filename)
    if not match:
        return None

    date_str = match.group(1)
    try:
        # Parse YYMMDD format
        year = int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        # Convert 2-digit year to 4-digit (assume 20xx for now)
        full_year = 2000 + year if year < 70 else 1900 + year

        # Validate date components
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None

        return datetime(full_year, month, day, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def _parse_xml_sidecar(video_path: str) -> SidecarMetadata | None:
    """Parse Canon XML sidecar file for additional metadata.

    Canon cameras create XML sidecar files with naming pattern:
    - Video: A012C001_230515_BY9X.MXF
    - XML:   A012C001_230515_BY9X.XML
    """
    path = Path(video_path)

    xml_patterns = [
        path.with_suffix(".XML"),
        path.with_suffix(".xml"),
    ]

    xml_path = None
    for pattern in xml_patterns:
        if pattern.exists():
            xml_path = pattern
            break

    if not xml_path:
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ns = {"canon": "http://www.canon.com/ns/VideoClip"}

        device: DeviceInfo | None = None
        gps: GPS | None = None
        created_at: datetime | None = None

        # Extract device info
        device_elem = root.find(".//canon:Device", ns) or root.find(".//{*}Device")
        if device_elem is not None:
            manufacturer_elem = device_elem.find("canon:Manufacturer", ns) or device_elem.find("{*}Manufacturer")
            model_elem = device_elem.find("canon:ModelName", ns) or device_elem.find("{*}ModelName")

            manufacturer = manufacturer_elem.text if manufacturer_elem is not None else None
            model_name = model_elem.text if model_elem is not None else None

            if manufacturer or model_name:
                device = DeviceInfo(
                    make=manufacturer,
                    model=model_name,
                    software=None,
                    type=MediaDeviceType.CAMERA,
                    detection_method=DetectionMethod.XML_SIDECAR,
                    confidence=1.0,
                )

        # Extract creation date - try multiple possible element names
        date_elements = [
            ".//canon:CreationDate",
            ".//canon:StartDate",
            ".//canon:Date",
            ".//{*}CreationDate",
            ".//{*}StartDate",
            ".//{*}Date",
        ]
        for date_xpath in date_elements:
            if date_xpath.startswith(".//canon:"):
                date_elem = root.find(date_xpath, ns)
            else:
                date_elem = root.find(date_xpath)

            if date_elem is not None and date_elem.text:
                try:
                    # Try ISO format first (2023-05-15T10:30:00)
                    date_text = date_elem.text.strip()
                    if "T" in date_text:
                        created_at = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                    else:
                        # Try date only (2023-05-15)
                        created_at = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue

        # Extract GPS from Location element
        location_elem = root.find(".//canon:Location", ns) or root.find(".//{*}Location")
        if location_elem is not None:
            lat_elem = location_elem.find("canon:Latitude", ns) or location_elem.find("{*}Latitude")
            lon_elem = location_elem.find("canon:Longitude", ns) or location_elem.find("{*}Longitude")
            alt_elem = location_elem.find("canon:Altitude", ns) or location_elem.find("{*}Altitude")

            lat = lat_elem.text if lat_elem is not None and lat_elem.text else None
            lon = lon_elem.text if lon_elem is not None and lon_elem.text else None
            alt = alt_elem.text if alt_elem is not None and alt_elem.text else None

            if lat and lon:
                try:
                    gps = GPS(
                        latitude=float(lat),
                        longitude=float(lon),
                        altitude=float(alt) if alt else None,
                    )
                except ValueError:
                    pass

        if device or gps or created_at:
            return SidecarMetadata(device=device, gps=gps, created_at=created_at)
        return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse Canon XML sidecar {xml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading Canon XML sidecar {xml_path}: {e}")
        return None


class CanonExtractor:
    """Metadata extractor for Canon cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a Canon camera."""
        tags = get_tags_lower(probe_data)

        # Check make tag (various names used by different formats)
        make = tags.get("make") or tags.get("manufacturer") or tags.get("company_name")
        if make and "CANON" in make.upper():
            return True

        # Check for Canon XML sidecar
        path = Path(file_path)
        xml_patterns = [
            path.with_suffix(".XML"),
            path.with_suffix(".xml"),
        ]
        for pattern in xml_patterns:
            if pattern.exists():
                try:
                    tree = ET.parse(pattern)
                    root = tree.getroot()
                    # Check for Canon namespace
                    if "canon.com" in str(root.tag).lower():
                        return True
                    # Check device manufacturer
                    device = root.find(".//{*}Device")
                    if device is not None:
                        mfr_elem = device.find(".//{*}Manufacturer")
                        if mfr_elem is not None and mfr_elem.text:
                            if "Canon" in mfr_elem.text:
                                return True
                except Exception:
                    pass

        return False

    def extract(self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata) -> Metadata:
        """Extract Canon-specific metadata."""
        tags = get_tags_lower(probe_data)

        # Get basic device info from tags
        make = tags.get("make") or tags.get("manufacturer") or "Canon"
        model = tags.get("model") or tags.get("model_name")

        # Parse XML sidecar for detailed metadata
        sidecar = _parse_xml_sidecar(file_path)

        # Build device info (prefer sidecar)
        if sidecar and sidecar.device:
            device = sidecar.device
        else:
            device = DeviceInfo(
                make=make if make else "Canon",
                model=model,
                software=tags.get("software"),
                type=MediaDeviceType.CAMERA,
                detection_method=DetectionMethod.METADATA,
                confidence=1.0,
            )

        # Merge metadata
        gps = sidecar.gps if sidecar and sidecar.gps else base_metadata.gps

        # Get creation date: prefer base_metadata, then sidecar, then filename
        created_at = base_metadata.created_at
        if created_at is None and sidecar and sidecar.created_at:
            created_at = sidecar.created_at
            logger.debug(f"Got creation date from XML sidecar: {created_at}")
        if created_at is None:
            created_at = _parse_date_from_filename(file_path)
            if created_at:
                logger.debug(f"Parsed creation date from filename: {created_at}")

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
            created_at=created_at,
            device=device,
            gps=gps,
            color_space=base_metadata.color_space,
            lens=base_metadata.lens,
        )


# Register this extractor
register_extractor("canon", CanonExtractor())
