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
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    GPS,
    DetectionMethod,
    DeviceInfo,
    MediaDeviceType,
    Metadata,
)

from .base import SidecarMetadata
from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


def _parse_xml_sidecar(video_path: str) -> SidecarMetadata | None:
    """Parse Canon XML sidecar file for additional metadata.

    Canon cameras create XML sidecar files with naming pattern:
    - Video: A012C001H200529BY_CANON.MXF
    - XML:   A012C001H200529BY_CANON.XML
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

        # Extract device info
        device_elem = root.find(".//canon:Device", ns) or root.find(".//{*}Device")
        if device_elem is not None:
            manufacturer_elem = device_elem.find(
                "canon:Manufacturer", ns
            ) or device_elem.find("{*}Manufacturer")
            model_elem = device_elem.find("canon:ModelName", ns) or device_elem.find(
                "{*}ModelName"
            )

            manufacturer = (
                manufacturer_elem.text if manufacturer_elem is not None else None
            )
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

        # Extract GPS from Location element
        location_elem = root.find(".//canon:Location", ns) or root.find(
            ".//{*}Location"
        )
        if location_elem is not None:
            lat_elem = location_elem.find("canon:Latitude", ns) or location_elem.find(
                "{*}Latitude"
            )
            lon_elem = location_elem.find("canon:Longitude", ns) or location_elem.find(
                "{*}Longitude"
            )
            alt_elem = location_elem.find("canon:Altitude", ns) or location_elem.find(
                "{*}Altitude"
            )

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

        if device or gps:
            return SidecarMetadata(device=device, gps=gps)
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

        # Check make tag
        make = tags.get("make") or tags.get("manufacturer")
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

    def extract(
        self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata
    ) -> Metadata:
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
            gps=gps,
            color_space=base_metadata.color_space,
            lens=base_metadata.lens,
        )


# Register this extractor
register_extractor("canon", CanonExtractor())
