"""Sony metadata extraction.

Handles Sony cameras:
- Professional: FX6, FX3, FX9, Venice
- Alpha series: A7S, A7R, A1, etc.
- Consumer: ZV-1, ZV-E1, etc.

Detection methods:
- make tag: "Sony"
- major_brand: "XAVC"
- XML sidecar files (M01.XML pattern)

Sony XML sidecar files contain:
- Device info (manufacturer, modelName)
- GPS coordinates (ExifGPS group)
- Color space (CaptureGammaEquation, CaptureColorPrimaries)
- Lens info (FocalLength, FNumber, etc.)
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    GPS,
    ColorSpace,
    DetectionMethod,
    DeviceInfo,
    LensInfo,
    MediaDeviceType,
    Metadata,
)

from .base import SidecarMetadata, parse_dms_coordinate
from .registry import get_tags_lower, register_extractor

logger = logging.getLogger(__name__)


def _parse_xml_sidecar(video_path: str) -> SidecarMetadata | None:
    """Parse Sony XML sidecar file for additional metadata.

    Sony cameras create XML sidecar files with naming pattern:
    - Video: 20251014_C0476.MP4
    - XML:   20251014_C0476M01.XML
    """
    path = Path(video_path)

    xml_patterns = [
        path.with_suffix(".XML"),
        path.parent / f"{path.stem}M01.XML",
        path.parent / f"{path.stem}M01.xml",
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

        ns = {"nrt": "urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20"}

        device: DeviceInfo | None = None
        gps: GPS | None = None
        color_space: ColorSpace | None = None
        lens: LensInfo | None = None

        # Extract device info
        device_elem = root.find(".//nrt:Device", ns) or root.find(".//{*}Device")
        if device_elem is not None:
            manufacturer = device_elem.get("manufacturer")
            model_name = device_elem.get("modelName")

            if manufacturer or model_name:
                device = DeviceInfo(
                    make=manufacturer,
                    model=model_name,
                    software=None,
                    type=MediaDeviceType.CAMERA,
                    detection_method=DetectionMethod.XML_SIDECAR,
                    confidence=1.0,
                )

        # Extract GPS from ExifGPS group
        gps_group = root.find(".//{*}Group[@name='ExifGPS']")
        if gps_group is not None:
            gps_items: dict[str, str | None] = {}
            for item in gps_group.findall(".//{*}Item"):
                name = item.get("name")
                if name is not None:
                    gps_items[name] = item.get("value")

            if gps_items.get("Status") != "V":
                lat_str = gps_items.get("Latitude")
                lon_str = gps_items.get("Longitude")
                lat_ref = gps_items.get("LatitudeRef")
                lon_ref = gps_items.get("LongitudeRef")
                alt_str = gps_items.get("Altitude")

                if lat_str and lon_str:
                    lat = parse_dms_coordinate(lat_str, lat_ref)
                    lon = parse_dms_coordinate(lon_str, lon_ref)

                    if lat is not None and lon is not None:
                        try:
                            gps = GPS(
                                latitude=lat,
                                longitude=lon,
                                altitude=float(alt_str) if alt_str else None,
                            )
                        except ValueError:
                            pass

        # Extract color space
        color_items: dict[str, str | None] = {}
        for group_name in ["CameraUnitMetadata", "VideoLayout", "AcquisitionRecord"]:
            group = root.find(f".//*[@name='{group_name}']")
            if group is not None:
                for item in group.findall(".//{*}Item"):
                    name = item.get("name")
                    if name:
                        color_items[name] = item.get("value")

        for item in root.findall(".//{*}Item"):
            name = item.get("name")
            if name and name in [
                "CaptureGammaEquation",
                "CaptureColorPrimaries",
                "CodingEquations",
            ]:
                color_items[name] = item.get("value")

        lut_file: str | None = None
        for related in root.findall(".//{*}RelatedTo"):
            if related.get("rel") == "LUT":
                lut_file = related.get("file")
                break

        gamma = color_items.get("CaptureGammaEquation")
        primaries = color_items.get("CaptureColorPrimaries")
        coding = color_items.get("CodingEquations")

        if gamma or primaries or coding or lut_file:
            color_space = ColorSpace(
                transfer=gamma,
                primaries=primaries,
                matrix=coding,
                lut_file=lut_file,
                detection_method=DetectionMethod.XML_SIDECAR,
            )

        # Extract lens info
        lens_items: dict[str, str | None] = {}
        for group_name in ["Camera", "Lens", "CameraUnitMetadata"]:
            group = root.find(f".//*[@name='{group_name}']")
            if group is not None:
                for item in group.findall(".//{*}Item"):
                    name = item.get("name")
                    if name:
                        lens_items[name] = item.get("value")

        for item in root.findall(".//{*}Item"):
            name = item.get("name")
            if name and name in [
                "FocalLength",
                "FocalLength35mm",
                "FocalLengthIn35mmFilm",
                "FNumber",
                "Iris",
                "FocusDistance",
            ]:
                lens_items[name] = item.get("value")

        focal_length = lens_items.get("FocalLength")
        focal_35mm = lens_items.get("FocalLength35mm") or lens_items.get(
            "FocalLengthIn35mmFilm"
        )
        f_number = lens_items.get("FNumber")
        iris = lens_items.get("Iris")
        focus_dist = lens_items.get("FocusDistance")

        if focal_length or focal_35mm or f_number or iris:
            lens = LensInfo(
                focal_length=float(focal_length) if focal_length else None,
                focal_length_35mm=float(focal_35mm) if focal_35mm else None,
                aperture=float(f_number) if f_number else None,
                focus_distance=float(focus_dist) if focus_dist else None,
                iris=iris,
                detection_method=DetectionMethod.XML_SIDECAR,
            )

        if device or gps or color_space or lens:
            return SidecarMetadata(
                device=device, gps=gps, color_space=color_space, lens=lens
            )
        return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse Sony XML sidecar {xml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading Sony XML sidecar {xml_path}: {e}")
        return None


class SonyExtractor:
    """Metadata extractor for Sony cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if file is from a Sony camera."""
        tags = get_tags_lower(probe_data)

        # Check make tag
        make = tags.get("make") or tags.get("manufacturer")
        if make and "SONY" in make.upper():
            return True

        # Check major_brand for XAVC
        major_brand = tags.get("major_brand", "")
        if major_brand.upper() == "XAVC":
            return True

        # Check for Sony XML sidecar
        path = Path(file_path)
        xml_patterns = [
            path.with_suffix(".XML"),
            path.parent / f"{path.stem}M01.XML",
            path.parent / f"{path.stem}M01.xml",
        ]
        for pattern in xml_patterns:
            if pattern.exists():
                # Verify it's a Sony XML by checking namespace
                try:
                    tree = ET.parse(pattern)
                    root = tree.getroot()
                    # Check for Sony namespace or Device manufacturer
                    if "professionalDisc" in str(root.tag).lower():
                        return True
                    device = root.find(".//{*}Device")
                    if device is not None:
                        mfr = device.get("manufacturer", "")
                        if "Sony" in mfr:
                            return True
                except Exception:
                    pass

        return False

    def extract(
        self, probe_data: dict[str, Any], file_path: str, base_metadata: Metadata
    ) -> Metadata:
        """Extract Sony-specific metadata."""
        tags = get_tags_lower(probe_data)

        # Get basic device info from tags
        make = tags.get("make") or tags.get("manufacturer") or "Sony"
        model = tags.get("model") or tags.get("model_name")

        # Parse XML sidecar for detailed metadata
        sidecar = _parse_xml_sidecar(file_path)

        # Build device info (prefer sidecar)
        if sidecar and sidecar.device:
            device = sidecar.device
        else:
            device = DeviceInfo(
                make=make if make else "Sony",
                model=model,
                software=tags.get("software"),
                type=MediaDeviceType.CAMERA,
                detection_method=DetectionMethod.METADATA,
                confidence=1.0,
            )

        # Merge metadata (prefer sidecar values)
        gps = sidecar.gps if sidecar and sidecar.gps else base_metadata.gps
        color_space = (
            sidecar.color_space
            if sidecar and sidecar.color_space
            else base_metadata.color_space
        )
        lens = sidecar.lens if sidecar and sidecar.lens else base_metadata.lens

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
            color_space=color_space,
            lens=lens,
        )


# Register this extractor
register_extractor("sony", SonyExtractor())
