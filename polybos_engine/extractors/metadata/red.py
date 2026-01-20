"""RED Digital Cinema metadata extractor.

Handles RED cameras:
- RED ONE (2007-2013)
- SCARLET (2011-2016)
- EPIC (2010-2016)
- DRAGON/Weapon (2014-2018)
- Helium (2016-present)
- KOMODO (2020-present)
- V-Raptor (2021-present)

R3D files store metadata in a proprietary header format.
ffprobe CANNOT read R3D natively, so we parse the header directly.

Header structure (reverse-engineered):
- 0x00-0x03: Size
- 0x04-0x07: Magic "RED2"
- 0x08+: TLV-like blocks with type codes

Notable fields found in header:
- Timecode (format HH:MM:SS:FF)
- Date (format YYYYMMDD)
- Firmware version
- Camera model (SCARLET, EPIC, KOMODO, etc.)
- Serial number
- Lens info (make, model, focal length, aperture)
"""

import logging
import re
from pathlib import Path
from typing import Any

from polybos_engine.schemas import (
    Codec,
    ColorSpace,
    DetectionMethod,
    DeviceInfo,
    LensInfo,
    MediaDeviceType,
    Metadata,
    VideoCodec,
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
    "red one": "RED ONE",
    "dsmc2": "DSMC2",
    "dsmc3": "DSMC3",
}


def _parse_r3d_header(file_path: str) -> dict[str, Any] | None:
    """Parse R3D file header for metadata.

    Returns dict with extracted metadata or None if not a valid R3D.
    """
    try:
        with open(file_path, "rb") as f:
            # Read header (first 1KB should contain all metadata)
            header = f.read(1024)

        if len(header) < 8:
            return None

        # Check magic
        magic = header[4:8]
        if magic != b"RED2":
            return None

        result: dict[str, Any] = {"make": "RED"}

        # Find timecode (format like "01:00:26:06")
        tc_match = re.search(rb"\d{2}:\d{2}:\d{2}:\d{2}", header)
        if tc_match:
            tc_str = tc_match.group().decode("ascii")
            result["timecode"] = tc_str

        # Find date (format YYYYMMDD)
        date_match = re.search(rb"20\d{6}", header)
        if date_match:
            date_str = date_match.group().decode("ascii")
            # Format as YYYY-MM-DD
            result["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        # Find firmware version (X.X.XX pattern)
        fw_match = re.search(rb"\d+\.\d+\.\d+", header)
        if fw_match:
            result["firmware"] = fw_match.group().decode("ascii")

        # Find camera model - look for known RED models
        models = [
            b"SCARLET",
            b"EPIC",
            b"DRAGON",
            b"WEAPON",
            b"HELIUM",
            b"KOMODO",
            b"RAPTOR",
            b"RED ONE",
            b"GEMINI",
            b"MONSTRO",
            b"RANGER",
            b"RAVEN",
            b"DSMC2",
            b"DSMC3",
        ]
        for model in models:
            if model in header.upper():
                result["model"] = model.decode("ascii")
                break

        # Find serial number (pattern like 221LS102VTLCZ)
        serial_match = re.search(rb"\d{3}[A-Z]{2}\d{3}[A-Z0-9]+", header)
        if serial_match:
            result["serial"] = serial_match.group().decode("ascii")

        # Find lens info (look for known lens brands/patterns)
        lens_patterns = [
            rb"Canon [^\x00]+",
            rb"Zeiss [^\x00]+",
            rb"Leica [^\x00]+",
            rb"Sigma [^\x00]+",
            rb"Cooke [^\x00]+",
            rb"Angenieux [^\x00]+",
            rb"Fujinon [^\x00]+",
            rb"RED [^\x00]*PRO [^\x00]*",
        ]
        for pattern in lens_patterns:
            lens_match = re.search(pattern, header)
            if lens_match:
                lens_str = lens_match.group().decode("utf-8", errors="ignore")
                # Clean up null bytes and control chars
                lens_str = re.sub(r"[\x00-\x1f]", "", lens_str).strip()
                if len(lens_str) > 3:
                    result["lens_name"] = lens_str
                    break

        # Find lens serial (pattern like 0018-0046-00E6)
        lens_serial_match = re.search(rb"\d{4}-\d{4}-\d{4}", header)
        if lens_serial_match:
            result["lens_serial"] = lens_serial_match.group().decode("ascii")

        return result if len(result) > 1 else None

    except Exception as e:
        logger.warning(f"Failed to parse R3D header: {e}")
        return None


class RedExtractor:
    """Extract metadata from RED cameras."""

    def detect(self, probe_data: dict[str, Any], file_path: str) -> bool:
        """Detect if this is a RED R3D file.

        Detection methods:
        1. File extension (.R3D)
        2. RED folder structure (RDM/RDC)
        3. R3D magic bytes
        """
        path = Path(file_path)

        # Check file extension
        if path.suffix.upper() == ".R3D":
            return True

        # Check folder structure (RDM = RED Digital Magazine, RDC = RED Digital Clip)
        parts = path.parts
        for part in parts:
            if part.upper().endswith(".RDM") or part.upper().endswith(".RDC"):
                return True

        return False

    def extract(
        self,
        probe_data: dict[str, Any],
        file_path: str,
        base_metadata: Metadata,
    ) -> Metadata:
        """Extract RED-specific metadata from R3D file."""
        path = Path(file_path)

        # Parse R3D header directly (ffprobe cannot read R3D)
        r3d_data = _parse_r3d_header(file_path)

        if r3d_data is None:
            # Return minimal metadata
            device = DeviceInfo(
                make="RED",
                model=None,
                type=MediaDeviceType.CINEMA_CAMERA,
                detection_method=DetectionMethod.METADATA,
                confidence=0.8,
            )
            base_metadata.device = device
            return base_metadata

        # Build device info
        # Note: serial_number stored in software field as DeviceInfo doesn't have serial
        serial = r3d_data.get("serial")
        firmware = r3d_data.get("firmware")
        software_str = firmware
        if serial:
            software_str = (
                f"{firmware} (S/N: {serial})" if firmware else f"S/N: {serial}"
            )

        device = DeviceInfo(
            make="RED",
            model=r3d_data.get("model"),
            software=software_str,
            type=MediaDeviceType.CINEMA_CAMERA,
            detection_method=DetectionMethod.METADATA,
            confidence=1.0,
        )

        # Build lens info
        lens: LensInfo | None = None
        lens_name = r3d_data.get("lens_name")
        if lens_name:
            # Try to parse focal length from lens name
            focal_match = re.search(r"(\d+)-(\d+)mm|(\d+)mm", lens_name)
            focal_length: float | None = None
            if focal_match:
                if focal_match.group(3):
                    focal_length = float(focal_match.group(3))
                elif focal_match.group(1):
                    # Zoom lens - use wide end
                    focal_length = float(focal_match.group(1))

            # Try to parse aperture
            aperture_match = re.search(r"f/?([\d.]+)", lens_name)
            aperture: float | None = None
            if aperture_match:
                aperture = float(aperture_match.group(1))

            # Store lens make/model/serial in iris field as LensInfo lacks those fields
            lens_serial = r3d_data.get("lens_serial")
            iris_info = lens_name
            if lens_serial:
                iris_info = f"{lens_name} (S/N: {lens_serial})"

            lens = LensInfo(
                focal_length=focal_length,
                aperture=aperture,
                iris=iris_info,  # Store full lens info here
                detection_method=DetectionMethod.METADATA,
            )

        # Get timecode string
        timecode: str | None = r3d_data.get("timecode")

        # Get file size
        file_size = path.stat().st_size if path.exists() else base_metadata.file_size

        # R3D uses REDCODE compression
        video_codec = VideoCodec(
            name="REDCODE",
            profile="RAW",
            bit_depth=16,  # RED shoots 16-bit
        )

        # Color space - RED shoots in REDWideGamutRGB / Log3G10
        color_space = ColorSpace(
            primaries="REDWideGamutRGB",
            transfer="Log3G10",
            matrix=None,
            detection_method=DetectionMethod.METADATA,
        )

        return Metadata(
            duration=base_metadata.duration,
            resolution=base_metadata.resolution,
            codec=Codec(video="REDCODE"),
            video_codec=video_codec,
            audio=base_metadata.audio,
            fps=base_metadata.fps,
            bitrate=base_metadata.bitrate,
            file_size=file_size,
            timecode=timecode,
            created_at=base_metadata.created_at,
            device=device,
            gps=base_metadata.gps,
            color_space=color_space,
            lens=lens,
        )


# Register the extractor
register_extractor("red", RedExtractor())
