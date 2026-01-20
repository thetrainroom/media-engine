"""AVCHD GPS extraction from H.264 SEI MDPM data.

Sony AVCHD cameras (HXR-NX5, HDR-CX series, etc.) embed GPS data
in the H.264 video stream using MDPM (Modified Digital Video Pack
Metadata) within SEI NAL units.

The MDPM format is identified by UUID: 17ee8c60-f84d-11d9-8cd6-0800200c9a66
followed by "MDPM" marker, then tag-value pairs.

GPS tags (from ExifTool H264.pm):
- 0xB0: GPSVersionID
- 0xB1: GPSLatitudeRef ('N' or 'S')
- 0xB2-B4: GPSLatitude (degrees, minutes, seconds as rationals)
- 0xB5: GPSLongitudeRef ('E' or 'W')
- 0xB6-B8: GPSLongitude (degrees, minutes, seconds as rationals)
- 0xB9: GPSAltitudeRef (0=above sea level, 1=below)
- 0xBA: GPSAltitude
- 0xBB-BD: GPSTimeStamp (hours, minutes, seconds)
- 0xBE: GPSStatus ('A'=active, 'V'=void)
- 0xBF: GPSMeasureMode
- 0xC0: GPSDOP
- 0xC2: GPSSpeed
- 0xCA: GPSDateStamp

Each tag is 1 byte followed by 4 bytes of value data (typically rational).
"""

import logging
from pathlib import Path

from polybos_engine.schemas import GPS, GPSTrack, GPSTrackPoint

logger = logging.getLogger(__name__)

# MDPM UUID used by Sony for embedded metadata in H.264 SEI
MDPM_UUID = bytes.fromhex("17ee8c60f84d11d98cd60800200c9a66")


def _parse_rational(value_bytes: bytes) -> float:
    """Parse 4-byte rational (2-byte numerator, 2-byte denominator)."""
    num = (value_bytes[0] << 8) | value_bytes[1]
    denom = (value_bytes[2] << 8) | value_bytes[3]
    return num / denom if denom > 0 else float(num)


def _extract_gps_from_mdpm_block(mdpm_data: bytes) -> dict[str, float | str] | None:
    """Extract GPS coordinates from a single MDPM block.

    Returns dict with latitude, longitude, altitude, status or None if invalid.
    """
    # Find GPS section start (tag 0xB0)
    try:
        gps_start = mdpm_data.index(b"\xb0")
    except ValueError:
        return None

    lat_ref: str | None = None
    lat_deg: float = 0.0
    lat_min: float = 0.0
    lat_sec: float = 0.0
    lon_ref: str | None = None
    lon_deg: float = 0.0
    lon_min: float = 0.0
    lon_sec: float = 0.0
    altitude: float | None = None
    status: str = "A"

    i = gps_start

    while i < len(mdpm_data) - 5:
        tag = mdpm_data[i]

        # Stop if we've passed GPS section
        if tag > 0xCA and tag < 0xE0:
            break
        if tag > 0xE6:
            break

        # Skip null bytes
        if tag == 0x00:
            i += 1
            continue

        value = mdpm_data[i + 1 : i + 5]

        if tag == 0xB1 and value[0] in (ord("N"), ord("S")):
            lat_ref = chr(value[0])
        elif tag == 0xB2:
            lat_deg = _parse_rational(value)
        elif tag == 0xB3:
            lat_min = _parse_rational(value)
        elif tag == 0xB4:
            lat_sec = _parse_rational(value)
        elif tag == 0xB5 and value[0] in (ord("E"), ord("W")):
            lon_ref = chr(value[0])
        elif tag == 0xB6:
            lon_deg = _parse_rational(value)
        elif tag == 0xB7:
            lon_min = _parse_rational(value)
        elif tag == 0xB8:
            lon_sec = _parse_rational(value)
        elif tag == 0xBA:
            altitude = _parse_rational(value)
        elif tag == 0xBE and value[0] in (ord("A"), ord("V")):
            status = chr(value[0])

        i += 5

    # Validate complete GPS reading
    if lat_ref is None or lon_ref is None:
        return None

    # Convert to decimal degrees
    lat = lat_deg + lat_min / 60 + lat_sec / 3600
    if lat_ref == "S":
        lat = -lat

    lon = lon_deg + lon_min / 60 + lon_sec / 3600
    if lon_ref == "W":
        lon = -lon

    result: dict[str, float | str] = {
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "status": status,
    }
    if altitude is not None:
        result["altitude"] = round(altitude, 1)

    return result


def extract_avchd_gps(file_path: str) -> GPS | None:
    """Extract GPS from AVCHD file embedded in H.264 SEI.

    Sony AVCHD cameras embed GPS data in the H.264 video stream using
    MDPM (Modified Digital Video Pack Metadata) within SEI NAL units.

    Args:
        file_path: Path to MTS/M2TS file

    Returns:
        GPS object with first valid GPS point, or None if no GPS found.
    """
    path = Path(file_path)

    # Only process MTS/M2TS files (AVCHD)
    if path.suffix.upper() not in (".MTS", ".M2TS"):
        return None

    try:
        with open(file_path, "rb") as f:
            # Detect packet size (188 for TS, 192 for MTS with timecode)
            header = f.read(8)
            f.seek(0)

            if len(header) < 8:
                return None

            if header[4] == 0x47:
                # 192-byte packets (4-byte timecode + 188-byte TS)
                pass  # packet_size = 192
            elif header[0] == 0x47:
                # Standard 188-byte TS packets
                pass  # packet_size = 188
            else:
                return None

            # Read file to find MDPM blocks
            data = f.read()

        # Find first valid GPS point
        pos = 0
        while True:
            pos = data.find(MDPM_UUID, pos)
            if pos == -1:
                break

            # Skip UUID (16) + "MDPM" marker (4) = 20 bytes
            mdpm_start = pos + 20
            mdpm_data = data[mdpm_start : mdpm_start + 200]

            gps_dict = _extract_gps_from_mdpm_block(mdpm_data)

            if gps_dict and gps_dict.get("status") == "A":
                lat = gps_dict["latitude"]
                lon = gps_dict["longitude"]
                alt = gps_dict.get("altitude")

                # Type narrowing for pyright
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    gps = GPS(
                        latitude=float(lat),
                        longitude=float(lon),
                        altitude=float(alt) if isinstance(alt, (int, float)) else None,
                    )
                    logger.info(f"Extracted GPS from AVCHD SEI: {lat:.6f}, {lon:.6f}")
                    return gps

            pos += 1

        return None

    except Exception as e:
        logger.warning(f"Failed to extract AVCHD GPS from {file_path}: {e}")
        return None


def extract_avchd_gps_track(file_path: str, max_points: int = 10000) -> GPSTrack | None:
    """Extract full GPS track from AVCHD file.

    Args:
        file_path: Path to MTS/M2TS file
        max_points: Maximum number of GPS points to extract

    Returns:
        GPSTrack object with all GPS points, or None if no GPS found.
    """
    path = Path(file_path)

    if path.suffix.upper() not in (".MTS", ".M2TS"):
        return None

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        gps_points: list[GPSTrackPoint] = []
        last_lat: float | None = None
        last_lon: float | None = None
        pos = 0

        while len(gps_points) < max_points:
            pos = data.find(MDPM_UUID, pos)
            if pos == -1:
                break

            mdpm_start = pos + 20
            mdpm_data = data[mdpm_start : mdpm_start + 200]

            gps_dict = _extract_gps_from_mdpm_block(mdpm_data)

            if gps_dict and gps_dict.get("status") == "A":
                lat = gps_dict["latitude"]
                lon = gps_dict["longitude"]
                alt = gps_dict.get("altitude")

                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    # Dedupe consecutive identical points
                    if lat != last_lat or lon != last_lon:
                        point = GPSTrackPoint(
                            latitude=float(lat),
                            longitude=float(lon),
                            altitude=(
                                float(alt) if isinstance(alt, (int, float)) else None
                            ),
                        )
                        gps_points.append(point)
                        last_lat = float(lat)
                        last_lon = float(lon)

            pos += 1

        if gps_points:
            logger.info(f"Extracted {len(gps_points)} GPS points from AVCHD SEI")
            return GPSTrack(points=gps_points, source="avchd_sei")

        return None

    except Exception as e:
        logger.warning(f"Failed to extract AVCHD GPS track from {file_path}: {e}")
        return None
