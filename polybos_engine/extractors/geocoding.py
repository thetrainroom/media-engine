"""Geocoding and POI lookup using OpenStreetMap Nominatim API."""

import logging
import time
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Nominatim requires a user agent
USER_AGENT = "PolybosMediaArchive/1.0"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

# Rate limiting - Nominatim allows max 1 request/second
_last_request_time: float = 0


def _rate_limit() -> None:
    """Ensure we don't exceed Nominatim's rate limit."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < 1.1:  # 1.1 seconds between requests
        time.sleep(1.1 - elapsed)
    _last_request_time = time.time()


@lru_cache(maxsize=1000)
def reverse_geocode(lat: float, lon: float) -> dict[str, Any] | None:
    """Reverse geocode coordinates to get address and location details.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dict with address details or None if lookup failed
    """
    _rate_limit()

    try:
        response = requests.get(
            f"{NOMINATIM_BASE}/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1,
                "extratags": 1,
                "namedetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            logger.warning(f"Nominatim error for ({lat}, {lon}): {data['error']}")
            return None

        return data
    except Exception as e:
        logger.warning(f"Reverse geocode failed for ({lat}, {lon}): {e}")
        return None


@lru_cache(maxsize=1000)
def find_nearby_pois(
    lat: float,
    lon: float,
    radius_meters: int = 500,
) -> list[dict[str, Any]]:
    """Find points of interest near coordinates.

    Uses Nominatim search with viewbox to find nearby landmarks.

    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius in meters

    Returns:
        List of POIs with name, type, and distance
    """
    _rate_limit()

    # Convert radius to approximate degree offset (rough approximation)
    # 1 degree latitude ≈ 111km, 1 degree longitude varies by latitude
    lat_offset = radius_meters / 111000
    lon_offset = radius_meters / (111000 * abs(lat) / 90) if lat != 0 else radius_meters / 111000

    # Create bounding box
    viewbox = f"{lon - lon_offset},{lat + lat_offset},{lon + lon_offset},{lat - lat_offset}"

    pois = []

    # Search for various POI types
    poi_queries = [
        "landmark",
        "monument",
        "lighthouse",
        "church",
        "castle",
        "museum",
        "historic",
        "viewpoint",
        "attraction",
    ]

    for query in poi_queries:
        _rate_limit()
        try:
            response = requests.get(
                f"{NOMINATIM_BASE}/search",
                params={
                    "q": query,
                    "format": "json",
                    "viewbox": viewbox,
                    "bounded": 1,
                    "limit": 5,
                    "addressdetails": 1,
                    "extratags": 1,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
            results = response.json()

            for result in results:
                # Calculate approximate distance
                result_lat = float(result.get("lat", 0))
                result_lon = float(result.get("lon", 0))
                dist = _haversine_distance(lat, lon, result_lat, result_lon)

                if dist <= radius_meters:
                    poi = {
                        "name": result.get("display_name", "").split(",")[0],
                        "type": result.get("type", query),
                        "class": result.get("class", ""),
                        "distance_m": int(dist),
                        "lat": result_lat,
                        "lon": result_lon,
                    }
                    # Avoid duplicates
                    if not any(p["name"] == poi["name"] for p in pois):
                        pois.append(poi)

        except Exception as e:
            logger.debug(f"POI search for '{query}' failed: {e}")
            continue

    # Sort by distance
    pois.sort(key=lambda x: x["distance_m"])
    return pois


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Earth's radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def get_location_context(
    lat: float | None,
    lon: float | None,
    user_location: str | None = None,
) -> dict[str, str]:
    """Build location context for AI processing.

    Combines user-provided location with geocoding and POI lookup.

    Args:
        lat: GPS latitude (optional)
        lon: GPS longitude (optional)
        user_location: User-provided location string (optional)

    Returns:
        Dict with location context fields:
        - location: Human-readable location
        - nearby_landmarks: Comma-separated list of nearby POIs
    """
    context: dict[str, str] = {}

    # Start with user-provided location
    if user_location:
        context["location"] = user_location

    # If we have GPS coordinates, enhance with geocoding
    if lat is not None and lon is not None:
        # Reverse geocode for address
        geo = reverse_geocode(lat, lon)
        if geo:
            address = geo.get("address", {})

            # Build location string if not provided by user
            if "location" not in context:
                parts = []
                for key in ["road", "village", "town", "city", "municipality", "county", "state", "country"]:
                    if key in address:
                        parts.append(address[key])
                        if len(parts) >= 3:
                            break
                if parts:
                    context["location"] = ", ".join(parts)

            # Check if the location itself is a POI
            name_details = geo.get("namedetails", {})
            extra_tags = geo.get("extratags", {})

            if name_details.get("name"):
                poi_name = name_details["name"]
                # Add as landmark if it's not just a road name
                if geo.get("type") not in ["road", "residential", "path", "track"]:
                    context["nearby_landmarks"] = poi_name

        # Find nearby POIs
        pois = find_nearby_pois(lat, lon, radius_meters=1000)
        if pois:
            # Filter to most relevant POIs (exclude generic types)
            relevant_pois = [
                p for p in pois
                if p["type"] not in ["yes", "residential", "road"]
                and p["name"]  # Has a name
            ]

            if relevant_pois:
                # Format POI names with types for context
                poi_strings = []
                for poi in relevant_pois[:5]:  # Top 5
                    poi_type = poi["type"].replace("_", " ")
                    if poi_type in poi["name"].lower():
                        poi_strings.append(poi["name"])
                    else:
                        poi_strings.append(f"{poi['name']} ({poi_type})")

                existing = context.get("nearby_landmarks", "")
                if existing:
                    poi_strings.insert(0, existing)
                context["nearby_landmarks"] = ", ".join(poi_strings)

    return context


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with Alnes lighthouse coordinates
    lat, lon = 62.4747, 5.6297  # Approximate coordinates for Alnes

    print(f"\nReverse geocoding ({lat}, {lon}):")
    geo = reverse_geocode(lat, lon)
    if geo:
        print(f"  Display name: {geo.get('display_name')}")
        print(f"  Type: {geo.get('type')}")
        print(f"  Name details: {geo.get('namedetails')}")

    print(f"\nFinding nearby POIs:")
    pois = find_nearby_pois(lat, lon, radius_meters=1000)
    for poi in pois:
        print(f"  - {poi['name']} ({poi['type']}) - {poi['distance_m']}m")

    print(f"\nLocation context:")
    context = get_location_context(lat, lon)
    for k, v in context.items():
        print(f"  {k}: {v}")
