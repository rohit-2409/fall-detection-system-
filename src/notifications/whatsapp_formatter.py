"""
Stage 6I Deterministic WhatsApp Message Formatter.
Standalone Fall Detection Project.

Formats verified fall emergency notifications for caregiver WhatsApp delivery.
Adheres strictly to GPS safety gating rules:
- Only include coordinates if fix_valid is True and coordinates are finite, in bounds, and non-zero.
- Never output 0,0, NaN, Inf, or browser/estimated coordinates.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

from src.fall_event_backend.models import FallEvent


def validate_gps_coordinates(lat_val: Any, lon_val: Any) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Validates latitude and longitude strictly according to Stage 6I rules:
    - Must be a real finite numeric value (not NaN, not Inf).
    - Latitude must be within [-90.0, 90.0].
    - Longitude must be within [-180.0, 180.0].
    - Must NOT be (0.0, 0.0).
    
    Returns:
        Tuple[bool, Optional[float], Optional[float]]: (is_valid, float_lat, float_lon)
    """
    if lat_val is None or lon_val is None:
        return False, None, None

    try:
        lat = float(lat_val)
        lon = float(lon_val)
    except (ValueError, TypeError):
        return False, None, None

    if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
        return False, None, None

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return False, None, None

    # Strict rejection of 0,0 (equator/prime meridian default GPS reset value)
    if abs(lat) < 1e-7 and abs(lon) < 1e-7:
        return False, None, None

    return True, lat, lon


def format_fall_message(event_or_data: Union[FallEvent, Dict[str, Any]]) -> str:
    """
    Deterministic message formatter for confirmed fall notifications.
    
    Produces one of three mutually exclusive formats:
    1. Valid GPS Fix:
       Includes GPS Status: FIX VALID, coordinates, satellites, HDOP, and Google Maps URL.
    2. No Valid Fix:
       Includes GPS Status: NO VALID FIX and Coordinates Not Available.
    3. Missing Location:
       Includes GPS Status: LOCATION UNAVAILABLE and Not available.
    """
    if isinstance(event_or_data, FallEvent):
        event_id = event_or_data.event_id
        detected_at = event_or_data.detected_at
        confidence = event_or_data.confidence
        source = event_or_data.source or "ESP32_MPU6050"
        meta = event_or_data.diagnostic_metadata or {}
    elif isinstance(event_or_data, dict):
        event_id = event_or_data.get("event_id", "UNKNOWN")
        detected_at = event_or_data.get("detected_at", "UNKNOWN")
        confidence = event_or_data.get("confidence")
        source = event_or_data.get("source", "ESP32_MPU6050")
        meta = event_or_data.get("diagnostic_metadata") or {}
    else:
        raise TypeError("event_or_data must be a FallEvent or dict")

    # Format confidence percentage
    if confidence is None:
        conf_str = "N/A"
    elif isinstance(confidence, (int, float)):
        if 0.0 <= confidence <= 1.0:
            conf_str = f"{confidence * 100:.1f}%"
        else:
            conf_str = f"{float(confidence):.1f}%"
    else:
        conf_str = str(confidence)

    base_header = (
        f"EMERGENCY: Confirmed Fall Detected\n\n"
        f"Event ID: {event_id}\n"
        f"Time: {detected_at}\n"
        f"Confidence: {conf_str}\n"
        f"Source: {source}\n\n"
    )

    location_info = meta.get("location") if isinstance(meta, dict) else None

    # Case 3: Missing location
    if location_info is None or not isinstance(location_info, dict):
        return (
            f"{base_header}"
            f"GPS Status: LOCATION UNAVAILABLE\n"
            f"Location: Not available"
        )

    # Inspect GPS fix
    fix_valid = bool(location_info.get("fix_valid", False))
    raw_lat = location_info.get("latitude")
    raw_lon = location_info.get("longitude")

    is_valid_coords, lat, lon = validate_gps_coordinates(raw_lat, raw_lon)

    # Case 1: Valid GPS fix with verified non-zero bounded coordinates
    if fix_valid and is_valid_coords:
        satellites = location_info.get("satellites", "N/A")
        hdop = location_info.get("hdop", "N/A")
        # Format clean lat/lon representation
        lat_str = f"{lat:.6f}".rstrip("0").rstrip(".") if isinstance(lat, float) else str(lat)
        lon_str = f"{lon:.6f}".rstrip("0").rstrip(".") if isinstance(lon, float) else str(lon)
        
        # Ensure at least 4 decimal places if originally precise
        if "." in str(raw_lat):
            lat_str = str(raw_lat)
        if "." in str(raw_lon):
            lon_str = str(raw_lon)

        return (
            f"{base_header}"
            f"GPS Status: FIX VALID\n"
            f"Location: {lat_str}, {lon_str}\n"
            f"Satellites: {satellites}\n"
            f"HDOP: {hdop}\n\n"
            f"Google Maps:\n"
            f"https://www.google.com/maps?q={lat_str},{lon_str}"
        )

    # Case 2: No valid GPS fix or coordinates failed validation
    return (
        f"{base_header}"
        f"GPS Status: NO VALID FIX\n"
        f"Location: Coordinates Not Available"
    )
