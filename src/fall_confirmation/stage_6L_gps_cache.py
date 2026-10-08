"""
Stage 6L Additive GPS Cache & Telemetry Subscriber.
Standalone Fall Detection Project.

Responsibilities:
  1. Subscribes to the existing GPS MQTT topic ('fall_detection/stage6/gps').
  2. Validates incoming GPS payloads (schema, lat/lon bounds, fix validity).
  3. Maintains the latest valid fix in thread-safe memory.
  4. Tracks freshness and detects stale GPS fixes.
  5. Provides clean query API: get_latest_location() -> Dict[str, Any].
  6. Supports direct serial/mock injection: update_location(...) for offline tests.
  7. Strict safety: READ-ONLY in-memory cache; ZERO database mutations.
"""

import datetime
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

logger = logging.getLogger("Stage6L_GpsCache")

DEFAULT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
DEFAULT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
DEFAULT_GPS_TOPIC = os.environ.get("MQTT_GPS_TOPIC", "fall_detection/stage6/gps")
DEFAULT_CLIENT_ID = "FALL-STAGE6L-GPS-CACHE"
DEFAULT_MAX_AGE_SECONDS = 15.0  # GPS data older than 15s is considered stale


class GpsTelemetryCache:
    """
    In-memory thread-safe cache for live GPS telemetry.
    Subscribes to MQTT GPS telemetry and provides fresh coordinates to fall events.
    """

    def __init__(
        self,
        broker_host: Optional[str] = None,
        broker_port: Optional[int] = None,
        topic: Optional[str] = None,
        client_id: Optional[str] = None,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        auto_connect: bool = False,
    ):
        self.broker_host = broker_host or os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
        self.broker_port = broker_port or int(os.environ.get("MQTT_BROKER_PORT", 1883))
        self.topic = topic or os.environ.get("MQTT_GPS_TOPIC", "fall_detection/stage6/gps")
        self.client_id = client_id or DEFAULT_CLIENT_ID
        self.max_age_seconds = max_age_seconds

        self._lock = threading.Lock()
        self._connected = False
        self._connected_event = threading.Event()

        # Cache fields
        self._latitude: Optional[float] = None
        self._longitude: Optional[float] = None
        self._satellites: int = 0
        self._hdop: Optional[float] = None
        self._fix_valid: bool = False
        self._last_update_monotonic: Optional[float] = None
        self._last_update_utc: Optional[str] = None
        self._total_updates: int = 0

        # MQTT client setup
        self._client: Optional[mqtt.Client] = None
        self._init_client()

        if auto_connect:
            self.start()

    def _init_client(self) -> None:
        if PAHO_V2:
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        else:
            self._client = mqtt.Client(client_id=self.client_id)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        code = rc.value if hasattr(rc, "value") else rc
        if code == 0:
            with self._lock:
                self._connected = True
            self._connected_event.set()
            logger.info("GPS Cache connected to MQTT broker at %s:%d", self.broker_host, self.broker_port)
            self._client.subscribe(self.topic, qos=0)
            logger.info("GPS Cache subscribed to topic: %s", self.topic)
        else:
            with self._lock:
                self._connected = False
            logger.warning("GPS Cache failed to connect to broker (rc=%s)", code)

    def _on_disconnect(self, client, userdata, disconnect_flags, rc=None, properties=None) -> None:
        with self._lock:
            self._connected = False
        self._connected_event.clear()
        logger.info("GPS Cache disconnected from broker (rc=%s)", rc)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            raw = msg.payload.decode("utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self.process_payload(data)
        except Exception as e:
            logger.debug("Failed to parse incoming GPS MQTT message: %s", e)

    @staticmethod
    def validate_coordinates(lat: Any, lon: Any) -> bool:
        """Validates that latitude and longitude are within standard geographical bounds."""
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            if not (-90.0 <= lat_f <= 90.0) or not (-180.0 <= lon_f <= 180.0):
                return False
            # Reject default 0.0, 0.0 uninitialized coordinates
            if abs(lat_f) < 1e-6 and abs(lon_f) < 1e-6:
                return False
            return True
        except (ValueError, TypeError):
            return False

    def process_payload(self, data: Dict[str, Any]) -> bool:
        """
        Processes and validates an incoming GPS payload dictionary.
        Returns True if payload contained a valid fix and was cached.
        """
        fix_valid = bool(data.get("fix_valid", False) or data.get("valid", False))
        lat = data.get("latitude")
        lon = data.get("longitude")
        satellites = int(data.get("satellites", 0)) if data.get("satellites") is not None else 0
        hdop = float(data.get("hdop")) if data.get("hdop") is not None else None

        if fix_valid and self.validate_coordinates(lat, lon):
            with self._lock:
                self._latitude = round(float(lat), 6)
                self._longitude = round(float(lon), 6)
                self._satellites = satellites
                self._hdop = round(hdop, 2) if hdop is not None else None
                self._fix_valid = True
                self._last_update_monotonic = time.monotonic()
                self._last_update_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._total_updates += 1
            return True
        else:
            # If payload explicitly declares fix lost/invalid, mark invalid but keep last known if desirable
            if not fix_valid:
                with self._lock:
                    self._fix_valid = False
            return False

    def update_location(
        self,
        latitude: float,
        longitude: float,
        satellites: int = 0,
        hdop: Optional[float] = None,
        valid: bool = True,
    ) -> bool:
        """
        Direct programmatic injection method (used by serial reader or mock tests).
        """
        payload = {
            "valid": valid,
            "fix_valid": valid,
            "latitude": latitude,
            "longitude": longitude,
            "satellites": satellites,
            "hdop": hdop,
        }
        return self.process_payload(payload)

    def get_latest_location(self) -> Dict[str, Any]:
        """
        Retrieves the latest GPS fix formatted for Stage 6F/6G/6H compatibility.
        Detects staleness against max_age_seconds.
        Never invents coordinates.
        """
        with self._lock:
            now_m = time.monotonic()
            is_stale = False
            age_sec = None

            if self._last_update_monotonic is not None:
                age_sec = round(now_m - self._last_update_monotonic, 2)
                if age_sec > self.max_age_seconds:
                    is_stale = True

            if self._fix_valid and not is_stale and self._latitude is not None and self._longitude is not None:
                return {
                    "fix_valid": True,
                    "latitude": self._latitude,
                    "longitude": self._longitude,
                    "satellites": self._satellites,
                    "hdop": self._hdop,
                    "timestamp": self._last_update_utc,
                    "age_seconds": age_sec,
                }

        # Safe explicit null representation when fix is invalid or stale
        return {
            "fix_valid": False,
            "latitude": None,
            "longitude": None,
            "satellites": 0,
            "hdop": None,
            "timestamp": None,
            "age_seconds": None,
        }

    def start(self, timeout: float = 3.0) -> bool:
        """Starts background MQTT network loop."""
        if self._connected:
            return True
        try:
            self._connected_event.clear()
            self._client.connect(self.broker_host, self.broker_port, keepalive=30)
            self._client.loop_start()
            return self._connected_event.wait(timeout=timeout)
        except Exception as e:
            logger.warning("GPS Cache could not start MQTT loop: %s", e)
            return False

    def stop(self) -> None:
        """Stops background MQTT network loop."""
        if self._client is not None:
            try:
                self._client.unsubscribe(self.topic)
            except Exception:
                pass
            try:
                self._client.disconnect()
            except Exception:
                pass
            try:
                self._client.loop_stop()
            except Exception:
                pass
        with self._lock:
            self._connected = False
        self._connected_event.clear()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def total_updates(self) -> int:
        with self._lock:
            return self._total_updates


# Module-level singleton helper
_global_gps_cache: Optional[GpsTelemetryCache] = None
_global_lock = threading.Lock()


def get_global_gps_cache() -> GpsTelemetryCache:
    """Returns or creates the global GpsTelemetryCache instance."""
    global _global_gps_cache
    with _global_lock:
        if _global_gps_cache is None:
            _global_gps_cache = GpsTelemetryCache()
        return _global_gps_cache


def reset_global_gps_cache(instance: Optional[GpsTelemetryCache] = None) -> None:
    """Resets global GPS cache singleton (useful for test teardown)."""
    global _global_gps_cache
    with _global_lock:
        if _global_gps_cache is not None and _global_gps_cache != instance:
            _global_gps_cache.stop()
        _global_gps_cache = instance
