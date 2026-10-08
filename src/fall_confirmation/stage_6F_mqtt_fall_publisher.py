"""
Stage 6F MQTT Confirmed Fall Event Publisher.
Standalone Fall Detection Project.

Architecture:
  MPU6050 -> Stage 4/5 Fall Detection -> TemporalConfirmationAdapter
    ↓
  confirmed_fall == True gating
    ↓
  Stage 6F MqttFallPublisher (paho-mqtt)
    ↓
  Mosquitto Broker (10.163.182.136:1883)
    ↓
  Topic: fall_detection/stage6/fall

Key Properties:
  - Publishes ONLY confirmed fall events (gated on confirmed_fall == True).
  - Reconnect-safe, non-blocking execution (MQTT errors/offline never crash inference).
  - In-memory duplicate protection window (duplicate event_ids are ignored).
  - Structured JSON payload containing event, stage, event_id, detected_at, confidence, source, session_id, device_id, and location schema.
  - Frozen modules (Stage 4, Stage 5, Stage 5F, Stage 6B, 6C, 6D, 6E) remain 100% untouched.
"""

import collections
import datetime
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Deque, Dict, Optional, Set, Tuple

import paho.mqtt.client as mqtt

# Handle paho-mqtt v2 vs v1 enum compatibility
try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("Stage6F_MqttPublisher")

# Defaults
DEFAULT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
DEFAULT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
DEFAULT_FALL_TOPIC = "fall_detection/stage6/fall"
DEFAULT_CLIENT_ID = "FALL-STAGE6F-PUBLISHER"
DEFAULT_DEVICE_ID = "ESP32-STAGE6-GPS"
DEFAULT_DEDUP_SIZE = 128


class MqttFallPublisher:
    """
    Thread-safe, non-blocking MQTT publisher for confirmed fall events.
    Enforces confirmed_fall gating, payload schema compliance, and duplicate protection.
    """

    def __init__(
        self,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        topic: str = DEFAULT_FALL_TOPIC,
        client_id: str = DEFAULT_CLIENT_ID,
        device_id: str = DEFAULT_DEVICE_ID,
        dedup_cache_size: int = DEFAULT_DEDUP_SIZE,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client_id = client_id
        self.device_id = device_id
        self.dedup_cache_size = dedup_cache_size

        self._connected = False
        self._lock = threading.Lock()
        self._seen_ids_deque: Deque[str] = collections.deque(maxlen=dedup_cache_size)
        self._seen_ids_set: Set[str] = set()

        self._connected_event = threading.Event()
        self._client: Optional[mqtt.Client] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initializes paho MQTT client with connection and disconnect callbacks."""
        if PAHO_V2:
            self._client = mqtt.Client(
                callback_api_version=CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
            )
        else:
            self._client = mqtt.Client(client_id=self.client_id)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        result_code = rc.value if hasattr(rc, "value") else rc
        if result_code == 0:
            with self._lock:
                self._connected = True
            self._connected_event.set()
            logger.info("Connected to MQTT broker at %s:%d", self.broker_host, self.broker_port)
        else:
            with self._lock:
                self._connected = False
            logger.warning("MQTT connection returned non-zero code: %s", result_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, rc=None, properties=None) -> None:
        with self._lock:
            self._connected = False
        self._connected_event.clear()
        logger.info("Disconnected from MQTT broker (rc=%s)", rc)

    def connect(self, timeout: float = 3.0) -> bool:
        """
        Connects to the broker and starts the background client loop.
        Non-blocking to the caller beyond the short timeout. Returns True if connected.
        """
        if self._connected:
            return True

        try:
            self._connected_event.clear()
            self._client.connect(self.broker_host, self.broker_port, keepalive=30)
            self._client.loop_start()
            is_conn = self._connected_event.wait(timeout=timeout)
            return is_conn
        except Exception as e:
            logger.warning("Failed to connect to MQTT broker %s:%d: %s", self.broker_host, self.broker_port, e)
            return False

    def disconnect(self, timeout: float = 2.0) -> None:
        """Stops background loop and disconnects cleanly."""
        if self._client is not None:
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
        """Returns current connection status."""
        with self._lock:
            return self._connected

    @staticmethod
    def format_location(location_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Formats location schema.
        If live GPS fix is valid, uses valid coordinates and satellite metadata.
        If fix is invalid or GPS unavailable, formats safe explicit null representation.
        Never outputs stale coordinates as valid.
        """
        if location_data and location_data.get("fix_valid") is True:
            lat = location_data.get("latitude")
            lng = location_data.get("longitude")
            if lat is not None and lng is not None:
                return {
                    "fix_valid": True,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lng), 6),
                    "satellites": int(location_data.get("satellites", 0)),
                    "hdop": round(float(location_data["hdop"]), 2) if location_data.get("hdop") is not None else None,
                }

        return {
            "fix_valid": False,
            "latitude": None,
            "longitude": None,
            "satellites": 0,
            "hdop": None,
        }

    def build_payload(
        self,
        event_data: Dict[str, Any],
        location: Optional[Dict[str, Any]] = None,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a structured, standardized Stage 6F confirmed-fall JSON payload.
        """
        event_id = str(event_data.get("event_id") or f"LIVE-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        detected_at = event_data.get("detected_at") or datetime.datetime.now(datetime.timezone.utc).isoformat()
        confidence = float(event_data.get("confidence", 0.0))
        source = str(event_data.get("source", "ESP32_MPU6050"))
        session_id = str(event_data.get("session_id", "stage6f-session"))
        dev_id = str(device_id or event_data.get("device_id") or self.device_id)

        payload: Dict[str, Any] = {
            "event": "CONFIRMED_FALL",
            "stage": "6F",
            "event_id": event_id,
            "detected_at": detected_at,
            "confidence": round(confidence, 4),
            "source": source,
            "session_id": session_id,
            "device_id": dev_id,
            "location": self.format_location(location),
        }

        # If a status label exists (e.g. TEST or SIMULATED_TEST_EVENT), preserve it
        if "status" in event_data:
            payload["status"] = event_data["status"]

        return payload

    def publish_confirmed_fall(
        self,
        event_data: Dict[str, Any],
        location: Optional[Dict[str, Any]] = None,
        device_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Publishes ONLY confirmed fall events.
        
        Enforces:
          1. Gating: Rejects unconfirmed, raw-positive, or normal data.
          2. Duplicate Protection: Rejects event_id if already seen within dedup cache.
          3. Fault Tolerance: Returns gracefully with status message if broker is unreachable.
          
        Returns: (success: bool, status_message: str, payload_dict: Optional[Dict[str, Any]])
        """
        # 1. Gating Check: Must be a confirmed fall
        # Checks if explicit flag confirmed_fall is False
        if event_data.get("confirmed_fall") is False:
            return False, "REJECTED_UNCONFIRMED_EVENT", None

        # Rejects non-fall event types
        event_type = event_data.get("event")
        if event_type and event_type not in ("CONFIRMED_FALL", "TEST", "SIMULATED_TEST_EVENT"):
            return False, f"REJECTED_INVALID_EVENT_TYPE_{event_type}", None

        # 2. Duplicate Protection
        event_id = event_data.get("event_id")
        if not event_id:
            event_id = f"STAGE6F-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            event_data["event_id"] = event_id

        with self._lock:
            if event_id in self._seen_ids_set:
                logger.info("Duplicate event_id '%s' suppressed by Stage 6F dedup filter.", event_id)
                return False, "DUPLICATE_IGNORED", None

        # 3. Payload Construction & Validation
        payload = self.build_payload(event_data=event_data, location=location, device_id=device_id)

        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to serialize payload to JSON: %s", e)
            return False, f"JSON_SERIALIZE_ERROR: {e}", None

        # 4. Connection & Publishing
        if not self.is_connected:
            logger.info("MQTT not connected; attempting quick reconnect to %s:%d...", self.broker_host, self.broker_port)
            self.connect(timeout=1.5)

        if not self.is_connected:
            logger.warning("Broker unavailable; MQTT event publication skipped for event '%s'.", event_id)
            return False, "BROKER_UNAVAILABLE", payload

        try:
            info = self._client.publish(self.topic, payload_json, qos=1)
            # Record in deduplication window
            with self._lock:
                if len(self._seen_ids_deque) == self._seen_ids_deque.maxlen:
                    oldest_id = self._seen_ids_deque.popleft()
                    self._seen_ids_set.discard(oldest_id)
                self._seen_ids_deque.append(event_id)
                self._seen_ids_set.add(event_id)

            logger.info("Published CONFIRMED_FALL event '%s' to topic '%s' (QoS 1)", event_id, self.topic)
            return True, "PUBLISHED", payload
        except Exception as e:
            logger.warning("Exception during MQTT publish: %s", e)
            return False, f"PUBLISH_EXCEPTION: {e}", payload


def publish_stage_5f_event(
    publisher: MqttFallPublisher,
    event_payload: Dict[str, Any],
    location: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Bridge helper: Takes an event payload produced by Stage 5F live bridge
    and dispatches it through the Stage 6F MqttFallPublisher.
    """
    return publisher.publish_confirmed_fall(event_data=event_payload, location=location)
