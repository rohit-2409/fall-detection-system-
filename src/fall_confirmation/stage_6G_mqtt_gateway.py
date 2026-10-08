"""
Stage 6G MQTT Backend Subscriber / Gateway.
Standalone Fall Detection Project.

Architecture:
  ESP32 + MPU6050 + NEO-6M GPS
    ↓
  Stage 4/5 Fall Detection -> TemporalConfirmationAdapter
    ↓
  CONFIRMED_FALL
    ↓
  Stage 6F MQTT Publisher
    ↓
  Mosquitto MQTT Broker (10.163.182.136:1883)
    ↓ Topic: fall_detection/stage6/fall
  NEW Stage 6G MQTT Gateway (paho-mqtt)
    ↓ Validates JSON & Gating (event == 'CONFIRMED_FALL')
    ↓ Preserves event_id, timestamps, confidence, GPS telemetry
  EXISTING Fall Event Backend REST API (POST http://127.0.0.1:5000/api/fall-events)
    ↓ Idempotent Storage
  SQLite Persistence (fall_events.db)
    ↓
  EXISTING Dashboard UI (GET /api/fall-events)

Key Properties:
  - Reconnect-safe, fault-tolerant paho-mqtt subscriber.
  - Strict gating: Forwards ONLY validated 'CONFIRMED_FALL' events.
  - Preserves original event_id, timestamps, confidence, and location without corruption.
  - Safe location mapping: Preserves valid 3D GPS or explicit null representation (never invents coords).
  - Backend REST API remains single source of truth (no direct DB manipulation).
  - Handles backend or broker downtime gracefully without crashing.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("Stage6G_MqttGateway")

# Default configurations
DEFAULT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
DEFAULT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
DEFAULT_FALL_TOPIC = "fall_detection/stage6/fall"
DEFAULT_CLIENT_ID = "FALL-STAGE6G-BACKEND"
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


class MqttBackendGateway:
    """
    Subscribes to MQTT confirmed fall events and forwards validated payloads
    into the existing REST Backend API.
    """

    def __init__(
        self,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        topic: str = DEFAULT_FALL_TOPIC,
        client_id: str = DEFAULT_CLIENT_ID,
        backend_url: str = DEFAULT_BACKEND_URL,
        http_timeout_seconds: float = 3.0,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.client_id = client_id
        self.backend_url = backend_url.rstrip("/")
        self.http_timeout = http_timeout_seconds

        self._connected = False
        self._lock = threading.Lock()
        self._connected_event = threading.Event()
        self._subscribed_event = threading.Event()

        # Telemetry / diagnostic counters
        self.messages_received = 0
        self.messages_validated = 0
        self.messages_rejected = 0
        self.backend_successes = 0
        self.backend_failures = 0

        self._client: Optional[mqtt.Client] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initializes paho MQTT client with lifecycle callbacks."""
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
        self._client.on_subscribe = self._on_subscribe

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        result_code = rc.value if hasattr(rc, "value") else rc
        if result_code == 0:
            with self._lock:
                self._connected = True
            self._connected_event.set()
            logger.info("Connected to MQTT broker at %s:%d", self.broker_host, self.broker_port)
            # Resubscribe upon reconnect to ensure topic continuity
            self._client.subscribe(self.topic, qos=1)
            logger.info("Subscribed to topic: %s (QoS 1)", self.topic)
        else:
            with self._lock:
                self._connected = False
            logger.warning("MQTT connection failed with code: %s", result_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, rc=None, properties=None) -> None:
        with self._lock:
            self._connected = False
        self._connected_event.clear()
        self._subscribed_event.clear()
        logger.info("Disconnected from MQTT broker (rc=%s)", rc)

    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None) -> None:
        self._subscribed_event.set()
        logger.info("Subscription confirmed for topic '%s' (mid=%s)", self.topic, mid)

    def start(self, timeout: float = 4.0) -> bool:
        """
        Starts the background MQTT network loop and waits for broker connection.
        Returns True if connected and subscribed within timeout.
        """
        if self._connected:
            return True

        try:
            self._connected_event.clear()
            self._subscribed_event.clear()
            self._client.connect(self.broker_host, self.broker_port, keepalive=30)
            self._client.loop_start()

            is_conn = self._connected_event.wait(timeout=timeout)
            if is_conn:
                self._subscribed_event.wait(timeout=timeout)
            return is_conn
        except Exception as e:
            logger.warning("Failed to start MQTT gateway on %s:%d: %s", self.broker_host, self.broker_port, e)
            return False

    def stop(self, timeout: float = 2.0) -> None:
        """Stops background loop and cleanly disconnects."""
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
        self._subscribed_event.clear()

    @property
    def is_connected(self) -> bool:
        """Returns True if currently connected to MQTT broker."""
        with self._lock:
            return self._connected

    @staticmethod
    def validate_mqtt_message(raw_payload: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates incoming raw MQTT payload.
        Enforces:
          1. Well-formed JSON.
          2. Required 'event' field present.
          3. Strict gating: event == 'CONFIRMED_FALL'.
        """
        try:
            data = json.loads(raw_payload)
        except Exception as e:
            return False, f"MALFORMED_JSON: {e}", None

        if not isinstance(data, dict):
            return False, "INVALID_ROOT_TYPE_EXPECTED_OBJECT", None

        event_name = data.get("event")
        if not event_name:
            return False, "MISSING_REQUIRED_EVENT_FIELD", data

        if event_name != "CONFIRMED_FALL":
            return False, f"REJECTED_EVENT_TYPE_{event_name}", data

        return True, "VALID", data

    @staticmethod
    def map_to_backend_payload(mqtt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps validated Stage 6F MQTT message to the existing Backend REST schema.
        Preserves event_id, detected_at, confidence, source, session_id,
        and packages GPS location cleanly in diagnostic_metadata.
        """
        event_id = str(mqtt_data.get("event_id") or f"GATEWAY-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        detected_at = str(mqtt_data.get("detected_at") or datetime.datetime.now(datetime.timezone.utc).isoformat())
        occurred_at = mqtt_data.get("occurred_at") or detected_at
        confidence = float(mqtt_data.get("confidence", 0.0))
        source = str(mqtt_data.get("source", "ESP32_MPU6050"))
        session_id = mqtt_data.get("session_id") or "stage6g-gateway-session"

        # Safely map location schema
        raw_loc = mqtt_data.get("location")
        if raw_loc and isinstance(raw_loc, dict) and raw_loc.get("fix_valid") is True:
            lat = raw_loc.get("latitude")
            lng = raw_loc.get("longitude")
            if lat is not None and lng is not None:
                location_meta = {
                    "fix_valid": True,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lng), 6),
                    "satellites": int(raw_loc.get("satellites", 0)),
                    "hdop": round(float(raw_loc["hdop"]), 2) if raw_loc.get("hdop") is not None else None,
                }
            else:
                location_meta = {
                    "fix_valid": False,
                    "latitude": None,
                    "longitude": None,
                    "satellites": 0,
                    "hdop": None,
                }
        else:
            location_meta = {
                "fix_valid": False,
                "latitude": None,
                "longitude": None,
                "satellites": 0,
                "hdop": None,
            }

        diagnostic_metadata = {
            "gateway_stage": "6G",
            "source_stage": mqtt_data.get("stage", "6F"),
            "device_id": mqtt_data.get("device_id", "ESP32-STAGE6-GPS"),
            "location": location_meta,
            "status": mqtt_data.get("status", "CONFIRMED"),
            "gateway_forwarded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Preserve any upstream diagnostic metadata if provided
        if "diagnostic_metadata" in mqtt_data and isinstance(mqtt_data["diagnostic_metadata"], dict):
            for k, v in mqtt_data["diagnostic_metadata"].items():
                if k not in diagnostic_metadata:
                    diagnostic_metadata[k] = v

        return {
            "event_id": event_id,
            "detected_at": detected_at,
            "occurred_at": occurred_at,
            "confidence": round(confidence, 4),
            "source": source,
            "session_id": session_id,
            "diagnostic_metadata": diagnostic_metadata,
        }

    def forward_to_backend(self, backend_payload: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
        """
        Posts mapped event to backend REST API (POST /api/fall-events) via urllib.
        Returns: (success: bool, response_message: str, http_status_code: Optional[int])
        """
        endpoint = f"{self.backend_url}/api/fall-events"
        data_bytes = json.dumps(backend_payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Stage6G-MqttGateway/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
                status_code = resp.getcode()
                resp_body = resp.read().decode("utf-8")
                # 201: Newly Created, 200: Idempotent Existing Record
                if status_code in (200, 201):
                    msg = "STORED" if status_code == 201 else "IDEMPOTENT_EXISTING"
                    return True, msg, status_code
                return False, f"UNEXPECTED_STATUS_{status_code}: {resp_body}", status_code

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return False, f"HTTP_{e.code}: {err_body}", e.code
        except urllib.error.URLError as e:
            return False, f"CONNECTION_ERROR: {e.reason}", None
        except Exception as e:
            return False, f"ERROR: {e}", None

    def process_message(self, raw_payload: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Processes single MQTT message end-to-end:
          1. Validates JSON & Gating
          2. Maps to Backend schema
          3. Forwards via HTTP POST
        """
        with self._lock:
            self.messages_received += 1

        is_valid, reason, mqtt_data = self.validate_mqtt_message(raw_payload)
        if not is_valid:
            with self._lock:
                self.messages_rejected += 1
            logger.warning("Rejected MQTT message: %s", reason)
            return False, f"VALIDATION_REJECTED: {reason}", None

        with self._lock:
            self.messages_validated += 1

        backend_payload = self.map_to_backend_payload(mqtt_data)
        event_id = backend_payload["event_id"]

        success, forward_msg, status_code = self.forward_to_backend(backend_payload)
        if success:
            with self._lock:
                self.backend_successes += 1
            logger.info("Successfully forwarded event '%s' to backend (%s, HTTP %s)", event_id, forward_msg, status_code)
            return True, forward_msg, backend_payload
        else:
            with self._lock:
                self.backend_failures += 1
            logger.warning("Failed to forward event '%s' to backend: %s (Status: %s)", event_id, forward_msg, status_code)
            return False, forward_msg, backend_payload

    def _on_message(self, client, userdata, message) -> None:
        """Invoked asynchronously whenever a message arrives on subscribed topic."""
        try:
            raw_payload = message.payload.decode("utf-8", errors="replace")
            logger.debug("Received MQTT message on %s: %s", message.topic, raw_payload[:120])
            self.process_message(raw_payload)
        except Exception as e:
            logger.error("Unexpected exception processing message: %s", e)


def run_gateway_service(
    broker_host: str = DEFAULT_BROKER_HOST,
    broker_port: int = DEFAULT_BROKER_PORT,
    topic: str = DEFAULT_FALL_TOPIC,
    client_id: str = DEFAULT_CLIENT_ID,
    backend_url: str = DEFAULT_BACKEND_URL,
) -> None:
    """Standalone runner for the Stage 6G MQTT Gateway process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("STAGE 6G: MQTT BACKEND SUBSCRIBER / GATEWAY")
    print("=" * 70)
    print(f"Broker:      {broker_host}:{broker_port}")
    print(f"Topic:       {topic}")
    print(f"Client ID:   {client_id}")
    print(f"Backend URL: {backend_url}")
    print("-" * 70)

    gateway = MqttBackendGateway(
        broker_host=broker_host,
        broker_port=broker_port,
        topic=topic,
        client_id=client_id,
        backend_url=backend_url,
    )

    connected = gateway.start(timeout=5.0)
    if not connected:
        print(f"[WARNING] Could not connect to MQTT broker {broker_host}:{broker_port}. Gateway will retry in background.")
    else:
        print(f"[READY] Gateway active and listening on topic '{topic}'. Forwarding to '{backend_url}'.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Gateway stopping...")
    finally:
        gateway.stop()
        print("[INFO] Gateway stopped cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6G MQTT Backend Subscriber / Gateway")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER_HOST, help="MQTT Broker Host")
    parser.add_argument("--port", type=int, default=DEFAULT_BROKER_PORT, help="MQTT Broker Port")
    parser.add_argument("--topic", type=str, default=DEFAULT_FALL_TOPIC, help="MQTT Topic")
    parser.add_argument("--client-id", type=str, default=DEFAULT_CLIENT_ID, help="MQTT Client ID")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND_URL, help="Backend REST API URL")

    args = parser.parse_args()
    run_gateway_service(
        broker_host=args.broker,
        broker_port=args.port,
        topic=args.topic,
        client_id=args.client_id,
        backend_url=args.backend,
    )
