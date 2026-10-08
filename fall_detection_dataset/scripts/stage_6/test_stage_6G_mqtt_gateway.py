"""
Stage 6G Automated Test Suite: MQTT Backend Subscriber / Gateway.
Standalone Fall Detection Project.

Validates:
  1. MQTT broker connection to 10.163.182.136:1883.
  2. Topic subscription to fall_detection/stage6/fall.
  3. Valid CONFIRMED_FALL event reception and end-to-end processing.
  4. JSON validation (well-formed accepted, malformed safely rejected).
  5. Event gating (CONFIRMED_FALL forwarded, non-confirmed rejected).
  6. Backend payload mapping and schema compliance.
  7. Event ID preservation (incoming event_id kept exactly intact).
  8. Location preservation (valid 3D GPS vs safe explicit null representation).
  9. Duplicate / idempotency handling (same event_id handled idempotently).
 10. Fault tolerance when backend REST API is offline (zero crashes).
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest
from typing import List, Dict, Any

import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_6G_mqtt_gateway import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_FALL_TOPIC,
    MqttBackendGateway,
)


class MockBackendServer:
    """Lightweight mock HTTP server simulating the Stage 5A Fall Event REST API."""

    def __init__(self, port: int = 58200):
        self.port = port
        self.recorded_events: List[Dict[str, Any]] = []
        self.seen_event_ids = set()
        self.response_status = 201
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        handler_self = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress console clutter during tests

            def do_POST(self):
                if self.path == "/api/fall-events":
                    content_len = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_len).decode("utf-8")
                    data = json.loads(body)
                    handler_self.recorded_events.append(data)

                    event_id = data.get("event_id")
                    is_new = event_id not in handler_self.seen_event_ids
                    handler_self.seen_event_ids.add(event_id)

                    status_code = 201 if is_new else 200

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {
                        "status": "success",
                        "is_new": is_new,
                        "event": data,
                    }
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

        # Allow immediate socket reuse
        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


class TestStage6GMqttGateway(unittest.TestCase):
    """Test suite for Stage 6G MQTT Backend Subscriber / Gateway."""

    @classmethod
    def setUpClass(cls):
        cls.mock_backend = MockBackendServer(port=58201)
        cls.mock_backend.start()

    @classmethod
    def tearDownClass(cls):
        cls.mock_backend.stop()

    def setUp(self):
        self.gateway = MqttBackendGateway(
            broker_host=DEFAULT_BROKER_HOST,
            broker_port=DEFAULT_BROKER_PORT,
            topic=DEFAULT_FALL_TOPIC,
            client_id="TEST-STAGE6G-GW",
            backend_url=f"http://127.0.0.1:{self.mock_backend.port}",
        )

    def tearDown(self):
        self.gateway.stop()

    def _publish_raw(self, topic: str, payload_str: str, qos: int = 1):
        """Helper to publish raw payload to MQTT broker."""
        if PAHO_V2:
            pub = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id="TEST-STAGE6G-HELPER-PUB")
        else:
            pub = mqtt.Client(client_id="TEST-STAGE6G-HELPER-PUB")
        pub.connect(DEFAULT_BROKER_HOST, DEFAULT_BROKER_PORT, keepalive=10)
        pub.loop_start()
        time.sleep(0.1)
        pub.publish(topic, payload_str, qos=qos)
        time.sleep(0.1)
        pub.loop_stop()
        pub.disconnect()

    def test_01_mqtt_connection(self):
        """TEST 1: Verifies gateway connects to Mosquitto broker."""
        connected = self.gateway.start(timeout=4.0)
        self.assertTrue(connected, f"Gateway failed to connect to {DEFAULT_BROKER_HOST}:{DEFAULT_BROKER_PORT}")
        self.assertTrue(self.gateway.is_connected)

        self.gateway.stop()
        time.sleep(0.1)
        self.assertFalse(self.gateway.is_connected)

    def test_02_topic_subscription(self):
        """TEST 2: Verifies subscription to fall_detection/stage6/fall."""
        self.assertEqual(self.gateway.topic, "fall_detection/stage6/fall")
        self.assertEqual(DEFAULT_FALL_TOPIC, "fall_detection/stage6/fall")
        started = self.gateway.start(timeout=4.0)
        self.assertTrue(started)
        self.assertTrue(self.gateway._subscribed_event.is_set())

    def test_03_valid_confirmed_fall_reception(self):
        """TEST 3: Verifies end-to-end receipt of CONFIRMED_FALL event via MQTT."""
        self.gateway.start(timeout=4.0)
        time.sleep(0.2)

        unique_id = f"STAGE6G-TEST-{int(time.time()*1000)}"
        test_payload = {
            "event": "CONFIRMED_FALL",
            "stage": "6F",
            "event_id": unique_id,
            "detected_at": "2026-10-06T15:30:20.123Z",
            "confidence": 0.9412,
            "source": "ESP32_MPU6050",
            "session_id": "stage6g-test-session",
            "device_id": "ESP32-STAGE6-GPS",
            "location": {
                "fix_valid": True,
                "latitude": 22.728439,
                "longitude": 75.869204,
                "satellites": 10,
                "hdop": 0.79,
            },
            "status": "SIMULATED_TEST_EVENT",
        }

        start_count = len(self.mock_backend.recorded_events)
        self._publish_raw(DEFAULT_FALL_TOPIC, json.dumps(test_payload))

        # Wait for gateway to receive and forward to backend
        time.sleep(1.0)
        self.assertGreater(len(self.mock_backend.recorded_events), start_count)
        last_event = self.mock_backend.recorded_events[-1]
        self.assertEqual(last_event["event_id"], unique_id)
        self.assertEqual(last_event["confidence"], 0.9412)

    def test_04_json_validation(self):
        """TEST 4: Verifies valid JSON is accepted and malformed JSON safely rejected."""
        valid_raw = json.dumps({"event": "CONFIRMED_FALL", "event_id": "EVT-1"})
        is_val, reason, data = self.gateway.validate_mqtt_message(valid_raw)
        self.assertTrue(is_val)
        self.assertEqual(reason, "VALID")

        malformed_raw = "{'bad_json': true,,}"
        is_val, reason, data = self.gateway.validate_mqtt_message(malformed_raw)
        self.assertFalse(is_val)
        self.assertIn("MALFORMED_JSON", reason)

    def test_05_event_gating(self):
        """TEST 5: Verifies CONFIRMED_FALL forwarded while UNCONFIRMED_FALL rejected."""
        # A: Confirmed fall -> valid
        is_val, _, _ = self.gateway.validate_mqtt_message(json.dumps({"event": "CONFIRMED_FALL"}))
        self.assertTrue(is_val)

        # B: Unconfirmed fall -> rejected
        is_val, reason, _ = self.gateway.validate_mqtt_message(json.dumps({"event": "UNCONFIRMED_FALL"}))
        self.assertFalse(is_val)
        self.assertIn("REJECTED_EVENT_TYPE_UNCONFIRMED_FALL", reason)

        # C: Missing event field -> rejected
        is_val, reason, _ = self.gateway.validate_mqtt_message(json.dumps({"some_data": 123}))
        self.assertFalse(is_val)
        self.assertEqual(reason, "MISSING_REQUIRED_EVENT_FIELD")

    def test_06_backend_mapping(self):
        """TEST 6: Verifies correct mapping from MQTT payload to backend request schema."""
        mqtt_event = {
            "event": "CONFIRMED_FALL",
            "stage": "6F",
            "event_id": "STAGE6G-MAP-001",
            "detected_at": "2026-10-06T15:30:20.123Z",
            "occurred_at": "2026-10-06T15:30:19.000Z",
            "confidence": 0.9567,
            "source": "ESP32_MPU6050",
            "session_id": "session-xyz",
            "device_id": "ESP32-STAGE6-GPS",
            "location": {
                "fix_valid": True,
                "latitude": 22.728439,
                "longitude": 75.869204,
                "satellites": 10,
                "hdop": 0.79,
            },
        }

        mapped = self.gateway.map_to_backend_payload(mqtt_event)

        self.assertEqual(mapped["event_id"], "STAGE6G-MAP-001")
        self.assertEqual(mapped["detected_at"], "2026-10-06T15:30:20.123Z")
        self.assertEqual(mapped["occurred_at"], "2026-10-06T15:30:19.000Z")
        self.assertEqual(mapped["confidence"], 0.9567)
        self.assertEqual(mapped["source"], "ESP32_MPU6050")
        self.assertEqual(mapped["session_id"], "session-xyz")
        self.assertIn("diagnostic_metadata", mapped)
        self.assertEqual(mapped["diagnostic_metadata"]["device_id"], "ESP32-STAGE6-GPS")

    def test_07_event_id_preservation(self):
        """TEST 7: Verifies the gateway preserves incoming event_id exactly without generating a new one."""
        exact_id = "LIVE-20261006-PRESERVE-EXACT-ID-999"
        mqtt_event = {
            "event": "CONFIRMED_FALL",
            "event_id": exact_id,
            "detected_at": "2026-10-06T15:30:20.123Z",
            "confidence": 0.92,
        }
        mapped = self.gateway.map_to_backend_payload(mqtt_event)
        self.assertEqual(mapped["event_id"], exact_id)

    def test_08_location_preservation(self):
        """TEST 8: Verifies valid GPS mapped cleanly and no-fix does not create fake coords."""
        # Case 1: Valid 3D Fix
        valid_gps_event = {
            "event": "CONFIRMED_FALL",
            "event_id": "EVT-LOC-VALID",
            "location": {
                "fix_valid": True,
                "latitude": 22.728439,
                "longitude": 75.869204,
                "satellites": 10,
                "hdop": 0.79,
            },
        }
        mapped_valid = self.gateway.map_to_backend_payload(valid_gps_event)
        loc_valid = mapped_valid["diagnostic_metadata"]["location"]
        self.assertTrue(loc_valid["fix_valid"])
        self.assertEqual(loc_valid["latitude"], 22.728439)
        self.assertEqual(loc_valid["longitude"], 75.869204)
        self.assertEqual(loc_valid["satellites"], 10)
        self.assertEqual(loc_valid["hdop"], 0.79)

        # Case 2: No Fix / Invalid Fix
        no_fix_event = {
            "event": "CONFIRMED_FALL",
            "event_id": "EVT-LOC-NOFIX",
            "location": {
                "fix_valid": False,
                "latitude": 0.0,
                "longitude": 0.0,
                "satellites": 0,
                "hdop": None,
            },
        }
        mapped_no_fix = self.gateway.map_to_backend_payload(no_fix_event)
        loc_no_fix = mapped_no_fix["diagnostic_metadata"]["location"]
        self.assertFalse(loc_no_fix["fix_valid"])
        self.assertIsNone(loc_no_fix["latitude"])  # Coordinates must be None when fix_valid=False
        self.assertIsNone(loc_no_fix["longitude"])
        self.assertEqual(loc_no_fix["satellites"], 0)
        self.assertIsNone(loc_no_fix["hdop"])

    def test_09_duplicate_idempotency(self):
        """TEST 9: Verifies sending same event_id twice returns 200 idempotent and prevents duplicates."""
        dup_id = f"STAGE6G-DUP-{int(time.time()*1000)}"
        mqtt_event = {
            "event": "CONFIRMED_FALL",
            "event_id": dup_id,
            "detected_at": "2026-10-06T15:30:20.123Z",
            "confidence": 0.9100,
        }
        mapped = self.gateway.map_to_backend_payload(mqtt_event)

        # First post -> 201 STORED
        s1, m1, code1 = self.gateway.forward_to_backend(mapped)
        self.assertTrue(s1)
        self.assertEqual(code1, 201)
        self.assertEqual(m1, "STORED")

        # Second post -> 200 IDEMPOTENT_EXISTING
        s2, m2, code2 = self.gateway.forward_to_backend(mapped)
        self.assertTrue(s2)
        self.assertEqual(code2, 200)
        self.assertEqual(m2, "IDEMPOTENT_EXISTING")

    def test_10_backend_unavailable_handling(self):
        """TEST 10: Verifies gateway handles offline backend gracefully without crashing."""
        offline_gateway = MqttBackendGateway(
            broker_host=DEFAULT_BROKER_HOST,
            broker_port=DEFAULT_BROKER_PORT,
            topic=DEFAULT_FALL_TOPIC,
            backend_url="http://127.0.0.1:59999",  # Offline port
        )

        test_payload = json.dumps({
            "event": "CONFIRMED_FALL",
            "event_id": "STAGE6G-OFFLINE-TEST",
            "detected_at": "2026-10-06T15:30:20.123Z",
            "confidence": 0.9000,
        })

        # process_message must return gracefully with failure indication, zero unhandled exceptions
        success, msg, payload = offline_gateway.process_message(test_payload)
        self.assertFalse(success)
        self.assertIn("CONNECTION_ERROR", msg)
        self.assertIsNotNone(payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
