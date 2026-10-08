"""
Stage 7B Phase 1 Automated Test Suite: Unified Backend + MQTT Gateway Integration.
Standalone Fall Detection Project.

Validates the Stage 7B Phase 1 integration requirements:
  Test 1: Gateway default broker is 127.0.0.1.
  Test 2: Gateway subscribes to fall_detection/stage6/fall.
  Test 3: A valid CONFIRMED_FALL MQTT payload is converted to an HTTP POST.
  Test 4: GPS metadata survives MQTT -> gateway -> backend mapping.
  Test 5: Invalid MQTT messages do not crash the gateway.
  Test 6: Unified backend lifecycle (Flask + MQTT Gateway in single coordinated process).
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import time
import unittest
import urllib.request
from typing import Any, Dict, List, Optional

import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_full_backend import UnifiedBackendServer
from src.fall_confirmation.stage_6G_mqtt_gateway import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_FALL_TOPIC,
    MqttBackendGateway,
)


class MockBackendReceiver:
    """Lightweight mock HTTP server simulating /api/fall-events endpoint."""

    def __init__(self, port: int = 58215):
        self.port = port
        self.received_payloads: List[Dict[str, Any]] = []
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        receiver_self = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress console clutter

            def do_POST(self):
                if self.path == "/api/fall-events":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length).decode("utf-8")
                    data = json.loads(body)
                    receiver_self.received_payloads.append(data)
                    self.send_response(201)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {"status": "success", "is_new": True, "event": data}
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


class TestStage7BPhase1Integration(unittest.TestCase):
    """Test suite validating Stage 7B Phase 1 Unified Backend & MQTT Gateway."""

    @classmethod
    def setUpClass(cls):
        cls.mock_receiver = MockBackendReceiver(port=58215)
        cls.mock_receiver.start()

    @classmethod
    def tearDownClass(cls):
        cls.mock_receiver.stop()

    def setUp(self):
        self.gateway = MqttBackendGateway(
            broker_host=DEFAULT_BROKER_HOST,
            broker_port=DEFAULT_BROKER_PORT,
            topic=DEFAULT_FALL_TOPIC,
            client_id="TEST-STAGE7B-GW",
            backend_url=f"http://127.0.0.1:{self.mock_receiver.port}",
        )

    def tearDown(self):
        self.gateway.stop()

    def _publish_mqtt(self, topic: str, payload_dict: Dict[str, Any]):
        """Helper to publish dictionary payload to MQTT broker."""
        if PAHO_V2:
            pub = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id="TEST-STAGE7B-PUB")
        else:
            pub = mqtt.Client(client_id="TEST-STAGE7B-PUB")
        pub.connect(DEFAULT_BROKER_HOST, DEFAULT_BROKER_PORT, keepalive=10)
        pub.loop_start()
        time.sleep(0.05)
        pub.publish(topic, json.dumps(payload_dict), qos=1)
        time.sleep(0.1)
        pub.loop_stop()
        pub.disconnect()

    def test_01_gateway_default_broker_is_127_0_0_1(self):
        """TEST 1: Verifies gateway default broker is 127.0.0.1 (not 10.163.182.136)."""
        # Save original env if present
        orig_env = os.environ.get("MQTT_BROKER_HOST")
        try:
            if "MQTT_BROKER_HOST" in os.environ:
                del os.environ["MQTT_BROKER_HOST"]

            # Re-read default
            gw = MqttBackendGateway()
            self.assertEqual(gw.broker_host, "127.0.0.1")
            self.assertEqual(DEFAULT_BROKER_HOST, "127.0.0.1")
            self.assertEqual(gw.broker_port, 1883)
            self.assertNotEqual(gw.broker_host, "10.163.182.136")
        finally:
            if orig_env is not None:
                os.environ["MQTT_BROKER_HOST"] = orig_env

    def test_02_gateway_subscribes_to_fall_detection_stage6_fall(self):
        """TEST 2: Verifies gateway subscribes to fall_detection/stage6/fall."""
        self.assertEqual(self.gateway.topic, "fall_detection/stage6/fall")
        self.assertEqual(DEFAULT_FALL_TOPIC, "fall_detection/stage6/fall")

        started = self.gateway.start(timeout=4.0)
        self.assertTrue(started, "Gateway failed to connect to local broker")
        self.assertTrue(self.gateway.is_connected)
        self.assertTrue(self.gateway._subscribed_event.is_set())

    def test_03_valid_confirmed_fall_mqtt_payload_converted_to_http_post(self):
        """TEST 3: Verifies valid CONFIRMED_FALL MQTT payload is converted to HTTP POST."""
        self.gateway.start(timeout=4.0)
        time.sleep(0.2)

        unique_id = f"STAGE7B-TEST-{int(time.time()*1000)}"
        test_payload = {
            "event": "CONFIRMED_FALL",
            "stage": "6L",
            "event_id": unique_id,
            "detected_at": "2026-10-07T12:00:00.000Z",
            "confidence": 0.8875,
            "source": "ESP32_MPU6050",
            "session_id": "stage7b-session-1",
            "device_id": "ESP32-STAGE6L-UNIFIED",
            "location": {
                "fix_valid": True,
                "latitude": 22.728523,
                "longitude": 75.869183,
                "satellites": 9,
                "hdop": 0.85,
            },
            "status": "CONFIRMED",
        }

        start_count = len(self.mock_receiver.received_payloads)
        self._publish_mqtt(DEFAULT_FALL_TOPIC, test_payload)

        # Wait for delivery
        time.sleep(1.0)
        self.assertGreater(len(self.mock_receiver.received_payloads), start_count)
        last_item = self.mock_receiver.received_payloads[-1]
        self.assertEqual(last_item["event_id"], unique_id)
        self.assertEqual(last_item["confidence"], 0.8875)
        self.assertEqual(last_item["source"], "ESP32_MPU6050")

    def test_04_gps_metadata_survives_mapping(self):
        """TEST 4: Verifies GPS metadata survives MQTT -> gateway -> backend mapping."""
        gps_payload = {
            "event": "CONFIRMED_FALL",
            "event_id": "STAGE7B-GPS-PRESERVE-01",
            "detected_at": "2026-10-07T12:05:00.000Z",
            "confidence": 0.9234,
            "location": {
                "fix_valid": True,
                "latitude": 22.728523,
                "longitude": 75.869183,
                "satellites": 11,
                "hdop": 0.82,
            },
        }

        mapped = self.gateway.map_to_backend_payload(gps_payload)
        self.assertIn("diagnostic_metadata", mapped)
        loc = mapped["diagnostic_metadata"]["location"]

        self.assertTrue(loc["fix_valid"])
        self.assertEqual(loc["latitude"], 22.728523)
        self.assertEqual(loc["longitude"], 75.869183)
        self.assertEqual(loc["satellites"], 11)
        self.assertEqual(loc["hdop"], 0.82)

    def test_05_invalid_mqtt_messages_do_not_crash_gateway(self):
        """TEST 5: Verifies invalid/malformed MQTT messages do not crash the gateway."""
        # 1. Non-JSON string
        is_val, reason, _ = self.gateway.validate_mqtt_message("NOT_JSON_AT_ALL")
        self.assertFalse(is_val)
        self.assertIn("MALFORMED_JSON", reason)

        # 2. JSON list instead of object
        is_val, reason, _ = self.gateway.validate_mqtt_message(json.dumps([1, 2, 3]))
        self.assertFalse(is_val)
        self.assertEqual(reason, "INVALID_ROOT_TYPE_EXPECTED_OBJECT")

        # 3. Missing event key
        is_val, reason, _ = self.gateway.validate_mqtt_message(json.dumps({"data": "xyz"}))
        self.assertFalse(is_val)
        self.assertEqual(reason, "MISSING_REQUIRED_EVENT_FIELD")

        # 4. Non-confirmed event type
        is_val, reason, _ = self.gateway.validate_mqtt_message(json.dumps({"event": "UNCONFIRMED_FALL"}))
        self.assertFalse(is_val)
        self.assertIn("REJECTED_EVENT_TYPE", reason)

        # Gateway process_message handles malformed message gracefully
        success, msg, _ = self.gateway.process_message("INVALID_PAYLOAD{{{")
        self.assertFalse(success)
        self.assertIn("VALIDATION_REJECTED", msg)

    def test_06_unified_backend_lifecycle(self):
        """TEST 6: Verifies UnifiedBackendServer starts Flask, connects gateway, and stops cleanly."""
        server = UnifiedBackendServer(
            host="127.0.0.1",
            port=58216,  # Ephemeral port
            broker_host=DEFAULT_BROKER_HOST,
            broker_port=DEFAULT_BROKER_PORT,
            topic=DEFAULT_FALL_TOPIC,
            enable_gateway=True,
        )

        try:
            started = server.start(timeout=5.0)
            self.assertTrue(started, "UnifiedBackendServer failed to start")

            # Check Flask health
            health_url = f"http://127.0.0.1:58216/api/health"
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data.get("status"), "healthy")

            # Check gateway is active
            self.assertIsNotNone(server.gateway)
            self.assertTrue(server.gateway.is_connected)

        finally:
            server.stop()
            time.sleep(0.2)
            self.assertFalse(server._is_running)


if __name__ == "__main__":
    unittest.main(verbosity=2)
