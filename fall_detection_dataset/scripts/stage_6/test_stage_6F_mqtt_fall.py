"""
Stage 6F Automated Test Suite: MQTT Confirmed Fall Event Publisher.
Standalone Fall Detection Project.

Validates:
  1. MQTT broker connection to 10.163.182.136:1883 (with local fallback).
  2. Publication topic verification: fall_detection/stage6/fall.
  3. Confirmed fall payload structure, schema, and field verification.
  4. JSON serialization and deserialization integrity.
  5. In-memory duplicate protection (duplicate event_id suppression).
  6. Fault tolerance when MQTT broker is unavailable (zero crashes).
  7. Gating protection (rejection of unconfirmed windows / raw positives).
  8. Safe GPS location schema representation (valid fix vs safe null representation).
"""

import json
import os
import sys
import threading
import time
import unittest

import paho.mqtt.client as mqtt

try:
    from paho.mqtt.enums import CallbackAPIVersion
    PAHO_V2 = True
except ImportError:
    PAHO_V2 = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_6F_mqtt_fall_publisher import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_FALL_TOPIC,
    MqttFallPublisher,
    publish_stage_5f_event,
)


class TestStage6FMqttFallPublisher(unittest.TestCase):
    """Test suite for Stage 6F confirmed fall MQTT publisher."""

    def setUp(self):
        self.publisher = MqttFallPublisher(
            broker_host=DEFAULT_BROKER_HOST,
            broker_port=DEFAULT_BROKER_PORT,
            topic=DEFAULT_FALL_TOPIC,
            client_id="TEST-STAGE6F-PUB",
        )

    def tearDown(self):
        self.publisher.disconnect()

    def test_01_broker_connection(self):
        """TEST 1: Verifies the Stage 6F publisher can connect to the broker."""
        connected = self.publisher.connect(timeout=4.0)
        self.assertTrue(connected, f"Failed to connect to broker at {DEFAULT_BROKER_HOST}:{DEFAULT_BROKER_PORT}")
        self.assertTrue(self.publisher.is_connected)

        self.publisher.disconnect()
        time.sleep(0.1)
        self.assertFalse(self.publisher.is_connected)

    def test_02_correct_topic(self):
        """TEST 2: Verifies publication uses the required topic: fall_detection/stage6/fall."""
        self.assertEqual(self.publisher.topic, "fall_detection/stage6/fall")
        self.assertEqual(DEFAULT_FALL_TOPIC, "fall_detection/stage6/fall")

    def test_03_confirmed_fall_payload_and_json_validity(self):
        """TEST 3 & 4: Verifies structured payload schema, JSON validity, and field presence."""
        received_messages = []
        msg_event = threading.Event()

        # Dedicated test subscriber
        if PAHO_V2:
            sub = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id="TEST-STAGE6F-SUB-03")
        else:
            sub = mqtt.Client(client_id="TEST-STAGE6F-SUB-03")

        def on_message(client, userdata, message):
            try:
                data = json.loads(message.payload.decode("utf-8"))
                received_messages.append(data)
            except Exception as e:
                received_messages.append({"error": str(e)})
            msg_event.set()

        sub.on_message = on_message
        sub.connect(DEFAULT_BROKER_HOST, DEFAULT_BROKER_PORT, keepalive=10)
        sub.subscribe(DEFAULT_FALL_TOPIC, qos=1)
        sub.loop_start()
        time.sleep(0.2)

        test_event = {
            "event": "CONFIRMED_FALL",
            "status": "SIMULATED_TEST_EVENT",
            "event_id": f"STAGE6F-TEST-{int(time.time()*1000)}",
            "detected_at": "2026-10-06T15:30:20.123Z",
            "confidence": 0.9412,
            "source": "ESP32_MPU6050",
            "session_id": "stage6f-test-session",
            "confirmed_fall": True,
        }

        test_location = {
            "fix_valid": True,
            "latitude": 22.728439,
            "longitude": 75.869204,
            "satellites": 10,
            "hdop": 0.79,
        }

        self.publisher.connect(timeout=3.0)
        success, status, published_payload = self.publisher.publish_confirmed_fall(
            event_data=test_event,
            location=test_location,
        )

        self.assertTrue(success, f"Publishing failed: {status}")
        self.assertEqual(status, "PUBLISHED")
        self.assertIsNotNone(published_payload)

        # Wait for subscriber to receive
        arrived = msg_event.wait(timeout=4.0)
        sub.loop_stop()
        sub.disconnect()

        self.assertTrue(arrived, "Subscriber did not receive message on topic within 4 seconds")
        self.assertEqual(len(received_messages), 1)

        payload = received_messages[0]
        # Validate required fields
        self.assertEqual(payload.get("event"), "CONFIRMED_FALL")
        self.assertEqual(payload.get("stage"), "6F")
        self.assertEqual(payload.get("event_id"), test_event["event_id"])
        self.assertEqual(payload.get("detected_at"), "2026-10-06T15:30:20.123Z")
        self.assertAlmostEqual(payload.get("confidence"), 0.9412, places=4)
        self.assertEqual(payload.get("source"), "ESP32_MPU6050")
        self.assertEqual(payload.get("session_id"), "stage6f-test-session")
        self.assertEqual(payload.get("device_id"), "ESP32-STAGE6-GPS")
        self.assertEqual(payload.get("status"), "SIMULATED_TEST_EVENT")

        # Location schema validation
        loc = payload.get("location", {})
        self.assertTrue(loc.get("fix_valid"))
        self.assertAlmostEqual(loc.get("latitude"), 22.728439, places=6)
        self.assertAlmostEqual(loc.get("longitude"), 75.869204, places=6)
        self.assertEqual(loc.get("satellites"), 10)
        self.assertAlmostEqual(loc.get("hdop"), 0.79, places=2)

    def test_05_duplicate_protection(self):
        """TEST 5: Verifies sending the same event_id twice suppresses the duplicate."""
        test_event = {
            "event": "CONFIRMED_FALL",
            "status": "SIMULATED_TEST_EVENT",
            "event_id": f"STAGE6F-DEDUP-{int(time.time()*1000)}",
            "confidence": 0.8850,
            "confirmed_fall": True,
        }

        self.publisher.connect(timeout=3.0)
        # First send -> should succeed
        s1, st1, p1 = self.publisher.publish_confirmed_fall(event_data=test_event)
        self.assertTrue(s1, f"First send failed: {st1}")
        self.assertEqual(st1, "PUBLISHED")

        # Second send with identical event_id -> must be suppressed
        s2, st2, p2 = self.publisher.publish_confirmed_fall(event_data=test_event)
        self.assertFalse(s2, "Duplicate send should NOT be published")
        self.assertEqual(st2, "DUPLICATE_IGNORED")
        self.assertIsNone(p2)

    def test_06_broker_unavailable_handling(self):
        """TEST 6: Verifies publisher handles broker unavailability without crashing the process."""
        # Point to closed port
        offline_pub = MqttFallPublisher(
            broker_host="127.0.0.1",
            broker_port=59999,  # Non-existent port
            client_id="TEST-STAGE6F-OFFLINE",
        )

        test_event = {
            "event": "CONFIRMED_FALL",
            "status": "SIMULATED_TEST_EVENT",
            "event_id": "STAGE6F-OFFLINE-TEST",
            "confidence": 0.7500,
            "confirmed_fall": True,
        }

        # Must not raise an exception; must return False gracefully
        success, status, payload = offline_pub.publish_confirmed_fall(event_data=test_event)
        self.assertFalse(success)
        self.assertEqual(status, "BROKER_UNAVAILABLE")
        self.assertIsNotNone(payload)  # Payload was still safely constructed

    def test_07_gating_unconfirmed_event_rejection(self):
        """TEST 7: Verifies non-confirmed events (raw positives, unconfirmed windows) are rejected."""
        unconfirmed_event = {
            "event": "RAW_POSITIVE_WINDOW",
            "confirmed_fall": False,
            "confidence": 0.9500,
        }
        success, status, payload = self.publisher.publish_confirmed_fall(event_data=unconfirmed_event)
        self.assertFalse(success)
        self.assertEqual(status, "REJECTED_UNCONFIRMED_EVENT")
        self.assertIsNone(payload)

    def test_08_location_schema_without_fix(self):
        """TEST 8: Verifies safe location representation when no valid GPS fix is available."""
        test_event = {
            "event": "CONFIRMED_FALL",
            "status": "SIMULATED_TEST_EVENT",
            "event_id": f"STAGE6F-NOFIX-{int(time.time()*1000)}",
            "confidence": 0.9100,
            "confirmed_fall": True,
        }

        # Case A: Location is None
        p_none = self.publisher.build_payload(event_data=test_event, location=None)
        loc_none = p_none["location"]
        self.assertFalse(loc_none["fix_valid"])
        self.assertIsNone(loc_none["latitude"])
        self.assertIsNone(loc_none["longitude"])
        self.assertEqual(loc_none["satellites"], 0)
        self.assertIsNone(loc_none["hdop"])

        # Case B: Location has fix_valid=False
        p_false = self.publisher.build_payload(
            event_data=test_event,
            location={"fix_valid": False, "latitude": 22.728, "longitude": 75.869},
        )
        loc_false = p_false["location"]
        self.assertFalse(loc_false["fix_valid"])
        self.assertIsNone(loc_false["latitude"])  # Stale coords must be suppressed when fix_valid=False
        self.assertIsNone(loc_false["longitude"])
        self.assertEqual(loc_false["satellites"], 0)
        self.assertIsNone(loc_false["hdop"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
