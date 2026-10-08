"""
Stage 6B Local MQTT Broker Automated Test Suite.
Standalone Fall Detection Project.

Tests local Mosquitto MQTT broker connectivity (127.0.0.1:1883),
publication, subscription, JSON serialization/deserialization,
and clean disconnect lifecycle.
"""

import json
import threading
import time
import unittest

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1883
TOPIC_BASE = "medicare/fall_detection/stage6/test"


class TestStage6BMQTT(unittest.TestCase):
    """Verifies local Mosquitto MQTT broker operation using paho-mqtt."""

    def test_01_broker_connection(self):
        """Verifies Python client can connect to 127.0.0.1:1883 and disconnect cleanly."""
        connected_event = threading.Event()
        disconnected_event = threading.Event()
        connect_rc = [-1]

        def on_connect(client, userdata, flags, rc, properties=None):
            connect_rc[0] = rc.value if hasattr(rc, "value") else rc
            connected_event.set()

        def on_disconnect(client, userdata, disconnect_flags, rc, properties=None):
            disconnected_event.set()

        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_conn"
        )
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        client.loop_start()

        # Wait for connection confirmation
        is_connected = connected_event.wait(timeout=3.0)
        self.assertTrue(is_connected, "Failed to connect to 127.0.0.1:1883 within 3 seconds")
        self.assertEqual(connect_rc[0], 0, f"Connect returned non-zero result code: {connect_rc[0]}")

        # Disconnect cleanly
        client.disconnect()
        is_disconnected = disconnected_event.wait(timeout=3.0)
        client.loop_stop()
        self.assertTrue(is_disconnected, "Failed to disconnect cleanly within 3 seconds")

    def test_02_publish_subscribe_string(self):
        """Verifies publish and subscribe of a plain string message."""
        test_topic = f"{TOPIC_BASE}/string"
        sent_msg = "hello-stage6b-python"
        received_msgs = []
        msg_event = threading.Event()

        def on_message(client, userdata, message):
            payload = message.payload.decode("utf-8")
            received_msgs.append(payload)
            msg_event.set()

        sub_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_sub_str"
        )
        sub_client.on_message = on_message
        sub_client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        sub_client.subscribe(test_topic)
        sub_client.loop_start()

        # Small delay to ensure subscription is registered on the broker
        time.sleep(0.1)

        pub_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_pub_str"
        )
        pub_client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        pub_client.loop_start()

        pub_client.publish(test_topic, payload=sent_msg, qos=1)

        got_msg = msg_event.wait(timeout=3.0)
        self.assertTrue(got_msg, f"Did not receive message on {test_topic} within 3s")
        self.assertEqual(received_msgs, [sent_msg])

        pub_client.loop_stop()
        pub_client.disconnect()
        sub_client.loop_stop()
        sub_client.disconnect()

    def test_03_publish_subscribe_json_payload(self):
        """Verifies publish and subscribe of a structured JSON payload."""
        test_topic = f"{TOPIC_BASE}/json"
        payload_data = {
            "event_id": "STAGE6B-TEST-001",
            "status": "CONFIRMED",
            "confidence": 0.9542,
            "source": "python_test_runner",
            "timestamp": "2026-10-06T18:22:00Z"
        }
        received_payloads = []
        msg_event = threading.Event()

        def on_message(client, userdata, message):
            try:
                data = json.loads(message.payload.decode("utf-8"))
                received_payloads.append(data)
            except Exception as e:
                received_payloads.append({"error": str(e)})
            msg_event.set()

        sub_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_sub_json"
        )
        sub_client.on_message = on_message
        sub_client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        sub_client.subscribe(test_topic)
        sub_client.loop_start()

        time.sleep(0.1)

        pub_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_pub_json"
        )
        pub_client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        pub_client.loop_start()

        pub_client.publish(test_topic, payload=json.dumps(payload_data), qos=1)

        got_msg = msg_event.wait(timeout=3.0)
        self.assertTrue(got_msg, "Did not receive JSON payload within 3s")
        self.assertEqual(len(received_payloads), 1)
        self.assertEqual(received_payloads[0]["event_id"], "STAGE6B-TEST-001")
        self.assertEqual(received_payloads[0]["status"], "CONFIRMED")
        self.assertAlmostEqual(received_payloads[0]["confidence"], 0.9542, places=4)
        self.assertEqual(received_payloads[0]["source"], "python_test_runner")

        pub_client.loop_stop()
        pub_client.disconnect()
        sub_client.loop_stop()
        sub_client.disconnect()

    def test_04_multiple_qos_transmissions(self):
        """Verifies QoS 0 and QoS 1 deliveries."""
        test_topic = f"{TOPIC_BASE}/qos"
        received = []
        done_event = threading.Event()

        def on_message(client, userdata, message):
            received.append(message.payload.decode("utf-8"))
            if len(received) == 2:
                done_event.set()

        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_stage6b_qos"
        )
        client.on_message = on_message
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=10)
        client.subscribe(test_topic, qos=1)
        client.loop_start()

        time.sleep(0.1)

        client.publish(test_topic, "qos_0_msg", qos=0)
        client.publish(test_topic, "qos_1_msg", qos=1)

        success = done_event.wait(timeout=3.0)
        self.assertTrue(success, f"Expected 2 messages, got {len(received)}")
        self.assertIn("qos_0_msg", received)
        self.assertIn("qos_1_msg", received)

        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
