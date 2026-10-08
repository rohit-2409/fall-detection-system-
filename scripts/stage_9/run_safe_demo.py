"""
Stage 9: Safe Demonstration Event Generator.
Standalone Wearable Fall Detection System.

Purpose:
  Enables controlled, repeatable demonstrations of the Incident Command Center,
  GPS mapping, incident acknowledgment, and resolution workflows during
  engineering reviews and final project presentations WITHOUT requiring a
  dangerous physical fall or tampering with the frozen Random Forest detector.

Safety & Audit Contract:
  1. Event ID prefix is strictly: 'DEMO-FALL-...'
  2. Source platform is explicitly: 'SIMULATED_DEMO_GENERATOR'
  3. Diagnostic metadata explicitly flags 'demo_mode': True
  4. Core ML model, scalers, and temporal confirmation rules are 100% UNTOUCHED.
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
DEFAULT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
DEFAULT_FALL_TOPIC = os.environ.get("MQTT_FALL_TOPIC", "fall_detection/stage6/fall")
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


def generate_demo_payload(
    confidence: float = 0.9412,
    lat: float = 22.728421,
    lon: float = 75.869163,
    satellites: int = 9,
    hdop: float = 0.85,
) -> Dict[str, Any]:
    """Generates an unmistakably marked demo event payload adhering to the Stage 6F contract."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    now_compact = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_id = f"DEMO-FALL-{now_compact}-01"

    payload = {
        "event": "CONFIRMED_FALL",
        "event_id": event_id,
        "status": "CONFIRMED",
        "detected_at": now_iso,
        "occurred_at": now_iso,
        "session_id": f"DEMO-SESSION-{now_compact[:8]}",
        "confidence": round(confidence, 4),
        "source": "SIMULATED_DEMO_GENERATOR",
        "location": {
            "fix_valid": True,
            "latitude": lat,
            "longitude": lon,
            "satellites": satellites,
            "hdop": hdop,
            "timestamp": now_iso,
            "is_cached": False,
        },
        "diagnostic_metadata": {
            "demo_mode": True,
            "simulation_notice": "SIMULATED DEMONSTRATION EVENT - FOR PRESENTATION ONLY",
            "kinematics": {
                "dynamic_acceleration_peak_g": 6.82,
                "gyroscope_magnitude_peak_dps": 482.3,
                "gait_stability_index": 0.12,
            },
            "location": {
                "fix_valid": True,
                "latitude": lat,
                "longitude": lon,
                "satellites": satellites,
                "hdop": hdop,
                "timestamp": now_iso,
                "is_cached": False,
            },
        },
    }
    return payload


def send_via_mqtt(
    payload: Dict[str, Any],
    broker_host: str = DEFAULT_BROKER_HOST,
    broker_port: int = DEFAULT_BROKER_PORT,
    topic: str = DEFAULT_FALL_TOPIC,
) -> bool:
    """Dispatches demo event through the full MQTT -> Gateway -> Backend pipeline."""
    try:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id="FALL-STAGE9-DEMO-INJECTOR")
        client.connect(broker_host, broker_port, keepalive=10)
        client.loop_start()

        body = json.dumps(payload)
        info = client.publish(topic, body, qos=1)
        info.wait_for_publish(timeout=3.0)

        client.loop_stop()
        client.disconnect()
        return True
    except Exception as ex:
        print(f"[ERROR] Failed to publish demo event via MQTT ({broker_host}:{broker_port}): {ex}")
        return False


def send_via_rest(
    payload: Dict[str, Any],
    backend_url: str = DEFAULT_BACKEND_URL,
) -> bool:
    """Dispatches demo event directly to the Flask REST backend API."""
    url = f"{backend_url}/api/fall-events"
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Stage9DemoInjector/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.status in (200, 201)
    except Exception as ex:
        print(f"[ERROR] Failed to post demo event to {url}: {ex}")
        return False


def print_banner():
    print("=" * 70)
    print("  FALL DETECTION SYSTEM: SAFE DEMONSTRATION EVENT INJECTOR")
    print("  [DEMO / EVALUATION MODE ONLY — PRODUCTION MODEL UNTOUCHED]")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inject a simulated demo fall event for command center testing")
    parser.add_argument("--via", choices=["mqtt", "rest"], default="mqtt", help="Transport mechanism (default: mqtt)")
    parser.add_argument("--confidence", type=float, default=0.9412, help="Simulated confidence (default: 0.9412)")
    parser.add_argument("--lat", type=float, default=22.728421, help="Simulated latitude (default: 22.728421)")
    parser.add_argument("--lon", type=float, default=75.869163, help="Simulated longitude (default: 75.869163)")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER_HOST, help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=DEFAULT_BROKER_PORT, help="MQTT broker port")
    parser.add_argument("--topic", type=str, default=DEFAULT_FALL_TOPIC, help="MQTT fall topic")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND_URL, help="Backend URL")

    args = parser.parse_args()

    print_banner()

    payload = generate_demo_payload(
        confidence=args.confidence,
        lat=args.lat,
        lon=args.lon,
    )

    print(f"\n[INFO] Generated Demonstration Event:")
    print(f"       Incident ID:  {payload['event_id']}")
    print(f"       Status:       {payload['status']}")
    print(f"       Confidence:   {payload['confidence'] * 100:.2f}%")
    print(f"       Source Tag:   {payload['source']}")
    print(f"       Location:     {payload['diagnostic_metadata']['location']['latitude']:.6f}, {payload['diagnostic_metadata']['location']['longitude']:.6f}")
    print(f"       Transport:    {args.via.upper()}")

    success = False
    if args.via == "mqtt":
        print(f"\n[INFO] Publishing to MQTT topic '{args.topic}' at {args.broker}:{args.broker_port}...")
        success = send_via_mqtt(payload, broker_host=args.broker, broker_port=args.broker_port, topic=args.topic)
        if not success:
            print("[WARNING] MQTT publication failed; falling back to REST POST...")
            success = send_via_rest(payload, backend_url=args.backend)
    else:
        print(f"\n[INFO] Sending direct HTTP POST to {args.backend}/api/fall-events...")
        success = send_via_rest(payload, backend_url=args.backend)

    print("-" * 70)
    if success:
        print("[SUCCESS] Demonstration event transmitted successfully!")
        print("\nNext Steps for Live Demonstration:")
        print("  1. Open Dashboard:  http://127.0.0.1:5000/dashboard")
        print("  2. Observe:         ACTIVE FALL INCIDENT banner (red pulse)")
        print("  3. View GPS Map:    Marker centered at simulated coordinates")
        print("  4. Demonstrate:     Click 'ACKNOWLEDGE INCIDENT' -> status updates")
        print("  5. Demonstrate:     Click 'RESOLVE INCIDENT' -> system returns to normal")
        print("=" * 70)
        sys.exit(0)
    else:
        print("[FAILURE] Could not dispatch demonstration event.")
        print("Verify that either the MQTT broker or the Flask backend is currently running.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
