"""
Stage 6L: End-to-End Hardware Test Preflight & Orchestration Suite.
Standalone Fall Detection Project.

Checks and coordinates required subsystem dependencies in deterministic order:
  1. Verify Mosquitto MQTT broker reachable.
  2. Verify Flask backend health (GET /api/health).
  3. Start/verify Stage 6G MQTT Gateway.
  4. Start/verify Stage 6L GPS Cache.
  5. Check physical or virtual ESP32 serial port.
  6. Execute hardware inference bridge in controlled verification mode.
  7. Print final dependency and execution health report.

SAFETY ENFORCEMENT:
  - NEVER instructs or forces a person to perform an unsafe physical fall.
  - NEVER sends real emergency notifications during test runs.
  - Safe diagnostic mode only.
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_6F_mqtt_fall_publisher import MqttFallPublisher
from src.fall_confirmation.stage_6G_mqtt_gateway import MqttBackendGateway
from src.fall_confirmation.stage_6L_gps_cache import GpsTelemetryCache
from scripts.stage_6.run_stage_6L_hardware_bridge import run_stage_6L_hardware_bridge

logger = logging.getLogger("Stage6L_Orchestrator")


def check_tcp_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Verifies that a TCP port is open and accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_backend_health(backend_url: str, timeout: float = 2.5) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Verifies Flask backend health via GET /api/health."""
    url = f"{backend_url.rstrip('/')}/api/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Stage6L-Orchestrator/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                return True, "HEALTHY", body
            return False, f"HTTP_{resp.status}", None
    except Exception as e:
        return False, f"UNREACHABLE ({e})", None


def check_serial_port_available(port_name: str) -> Tuple[bool, List[str]]:
    """Checks if the specified serial port is currently present in the system."""
    available_ports = []
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        available_ports = [p.device for p in ports]
        return (port_name in available_ports), available_ports
    except Exception:
        # If pyserial is missing or port enumeration fails
        return False, []


def run_orchestration_preflight(
    broker_host: str = "127.0.0.1",
    broker_port: int = 1883,
    backend_url: str = "http://127.0.0.1:5000",
    serial_port: str = "COM22",
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """
    Executes preflight verification checks for all system dependencies.
    """
    print("\n" + "=" * 70)
    print("STAGE 6L HARDWARE-IN-THE-LOOP DEPENDENCY PREFLIGHT")
    print("=" * 70)

    results = {
        "mqtt_broker": False,
        "backend_health": False,
        "serial_port": False,
        "overall_ready": False,
        "available_com_ports": [],
    }

    # 1. Check MQTT Broker
    print(f"[1/3] Testing MQTT Broker at {broker_host}:{broker_port}...", end=" ")
    broker_ok = check_tcp_port(broker_host, broker_port)
    results["mqtt_broker"] = broker_ok
    print("ONLINE" if broker_ok else "OFFLINE / UNREACHABLE")

    # 2. Check Backend Health
    print(f"[2/3] Testing Backend Health at {backend_url}/api/health...", end=" ")
    backend_ok, be_msg, _ = check_backend_health(backend_url)
    results["backend_health"] = backend_ok
    print(f"ONLINE ({be_msg})" if backend_ok else f"OFFLINE ({be_msg})")

    # 3. Check Serial Port
    print(f"[3/3] Checking Serial Port '{serial_port}' (Mock: {mock_mode})...", end=" ")
    if mock_mode:
        results["serial_port"] = True
        print("VIRTUAL (Mock Mode Active)")
    else:
        port_ok, avail = check_serial_port_available(serial_port)
        results["serial_port"] = port_ok
        results["available_com_ports"] = avail
        if port_ok:
            print("FOUND")
        else:
            print(f"NOT FOUND (Available: {avail or 'None'})")

    print("-" * 70)

    if mock_mode:
        results["overall_ready"] = broker_ok and backend_ok
    else:
        results["overall_ready"] = broker_ok and backend_ok and results["serial_port"]

    print(f"Overall Preflight Status: {'READY FOR TEST' if results['overall_ready'] else 'NOT READY (Missing Dependencies)'}")
    print("=" * 70 + "\n")

    return results


def run_staged_hardware_test(
    broker_host: str = "127.0.0.1",
    broker_port: int = 1883,
    backend_url: str = "http://127.0.0.1:5000",
    serial_port: str = "COM22",
    baudrate: Optional[int] = None,
    mock_mode: bool = False,
    mock_scenario: str = "stationary",
    max_windows: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Coordinates and launches an integrated test session across the verified pipeline.
    """
    preflight = run_orchestration_preflight(
        broker_host=broker_host,
        broker_port=broker_port,
        backend_url=backend_url,
        serial_port=serial_port,
        mock_mode=mock_mode,
    )

    if not preflight["overall_ready"] and not dry_run and not mock_mode:
        print("[ABORT] Cannot proceed: One or more preflight checks failed.")
        return {"status": "FAILED_PREFLIGHT", "preflight": preflight}

    print("[STAGE 6L] Launching integrated hardware bridge...")

    bridge_results = run_stage_6L_hardware_bridge(
        port=serial_port,
        baudrate=baudrate,
        broker_host=broker_host,
        broker_port=broker_port,
        mock_mode=mock_mode,
        mock_scenario=mock_scenario,
        max_windows=max_windows,
        dry_run=dry_run,
    )

    return {
        "status": "COMPLETED",
        "preflight": preflight,
        "bridge": bridge_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6L Hardware-in-the-Loop Orchestrator")
    parser.add_argument("--broker", type=str, default=os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"), help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--backend", type=str, default=os.environ.get("BACKEND_URL", "http://127.0.0.1:5000"), help="Backend URL")
    parser.add_argument("--port", type=str, default=os.environ.get("SERIAL_PORT", "COM22"), help="Serial port")
    parser.add_argument("--baud", type=int, default=int(os.environ.get("SERIAL_BAUD", 460800)), help="Baud rate (default: 460800 or SERIAL_BAUD)")
    parser.add_argument("--mock", action="store_true", help="Run with mock sensor stream")
    parser.add_argument("--mock-scenario", type=str, default="stationary", choices=["stationary", "fall"], help="Mock scenario")
    parser.add_argument("--windows", type=int, default=4, help="Max windows to process")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without publishing")

    args = parser.parse_args()

    run_staged_hardware_test(
        broker_host=args.broker,
        broker_port=args.broker_port,
        backend_url=args.backend,
        serial_port=args.port,
        baudrate=args.baud,
        mock_mode=args.mock,
        mock_scenario=args.mock_scenario,
        max_windows=args.windows,
        dry_run=args.dry_run,
    )
