"""
Stage 9 Production Launcher & Process Orchestrator.
Standalone Wearable Fall Detection System.

Coordinates verified startup across:
  [1] Environment & dependency check
  [2] Database integrity check
  [3] MQTT broker connectivity probe
  [4] Backend startup (with duplicate instance detection)
  [5] Dashboard availability verification
  [6] ESP32 physical COM port detection
  [7] Hardware bridge launch (physical UART or mock simulation)
  [8] Composite system health verification

Handles graceful shutdown (Ctrl+C / SIGINT) without leaving orphan processes.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_9.preflight_check import (
    check_dashboard_and_backend,
    check_database,
    check_dependencies,
    check_esp32_port,
    check_model_integrity,
    check_mqtt_broker,
    is_port_in_use,
)

DEFAULT_SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM22")
DEFAULT_SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", 460800))
DEFAULT_MQTT_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
DEFAULT_MQTT_PORT = int(os.environ.get("MQTT_BROKER_PORT", 1883))
DEFAULT_BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 5000))
DEFAULT_BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")


class ProductionLauncher:
    """Manages full lifecycle of backend, dashboard, and hardware bridge."""

    def __init__(
        self,
        port: int = DEFAULT_BACKEND_PORT,
        host: str = DEFAULT_BACKEND_HOST,
        serial_port: str = DEFAULT_SERIAL_PORT,
        serial_baud: int = DEFAULT_SERIAL_BAUD,
        open_browser: bool = True,
        start_bridge: bool = False,
        mock_bridge: bool = False,
    ):
        self.port = port
        self.host = host
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.open_browser = open_browser
        self.start_bridge = start_bridge
        self.mock_bridge = mock_bridge

        self.backend_process: Optional[subprocess.Popen] = None
        self.bridge_process: Optional[subprocess.Popen] = None
        self._shutdown_event = threading.Event()
        self.backend_already_running = False

    def print_startup_banner(self):
        print("=" * 50)
        print("FALL DETECTION SYSTEM")
        print("Production Startup")
        print("=" * 50)
        print("\n[1] Environment check")
        print("[2] Database check")
        print("[3] MQTT broker check")
        print("[4] Backend startup")
        print("[5] Dashboard availability")
        print("[6] ESP32 detection")
        print("[7] Hardware bridge startup")
        print("[8] System health verification")
        print("\n" + "=" * 50 + "\n")

    def run_preflight(self) -> bool:
        """Executes steps 1 to 3 of preflight."""
        # [1] Environment check
        print("[1] Environment & Dependency Check:")
        dep_ok, dep_logs = check_dependencies()
        model_ok, model_logs = check_model_integrity()
        if not (dep_ok and model_ok):
            print("    [FAIL] DEPENDENCY OR MODEL INTEGRITY CHECK FAILED:")
            for log in dep_logs + model_logs:
                if "[FAIL]" in log:
                    print(f"      {log}")
            return False
        print("    [OK] Python runtime, libraries, and frozen ML model hashes verified.")

        # [2] Database check
        print("\n[2] Database Integrity Check:")
        db_ok, db_logs = check_database()
        if not db_ok:
            print("    [FAIL] DATABASE CHECK FAILED:")
            for log in db_logs:
                print(f"      {log}")
            return False
        for log in db_logs:
            if "[OK]" in log or "[INFO]" in log:
                print(f"    {log}")

        # [3] MQTT broker check
        print("\n[3] MQTT Broker Check:")
        mqtt_ok, mqtt_logs = check_mqtt_broker(self.host, DEFAULT_MQTT_PORT)
        for log in mqtt_logs:
            print(f"    {log}")

        return True

    def start_backend(self) -> bool:
        """Starts Flask backend and embedded gateway or attaches to running instance."""
        print("\n[4] Backend Startup:")

        # Check for existing process on port
        if is_port_in_use(self.port, self.host):
            # Probe if it is our backend
            health_url = f"http://{self.host}:{self.port}/api/health"
            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "LauncherProbe/1.0"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status == 200:
                        self.backend_already_running = True
                        print(f"    [OK] Backend service already active on port {self.port}. Attaching cleanly.")
                        return True
            except Exception:
                print(f"    [FAIL] Port {self.port} is in use by another process. Cannot bind backend.")
                return False

        # Spawn backend subprocess
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "stage_6", "run_full_backend.py"),
            "--port", str(self.port),
            "--host", self.host,
        ]
        try:
            self.backend_process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            # Wait for backend readiness
            health_url = f"http://{self.host}:{self.port}/api/health"
            ready = False
            start_t = time.time()
            while time.time() - start_t < 8.0:
                try:
                    req = urllib.request.Request(health_url, headers={"User-Agent": "LauncherProbe/1.0"})
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            ready = True
                            break
                except Exception:
                    time.sleep(0.2)

            if ready:
                print(f"    [OK] Unified Backend started on http://{self.host}:{self.port}")
                return True
            else:
                print(f"    [FAIL] Backend did not respond within 8 seconds.")
                return False
        except Exception as ex:
            print(f"    [FAIL] Failed to launch backend: {ex}")
            return False

    def verify_dashboard(self) -> bool:
        """Verifies dashboard routes respond with HTTP 200."""
        print("\n[5] Dashboard Availability:")
        dash_url = f"http://{self.host}:{self.port}/dashboard"
        try:
            req = urllib.request.Request(dash_url, headers={"User-Agent": "LauncherProbe/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"    [OK] Command Center Dashboard is READY at: {dash_url}")
                    return True
                else:
                    print(f"    [FAIL] Dashboard returned unexpected HTTP status {resp.status}")
                    return False
        except Exception as ex:
            print(f"    [FAIL] Could not reach dashboard: {ex}")
            return False

    def check_hardware_esp32(self) -> bool:
        """Detects physical ESP32 UART port."""
        print("\n[6] ESP32 Detection:")
        detected, logs = check_esp32_port(self.serial_port, self.serial_baud)
        for log in logs:
            print(f"    {log}")
        return detected

    def launch_bridge_if_requested(self, esp32_connected: bool):
        """Starts hardware inference bridge if requested."""
        print("\n[7] Hardware Bridge Status:")
        if not self.start_bridge and not self.mock_bridge:
            print("    [INFO] Hardware bridge launch skipped (Start via separate terminal or --start-bridge).")
            return

        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "stage_6", "run_stage_6L_hardware_bridge.py")]
        if self.mock_bridge or not esp32_connected:
            cmd.append("--mock-mode")
            print("    [OK] Launching Stage 6L Bridge in MOCK SIMULATION mode...")
        else:
            cmd.extend(["--port", self.serial_port, "--baud", str(self.serial_baud)])
            print(f"    [OK] Launching Stage 6L Bridge on {self.serial_port} @ {self.serial_baud} baud...")

        try:
            self.bridge_process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            print("    [OK] Hardware Bridge active in background.")
        except Exception as ex:
            print(f"    [WARNING] Could not start Hardware Bridge: {ex}")

    def verify_system_health(self):
        """Queries /api/system/health and presents current system status honestly."""
        print("\n[8] System Health Verification:")
        health_url = f"http://{self.host}:{self.port}/api/system/health"
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "LauncherProbe/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode())
                status = data.get("status", "UNKNOWN")
                print(f"\n    ================================================")
                print(f"    SYSTEM STATUS:  [{status}]")
                print(f"    ================================================")
                print(f"    Backend:        {data.get('backend', {}).get('status', 'UNKNOWN')}")
                print(f"    Database:       {data.get('database', {}).get('status', 'UNKNOWN')}")
                print(f"    MQTT:           {data.get('mqtt', {}).get('status', 'UNKNOWN')}")
                print(f"    Gateway:        {data.get('gateway', {}).get('status', 'UNKNOWN')}")
                print(f"    ESP32:          {data.get('esp32', {}).get('status', 'UNKNOWN')}")
                print(f"    GPS:            {data.get('gps', {}).get('status', 'UNKNOWN')}")
                print(f"    ================================================\n")
        except Exception as ex:
            print(f"    [WARNING] Could not query system health endpoint: {ex}")

    def shutdown(self):
        """Gracefully shuts down spawned processes without orphaned children."""
        print("\n[SHUTDOWN] Terminating system processes cleanly...")
        self._shutdown_event.set()

        if self.bridge_process:
            try:
                print("  Stopping Hardware Bridge...")
                self.bridge_process.terminate()
                self.bridge_process.wait(timeout=3.0)
            except Exception:
                try:
                    self.bridge_process.kill()
                except Exception:
                    pass
            self.bridge_process = None

        if self.backend_process and not self.backend_already_running:
            try:
                print("  Stopping Unified Backend...")
                self.backend_process.terminate()
                self.backend_process.wait(timeout=3.0)
            except Exception:
                try:
                    self.backend_process.kill()
                except Exception:
                    pass
            self.backend_process = None

        print("[SHUTDOWN] System stopped cleanly. Goodbye.")

    def run(self) -> int:
        """Main launcher workflow."""
        self.print_startup_banner()

        # Step 1 - 3
        if not self.run_preflight():
            return 1

        # Step 4
        if not self.start_backend():
            return 1

        # Step 5
        if not self.verify_dashboard():
            return 1

        # Step 6
        esp32_connected = self.check_hardware_esp32()

        # Step 7
        self.launch_bridge_if_requested(esp32_connected)

        # Step 8
        self.verify_system_health()

        # Open web dashboard in browser
        dash_url = f"http://{self.host}:{self.port}/dashboard"
        if self.open_browser:
            print(f"[INFO] Opening Command Center Dashboard in browser: {dash_url}")
            try:
                webbrowser.open(dash_url)
            except Exception:
                pass

        print("\n[READY] Press Ctrl+C at any time to exit and shutdown services cleanly.")

        # If backend was already running and no bridge was spawned, exit cleanly
        if self.backend_already_running and not self.bridge_process:
            print("[INFO] Attached to existing background instance. Launcher exiting standby.")
            return 0

        # Wait on background processes
        try:
            while not self._shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

        return 0


def main():
    parser = argparse.ArgumentParser(description="Fall Detection System Production Launcher")
    parser.add_argument("--port", type=int, default=DEFAULT_BACKEND_PORT, help="Backend bind port")
    parser.add_argument("--host", type=str, default=DEFAULT_BACKEND_HOST, help="Backend bind host")
    parser.add_argument("--serial-port", type=str, default=DEFAULT_SERIAL_PORT, help="ESP32 serial port")
    parser.add_argument("--serial-baud", type=int, default=DEFAULT_SERIAL_BAUD, help="ESP32 serial baud rate")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--start-bridge", action="store_true", help="Automatically launch Hardware Bridge")
    parser.add_argument("--mock-bridge", action="store_true", help="Launch Hardware Bridge in mock mode")

    args = parser.parse_args()

    launcher = ProductionLauncher(
        port=args.port,
        host=args.host,
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        open_browser=(not args.no_browser),
        start_bridge=args.start_bridge,
        mock_bridge=args.mock_bridge,
    )

    exit_code = launcher.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
