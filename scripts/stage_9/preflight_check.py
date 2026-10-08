"""
Stage 9: Production Preflight & System Readiness Verification Engine.
Standalone Wearable Fall Detection System.

Performs read-only validation across:
  1. Python runtime & package dependencies
  2. Frozen ML model & pipeline SHA-256 hashes
  3. SQLite database schema & integrity
  4. Mosquitto MQTT broker availability
  5. ESP32 serial communication & COM port status
  6. Backend REST API & Stage 8B Command Center dashboard availability
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# Frozen ML Artifact Hashes (Immutable Project Constants)
# -----------------------------------------------------------------------------
EXPECTED_HASHES = {
    os.path.join("fall_detection_dataset", "models", "stage_4D", "stage_4D_random_forest.joblib"):
        "a499a0dd0107d9b25de01012a2efe8293f1b1e2092d1af51b08237efd3a8584c",
    os.path.join("fall_detection_dataset", "models", "stage_4D", "scaler.joblib"):
        "ea73df7b4756911221b670a69d7195baae65492bfea43cdf80c3d6dfcde346af",
    os.path.join("fall_detection_dataset", "models", "stage_4D", "feature_order.json"):
        "995941391e4fe95a2899131583a30cf3be895d1c78401ff66bfc61333262d65b",
    os.path.join("src", "fall_confirmation", "stage_4G_live_validator.py"):
        "64bfac3347f3bb87dfd7c7967488cf4359f0319c63cd17583c681aace28d2728",
    os.path.join("src", "fall_confirmation", "temporal_adapter.py"):
        "0dc59c8f992e03ac128ad45a19600b6dbc05185d07e92f0f89c2695ad08db198",
}

REQUIRED_MODULES = [
    ("flask", "Flask web server & REST API framework"),
    ("numpy", "Numerical signal processing & feature extraction math"),
    ("scipy", "Butterworth/causal signal filtering"),
    ("pandas", "Dataset parsing & telemetry recording"),
    ("pyarrow", "Parquet columnar storage interface"),
    ("sklearn", "Scikit-Learn Random Forest inference runtime"),
    ("joblib", "Model serialization and deserialization"),
    ("serial", "PySerial UART serial communications"),
    ("paho.mqtt", "Eclipse Paho MQTT client protocol engine"),
]

DEFAULT_MQTT_HOST = "127.0.0.1"
DEFAULT_MQTT_PORT = 1883
DEFAULT_FALL_TOPIC = "fall_detection/stage6/fall"


def check_dependencies() -> Tuple[bool, List[str]]:
    """Validates Python version and required libraries."""
    logs = []
    all_ok = True

    # 1. Python runtime
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        logs.append(f"[OK] Python Runtime: {v_str} (Compatible >= 3.10)")
    else:
        logs.append(f"[FAIL] Python Runtime: {v_str} (Requires Python >= 3.10)")
        all_ok = False

    # 2. Required modules
    for mod_name, desc in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            try:
                clean_name = "pyserial" if mod_name == "serial" else ("paho-mqtt" if mod_name == "paho.mqtt" else mod_name)
                ver = importlib.metadata.version(clean_name)
            except Exception:
                ver = getattr(mod, "__version__", "installed")
            logs.append(f"[OK] Module '{mod_name}' ({ver}): {desc}")
        except ImportError:
            logs.append(f"[FAIL] Module '{mod_name}' is MISSING: {desc}")
            all_ok = False

    return all_ok, logs


def check_model_integrity() -> Tuple[bool, List[str]]:
    """Calculates SHA-256 hashes of frozen detection core and validates against golden baseline."""
    logs = []
    all_ok = True

    for rel_path, expected_hash in EXPECTED_HASHES.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            logs.append(f"[FAIL] Missing model asset: {rel_path}")
            all_ok = False
            continue

        try:
            with open(full_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if actual_hash == expected_hash:
                logs.append(f"[OK] Hash Verified: {os.path.basename(rel_path)} ({actual_hash[:16]}...)")
            else:
                logs.append(f"[FAIL] HASH MISMATCH: {rel_path}\n       Expected: {expected_hash}\n       Actual:   {actual_hash}")
                all_ok = False
        except Exception as ex:
            logs.append(f"[FAIL] Error reading {rel_path}: {ex}")
            all_ok = False

    return all_ok, logs


def check_database() -> Tuple[bool, List[str]]:
    """Performs non-destructive, read-only SQLite schema and integrity validation."""
    logs = []
    all_ok = True

    db_path = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "fall_events.db")
    if not os.path.exists(db_path):
        logs.append(f"[FAIL] Database file not found at: {db_path}")
        return False, logs

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        # Integrity check
        cur.execute("PRAGMA integrity_check;")
        res = cur.fetchone()
        if res and res[0] == "ok":
            logs.append("[OK] SQLite PRAGMA integrity_check: ok")
        else:
            logs.append(f"[FAIL] SQLite integrity check failed: {res}")
            all_ok = False

        # Table check
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events';")
        if cur.fetchone():
            logs.append("[OK] Required table 'fall_events' exists")
        else:
            logs.append("[FAIL] Table 'fall_events' missing from database")
            all_ok = False

        # Event count & latest event
        cur.execute("SELECT COUNT(*) FROM fall_events;")
        count = cur.fetchone()[0]
        cur.execute("SELECT event_id, detected_at, status FROM fall_events ORDER BY created_at DESC LIMIT 1;")
        latest = cur.fetchone()
        conn.close()

        logs.append(f"[OK] Total Recorded Events: {count}")
        if latest:
            logs.append(f"[INFO] Latest Event: {latest[0]} ({latest[2]} at {latest[1]})")

    except Exception as ex:
        logs.append(f"[FAIL] Database verification error: {ex}")
        all_ok = False

    return all_ok, logs


def check_mqtt_broker(host: str = "127.0.0.1", port: int = 1883, timeout: float = 2.0) -> Tuple[bool, List[str]]:
    """Probes the Mosquitto MQTT broker TCP listener."""
    logs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        logs.append(f"[OK] MQTT Broker is ONLINE at {host}:{port}")
        return True, logs
    except Exception as ex:
        logs.append(f"[WARNING] MQTT Broker not reachable at {host}:{port}: {ex}")
        logs.append("[INFO] System can operate, but real-time hardware MQTT publishing will be suspended.")
        return False, logs


def check_esp32_port(port: str = "COM22", baud: int = 460800) -> Tuple[bool, List[str]]:
    """Scans for physical ESP32 COM port without locking or interrupting active telemetry."""
    logs = []
    try:
        import serial.tools.list_ports
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        if port in available_ports:
            logs.append(f"[OK] ESP32 Serial Port '{port}' DETECTED (Configured baud: {baud})")
            return True, logs
        else:
            logs.append(f"[INFO] ESP32 Serial Port '{port}' NOT DETECTED (Available ports: {available_ports or 'None'})")
            logs.append(f"[INFO] Hardware bridge can run in mock simulation mode if hardware is disconnected.")
            return False, logs
    except Exception as ex:
        logs.append(f"[WARNING] Serial port scan error: {ex}")
        return False, logs


def check_dashboard_and_backend(host: str = "127.0.0.1", port: int = 5000, timeout: float = 2.5) -> Tuple[bool, List[str]]:
    """Verifies HTTP endpoints for REST API, system health, and web dashboard."""
    logs = []
    all_ok = True
    base_url = f"http://{host}:{port}"

    endpoints = [
        ("/", "Root Web Dashboard"),
        ("/dashboard", "Incident Command Center Interface"),
        ("/api/health", "Legacy Service Health Endpoint"),
        ("/api/system/health", "Stage 8A Comprehensive System Health Monitor"),
    ]

    for path, name in endpoints:
        url = f"{base_url}{path}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PreflightCheck/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    logs.append(f"[OK] {name}: HTTP 200 at {url}")
                else:
                    logs.append(f"[FAIL] {name}: HTTP {resp.status} at {url}")
                    all_ok = False
        except urllib.error.URLError as ex:
            logs.append(f"[FAIL] {name}: Cannot connect to {url} ({ex.reason})")
            all_ok = False
        except Exception as ex:
            logs.append(f"[FAIL] {name}: Request error: {ex}")
            all_ok = False

    return all_ok, logs


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is currently bound by another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def run_full_preflight() -> Dict[str, Any]:
    """Runs the complete Stage 9 preflight suite and returns structured report."""
    dep_ok, dep_logs = check_dependencies()
    model_ok, model_logs = check_model_integrity()
    db_ok, db_logs = check_database()
    mqtt_ok, mqtt_logs = check_mqtt_broker()
    esp32_ok, esp32_logs = check_esp32_port()

    # Core system viability requires dependencies, models, and database
    core_ready = dep_ok and model_ok and db_ok

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "core_ready": core_ready,
        "dependencies": {"status": "PASS" if dep_ok else "FAIL", "logs": dep_logs},
        "model_integrity": {"status": "PASS" if model_ok else "FAIL", "logs": model_logs},
        "database": {"status": "PASS" if db_ok else "FAIL", "logs": db_logs},
        "mqtt": {"status": "ONLINE" if mqtt_ok else "OFFLINE", "logs": mqtt_logs},
        "esp32": {"status": "CONNECTED" if esp32_ok else "NOT_DETECTED", "logs": esp32_logs},
    }
    return report


def print_banner():
    print("=" * 70)
    print("  FALL DETECTION SYSTEM: STAGE 9 PRODUCTION PREFLIGHT ENGINE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stage 9 System Readiness & Preflight Verification")
    parser.add_argument("--all", action="store_true", help="Run full system preflight")
    parser.add_argument("--dependencies", action="store_true", help="Check Python dependencies")
    parser.add_argument("--models", action="store_true", help="Verify frozen ML model hashes")
    parser.add_argument("--database", action="store_true", help="Check database integrity")
    parser.add_argument("--mqtt", action="store_true", help="Probe MQTT broker")
    parser.add_argument("--esp32", action="store_true", help="Detect ESP32 COM port")
    parser.add_argument("--backend", action="store_true", help="Verify live dashboard endpoints")
    parser.add_argument("--json", action="store_true", help="Output JSON results")

    args = parser.parse_args()

    # Default to --all if no specific flag supplied
    run_all = args.all or not (args.dependencies or args.models or args.database or args.mqtt or args.esp32 or args.backend)

    if args.json:
        report = run_full_preflight()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["core_ready"] else 1)

    print_banner()

    overall_success = True

    if run_all or args.dependencies:
        print("\n[1] DEPENDENCY VERIFICATION:")
        ok, logs = check_dependencies()
        for log in logs:
            print(f"    {log}")
        if not ok:
            overall_success = False

    if run_all or args.models:
        print("\n[2] FROZEN MODEL INTEGRITY (SHA-256):")
        ok, logs = check_model_integrity()
        for log in logs:
            print(f"    {log}")
        if not ok:
            overall_success = False

    if run_all or args.database:
        print("\n[3] DATABASE INTEGRITY & SCHEMA:")
        ok, logs = check_database()
        for log in logs:
            print(f"    {log}")
        if not ok:
            overall_success = False

    if run_all or args.mqtt:
        print("\n[4] MQTT BROKER CONNECTIVITY:")
        _, logs = check_mqtt_broker()
        for log in logs:
            print(f"    {log}")

    if run_all or args.esp32:
        print("\n[5] HARDWARE SENSOR COM PORT:")
        _, logs = check_esp32_port()
        for log in logs:
            print(f"    {log}")

    if args.backend:
        print("\n[6] LIVE BACKEND & DASHBOARD VERIFICATION:")
        ok, logs = check_dashboard_and_backend()
        for log in logs:
            print(f"    {log}")
        if not ok:
            overall_success = False

    print("\n" + "=" * 70)
    if overall_success:
        print("  RESULT: PREFLIGHT VERIFICATION PASSED (SYSTEM READY)")
        print("=" * 70)
        sys.exit(0)
    else:
        print("  RESULT: PREFLIGHT VERIFICATION FAILED")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
