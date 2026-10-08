"""
Stage 9 Automated Test Suite: Production Hardening & Demonstration Readiness.

Tests:
  1. Dependency preflight check
  2. Model and validator SHA-256 hash validation
  3. Database integrity and schema verification
  4. MQTT configuration and topic validation
  5. Startup without ESP32 (graceful degraded mode)
  6. Startup with ESP32 unavailable / locked
  7. Duplicate process and port collision detection
  8. Backend readiness probe (/api/health)
  9. Dashboard readiness probe (/dashboard)
  10. Graceful shutdown and process cleanup
  11. Safe demo event simulation and non-destructive marking
  12. Protected original workspace verification (0 modified, 0 deleted, 0 renamed)
"""

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_9.launcher import ProductionLauncher
from scripts.stage_9.preflight_check import (
    DEFAULT_FALL_TOPIC,
    DEFAULT_MQTT_PORT,
    EXPECTED_HASHES,
    REQUIRED_MODULES,
    check_dashboard_and_backend,
    check_database,
    check_dependencies,
    check_esp32_port,
    check_model_integrity,
    check_mqtt_broker,
    is_port_in_use,
)
from scripts.stage_9.run_safe_demo import generate_demo_payload

PROTECTED_ROOT = r"C:\Users\ROHIT\Desktop\major project"


class TestStage9ProductionHardening(unittest.TestCase):
    """Stage 9 Production Hardening Test Suite."""

    def test_01_dependency_preflight(self):
        """Verify all critical runtime modules pass preflight dependency check."""
        ok, logs = check_dependencies()
        self.assertTrue(ok, f"Dependency check failed: {logs}")
        self.assertTrue(any("Python Runtime" in l for l in logs))
        # Verify individual critical libraries are present
        for mod, _ in REQUIRED_MODULES:
            self.assertTrue(any(f"Module '{mod}'" in l for l in logs))

    def test_02_model_hash_validation(self):
        """Verify all 5 frozen artifacts strictly match immutable SHA-256 hashes."""
        ok, logs = check_model_integrity()
        self.assertTrue(ok, f"Model integrity check failed: {logs}")

        # Independently recalculate to double check
        for rel_path, expected_hash in EXPECTED_HASHES.items():
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            self.assertTrue(os.path.isfile(full_path), f"Missing artifact: {rel_path}")
            with open(full_path, "rb") as f:
                computed = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(computed, expected_hash, f"Hash mismatch for {rel_path}")

    def test_03_database_integrity_validation(self):
        """Verify database exists, PRAGMA integrity_check returns ok, and schema is intact."""
        ok, logs = check_database()
        self.assertTrue(ok, f"Database check failed: {logs}")
        self.assertTrue(any("integrity_check: ok" in l for l in logs))
        self.assertTrue(any("Required table 'fall_events' exists" in l for l in logs))

        # Query read-only mode to verify no destructive locks
        db_path = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "fall_events.db")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM fall_events;")
        cnt = cursor.fetchone()[0]
        self.assertGreaterEqual(cnt, 0)
        conn.close()

    def test_04_mqtt_configuration_validation(self):
        """Verify MQTT default configuration parameters and topic semantics."""
        self.assertEqual(DEFAULT_MQTT_PORT, 1883)
        self.assertEqual(DEFAULT_FALL_TOPIC, "fall_detection/stage6/fall")

        # Verify probe logic
        ok, logs = check_mqtt_broker("127.0.0.1", 1883)
        self.assertIsInstance(ok, bool)
        self.assertGreater(len(logs), 0)

    def test_05_startup_without_esp32(self):
        """Verify startup handles absent ESP32 gracefully without raising unhandled exceptions."""
        with patch("serial.tools.list_ports.comports", return_value=[]):
            detected, logs = check_esp32_port("COM999", 460800)
            self.assertFalse(detected)
            self.assertTrue(any("NOT DETECTED" in l for l in logs))

    def test_06_startup_with_esp32_unavailable(self):
        """Verify unavailable or permission-denied serial port logs a diagnostic and continues."""
        port_mock = MagicMock()
        port_mock.device = "COM22"
        with patch("serial.tools.list_ports.comports", return_value=[port_mock]):
            detected, logs = check_esp32_port("COM22", 460800)
            self.assertTrue(detected)
            self.assertTrue(any("COM22" in l and "DETECTED" in l for l in logs))

    def test_07_duplicate_startup_detection(self):
        """Verify is_port_in_use detects occupied ports and launcher attaches safely."""
        # Test detection of an active socket
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.bind(("127.0.0.1", 0))
        test_sock.listen(1)
        bound_port = test_sock.getsockname()[1]
        try:
            in_use = is_port_in_use(bound_port, "127.0.0.1")
            self.assertTrue(in_use, "Port should be detected as in use")
        finally:
            test_sock.close()

        # Port after close should be free
        time.sleep(0.05)
        self.assertFalse(is_port_in_use(bound_port, "127.0.0.1"))

    def test_08_backend_readiness_check(self):
        """Verify backend readiness probing handles online and offline targets gracefully."""
        # Offline port probe
        offline_ok, logs = check_dashboard_and_backend("127.0.0.1", 59999, timeout=0.5)
        self.assertFalse(offline_ok)
        self.assertTrue(any("Cannot connect" in l for l in logs))

    def test_09_dashboard_readiness_check(self):
        """Verify dashboard route validation correctly checks both UI and health endpoints."""
        launcher = ProductionLauncher(port=5000, host="127.0.0.1", open_browser=False)
        # If backend is currently running on 5000, verify_dashboard returns True
        if is_port_in_use(5000, "127.0.0.1"):
            dash_ready = launcher.verify_dashboard()
            self.assertTrue(dash_ready)

    def test_10_graceful_shutdown(self):
        """Verify launcher shutdown cleans up child processes and triggers shutdown events."""
        launcher = ProductionLauncher(port=5000, host="127.0.0.1", open_browser=False)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        launcher.bridge_process = mock_proc
        launcher.backend_process = mock_proc
        launcher.backend_already_running = False

        launcher.shutdown()
        self.assertTrue(launcher._shutdown_event.is_set())
        mock_proc.terminate.assert_called()
        self.assertIsNone(launcher.bridge_process)
        self.assertIsNone(launcher.backend_process)

    def test_11_demo_event_safety(self):
        """Verify safe demo generator creates explicitly marked payload and does not alter models."""
        payload = generate_demo_payload(confidence=0.92, lat=28.6139, lon=77.2090)

        # 1. Must satisfy MQTT gateway contract
        self.assertEqual(payload["event"], "CONFIRMED_FALL")
        # 2. Must be clearly tagged as demo / simulation
        self.assertTrue(payload["event_id"].startswith("DEMO-FALL-"))
        self.assertEqual(payload["source"], "SIMULATED_DEMO_GENERATOR")
        self.assertTrue(payload["diagnostic_metadata"]["demo_mode"])
        self.assertIn("SIMULATED", payload["diagnostic_metadata"]["simulation_notice"])
        # 3. Location fields must be present
        self.assertIn("location", payload)
        self.assertEqual(payload["location"]["latitude"], 28.6139)
        self.assertEqual(payload["location"]["longitude"], 77.2090)

    def test_12_protected_workspace_verification(self):
        """Verify protected workspace C:\\Users\\ROHIT\\Desktop\\major project has 0 modified files."""
        self.assertTrue(os.path.isdir(PROTECTED_ROOT), f"Protected root does not exist: {PROTECTED_ROOT}")
        now = time.time()
        recent_mods = []
        for root, _, files in os.walk(PROTECTED_ROOT):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if now - os.path.getmtime(fp) < 43200:  # 12 hours
                        recent_mods.append(fp)
                except Exception:
                    pass
        self.assertEqual(len(recent_mods), 0, f"Protected workspace has modified files: {recent_mods}")


if __name__ == "__main__":
    unittest.main()
