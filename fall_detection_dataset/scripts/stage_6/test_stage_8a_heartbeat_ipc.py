"""
Stage 8A Heartbeat IPC Automated Test Suite.
Validates the cross-process heartbeat communication between Stage 6L and Flask backend.

Covers:
  Test A: Heartbeat endpoint exists and responds to POST /api/system/heartbeat.
  Test B: Valid ESP32 heartbeat updates monitor state to CONNECTED.
  Test C: GPS available heartbeat updates GPS state to AVAILABLE with metadata.
  Test D: GPS unavailable heartbeat sets GPS state to UNAVAILABLE without fabricating coords.
  Test E: Malformed heartbeat payloads are safely rejected with HTTP 400.
  Test F: Stage 6L heartbeat sender handles backend unavailability gracefully.
  Test G: Full inference and confirmation continues seamlessly even if heartbeat fails.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_bridge import (
    HardwareBridgeHeartbeatSender,
    run_stage_6L_hardware_bridge,
)
from src.fall_event_backend.api import create_app
from src.fall_confirmation.stage_6L_gps_cache import GpsTelemetryCache
from src.system_health.models import (
    ESP32_CONNECTED,
    GPS_AVAILABLE,
    GPS_UNAVAILABLE,
    SYSTEM_READY,
    SYSTEM_DEGRADED,
)
from src.system_health.monitor import SystemHealthMonitor, get_global_health_monitor


class TestStage8AHeartbeatIpc(unittest.TestCase):
    """Hermetic unit tests for the additive Stage 8A IPC heartbeat subsystem."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_events.db")

        # Initialize minimal SQLite schema for health monitor
        conn = sqlite3.connect(self.test_db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE fall_events (
                event_id TEXT PRIMARY KEY,
                detected_at TEXT NOT NULL,
                occurred_at TEXT,
                session_id TEXT,
                confidence REAL,
                source TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                diagnostic_metadata TEXT,
                acknowledged_at TEXT,
                acknowledged_by TEXT,
                cancelled_at TEXT,
                cancelled_by TEXT,
                cancellation_reason TEXT,
                resolved_at TEXT,
                resolved_by TEXT
            )
        """)
        conn.commit()
        conn.close()

        self.app = create_app()
        self.client = self.app.test_client()
        self.monitor = get_global_health_monitor()
        self._orig_db_path = self.monitor.db_path
        self._orig_esp32_state = self.monitor._esp32_state
        self._orig_last_esp32_heartbeat = self.monitor._last_esp32_heartbeat
        self._orig_esp32_port = self.monitor._esp32_port
        self._orig_esp32_baud = self.monitor._esp32_baud
        self._orig_gps_state = self.monitor._gps_state
        self._orig_last_gps_metadata = self.monitor._last_gps_metadata
        self.monitor.db_path = self.test_db_path

    def tearDown(self):
        self.monitor.db_path = self._orig_db_path
        self.monitor._esp32_state = self._orig_esp32_state
        self.monitor._last_esp32_heartbeat = self._orig_last_esp32_heartbeat
        self.monitor._esp32_port = self._orig_esp32_port
        self.monitor._esp32_baud = self._orig_esp32_baud
        self.monitor._gps_state = self._orig_gps_state
        self.monitor._last_gps_metadata = self._orig_last_gps_metadata
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A: Heartbeat Endpoint Exists and Responds HTTP 200
    # -------------------------------------------------------------------------
    def test_a_heartbeat_endpoint_returns_200(self):
        """Test A: POST /api/system/heartbeat exists and returns HTTP 200."""
        payload = {
            "port": "COM22",
            "baud": 460800,
            "gps_available": True,
            "gps_metadata": {
                "fix_valid": True,
                "latitude": 22.728523,
                "longitude": 75.869183,
                "satellites": 9,
                "hdop": 0.85,
            },
        }
        resp = self.client.post("/api/system/heartbeat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")

    # -------------------------------------------------------------------------
    # Test B: Valid ESP32 Heartbeat Updates Monitor State
    # -------------------------------------------------------------------------
    def test_b_valid_esp32_heartbeat_updates_state(self):
        """Test B: Heartbeat sets ESP32 to CONNECTED and records recent timestamp."""
        payload = {"port": "COM22", "baud": 460800}
        resp = self.client.post("/api/system/heartbeat", json=payload)
        self.assertEqual(resp.status_code, 200)

        health = self.monitor.get_status_dict()
        self.assertEqual(health["esp32"]["status"], ESP32_CONNECTED)
        self.assertEqual(health["esp32"]["port"], "COM22")
        self.assertEqual(health["esp32"]["baud"], 460800)
        self.assertIsNotNone(health["esp32"]["last_heartbeat_s"])
        self.assertLess(health["esp32"]["last_heartbeat_s"], 1.0)

    # -------------------------------------------------------------------------
    # Test C: GPS Available Heartbeat Updates GPS State
    # -------------------------------------------------------------------------
    def test_c_gps_available_heartbeat_updates_state(self):
        """Test C: Valid GPS telemetry sets GPS status to AVAILABLE with metadata."""
        payload = {
            "port": "COM22",
            "baud": 460800,
            "gps_available": True,
            "gps_metadata": {
                "fix_valid": True,
                "latitude": 22.728523,
                "longitude": 75.869183,
                "satellites": 9,
                "hdop": 0.85,
            },
        }
        resp = self.client.post("/api/system/heartbeat", json=payload)
        self.assertEqual(resp.status_code, 200)

        health = self.monitor.get_status_dict()
        self.assertEqual(health["gps"]["status"], GPS_AVAILABLE)
        self.assertTrue(health["gps"]["fix_valid"])
        self.assertEqual(health["gps"]["satellites"], 9)

    # -------------------------------------------------------------------------
    # Test D: GPS Unavailable Heartbeat Updates State Safely
    # -------------------------------------------------------------------------
    def test_d_gps_unavailable_heartbeat_updates_state(self):
        """Test D: GPS unavailable heartbeat safely reflects UNAVAILABLE without coords."""
        payload = {
            "port": "COM22",
            "baud": 460800,
            "gps_available": False,
            "gps_metadata": {
                "fix_valid": False,
                "latitude": None,
                "longitude": None,
                "satellites": 0,
                "hdop": None,
            },
        }
        resp = self.client.post("/api/system/heartbeat", json=payload)
        self.assertEqual(resp.status_code, 200)

        health = self.monitor.get_status_dict()
        self.assertEqual(health["gps"]["status"], GPS_UNAVAILABLE)
        self.assertFalse(health["gps"]["fix_valid"])

    # -------------------------------------------------------------------------
    # Test E: Malformed Heartbeat Payloads Return HTTP 400
    # -------------------------------------------------------------------------
    def test_e_malformed_heartbeat_payload(self):
        """Test E: Malformed requests return HTTP 400."""
        # Non-dict / non-JSON
        resp = self.client.post("/api/system/heartbeat", data="not json", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

        # Invalid port type
        resp = self.client.post("/api/system/heartbeat", json={"port": 12345})
        self.assertEqual(resp.status_code, 400)

        # Invalid gps_available type
        resp = self.client.post("/api/system/heartbeat", json={"gps_available": "yes"})
        self.assertEqual(resp.status_code, 400)

        # Invalid gps_metadata type
        resp = self.client.post("/api/system/heartbeat", json={"gps_metadata": ["not", "a", "dict"]})
        self.assertEqual(resp.status_code, 400)

    # -------------------------------------------------------------------------
    # Test F: Backend Unavailable Behavior in Stage 6L
    # -------------------------------------------------------------------------
    def test_f_backend_unavailable_behavior_in_stage_6l(self):
        """Test F: Sender handles offline backend cleanly without raising or crashing."""
        # Point to an unreachable port
        sender = HardwareBridgeHeartbeatSender(
            backend_url="http://127.0.0.1:59999",
            interval_seconds=0.05,
            timeout_seconds=0.2,
        )
        sender.start()
        try:
            # Enqueue heartbeat to dead backend
            sender.queue_heartbeat("COM22", 460800, False, {})
            # Wait for worker loop to attempt dispatch
            time.sleep(0.3)
            # Worker must still be alive and not crashed
            self.assertTrue(sender._worker_thread.is_alive())
        finally:
            sender.stop()

    # -------------------------------------------------------------------------
    # Test G: Confirmation/Inference Continues Even if Heartbeat Request Fails
    # -------------------------------------------------------------------------
    def test_g_inference_continues_when_heartbeat_fails(self):
        """Test G: Mock hardware bridge completes inference even when backend is unreachable."""
        # Run bridge with backend pointing to an unreachable port
        res = run_stage_6L_hardware_bridge(
            port="COM22",
            max_windows=4,
            dry_run=True,
            mock_mode=True,
            mock_scenario="stationary",
            backend_url="http://127.0.0.1:59999",
        )
        self.assertEqual(res["total_windows"], 4)
        self.assertGreater(res["total_samples"], 0)
        self.assertEqual(res["confirmed_falls"], 0)


if __name__ == "__main__":
    unittest.main()
