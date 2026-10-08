"""
Stage 8A Automated Test Suite: Production Startup, Health Monitoring & Recovery.
Standalone Fall Detection System.

Covers:
  1. System health endpoint GET /api/system/health returns HTTP 200 with complete diagnostics.
  2. Legacy /api/health endpoint remains completely unbroken and functional.
  3. Database health monitoring detects both normal READY and simulated ERROR states safely.
  4. MQTT connection monitoring tracks CONNECTED, DISCONNECTED, and RECONNECTING states.
  5. ESP32 serial monitoring distinguishes CONNECTED, STALE (elapsed > 3s), and DISCONNECTED.
  6. GPS telemetry monitoring distinguishes AVAILABLE and UNAVAILABLE without fabricating coordinates.
  7. Composite overall status correctly computes SYSTEM_READY, SYSTEM_DEGRADED, and SYSTEM_ERROR.
  8. Health JSON response schema adheres to specification with valid uptime and ISO timestamps.
  9. Graceful startup operates in decoupled mode without requiring physical ESP32 on COM22.
  10. Graceful shutdown cleanly halts gateway and HTTP server without resource leaks.

SAFETY:
  - 100% Mock/Software-in-the-loop.
  - Zero COM22 hardware requirements.
  - Zero synthetic fall publications to production MQTT.
  - Zero database resetting.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.api import create_app
from src.notifications.notification_listener import create_notifying_app
from src.system_health.logger import OperationalLogger
from src.system_health.models import (
    BACKEND_ERROR,
    BACKEND_READY,
    DATABASE_ERROR,
    DATABASE_READY,
    ESP32_CONNECTED,
    ESP32_DISCONNECTED,
    ESP32_STALE,
    GATEWAY_DISCONNECTED,
    GATEWAY_READY,
    GPS_AVAILABLE,
    GPS_UNAVAILABLE,
    MQTT_CONNECTED,
    MQTT_DISCONNECTED,
    MQTT_RECONNECTING,
    SYSTEM_DEGRADED,
    SYSTEM_ERROR,
    SYSTEM_READY,
    SYSTEM_STARTING,
)
from src.system_health.monitor import SystemHealthMonitor, get_global_health_monitor


class TestStage8AHealthMonitoring(unittest.TestCase):
    """Hermetic unit and integration tests for Stage 8A Health Monitoring."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_health.db")

        # Create valid SQLite schema
        import sqlite3
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

        self.monitor = SystemHealthMonitor(db_path=self.test_db_path)
        self.app = create_notifying_app()
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: System Health Endpoint Exists and Returns HTTP 200
    # -------------------------------------------------------------------------
    def test_01_system_health_endpoint_exists_and_returns_200(self):
        """GET /api/system/health returns HTTP 200 and valid JSON body."""
        resp = self.client.get("/api/system/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertIn("backend", data)
        self.assertIn("database", data)
        self.assertIn("mqtt", data)
        self.assertIn("gateway", data)
        self.assertIn("esp32", data)
        self.assertIn("gps", data)
        self.assertIn("uptime_seconds", data)
        self.assertIn("timestamp", data)

    # -------------------------------------------------------------------------
    # Test 2: Existing /api/health Endpoint Still Works Unbroken
    # -------------------------------------------------------------------------
    def test_02_legacy_api_health_endpoint_still_works_unbroken(self):
        """Legacy GET /api/health retains exact backward-compatible structure."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "fall_event_backend")
        self.assertEqual(data.get("stage"), "5B")
        self.assertEqual(data.get("emergency_notifications"), "DISABLED")

    # -------------------------------------------------------------------------
    # Test 3: Database Health Probing
    # -------------------------------------------------------------------------
    def test_03_database_health_reporting_and_detection(self):
        """Verifies database READY check and ERROR detection on missing file."""
        status, err = self.monitor.check_database_health()
        self.assertEqual(status, DATABASE_READY)
        self.assertIsNone(err)

        # Simulate corrupt/missing database
        bad_monitor = SystemHealthMonitor(db_path=os.path.join(self.temp_dir, "nonexistent.db"))
        bad_status, bad_err = bad_monitor.check_database_health()
        self.assertEqual(bad_status, DATABASE_ERROR)
        self.assertIsNotNone(bad_err)

    # -------------------------------------------------------------------------
    # Test 4: MQTT Connection State Tracking
    # -------------------------------------------------------------------------
    def test_04_mqtt_disconnected_and_reconnecting_state(self):
        """Tracks MQTT broker CONNECTED, RECONNECTING, and DISCONNECTED states."""
        self.monitor.update_mqtt_state(MQTT_CONNECTED, host="127.0.0.1", port=1883)
        self.assertEqual(self.monitor._mqtt_state, MQTT_CONNECTED)

        self.monitor.update_mqtt_state(MQTT_RECONNECTING)
        self.assertEqual(self.monitor._mqtt_state, MQTT_RECONNECTING)

        self.monitor.update_mqtt_state(MQTT_DISCONNECTED)
        self.assertEqual(self.monitor._mqtt_state, MQTT_DISCONNECTED)

    # -------------------------------------------------------------------------
    # Test 5: ESP32 Serial Monitoring (Connected, Stale, Disconnected)
    # -------------------------------------------------------------------------
    def test_05_esp32_connected_stale_and_disconnected_states(self):
        """Distinguishes live heartbeat (CONNECTED), timeout (STALE), and DISCONNECTED."""
        # 1. Live heartbeat -> CONNECTED
        self.monitor.update_esp32_heartbeat(port="COM22", baud=460800)
        self.assertEqual(self.monitor.assess_esp32_status(), ESP32_CONNECTED)

        # 2. Simulate > 3.0s elapsed without heartbeat -> STALE
        self.monitor._last_esp32_heartbeat = time.time() - 4.5
        self.assertEqual(self.monitor.assess_esp32_status(), ESP32_STALE)

        # 3. Explicit disconnect -> DISCONNECTED
        self.monitor.mark_esp32_disconnected("USB unplugged")
        self.assertEqual(self.monitor.assess_esp32_status(), ESP32_DISCONNECTED)

    # -------------------------------------------------------------------------
    # Test 6: GPS Telemetry Monitoring
    # -------------------------------------------------------------------------
    def test_06_gps_available_and_unavailable_states(self):
        """Monitors GPS satellite lock availability without creating fake coordinates."""
        self.monitor.update_gps_state(True, {"latitude": 22.72, "longitude": 75.86, "satellites": 6, "fix_valid": True})
        self.assertEqual(self.monitor._gps_state, GPS_AVAILABLE)
        self.assertTrue(self.monitor._last_gps_metadata["fix_valid"])

        self.monitor.update_gps_state(False, {"fix_valid": False})
        self.assertEqual(self.monitor._gps_state, GPS_UNAVAILABLE)

    # -------------------------------------------------------------------------
    # Test 7: Overall Status Assessment (Ready, Degraded, Error)
    # -------------------------------------------------------------------------
    def test_07_overall_degraded_and_error_states(self):
        """Verifies overall status transitions between READY, DEGRADED, and ERROR."""
        # Baseline healthy state
        self.monitor.update_backend_state(BACKEND_READY)
        self.monitor.update_database_state(DATABASE_READY)
        self.monitor.update_mqtt_state(MQTT_CONNECTED)
        self.monitor.update_gateway_state(GATEWAY_READY)
        self.monitor.update_esp32_heartbeat(port="COM22", baud=460800)
        self.monitor.update_gps_state(True, {"fix_valid": True})

        self.assertEqual(self.monitor.compute_overall_status(), SYSTEM_READY)

        # Degraded if ESP32 goes stale/disconnected
        self.monitor._last_esp32_heartbeat = time.time() - 5.0
        self.assertEqual(self.monitor.compute_overall_status(), SYSTEM_DEGRADED)

        # Restore ESP32, but GPS lost -> still DEGRADED
        self.monitor.update_esp32_heartbeat(port="COM22", baud=460800)
        self.monitor.update_gps_state(False)
        self.assertEqual(self.monitor.compute_overall_status(), SYSTEM_DEGRADED)

        # Critical failure: Database error -> SYSTEM_ERROR
        self.monitor.update_database_state(DATABASE_ERROR, "Disk I/O error")
        self.assertEqual(self.monitor.compute_overall_status(), SYSTEM_ERROR)

    # -------------------------------------------------------------------------
    # Test 8: Health Response Schema, Timestamp, and Uptime
    # -------------------------------------------------------------------------
    def test_08_health_response_schema_timestamp_and_uptime(self):
        """Validates JSON structure, ISO 8601 timestamps, and numeric uptime."""
        status_dict = self.monitor.get_status_dict()
        self.assertIsInstance(status_dict["uptime_seconds"], (int, float))
        self.assertGreaterEqual(status_dict["uptime_seconds"], 0.0)

        # Validate ISO 8601 timestamp
        ts = status_dict["timestamp"]
        parsed_dt = datetime.datetime.fromisoformat(ts)
        self.assertIsNotNone(parsed_dt)

        # Subsystems are dictionaries with status keys
        for key in ("backend", "database", "mqtt", "gateway", "esp32", "gps"):
            self.assertIn("status", status_dict[key])

    # -------------------------------------------------------------------------
    # Test 9: Graceful Startup without Physical ESP32
    # -------------------------------------------------------------------------
    def test_09_graceful_startup_without_esp32_does_not_crash(self):
        """Backend startup in decoupled mode succeeds even when hardware is absent."""
        from scripts.stage_6.run_full_backend import UnifiedBackendServer

        server = UnifiedBackendServer(
            host="127.0.0.1",
            port=55123,  # non-colliding test port
            enable_gateway=False,  # decoupled mode
        )
        try:
            with patch("serial.tools.list_ports.comports", return_value=[]):
                started = server.start(timeout=5.0)
                self.assertTrue(started)
                # ESP32 is reported as disconnected without crashing the server
                self.assertEqual(server.health_monitor.assess_esp32_status(), ESP32_DISCONNECTED)
                self.assertTrue(server._is_running)
        finally:
            server.stop()

    # -------------------------------------------------------------------------
    # Test 10: Graceful Shutdown Releases Resources Cleanly
    # -------------------------------------------------------------------------
    def test_10_graceful_shutdown_releases_resources(self):
        """Server stops cleanly, disengages gateway, and shuts down HTTP server."""
        from scripts.stage_6.run_full_backend import UnifiedBackendServer

        server = UnifiedBackendServer(
            host="127.0.0.1",
            port=55124,
            enable_gateway=False,
        )
        started = server.start(timeout=5.0)
        self.assertTrue(started)

        # Stop server
        server.stop()
        self.assertFalse(server._is_running)
        self.assertIsNone(server.http_server)
        self.assertIsNone(server.gateway)


if __name__ == "__main__":
    unittest.main()
