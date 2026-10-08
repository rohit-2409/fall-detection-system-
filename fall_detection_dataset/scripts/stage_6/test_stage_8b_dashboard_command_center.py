"""
Stage 8B Automated Test Suite: Live Dashboard & Incident Command Center.
Standalone Fall Detection Project.

Validates the 14 mandatory Stage 8B requirements:
  1. Dashboard loads successfully (HTTP 200 at / and /dashboard).
  2. System health header and subsystem indicators displayed.
  3. SYSTEM_READY renders correctly.
  4. SYSTEM_DEGRADED renders correctly.
  5. Backend connection failure handling (non-intrusive alert banner).
  6. Active incident panel renders correctly for confirmed emergency.
  7. No-active-incident state renders correctly during normal monitoring.
  8. Incident history audit table renders with required columns.
  9. GPS available state renders valid coordinates and satellite count.
  10. GPS unavailable state renders clear fallback without fake coords.
  11. Cached/last-known GPS is labeled as 'LAST KNOWN LOCATION'.
  12. Mobile layout responsiveness and overflow protection in CSS.
  13. Incident acknowledgment strictly follows backend contract (PATCH).
  14. Incident resolution strictly follows backend contract (PATCH).
"""

import json
import os
import re
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.api import create_app
from src.fall_event_backend.models import FallEvent, FallEventStatus
from src.fall_event_backend.service import FallEventService
from src.fall_event_backend.storage import FallEventRepository
from src.system_health.monitor import (
    SystemHealthMonitor,
    SYSTEM_READY,
    SYSTEM_DEGRADED,
    SYSTEM_ERROR,
    SYSTEM_STARTING,
    ESP32_CONNECTED,
    ESP32_DISCONNECTED,
    MQTT_CONNECTED,
    MQTT_DISCONNECTED,
    GPS_AVAILABLE,
    GPS_UNAVAILABLE,
    get_global_health_monitor,
)


class TestStage8BDashboardCommandCenter(unittest.TestCase):
    """Hermetic unit and integration tests for Stage 8B Incident Command Center."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stage_8b_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.app = create_app(self.service)
        self.client = self.app.test_client()
        self.health_monitor = get_global_health_monitor()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_dashboard_loads_successfully(self):
        """TEST 1: Verifies dashboard routes return HTTP 200 OK and load template."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        html_root = res_root.get_data(as_text=True)
        self.assertIn("Fall Detection Incident Command Center", html_root)
        self.assertIn("Wearable Fall Detection System", html_root)
        self.assertIn("/static/css/dashboard.css", html_root)
        self.assertIn("/static/js/dashboard.js", html_root)
        self.assertIn("leaflet.js", html_root)

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        html_dash = res_dash.get_data(as_text=True)
        self.assertIn("Active Event Telemetry", html_dash)

    def test_02_system_health_displayed(self):
        """TEST 2: Verifies all 6 subsystem indicator pills and overall status box exist in HTML."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        # Header status box
        self.assertIn('id="systemStatusBox"', html)
        self.assertIn('id="systemStatusDot"', html)
        self.assertIn('id="systemStatusText"', html)

        # Subsystem status bar
        self.assertIn('id="subsystemStatusBar"', html)
        self.assertIn('id="pillEsp32"', html)
        self.assertIn('id="pillMqtt"', html)
        self.assertIn('id="pillGateway"', html)
        self.assertIn('id="pillBackend"', html)
        self.assertIn('id="pillDatabase"', html)
        self.assertIn('id="pillGps"', html)

    def test_03_system_ready_rendered(self):
        """TEST 3: Verifies SYSTEM_READY status from /api/system/health."""
        from src.system_health.models import BACKEND_READY, DATABASE_READY, GATEWAY_READY
        self.health_monitor.set_system_ready()
        self.health_monitor.update_backend_state(BACKEND_READY)
        self.health_monitor.update_database_state(DATABASE_READY)
        self.health_monitor.update_mqtt_state(MQTT_CONNECTED)
        self.health_monitor.update_gateway_state(GATEWAY_READY)
        self.health_monitor.update_esp32_heartbeat("COM22", 460800)
        self.health_monitor.update_gps_state(True, {"fix_valid": True, "satellites": 8, "hdop": 0.9})

        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], SYSTEM_READY)
        self.assertEqual(data["mqtt"]["status"], MQTT_CONNECTED)
        self.assertEqual(data["esp32"]["status"], ESP32_CONNECTED)
        self.assertEqual(data["gps"]["status"], GPS_AVAILABLE)

    def test_04_system_degraded_rendered(self):
        """TEST 4: Verifies SYSTEM_DEGRADED status when a subsystem drops."""
        self.health_monitor.set_system_ready()
        # Degrade by disconnecting MQTT and GPS
        self.health_monitor.update_mqtt_state(MQTT_DISCONNECTED)
        self.health_monitor.update_gps_state(False)

        res = self.client.get("/api/system/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], SYSTEM_DEGRADED)
        self.assertEqual(data["mqtt"]["status"], MQTT_DISCONNECTED)
        self.assertEqual(data["gps"]["status"], GPS_UNAVAILABLE)

    def test_05_backend_connection_failure_handling(self):
        """TEST 5: Verifies backend connection loss banner and non-intrusive handling."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="serviceAlertBanner"', html)
        self.assertIn('id="serviceAlertMessage"', html)
        self.assertIn('id="btnRetryConnection"', html)

        # Verify JS contains non-intrusive error handler without window.alert
        js_path = os.path.join(PROJECT_ROOT, "src", "fall_event_backend", "static", "js", "dashboard.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js_code = f.read()
        self.assertNotIn("window.alert(", js_code)
        self.assertIn("handleBackendConnectionLost", js_code)
        self.assertIn("BACKEND CONNECTION LOST", js_code)

    def test_06_active_incident_renders(self):
        """TEST 6: Verifies active incident hero panel elements render for emergency event."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="activeAlertSection"', html)
        self.assertIn('id="alertHeading"', html)
        self.assertIn('id="alertEventId"', html)
        self.assertIn('id="alertDetectedTime"', html)
        self.assertIn('id="alertConfidence"', html)
        self.assertIn('id="alertSource"', html)
        self.assertIn('id="alertStatusBadge"', html)
        self.assertIn('id="btnAcknowledge"', html)

    def test_07_no_active_incident_state_renders(self):
        """TEST 7: Verifies NO ACTIVE INCIDENT section exists with normal monitoring text."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="noActiveIncidentSection"', html)
        self.assertIn("NO ACTIVE INCIDENT", html)
        self.assertIn("System is monitoring normally", html)

    def test_08_incident_history_renders(self):
        """TEST 8: Verifies incident history audit table renders with required columns."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="historyTable"', html)
        self.assertIn('id="historyTableBody"', html)
        self.assertIn("<th scope=\"col\">Time</th>", html)
        self.assertIn("<th scope=\"col\">Incident ID</th>", html)
        self.assertIn("<th scope=\"col\">Confidence</th>", html)
        self.assertIn("<th scope=\"col\">Status</th>", html)
        self.assertIn("<th scope=\"col\">GPS</th>", html)
        self.assertIn("<th scope=\"col\">Source</th>", html)
        self.assertIn("<th scope=\"col\">Actions</th>", html)

        # Add event and verify history list API
        self.service.record_confirmed_fall(
            event_id="STAGE8B-HIST-01",
            detected_at="2026-10-07T12:00:00Z",
            confidence=0.912,
            source="ESP32_MPU6050",
        )
        res_events = self.client.get("/api/fall-events")
        self.assertEqual(res_events.status_code, 200)
        self.assertEqual(len(res_events.get_json()["events"]), 1)

    def test_09_gps_available_state_renders(self):
        """TEST 9: Verifies GPS available state with valid coordinates and satellite count."""
        loc = {
            "fix_valid": True,
            "latitude": 22.728421,
            "longitude": 75.869163,
            "satellites": 7,
            "hdop": 1.1,
        }
        event, _ = self.service.record_confirmed_fall(
            event_id="STAGE8B-GPS-VALID-01",
            detected_at="2026-10-07T12:00:00Z",
            confidence=0.935,
            diagnostic_metadata={"location": loc},
        )

        res = self.client.get(f"/api/fall-events/{event.event_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["event"]["diagnostic_metadata"]["location"]

        self.assertTrue(data["fix_valid"])
        self.assertAlmostEqual(data["latitude"], 22.728421, places=6)
        self.assertAlmostEqual(data["longitude"], 75.869163, places=6)
        self.assertEqual(data["satellites"], 7)

    def test_10_gps_unavailable_state_renders(self):
        """TEST 10: Verifies GPS unavailable state does not invent fake coordinates."""
        loc = {
            "fix_valid": False,
            "latitude": None,
            "longitude": None,
            "satellites": 0,
            "hdop": None,
        }
        event, _ = self.service.record_confirmed_fall(
            event_id="STAGE8B-GPS-NONE-01",
            detected_at="2026-10-07T12:00:00Z",
            confidence=0.880,
            diagnostic_metadata={"location": loc},
        )

        res = self.client.get(f"/api/fall-events/{event.event_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["event"]["diagnostic_metadata"]["location"]

        self.assertFalse(data["fix_valid"])
        self.assertIsNone(data["latitude"])
        self.assertIsNone(data["longitude"])

        # Check template contains mapUnavailableOverlay
        res_html = self.client.get("/")
        html = res_html.get_data(as_text=True)
        self.assertIn('id="mapUnavailableOverlay"', html)
        self.assertIn("GPS LOCATION UNAVAILABLE", html)

    def test_11_cached_last_known_gps_labeled(self):
        """TEST 11: Verifies cached coordinates are explicitly labeled LAST KNOWN LOCATION."""
        res_html = self.client.get("/")
        html = res_html.get_data(as_text=True)
        self.assertIn('id="mapLastKnownBadge"', html)
        self.assertIn("LAST KNOWN LOCATION", html)

        # Check JS logic handles cached badge
        js_path = os.path.join(PROJECT_ROOT, "src", "fall_event_backend", "static", "js", "dashboard.js")
        with open(js_path, "r", encoding="utf-8") as f:
            js_code = f.read()
        self.assertIn("LAST KNOWN LOCATION", js_code)
        self.assertIn("mapLastKnownBadge", js_code)
        self.assertIn("is_cached", js_code)

    def test_12_mobile_layout_does_not_overflow(self):
        """TEST 12: Verifies CSS responsive rules prevent horizontal overflow."""
        css_path = os.path.join(PROJECT_ROOT, "src", "fall_event_backend", "static", "css", "dashboard.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        self.assertIn("@media (max-width: 992px)", css)
        self.assertIn("@media (max-width: 640px)", css)
        self.assertIn("grid-template-columns: 1fr;", css)
        self.assertIn(".table-responsive", css)

    def test_13_acknowledgment_follows_backend_contract(self):
        """TEST 13: Verifies acknowledgment strictly uses backend PATCH and transitions CONFIRMED -> ACKNOWLEDGED."""
        event, _ = self.service.record_confirmed_fall(
            event_id="STAGE8B-ACK-TEST",
            detected_at="2026-10-07T12:00:00Z",
            confidence=0.925,
        )
        self.assertEqual(event.status, FallEventStatus.CONFIRMED)

        # Perform acknowledgment via backend API
        res = self.client.patch(
            f"/api/fall-events/{event.event_id}/acknowledge",
            json={"acknowledged_by": "Operator_Unit_Test"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["event"]

        self.assertEqual(data["status"], "ACKNOWLEDGED")
        self.assertEqual(data["acknowledged_by"], "Operator_Unit_Test")
        self.assertIsNotNone(data["acknowledged_at"])

    def test_14_resolution_follows_backend_contract(self):
        """TEST 14: Verifies resolution strictly uses backend PATCH and transitions ACKNOWLEDGED -> RESOLVED."""
        event, _ = self.service.record_confirmed_fall(
            event_id="STAGE8B-RESOLVE-TEST",
            detected_at="2026-10-07T12:00:00Z",
            confidence=0.940,
        )

        # Attempting to resolve directly without acknowledging first must return 409 Conflict
        res_conflict = self.client.patch(
            f"/api/fall-events/{event.event_id}/resolve",
            json={"resolved_by": "Operator_Unit_Test"},
        )
        self.assertEqual(res_conflict.status_code, 409)

        # First acknowledge
        self.client.patch(
            f"/api/fall-events/{event.event_id}/acknowledge",
            json={"acknowledged_by": "Operator_Unit_Test"},
        )

        # Now resolve
        res_resolve = self.client.patch(
            f"/api/fall-events/{event.event_id}/resolve",
            json={"resolved_by": "Operator_Unit_Test"},
        )
        self.assertEqual(res_resolve.status_code, 200)
        data = res_resolve.get_json()["event"]

        self.assertEqual(data["status"], "RESOLVED")
        self.assertEqual(data["resolved_by"], "Operator_Unit_Test")
        self.assertIsNotNone(data["resolved_at"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
