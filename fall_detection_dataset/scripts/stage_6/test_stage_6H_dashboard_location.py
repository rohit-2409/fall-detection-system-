"""
Stage 6H Automated Test Suite: Dashboard Alert + GPS Location Display.
Standalone Fall Detection Project.

Validates:
  1. Existing dashboard loads successfully (HTTP 200 at / and /dashboard).
  2. Existing active fall alert panel elements remain present.
  3. Existing response window countdown elements remain present.
  4. Stage 6H Location Telemetry section elements exist in the DOM.
  5. Valid GPS fix telemetry data representation (Latitude, Longitude, Satellites, HDOP).
  6. No-fix GPS telemetry data representation (safely shows 'Not available', no fake coords).
  7. Missing location metadata handled gracefully without crashing.
  8. Map action URL generation uses exact coordinates and does NOT use browser geolocation.
  9. Existing user controls remain intact (Acknowledge, Cancel, Historical events).
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


class TestStage6HDashboardLocation(unittest.TestCase):
    """Test suite validating Stage 6H GPS location display and emergency dashboard integrity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stage_6h_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.app = create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_dashboard_loads_successfully(self):
        """TEST 1: Verifies existing dashboard routes return HTTP 200 OK."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("Wearable Fall Detection System", res_root.get_data(as_text=True))

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn("Active Event Telemetry", res_dash.get_data(as_text=True))

    def test_02_existing_fall_alert_panel_present(self):
        """TEST 2: Verifies core active alert banner structure is intact."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="activeAlertSection"', html)
        self.assertIn('id="alertEventId"', html)
        self.assertIn('id="alertDetectedTime"', html)
        self.assertIn('id="alertConfidence"', html)
        self.assertIn('id="alertSource"', html)
        self.assertIn('id="alertSessionId"', html)

    def test_03_existing_countdown_elements_present(self):
        """TEST 3: Verifies countdown progress and timer value elements are intact."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        self.assertIn('id="alertCountdownText"', html)
        self.assertIn('id="countdownProgressBar"', html)
        self.assertIn("Response Window:", html)

    def test_04_location_telemetry_section_exists(self):
        """TEST 4: Verifies Stage 6H location telemetry DOM elements exist in HTML template."""
        res = self.client.get("/")
        html = res.get_data(as_text=True)

        # Alert panel location elements
        self.assertIn('id="alertLocationSection"', html)
        self.assertIn('id="alertGpsStatusBadge"', html)
        self.assertIn('id="alertLatitude"', html)
        self.assertIn('id="alertLongitude"', html)
        self.assertIn('id="alertSatellites"', html)
        self.assertIn('id="alertHdop"', html)
        self.assertIn('id="alertMapLink"', html)

        # Status card location elements
        self.assertIn('id="viewGpsStatusBadge"', html)
        self.assertIn('id="viewLatitude"', html)
        self.assertIn('id="viewLongitude"', html)
        self.assertIn('id="viewSatellitesHdop"', html)
        self.assertIn('id="viewMapRow"', html)
        self.assertIn('id="viewMapLink"', html)

    def test_05_valid_gps_telemetry_api_and_representation(self):
        """TEST 5: Tests that an event with a valid 3D GPS fix stores and returns complete GPS metadata."""
        test_loc = {
            "fix_valid": True,
            "latitude": 22.728439,
            "longitude": 75.869204,
            "satellites": 10,
            "hdop": 0.79,
        }
        event, is_new = self.service.record_confirmed_fall(
            event_id="STAGE6H-TEST-VALID-GPS",
            detected_at="2026-10-06T15:30:20.123Z",
            confidence=0.9412,
            source="ESP32_MPU6050",
            diagnostic_metadata={"location": test_loc, "device_id": "ESP32-STAGE6-GPS"},
        )
        self.assertTrue(is_new)

        res = self.client.get(f"/api/fall-events/{event.event_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        loc = data["event"]["diagnostic_metadata"]["location"]

        self.assertTrue(loc["fix_valid"])
        self.assertAlmostEqual(loc["latitude"], 22.728439, places=6)
        self.assertAlmostEqual(loc["longitude"], 75.869204, places=6)
        self.assertEqual(loc["satellites"], 10)
        self.assertAlmostEqual(loc["hdop"], 0.79, places=2)

    def test_06_no_gps_fix_representation(self):
        """TEST 6: Tests that an event with fix_valid=false safely represents no-fix without fake coordinates."""
        no_fix_loc = {
            "fix_valid": False,
            "latitude": None,
            "longitude": None,
            "satellites": 0,
            "hdop": None,
        }
        event, is_new = self.service.record_confirmed_fall(
            event_id="STAGE6H-TEST-NO-FIX",
            detected_at="2026-10-06T15:30:20.123Z",
            confidence=0.8800,
            source="ESP32_MPU6050",
            diagnostic_metadata={"location": no_fix_loc},
        )
        self.assertTrue(is_new)

        res = self.client.get(f"/api/fall-events/{event.event_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        loc = data["event"]["diagnostic_metadata"]["location"]

        self.assertFalse(loc["fix_valid"])
        self.assertIsNone(loc["latitude"])
        self.assertIsNone(loc["longitude"])
        self.assertEqual(loc["satellites"], 0)
        self.assertIsNone(loc["hdop"])

    def test_07_missing_location_metadata_resilience(self):
        """TEST 7: Tests that events without location metadata load safely without breaking the API or UI."""
        # Event with no diagnostic metadata at all
        event, is_new = self.service.record_confirmed_fall(
            event_id="STAGE6H-TEST-MISSING-LOC",
            detected_at="2026-10-06T15:30:20.123Z",
            confidence=0.9100,
            source="ESP32_MPU6050",
            diagnostic_metadata=None,
        )
        self.assertTrue(is_new)

        res = self.client.get(f"/api/fall-events/{event.event_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["event"]["diagnostic_metadata"])

    def test_08_map_action_security_and_no_browser_geolocation(self):
        """TEST 8: Verifies map action links use coordinates and that browser geolocation is NOT utilized."""
        res_js = self.client.get("/static/js/dashboard.js")
        self.assertEqual(res_js.status_code, 200)
        js_code = res_js.get_data(as_text=True)

        # MUST NOT contain navigator.geolocation or location permission APIs
        self.assertNotIn("navigator.geolocation", js_code)
        self.assertNotIn("getCurrentPosition", js_code)
        self.assertNotIn("watchPosition", js_code)

        # Must construct map URL using encoded latitude and longitude
        self.assertIn("https://www.google.com/maps?q=", js_code)
        self.assertIn("encodeURIComponent", js_code)

    def test_09_existing_controls_remain_functional(self):
        """TEST 9: Verifies existing controls (acknowledge, cancel, historical log) remain functional."""
        event, _ = self.service.record_confirmed_fall(
            event_id="STAGE6H-TEST-CONTROLS",
            detected_at="2026-10-06T15:30:20.123Z",
            confidence=0.9500,
        )

        # Acknowledge endpoint
        res_ack = self.client.patch(
            f"/api/fall-events/{event.event_id}/acknowledge",
            json={"acknowledged_by": "TestUser"},
        )
        self.assertEqual(res_ack.status_code, 200)
        self.assertEqual(res_ack.get_json()["event"]["status"], "ACKNOWLEDGED")

        # Second event for Cancel endpoint
        event2, _ = self.service.record_confirmed_fall(
            event_id="STAGE6H-TEST-CANCEL",
            detected_at="2026-10-06T15:30:25.123Z",
            confidence=0.9200,
        )
        res_cancel = self.client.patch(
            f"/api/fall-events/{event2.event_id}/cancel",
            json={"cancelled_by": "TestUser", "reason": "Verified safe"},
        )
        self.assertEqual(res_cancel.status_code, 200)
        self.assertEqual(res_cancel.get_json()["event"]["status"], "CANCELLED")

        # History list
        res_list = self.client.get("/api/fall-events")
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(len(res_list.get_json()["events"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
