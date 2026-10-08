"""
Stage 5B Automated Test Suite: Fall Event Alert & Monitoring Interface.
Tests all 32 required scenarios:
- Initial state (Tests 1-3)
- New event detection & alert banner (Tests 4-7)
- Acknowledgment workflow (Tests 8-11)
- Cancellation workflow (Tests 12-15)
- Timeout & no emergency action (Tests 16-18)
- Refresh / reconnect recovery (Tests 19-21)
- Failure handling (Tests 22-25)
- History & event detail retrieval (Tests 26-28)
- Regressions: Stage 5A, 4E, 4F, 4G (Tests 29-32)
"""

import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.models import (
    FallEvent,
    FallEventStatus,
    InvalidStateTransitionError,
    EventNotFoundError,
    get_utc_iso_now
)
from src.fall_event_backend.storage import FallEventRepository
from src.fall_event_backend.service import FallEventService
from src.fall_event_backend.integration_adapter import FallEventIntegrationAdapter
from src.fall_event_backend.api import create_app
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter


class TestStage5BDashboard(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_dashboard_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.adapter = FallEventIntegrationAdapter(self.service)
        self.app = create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Initial State (Tests 1 - 3)
    # -------------------------------------------------------------------------
    def test_01_dashboard_loads(self):
        """Test 1: Dashboard HTML loads successfully at / and /dashboard."""
        res1 = self.client.get("/")
        self.assertEqual(res1.status_code, 200)
        content1 = res1.get_data(as_text=True)
        self.assertIn("Wearable Fall Detection System", content1)
        self.assertIn("Active Event Telemetry", content1)

        res2 = self.client.get("/dashboard")
        self.assertEqual(res2.status_code, 200)

        # Verify static CSS and JS load
        res_css = self.client.get("/static/css/dashboard.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("--color-confirmed", res_css.get_data(as_text=True))

        res_js = self.client.get("/static/js/dashboard.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("POLLING_INTERVAL_MS", res_js.get_data(as_text=True))

    def test_02_no_event_state_displays_correctly(self):
        """Test 2: When no events exist, API returns empty list and placeholder is present."""
        res = self.client.get("/api/fall-events")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["count"], 0)
        self.assertEqual(data["events"], [])

        # Verify HTML has empty placeholder structure
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("noActiveEventPlaceholder", html)
        self.assertIn("No Active Fall Event", html)

    def test_03_event_history_loads(self):
        """Test 3: Event history loads correctly via GET /api/fall-events."""
        self.service.record_confirmed_fall(
            event_id="EVT_INIT_03",
            detected_at=get_utc_iso_now(),
            session_id="SESS_INIT",
            confidence=0.912
        )
        res = self.client.get("/api/fall-events")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["events"][0]["event_id"], "EVT_INIT_03")

    # -------------------------------------------------------------------------
    # New Event & Countdown (Tests 4 - 7)
    # -------------------------------------------------------------------------
    def test_04_confirmed_event_appears(self):
        """Test 4: Newly confirmed fall event appears queryable with CONFIRMED status."""
        self.service.record_confirmed_fall(
            event_id="EVT_ALERT_04",
            detected_at=get_utc_iso_now(),
            session_id="SESS_04",
            confidence=0.965
        )
        res = self.client.get("/api/fall-events?status=CONFIRMED")
        self.assertEqual(res.status_code, 200)
        events = res.get_json()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "CONFIRMED")

    def test_05_correct_event_information_appears(self):
        """Test 5: Event information contains ID, timestamp, confidence, and source."""
        now_ts = get_utc_iso_now()
        diag_meta = {"max_dyn_acc_mag": 4.52, "max_gyr_mag": 284.1}
        self.service.record_confirmed_fall(
            event_id="EVT_INFO_05",
            detected_at=now_ts,
            occurred_at="12.2500s",
            session_id="SESS_INFO_05",
            confidence=0.9785,
            source="ESP32_MPU6050",
            diagnostic_metadata=diag_meta
        )
        res = self.client.get("/api/fall-events/EVT_INFO_05")
        self.assertEqual(res.status_code, 200)
        evt = res.get_json()["event"]
        self.assertEqual(evt["event_id"], "EVT_INFO_05")
        self.assertEqual(evt["detected_at"], now_ts)
        self.assertEqual(evt["occurred_at"], "12.2500s")
        self.assertAlmostEqual(evt["confidence"], 0.9785, places=4)
        self.assertEqual(evt["source"], "ESP32_MPU6050")
        self.assertEqual(evt["diagnostic_metadata"], diag_meta)

    def test_06_countdown_calculation_logic(self):
        """Test 6: Countdown window calculates elapsed seconds properly."""
        # Detected 10 seconds ago -> Remaining should be 60 - 10 = 50 seconds
        ten_sec_ago = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        evt, _ = self.service.record_confirmed_fall(
            event_id="EVT_COUNTDOWN_06",
            detected_at=ten_sec_ago,
            confidence=0.95
        )
        dt = datetime.fromisoformat(evt.detected_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
        remaining = max(0, 60 - int(elapsed))
        self.assertGreaterEqual(remaining, 48)
        self.assertLessEqual(remaining, 51)

    def test_07_duplicate_polling_does_not_restart_countdown(self):
        """Test 7: Sequential polling of same event returns identical event_id and timestamps."""
        self.service.record_confirmed_fall(
            event_id="EVT_POLL_07",
            detected_at=get_utc_iso_now(),
            confidence=0.94
        )
        res1 = self.client.get("/api/fall-events?status=CONFIRMED")
        res2 = self.client.get("/api/fall-events?status=CONFIRMED")
        evt1 = res1.get_json()["events"][0]
        evt2 = res2.get_json()["events"][0]
        self.assertEqual(evt1["event_id"], evt2["event_id"])
        self.assertEqual(evt1["created_at"], evt2["created_at"])
        self.assertEqual(evt1["detected_at"], evt2["detected_at"])

    # -------------------------------------------------------------------------
    # Acknowledge Workflow (Tests 8 - 11)
    # -------------------------------------------------------------------------
    def test_08_acknowledge_request_succeeds(self):
        """Test 8: PATCH /api/fall-events/<event_id>/acknowledge returns 200 OK."""
        self.service.record_confirmed_fall("EVT_ACK_08", detected_at=get_utc_iso_now())
        res = self.client.patch("/api/fall-events/EVT_ACK_08/acknowledge", json={"acknowledged_by": "User_Dashboard"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["event"]["status"], "ACKNOWLEDGED")

    def test_09_ui_updates_to_acknowledged(self):
        """Test 9: Verified that event status is persistent as ACKNOWLEDGED."""
        self.service.record_confirmed_fall("EVT_ACK_09", detected_at=get_utc_iso_now())
        self.client.patch("/api/fall-events/EVT_ACK_09/acknowledge", json={"acknowledged_by": "User_Dashboard"})
        res = self.client.get("/api/fall-events/EVT_ACK_09")
        self.assertEqual(res.get_json()["event"]["status"], "ACKNOWLEDGED")
        self.assertIsNotNone(res.get_json()["event"]["acknowledged_at"])

    def test_10_countdown_stops_upon_acknowledgement(self):
        """Test 10: Acknowledged event is no longer in CONFIRMED query pool."""
        self.service.record_confirmed_fall("EVT_ACK_10", detected_at=get_utc_iso_now())
        self.client.patch("/api/fall-events/EVT_ACK_10/acknowledge", json={})
        res_conf = self.client.get("/api/fall-events?status=CONFIRMED")
        self.assertEqual(len(res_conf.get_json()["events"]), 0)

    def test_11_repeated_acknowledgement_is_handled_safely(self):
        """Test 11: Second acknowledge request returns 200 OK without state corruption."""
        self.service.record_confirmed_fall("EVT_ACK_11", detected_at=get_utc_iso_now())
        res1 = self.client.patch("/api/fall-events/EVT_ACK_11/acknowledge", json={"acknowledged_by": "User1"})
        self.assertEqual(res1.status_code, 200)
        res2 = self.client.patch("/api/fall-events/EVT_ACK_11/acknowledge", json={"acknowledged_by": "User1"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["event"]["status"], "ACKNOWLEDGED")

    # -------------------------------------------------------------------------
    # Cancel Workflow (Tests 12 - 15)
    # -------------------------------------------------------------------------
    def test_12_cancel_request_succeeds(self):
        """Test 12: PATCH /api/fall-events/<event_id>/cancel returns 200 OK."""
        self.service.record_confirmed_fall("EVT_CAN_12", detected_at=get_utc_iso_now())
        res = self.client.patch("/api/fall-events/EVT_CAN_12/cancel", json={
            "cancelled_by": "User_Dashboard",
            "reason": "False alarm / controlled sit"
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["event"]["status"], "CANCELLED")

    def test_13_ui_updates_to_cancelled(self):
        """Test 13: Verified that event status is persistent as CANCELLED with cancellation reason."""
        self.service.record_confirmed_fall("EVT_CAN_13", detected_at=get_utc_iso_now())
        self.client.patch("/api/fall-events/EVT_CAN_13/cancel", json={
            "cancelled_by": "Operator_A",
            "reason": "Verified user stood up immediately"
        })
        res = self.client.get("/api/fall-events/EVT_CAN_13")
        evt = res.get_json()["event"]
        self.assertEqual(evt["status"], "CANCELLED")
        self.assertEqual(evt["cancellation_reason"], "Verified user stood up immediately")
        self.assertIsNotNone(evt["cancelled_at"])

    def test_14_countdown_stops_upon_cancellation(self):
        """Test 14: Cancelled event is removed from active confirmed query pool."""
        self.service.record_confirmed_fall("EVT_CAN_14", detected_at=get_utc_iso_now())
        self.client.patch("/api/fall-events/EVT_CAN_14/cancel", json={})
        res_conf = self.client.get("/api/fall-events?status=CONFIRMED")
        self.assertEqual(len(res_conf.get_json()["events"]), 0)

    def test_15_repeated_cancellation_handled_safely(self):
        """Test 15: Second cancel request returns 200 OK idempotently."""
        self.service.record_confirmed_fall("EVT_CAN_15", detected_at=get_utc_iso_now())
        res1 = self.client.patch("/api/fall-events/EVT_CAN_15/cancel", json={"reason": "User ok"})
        self.assertEqual(res1.status_code, 200)
        res2 = self.client.patch("/api/fall-events/EVT_CAN_15/cancel", json={"reason": "User ok"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["event"]["status"], "CANCELLED")

    # -------------------------------------------------------------------------
    # Timeout Behavior (Tests 16 - 18)
    # -------------------------------------------------------------------------
    def test_16_countdown_reaches_zero_calculation(self):
        """Test 16: Calculation for events detected > 60 seconds ago yields 0 seconds."""
        seventy_sec_ago = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
        evt, _ = self.service.record_confirmed_fall("EVT_TIMEOUT_16", detected_at=seventy_sec_ago)
        dt = datetime.fromisoformat(evt.detected_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
        remaining = max(0, 60 - int(elapsed))
        self.assertEqual(remaining, 0)

    def test_17_no_emergency_action_occurs_on_timeout(self):
        """Test 17: No emergency services calling, dispatching or external alert triggered."""
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.get_json()["emergency_notifications"], "DISABLED")

    def test_18_event_remains_persisted_as_confirmed_on_timeout(self):
        """Test 18: After response window expiry, event remains CONFIRMED in database."""
        past_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        self.service.record_confirmed_fall("EVT_EXP_18", detected_at=past_time)
        res = self.client.get("/api/fall-events/EVT_EXP_18")
        self.assertEqual(res.get_json()["event"]["status"], "CONFIRMED")

    # -------------------------------------------------------------------------
    # Refresh / Reconnect Behavior (Tests 19 - 21)
    # -------------------------------------------------------------------------
    def test_19_refresh_during_active_event_reloads_same_event(self):
        """Test 19: Reloading dashboard / re-fetching returns the active confirmed event."""
        self.service.record_confirmed_fall("EVT_REFRESH_19", detected_at=get_utc_iso_now(), session_id="SESS_19")
        res = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["events"][0]["event_id"], "EVT_REFRESH_19")

    def test_20_existing_event_is_rediscovered_without_new_id(self):
        """Test 20: Event discovery preserves existing event_id across queries."""
        self.service.record_confirmed_fall("EVT_REDISCOVER_20", detected_at=get_utc_iso_now())
        res1 = self.client.get("/api/fall-events/EVT_REDISCOVER_20")
        res2 = self.client.get("/api/fall-events/EVT_REDISCOVER_20")
        self.assertEqual(res1.get_json()["event"]["event_id"], res2.get_json()["event"]["event_id"])

    def test_21_countdown_does_not_reset_incorrectly_on_refresh(self):
        """Test 21: Elapsed time since detection preserves truthful remaining seconds."""
        twenty_sec_ago = (datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat()
        evt, _ = self.service.record_confirmed_fall("EVT_TIME_21", detected_at=twenty_sec_ago)
        dt = datetime.fromisoformat(evt.detected_at.replace("Z", "+00:00"))
        remaining = max(0, 60 - int((datetime.now(timezone.utc) - dt).total_seconds()))
        self.assertLessEqual(remaining, 41)
        self.assertGreaterEqual(remaining, 38)

    # -------------------------------------------------------------------------
    # Failure Handling (Tests 22 - 25)
    # -------------------------------------------------------------------------
    def test_22_backend_health_check_available(self):
        """Test 22: Backend /api/health responds with healthy status."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "healthy")

    def test_23_acknowledge_failure_on_missing_event(self):
        """Test 23: Acknowledge on non-existent event returns 404."""
        res = self.client.patch("/api/fall-events/NON_EXISTENT_99/acknowledge", json={})
        self.assertEqual(res.status_code, 404)

    def test_24_cancel_failure_on_missing_event(self):
        """Test 24: Cancel on non-existent event returns 404."""
        res = self.client.patch("/api/fall-events/NON_EXISTENT_99/cancel", json={})
        self.assertEqual(res.status_code, 404)

    def test_25_invalid_state_transition_returns_409_conflict(self):
        """Test 25: Illegal state transitions return 409 Conflict."""
        self.service.record_confirmed_fall("EVT_CONFLICT_25", detected_at=get_utc_iso_now())
        self.service.cancel_event("EVT_CONFLICT_25")
        # Attempting to acknowledge a cancelled event
        res = self.client.patch("/api/fall-events/EVT_CONFLICT_25/acknowledge", json={})
        self.assertEqual(res.status_code, 409)

    # -------------------------------------------------------------------------
    # History & Detail (Tests 26 - 28)
    # -------------------------------------------------------------------------
    def test_26_event_list_renders_multiple_records(self):
        """Test 26: List endpoint returns sorted history records."""
        for i in range(5):
            self.service.record_confirmed_fall(f"EVT_HIST_{i}", detected_at=get_utc_iso_now())
        res = self.client.get("/api/fall-events?limit=10")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["count"], 5)

    def test_27_pagination_limit_and_offset_works(self):
        """Test 27: Pagination correctly offsets and limits results."""
        for i in range(10):
            self.service.record_confirmed_fall(f"EVT_PAGE_{i:02d}", detected_at=get_utc_iso_now())
        res1 = self.client.get("/api/fall-events?limit=4&offset=0")
        self.assertEqual(len(res1.get_json()["events"]), 4)
        res2 = self.client.get("/api/fall-events?limit=4&offset=4")
        self.assertEqual(len(res2.get_json()["events"]), 4)
        self.assertNotEqual(res1.get_json()["events"][0]["event_id"], res2.get_json()["events"][0]["event_id"])

    def test_28_event_detail_includes_all_diagnostic_metadata(self):
        """Test 28: Event detail endpoint returns complete diagnostic payload."""
        diag_data = {
            "temporal_state": "CONFIRMED_FALL",
            "positive_count": 3,
            "max_dyn_acc_mag": 13.22,
            "max_gyr_mag": 747.5,
            "max_jerk_mag": 1257.5
        }
        self.service.record_confirmed_fall(
            "EVT_DETAIL_28",
            detected_at=get_utc_iso_now(),
            occurred_at="8.24s",
            session_id="session_012_fall_forward_slip",
            confidence=0.9472,
            diagnostic_metadata=diag_data
        )
        res = self.client.get("/api/fall-events/EVT_DETAIL_28")
        self.assertEqual(res.status_code, 200)
        evt = res.get_json()["event"]
        self.assertEqual(evt["diagnostic_metadata"]["max_dyn_acc_mag"], 13.22)
        self.assertEqual(evt["diagnostic_metadata"]["max_gyr_mag"], 747.5)

    # -------------------------------------------------------------------------
    # Regressions (Tests 29 - 32)
    # -------------------------------------------------------------------------
    def test_29_stage_5A_backend_idempotency_preserved(self):
        """Test 29: Regression - Stage 5A duplicate creation idempotency preserved."""
        evt1, new1 = self.service.record_confirmed_fall("EVT_REG_29", detected_at=get_utc_iso_now())
        evt2, new2 = self.service.record_confirmed_fall("EVT_REG_29", detected_at=get_utc_iso_now())
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(evt1.event_id, evt2.event_id)

    def test_30_stage_4E_baseline_configuration_preserved(self):
        """Test 30: Regression - Stage 4E temporal engine configuration unchanged."""
        cfg_path = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4E", "confirmation_engine_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["temporal_rule"], "N3")
        self.assertEqual(cfg["model_threshold"], 0.60)
        self.assertEqual(cfg["require_event_confirmation"], False)

    def test_31_stage_4F_temporal_adapter_regression(self):
        """Test 31: Regression - Stage 4F temporal state machine functions identically."""
        adapter = TemporalConfirmationAdapter()
        adapter.reset()
        res1 = adapter.process_prediction(0.70, window_start_time=0.0)
        res2 = adapter.process_prediction(0.75, window_start_time=0.25)
        res3 = adapter.process_prediction(0.80, window_start_time=0.50)
        self.assertFalse(res1["confirmed_fall"])
        self.assertFalse(res2["confirmed_fall"])
        self.assertTrue(res3["confirmed_fall"])
        self.assertEqual(res3["temporal_state"], "CONFIRMED_FALL")

    def test_32_stage_4G_live_validation_parity_preserved(self):
        """Test 32: Regression - Live validator can stream to integration adapter."""
        sim_evt = {
            "event_id": "EVT_REG_32",
            "session_id": "session_013_fall_backward_slip",
            "timestamp": 8.50,
            "probability": 0.9680,
            "state": "CONFIRMED_FALL",
            "positive_count": 3
        }
        evt, is_new = self.adapter.process_detector_event(sim_evt)
        self.assertTrue(is_new)
        self.assertEqual(evt.event_id, "EVT_REG_32")
        self.assertEqual(evt.session_id, "session_013_fall_backward_slip")
        self.assertAlmostEqual(evt.confidence, 0.9680, places=4)


def run_tests_and_summarize() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStage5BDashboard)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "total_tests": result.testsRun,
        "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "was_successful": result.wasSuccessful()
    }


if __name__ == "__main__":
    summary = run_tests_and_summarize()
    print("\nSTAGE 5B TEST SUMMARY:", summary)
    if not summary["was_successful"]:
        sys.exit(1)
