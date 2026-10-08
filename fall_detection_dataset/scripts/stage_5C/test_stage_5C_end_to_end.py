"""
Stage 5C Comprehensive Test Suite: End-to-End Reliability & Integration.

Validates:
- Phase 2: End-to-End Lifecycle (Startup, Generation, Acknowledge, Cancel, Resolve)
- Phase 3: Duplicate Event Stress Test (10, 50, 100 repeated submissions)
- Phase 4: Polling / UI Reliability (Discovery, no DOM thrashing, timer persistence)
- Phase 5: Refresh / Reconnect Recovery (Elapsed-time preservation, outage recovery)
- Phase 6: Database Recovery (WAL mode, connection re-open, persistent state)
- Phase 8: Sensor Stream Interruption (Gaps > 0.50s trigger state-machine reset)
- Phase 10: Failure Injection (404 Not Found, 409 Conflict, 400 Bad Request)
- Phase 12: Regressions (Stage 5A, Stage 5B, Stage 4E, Stage 4F, Stage 4G)
"""

import json
import os
import shutil
import sqlite3
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
from src.fall_confirmation.stage_4G_live_validator import (
    Stage4GLiveValidator,
    compute_dynamic_acceleration,
    compute_jerk_magnitude
)


class TestStage5CEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stage_5c_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.adapter = FallEventIntegrationAdapter(self.service)
        self.app = create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # PHASE 2: End-to-End Test Lifecycle (A - E)
    # -------------------------------------------------------------------------
    def test_2A_normal_startup(self):
        """Phase 2A: Verify backend startup, health check, dashboard routes, and assets."""
        # 1. Health check
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        health_data = res_health.get_json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertEqual(health_data["service"], "fall_event_backend")
        self.assertEqual(health_data["emergency_notifications"], "DISABLED")

        # 2. Root route & dashboard route
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("Wearable Fall Detection System", res_root.get_data(as_text=True))

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)

        # 3. Static CSS and JS
        res_css = self.client.get("/static/css/dashboard.css")
        self.assertEqual(res_css.status_code, 200)
        res_js = self.client.get("/static/js/dashboard.js")
        self.assertEqual(res_js.status_code, 200)

        # 4. Database accessible
        self.assertTrue(os.path.exists(self.db_path))

    def test_2B_confirmed_fall_event_lifecycle(self):
        """Phase 2B: Generate controlled test event, verify persistence, status, API retrieval."""
        detector_payload = {
            "event_id": "EVT_5C_E2E_01",
            "session_id": "session_5C_validation",
            "timestamp": 14.25,
            "probability": 0.9450,
            "state": "CONFIRMED_FALL",
            "positive_count": 3,
            "pre_event_activity": "SAFE_FALL_SIMULATION"
        }
        evt, is_new = self.adapter.process_detector_event(detector_payload)
        self.assertTrue(is_new)
        self.assertEqual(evt.event_id, "EVT_5C_E2E_01")
        self.assertEqual(evt.status, FallEventStatus.CONFIRMED.value)

        # Query via REST API
        res = self.client.get("/api/fall-events/EVT_5C_E2E_01")
        self.assertEqual(res.status_code, 200)
        api_evt = res.get_json()["event"]
        self.assertEqual(api_evt["event_id"], "EVT_5C_E2E_01")
        self.assertEqual(api_evt["status"], "CONFIRMED")
        self.assertEqual(api_evt["session_id"], "session_5C_validation")
        self.assertAlmostEqual(api_evt["confidence"], 0.9450, places=4)
        self.assertEqual(api_evt["source"], "ESP32_MPU6050")
        self.assertEqual(api_evt["diagnostic_metadata"]["temporal_state"], "CONFIRMED_FALL")

    def test_2C_acknowledge_flow(self):
        """Phase 2C: Verify CONFIRMED -> ACKNOWLEDGED transition and repeated call safety."""
        self.service.record_confirmed_fall("EVT_5C_ACK_01", detected_at=get_utc_iso_now())
        
        # 1. Acknowledge
        res1 = self.client.patch("/api/fall-events/EVT_5C_ACK_01/acknowledge", json={
            "acknowledged_by": "Test_Operator"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()["event"]
        self.assertEqual(data1["status"], "ACKNOWLEDGED")
        self.assertEqual(data1["acknowledged_by"], "Test_Operator")
        self.assertIsNotNone(data1["acknowledged_at"])

        # 2. Repeated acknowledge is safe and idempotent
        res2 = self.client.patch("/api/fall-events/EVT_5C_ACK_01/acknowledge", json={
            "acknowledged_by": "Test_Operator"
        })
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["event"]["status"], "ACKNOWLEDGED")

        # 3. Verify no duplicate records created
        res_list = self.client.get("/api/fall-events")
        self.assertEqual(res_list.get_json()["count"], 1)

    def test_2D_cancel_flow(self):
        """Phase 2D: Verify CONFIRMED -> CANCELLED transition, reason persistence, and repeated safety."""
        self.service.record_confirmed_fall("EVT_5C_CAN_01", detected_at=get_utc_iso_now())

        # 1. Cancel
        res1 = self.client.patch("/api/fall-events/EVT_5C_CAN_01/cancel", json={
            "cancelled_by": "Test_Operator",
            "reason": "Controlled sit simulation"
        })
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()["event"]
        self.assertEqual(data1["status"], "CANCELLED")
        self.assertEqual(data1["cancellation_reason"], "Controlled sit simulation")
        self.assertIsNotNone(data1["cancelled_at"])

        # 2. Repeated cancellation is safe and idempotent
        res2 = self.client.patch("/api/fall-events/EVT_5C_CAN_01/cancel", json={
            "reason": "Controlled sit simulation"
        })
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.get_json()["event"]["status"], "CANCELLED")

        # 3. Verify event is NOT marked resolved
        self.assertNotEqual(res2.get_json()["event"]["status"], "RESOLVED")

    def test_2E_resolve_flow(self):
        """Phase 2E: Verify CONFIRMED -> ACKNOWLEDGED -> RESOLVED transition and audit fields."""
        self.service.record_confirmed_fall("EVT_5C_RES_01", detected_at=get_utc_iso_now())

        # Transition 1: Acknowledge
        self.client.patch("/api/fall-events/EVT_5C_RES_01/acknowledge", json={"acknowledged_by": "Nurse_1"})

        # Transition 2: Resolve
        res_res = self.client.patch("/api/fall-events/EVT_5C_RES_01/resolve", json={
            "resolved_by": "Doctor_1"
        })
        self.assertEqual(res_res.status_code, 200)
        evt_res = res_res.get_json()["event"]
        self.assertEqual(evt_res["status"], "RESOLVED")
        self.assertEqual(evt_res["resolved_by"], "Doctor_1")
        self.assertIsNotNone(evt_res["resolved_at"])

    # -------------------------------------------------------------------------
    # PHASE 3: Duplicate Event Stress Test (10, 50, 100 submissions)
    # -------------------------------------------------------------------------
    def test_3_duplicate_event_stress_test(self):
        """Phase 3: Repeatedly submit identical event_id across 10, 50, and 100 calls."""
        evt_id = "EVT_STRESS_REPEAT_01"
        payload = {
            "event_id": evt_id,
            "session_id": "stress_session",
            "timestamp": 5.0,
            "probability": 0.92,
            "state": "CONFIRMED_FALL",
            "positive_count": 3
        }

        # Initial creation
        evt_orig, is_new_first = self.adapter.process_detector_event(payload)
        self.assertTrue(is_new_first)

        # Batch 1: 10 duplicate submissions
        for _ in range(10):
            evt, is_new = self.adapter.process_detector_event(payload)
            self.assertFalse(is_new)
            self.assertEqual(evt.event_id, evt_id)

        # Verify DB still contains exactly 1 row
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM fall_events WHERE event_id = ?", (evt_id,))
        self.assertEqual(cur.fetchone()[0], 1)

        # Batch 2: 50 duplicate submissions
        for _ in range(50):
            evt, is_new = self.adapter.process_detector_event(payload)
            self.assertFalse(is_new)

        cur.execute("SELECT COUNT(*) FROM fall_events WHERE event_id = ?", (evt_id,))
        self.assertEqual(cur.fetchone()[0], 1)

        # Batch 3: 100 duplicate submissions via REST API POST
        for _ in range(100):
            res = self.client.post("/api/fall-events", json={
                "event_id": evt_id,
                "detected_at": get_utc_iso_now(),
                "session_id": "stress_session",
                "confidence": 0.92
            })
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["status"], "success")
            self.assertFalse(res.get_json()["is_new"])

        cur.execute("SELECT COUNT(*) FROM fall_events")
        self.assertEqual(cur.fetchone()[0], 1)
        conn.close()

    # -------------------------------------------------------------------------
    # PHASE 4: Polling / UI Reliability Test
    # -------------------------------------------------------------------------
    def test_4_polling_ui_reliability(self):
        """Phase 4: Validate discovery, repeated polling idempotency, and alert clearing."""
        # 1. Dashboard starts with no event
        res0 = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
        self.assertEqual(len(res0.get_json()["events"]), 0)

        # 2. Confirmed event is inserted
        now_ts = get_utc_iso_now()
        self.service.record_confirmed_fall("EVT_POLL_TEST_4", detected_at=now_ts, confidence=0.96)

        # 3. Dashboard polling discovers the event
        res1 = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
        events1 = res1.get_json()["events"]
        self.assertEqual(len(events1), 1)
        self.assertEqual(events1[0]["event_id"], "EVT_POLL_TEST_4")

        # 4. Repeated sequential polling returns identical metadata without mutation
        for _ in range(10):
            poll_res = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
            poll_evt = poll_res.get_json()["events"][0]
            self.assertEqual(poll_evt["event_id"], "EVT_POLL_TEST_4")
            self.assertEqual(poll_evt["detected_at"], now_ts)
            self.assertEqual(poll_evt["status"], "CONFIRMED")

        # 5. Acknowledge removes active alert from CONFIRMED pool
        self.client.patch("/api/fall-events/EVT_POLL_TEST_4/acknowledge", json={})
        res_after = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
        self.assertEqual(len(res_after.get_json()["events"]), 0)

        # 6. Polling continues normally and history reflects acknowledged event
        res_hist = self.client.get("/api/fall-events")
        self.assertEqual(res_hist.get_json()["count"], 1)
        self.assertEqual(res_hist.get_json()["events"][0]["status"], "ACKNOWLEDGED")

    # -------------------------------------------------------------------------
    # PHASE 5: Refresh / Reconnect Testing
    # -------------------------------------------------------------------------
    def test_5_refresh_and_reconnect_recovery(self):
        """Phase 5: Verify elapsed time recovery on page refresh without resetting to 60s."""
        # 1. Event detected 25 seconds ago
        twenty_five_sec_ago = (datetime.now(timezone.utc) - timedelta(seconds=25)).isoformat()
        evt, _ = self.service.record_confirmed_fall("EVT_REFRESH_5", detected_at=twenty_five_sec_ago)

        # 2. UI refreshes / queries active confirmed event
        res = self.client.get("/api/fall-events?status=CONFIRMED&limit=1")
        self.assertEqual(res.status_code, 200)
        active_evt = res.get_json()["events"][0]
        self.assertEqual(active_evt["event_id"], "EVT_REFRESH_5")

        # 3. Calculation of remaining time
        dt = datetime.fromisoformat(active_evt["detected_at"].replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
        remaining = max(0, 60 - int(elapsed))
        
        # Verify it did NOT restart from 60 seconds (should be ~35s)
        self.assertLessEqual(remaining, 37)
        self.assertGreaterEqual(remaining, 33)

        # 4. Outage recovery simulation: health check confirms service availability
        res_reconnect = self.client.get("/api/health")
        self.assertEqual(res_reconnect.status_code, 200)
        self.assertEqual(res_reconnect.get_json()["status"], "healthy")

    # -------------------------------------------------------------------------
    # PHASE 6: Database Recovery Test
    # -------------------------------------------------------------------------
    def test_6_database_recovery_and_restart(self):
        """Phase 6: Verify WAL mode, data persistence across repository instantiation."""
        # 1. Verify WAL mode
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        self.assertEqual(mode.lower(), "wal")
        conn.close()

        # 2. Write event
        self.service.record_confirmed_fall("EVT_RESTART_6", detected_at=get_utc_iso_now(), session_id="restart_sess")

        # 3. Simulate backend/server shutdown: instantiate entirely new repo and service
        new_repo = FallEventRepository(self.db_path)
        new_service = FallEventService(new_repo)
        new_app = create_app(new_service)
        new_client = new_app.test_client()

        # 4. Verify persisted event is intact
        res = new_client.get("/api/fall-events/EVT_RESTART_6")
        self.assertEqual(res.status_code, 200)
        evt = res.get_json()["event"]
        self.assertEqual(evt["event_id"], "EVT_RESTART_6")
        self.assertEqual(evt["session_id"], "restart_sess")
        self.assertEqual(evt["status"], "CONFIRMED")

    # -------------------------------------------------------------------------
    # PHASE 8: Sensor Stream Interruption Test
    # -------------------------------------------------------------------------
    def test_8_sensor_stream_interruption_gap_reset(self):
        """Phase 8: Gaps > 0.50s in window timestamps must reset the temporal confirmation state."""
        adapter = TemporalConfirmationAdapter()
        adapter.reset()

        # Push 2 positive windows (rule N3 requires 3 consecutive)
        res1 = adapter.process_prediction(0.85, window_start_time=0.00)
        self.assertEqual(res1["positive_count"], 1)
        self.assertFalse(res1["confirmed_fall"])

        res2 = adapter.process_prediction(0.85, window_start_time=0.25)
        self.assertEqual(res2["positive_count"], 2)
        self.assertFalse(res2["confirmed_fall"])

        # Stream interruption: next sample arrives 1.25s later (> 0.50s stream_gap_threshold_s)
        res3 = adapter.process_prediction(0.85, window_start_time=1.50)
        # Because a gap occurred, the adapter automatically reset; this window becomes positive count 1
        self.assertEqual(res3["positive_count"], 1)
        self.assertFalse(res3["confirmed_fall"])

        # Then 2 more continuous windows trigger confirmation normally
        res4 = adapter.process_prediction(0.85, window_start_time=1.75)
        res5 = adapter.process_prediction(0.85, window_start_time=2.00)
        self.assertEqual(res5["positive_count"], 3)
        self.assertTrue(res5["confirmed_fall"])

    # -------------------------------------------------------------------------
    # PHASE 10: Failure Injection
    # -------------------------------------------------------------------------
    def test_10_failure_injection(self):
        """Phase 10: Test 404, 409, 400, and illegal state transitions."""
        # 1. 404 on missing event
        res_404 = self.client.get("/api/fall-events/NON_EXISTENT_ID")
        self.assertEqual(res_404.status_code, 404)
        self.assertEqual(res_404.get_json()["status"], "error")

        # 2. 409 Conflict on invalid state transition (CANCELLED -> ACKNOWLEDGED)
        self.service.record_confirmed_fall("EVT_FAIL_10", detected_at=get_utc_iso_now())
        self.service.cancel_event("EVT_FAIL_10")
        res_409 = self.client.patch("/api/fall-events/EVT_FAIL_10/acknowledge", json={})
        self.assertEqual(res_409.status_code, 409)
        self.assertEqual(res_409.get_json()["status"], "error")

        # 3. 400 on malformed request
        res_400 = self.client.post("/api/fall-events", data="invalid json string", content_type="application/json")
        self.assertEqual(res_400.status_code, 400)

        # 4. Missing required event_id on creation
        res_bad_id = self.client.post("/api/fall-events", json={"detected_at": get_utc_iso_now()})
        self.assertEqual(res_bad_id.status_code, 400)

    # -------------------------------------------------------------------------
    # PHASE 12: Regressions (Stage 5A, 5B, 4E, 4F, 4G)
    # -------------------------------------------------------------------------
    def test_12_regressions(self):
        """Phase 12: Regressions across Stage 4E, 4F, 4G, 5A, 5B."""
        # Stage 4E config
        cfg_path = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4E", "confirmation_engine_config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["temporal_rule"], "N3")
        self.assertEqual(cfg["model_threshold"], 0.60)
        self.assertEqual(cfg["require_event_confirmation"], False)

        # Stage 4F adapter
        temp_adapter = TemporalConfirmationAdapter()
        temp_adapter.reset()
        r1 = temp_adapter.process_prediction(0.70, window_start_time=0.0)
        r2 = temp_adapter.process_prediction(0.70, window_start_time=0.25)
        r3 = temp_adapter.process_prediction(0.70, window_start_time=0.50)
        self.assertTrue(r3["confirmed_fall"])

        # Stage 4G integration parity
        detector_out = {
            "session_id": "session_012_fall_forward_slip",
            "timestamp": 6.49,
            "probability": 0.9014,
            "state": "CONFIRMED_FALL",
            "positive_count": 3
        }
        evt, is_new = self.adapter.process_detector_event(detector_out)
        self.assertTrue(is_new)
        self.assertEqual(evt.session_id, "session_012_fall_forward_slip")
        self.assertEqual(evt.status, "CONFIRMED")


def run_tests() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStage5CEndToEnd)
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
    res = run_tests()
    print("\nSTAGE 5C END-TO-END TEST SUMMARY:", res)
    if not res["was_successful"]:
        sys.exit(1)
