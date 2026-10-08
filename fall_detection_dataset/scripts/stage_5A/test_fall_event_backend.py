"""
Stage 5A Fall Event Backend Automated Test Suite.
Covers:
- Event creation & validation (Tests 1-3)
- Duplicate protection & Idempotency (Tests 4-6)
- State machine lifecycle & illegal transition rejection (Tests 7-10)
- Event query & retrieval (Tests 11-13)
- Storage integrity & malformed payload rejection (Tests 14-15)
- Regression validation: Stage 4E, Stage 4F, Stage 4G (Tests 16-18)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Dict, Any, List, Optional, Tuple

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


class TestFallEventBackend(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_fall_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.adapter = FallEventIntegrationAdapter(self.service)
        self.app = create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Event Creation & Validation
    # -------------------------------------------------------------------------
    def test_01_confirmed_fall_creates_exactly_one_event(self):
        """Test 1: Confirmed fall creates exactly one event."""
        evt, is_new = self.service.record_confirmed_fall(
            event_id="EVT_SESSION_01_001",
            detected_at=get_utc_iso_now(),
            session_id="SESSION_01",
            confidence=0.9542
        )
        self.assertTrue(is_new)
        self.assertEqual(evt.event_id, "EVT_SESSION_01_001")
        self.assertEqual(evt.status, FallEventStatus.CONFIRMED)

        # Verify exactly one record in storage
        events = self.repo.list_all()
        self.assertEqual(len(events), 1)

    def test_02_event_contains_required_metadata(self):
        """Test 2: Event contains all required metadata fields."""
        diag_meta = {"peak_dyn_acc": 4.12, "peak_gyr": 265.4}
        evt, is_new = self.service.record_confirmed_fall(
            event_id="EVT_META_002",
            detected_at="2026-10-02T12:00:00Z",
            occurred_at="14.2500s",
            session_id="SESSION_META",
            confidence=0.9812,
            source="ESP32_MPU6050",
            diagnostic_metadata=diag_meta
        )
        self.assertEqual(evt.event_id, "EVT_META_002")
        self.assertEqual(evt.detected_at, "2026-10-02T12:00:00Z")
        self.assertEqual(evt.occurred_at, "14.2500s")
        self.assertEqual(evt.session_id, "SESSION_META")
        self.assertAlmostEqual(evt.confidence, 0.9812, places=4)
        self.assertEqual(evt.source, "ESP32_MPU6050")
        self.assertEqual(evt.status, FallEventStatus.CONFIRMED)
        self.assertIsNotNone(evt.created_at)
        self.assertIsNotNone(evt.updated_at)
        self.assertEqual(evt.diagnostic_metadata, diag_meta)

    def test_03_invalid_event_data_rejected(self):
        """Test 3: Invalid or empty event data is rejected."""
        with self.assertRaises(ValueError):
            self.service.record_confirmed_fall(event_id="", detected_at=get_utc_iso_now())
        with self.assertRaises(ValueError):
            self.service.record_confirmed_fall(event_id="EVT_VALID", detected_at="")

    # -------------------------------------------------------------------------
    # Duplicate Protection & Idempotency
    # -------------------------------------------------------------------------
    def test_04_same_event_id_cannot_create_second_event(self):
        """Test 4: Same event ID cannot create a second database record."""
        evt1, is_new1 = self.service.record_confirmed_fall(
            event_id="EVT_DUP_004",
            detected_at=get_utc_iso_now(),
            confidence=0.92
        )
        self.assertTrue(is_new1)

        evt2, is_new2 = self.service.record_confirmed_fall(
            event_id="EVT_DUP_004",
            detected_at=get_utc_iso_now(),
            confidence=0.92
        )
        self.assertFalse(is_new2)
        self.assertEqual(evt1.event_id, evt2.event_id)
        self.assertEqual(len(self.repo.list_all()), 1)

    def test_05_retried_request_remains_idempotent(self):
        """Test 5: Retried HTTP request remains idempotent."""
        payload = {
            "event_id": "EVT_HTTP_RETRY_005",
            "detected_at": get_utc_iso_now(),
            "confidence": 0.945,
            "session_id": "RETRY_SESSION"
        }
        res1 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res1.status_code, 201)
        data1 = res1.get_json()
        self.assertTrue(data1["is_new"])

        # Immediate retry
        res2 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertFalse(data2["is_new"])
        self.assertEqual(data1["event"]["event_id"], data2["event"]["event_id"])
        self.assertEqual(len(self.repo.list_all()), 1)

    def test_06_repeated_callback_does_not_create_duplicate_events(self):
        """Test 6: Repeated detector callback does not create duplicate events."""
        detector_output = {
            "event_id": "EVT_CALLBACK_006",
            "timestamp": 12.50,
            "probability": 0.967,
            "state": "CONFIRMED_FALL",
            "positive_count": 3
        }
        callback = self.adapter.create_detector_callback()
        evt1, new1 = callback(detector_output)
        evt2, new2 = callback(detector_output)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(len(self.repo.list_all()), 1)

    # -------------------------------------------------------------------------
    # State Machine Lifecycle & Transition Safety
    # -------------------------------------------------------------------------
    def test_07_confirmed_to_acknowledged_works(self):
        """Test 7: CONFIRMED -> ACKNOWLEDGED transition works and updates timestamp."""
        evt, _ = self.service.record_confirmed_fall(
            event_id="EVT_ACK_007",
            detected_at=get_utc_iso_now()
        )
        self.assertEqual(evt.status, FallEventStatus.CONFIRMED)

        ack_evt = self.service.acknowledge_event("EVT_ACK_007", acknowledged_by="Nurse_Smith")
        self.assertEqual(ack_evt.status, FallEventStatus.ACKNOWLEDGED)
        self.assertEqual(ack_evt.acknowledged_by, "Nurse_Smith")
        self.assertIsNotNone(ack_evt.acknowledged_at)

        # Idempotent re-acknowledgement
        ack_evt_again = self.service.acknowledge_event("EVT_ACK_007", acknowledged_by="Nurse_Smith")
        self.assertEqual(ack_evt_again.status, FallEventStatus.ACKNOWLEDGED)

    def test_08_confirmed_to_cancelled_works(self):
        """Test 8: CONFIRMED -> CANCELLED transition works and records reason."""
        evt, _ = self.service.record_confirmed_fall(
            event_id="EVT_CAN_008",
            detected_at=get_utc_iso_now()
        )
        can_evt = self.service.cancel_event("EVT_CAN_008", cancelled_by="User_Prompt", reason="False alarm verification")
        self.assertEqual(can_evt.status, FallEventStatus.CANCELLED)
        self.assertEqual(can_evt.cancelled_by, "User_Prompt")
        self.assertEqual(can_evt.cancellation_reason, "False alarm verification")
        self.assertIsNotNone(can_evt.cancelled_at)

        # Idempotent re-cancellation
        can_again = self.service.cancel_event("EVT_CAN_008")
        self.assertEqual(can_again.status, FallEventStatus.CANCELLED)

    def test_09_acknowledged_to_resolved_works(self):
        """Test 9: ACKNOWLEDGED -> RESOLVED transition works."""
        self.service.record_confirmed_fall(
            event_id="EVT_RES_009",
            detected_at=get_utc_iso_now()
        )
        self.service.acknowledge_event("EVT_RES_009", acknowledged_by="Medic_1")
        res_evt = self.service.resolve_event("EVT_RES_009", resolved_by="Medic_1")
        self.assertEqual(res_evt.status, FallEventStatus.RESOLVED)
        self.assertEqual(res_evt.resolved_by, "Medic_1")
        self.assertIsNotNone(res_evt.resolved_at)

    def test_10_invalid_transitions_are_rejected(self):
        """Test 10: Invalid state transitions are strictly rejected."""
        self.service.record_confirmed_fall("EVT_INV_010", detected_at=get_utc_iso_now())
        self.service.cancel_event("EVT_INV_010")

        # CANCELLED -> ACKNOWLEDGED is ILLEGAL
        with self.assertRaises(InvalidStateTransitionError):
            self.service.acknowledge_event("EVT_INV_010")

        # CANCELLED -> RESOLVED is ILLEGAL
        with self.assertRaises(InvalidStateTransitionError):
            self.service.resolve_event("EVT_INV_010")

        # CONFIRMED -> RESOLVED (without ACK) is ILLEGAL
        self.service.record_confirmed_fall("EVT_INV_010B", detected_at=get_utc_iso_now())
        with self.assertRaises(InvalidStateTransitionError):
            self.service.resolve_event("EVT_INV_010B")

    # -------------------------------------------------------------------------
    # Retrieval & Listing
    # -------------------------------------------------------------------------
    def test_11_event_can_be_retrieved(self):
        """Test 11: Event can be retrieved by event_id."""
        self.service.record_confirmed_fall("EVT_GET_011", detected_at=get_utc_iso_now(), session_id="SESS_11")
        res = self.client.get("/api/fall-events/EVT_GET_011")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["event"]["event_id"], "EVT_GET_011")

    def test_12_event_list_works(self):
        """Test 12: Event list works with filters."""
        self.service.record_confirmed_fall("EVT_L1", detected_at=get_utc_iso_now(), session_id="S1")
        self.service.record_confirmed_fall("EVT_L2", detected_at=get_utc_iso_now(), session_id="S1")
        self.service.record_confirmed_fall("EVT_L3", detected_at=get_utc_iso_now(), session_id="S2")
        self.service.acknowledge_event("EVT_L2")

        # Filter by session_id
        res_s1 = self.client.get("/api/fall-events?session_id=S1")
        self.assertEqual(res_s1.status_code, 200)
        self.assertEqual(res_s1.get_json()["count"], 2)

        # Filter by status
        res_ack = self.client.get("/api/fall-events?status=ACKNOWLEDGED")
        self.assertEqual(res_ack.status_code, 200)
        self.assertEqual(res_ack.get_json()["count"], 1)

    def test_13_unknown_event_id_handled_safely(self):
        """Test 13: Unknown event ID returns 404 cleanly."""
        res = self.client.get("/api/fall-events/NON_EXISTENT_EVT")
        self.assertEqual(res.status_code, 404)

        res_ack = self.client.patch("/api/fall-events/NON_EXISTENT_EVT/acknowledge", json={})
        self.assertEqual(res_ack.status_code, 404)

    # -------------------------------------------------------------------------
    # Failure Handling & Malformed Requests
    # -------------------------------------------------------------------------
    def test_14_database_integrity_under_concurrency_or_duplicates(self):
        """Test 14: Storage handles duplicate collisions without corrupting state."""
        evt = FallEvent(
            event_id="EVT_COLLIDE_014",
            detected_at=get_utc_iso_now(),
            confidence=0.91
        )
        e1, new1 = self.repo.create(evt)
        e2, new2 = self.repo.create(evt)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(e1.event_id, e2.event_id)

    def test_15_malformed_requests_rejected_safely(self):
        """Test 15: Malformed HTTP requests are rejected with 400 Bad Request."""
        res_empty = self.client.post("/api/fall-events", data="not json", content_type="application/json")
        self.assertEqual(res_empty.status_code, 400)

        res_bad_fields = self.client.post("/api/fall-events", json={"wrong": "payload"})
        self.assertEqual(res_bad_fields.status_code, 400)

    # -------------------------------------------------------------------------
    # Regression Tests
    # -------------------------------------------------------------------------
    def test_16_stage_4E_temporal_engine_configuration_preserved(self):
        """Test 16: Regression - Stage 4E temporal confirmation baseline preserved."""
        cfg_path = os.path.join(
            PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4E", "confirmation_engine_config.json"
        )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["temporal_rule"], "N3")
        self.assertEqual(cfg["model_threshold"], 0.60)
        self.assertEqual(cfg["require_event_confirmation"], False)

    def test_17_stage_4F_temporal_adapter_regression(self):
        """Test 17: Regression - Stage 4F temporal state machine functions correctly."""
        adapter = TemporalConfirmationAdapter()
        adapter.reset()

        # Window 1: P=0.70 -> raw_pos=True, confirmed=False
        r1 = adapter.process_prediction(0.70, window_start_time=0.0)
        self.assertTrue(r1["raw_positive"])
        self.assertFalse(r1["confirmed_fall"])

        # Window 2: P=0.75 -> raw_pos=True, confirmed=False
        r2 = adapter.process_prediction(0.75, window_start_time=0.25)
        self.assertTrue(r2["raw_positive"])
        self.assertFalse(r2["confirmed_fall"])

        # Window 3: P=0.80 -> confirmed=True!
        r3 = adapter.process_prediction(0.80, window_start_time=0.50)
        self.assertTrue(r3["confirmed_fall"])
        self.assertEqual(r3["temporal_state"], "CONFIRMED_FALL")

        # Window 4: P=0.85 -> Debounced into COOLDOWN, confirmed=False!
        r4 = adapter.process_prediction(0.85, window_start_time=0.75)
        self.assertFalse(r4["confirmed_fall"])
        self.assertEqual(r4["temporal_state"], "COOLDOWN")

    def test_18_stage_4G_integration_adapter_emits_valid_fall_event(self):
        """Test 18: Regression - Integration adapter receives confirmed fall and persists it."""
        detector_result = {
            "event_id": "EVT_STAGE_4G_SIM_18",
            "session_id": "session_012_fall_forward_slip",
            "timestamp": 8.24,
            "probability": 0.9472,
            "state": "CONFIRMED_FALL",
            "positive_count": 3,
            "pre_event_activity": "SAFE_FALL_SIMULATION"
        }
        meta_addition = {"max_dyn_acc_mag": 13.22, "max_jerk_mag": 1257.5}
        evt, is_new = self.adapter.process_detector_event(
            detector_result,
            additional_metadata=meta_addition
        )
        self.assertTrue(is_new)
        self.assertEqual(evt.event_id, "EVT_STAGE_4G_SIM_18")
        self.assertEqual(evt.session_id, "session_012_fall_forward_slip")
        self.assertAlmostEqual(evt.confidence, 0.9472, places=4)
        self.assertEqual(evt.diagnostic_metadata["max_dyn_acc_mag"], 13.22)

        # Verify event is queryable via API
        res = self.client.get("/api/fall-events/EVT_STAGE_4G_SIM_18")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["event"]["status"], "CONFIRMED")


def run_tests_and_generate_report() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFallEventBackend)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    test_summary = {
        "total_tests": result.testsRun,
        "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "was_successful": result.wasSuccessful()
    }
    return test_summary


if __name__ == "__main__":
    summary = run_tests_and_generate_report()
    print("\nTEST SUMMARY:", summary)
    if not summary["was_successful"]:
        sys.exit(1)
