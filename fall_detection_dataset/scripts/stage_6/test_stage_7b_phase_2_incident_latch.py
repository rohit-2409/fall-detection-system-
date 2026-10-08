"""
Stage 7B Phase 2 Automated Test Suite: One-Physical-Fall = One-Logical-Event.
Standalone Fall Detection Project.

Validates the Stage 7B Phase 2 incident latching & re-arm requirements:
  TEST 1: One confirmed fall produces exactly one logical event.
  TEST 2: Additional high-probability windows during post-fall lock do NOT create another event.
  TEST 3: After the lock period, continued ground-level movement does NOT immediately re-arm detector.
  TEST 4: Detector only re-arms after configured stable/upright condition (e.g. 5s continuous stability).
  TEST 5: After successful re-arm, a genuinely new fall can produce a new event.
  TEST 6: The same incident retains the same logical event ID.
  TEST 7: Only one MQTT publication occurs for one physical incident.
  TEST 8: Existing Phase 1 MQTT -> Gateway -> Backend -> SQLite behavior remains intact.
"""

import collections
import datetime
import json
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_bridge import run_stage_6L_hardware_bridge
from src.fall_confirmation.stage_6F_mqtt_fall_publisher import MqttFallPublisher
from src.fall_confirmation.stage_6G_mqtt_gateway import MqttBackendGateway
from src.fall_confirmation.stage_6L_gps_cache import GpsTelemetryCache
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter


class MockMqttPublisher(MqttFallPublisher):
    """Captures published events in-memory for testing without external broker."""

    def __init__(self):
        super().__init__(broker_host="127.0.0.1", broker_port=1883)
        self.published_events: List[Dict[str, Any]] = []

    def connect(self, timeout: float = 2.0) -> bool:
        self.is_connected = True
        return True

    def publish_confirmed_fall(
        self,
        event_data: Dict[str, Any],
        location: Optional[Dict[str, Any]] = None,
        timeout: float = 2.0,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        payload = event_data.copy()
        if location:
            payload["location"] = location
        self.published_events.append(payload)
        return True, "MOCK_PUBLISHED", payload


class TestStage7BPhase2IncidentLatch(unittest.TestCase):
    """Complete verification of Stage 7B Phase 2 incident latching & state machine."""

    def setUp(self):
        self.gps_cache = GpsTelemetryCache()
        self.gps_cache.update_location(18.5204, 73.8567, satellites=8, hdop=1.0, valid=True)

    # -------------------------------------------------------------------------
    # TEST 1: One confirmed fall produces exactly one logical event
    # -------------------------------------------------------------------------
    def test_01_single_physical_fall_produces_one_event(self):
        """TEST 1: 3 consecutive candidate windows produce exactly 1 confirmed fall event."""
        adapter = TemporalConfirmationAdapter()
        adapter.reset()

        # Windows: 2 candidate windows, 3rd triggers N3 confirmation
        res1 = adapter.process_prediction(0.85, window_start_time=0.0)
        self.assertFalse(res1["confirmed_fall"])
        self.assertEqual(res1["temporal_state"], TemporalConfirmationAdapter.STATE_POSSIBLE_FALL)

        res2 = adapter.process_prediction(0.88, window_start_time=0.25)
        self.assertFalse(res2["confirmed_fall"])
        self.assertEqual(res2["temporal_state"], TemporalConfirmationAdapter.STATE_POSSIBLE_FALL)

        res3 = adapter.process_prediction(0.92, window_start_time=0.50)
        self.assertTrue(res3["confirmed_fall"])
        self.assertEqual(res3["temporal_state"], TemporalConfirmationAdapter.STATE_CONFIRMED_FALL)
        self.assertIsNotNone(res3["incident_id"])
        self.assertEqual(adapter.total_confirmed_events, 1)

    # -------------------------------------------------------------------------
    # TEST 2: Additional high-probability windows during post-fall lock do NOT create another event
    # -------------------------------------------------------------------------
    def test_02_post_fall_lock_suppresses_duplicate_events(self):
        """TEST 2: Continued high probabilities during post-fall monitoring/lock do NOT create events."""
        adapter = TemporalConfirmationAdapter(lock_duration_s=60.0)
        adapter.reset()

        # Trigger initial fall
        adapter.process_prediction(0.85, window_start_time=0.0)
        adapter.process_prediction(0.88, window_start_time=0.25)
        res_fall = adapter.process_prediction(0.90, window_start_time=0.50)
        self.assertTrue(res_fall["confirmed_fall"])
        first_incident_id = res_fall["incident_id"]

        # Now simulate 40 subsequent windows with high fall probability (rolling/lying on floor)
        subsequent_alarms = []
        for i in range(1, 41):
            t_win = 0.50 + (i * 0.25)
            # High probability, high dynamic acceleration, horizontal orientation
            res = adapter.process_prediction(
                probability=0.95,
                window_start_time=t_win,
                dyn_acc_max=2.5,
                gyr_mag_max=150.0,
                acc_z_mean=0.10,  # recumbent on floor
            )
            if res["confirmed_fall"]:
                subsequent_alarms.append(i)
            # Must remain locked
            self.assertIn(
                res["temporal_state"],
                (TemporalConfirmationAdapter.STATE_POST_FALL_MONITORING,
                 TemporalConfirmationAdapter.STATE_LOCKED_ALARM),
            )
            self.assertEqual(res["incident_id"], first_incident_id)

        self.assertEqual(len(subsequent_alarms), 0)
        self.assertEqual(adapter.total_confirmed_events, 1)

    # -------------------------------------------------------------------------
    # TEST 3: After lock period, ground-level movement does NOT immediately re-arm
    # -------------------------------------------------------------------------
    def test_03_recumbent_movement_after_lock_prevents_rearm(self):
        """TEST 3: When lock expires, lying on ground or moving prevents re-arming."""
        # Short lock duration of 2.0s for controlled test
        adapter = TemporalConfirmationAdapter(lock_duration_s=2.0, min_stable_duration_s=5.0)
        adapter.reset()

        # Trigger fall
        adapter.process_prediction(0.80, window_start_time=0.0)
        adapter.process_prediction(0.85, window_start_time=0.25)
        adapter.process_prediction(0.90, window_start_time=0.50)

        # Advance past 8 debounce windows and 2.0s lock duration
        t = 0.75
        for _ in range(20):
            adapter.process_prediction(
                0.20,
                window_start_time=t,
                dyn_acc_max=0.1,
                gyr_mag_max=5.0,
                acc_z_mean=0.10,  # lying down
            )
            t += 0.25

        # System should now be in WAIT_UPRIGHT_STABLE, but NOT rearmed because acc_z_mean is 0.10g
        res = adapter.process_prediction(
            0.15,
            window_start_time=t,
            dyn_acc_max=0.1,
            gyr_mag_max=5.0,
            acc_z_mean=0.10,  # lying flat
        )
        self.assertEqual(res["temporal_state"], TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)
        self.assertEqual(res["stable_elapsed_s"], 0.0)
        self.assertFalse(res["confirmed_fall"])

        # Also test with dynamic motion while lying down
        t += 0.25
        res_motion = adapter.process_prediction(
            0.15,
            window_start_time=t,
            dyn_acc_max=0.60,  # high motion on floor
            gyr_mag_max=50.0,
            acc_z_mean=0.10,
        )
        self.assertEqual(res_motion["temporal_state"], TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)
        self.assertEqual(res_motion["stable_elapsed_s"], 0.0)

    # -------------------------------------------------------------------------
    # TEST 4: Detector only re-arms after configured stable/upright condition
    # -------------------------------------------------------------------------
    def test_04_rearm_requires_sustained_upright_and_stable(self):
        """TEST 4: Rearm only occurs after sustained upright posture + calm motion for min_stable_duration."""
        adapter = TemporalConfirmationAdapter(
            lock_duration_s=1.0,
            min_stable_duration_s=2.0,  # 2.0s = 8 windows of 0.25s
            min_upright_acc_g=0.65,
            max_stable_dyn_acc_g=0.35,
        )
        adapter.reset()

        # Trigger fall
        adapter.process_prediction(0.80, window_start_time=0.0)
        adapter.process_prediction(0.85, window_start_time=0.25)
        adapter.process_prediction(0.90, window_start_time=0.50)

        # Advance past debounce and lock into WAIT_UPRIGHT_STABLE
        t = 0.75
        for _ in range(15):
            adapter.process_prediction(0.10, window_start_time=t, acc_z_mean=0.10)
            t += 0.25

        self.assertEqual(adapter.state, TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)

        # Feed 4 windows of upright (1.0s < 2.0s required)
        for _ in range(4):
            res_upright = adapter.process_prediction(
                0.05,
                window_start_time=t,
                dyn_acc_max=0.05,
                gyr_mag_max=5.0,
                acc_z_mean=0.98,  # Upright standing
            )
            t += 0.25
        self.assertEqual(res_upright["temporal_state"], TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)
        self.assertGreater(res_upright["stable_elapsed_s"], 0.0)

        # Feed a motion shock: resets stability counter
        res_shock = adapter.process_prediction(
            0.05,
            window_start_time=t,
            dyn_acc_max=0.85,  # Shock / movement
            gyr_mag_max=5.0,
            acc_z_mean=0.98,
        )
        t += 0.25
        self.assertEqual(res_shock["stable_elapsed_s"], 0.0)

        # Now feed 8 continuous windows of upright + calm (2.0s continuous)
        for i in range(7):
            res_calm = adapter.process_prediction(
                0.05,
                window_start_time=t,
                dyn_acc_max=0.05,
                gyr_mag_max=3.0,
                acc_z_mean=0.98,
            )
            t += 0.25
            self.assertEqual(res_calm["temporal_state"], TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)

        # 8th window completes the 2.0s duration -> transitions to REARMED
        res_rearm = adapter.process_prediction(
            0.05,
            window_start_time=t,
            dyn_acc_max=0.05,
            gyr_mag_max=3.0,
            acc_z_mean=0.98,
        )
        self.assertEqual(res_rearm["temporal_state"], TemporalConfirmationAdapter.STATE_REARMED)

        # Next incoming window is back in IDLE
        t += 0.25
        res_idle = adapter.process_prediction(
            0.05,
            window_start_time=t,
            dyn_acc_max=0.05,
            gyr_mag_max=3.0,
            acc_z_mean=0.98,
        )
        self.assertEqual(res_idle["temporal_state"], TemporalConfirmationAdapter.STATE_IDLE)
        self.assertIsNone(res_idle["incident_id"])

    # -------------------------------------------------------------------------
    # TEST 5: After successful re-arm, a new fall produces a new event
    # -------------------------------------------------------------------------
    def test_05_new_fall_after_rearm_produces_new_event(self):
        """TEST 5: Genuinely new fall incident after re-arm produces a distinct second fall event."""
        adapter = TemporalConfirmationAdapter(lock_duration_s=0.5, min_stable_duration_s=0.5)
        adapter.reset()

        # Fall 1
        adapter.process_prediction(0.80, window_start_time=0.0)
        adapter.process_prediction(0.85, window_start_time=0.25)
        res_f1 = adapter.process_prediction(0.90, window_start_time=0.50)
        id_1 = res_f1["incident_id"]
        self.assertTrue(res_f1["confirmed_fall"])

        # Advance past lock and stabilize
        t = 0.75
        for _ in range(12):
            adapter.process_prediction(
                0.05,
                window_start_time=t,
                dyn_acc_max=0.05,
                gyr_mag_max=2.0,
                acc_z_mean=0.99,
            )
            t += 0.25

        self.assertEqual(adapter.state, TemporalConfirmationAdapter.STATE_IDLE)

        # Fall 2
        adapter.process_prediction(0.82, window_start_time=t)
        t += 0.25
        adapter.process_prediction(0.87, window_start_time=t)
        t += 0.25
        res_f2 = adapter.process_prediction(0.93, window_start_time=t)
        id_2 = res_f2["incident_id"]

        self.assertTrue(res_f2["confirmed_fall"])
        self.assertEqual(adapter.total_confirmed_events, 2)
        self.assertNotEqual(id_1, id_2)

    # -------------------------------------------------------------------------
    # TEST 6: The same incident retains the same logical event ID
    # -------------------------------------------------------------------------
    def test_06_incident_id_persistence_and_acknowledgement(self):
        """TEST 6: Incident ID persists across all post-fall states; acknowledgement advances lock."""
        adapter = TemporalConfirmationAdapter(lock_duration_s=60.0)
        adapter.reset()

        adapter.process_prediction(0.80, window_start_time=0.0)
        adapter.process_prediction(0.85, window_start_time=0.25)
        res_fall = adapter.process_prediction(0.90, window_start_time=0.50)
        incident_id = res_fall["incident_id"]

        # Check in debounce
        res_deb = adapter.process_prediction(0.50, window_start_time=0.75)
        self.assertEqual(res_deb["incident_id"], incident_id)

        # Advance into LOCKED_ALARM
        for i in range(10):
            res_lock = adapter.process_prediction(0.50, window_start_time=1.0 + (i * 0.25))
            self.assertEqual(res_lock["incident_id"], incident_id)

        self.assertEqual(res_lock["temporal_state"], TemporalConfirmationAdapter.STATE_LOCKED_ALARM)

        # User acknowledges event via dashboard / API
        ack_success = adapter.acknowledge_incident(incident_id)
        self.assertTrue(ack_success)
        self.assertTrue(adapter.is_acknowledged)

        # System should advance out of LOCKED_ALARM directly into WAIT_UPRIGHT_STABLE
        self.assertEqual(adapter.state, TemporalConfirmationAdapter.STATE_WAIT_UPRIGHT_STABLE)

    # -------------------------------------------------------------------------
    # TEST 7: Only one MQTT publication occurs for one physical incident
    # -------------------------------------------------------------------------
    def test_07_hardware_bridge_publishes_exactly_once(self):
        """TEST 7: Hardware bridge produces exactly one MQTT publish for a fall scenario."""
        mock_pub = MockMqttPublisher()
        res = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=12,  # Span past confirmation and into cooldown/monitoring
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )

        self.assertEqual(res["confirmed_falls"], 1)
        self.assertEqual(res["mqtt_published"], 1)
        self.assertEqual(len(mock_pub.published_events), 1)

        event = mock_pub.published_events[0]
        self.assertEqual(event["event"], "CONFIRMED_FALL")
        self.assertTrue(event["confirmed_fall"])
        self.assertIn("STAGE6L-FALL-", event["event_id"])

    # -------------------------------------------------------------------------
    # TEST 8: Existing Phase 1 MQTT -> Gateway -> Backend -> SQLite behavior intact
    # -------------------------------------------------------------------------
    def test_08_phase_1_mqtt_to_backend_pipeline_compatibility(self):
        """TEST 8: Published event conforms to backend API contract and survives gateway transformation."""
        mock_pub = MockMqttPublisher()
        res = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=5,
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertEqual(len(mock_pub.published_events), 1)
        mqtt_evt = mock_pub.published_events[0]

        # Convert using Stage 6G Gateway logic
        backend_payload = MqttBackendGateway.map_to_backend_payload(mqtt_evt)

        # Verify all mandatory backend fields are present and typed correctly
        self.assertIn("event_id", backend_payload)
        self.assertIn("detected_at", backend_payload)
        self.assertIn("occurred_at", backend_payload)
        self.assertIn("confidence", backend_payload)
        self.assertIn("source", backend_payload)
        self.assertIn("session_id", backend_payload)
        self.assertIn("diagnostic_metadata", backend_payload)

        # Verify location mapping
        loc = backend_payload.get("diagnostic_metadata", {}).get("location", {})
        self.assertEqual(loc.get("latitude"), 18.5204)
        self.assertEqual(loc.get("longitude"), 73.8567)
        self.assertTrue(loc.get("fix_valid"))
        self.assertEqual(backend_payload.get("diagnostic_metadata", {}).get("status"), "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
