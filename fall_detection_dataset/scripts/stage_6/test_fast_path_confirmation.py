"""
Stage 6: Fast-Path Confirmation Unit & Integration Test Suite.
Validates the additive Candidate 3B Fast-Path Confirmation capability
in the Wearable Fall Detection System.

Covers:
  1. Fast path disabled: existing baseline N=3 behavior remains unchanged.
  2. Fast path enabled + all thresholds satisfied: immediate confirmation.
  3. Probability below 0.85: fast path does not confirm.
  4. Dynamic acceleration below 2.50 g: fast path does not confirm.
  5. Gyro below 200 dps: fast path does not confirm.
  6. All fast-path conditions satisfied: confirmation occurs without waiting for N=3.
  7. Baseline N=3 still works when fast path is enabled.
  8. Fast path does not modify the baseline counter unexpectedly.
  9. Existing negative case: probs = [0.75, 0.82, 0.30] does not confirm under default settings.
 10. Existing 60-second post-fall lockout and re-arm state machine remain unchanged.
 11. Event generation and dictionary schema remain intact with additive fast_path_triggered flag.
 12. Configuration loader default values guarantee fast path is disabled by default.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.config_loader import load_confirmation_config
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter


class TestFastPathConfirmation(unittest.TestCase):
    """Dedicated regression suite for Candidate 3B Fast-Path confirmation."""

    def setUp(self):
        self.default_adapter = TemporalConfirmationAdapter()
        self.default_adapter.reset()

        self.fast_adapter = TemporalConfirmationAdapter(fast_path_enabled=True)
        self.fast_adapter.reset()

    # -------------------------------------------------------------------------
    # Requirement 1: Fast path disabled: existing behavior remains unchanged
    # -------------------------------------------------------------------------
    def test_01_fast_path_disabled_preserves_baseline_single_window_rejection(self):
        """Under default settings (fast_path_enabled=False), a single high-impact window is rejected."""
        self.assertFalse(self.default_adapter.fast_path_enabled)
        # Even with high probability, high accel, high gyro:
        res = self.default_adapter.process_prediction(
            probability=0.92,
            dyn_acc_max=3.20,
            gyr_mag_max=280.0,
            window_start_time=0.0,
        )
        self.assertFalse(res["confirmed_fall"], "Single window must NOT confirm fall when fast path disabled")
        self.assertFalse(res["fast_path_triggered"], "fast_path_triggered must be False when disabled")
        self.assertEqual(res["positive_count"], 1)
        self.assertEqual(res["temporal_state"], "POSSIBLE_FALL")

    # -------------------------------------------------------------------------
    # Requirement 2: Fast path enabled + all thresholds satisfied: immediate confirmation
    # -------------------------------------------------------------------------
    def test_02_fast_path_enabled_all_thresholds_satisfied_immediate_confirmation(self):
        """When fast path is enabled and all 3 criteria are met, confirmation fires on window 1."""
        self.assertTrue(self.fast_adapter.fast_path_enabled)
        res = self.fast_adapter.process_prediction(
            probability=0.85,
            dyn_acc_max=2.50,
            gyr_mag_max=200.0,
            window_start_time=0.0,
        )
        self.assertTrue(res["confirmed_fall"], "Must confirm immediately on window 1")
        self.assertTrue(res["fast_path_triggered"], "fast_path_triggered must be True")
        self.assertEqual(res["temporal_state"], "CONFIRMED_FALL")
        self.assertIsNotNone(res["incident_id"])
        self.assertTrue(res["incident_id"].startswith("STAGE6L-FALL-"))

    # -------------------------------------------------------------------------
    # Requirement 3: Probability below 0.85: fast path does not confirm
    # -------------------------------------------------------------------------
    def test_03_probability_below_threshold_does_not_fast_confirm(self):
        """If probability is below 0.85 (e.g. 0.84), fast path does not trigger."""
        res = self.fast_adapter.process_prediction(
            probability=0.84,
            dyn_acc_max=3.50,
            gyr_mag_max=350.0,
            window_start_time=0.0,
        )
        self.assertFalse(res["confirmed_fall"], "Probability < 0.85 must not trigger fast path")
        self.assertFalse(res["fast_path_triggered"])
        self.assertEqual(res["positive_count"], 1)
        self.assertEqual(res["temporal_state"], "POSSIBLE_FALL")

    # -------------------------------------------------------------------------
    # Requirement 4: Dynamic acceleration below 2.50 g: fast path does not confirm
    # -------------------------------------------------------------------------
    def test_04_dyn_acc_below_threshold_does_not_fast_confirm(self):
        """If dynamic acceleration is below 2.50 g (e.g. 2.49 g), fast path does not trigger."""
        res = self.fast_adapter.process_prediction(
            probability=0.95,
            dyn_acc_max=2.49,
            gyr_mag_max=300.0,
            window_start_time=0.0,
        )
        self.assertFalse(res["confirmed_fall"], "Dynamic accel < 2.50g must not trigger fast path")
        self.assertFalse(res["fast_path_triggered"])
        self.assertEqual(res["positive_count"], 1)

    # -------------------------------------------------------------------------
    # Requirement 5: Gyro below 200 dps: fast path does not confirm
    # -------------------------------------------------------------------------
    def test_05_gyro_below_threshold_does_not_fast_confirm(self):
        """If gyro magnitude is below 200 dps (e.g. 199.9 dps), fast path does not trigger."""
        res = self.fast_adapter.process_prediction(
            probability=0.95,
            dyn_acc_max=3.50,
            gyr_mag_max=199.9,
            window_start_time=0.0,
        )
        self.assertFalse(res["confirmed_fall"], "Gyro < 200 dps must not trigger fast path")
        self.assertFalse(res["fast_path_triggered"])
        self.assertEqual(res["positive_count"], 1)

    # -------------------------------------------------------------------------
    # Requirement 6: All fast-path conditions satisfied: no waiting for N=3
    # -------------------------------------------------------------------------
    def test_06_fast_path_latency_is_window_zero(self):
        """Validates that confirmation latency is 0-250ms (window index 0), without waiting for N=3."""
        res = self.fast_adapter.process_prediction(
            probability=0.88,
            dyn_acc_max=2.75,
            gyr_mag_max=240.0,
            window_start_time=0.0,
        )
        self.assertTrue(res["confirmed_fall"])
        self.assertEqual(res["positive_count"], 1)
        self.assertTrue(res["fast_path_triggered"])

    # -------------------------------------------------------------------------
    # Requirement 7: Baseline N=3 still works when fast path is enabled
    # -------------------------------------------------------------------------
    def test_07_baseline_n3_remains_active_parallel_fallback(self):
        """Moderate falls below fast thresholds still confirm via baseline N=3 rule."""
        # 3 consecutive windows above 0.60 but below fast thresholds (e.g. dyn_acc=1.2g, gyr=120dps)
        w0 = self.fast_adapter.process_prediction(0.70, dyn_acc_max=1.2, gyr_mag_max=120.0, window_start_time=0.0)
        self.assertFalse(w0["confirmed_fall"])
        self.assertFalse(w0["fast_path_triggered"])
        self.assertEqual(w0["positive_count"], 1)

        w1 = self.fast_adapter.process_prediction(0.72, dyn_acc_max=1.3, gyr_mag_max=130.0, window_start_time=0.25)
        self.assertFalse(w1["confirmed_fall"])
        self.assertFalse(w1["fast_path_triggered"])
        self.assertEqual(w1["positive_count"], 2)

        w2 = self.fast_adapter.process_prediction(0.75, dyn_acc_max=1.4, gyr_mag_max=140.0, window_start_time=0.50)
        self.assertTrue(w2["confirmed_fall"], "Window 3 must confirm via baseline N=3")
        self.assertFalse(w2["fast_path_triggered"], "Baseline confirmation must have fast_path_triggered=False")
        self.assertEqual(w2["positive_count"], 3)

    # -------------------------------------------------------------------------
    # Requirement 8: Fast path does not modify the baseline counter unexpectedly
    # -------------------------------------------------------------------------
    def test_08_consecutive_counter_tracking_and_sub_threshold_reset(self):
        """Positive counter accurately tracks prob >= 0.60 and resets on sub-threshold window."""
        w0 = self.fast_adapter.process_prediction(0.65, dyn_acc_max=1.0, gyr_mag_max=50.0, window_start_time=0.0)
        self.assertEqual(w0["positive_count"], 1)

        # Drop below baseline threshold (0.60)
        w1 = self.fast_adapter.process_prediction(0.58, dyn_acc_max=1.0, gyr_mag_max=50.0, window_start_time=0.25)
        self.assertEqual(w1["positive_count"], 0)
        self.assertEqual(w1["temporal_state"], "IDLE")

        # Next positive window restarts count at 1
        w2 = self.fast_adapter.process_prediction(0.62, dyn_acc_max=1.0, gyr_mag_max=50.0, window_start_time=0.50)
        self.assertEqual(w2["positive_count"], 1)

    # -------------------------------------------------------------------------
    # Requirement 9: Existing negative case: [0.75, 0.82, 0.30] does not confirm
    # -------------------------------------------------------------------------
    def test_09_canonical_two_window_spike_does_not_confirm(self):
        """Canonical negative sequence [0.75, 0.82, 0.30] produces 0 confirmed falls under default config."""
        probs = [0.75, 0.82, 0.30]
        alarms = []
        for i, p in enumerate(probs):
            res = self.default_adapter.process_prediction(p, window_start_time=i * 0.25)
            if res["confirmed_fall"]:
                alarms.append(i)
        self.assertEqual(len(alarms), 0, "Sequence [0.75, 0.82, 0.30] must produce zero alarms under default config")

        # Also when fast path is enabled with typical sub-fast physical metrics:
        self.fast_adapter.reset()
        alarms_fast = []
        for i, p in enumerate(probs):
            res = self.fast_adapter.process_prediction(
                p, dyn_acc_max=1.8, gyr_mag_max=150.0, window_start_time=i * 0.25
            )
            if res["confirmed_fall"]:
                alarms_fast.append(i)
        self.assertEqual(len(alarms_fast), 0, "Sequence with sub-fast metrics must produce zero alarms")

    # -------------------------------------------------------------------------
    # Requirement 10: Existing state-machine lockout behavior remains unchanged
    # -------------------------------------------------------------------------
    def test_10_post_fall_lockout_and_rearm_state_machine_unchanged(self):
        """After fast-path confirmation, 60-second post-fall lockout debounces further alerts."""
        res0 = self.fast_adapter.process_prediction(
            probability=0.90,
            dyn_acc_max=3.0,
            gyr_mag_max=250.0,
            window_start_time=0.0,
        )
        self.assertTrue(res0["confirmed_fall"])
        self.assertEqual(res0["temporal_state"], "CONFIRMED_FALL")

        # Subsequent window enters post-fall monitoring / lockout
        res1 = self.fast_adapter.process_prediction(
            probability=0.98,
            dyn_acc_max=4.0,
            gyr_mag_max=350.0,
            window_start_time=0.25,
        )
        self.assertFalse(res1["confirmed_fall"], "Must NOT confirm fall during post-fall monitoring")
        self.assertTrue(res1["cooldown_active"])
        self.assertEqual(res1["temporal_state"], "POST_FALL_MONITORING")

    # -------------------------------------------------------------------------
    # Requirement 11: Event generation and dictionary schema remain intact
    # -------------------------------------------------------------------------
    def test_11_event_dictionary_schema_completeness(self):
        """The output dictionary contains all required legacy keys plus additive fast_path_triggered."""
        res = self.fast_adapter.process_prediction(
            probability=0.90,
            dyn_acc_max=3.0,
            gyr_mag_max=250.0,
            window_start_time=0.0,
        )
        expected_keys = {
            "raw_probability",
            "raw_positive",
            "confirmed_fall",
            "fast_path_triggered",
            "temporal_state",
            "positive_count",
            "threshold",
            "confirmation_rule",
            "cooldown_active",
            "incident_id",
            "lock_elapsed_s",
            "stable_elapsed_s",
            "is_acknowledged",
        }
        for key in expected_keys:
            self.assertIn(key, res, f"Expected key '{key}' missing from temporal adapter output")

        self.assertIsInstance(res["raw_probability"], float)
        self.assertIsInstance(res["raw_positive"], bool)
        self.assertIsInstance(res["confirmed_fall"], bool)
        self.assertIsInstance(res["fast_path_triggered"], bool)
        self.assertIsInstance(res["cooldown_active"], bool)

    # -------------------------------------------------------------------------
    # Requirement 12: Configuration loader guarantees fast path disabled by default
    # -------------------------------------------------------------------------
    def test_12_config_loader_default_values(self):
        """Configuration loader defaults guarantee fast path is disabled by default."""
        cfg = load_confirmation_config()
        self.assertIn("fast_path_enabled", cfg)
        self.assertFalse(cfg["fast_path_enabled"], "fast_path_enabled must be false by default")
        self.assertEqual(cfg["fast_path_threshold"], 0.85)
        self.assertEqual(cfg["fast_path_accel_threshold_g"], 2.50)
        self.assertEqual(cfg["fast_path_gyro_threshold_dps"], 200.0)


if __name__ == "__main__":
    unittest.main()
