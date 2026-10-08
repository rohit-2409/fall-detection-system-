"""
Temporal Confirmation Adapter Module (Stage 4F).

Provides a production-ready, modular, and reversible adapter wrapping the Stage 4E
temporal confirmation logic. Implements state tracking, duplicate-event debouncing,
gap detection, and explicit reset handling.

State Machine (Stage 7B Phase 2):
  IDLE -> POSSIBLE_FALL -> CONFIRMED_FALL -> POST_FALL_MONITORING -> LOCKED_ALARM -> WAIT_UPRIGHT_STABLE -> REARMED -> IDLE
"""

import datetime
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Import the Stage 4E engine and config loader
from src.fall_confirmation.config_loader import load_confirmation_config

# Ensure fall_confirmation_engine is importable
ENGINE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "fall_detection_dataset",
        "scripts",
        "stage_4E",
    )
)
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

try:
    from fall_confirmation_engine import FallConfirmationEngine
except ImportError:
    # Standalone fallback if path resolution differs
    class FallConfirmationEngine:
        def __init__(self, model_threshold=0.60, temporal_rule="N3", **kwargs):
            self.model_threshold = model_threshold
            self.temporal_rule = temporal_rule
            self.history = []
        def reset(self):
            self.history.clear()
        def process_window(self, prob, **kwargs):
            is_cand = prob >= self.model_threshold
            self.history.append(is_cand)
            if len(self.history) > 3:
                self.history.pop(0)
            n3 = len(self.history) == 3 and all(self.history)
            return {"is_candidate": is_cand, "temporal_confirmed": n3, "fall_alarm": n3}


class StateString(str):
    """
    Subclass of str supporting dual-identity equivalence between
    'POST_FALL_MONITORING' and 'COOLDOWN' for Stage 4F/5A/5F backward compatibility.
    """
    def __eq__(self, other: Any) -> bool:
        if str(self) == "POST_FALL_MONITORING" and other in ("POST_FALL_MONITORING", "COOLDOWN"):
            return True
        if str(self) == "COOLDOWN" and other in ("POST_FALL_MONITORING", "COOLDOWN"):
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self))


class TemporalConfirmationAdapter:
    """
    Modular stateful adapter for live or replay fall-detection inference.
    Implements Stage 7B Phase 2 One-Physical-Fall = One-Logical-Event state machine:
      IDLE -> POSSIBLE_FALL -> CONFIRMED_FALL -> POST_FALL_MONITORING ->
      LOCKED_ALARM -> WAIT_UPRIGHT_STABLE -> REARMED -> IDLE
    """

    STATE_IDLE = "IDLE"
    STATE_POSSIBLE_FALL = "POSSIBLE_FALL"
    STATE_CONFIRMED_FALL = "CONFIRMED_FALL"
    STATE_POST_FALL_MONITORING = StateString("POST_FALL_MONITORING")
    STATE_LOCKED_ALARM = "LOCKED_ALARM"
    STATE_WAIT_UPRIGHT_STABLE = "WAIT_UPRIGHT_STABLE"
    STATE_REARMED = "REARMED"
    # Backwards compatibility alias for Stage 4F / 5F callers
    STATE_COOLDOWN = STATE_POST_FALL_MONITORING

    def __init__(
        self,
        config_path: Optional[str] = None,
        lock_duration_s: Optional[float] = None,
        min_stable_duration_s: Optional[float] = None,
        min_upright_acc_g: Optional[float] = None,
        max_stable_dyn_acc_g: Optional[float] = None,
        max_stable_gyr_dps: Optional[float] = None,
        max_stable_prob: Optional[float] = None,
        fast_path_enabled: Optional[bool] = None,
        fast_path_threshold: Optional[float] = None,
        fast_path_accel_threshold_g: Optional[float] = None,
        fast_path_gyro_threshold_dps: Optional[float] = None,
    ):
        self.config = load_confirmation_config(config_path)

        self.threshold = float(self.config.get("model_threshold", 0.60))
        self.rule = str(self.config.get("temporal_rule", "N3"))
        self.event_gating = bool(self.config.get("require_event_confirmation", False))
        self.min_dyn_acc = float(self.config.get("min_dyn_acc", 1.80))
        self.min_gyr_mag = float(self.config.get("min_gyr_mag", 100.0))
        self.cooldown_limit = int(self.config.get("cooldown_windows", 8))
        self.gap_threshold_s = float(self.config.get("stream_gap_threshold_s", 0.50))

        # Fast path configuration (Candidate 3B opt-in)
        self.fast_path_enabled = bool(
            fast_path_enabled if fast_path_enabled is not None else self.config.get("fast_path_enabled", False)
        )
        self.fast_path_threshold = float(
            fast_path_threshold if fast_path_threshold is not None else self.config.get("fast_path_threshold", 0.85)
        )
        self.fast_path_accel_threshold_g = float(
            fast_path_accel_threshold_g if fast_path_accel_threshold_g is not None else self.config.get("fast_path_accel_threshold_g", 2.50)
        )
        self.fast_path_gyro_threshold_dps = float(
            fast_path_gyro_threshold_dps if fast_path_gyro_threshold_dps is not None else self.config.get("fast_path_gyro_threshold_dps", 200.0)
        )

        # Stage 7B Phase 2 incident lock & upright/stable re-arm parameters
        self.lock_duration_s = float(
            lock_duration_s if lock_duration_s is not None else self.config.get("lock_duration_s", 60.0)
        )
        self.min_stable_duration_s = float(
            min_stable_duration_s if min_stable_duration_s is not None else self.config.get("min_stable_duration_s", 5.0)
        )
        self.min_upright_acc_g = float(
            min_upright_acc_g if min_upright_acc_g is not None else self.config.get("min_upright_acc_g", 0.65)
        )
        self.max_stable_dyn_acc_g = float(
            max_stable_dyn_acc_g if max_stable_dyn_acc_g is not None else self.config.get("max_stable_dyn_acc_g", 0.35)
        )
        self.max_stable_gyr_dps = float(
            max_stable_gyr_dps if max_stable_gyr_dps is not None else self.config.get("max_stable_gyr_dps", 40.0)
        )
        self.max_stable_prob = float(
            max_stable_prob if max_stable_prob is not None else self.config.get("max_stable_prob", 0.40)
        )

        # Internal engine
        self.engine = FallConfirmationEngine(
            model_threshold=self.threshold,
            temporal_rule=self.rule,
            require_event_confirmation=self.event_gating,
            min_dyn_acc=self.min_dyn_acc,
            min_gyr_mag=self.min_gyr_mag,
        )

        # State machine variables
        self.state = self.STATE_IDLE
        self.consecutive_positives = 0
        self.cooldown_counter = 0
        self.last_window_time: Optional[float] = None
        self.total_confirmed_events = 0

        # Phase 2 incident latching & stability tracking
        self.incident_count = 0
        self.current_incident_id: Optional[str] = None
        self.lock_elapsed_s = 0.0
        self.stable_elapsed_s = 0.0
        self.is_acknowledged = False

    def reset(self):
        """Explicitly resets confirmation engine state."""
        self.engine.reset()
        self.state = self.STATE_IDLE
        self.consecutive_positives = 0
        self.cooldown_counter = 0
        self.last_window_time = None
        self.current_incident_id = None
        self.lock_elapsed_s = 0.0
        self.stable_elapsed_s = 0.0
        self.is_acknowledged = False

    def acknowledge_incident(self, incident_id: Optional[str] = None) -> bool:
        """
        Acknowledges the active incident.
        If in LOCKED_ALARM, immediately advances state to WAIT_UPRIGHT_STABLE.
        Returns True if an active incident was acknowledged, False otherwise.
        """
        if self.current_incident_id is None:
            return False
        if incident_id is not None and incident_id != self.current_incident_id:
            return False
        self.is_acknowledged = True
        if self.state == self.STATE_LOCKED_ALARM:
            self.state = self.STATE_WAIT_UPRIGHT_STABLE
            self.stable_elapsed_s = 0.0
        return True

    def process_prediction(
        self,
        probability: float,
        window_start_time: float = 0.0,
        dyn_acc_max: float = 0.0,
        gyr_mag_max: float = 0.0,
        jerk_mag_max: float = 0.0,
        acc_mag_max: float = 0.0,
        acc_z_mean: Optional[float] = None,
        acc_mean: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single window's model probability through the temporal confirmation layer.

        Parameters
        ----------
        probability : float
            Instantaneous fall probability from the ML model (0.0 to 1.0).
        window_start_time : float
            Timestamp of the window start in seconds.
        dyn_acc_max : float
            Peak dynamic acceleration in the window (g).
        gyr_mag_max : float
            Peak angular velocity magnitude in the window (deg/s).
        jerk_mag_max : float
            Peak jerk magnitude in the window.
        acc_mag_max : float
            Peak acceleration magnitude in the window (g).
        acc_z_mean : Optional[float]
            Mean Z-axis acceleration over the window (g).
        acc_mean : Optional[Tuple[float, float, float]]
            Mean 3D acceleration over the window (g, g, g).

        Returns
        -------
        Dict with keys:
            raw_probability: float
            raw_positive: bool (prob >= threshold)
            confirmed_fall: bool (true only on trigger window)
            temporal_state: str (IDLE, POSSIBLE_FALL, CONFIRMED_FALL,
                                 POST_FALL_MONITORING, LOCKED_ALARM,
                                 WAIT_UPRIGHT_STABLE, REARMED)
            positive_count: int
            threshold: float
            confirmation_rule: str
            cooldown_active: bool
            incident_id: Optional[str]
            lock_elapsed_s: float
            stable_elapsed_s: float
            is_acknowledged: bool
        """
        # Calculate time step delta
        if self.last_window_time is not None:
            time_delta = window_start_time - self.last_window_time
        else:
            time_delta = 0.0

        # Stream gap detection:
        # ONLY reset candidate buffer when in IDLE or POSSIBLE_FALL.
        # Active locked incidents MUST NOT be unlocked or reset by transient network/serial gaps.
        if self.last_window_time is not None and (time_delta < 0 or time_delta > self.gap_threshold_s):
            if self.state in (self.STATE_IDLE, self.STATE_POSSIBLE_FALL):
                self.reset()
            elif self.state == self.STATE_WAIT_UPRIGHT_STABLE:
                # If stream is interrupted during stability wait, reset continuity timer
                self.stable_elapsed_s = 0.0

        # Determine dt for state timers (fallback to nominal 0.25s window step if timestamps don't advance)
        dt = time_delta if time_delta > 0 else 0.25
        self.last_window_time = window_start_time

        # Raw single-window flag
        raw_positive = (probability >= self.threshold)
        confirmed_fall = False

        # -------------------------------------------------------------
        # STATE 1: POST_FALL_MONITORING (Debouncing post-fall shock)
        # -------------------------------------------------------------
        if self.state == self.STATE_POST_FALL_MONITORING:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_limit:
                self.state = self.STATE_LOCKED_ALARM
                self.cooldown_counter = 0
                self.lock_elapsed_s = 0.0

        # -------------------------------------------------------------
        # STATE 2: LOCKED_ALARM (Post-fall lock period, default 60s)
        # -------------------------------------------------------------
        elif self.state == self.STATE_LOCKED_ALARM:
            self.lock_elapsed_s += dt
            if self.is_acknowledged or self.lock_elapsed_s >= self.lock_duration_s:
                self.state = self.STATE_WAIT_UPRIGHT_STABLE
                self.stable_elapsed_s = 0.0

        # -------------------------------------------------------------
        # STATE 3: WAIT_UPRIGHT_STABLE (Sustained upright + low motion)
        # -------------------------------------------------------------
        elif self.state == self.STATE_WAIT_UPRIGHT_STABLE:
            # Check upright posture
            if acc_z_mean is not None:
                is_upright = (abs(acc_z_mean) >= self.min_upright_acc_g)
            elif acc_mean is not None:
                is_upright = (abs(acc_mean[2]) >= self.min_upright_acc_g)
            else:
                is_upright = True

            # Check low motion
            is_low_motion = (dyn_acc_max <= self.max_stable_dyn_acc_g) and (gyr_mag_max <= self.max_stable_gyr_dps)

            # Check low model probability
            is_low_prob = (probability < self.max_stable_prob)

            if is_upright and is_low_motion and is_low_prob:
                self.stable_elapsed_s += dt
                if self.stable_elapsed_s >= self.min_stable_duration_s:
                    self.state = self.STATE_REARMED
            else:
                # Any posture, motion, or model probability violation resets continuous stability timer
                self.stable_elapsed_s = 0.0

        # -------------------------------------------------------------
        # STATE 4 & 5: IDLE or POSSIBLE_FALL
        # -------------------------------------------------------------
        elif self.state in (self.STATE_IDLE, self.STATE_POSSIBLE_FALL):
            if raw_positive:
                self.consecutive_positives += 1
            else:
                self.consecutive_positives = 0

            # Feed to Stage 4E engine (baseline confirmation path)
            engine_res = self.engine.process_window(
                prob=probability,
                dyn_acc_max=dyn_acc_max,
                gyr_mag_max=gyr_mag_max,
                jerk_mag_max=jerk_mag_max,
                acc_mag_max=acc_mag_max,
                window_start_time=window_start_time,
            )

            baseline_confirmed = bool(engine_res["fall_alarm"])

            # Fast-path evaluation (Candidate 3B opt-in)
            fast_confirmed = False
            if self.fast_path_enabled:
                if (
                    probability >= self.fast_path_threshold
                    and dyn_acc_max >= self.fast_path_accel_threshold_g
                    and gyr_mag_max >= self.fast_path_gyro_threshold_dps
                ):
                    fast_confirmed = True

            confirmed_fall = fast_confirmed or baseline_confirmed

            if confirmed_fall:
                self.state = self.STATE_CONFIRMED_FALL
                self.total_confirmed_events += 1
                self.incident_count += 1
                now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                self.current_incident_id = f"STAGE6L-FALL-{now_str}-{self.incident_count:02d}"
                self.cooldown_counter = 0
                self.lock_elapsed_s = 0.0
                self.stable_elapsed_s = 0.0
                self.is_acknowledged = False
            elif self.consecutive_positives > 0:
                self.state = self.STATE_POSSIBLE_FALL
            else:
                self.state = self.STATE_IDLE

        state_for_report = self.state
        cooldown_active = (state_for_report in (
            self.STATE_POST_FALL_MONITORING,
            self.STATE_LOCKED_ALARM,
            self.STATE_WAIT_UPRIGHT_STABLE,
            self.STATE_REARMED,
        ))

        out_dict = {
            "raw_probability": round(probability, 4),
            "raw_positive": raw_positive,
            "confirmed_fall": confirmed_fall,
            "fast_path_triggered": fast_confirmed if 'fast_confirmed' in locals() else False,
            "temporal_state": state_for_report,
            "positive_count": self.consecutive_positives if state_for_report in (
                self.STATE_IDLE, self.STATE_POSSIBLE_FALL, self.STATE_CONFIRMED_FALL
            ) else 0,
            "threshold": self.threshold,
            "confirmation_rule": self.rule,
            "cooldown_active": cooldown_active,
            "incident_id": self.current_incident_id,
            "lock_elapsed_s": round(self.lock_elapsed_s, 2),
            "stable_elapsed_s": round(self.stable_elapsed_s, 2),
            "is_acknowledged": self.is_acknowledged,
        }

        # Transition next-state lifecycle
        if self.state == self.STATE_CONFIRMED_FALL:
            self.state = self.STATE_POST_FALL_MONITORING
        elif self.state == self.STATE_REARMED:
            # Self-transition back to IDLE ready for next incident
            self.engine.reset()
            self.state = self.STATE_IDLE
            self.consecutive_positives = 0
            self.cooldown_counter = 0
            self.lock_elapsed_s = 0.0
            self.stable_elapsed_s = 0.0
            self.is_acknowledged = False
            self.current_incident_id = None

        return out_dict

    process_window = process_prediction
