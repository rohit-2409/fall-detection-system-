"""
Stage 4E: Fall Confirmation Engine.

Implements an offline temporal and event-confirmation layer to filter isolated
false alarms produced by single-window machine learning classifiers while
preserving sensitivity to genuine falls.

Architecture:
  window -> model probability -> candidate fall -> event confirmation -> temporal confirmation -> final decision
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class FallConfirmationEngine:
    """
    Stateful confirmation engine processing a chronological stream of 50-sample
    sliding windows (0.50 s window, 0.25 s step at 100 Hz).
    """

    def __init__(
        self,
        model_threshold: float = 0.70,
        temporal_rule: str = "N2",
        require_event_confirmation: bool = True,
        min_dyn_acc: float = 1.80,    # Derived from Train/Val (p10 of falls is ~1.5-2.0g)
        min_gyr_mag: float = 100.0,    # Derived from Train/Val (p10 of falls is ~134 deg/s)
        buffer_len: int = 5
    ):
        """
        Parameters
        ----------
        model_threshold : float
            Probability threshold for declaring a candidate positive window.
        temporal_rule : str
            One of: 'N1' (single window), 'N2' (2 consecutive), 'N3' (3 consecutive),
            '2_of_3' (2 within 3 consecutive), '3_of_4' (3 within 4 consecutive).
        require_event_confirmation : bool
            Whether to enforce dynamic event gating (checking DynamicAccMag and GyrMag).
        min_dyn_acc : float
            Minimum dynamic acceleration magnitude (g) required within the event window.
        min_gyr_mag : float
            Minimum angular velocity magnitude (deg/s) required within the event window.
        buffer_len : int
            Length of the historical window buffer.
        """
        self.model_threshold = model_threshold
        self.temporal_rule = temporal_rule
        self.require_event_confirmation = require_event_confirmation
        self.min_dyn_acc = min_dyn_acc
        self.min_gyr_mag = min_gyr_mag
        self.buffer_len = buffer_len

        self.prob_history = deque(maxlen=buffer_len)
        self.candidate_history = deque(maxlen=buffer_len)
        self.event_confirmed_history = deque(maxlen=buffer_len)
        self.meta_history = deque(maxlen=buffer_len)
        self.alarm_active = False

    def reset(self):
        """Resets internal state between continuous recordings."""
        self.prob_history.clear()
        self.candidate_history.clear()
        self.event_confirmed_history.clear()
        self.meta_history.clear()
        self.alarm_active = False

    def process_window(
        self,
        prob: float,
        dyn_acc_max: float = 0.0,
        gyr_mag_max: float = 0.0,
        jerk_mag_max: float = 0.0,
        acc_mag_max: float = 0.0,
        window_start_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Processes a single window in chronological order.

        Returns
        -------
        Dict containing decision stages: candidate, event_confirmed, temporal_confirmed, fall_alarm.
        """
        # Stage 1: Candidate Detection
        is_candidate = (prob >= self.model_threshold)

        # Stage 2: Physical Dynamic Event Confirmation
        # Check if the window exhibits genuine dynamic fall dynamics rather than pure static tilt
        if self.require_event_confirmation:
            event_confirmed = is_candidate and (dyn_acc_max >= self.min_dyn_acc) and (gyr_mag_max >= self.min_gyr_mag)
        else:
            event_confirmed = is_candidate

        # Update FIFO history
        self.prob_history.append(prob)
        self.candidate_history.append(is_candidate)
        self.event_confirmed_history.append(event_confirmed)
        self.meta_history.append({
            "prob": prob,
            "dyn_acc_max": dyn_acc_max,
            "gyr_mag_max": gyr_mag_max,
            "time": window_start_time
        })

        # Stage 3: Temporal Rule Evaluation
        temporal_confirmed = self._evaluate_temporal_rule()

        # Final Alarm Trigger (latch or transient)
        fall_alarm = temporal_confirmed and event_confirmed

        return {
            "window_time": window_start_time,
            "probability": prob,
            "is_candidate": is_candidate,
            "event_confirmed": event_confirmed,
            "temporal_confirmed": temporal_confirmed,
            "fall_alarm": fall_alarm
        }

    def _evaluate_temporal_rule(self) -> bool:
        """Evaluates whether recent history satisfies the configured temporal rule."""
        hist = list(self.event_confirmed_history)
        n = len(hist)

        if self.temporal_rule == "N1":
            return bool(hist[-1]) if n >= 1 else False

        elif self.temporal_rule == "N2":
            return bool(n >= 2 and hist[-1] and hist[-2])

        elif self.temporal_rule == "N3":
            return bool(n >= 3 and hist[-1] and hist[-2] and hist[-3])

        elif self.temporal_rule == "2_of_3":
            if n < 2:
                return False
            sub = hist[-3:] if n >= 3 else hist[-2:]
            return sum(sub) >= 2

        elif self.temporal_rule == "3_of_4":
            if n < 3:
                return False
            sub = hist[-4:] if n >= 4 else hist[-3:]
            return sum(sub) >= 3

        else:
            return bool(hist[-1]) if n >= 1 else False


def simulate_stream_on_recording(
    recording_df: pd.DataFrame,
    clf,
    scaler,
    feature_columns: List[str],
    engine: FallConfirmationEngine,
) -> Dict[str, Any]:
    """
    Simulates chronological stream processing on an individual recording trial.

    Returns summary of alarms, detection latency, and window decisions.
    """
    engine.reset()
    rec_sorted = recording_df.sort_values("window_start_sample").copy()

    X_raw = rec_sorted[feature_columns].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    probs = clf.predict_proba(X_scaled)[:, 1]

    dyn_acc_arr = rec_sorted["DynamicAccMag_max"].values
    gyr_mag_arr = rec_sorted["GyrMag_max"].values
    jerk_mag_arr = rec_sorted["JerkMag_max"].values
    acc_mag_arr = rec_sorted["AccMag_max"].values
    time_arr = rec_sorted["window_start_time"].values
    labels_arr = rec_sorted["label"].values

    total_windows = len(rec_sorted)
    alarms = []
    first_alarm_time = None
    candidate_windows = 0

    for i in range(total_windows):
        p = float(probs[i])
        res = engine.process_window(
            prob=p,
            dyn_acc_max=float(dyn_acc_arr[i]),
            gyr_mag_max=float(gyr_mag_arr[i]),
            jerk_mag_max=float(jerk_mag_arr[i]),
            acc_mag_max=float(acc_mag_arr[i]),
            window_start_time=float(time_arr[i]),
        )
        if res["is_candidate"]:
            candidate_windows += 1
        if res["fall_alarm"]:
            alarms.append(i)
            if first_alarm_time is None:
                first_alarm_time = float(time_arr[i])

    # Cluster alarms into discrete events (runs separated by >= 2 non-alarm windows)
    discrete_events = 0
    in_event = False
    for i in range(total_windows):
        if i in alarms:
            if not in_event:
                discrete_events += 1
                in_event = True
        else:
            in_event = False

    is_fall_rec = rec_sorted["activity_id"].iloc[0].startswith("F") or (
        rec_sorted["dataset"].iloc[0] == "KFall" and int(rec_sorted["activity_id"].iloc[0][1:]) >= 22
    )

    fall_impact_time = None
    latency = None
    if is_fall_rec and "fall_impact_sample" in rec_sorted.columns:
        f_imp = rec_sorted["fall_impact_sample"].iloc[0]
        if f_imp > 0:
            fall_impact_time = f_imp / 100.0  # 100 Hz
            if first_alarm_time is not None:
                latency = round(first_alarm_time - (fall_impact_time - 0.40), 3)  # latency relative to fall onset/descent

    return {
        "recording_id": rec_sorted["recording_id"].iloc[0],
        "dataset": rec_sorted["dataset"].iloc[0],
        "subject_id": rec_sorted["subject_id"].iloc[0],
        "activity_id": rec_sorted["activity_id"].iloc[0],
        "activity_name": rec_sorted["activity_name"].iloc[0],
        "is_fall_recording": is_fall_rec,
        "total_windows": total_windows,
        "candidate_windows": candidate_windows,
        "alarm_windows": len(alarms),
        "discrete_alarm_events": discrete_events,
        "first_alarm_time": first_alarm_time,
        "fall_impact_time": fall_impact_time,
        "latency_seconds": latency,
        "detected": (discrete_events > 0),
    }
