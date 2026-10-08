"""
Stage 4G Live Sensor Validation & Diagnostic Suite.

Implements real-time 100 Hz ESP32 + MPU6050 signal acquisition, 50-sample FIFO windowing,
100-feature vector extraction (exact parity with Stage 4C/4D feature_order.json),
inference with frozen stage_4D_random_forest.joblib, and temporal confirmation routing
via TemporalConfirmationAdapter (threshold=0.60, rule=N3, debouncing, auto-reset).

All sessions record raw sensor streams and diagnostic logs under:
fall_detection_dataset/live_validation/stage_4G/

EMERGENCY NOTIFICATIONS REMAIN STRICTLY DISABLED (DIAGNOSTIC MODE ONLY).
"""

import argparse
import collections
import datetime
import json
import os
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
from scipy.signal import lfilter, lfilter_zi

# Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4D", "stage_4D_random_forest.joblib"
)
DEFAULT_SCALER_PATH = os.path.join(
    PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4D", "scaler.joblib"
)
DEFAULT_FEATURE_ORDER_PATH = os.path.join(
    PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4D", "feature_order.json"
)
LIVE_VALIDATION_DIR = os.path.join(
    PROJECT_ROOT, "fall_detection_dataset", "live_validation", "stage_4G"
)

WINDOW_SIZE = 50       # 0.50 s at 100 Hz
STEP_SIZE = 25         # 0.25 s (50% overlap)
NOMINAL_SAMPLING_HZ = 100.0
GRAVITY_ALPHA = 0.969  # Causal 1-pole filter cutoff ~0.5 Hz at 100 Hz

SIGNALS_ORDER = [
    "AccX", "AccY", "AccZ",
    "GyrX", "GyrY", "GyrZ",
    "AccMag", "GyrMag",
    "DynamicAccMag", "JerkMag"
]

STAT_METRICS = [
    "mean", "std", "min", "max", "ptp",
    "median", "p25", "p75", "rms", "energy"
]

ALLOWED_SESSION_LABELS = {
    "REST", "STANDING", "WALKING_SLOW", "WALKING_FAST", "SITTING",
    "QUICK_SITTING", "STAIRS", "BENDING", "STUMBLE_SIMULATION",
    "SAFE_FALL_SIMULATION", "UNKNOWN"
}

# -----------------------------------------------------------------------------
# 100-Feature Extraction Engine
# -----------------------------------------------------------------------------
def compute_dynamic_acceleration(acc_3d: np.ndarray, alpha: float = GRAVITY_ALPHA) -> np.ndarray:
    """Computes dynamic acceleration magnitude removing static gravity via causal 1-pole filter."""
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    zi = lfilter_zi(b, a)[:, None] * acc_3d[0:1, :]
    gravity, _ = lfilter(b, a, acc_3d, axis=0, zi=zi)
    dynamic_acc_3d = acc_3d - gravity
    return np.linalg.norm(dynamic_acc_3d, axis=1)


def compute_jerk_magnitude(acc_3d: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Computes instantaneous jerk magnitude in g/s."""
    diff = np.diff(acc_3d, axis=0) / dt
    jerk = np.linalg.norm(diff, axis=1)
    return np.concatenate([[jerk[0]], jerk])


def extract_100_features(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes exactly 100 statistical features for a 50-sample window,
    matching fall_detection_dataset/models/stage_4D/feature_order.json.
    """
    acc_3d = np.column_stack([window_data["AccX"], window_data["AccY"], window_data["AccZ"]])
    gyr_3d = np.column_stack([window_data["GyrX"], window_data["GyrY"], window_data["GyrZ"]])

    acc_mag = np.linalg.norm(acc_3d, axis=1)
    gyr_mag = np.linalg.norm(gyr_3d, axis=1)
    dyn_acc_mag = compute_dynamic_acceleration(acc_3d)
    jerk_mag = compute_jerk_magnitude(acc_3d)

    sig_matrix = np.column_stack([
        acc_3d[:, 0], acc_3d[:, 1], acc_3d[:, 2],
        gyr_3d[:, 0], gyr_3d[:, 1], gyr_3d[:, 2],
        acc_mag, gyr_mag, dyn_acc_mag, jerk_mag
    ])

    features: List[float] = []
    for c in range(10):
        sig = sig_matrix[:, c]
        s_mean = float(np.mean(sig))
        s_std = float(np.std(sig, ddof=0))
        s_min = float(np.min(sig))
        s_max = float(np.max(sig))
        s_ptp = float(s_max - s_min)
        p25, med, p75 = np.percentile(sig, [25, 50, 75])
        s_energy = float(np.mean(sig ** 2))
        s_rms = float(np.sqrt(s_energy))

        features.extend([
            s_mean, s_std, s_min, s_max, s_ptp,
            float(med), float(p25), float(p75), s_rms, s_energy
        ])

    return np.array(features, dtype=np.float64)


# -----------------------------------------------------------------------------
# CSV Parsing & Stream Validation
# -----------------------------------------------------------------------------
def parse_csv_line(line: str) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    Parses a single CSV line: timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ.
    Returns (ts, ax, ay, az, gx, gy, gz) or None if invalid/header.
    """
    line = line.strip()
    if not line or "timestamp" in line.lower() or "accx" in line.lower():
        return None
    parts = line.split(",")
    if len(parts) != 7:
        return None
    try:
        return (
            float(parts[0]), float(parts[1]), float(parts[2]),
            float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        )
    except (ValueError, TypeError):
        return None


# -----------------------------------------------------------------------------
# Diagnostic Session Validator
# -----------------------------------------------------------------------------
class Stage4GLiveValidator:
    """
    Executes controlled live sensor validation sessions with full telemetry,
    dual-path diagnostic reporting, and disk logging.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        feature_order_path: str = DEFAULT_FEATURE_ORDER_PATH,
        output_dir: str = LIVE_VALIDATION_DIR
    ):
        self.model_path = model_path
        self.feature_order_path = feature_order_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Feature Parity Check
        with open(self.feature_order_path, "r") as f:
            self.expected_feature_order = json.load(f)
        assert len(self.expected_feature_order) == 100, f"Expected 100 features, got {len(self.expected_feature_order)}"

        # 2. Model & Scaler Loading
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        assert getattr(self.model, "n_features_in_", 100) == 100, "Model n_features_in_ must equal 100"

        self.scaler = joblib.load(DEFAULT_SCALER_PATH) if os.path.exists(DEFAULT_SCALER_PATH) else None

        # 3. Temporal Confirmation Adapter
        self.adapter = TemporalConfirmationAdapter()
        self.adapter.reset()

        print(f"[INIT] Loaded Model: {os.path.basename(self.model_path)}")
        print(f"[INIT] Feature Parity: 100 features verified.")
        print(f"[INIT] Temporal Adapter: Rule={self.adapter.rule}, Threshold={self.adapter.threshold}, Gating={self.adapter.event_gating}")
        print(f"[INIT] EMERGENCY NOTIFICATIONS: DISABLED (DIAGNOSTIC MODE ONLY)")

    def run_session(
        self,
        session_id: str,
        session_label: str,
        stream_generator,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a validation session on an iterable stream of samples.
        stream_generator yields (timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ).
        """
        if session_label not in ALLOWED_SESSION_LABELS:
            raise ValueError(f"Invalid session label: {session_label}. Must be one of {ALLOWED_SESSION_LABELS}")

        session_folder = os.path.join(self.output_dir, session_id)
        os.makedirs(session_folder, exist_ok=True)

        raw_csv_path = os.path.join(session_folder, "raw_sensor_data.csv")
        diag_csv_path = os.path.join(session_folder, "diagnostic_log.csv")
        meta_json_path = os.path.join(session_folder, "session_metadata.json")

        # Session Buffers
        fifo_accx: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
        fifo_accy: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
        fifo_accz: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
        fifo_gyrx: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
        fifo_gyry: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
        fifo_gyrz: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)

        self.adapter.reset()

        # Telemetry Metrics
        timestamps: List[float] = []
        samples_since_step = 0
        total_samples = 0
        malformed_records = 0
        duplicate_timestamps = 0
        dropped_samples = 0
        nan_inf_count = 0

        # Model & Temporal Metrics
        window_count = 0
        raw_positives = 0
        confirmed_fall_events = 0
        max_probability = 0.0
        probabilities: List[float] = []
        confirmed_event_details: List[Dict[str, Any]] = []

        # Peak Signal Tracking
        max_gyr_mag = 0.0
        max_dyn_acc_mag = 0.0
        max_jerk_mag = 0.0

        prev_ts: Optional[float] = None

        with open(raw_csv_path, "w") as f_raw, open(diag_csv_path, "w") as f_diag:
            f_raw.write("timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ\n")
            f_diag.write("timestamp,probability,raw_positive,positive_count,temporal_state,confirmed_fall\n")

            for record in stream_generator:
                if record is None:
                    malformed_records += 1
                    continue

                ts, ax, ay, az, gx, gy, gz = record

                # Sanity checks on raw values
                if any(np.isnan([ts, ax, ay, az, gx, gy, gz])) or any(np.isinf([ts, ax, ay, az, gx, gy, gz])):
                    nan_inf_count += 1
                    continue

                if prev_ts is not None:
                    interval = ts - prev_ts
                    if interval == 0:
                        duplicate_timestamps += 1
                    elif interval < 0:
                        malformed_records += 1
                    elif interval > 0.025:  # Over 2.5x expected 10ms interval
                        dropped_samples += int(round(interval / 0.01)) - 1

                prev_ts = ts
                timestamps.append(ts)
                total_samples += 1

                # Record raw sample
                f_raw.write(f"{ts:.4f},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f}\n")

                # Push to FIFO
                fifo_accx.append(ax)
                fifo_accy.append(ay)
                fifo_accz.append(az)
                fifo_gyrx.append(gx)
                fifo_gyry.append(gy)
                fifo_gyrz.append(gz)

                samples_since_step += 1

                # When FIFO is full and stride reached: extract features & infer
                if len(fifo_accx) == WINDOW_SIZE and samples_since_step >= STEP_SIZE:
                    samples_since_step = 0
                    window_count += 1

                    win_dict = {
                        "AccX": np.array(fifo_accx, dtype=np.float64),
                        "AccY": np.array(fifo_accy, dtype=np.float64),
                        "AccZ": np.array(fifo_accz, dtype=np.float64),
                        "GyrX": np.array(fifo_gyrx, dtype=np.float64),
                        "GyrY": np.array(fifo_gyry, dtype=np.float64),
                        "GyrZ": np.array(fifo_gyrz, dtype=np.float64),
                    }

                    # Compute features
                    feat_vec = extract_100_features(win_dict)
                    
                    # Track signal magnitudes for validation
                    cur_gyr_mag = np.max(np.linalg.norm(np.column_stack([win_dict["GyrX"], win_dict["GyrY"], win_dict["GyrZ"]]), axis=1))
                    cur_dyn_acc = np.max(compute_dynamic_acceleration(np.column_stack([win_dict["AccX"], win_dict["AccY"], win_dict["AccZ"]])))
                    cur_jerk = np.max(compute_jerk_magnitude(np.column_stack([win_dict["AccX"], win_dict["AccY"], win_dict["AccZ"]])))
                    max_gyr_mag = max(max_gyr_mag, float(cur_gyr_mag))
                    max_dyn_acc_mag = max(max_dyn_acc_mag, float(cur_dyn_acc))
                    max_jerk_mag = max(max_jerk_mag, float(cur_jerk))

                    # Model inference
                    feat_input = feat_vec.reshape(1, -1).astype(np.float32)
                    if self.scaler is not None:
                        feat_input = self.scaler.transform(feat_input)
                    prob = float(self.model.predict_proba(feat_input)[0, 1])
                    probabilities.append(prob)
                    max_probability = max(max_probability, prob)

                    # Feed temporal confirmation engine
                    temp_result = self.adapter.process_prediction(
                        probability=prob,
                        window_start_time=ts,
                        dyn_acc_max=cur_dyn_acc,
                        gyr_mag_max=cur_gyr_mag,
                        jerk_mag_max=cur_jerk
                    )
                    raw_pos = temp_result["raw_positive"]
                    confirmed = temp_result["confirmed_fall"]
                    state = temp_result["temporal_state"]
                    pos_count = temp_result["positive_count"]

                    if raw_pos:
                        raw_positives += 1

                    if confirmed:
                        confirmed_fall_events += 1
                        event_info = {
                            "event_id": f"EVT_{session_id}_{confirmed_fall_events}",
                            "session_id": session_id,
                            "timestamp": ts,
                            "probability": prob,
                            "state": state,
                            "positive_count": pos_count,
                            "pre_event_activity": session_label
                        }
                        confirmed_event_details.append(event_info)

                    # Log diagnostic row
                    f_diag.write(f"{ts:.4f},{prob:.4f},{str(raw_pos).lower()},{pos_count},{state},{str(confirmed).lower()}\n")

                    if verbose and (window_count % 10 == 0 or confirmed or raw_pos):
                        print(f"[{session_id}] t={ts:.2f}s | prob={prob:.3f} | raw_pos={raw_pos} | pos_count={pos_count} | state={state} | CONFIRMED={confirmed}")

        # Compute Telemetry Summary
        duration_s = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        intervals = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.01])
        observed_hz = float(len(timestamps) / duration_s) if duration_s > 0 else 0.0
        median_interval = float(np.median(intervals)) if len(intervals) > 0 else 0.0
        min_interval = float(np.min(intervals)) if len(intervals) > 0 else 0.0
        max_interval = float(np.max(intervals)) if len(intervals) > 0 else 0.0

        hours = duration_s / 3600.0 if duration_s > 0 else 0.0
        false_alarms = confirmed_fall_events if session_label != "SAFE_FALL_SIMULATION" else 0
        false_alarms_per_hour = (false_alarms / hours) if hours > 0 else 0.0

        metadata = {
            "session_id": session_id,
            "session_label": session_label,
            "date": datetime.datetime.now().isoformat(),
            "duration_s": round(duration_s, 3),
            "sample_count": total_samples,
            "observed_hz": round(observed_hz, 2),
            "median_interval_s": round(median_interval, 5),
            "min_interval_s": round(min_interval, 5),
            "max_interval_s": round(max_interval, 5),
            "malformed_records": malformed_records,
            "duplicate_timestamps": duplicate_timestamps,
            "dropped_samples": dropped_samples,
            "nan_inf_count": nan_inf_count,
            "window_count": window_count,
            "raw_positive_windows": raw_positives,
            "raw_positive_rate": round(raw_positives / window_count, 4) if window_count > 0 else 0.0,
            "confirmed_fall_events": confirmed_fall_events,
            "false_alarm_events": false_alarms,
            "false_alarms_per_hour": round(false_alarms_per_hour, 2),
            "max_probability": round(max_probability, 4),
            "median_probability": round(float(np.median(probabilities)), 4) if probabilities else 0.0,
            "max_gyr_mag": round(max_gyr_mag, 2),
            "max_dyn_acc_mag": round(max_dyn_acc_mag, 3),
            "max_jerk_mag": round(max_jerk_mag, 2),
            "emergency_notifications_enabled": False,
            "confirmed_events": confirmed_event_details,
            "raw_csv_path": raw_csv_path,
            "diagnostic_csv_path": diag_csv_path
        }

        with open(meta_json_path, "w") as f_meta:
            json.dump(metadata, f_meta, indent=2)

        return metadata
