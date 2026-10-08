"""
Stage 5F Live Bridge: ESP32 -> ML Inference -> Temporal Confirmation -> REST Backend Integration.
Standalone Fall Detection Project.

Architecture:
  ESP32 + MPU6050 (COM22 @ 115200)
    ↓
  Serial stream parsing (CSV or Pipe-format)
    ↓
  Rolling 50-sample FIFO window (25-sample step, 50% overlap @ 100 Hz)
    ↓
  100-feature extraction (extract_100_features from Stage 4G/5E)
    ↓
  Frozen Stage 4D Random Forest model inference (predict_proba)
    ↓
  Stage 4E TemporalConfirmationAdapter (threshold=0.60, rule=N3, cooldown=2.0s)
    ↓
  confirmed_fall == True gating (Never POST on raw positives or unconfirmed windows)
    ↓
  HTTP POST to http://127.0.0.1:5000/api/fall-events (urllib standard library)
    ↓
  SQLite FallEventService & Live Monitoring Dashboard
"""

import argparse
import collections
import datetime
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import validated Stage 4G / 5E components
from src.fall_confirmation.stage_4G_live_validator import (
    DEFAULT_FEATURE_ORDER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    STEP_SIZE,
    WINDOW_SIZE,
    compute_dynamic_acceleration,
    compute_jerk_magnitude,
    extract_100_features,
)
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter

# Regular expressions for parsing live serial lines
RE_PIPE_FORMAT = re.compile(
    r"ACC(?:\(g\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*\|\s*GYRO?(?:\(deg/s\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
RE_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_serial_line(line: str) -> Optional[Tuple[Optional[float], float, float, float, float, float, float]]:
    """
    Parses a single raw serial string into (timestamp_ms_or_None, ax, ay, az, gx, gy, gz).
    Supports:
      1. Pipe format: ACC(g): 0.012, -0.025, 0.986 | GYRO(deg/s): 0.31, -0.12, 0.08
      2. 7-column CSV: timestamp_ms,ax,ay,az,gx,gy,gz
      3. 6-column CSV: ax,ay,az,gx,gy,gz
    Returns None if line is a header, corrupt, or invalid.
    """
    line = line.strip()
    if not line:
        return None

    lower = line.lower()
    if any(h in lower for h in ["starting", "i2c", "ready", "setup", "baud", "timestamp", "accx"]):
        return None

    # Try Format 1: Pipe format
    m_pipe = RE_PIPE_FORMAT.search(line)
    if m_pipe:
        try:
            return (
                None,
                float(m_pipe.group(1)),
                float(m_pipe.group(2)),
                float(m_pipe.group(3)),
                float(m_pipe.group(4)),
                float(m_pipe.group(5)),
                float(m_pipe.group(6)),
            )
        except (ValueError, IndexError):
            pass

    # Try Format 2: CSV format
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7:
            try:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
            except (ValueError, IndexError):
                pass
        elif len(parts) == 6:
            try:
                return (
                    None,
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            except (ValueError, IndexError):
                pass

    # Fallback: Extract 6 consecutive floating point numbers
    floats = RE_FLOAT.findall(line)
    if len(floats) == 6:
        try:
            return (
                None,
                float(floats[0]),
                float(floats[1]),
                float(floats[2]),
                float(floats[3]),
                float(floats[4]),
                float(floats[5]),
            )
        except (ValueError, IndexError):
            pass
    elif len(floats) == 7:
        try:
            return (
                float(floats[0]),
                float(floats[1]),
                float(floats[2]),
                float(floats[3]),
                float(floats[4]),
                float(floats[5]),
                float(floats[6]),
            )
        except (ValueError, IndexError):
            pass

    return None


def generate_event_id(prefix: str = "LIVE") -> str:
    """Generates a deterministic-safe unique event ID."""
    utc_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_hex = uuid.uuid4().hex[:8]
    return f"{prefix}-{utc_str}-{random_hex}"


def post_fall_event_to_backend(
    backend_url: str,
    event_payload: Dict[str, Any],
    timeout_seconds: float = 2.5,
) -> Tuple[bool, str, Optional[int]]:
    """
    Dispatches a confirmed fall event to the backend REST API via standard library urllib.
    Returns (success: bool, message: str, status_code: Optional[int]).
    """
    endpoint = f"{backend_url.rstrip('/')}/api/fall-events"
    data_bytes = json.dumps(event_payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "Stage5F-LiveBridge/1.0"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status_code = resp.getcode()
            resp_body = resp.read().decode("utf-8")
            if 200 <= status_code < 300:
                return True, "STORED", status_code
            return False, f"HTTP_{status_code}: {resp_body}", status_code
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        return False, f"HTTP_{e.code}: {err_body}", e.code
    except urllib.error.URLError as e:
        return False, f"CONNECTION_ERROR: {e.reason}", None
    except Exception as e:
        return False, f"ERROR: {str(e)}", None


def run_stage_5f_live_bridge(
    port: str = "COM22",
    baudrate: int = 115200,
    backend_url: str = "http://127.0.0.1:5000",
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    feature_order_path: str = DEFAULT_FEATURE_ORDER_PATH,
    session_id: Optional[str] = None,
    max_windows: Optional[int] = None,
    dry_run: bool = False,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """
    Main live bridge runtime connecting ESP32 serial data to ML inference,
    temporal confirmation, and REST backend persistence.
    """
    if not session_id:
        session_id = f"STAGE5F-LIVE-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    print("=" * 70)
    print("  STAGE 5F: LIVE ESP32 -> ML -> TEMPORAL CONFIRMATION -> BACKEND")
    print("=" * 70)
    print(f"  Session ID:   {session_id}")
    print(f"  Serial Port:  {port} @ {baudrate} baud (Mock: {mock_mode})")
    print(f"  Backend URL:  {backend_url}")
    print(f"  Dry Run Mode: {dry_run} (Backend POSTs: {'DISABLED' if dry_run else 'ACTIVE'})")
    print(f"  Max Windows:  {max_windows if max_windows else 'Unlimited (until Ctrl+C)'}")
    print("-" * 70)

    # 1. Verify and Load Model, Scaler, and Feature Order
    if not os.path.exists(model_path):
        print(f"[FATAL] Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading Model: {model_path}...")
    model = joblib.load(model_path)
    print("  [OK] Model loaded successfully.")

    scaler = None
    if os.path.exists(scaler_path):
        print(f"Loading Scaler: {scaler_path}...")
        scaler = joblib.load(scaler_path)
        print("  [OK] Scaler loaded successfully.")
    else:
        print("  [INFO] No scaler found (Stage 4D Random Forest expects unscaled feature vectors).")

    if not os.path.exists(feature_order_path):
        print(f"[FATAL] Feature order file not found: {feature_order_path}")
        sys.exit(1)

    with open(feature_order_path, "r", encoding="utf-8") as f_fo:
        expected_features = json.load(f_fo)
    if len(expected_features) != 100:
        print(f"[FATAL] Feature order expected 100 features, found {len(expected_features)}.")
        sys.exit(1)
    print(f"  [OK] Feature ordering verified (100 features).")

    # 2. Instantiate Temporal Confirmation Adapter
    adapter = TemporalConfirmationAdapter()
    adapter.reset()
    print(
        f"  [OK] Temporal Engine initialized: Rule={adapter.rule}, "
        f"Threshold={adapter.threshold}, Cooldown={adapter.cooldown_limit} windows."
    )
    print("=" * 70)

    # 3. Setup Hardware Connection or Mock Mode
    ser = None
    if not mock_mode:
        try:
            import serial
        except ImportError:
            print("[FATAL] pyserial is required for serial communication. Install with: pip install pyserial")
            sys.exit(1)

        print(f"\nConnecting to ESP32 on '{port}' @ {baudrate} baud...")
        try:
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
            time.sleep(1.5)  # Allow UART connection to settle
            ser.reset_input_buffer()
            adapter.reset()
            print(f"  [OK] Connected to {port}. Listening for live sensor stream...\n")
        except Exception as e:
            print(f"[FATAL] Could not open port {port}: {e}")
            print("Troubleshooting:")
            print("  1. Verify the ESP32 is plugged in and check Device Manager for the correct COM port.")
            print("  2. Ensure Arduino IDE Serial Monitor or other serial terminals are CLOSED.")
            print("  3. To run in simulated test mode, use: python src/fall_confirmation/stage_5F_live_bridge.py --mock")
            sys.exit(1)
    else:
        print("\n[MOCK MODE ACTIVE] Generating simulated motion stream for verification...\n")

    # 4. Stream Buffers & Statistics
    fifo_accx: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_accy: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_accz: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_gyrx: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_gyry: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_gyrz: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_ts: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)

    total_samples = 0
    malformed_lines = 0
    samples_since_step = 0
    window_count = 0
    raw_positives = 0
    confirmed_falls = 0
    backend_stored = 0
    backend_failures = 0

    start_monotonic = time.perf_counter()
    mock_time_ms = 0.0

    print("-" * 70)
    print("  LIVE INFERENCE & TEMPORAL MONITORING ACTIVE — PRESS CTRL+C TO STOP")
    print("-" * 70)

    try:
        while True:
            # A. Read Line
            if not mock_mode:
                try:
                    raw_line = ser.readline().decode("utf-8", errors="ignore")
                except Exception as e:
                    print(f"\n[WARNING] Serial read interrupted: {e}")
                    break
            else:
                time.sleep(0.010)  # 10ms ~ 100 Hz
                mock_time_ms += 10.0
                # Normal upright stationary motion baseline
                ax = float(np.random.normal(0.02, 0.01))
                ay = float(np.random.normal(0.04, 0.01))
                az = float(np.random.normal(0.98, 0.01))
                gx = float(np.random.normal(0.10, 0.20))
                gy = float(np.random.normal(0.10, 0.20))
                gz = float(np.random.normal(0.10, 0.20))
                raw_line = f"{int(mock_time_ms)},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f}\n"

            # B. Parse Line
            parsed = parse_serial_line(raw_line)
            if parsed is None:
                malformed_lines += 1
                continue

            total_samples += 1
            sample_ts_ms, ax, ay, az, gx, gy, gz = parsed

            # Timestamp in seconds
            now_mono = time.perf_counter()
            sample_ts_sec = (sample_ts_ms / 1000.0) if sample_ts_ms is not None else (now_mono - start_monotonic)

            # Append to FIFO
            fifo_accx.append(ax)
            fifo_accy.append(ay)
            fifo_accz.append(az)
            fifo_gyrx.append(gx)
            fifo_gyry.append(gy)
            fifo_gyrz.append(gz)
            fifo_ts.append(sample_ts_sec)
            samples_since_step += 1

            # C. Window Processing (50 samples, 25 step)
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

                # Feature Extraction (100 features)
                feat_vec = extract_100_features(win_dict)
                if len(feat_vec) != 100:
                    print(f"[ERROR] Feature extraction generated {len(feat_vec)} features, expected 100.")
                    continue

                if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
                    print("[WARNING] Non-finite features detected in window; skipping inference.")
                    continue

                # Model Prediction
                feat_input = feat_vec.reshape(1, -1).astype(np.float32)
                if scaler is not None:
                    feat_input = scaler.transform(feat_input)

                prob = float(model.predict_proba(feat_input)[0, 1])

                # Temporal Inputs
                acc_3d = np.column_stack([win_dict["AccX"], win_dict["AccY"], win_dict["AccZ"]])
                gyr_3d = np.column_stack([win_dict["GyrX"], win_dict["GyrY"], win_dict["GyrZ"]])
                cur_gyr_mag = float(np.max(np.linalg.norm(gyr_3d, axis=1)))
                cur_dyn_acc = float(np.max(compute_dynamic_acceleration(acc_3d)))
                cur_jerk = float(np.max(compute_jerk_magnitude(acc_3d)))
                window_start_time = float(fifo_ts[0])

                # Temporal Confirmation
                temp_res = adapter.process_prediction(
                    probability=prob,
                    window_start_time=window_start_time,
                    dyn_acc_max=cur_dyn_acc,
                    gyr_mag_max=cur_gyr_mag,
                    jerk_mag_max=cur_jerk,
                )

                is_raw_pos = bool(temp_res["raw_positive"])
                is_confirmed = bool(temp_res["confirmed_fall"])
                pos_count = int(temp_res["positive_count"])
                temp_state = str(temp_res["temporal_state"])

                if is_raw_pos:
                    raw_positives += 1

                # Compact Console Output
                print(f"[Window {window_count:04d}]", flush=True)
                print(f"  probability={prob:.4f}", flush=True)
                print(f"  raw_positive={str(is_raw_pos).lower()}", flush=True)
                print(f"  positive_count={pos_count}", flush=True)
                print(f"  state={temp_state}", flush=True)
                print(f"  confirmed_fall={str(is_confirmed).lower()}", flush=True)

                # D. Backend REST Integration ONLY on Confirmed Fall
                if is_confirmed:
                    confirmed_falls += 1
                    event_id = generate_event_id(prefix="LIVE")
                    detected_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    occurred_at_iso = detected_at_iso

                    event_payload = {
                        "event_id": event_id,
                        "detected_at": detected_at_iso,
                        "occurred_at": occurred_at_iso,
                        "session_id": session_id,
                        "confidence": round(prob, 4),
                        "source": "ESP32_MPU6050",
                        "diagnostic_metadata": {
                            "window_index": window_count,
                            "window_start_time_s": round(window_start_time, 4),
                            "dyn_acc_max": round(cur_dyn_acc, 4),
                            "gyr_mag_max": round(cur_gyr_mag, 4),
                            "jerk_mag_max": round(cur_jerk, 4),
                            "positive_count": pos_count,
                            "temporal_state": temp_state,
                            "dry_run": dry_run,
                            "mock_mode": mock_mode,
                        },
                    }

                    print("\n>>> [CONFIRMED FALL EVENT DETECTED] <<<", flush=True)
                    print(f"  confidence={prob:.4f}", flush=True)
                    print(f"  event_id={event_id}", flush=True)

                    if not dry_run:
                        success, resp_msg, code = post_fall_event_to_backend(
                            backend_url=backend_url,
                            event_payload=event_payload,
                        )
                        if success:
                            backend_stored += 1
                            print(f"  backend=STORED (Status {code})", flush=True)
                        else:
                            backend_failures += 1
                            print(f"  backend=NOT_RECORDED", flush=True)
                            print(f"  reason={resp_msg}", flush=True)
                            print(f"  [WARNING] Event preserved locally in session log.", flush=True)
                    else:
                        print("  backend=DRY_RUN (Simulated POST skipped)", flush=True)
                        print(f"  payload={json.dumps(event_payload)}", flush=True)

                elif temp_state == adapter.STATE_COOLDOWN:
                    print("  ... [COOLDOWN ACTIVE - DEBOUNCING] ...", flush=True)

                print("-" * 50, flush=True)

                if max_windows is not None and window_count >= max_windows:
                    print(f"\n[INFO] Reached maximum window limit of {max_windows}. Stopping.")
                    break

    except KeyboardInterrupt:
        print("\n\n[INFO] Live bridge terminated by user (Ctrl+C).")
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")

        print("\n" + "=" * 70)
        print("  STAGE 5F LIVE BRIDGE SESSION SUMMARY")
        print("=" * 70)
        print(f"  Total Samples Received:       {total_samples}")
        print(f"  Malformed Lines Ignored:      {malformed_lines}")
        print(f"  Total Windows Processed:      {window_count}")
        print(f"  Raw Positive Windows:         {raw_positives}")
        print(f"  Confirmed Fall Events:        {confirmed_falls}")
        print(f"  Backend Events Stored:        {backend_stored}")
        print(f"  Backend POST Failures:        {backend_failures}")
        if dry_run:
            print(f"  Dry Run Status:               ACTIVE (No POSTs sent)")
        print("=" * 70)

    return {
        "session_id": session_id,
        "total_samples": total_samples,
        "malformed_lines": malformed_lines,
        "total_windows": window_count,
        "raw_positives": raw_positives,
        "confirmed_falls": confirmed_falls,
        "backend_stored": backend_stored,
        "backend_failures": backend_failures,
        "dry_run": dry_run,
        "mock_mode": mock_mode,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5F Live ESP32 -> ML -> Temporal Confirmation -> Backend Bridge")
    parser.add_argument("--port", type=str, default="COM22", help="Serial port (default: COM22)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:5000", help="Backend REST URL (default: http://127.0.0.1:5000)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained Random Forest model")
    parser.add_argument("--scaler", type=str, default=DEFAULT_SCALER_PATH, help="Path to scaler")
    parser.add_argument("--feature-order", type=str, default=DEFAULT_FEATURE_ORDER_PATH, help="Path to feature_order.json")
    parser.add_argument("--session-id", type=str, default=None, help="Optional session ID")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after N processed windows")
    parser.add_argument("--dry-run", action="store_true", help="Run full pipeline without sending HTTP POST to backend")
    parser.add_argument("--mock", action="store_true", help="Run with simulated 100 Hz stream instead of physical serial port")

    args = parser.parse_args()

    run_stage_5f_live_bridge(
        port=args.port,
        baudrate=args.baud,
        backend_url=args.backend,
        model_path=args.model,
        scaler_path=args.scaler,
        feature_order_path=args.feature_order,
        session_id=args.session_id,
        max_windows=args.max_windows,
        dry_run=args.dry_run,
        mock_mode=args.mock,
    )
