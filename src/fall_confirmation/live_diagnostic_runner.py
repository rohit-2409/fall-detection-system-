"""
Stage 4F Live Diagnostic Runner.

Connects to ESP32 stream (or mock simulator) at 100 Hz, performs rolling window
feature extraction and machine learning inference, and routes all raw predictions
through the Stage 4E TemporalConfirmationAdapter.

Outputs both raw model statistics and confirmed temporal state in real-time diagnostic format.
"""

import argparse
import collections
import datetime
import os
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple
import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "sisfall", "sisfall_random_forest.joblib"
)

WINDOW_SIZE = 50   # 50 samples = 0.50 s at 100 Hz
STEP_SIZE = 25     # 25 samples = 0.25 s (50% overlap)

BASE_SIGNALS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
DERIVED_SIGNALS = ["AccMag", "GyrMag"]
ALL_SIGNALS = BASE_SIGNALS + DERIVED_SIGNALS
STAT_NAMES = ["mean", "std", "min", "max", "ptp", "rms"]
FEATURE_NAMES = [f"{sig}_{stat}" for sig in ALL_SIGNALS for stat in STAT_NAMES]


def compute_48_features(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes 48 features matching baseline SisFall pipeline."""
    features: List[float] = []
    for sig in ALL_SIGNALS:
        arr = window_data[sig]
        features.append(float(np.mean(arr)))
        features.append(float(np.std(arr, ddof=0)))
        features.append(float(np.min(arr)))
        features.append(float(np.max(arr)))
        features.append(float(np.ptp(arr)))
        features.append(float(np.sqrt(np.mean(arr ** 2))))
    return np.array(features, dtype=np.float64)


def parse_csv_line(line: str) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Parses 7-column CSV line: timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ"""
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


def run_live_diagnostic(
    port: str = "COM22",
    baudrate: int = 115200,
    model_path: str = DEFAULT_MODEL_PATH,
    mock_mode: bool = False,
    max_windows: Optional[int] = None,
):
    print("=" * 70)
    print("  STAGE 4F: LIVE FALL DETECTION INFERENCE — DIAGNOSTIC MODE")
    print("=" * 70)

    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"[FATAL] Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading Model: {model_path}...")
    model = joblib.load(model_path)
    print("  Model loaded successfully!")

    # 2. Instantiate Temporal Confirmation Adapter
    adapter = TemporalConfirmationAdapter()
    adapter.reset()
    print(f"Temporal Engine Initialized: Rule = {adapter.rule}, Threshold = {adapter.threshold}, Event Gating = {adapter.event_gating}")

    # 3. Setup Connection
    ser = None
    if not mock_mode:
        try:
            import serial
        except ImportError:
            print("[FATAL] pyserial is required for serial communication. Install with: pip install pyserial")
            sys.exit(1)

        print(f"\nConnecting to ESP32 on port '{port}' @ {baudrate} baud...")
        try:
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
            time.sleep(1.5)
            ser.reset_input_buffer()
            adapter.reset()  # Reset on connection start
            print(f"  Connected to {port}. Listening for 100 Hz CSV stream...\n")
        except Exception as e:
            print(f"[ERROR] Could not open port {port}: {e}")
            print("To run in simulation mode, use: python src/fall_confirmation/live_diagnostic_runner.py --mock")
            sys.exit(1)
    else:
        print("\n[MOCK MODE ACTIVE] Generating simulated 100 Hz motion stream for verification...\n")

    # 4. Rolling Buffers
    raw_buffer: Deque[Tuple[float, float, float, float, float, float, float]] = collections.deque(maxlen=WINDOW_SIZE)
    new_samples = 0
    window_count = 0
    raw_positives = 0
    confirmed_falls = 0
    mock_time_ms = 0.0

    print("-" * 70)
    print("  DIAGNOSTIC STREAM ACTIVE — PRESS CTRL+C TO STOP")
    print("-" * 70)

    try:
        while True:
            if not mock_mode:
                try:
                    line_str = ser.readline().decode("utf-8", errors="ignore")
                except Exception as e:
                    print(f"\n[WARNING] Serial interrupted: {e}")
                    break
            else:
                time.sleep(0.010)  # 10ms ~ 100 Hz
                mock_time_ms += 10.0
                # Generate upright standing noise in SisFall coordinates
                ax = float(np.random.normal(0.01, 0.01))
                ay = float(np.random.normal(-0.96, 0.02))
                az = float(np.random.normal(-0.15, 0.02))
                gx = float(np.random.normal(0.02, 0.10))
                gy = float(np.random.normal(-0.01, 0.10))
                gz = float(np.random.normal(0.01, 0.10))
                line_str = f"{int(mock_time_ms)},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f}\n"

            parsed = parse_csv_line(line_str)
            if parsed is None:
                continue

            raw_buffer.append(parsed)
            new_samples += 1

            if len(raw_buffer) == WINDOW_SIZE and new_samples >= STEP_SIZE:
                new_samples = 0
                window_count += 1

                buf = list(raw_buffer)
                ts_arr = np.array([s[0] for s in buf])
                ax_arr = np.array([s[1] for s in buf])
                ay_arr = np.array([s[2] for s in buf])
                az_arr = np.array([s[3] for s in buf])
                gx_arr = np.array([s[4] for s in buf])
                gy_arr = np.array([s[5] for s in buf])
                gz_arr = np.array([s[6] for s in buf])

                acc_mag = np.sqrt(ax_arr ** 2 + ay_arr ** 2 + az_arr ** 2)
                gyr_mag = np.sqrt(gx_arr ** 2 + gy_arr ** 2 + gz_arr ** 2)

                win_dict = {
                    "AccX": ax_arr, "AccY": ay_arr, "AccZ": az_arr,
                    "GyrX": gx_arr, "GyrY": gy_arr, "GyrZ": gz_arr,
                    "AccMag": acc_mag, "GyrMag": gyr_mag
                }

                # Feature extraction and model probability
                feat = compute_48_features(win_dict)
                probs = model.predict_proba(feat.reshape(1, -1))[0]
                prob_fall = float(probs[1])

                # Pass through Stage 4E confirmation adapter
                w_start_sec = ts_arr[0] / 1000.0 if not mock_mode else mock_time_ms / 1000.0
                conf_res = adapter.process_prediction(
                    probability=prob_fall,
                    window_start_time=w_start_sec
                )

                if conf_res["raw_positive"]:
                    raw_positives += 1
                if conf_res["confirmed_fall"]:
                    confirmed_falls += 1

                # Live Diagnostic Output Format (Step 13)
                curr_clock = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"{curr_clock} | Window #{window_count:04d}", flush=True)
                print(f"  probability={conf_res['raw_probability']:.4f}", flush=True)
                print(f"  raw_positive={str(conf_res['raw_positive']).lower()}", flush=True)
                print(f"  positive_count={conf_res['positive_count']}", flush=True)
                print(f"  state={conf_res['temporal_state']}", flush=True)
                print(f"  confirmed_fall={str(conf_res['confirmed_fall']).lower()}", flush=True)

                if conf_res["confirmed_fall"]:
                    print("  >>> [CONFIRMED FALL EVENT DETECTED] <<<", flush=True)
                elif conf_res["temporal_state"] == adapter.STATE_COOLDOWN:
                    print("  ... [COOLDOWN ACTIVE - DEBOUNCING] ...", flush=True)
                print("-" * 50, flush=True)

                if max_windows is not None and window_count >= max_windows:
                    print(f"\n[INFO] Reached window limit of {max_windows}. Stopping.")
                    break

    except KeyboardInterrupt:
        print("\n\n[INFO] Diagnostic session terminated by user (Ctrl+C).")
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")

        print("\n" + "=" * 70)
        print("  STAGE 4F DIAGNOSTIC SESSION SUMMARY")
        print("=" * 70)
        print(f"  Total Windows Processed:    {window_count}")
        print(f"  Raw Positive Windows:       {raw_positives}")
        print(f"  Confirmed Fall Events:      {confirmed_falls}")
        print(f"  Filtered False Alarms:      {max(0, raw_positives - confirmed_falls)}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4F Live Diagnostic Inference")
    parser.add_argument("--port", type=str, default="COM22", help="Serial port (default: COM22)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to ML model")
    parser.add_argument("--mock", action="store_true", help="Run in mock streaming mode")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after N windows")
    args = parser.parse_args()

    run_live_diagnostic(
        port=args.port,
        baudrate=args.baud,
        model_path=args.model,
        mock_mode=args.mock,
        max_windows=args.max_windows
    )
