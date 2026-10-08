"""
SisFall Live Inference Module (ESP32 + MPU6050 Stream Reader).

Connects to ESP32 over serial (default: COM22 @ 115200 baud), ingests 100 Hz CSV
inertial data (timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ), maintains a rolling
50-sample buffer, generates a window every 25 new samples (50% overlap), computes
the exact 48 B3 statistical features, and performs real-time classification using
the frozen Random Forest model (models/sisfall/sisfall_random_forest.joblib).

Safety Notice:
Existing KFall pipeline and SisFall B1-B5 modules and models are strictly frozen.
"""

import argparse
import collections
import os
import sys
import time
from typing import Deque, Dict, List, Optional, Tuple
import joblib
import numpy as np

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Model path
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sisfall", "sisfall_random_forest.joblib")

# Windowing parameters (matching frozen SisFall B2/B3 pipeline)
WINDOW_SIZE = 50   # 50 samples = 0.5 seconds at 100 Hz
STEP_SIZE = 25     # 25 samples = 0.25 seconds (50% overlap)

# 8 Signals: 6 raw channels + 2 derived magnitude channels
BASE_SIGNALS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
DERIVED_SIGNALS = ["AccMag", "GyrMag"]
ALL_SIGNALS = BASE_SIGNALS + DERIVED_SIGNALS

# 6 Summary statistics per signal
STAT_NAMES = ["mean", "std", "min", "max", "ptp", "rms"]

# Ordered list of all 48 features (matching frozen B3 definition)
FEATURE_NAMES = [f"{sig}_{stat}" for sig in ALL_SIGNALS for stat in STAT_NAMES]


def compute_48_features(window_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Computes exactly the 48 statistical features from a 50-sample window
    in the exact canonical order expected by the frozen Random Forest model.

    Parameters
    ----------
    window_data : Dict[str, np.ndarray]
        Dictionary containing arrays of length 50 for:
        AccX, AccY, AccZ, GyrX, GyrY, GyrZ, AccMag, GyrMag.

    Returns
    -------
    np.ndarray
        1D feature vector of shape (48,).
    """
    features: List[float] = []

    for sig in ALL_SIGNALS:
        arr = window_data[sig]

        # 1. Mean: mean(x)
        features.append(float(np.mean(arr)))

        # 2. Standard deviation: std(x, ddof=0)
        features.append(float(np.std(arr, ddof=0)))

        # 3. Minimum: min(x)
        features.append(float(np.min(arr)))

        # 4. Maximum: max(x)
        features.append(float(np.max(arr)))

        # 5. Peak-to-peak: max(x) - min(x)
        features.append(float(np.ptp(arr)))

        # 6. Root Mean Square: sqrt(mean(x^2))
        features.append(float(np.sqrt(np.mean(arr ** 2))))

    return np.array(features, dtype=np.float64)


def parse_csv_line(line: str) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    Parses and validates a single CSV data line from ESP32.
    Expected format: timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ

    Returns (timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ) or None if invalid/header.
    """
    line = line.strip()
    if not line:
        return None

    # Ignore header line
    if "timestamp" in line.lower() or "accx" in line.lower():
        return None

    parts = line.split(",")
    if len(parts) != 7:
        return None

    try:
        ts = float(parts[0])
        ax = float(parts[1])
        ay = float(parts[2])
        az = float(parts[3])
        gx = float(parts[4])
        gy = float(parts[5])
        gz = float(parts[6])
        return (ts, ax, ay, az, gx, gy, gz)
    except (ValueError, TypeError):
        return None


def run_live_inference(
    port: str = "COM22",
    baudrate: int = 115200,
    model_path: str = DEFAULT_MODEL_PATH,
    mock_mode: bool = False,
    max_windows: Optional[int] = None,
):
    """
    Runs real-time inference loop reading from ESP32 over serial.
    """
    print("=" * 65)
    print("  WEARABLE FALL DETECTION — LIVE ML INFERENCE SYSTEM")
    print("=" * 65)

    # 1. Load Frozen Random Forest Model
    if not os.path.exists(model_path):
        print(f"[FATAL] Model artifact not found at: {model_path}")
        print("Please verify the model file exists before running.")
        sys.exit(1)

    print(f"Loading frozen model from: {model_path}...")
    try:
        model = joblib.load(model_path)
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"[FATAL] Failed to load model artifact: {e}")
        sys.exit(1)

    print(f"Feature schema: {len(FEATURE_NAMES)} features (AccX, AccY, AccZ, GyrX, GyrY, GyrZ, AccMag, GyrMag)")
    print(f"Window configuration: {WINDOW_SIZE} samples (0.50 s @ 100 Hz), step = {STEP_SIZE} samples (0.25 s)")

    # 2. Open Serial Port or Setup Mock
    ser = None
    if not mock_mode:
        try:
            import serial
            import serial.tools.list_ports
        except ImportError:
            print("[FATAL] 'pyserial' library is required to connect to COM ports.")
            print("Install it using: pip install pyserial")
            sys.exit(1)

        print(f"\nConnecting to ESP32 on port '{port}' at {baudrate} baud...")
        try:
            ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
            time.sleep(1.5)  # Allow ESP32 serial reset to stabilize
            ser.reset_input_buffer()
            print(f"  Successfully connected to {port}!")
            print("  Listening for incoming 100 Hz CSV stream...\n")
        except serial.SerialException as e:
            print(f"\n[ERROR] Unable to open serial port '{port}': {e}")
            print("\nAvailable serial ports detected on your machine:")
            ports = list(serial.tools.list_ports.comports())
            if ports:
                for p in ports:
                    print(f"  - {p.device}: {p.description}")
            else:
                print("  (No active COM ports detected)")
            print("\nTroubleshooting tips:")
            print("  1. Verify the ESP32 is plugged into USB.")
            print("  2. Ensure Arduino IDE Serial Monitor / Plotter is CLOSED (holds COM port exclusively).")
            print("  3. If ESP32 is on another port (e.g. COM3), run with: python src/sisfall_live_inference.py --port COM3")
            print("  4. To test with simulated live data right now, run: python src/sisfall_live_inference.py --mock")
            sys.exit(1)
    else:
        print("\n[MOCK MODE ACTIVATED] Simulating 100 Hz streaming sensor data for testing...\n")

    # 3. Rolling Sample Buffers
    raw_buffer: Deque[Tuple[float, float, float, float, float, float, float]] = collections.deque(maxlen=WINDOW_SIZE)
    new_samples_count = 0
    window_counter = 0

    total_normal_windows = 0
    total_fall_windows = 0
    malformed_lines_count = 0

    mock_sim_timestamp = 0.0

    print("-----------------------------------------------------------------")
    print("  LIVE INFERENCE ACTIVE — PRESS CTRL+C TO STOP AT ANY TIME")
    print("-----------------------------------------------------------------")

    try:
        while True:
            # Read sample from Serial or Mock
            if not mock_mode:
                try:
                    line_bytes = ser.readline()
                    line_str = line_bytes.decode("utf-8", errors="ignore")
                except serial.SerialException as e:
                    print(f"\n[WARNING] Serial communication interrupted: {e}")
                    print("Connection lost. Exiting cleanly.")
                    break
            else:
                # Mock generation at ~100 Hz (10 ms sleep)
                time.sleep(0.010)
                mock_sim_timestamp += 10.0
                # Generate realistic standing/walking baseline in SisFall coordinate frame
                mock_ax = float(np.random.normal(0.01, 0.01))
                mock_ay = float(np.random.normal(-0.96, 0.02))
                mock_az = float(np.random.normal(-0.15, 0.02))
                mock_gx = float(np.random.normal(0.02, 0.10))
                mock_gy = float(np.random.normal(-0.01, 0.10))
                mock_gz = float(np.random.normal(0.01, 0.10))
                line_str = f"{int(mock_sim_timestamp)},{mock_ax:.6f},{mock_ay:.6f},{mock_az:.6f},{mock_gx:.6f},{mock_gy:.6f},{mock_gz:.6f}\n"

            parsed = parse_csv_line(line_str)
            if parsed is None:
                if line_str.strip() and not ("timestamp" in line_str.lower() or "accx" in line_str.lower()):
                    malformed_lines_count += 1
                continue

            raw_buffer.append(parsed)
            new_samples_count += 1

            # Window trigger: rolling buffer has 50 samples and at least 25 new samples arrived
            if len(raw_buffer) == WINDOW_SIZE and new_samples_count >= STEP_SIZE:
                new_samples_count = 0
                window_counter += 1

                # Extract channel arrays
                buffer_list = list(raw_buffer)
                timestamps = np.array([s[0] for s in buffer_list])
                acc_x = np.array([s[1] for s in buffer_list])
                acc_y = np.array([s[2] for s in buffer_list])
                acc_z = np.array([s[3] for s in buffer_list])
                gyr_x = np.array([s[4] for s in buffer_list])
                gyr_y = np.array([s[5] for s in buffer_list])
                gyr_z = np.array([s[6] for s in buffer_list])

                # Compute the 2 derived magnitude signals (matching B3 exactly)
                acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                gyr_mag = np.sqrt(gyr_x ** 2 + gyr_y ** 2 + gyr_z ** 2)

                window_dict = {
                    "AccX": acc_x,
                    "AccY": acc_y,
                    "AccZ": acc_z,
                    "GyrX": gyr_x,
                    "GyrY": gyr_y,
                    "GyrZ": gyr_z,
                    "AccMag": acc_mag,
                    "GyrMag": gyr_mag,
                }

                # Compute 48 features in exact canonical order
                feature_vector = compute_48_features(window_dict)
                X_input = feature_vector.reshape(1, -1)

                # Predict using frozen Random Forest model
                pred_label = int(model.predict(X_input)[0])
                probs = model.predict_proba(X_input)[0]

                prob_normal = probs[0] * 100.0
                prob_fall = probs[1] * 100.0

                ts_start = int(timestamps[0])
                ts_end = int(timestamps[-1])
                ts_duration = ts_end - ts_start

                if pred_label == 1:
                    total_fall_windows += 1
                    status_text = "FALL"
                    prob_display = f"{prob_fall:.1f}%"
                    alert_prefix = "[FALL ALERT] "
                else:
                    total_normal_windows += 1
                    status_text = "NORMAL"
                    prob_display = f"{prob_normal:.1f}%"
                    alert_prefix = ""

                # Print beginner-readable output
                print(f"{alert_prefix}Window: {window_counter}", flush=True)
                print(f"Samples: {WINDOW_SIZE}", flush=True)
                print(f"Timestamp Range: {ts_start} ms -> {ts_end} ms (Duration: {ts_duration} ms)", flush=True)
                print(f"Prediction: {status_text}", flush=True)
                print(f"Probability: {prob_display}", flush=True)
                print("-" * 50, flush=True)

                if max_windows is not None and window_counter >= max_windows:
                    print(f"[INFO] Reached requested limit of {max_windows} window(s). Stopping inference.")
                    break

    except KeyboardInterrupt:
        print("\n\n[INFO] Live inference stopped by user (Ctrl+C).")
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")

        print("\n" + "=" * 65)
        print("  LIVE INFERENCE SESSION SUMMARY")
        print("=" * 65)
        print(f"  Total Windows Evaluated: {window_counter}")
        print(f"  Normal (ADL) Windows:    {total_normal_windows}")
        print(f"  Fall Windows Detected:   {total_fall_windows}")
        if malformed_lines_count > 0:
            print(f"  Malformed Lines Ignored: {malformed_lines_count}")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Fall Detection Inference from ESP32 Stream")
    parser.add_argument("--port", type=str, default="COM22", help="Serial port (default: COM22)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to frozen Random Forest model")
    parser.add_argument("--mock", action="store_true", help="Run in mock simulation mode without hardware")
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after N windows (useful for testing)")
    args = parser.parse_args()

    run_live_inference(
        port=args.port,
        baudrate=args.baud,
        model_path=args.model,
        mock_mode=args.mock,
        max_windows=args.max_windows,
    )
