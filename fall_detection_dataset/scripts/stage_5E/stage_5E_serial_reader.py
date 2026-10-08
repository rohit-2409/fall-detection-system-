"""
Stage 5E Hardware Serial Ingestion & Data Quality Analyzer.

Connects to ESP32 over serial (default: COM22 @ 115200 baud),
ingests real MPU6050 accelerometer & gyroscope streams,
adds local monotonic timestamps, analyzes timing jitter and signal quality,
and saves raw CSV and structured telemetry JSON.

Supports both:
1. Label format: ACC(g): X, Y, Z | GYRO(deg/s): X, Y, Z
2. Standard CSV format: timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ
3. Raw 6-channel CSV: AccX,AccY,AccZ,GyrX,GyrY,GyrZ
"""

import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Regular expressions for robust line parsing
# Format 1: ACC(g): 0.012, -0.025, 0.986 | GYRO(deg/s): 0.31, -0.12, 0.08
RE_PIPE_FORMAT = re.compile(
    r"ACC(?:\(g\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*\|\s*GYRO?(?:\(deg/s\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE
)

# Format 2: Separate labels like AccX: ... AccY: ...
RE_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_sensor_line(line: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Parses a single line of raw sensor text into:
    (AccX, AccY, AccZ, GyrX, GyrY, GyrZ) in units of (g, g, g, deg/s, deg/s, deg/s).
    Returns None if line is a header, debug message, or invalid.
    """
    line = line.strip()
    if not line:
        return None

    # Check for header strings
    lower = line.lower()
    if "starting" in lower or "i2c" in lower or "ready" in lower or "setup" in lower or "baud" in lower:
        return None

    # Try Format 1: Pipe-separated ACC | GYRO
    m_pipe = RE_PIPE_FORMAT.search(line)
    if m_pipe:
        try:
            return (
                float(m_pipe.group(1)),
                float(m_pipe.group(2)),
                float(m_pipe.group(3)),
                float(m_pipe.group(4)),
                float(m_pipe.group(5)),
                float(m_pipe.group(6))
            )
        except (ValueError, IndexError):
            pass

    # Try Format 2: Comma-separated tokens (7 or 6 columns)
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
        # Header check
        if any(h in lower for h in ["timestamp", "accx", "gyrx"]):
            return None
        if len(parts) == 7:
            # timestamp, ax, ay, az, gx, gy, gz
            try:
                return (
                    float(parts[1]), float(parts[2]), float(parts[3]),
                    float(parts[4]), float(parts[5]), float(parts[6])
                )
            except (ValueError, IndexError):
                pass
        elif len(parts) == 6:
            # ax, ay, az, gx, gy, gz
            try:
                return (
                    float(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]), float(parts[5])
                )
            except (ValueError, IndexError):
                pass

    # Fallback: Extract all 6 consecutive floats
    floats = RE_FLOAT.findall(line)
    if len(floats) == 6:
        try:
            return (
                float(floats[0]), float(floats[1]), float(floats[2]),
                float(floats[3]), float(floats[4]), float(floats[5])
            )
        except ValueError:
            pass
    elif len(floats) == 7 and not line.startswith("ACC"):
        # Likely index 0 is timestamp
        try:
            return (
                float(floats[1]), float(floats[2]), float(floats[3]),
                float(floats[4]), float(floats[5]), float(floats[6])
            )
        except ValueError:
            pass

    return None


def analyze_sensor_stream(
    timestamps: List[float],
    accx: List[float],
    accy: List[float],
    accz: List[float],
    gyrx: List[float],
    gyry: List[float],
    gyrz: List[float],
    malformed_count: int,
    nan_inf_count: int,
    nominal_hz: float = 100.0
) -> Dict[str, Any]:
    """
    Computes rigorous timing, jitter, sample rate, and channel distribution statistics
    without silently interpolating or modifying the raw data.
    """
    total_samples = len(timestamps)
    if total_samples < 2:
        return {
            "error": "Insufficient samples collected (less than 2 samples).",
            "total_samples": total_samples,
            "malformed_samples": malformed_count
        }

    elapsed_time = timestamps[-1] - timestamps[0]
    intervals = np.diff(timestamps)

    # Timing analysis
    mean_interval = float(np.mean(intervals))
    median_interval = float(np.median(intervals))
    std_interval = float(np.std(intervals, ddof=0))
    min_interval = float(np.min(intervals))
    max_interval = float(np.max(intervals))
    
    # Timing jitter: RMS deviation from nominal 10ms (0.010s) interval
    nominal_dt = 1.0 / nominal_hz
    timing_jitter_rms = float(np.sqrt(np.mean((intervals - nominal_dt) ** 2)))
    
    # Measured frequency
    measured_sample_rate = float(total_samples / max(0.0001, elapsed_time))

    # Duplicate timestamps
    duplicate_timestamps = int(np.sum(intervals == 0))

    # Channels
    ax_arr = np.array(accx)
    ay_arr = np.array(accy)
    az_arr = np.array(accz)
    gx_arr = np.array(gyrx)
    gy_arr = np.array(gyry)
    gz_arr = np.array(gyrz)

    acc_mag = np.sqrt(ax_arr ** 2 + ay_arr ** 2 + az_arr ** 2)
    gyr_mag = np.sqrt(gx_arr ** 2 + gy_arr ** 2 + gz_arr ** 2)

    channel_stats = {
        "AccX": {"min": float(np.min(ax_arr)), "max": float(np.max(ax_arr)), "mean": float(np.mean(ax_arr)), "std": float(np.std(ax_arr))},
        "AccY": {"min": float(np.min(ay_arr)), "max": float(np.max(ay_arr)), "mean": float(np.mean(ay_arr)), "std": float(np.std(ay_arr))},
        "AccZ": {"min": float(np.min(az_arr)), "max": float(np.max(az_arr)), "mean": float(np.mean(az_arr)), "std": float(np.std(az_arr))},
        "GyrX": {"min": float(np.min(gx_arr)), "max": float(np.max(gx_arr)), "mean": float(np.mean(gx_arr)), "std": float(np.std(gx_arr))},
        "GyrY": {"min": float(np.min(gy_arr)), "max": float(np.max(gy_arr)), "mean": float(np.mean(gy_arr)), "std": float(np.std(gy_arr))},
        "GyrZ": {"min": float(np.min(gz_arr)), "max": float(np.max(gz_arr)), "mean": float(np.mean(gz_arr)), "std": float(np.std(gz_arr))},
        "AccMag": {"min": float(np.min(acc_mag)), "max": float(np.max(acc_mag)), "mean": float(np.mean(acc_mag)), "std": float(np.std(acc_mag))},
        "GyrMag": {"min": float(np.min(gyr_mag)), "max": float(np.max(gyr_mag)), "mean": float(np.mean(gyr_mag)), "std": float(np.std(gyr_mag))}
    }

    # 100 Hz Tolerance check: 90 Hz <= measured <= 110 Hz
    sample_rate_compatible = (90.0 <= measured_sample_rate <= 110.0)

    # Unit Sanity: For stationary or ADL movement, mean AccMag should be ~1.0g (0.8g - 1.2g)
    unit_compatible = (0.75 <= channel_stats["AccMag"]["mean"] <= 1.30)

    return {
        "total_samples": total_samples,
        "elapsed_time_s": round(elapsed_time, 4),
        "nominal_sample_rate_hz": nominal_hz,
        "measured_sample_rate_hz": round(measured_sample_rate, 2),
        "sample_rate_compatible": sample_rate_compatible,
        "median_interval_s": round(median_interval, 6),
        "mean_interval_s": round(mean_interval, 6),
        "std_interval_s": round(std_interval, 6),
        "min_interval_s": round(min_interval, 6),
        "max_interval_s": round(max_interval, 6),
        "timing_jitter_rms_s": round(timing_jitter_rms, 6),
        "duplicate_timestamps": duplicate_timestamps,
        "malformed_samples": malformed_count,
        "missing_invalid_values": nan_inf_count,
        "unit_compatible_1g": unit_compatible,
        "channel_statistics": channel_stats
    }


def record_serial_stream(
    port: str = "COM22",
    baudrate: int = 115200,
    duration_s: float = 30.0,
    output_dir: str = "fall_detection_dataset/live_validation/stage_5E",
    session_id: str = "stage_5E_recording"
) -> Dict[str, Any]:
    """
    Connects to the physical ESP32 serial port, collects live MPU6050 samples,
    writes raw CSV, and generates statistical data quality analysis.
    """
    try:
        import serial
    except ImportError:
        raise ImportError("pyserial is required. Install via: pip install pyserial")

    os.makedirs(output_dir, exist_ok=True)
    raw_csv_path = os.path.join(output_dir, f"{session_id}_raw.csv")
    meta_json_path = os.path.join(output_dir, f"{session_id}_quality.json")

    print(f"\n[STAGE 5E] Connecting to ESP32 on port '{port}' @ {baudrate} baud...")
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
    time.sleep(1.0)  # Allow USB UART to stabilize
    ser.reset_input_buffer()

    timestamps: List[float] = []
    accx: List[float] = []
    accy: List[float] = []
    accz: List[float] = []
    gyrx: List[float] = []
    gyry: List[float] = []
    gyrz: List[float] = []

    malformed_count = 0
    nan_inf_count = 0

    print(f"[STAGE 5E] Recording {duration_s:.1f} seconds of live MPU6050 telemetry...")
    t_start = time.perf_counter()
    last_print = t_start

    with open(raw_csv_path, "w", encoding="utf-8") as f_out:
        f_out.write("timestamp_s,AccX,AccY,AccZ,GyrX,GyrY,GyrZ\n")

        while True:
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            if elapsed >= duration_s:
                break

            # Status heartbeat every 5 seconds
            if t_now - last_print >= 5.0:
                last_print = t_now
                print(f"  -> Elapsed: {elapsed:5.1f}s / {duration_s:5.1f}s | Samples: {len(timestamps):,}")

            raw_line = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw_line:
                continue

            parsed = parse_sensor_line(raw_line)
            if parsed is None:
                malformed_count += 1
                continue

            ax, ay, az, gx, gy, gz = parsed
            if any(math.isnan(v) or math.isinf(v) for v in [ax, ay, az, gx, gy, gz]):
                nan_inf_count += 1
                continue

            rel_ts = t_now - t_start
            timestamps.append(rel_ts)
            accx.append(ax)
            accy.append(ay)
            accz.append(az)
            gyrx.append(gx)
            gyry.append(gy)
            gyrz.append(gz)

            f_out.write(f"{rel_ts:.6f},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f}\n")

    ser.close()
    print(f"[STAGE 5E] Recording complete: {len(timestamps):,} samples captured in {elapsed:.2f}s.")

    # Quality Analysis
    quality_report = analyze_sensor_stream(
        timestamps, accx, accy, accz, gyrx, gyry, gyrz,
        malformed_count, nan_inf_count
    )
    quality_report["session_id"] = session_id
    quality_report["port"] = port
    quality_report["baudrate"] = baudrate
    quality_report["recorded_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    quality_report["raw_csv_path"] = os.path.abspath(raw_csv_path)

    with open(meta_json_path, "w", encoding="utf-8") as f_json:
        json.dump(quality_report, f_json, indent=2)

    print(f"[STAGE 5E] Raw CSV saved to: {raw_csv_path}")
    print(f"[STAGE 5E] Quality JSON saved to: {meta_json_path}")
    print(f"  - Measured Rate: {quality_report.get('measured_sample_rate_hz', 0)} Hz (Nominal: 100 Hz)")
    print(f"  - Malformed: {malformed_count}, Jitter RMS: {quality_report.get('timing_jitter_rms_s', 0)}s")

    return quality_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5E Hardware Serial Ingestion & Quality Analyzer")
    parser.add_argument("--port", type=str, default="COM22", help="Serial port (default: COM22)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds (default: 30.0)")
    parser.add_argument("--session-id", type=str, default="stage_5E_recording", help="Unique session ID")
    parser.add_argument("--output-dir", type=str, default="fall_detection_dataset/live_validation/stage_5E", help="Output directory")
    args = parser.parse_args()

    record_serial_stream(
        port=args.port,
        baudrate=args.baudrate,
        duration_s=args.duration,
        output_dir=args.output_dir,
        session_id=args.session_id
    )
