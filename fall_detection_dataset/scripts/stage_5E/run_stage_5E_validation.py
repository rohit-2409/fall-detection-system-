"""
Stage 5E Hardware Validation Runner.

Executes the 4 required hardware validation tests:
1. TEST 1: 30-second stationary recording (Rest / Flat Baseline)
2. TEST 2: 30-second standing/movement recording (ADL Micro-movements)
3. TEST 3: 60-second normal walking recording (ADL Walking)
4. TEST 4: Controlled padded fall simulation (Impact dynamics)

For each test:
- Records live stream from ESP32 over serial (or from validated hardware stream)
- Computes timing jitter, sampling frequency (target 100 Hz), data quality metrics
- Saves raw CSV and telemetry JSON to fall_detection_dataset/live_validation/stage_5E/
- Feeds raw CSV to stage_5E_offline_replay for feature extraction & inference
- Evaluates Stage 4D Random Forest and Stage 4E Rule N3 confirmation
- Outputs comprehensive test reports in fall_detection_dataset/reports/stage_5E/
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fall_detection_dataset.scripts.stage_5E.stage_5E_serial_reader import record_serial_stream
from fall_detection_dataset.scripts.stage_5E.stage_5E_offline_replay import replay_hardware_file

STAGE_5E_DATA_DIR = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "live_validation", "stage_5E")
STAGE_5E_REPORT_DIR = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "reports", "stage_5E")
os.makedirs(STAGE_5E_DATA_DIR, exist_ok=True)
os.makedirs(STAGE_5E_REPORT_DIR, exist_ok=True)


def execute_test_session(
    test_id: str,
    test_title: str,
    session_label: str,
    duration_s: float,
    port: str = "COM22",
    baudrate: int = 115200,
    source_csv_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single Stage 5E validation session: acquisition -> quality analysis -> replay.
    """
    print("\n" + "=" * 75)
    print(f"  STAGE 5E: {test_id} — {test_title}")
    print(f"  Target Duration: {duration_s}s | Label: {session_label} | Port: {port}")
    print("=" * 75)

    raw_csv_path = None

    if source_csv_override and os.path.exists(source_csv_override):
        print(f"[REUSE] Using validated hardware recording: {source_csv_override}")
        raw_csv_path = source_csv_override
    else:
        # Record directly from serial
        try:
            quality_meta = record_serial_stream(
                port=port,
                baudrate=baudrate,
                duration_s=duration_s,
                output_dir=STAGE_5E_DATA_DIR,
                session_id=test_id
            )
            raw_csv_path = quality_meta["raw_csv_path"]
        except Exception as e:
            print(f"[SERIAL ERROR] Failed to record from port '{port}': {e}")
            raise e

    # Replay through exact feature extraction, Random Forest, and temporal confirmation
    print(f"[STAGE 5E] Running Model & Feature Compatibility Replay on: {raw_csv_path}...")
    replay_summary = replay_hardware_file(
        csv_path=raw_csv_path,
        output_dir=STAGE_5E_REPORT_DIR,
        session_label=session_label
    )

    return {
        "test_id": test_id,
        "test_title": test_title,
        "session_label": session_label,
        "duration_s": duration_s,
        "raw_csv_path": raw_csv_path,
        "replay_summary": replay_summary
    }


def run_stage_5e_full_suite(
    port: str = "COM22",
    baudrate: int = 115200,
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Runs the complete 4-test Stage 5E suite:
    TEST 1: 30s Stationary
    TEST 2: 30s Standing / Micro-movement
    TEST 3: 60s Normal Walking
    TEST 4: Controlled Padded Fall Simulation
    """
    print("=" * 75)
    print("  STAGE 5E: REAL HARDWARE DATA VALIDATION SUITE")
    print("  ESP32 + MPU6050 (100 Hz Target, 6 Channels: AccX,Y,Z, GyrX,Y,Z)")
    print("=" * 75)

    suite_results: List[Dict[str, Any]] = []

    tests = [
        ("stage_5E_test1_stationary_30s", "30-Second Stationary Recording", "STATIONARY", 30.0),
        ("stage_5E_test2_standing_30s", "30-Second Standing / Movement Recording", "STANDING", 30.0),
        ("stage_5E_test3_walking_60s", "60-Second Normal Walking Recording", "WALKING", 60.0),
        ("stage_5E_test4_padded_fall", "Controlled Padded Fall Simulation", "CONTROLLED_FALL", 15.0),
    ]

    for test_id, title, label, duration in tests:
        if interactive:
            print(f"\n[ACTION REQUIRED] Ready to record: {title} ({duration}s).")
            print("Position sensor as instructed. Press ENTER when ready to start recording...")
            input()

        res = execute_test_session(
            test_id=test_id,
            test_title=title,
            session_label=label,
            duration_s=duration,
            port=port,
            baudrate=baudrate
        )
        suite_results.append(res)

    suite_summary = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "port": port,
        "baudrate": baudrate,
        "tests_completed": len(suite_results),
        "results": suite_results
    }

    suite_json_path = os.path.join(STAGE_5E_REPORT_DIR, "stage_5E_suite_summary.json")
    with open(suite_json_path, "w", encoding="utf-8") as f_out:
        json.dump(suite_summary, f_out, indent=2)

    print("\n" + "=" * 75)
    print(f"  STAGE 5E SUITE COMPLETE: {len(suite_results)}/4 TESTS FINISHED")
    print(f"  Suite Summary JSON: {suite_json_path}")
    print("=" * 75)

    return suite_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5E Hardware Validation Runner")
    parser.add_argument("--port", type=str, default="COM22", help="Serial port (default: COM22)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--interactive", action="store_true", help="Prompt user before each test recording")
    parser.add_argument("--single-test", type=str, choices=["1", "2", "3", "4"], help="Run single test only (1, 2, 3, or 4)")
    args = parser.parse_args()

    if args.single_test:
        test_map = {
            "1": ("stage_5E_test1_stationary_30s", "30-Second Stationary Recording", "STATIONARY", 30.0),
            "2": ("stage_5E_test2_standing_30s", "30-Second Standing / Movement Recording", "STANDING", 30.0),
            "3": ("stage_5E_test3_walking_60s", "60-Second Normal Walking Recording", "WALKING", 60.0),
            "4": ("stage_5E_test4_padded_fall", "Controlled Padded Fall Simulation", "CONTROLLED_FALL", 15.0),
        }
        tid, title, lbl, dur = test_map[args.single_test]
        execute_test_session(tid, title, lbl, dur, port=args.port, baudrate=args.baudrate)
    else:
        run_stage_5e_full_suite(port=args.port, baudrate=args.baudrate, interactive=args.interactive)
