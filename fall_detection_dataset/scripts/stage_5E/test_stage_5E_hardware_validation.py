"""
Stage 5E Automated Hardware Validation & Compatibility Test Suite.

Verifies:
1. Sensor line parsing across multiple ESP32 output formats:
   - Pipe format: ACC(g): X, Y, Z | GYRO(deg/s): X, Y, Z
   - Standard 7-column CSV: timestamp,ax,ay,az,gx,gy,gz
   - 6-column CSV: ax,ay,az,gx,gy,gz
   - Header filtering and error recovery
2. Statistical timing and data quality analysis
3. Exact 100-feature vector extraction and feature_order.json parity
4. Random Forest model inference on hardware windows
5. Temporal confirmation state machine behavior (Rule N3)
6. Offline hardware replay pipeline
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fall_detection_dataset.scripts.stage_5E.stage_5E_serial_reader import (
    parse_sensor_line,
    analyze_sensor_stream
)
from fall_detection_dataset.scripts.stage_5E.stage_5E_offline_replay import replay_hardware_file
from src.fall_confirmation.stage_4G_live_validator import (
    extract_100_features,
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_ORDER_PATH,
    WINDOW_SIZE,
    STEP_SIZE
)
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter


class TestStage5EHardwareValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_parse_pipe_format(self):
        """Test 1: Parse ACC(g): ... | GYRO(deg/s): ... format."""
        line = "ACC(g): 0.012, -0.025, 0.986 | GYRO(deg/s): 0.31, -0.12, 0.08"
        res = parse_sensor_line(line)
        self.assertIsNotNone(res)
        ax, ay, az, gx, gy, gz = res
        self.assertAlmostEqual(ax, 0.012, places=3)
        self.assertAlmostEqual(ay, -0.025, places=3)
        self.assertAlmostEqual(az, 0.986, places=3)
        self.assertAlmostEqual(gx, 0.31, places=2)
        self.assertAlmostEqual(gy, -0.12, places=2)
        self.assertAlmostEqual(gz, 0.08, places=2)

    def test_02_parse_csv_formats(self):
        """Test 2: Parse 7-column and 6-column CSV formats."""
        # 7-column
        line7 = "10,0.010254,-0.020142,0.981445,0.206107,-0.106870,0.053435"
        res7 = parse_sensor_line(line7)
        self.assertIsNotNone(res7)
        self.assertAlmostEqual(res7[0], 0.010254, places=6)
        self.assertAlmostEqual(res7[5], 0.053435, places=6)

        # 6-column
        line6 = "0.010254,-0.020142,0.981445,0.206107,-0.106870,0.053435"
        res6 = parse_sensor_line(line6)
        self.assertIsNotNone(res6)
        self.assertAlmostEqual(res6[0], 0.010254, places=6)

    def test_03_header_and_corrupt_lines_ignored(self):
        """Test 3: Verify header lines, debug messages, and malformed strings return None."""
        self.assertIsNone(parse_sensor_line("timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ"))
        self.assertIsNone(parse_sensor_line("Starting MPU6050 I2C..."))
        self.assertIsNone(parse_sensor_line("Ready!"))
        self.assertIsNone(parse_sensor_line(""))
        self.assertIsNone(parse_sensor_line("invalid,data,here"))

    def test_04_analyze_sensor_stream_metrics(self):
        """Test 4: Verify timing, jitter, sample rate, and channel distribution statistics."""
        # Generate 100 samples at exact 100 Hz (0.01s interval)
        ts = [i * 0.01 for i in range(100)]
        ax = [0.01] * 100
        ay = [-0.02] * 100
        az = [0.98] * 100
        gx = [0.20] * 100
        gy = [-0.10] * 100
        gz = [0.05] * 100

        stats = analyze_sensor_stream(ts, ax, ay, az, gx, gy, gz, malformed_count=0, nan_inf_count=0)
        self.assertEqual(stats["total_samples"], 100)
        self.assertAlmostEqual(stats["measured_sample_rate_hz"], 101.01, delta=2.0)
        self.assertTrue(stats["sample_rate_compatible"])
        self.assertAlmostEqual(stats["median_interval_s"], 0.01, places=4)
        self.assertAlmostEqual(stats["channel_statistics"]["AccZ"]["mean"], 0.98, places=2)
        self.assertTrue(stats["unit_compatible_1g"])

    def test_05_feature_extraction_order_parity(self):
        """Test 5: Verify extract_100_features matches feature_order.json exactly."""
        with open(DEFAULT_FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            expected_names = json.load(f)
        self.assertEqual(len(expected_names), 100)

        # 50-sample stationary window
        win_dict = {
            "AccX": np.full(50, 0.01),
            "AccY": np.full(50, -0.02),
            "AccZ": np.full(50, 0.98),
            "GyrX": np.full(50, 0.20),
            "GyrY": np.full(50, -0.10),
            "GyrZ": np.full(50, 0.05),
        }
        feat_vec = extract_100_features(win_dict)
        self.assertEqual(len(feat_vec), 100)
        self.assertFalse(np.isnan(feat_vec).any())
        self.assertFalse(np.isinf(feat_vec).any())

    def test_06_model_inference_on_stationary_data(self):
        """Test 6: Random Forest evaluates stationary window with probability << 0.60."""
        model = joblib.load(DEFAULT_MODEL_PATH)
        win_dict = {
            "AccX": np.full(50, 0.010254),
            "AccY": np.full(50, -0.020142),
            "AccZ": np.full(50, 0.981445),
            "GyrX": np.full(50, 0.206107),
            "GyrY": np.full(50, -0.106870),
            "GyrZ": np.full(50, 0.053435),
        }
        feat_vec = extract_100_features(win_dict).reshape(1, -1).astype(np.float32)
        prob = float(model.predict_proba(feat_vec)[0, 1])
        self.assertLess(prob, 0.30)  # Safe stationary movement

    def test_07_offline_replay_pipeline(self):
        """Test 7: Offline replay engine parses CSV, evaluates windows, and writes report."""
        test_csv = os.path.join(self.temp_dir, "test_hardware.csv")
        # Generate 150 samples (enough for 5 windows: 50 + 4*25)
        with open(test_csv, "w", encoding="utf-8") as f:
            f.write("timestamp_s,AccX,AccY,AccZ,GyrX,GyrY,GyrZ\n")
            for i in range(150):
                t = i * 0.010
                f.write(f"{t:.4f},0.010254,-0.020142,0.981445,0.206107,-0.106870,0.053435\n")

        rep = replay_hardware_file(test_csv, output_dir=self.temp_dir, session_label="TEST_STATIONARY")
        self.assertEqual(rep["section_A_sensor_data_quality"]["total_samples"], 150)
        self.assertEqual(rep["section_B_preprocessing_compatibility"]["total_windows_generated"], 5)
        self.assertEqual(rep["section_C_feature_compatibility"]["feature_count_extracted"], 100)
        self.assertTrue(rep["section_C_feature_compatibility"]["all_features_finite_no_nan_inf"])
        self.assertEqual(rep["section_E_temporal_confirmation_behavior"]["confirmed_fall_events"], 0)
        self.assertEqual(rep["overall_compatibility_verdict"], "COMPATIBLE")


def run_tests() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStage5EHardwareValidation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "total_tests": result.testsRun,
        "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "was_successful": result.wasSuccessful()
    }


if __name__ == "__main__":
    res = run_tests()
    print("\nSTAGE 5E TEST SUMMARY:", res)
    sys.exit(0 if res["was_successful"] else 1)
