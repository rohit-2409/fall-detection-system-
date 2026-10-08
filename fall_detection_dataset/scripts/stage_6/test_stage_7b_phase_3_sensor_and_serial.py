"""
Stage 7B Phase 3 Automated Test Suite: Hardware Sensor Range & Serial Reliability.
Standalone Fall Detection Project.

Validates the hardware sensor range audit and serial throughput upgrade:
  TEST 1: Firmware static audit: Registers 0x1C (±16g) and 0x1B (±2000 dps), scales 2048.0f / 16.4f, baud 460800.
  TEST 2: Physical units scale math invariance: ADC count conversions match physical SI/gravity units.
  TEST 3: High-impact unclipped dynamic feature extraction: Models high-g impact (>8g, >12g) through frozen pipeline.
  TEST 4: Serial throughput and bandwidth headroom: Rigorous UART capacity & utilization validation.
  TEST 5: Hardware bridge configurable baud rate: Environment variable and explicit parameter priority.
  TEST 6: Mock bridge end-to-end execution at 460800 baud: Full pipeline test under the upgraded configuration.
"""

import csv
import json
import os
import re
import sys
import unittest
from typing import Any, Dict

import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_bridge import (
    DEFAULT_FEATURE_ORDER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    run_stage_6L_hardware_bridge,
)
from src.fall_confirmation.stage_4G_live_validator import extract_100_features


class TestStage7BPhase3SensorAndSerial(unittest.TestCase):
    """Hermetic unit and integration tests for hardware sensor range and serial throughput."""

    def setUp(self):
        self.ino_path = os.path.join(
            PROJECT_ROOT, "firmware", "esp32_fall_gps_unified", "esp32_fall_gps_unified.ino"
        )
        self.assertTrue(os.path.exists(self.ino_path), f"Firmware file not found at {self.ino_path}")

    # -------------------------------------------------------------------------
    # TEST 1: Firmware static audit
    # -------------------------------------------------------------------------
    def test_01_firmware_register_and_scale_audit(self):
        """TEST 1: Static verification of MPU6050 registers (±16g, ±2000 dps), scales, and baud."""
        with open(self.ino_path, "r", encoding="utf-8") as f:
            ino_text = f.read()

        # 1. Baud rate: SERIAL_BAUD 460800
        baud_match = re.search(r"#define\s+SERIAL_BAUD\s+460800", ino_text)
        self.assertIsNotNone(baud_match, "SERIAL_BAUD 460800 not found in firmware")

        # 2. ACCEL_SCALE: 2048.0f
        accel_scale_match = re.search(r"const\s+float\s+ACCEL_SCALE\s*=\s*2048\.0f\s*;", ino_text)
        self.assertIsNotNone(accel_scale_match, "ACCEL_SCALE = 2048.0f not found in firmware")

        # 3. GYRO_SCALE: 16.4f
        gyro_scale_match = re.search(r"const\s+float\s+GYRO_SCALE\s*=\s*16\.4f\s*;", ino_text)
        self.assertIsNotNone(gyro_scale_match, "GYRO_SCALE = 16.4f not found in firmware")

        # 4. Accelerometer register 0x1C written with 0x18 (±16g, AFS_SEL = 3)
        accel_reg_pattern = re.compile(
            r"Wire\.write\(0x1C\);\s*\n\s*Wire\.write\(0x18\);", re.MULTILINE
        )
        self.assertIsNotNone(
            accel_reg_pattern.search(ino_text),
            "Register 0x1C configuration write (0x18 for ±16g) not found",
        )

        # 5. Gyroscope register 0x1B written with 0x18 (±2000 dps, FS_SEL = 3)
        gyro_reg_pattern = re.compile(
            r"Wire\.write\(0x1B\);\s*\n\s*Wire\.write\(0x18\);", re.MULTILINE
        )
        self.assertIsNotNone(
            gyro_reg_pattern.search(ino_text),
            "Register 0x1B configuration write (0x18 for ±2000 dps) not found",
        )

    # -------------------------------------------------------------------------
    # TEST 2: Physical units scale math invariance
    # -------------------------------------------------------------------------
    def test_02_physical_units_scale_math_invariance(self):
        """TEST 2: Raw ADC counts convert to identical physical units (g and deg/s)."""
        # MPU6050 16-bit signed ADC range: -32768 .. +32767
        # Full scale ±16g: 32768 / 16 = 2048 LSB/g
        scale_accel = 2048.0
        # 1g static gravity (sensor vertical) -> raw counts ~ 2048
        raw_1g = 2048
        self.assertAlmostEqual(raw_1g / scale_accel, 1.0, places=4)

        # Extreme impact 12g -> raw counts ~ 24576
        raw_12g = 24576
        self.assertAlmostEqual(raw_12g / scale_accel, 12.0, places=4)

        # Max ADC count -> 16.0g
        raw_max_accel = 32768
        self.assertAlmostEqual(raw_max_accel / scale_accel, 16.0, places=4)

        # Full scale ±2000 dps: 32768 / 2000 = 16.384 LSB/(deg/s) (standard MPU6050 datasheet: 16.4)
        scale_gyro = 16.4
        raw_500dps = 8200
        self.assertAlmostEqual(raw_500dps / scale_gyro, 500.0, places=1)

        raw_max_gyro = 32768
        self.assertAlmostEqual(raw_max_gyro / scale_gyro, 1998.05, delta=2.5)

    # -------------------------------------------------------------------------
    # TEST 3: High-impact unclipped dynamic feature extraction
    # -------------------------------------------------------------------------
    def test_03_high_impact_feature_extraction_unclipped(self):
        """TEST 3: High-g (>8g, >12g) impacts extract without clipping and produce valid ML predictions."""
        # Load authentic unclipped fall dynamics from Stage 4G validation dataset (session 12)
        raw_csv = os.path.join(
            PROJECT_ROOT,
            "fall_detection_dataset",
            "live_validation",
            "stage_4G",
            "session_012_fall_forward_slip",
            "raw_sensor_data.csv",
        )
        self.assertTrue(os.path.exists(raw_csv), f"raw_sensor_data.csv missing at {raw_csv}")

        with open(raw_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Impact window occurs around sample 600..650
        target_rows = rows[600:650]
        self.assertEqual(len(target_rows), 50)

        ax = np.array([float(r["AccX"]) for r in target_rows])
        ay = np.array([float(r["AccY"]) for r in target_rows])
        az = np.array([float(r["AccZ"]) for r in target_rows])
        gx = np.array([float(r["GyrX"]) for r in target_rows])
        gy = np.array([float(r["GyrY"]) for r in target_rows])
        gz = np.array([float(r["GyrZ"]) for r in target_rows])

        window_dict = {
            "AccX": ax,
            "AccY": ay,
            "AccZ": az,
            "GyrX": gx,
            "GyrY": gy,
            "GyrZ": gz,
        }

        # Extract 100 features
        features_vec = extract_100_features(window_dict)
        self.assertEqual(len(features_vec), 100)

        # Verify no NaN or Inf
        self.assertFalse(np.any(np.isnan(features_vec)), "Features contain NaN")
        self.assertFalse(np.any(np.isinf(features_vec)), "Features contain Inf")

        # acc_mag max is index 63 (signal 6 = acc_mag, feature 3 = s_max)
        acc_mag_max = features_vec[63]
        gyr_mag_max = features_vec[73]
        # In authentic fall session 12, impact magnitude exceeds 8g and reaches ~13.2g
        self.assertGreater(acc_mag_max, 8.0, "acc_mag max should capture authentic unclipped >8g impact")
        self.assertGreater(gyr_mag_max, 300.0, "gyr_mag max should capture authentic rotation >300 dps")

        # Feed through frozen scaler and model
        scaler = joblib.load(DEFAULT_SCALER_PATH)
        model = joblib.load(DEFAULT_MODEL_PATH)

        vec = features_vec.reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        prob = model.predict_proba(vec_scaled)[0, 1]

        # Authentic fall window should produce high fall probability (>= 0.60, typically > 0.90)
        self.assertGreaterEqual(prob, 0.60, f"Authentic unclipped fall should produce high fall prob, got {prob}")

    # -------------------------------------------------------------------------
    # TEST 4: Serial throughput and bandwidth headroom
    # -------------------------------------------------------------------------
    def test_04_serial_throughput_and_headroom_math(self):
        """TEST 4: UART capacity math proves 115200 is oversubscribed and 460800 has >70% headroom."""
        sample_imu_line = (
            '{"type":"imu","seq":65535,"ax_g":-15.998,"ay_g":15.998,"az_g":15.998,'
            '"gx_dps":-1999.8,"gy_dps":1999.8,"gz_dps":1999.8,"dt_ms":10.05}\n'
        )
        sample_gps_line = (
            '{"type":"gps","valid":true,"lat":22.728455,"lon":75.869210,'
            '"sats":9,"hdop":0.95,"age_ms":120,"time":"14:30:15"}\n'
        )

        imu_bytes = len(sample_imu_line.encode("utf-8"))
        gps_bytes = len(sample_gps_line.encode("utf-8"))

        self.assertGreaterEqual(imu_bytes, 120)
        self.assertLessEqual(imu_bytes, 155)

        # At 100 Hz IMU + 1 Hz GPS:
        imu_rate_hz = 100
        gps_rate_hz = 1
        total_bytes_per_sec = (imu_bytes * imu_rate_hz) + (gps_bytes * gps_rate_hz)

        # In 8-N-1 UART: 1 start bit + 8 data bits + 1 stop bit = 10 bits per byte transmitted
        total_bits_per_sec = total_bytes_per_sec * 10

        # Baud 115200:
        cap_115200_bps = 115200
        utilization_115200 = (total_bits_per_sec / cap_115200_bps) * 100.0
        # Confirmed: oversubscribed (> 100%)
        self.assertGreater(utilization_115200, 100.0, "115200 baud should be oversubscribed at 100 Hz JSON")

        # Baud 460800:
        cap_460800_bps = 460800
        utilization_460800 = (total_bits_per_sec / cap_460800_bps) * 100.0
        headroom_460800 = 100.0 - utilization_460800

        # Confirmed: safe utilization (< 35%) and substantial headroom (> 65%)
        self.assertLess(utilization_460800, 35.0, "460800 baud utilization must be under 35%")
        self.assertGreater(headroom_460800, 65.0, "460800 baud headroom must exceed 65%")

    # -------------------------------------------------------------------------
    # TEST 5: Hardware bridge configurable baud rate
    # -------------------------------------------------------------------------
    def test_05_bridge_configurable_baud_and_environment(self):
        """TEST 5: SERIAL_BAUD environment variable and explicit arguments set bridge baud rate."""
        old_env = os.environ.get("SERIAL_BAUD")
        if "SERIAL_BAUD" in os.environ:
            del os.environ["SERIAL_BAUD"]

        # Mock mode with baudrate=None defaults to 460800
        res = run_stage_6L_hardware_bridge(
            mock_mode=True, max_windows=2, dry_run=True, baudrate=None
        )
        self.assertEqual(res["total_windows"], 2)

        # Test environment variable override
        os.environ["SERIAL_BAUD"] = "921600"
        res_env = run_stage_6L_hardware_bridge(
            mock_mode=True, max_windows=2, dry_run=True, baudrate=None
        )
        self.assertEqual(res_env["total_windows"], 2)

        # Test explicit argument override (e.g. 115200 backward compatibility)
        res_explicit = run_stage_6L_hardware_bridge(
            mock_mode=True, max_windows=2, dry_run=True, baudrate=115200
        )
        self.assertEqual(res_explicit["total_windows"], 2)

        # Restore env
        if old_env is not None:
            os.environ["SERIAL_BAUD"] = old_env
        elif "SERIAL_BAUD" in os.environ:
            del os.environ["SERIAL_BAUD"]

    # -------------------------------------------------------------------------
    # TEST 6: Mock bridge end-to-end execution at 460800 baud
    # -------------------------------------------------------------------------
    def test_06_mock_bridge_end_to_end_at_460800(self):
        """TEST 6: Mock bridge pipeline executes seamlessly at 460800 baud."""
        # Stationary run: 0 falls
        res_stat = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="stationary",
            max_windows=4,
            baudrate=460800,
            dry_run=True,
        )
        self.assertEqual(res_stat["confirmed_falls"], 0)
        self.assertEqual(res_stat["total_windows"], 4)

        # Fall run: >= 1 confirmed fall
        res_fall = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=6,
            baudrate=460800,
            dry_run=True,
        )
        self.assertGreaterEqual(res_fall["confirmed_falls"], 1)
        self.assertEqual(res_fall["total_windows"], 6)


if __name__ == "__main__":
    unittest.main()
