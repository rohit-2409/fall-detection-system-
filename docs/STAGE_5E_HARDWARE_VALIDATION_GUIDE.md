# Stage 5E — Real Hardware Data Validation User Guide

This guide details how to collect real MPU6050 sensor data from the ESP32 microcontroller over USB Serial and run offline compatibility replay through the validated Stage 4D/4E fall detection pipeline.

---

## Prerequisites

1. **Hardware Setup:**
   * ESP32 Dev Module connected via USB.
   * MPU6050 sensor connected via I2C (SDA $\rightarrow$ GPIO 21, SCL $\rightarrow$ GPIO 22, VCC $\rightarrow$ 3.3V, GND $\rightarrow$ GND).
   * Verify COM port in Windows Device Manager (default: `COM22`).
   * **Important:** Ensure the Arduino IDE Serial Monitor or any other terminal software is **CLOSED** so the serial port is free.

2. **Python Environment:**
   * Python 3.10+ with `pyserial`, `numpy`, `pandas`, `scipy`, and `joblib`.

---

## Command-Line Usage

### 1. Run Complete Automated 4-Test Hardware Validation Suite
To execute all 4 physical tests sequentially (Stationary $\rightarrow$ Standing $\rightarrow$ Walking $\rightarrow$ Controlled Fall) with automated prompts and offline replay analysis:

```powershell
python fall_detection_dataset/scripts/stage_5E/run_stage_5E_validation.py --port COM22 --duration 30
```

* `--port`: Serial COM port (e.g., `COM22`, `COM3`, `/dev/ttyUSB0`). Default is `COM22`.
* `--baudrate`: Baud rate (default: `115200`).
* `--duration`: Default recording duration in seconds for stationary/standing tests (default: `30`).

---

### 2. Run Single Hardware Recording Session
To collect a custom-duration recording session without running the full suite:

```powershell
python fall_detection_dataset/scripts/stage_5E/stage_5E_serial_reader.py --port COM22 --duration 30 --output-dir fall_detection_dataset/live_validation/stage_5E --session-id my_session_30s
```

Outputs generated:
* `fall_detection_dataset/live_validation/stage_5E/my_session_30s_raw.csv`
* `fall_detection_dataset/live_validation/stage_5E/my_session_30s_quality.json`

---

### 3. Run Offline Model Compatibility Replay
To replay any captured raw hardware CSV through the production 100-feature extractor, frozen Random Forest classifier, and Stage 4E temporal confirmation engine:

```powershell
python fall_detection_dataset/scripts/stage_5E/stage_5E_offline_replay.py --input-csv fall_detection_dataset/live_validation/stage_5E/my_session_30s_raw.csv --label STATIONARY --output-dir fall_detection_dataset/reports/stage_5E
```

Outputs generated:
* `fall_detection_dataset/reports/stage_5E/my_session_30s_predictions.csv`
* `fall_detection_dataset/reports/stage_5E/my_session_30s_compatibility_report.json`

---

### 4. Run Automated Stage 5E Test Suite
To verify the unit and regression tests for Stage 5E components:

```powershell
python fall_detection_dataset/scripts/stage_5E/test_stage_5E_hardware_validation.py
```

---

## Output Files Reference

* **Raw Sensor Data:** `fall_detection_dataset/live_validation/stage_5E/*_raw.csv`
* **Timing & Quality Metrics:** `fall_detection_dataset/live_validation/stage_5E/*_quality.json`
* **Window Predictions:** `fall_detection_dataset/reports/stage_5E/*_predictions.csv`
* **Compatibility Reports:** `fall_detection_dataset/reports/stage_5E/*_compatibility_report.json`
* **Full Suite Summary:** `fall_detection_dataset/reports/stage_5E/stage_5E_suite_summary.json`
* **Model Hashes (Pre & Post):** `fall_detection_dataset/reports/stage_5E/stage_5E_post_test_hashes.json`
