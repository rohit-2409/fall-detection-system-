# Stage 6L Phase 2 — Hardware Integration Implementation Report

**Project:** Standalone Fall Detection System  
**Mode:** Phase 2: Hardware Integration Implementation (Additive Only)  
**Active Development Workspace:** `C:\Users\ROHIT\Desktop\major project - Copy`  
**Protected Original Workspace:** `C:\Users\ROHIT\Desktop\major project`  
**Date:** 2026-10-07  

---

## 1. Implementation Summary

Stage 6L Phase 2 completes the implementation of the missing end-to-end hardware integration identified in the Phase 1 audit. The implementation resolves all five identified blockers in a strictly **additive** manner without modifying or refactoring any frozen Stage 4, Stage 5, or Stage 6 components.

### Architectural Flow Implemented

```text
       MPU6050 (I2C) + NEO-6M GPS (UART2)
                        │
                        ▼
       ESP32 Unified Firmware (100 Hz Non-Blocking)
           ├── Serial (USB COM) ────► JSON Lines Stream
           └── Wi-Fi / MQTT     ────► GPS Telemetry (fall_detection/stage6/gps)
                                           │
                                           ▼
                                 Additive GPS Cache (In-Memory)
                                           │ (query latest fix)
                                           ▼
COM Port ──► Serial Parser ──► 100-Feature Extractor ──► Random Forest Model
                                                                │
                                                                ▼
                                                   Temporal Confirmation (N3)
                                                                │
                                                    [Confirmed Fall Detected]
                                                                │
                                              ┌─────────────────┴─────────────────┐
                                              │ Attach GPS Location (if fresh)     │
                                              └─────────────────┬─────────────────┘
                                                                ▼
                                                Stage 6F MQTT Fall Publisher
                                                  (fall_detection/stage6/fall)
                                                                │
                                                                ▼
                                                   Mosquitto MQTT Broker (1883)
                                                                │
                                                                ▼
                                                   Stage 6G MQTT Gateway
                                                                │
                                                                ▼
                                                    Flask Event Backend REST API
                                                                │
                                                                ▼
                                                   SQLite Database (fall_events)
                                                                │
                                       ┌────────────────────────┴────────────────────────┐
                                       ▼                                                 ▼
                             Stage 6J Reliable Dispatch Engine               Stage 6K Monitoring Dashboard
                             (WhatsApp API + Exponential Backoff)            (Real-Time Web UI)
```

---

## 2. Files Created

| File | Purpose |
|---|---|
| `firmware/esp32_fall_gps_unified/esp32_fall_gps_unified.ino` | Production ESP32 firmware combining MPU6050 sampling (100 Hz), NEO-6M GPS (UART2), USB JSON Lines serial stream, and background Wi-Fi/MQTT GPS publishing. |
| `src/fall_confirmation/stage_6L_gps_cache.py` | Additive in-memory GPS telemetry cache subscribing to `fall_detection/stage6/gps`, verifying coordinate bounds and freshness, providing clean location attachments without touching SQLite. |
| `scripts/stage_6/run_stage_6L_hardware_bridge.py` | Core hardware bridge reading ESP32 serial stream (JSON Lines / CSV / Pipe), computing 100 features, evaluating Random Forest, checking temporal confirmation, attaching live GPS, and publishing confirmed falls via Stage 6F MQTT publisher. |
| `scripts/stage_6/run_stage_6L_hardware_test.py` | End-to-end preflight orchestrator checking Mosquitto broker health, Flask backend health, COM port availability, and coordinating bridge startup. |
| `fall_detection_dataset/scripts/stage_6/run_stage_6L_hardware_bridge.py` | Root compatibility wrapper exposing CLI interface for `run_stage_6L_hardware_bridge.py`. |
| `fall_detection_dataset/scripts/stage_6/run_stage_6L_hardware_test.py` | Root compatibility wrapper exposing CLI interface for `run_stage_6L_hardware_test.py`. |
| `fall_detection_dataset/scripts/stage_6/test_stage_6L_hardware_integration.py` | Automated test suite containing 20 comprehensive unit and integration tests for Stage 6L. |
| `.env.example` | Non-sensitive configuration template for broker, ports, and optional Wi-Fi credentials. |

---

## 3. Files Modified

| File | Modification Details |
|---|---|
| `scratch/run_all_regressions.py` | Added `Stage 6L` suite (`test_stage_6L_hardware_integration.py`) to the global regression runner to maintain tracking of all 13 suites. |

*Note: No existing codebase files were modified. All changes are strictly additive.*

---

## 4. Frozen Files Confirmed Unchanged

The following components were audited and confirmed **100% UNCHANGED**:

- **Stage 4 Models & Artifacts:**
  - `fall_detection_dataset/models/stage_4D_random_forest.joblib` (UNTOUCHED)
  - `fall_detection_dataset/models/stage_4D_scaler.joblib` (UNTOUCHED)
  - `fall_detection_dataset/models/stage_4D_feature_order.json` (UNTOUCHED)
- **Stage 4 Inference & Gating:**
  - `src/fall_confirmation/temporal_adapter.py` (UNTOUCHED)
  - `src/fall_confirmation/temporal_rule_engine.py` (UNTOUCHED)
- **Stage 4G Live Validator:**
  - `src/fall_confirmation/stage_4G_live_validator.py` (UNTOUCHED)
- **Stage 5A–5E Fall Event Backend & Dashboard:**
  - `src/fall_event_backend/` (UNTOUCHED)
  - `templates/dashboard.html` (UNTOUCHED)
- **Stage 5F Live Bridge:**
  - `src/fall_confirmation/stage_5F_live_bridge.py` (UNTOUCHED — retained for REST fallback and regression)
- **Stage 6F MQTT Fall Publisher:**
  - `src/mqtt/mqtt_publisher.py` (UNTOUCHED — imported and called as-is)
- **Stage 6G MQTT Gateway:**
  - `src/mqtt/mqtt_subscriber.py` (UNTOUCHED)
- **Stage 6H Location Display:**
  - `templates/dashboard.html` (UNTOUCHED)
- **Stage 6I WhatsApp Dispatcher:**
  - `src/notifications/whatsapp_notifier.py` (UNTOUCHED)
- **Stage 6J Reliable Delivery Worker:**
  - `src/notifications/delivery_worker.py` (UNTOUCHED)
- **Stage 6K Monitoring Dashboard:**
  - `src/notifications/monitoring_routes.py` (UNTOUCHED)
  - `templates/monitoring.html` (UNTOUCHED)

---

## 5. Unified Firmware Design

- **Location:** `firmware/esp32_fall_gps_unified/esp32_fall_gps_unified.ino`
- **Sensors Managed:**
  1. **MPU6050 6-DOF IMU:**
     - Connected via I2C (SDA = GPIO 21, SCL = GPIO 22) @ 400 kHz Fast-Mode.
     - Sampled at 100 Hz (10 ms interval) via non-blocking `millis()` loop.
     - Configured for $\pm 8g$ accelerometer range (sensitivity: 4096 LSB/g) and $\pm 2000^\circ/\text{s}$ gyroscope range (sensitivity: 16.4 LSB/($^\circ$/s)).
     - Full gravity vector preserved; axes mapped to standard coordinate frame ($A_X, A_Y, A_Z$ in $g$, $G_X, G_Y, G_Z$ in $^\circ/\text{s}$).
  2. **NEO-6M GPS Module:**
     - Connected via UART2 (RX = GPIO 16, TX = GPIO 17) @ 9600 baud.
     - Configured pins clearly exposed at the sketch header.
     - Continuous non-blocking character feeding via `TinyGPSPlus`.
     - Validates latitude, longitude, satellite count, and fix age.
- **Output Mechanisms:**
  1. **USB Serial Stream:** Emits deterministic JSON Lines stream at 115200 baud for IMU samples and periodic GPS status reports.
  2. **Wi-Fi + MQTT Background Telemetry:** Connects to local Wi-Fi and publishes GPS JSON payloads to `fall_detection/stage6/gps` every 2000 ms. Reconnection is non-blocking with exponential backoff.
- **Safety & Secret Hygiene:**
  - Wi-Fi SSID, password, broker IP, and port are parameterized via configuration placeholders (`YOUR_WIFI_SSID`, `YOUR_WIFI_PASSWORD`, `YOUR_BROKER_IP`).
  - No secrets or credentials are ever transmitted over serial or MQTT.

---

## 6. Serial Protocol

The firmware outputs deterministic, newline-delimited JSON Lines (`\n` terminated):

### IMU Telemetry (100 Hz)
```json
{
  "type": "imu",
  "timestamp_ms": 123450,
  "ax_g": 0.0125,
  "ay_g": -0.0241,
  "az_g": 0.9854,
  "gx_dps": 0.31,
  "gy_dps": -0.12,
  "gz_dps": 0.08
}
```

### GPS Telemetry (1 Hz / 2-second rate)
```json
{
  "type": "gps",
  "timestamp_ms": 124000,
  "valid": true,
  "latitude": 22.719568,
  "longitude": 75.857725,
  "satellites": 8,
  "hdop": 1.2
}
```

### Backward Compatibility
The Stage 6L serial parser (`parse_serial_stream_line`) also transparently parses:
- 7-Column CSV: `timestamp_ms,ax,ay,az,gx,gy,gz`
- 6-Column CSV: `ax,ay,az,gx,gy,gz`
- Legacy Stage 5F pipe format: `ACC(g): ax, ay, az | GYRO(deg/s): gx, gy, gz`

---

## 7. GPS Cache

- **Location:** `src/fall_confirmation/stage_6L_gps_cache.py`
- **Class:** `GpsTelemetryCache`
- **Features:**
  - Background MQTT subscriber to `fall_detection/stage6/gps` (thread-safe).
  - Validates coordinate bounds: $-90 \le \text{lat} \le 90$, $-180 \le \text{lon} \le 180$.
  - Rejects $(0.0, 0.0)$ null island coordinates and unparsed NMEA tokens.
  - Computes age of latest fix (`age_seconds`) and detects staleness (threshold configurable, default 30.0s).
  - API method: `get_latest_location()` returns schema compatible with Stage 6F/6H:
    ```python
    {
        "fix_valid": bool,
        "latitude": Optional[float],
        "longitude": Optional[float],
        "satellites": Optional[int],
        "hdop": Optional[float],
        "timestamp": Optional[str],
        "age_seconds": Optional[float]
    }
    ```
  - Direct programmatic injection support (`update_location(...)`) for unit testing and offline simulation.
  - Zero database writes — strictly in-memory.

---

## 8. Integrated Hardware Bridge

- **Location:** `scripts/stage_6/run_stage_6L_hardware_bridge.py`
- **Function:** `run_stage_6L_hardware_bridge(...)`
- **Pipeline Implementation:**
  1. Opens serial port (`COM22` or configured port) @ 115200 baud with fallback to `MockSensorStream`.
  2. Receives and parses serial stream; extracts 100 Hz IMU data.
  3. Maintains sliding FIFO buffer of $W = 50$ samples (0.50 s) with $S = 25$ sample step (0.25 s).
  4. Calls frozen `extract_100_features(win_dict)` from Stage 4G to generate exact 100 kinematic features.
  5. Scales features using frozen `stage_4D_scaler.joblib`.
  6. Computes class 1 posterior probability $P$ via frozen `stage_4D_random_forest.joblib`.
  7. Passes probability and dynamic kinematic markers (`cur_dyn_acc`, `cur_gyr_mag`, `cur_jerk`) into frozen `TemporalConfirmationAdapter()`.
  8. Gating rule N3 requires $P \ge 0.60$ for 3 consecutive overlapping windows (0.75 s duration) with dynamic acceleration $\ge 1.80g$ and angular velocity $\ge 100.0^\circ/\text{s}$.
  9. Upon confirmed fall: queries `GpsTelemetryCache` for latest valid, non-stale coordinates.
  10. Invokes frozen `MqttFallPublisher` (`src/mqtt/mqtt_publisher.py`) to publish confirmed fall event to `fall_detection/stage6/fall`.
  11. Stage 6G gateway picks up the MQTT payload and forwards it into the Flask backend, achieving full end-to-end integration without bypassing MQTT.

---

## 9. MQTT Integration

| Role | Topic | Payload Format |
|---|---|---|
| ESP32 GPS Telemetry | `fall_detection/stage6/gps` | `{"fix_valid": true, "latitude": 22.7196, "longitude": 75.8577, "satellites": 8, "hdop": 1.1, "timestamp": "..."}` |
| Stage 6L Fall Event | `fall_detection/stage6/fall` | `{"event": "CONFIRMED_FALL", "stage": "6L", "event_id": "STAGE6L-FALL-...", "confidence": 0.9014, "location": {...}}` |

- **Broker:** Mosquitto 2.1.2 running on `127.0.0.1:1883`.
- **Publisher:** Uses frozen `src.mqtt.mqtt_publisher.MqttFallPublisher`.
- **Consumer:** Uses frozen `src.mqtt.mqtt_subscriber.MqttGatewaySubscriber`.

---

## 10. Network & Configuration Architecture

- **Environment Template:** `.env.example`
  ```env
  MQTT_BROKER_HOST=127.0.0.1
  MQTT_BROKER_PORT=1883
  MQTT_FALL_TOPIC=fall_detection/stage6/fall
  MQTT_GPS_TOPIC=fall_detection/stage6/gps
  BACKEND_URL=http://127.0.0.1:5000
  SERIAL_PORT=COM22
  SERIAL_BAUD=115200
  WIFI_SSID=
  WIFI_PASSWORD=
  ```
- **Dynamic Configuration:**
  Both `GpsTelemetryCache` and `run_stage_6L_hardware_bridge` dynamically read environment variables (`os.environ`) at runtime with fallback defaults.
  No IP addresses or passwords are hardcoded in the application logic.

---

## 11. End-to-End Orchestrator

- **Location:** `scripts/stage_6/run_stage_6L_hardware_test.py`
- **Features:**
  1. Performs preflight checks:
     - Mosquitto TCP socket connection test (port 1883).
     - Flask backend REST health check (`/api/health`).
     - Serial COM port existence check via `pyserial.tools.list_ports`.
  2. Spawns/manages Stage 6G Gateway if requested.
  3. Initializes GPS Telemetry Cache.
  4. Launches Hardware Inference Bridge.
  5. Monitors event flow safely.
  6. Provides clean startup error messages when dependencies are offline.
  7. Does NOT trigger synthetic falls or real notifications unless explicitly placed in simulated test mode.

---

## 12. Stage 6L Automated Test Suite

- **Location:** `fall_detection_dataset/scripts/stage_6/test_stage_6L_hardware_integration.py`
- **Test Categories:**
  - **GPS Cache Unit Tests (Tests 1–6):**
    - `test_01_gps_cache_initial_empty`
    - `test_02_gps_cache_valid_mqtt_payload`
    - `test_03_gps_cache_rejects_out_of_range_coords`
    - `test_04_gps_cache_stale_detection`
    - `test_05_gps_cache_latest_valid_fix_schema`
    - `test_06_gps_cache_direct_injection`
  - **Serial Parser Unit Tests (Tests 7–12):**
    - `test_07_parse_json_lines_imu`
    - `test_08_parse_json_lines_gps`
    - `test_09_parse_legacy_pipe_format`
    - `test_10_parse_legacy_7_column_csv`
    - `test_11_parse_legacy_6_column_csv`
    - `test_12_parse_rejects_malformed_and_headers`
  - **Hardware Bridge Integration Tests (Tests 13–17):**
    - `test_13_hardware_bridge_stationary_zero_falls`
    - `test_14_hardware_bridge_dynamic_fall_detected`
    - `test_15_hardware_bridge_live_gps_attached`
    - `test_16_hardware_bridge_missing_gps_safe_null`
    - `test_17_hardware_bridge_dry_run_skips_publishing`
  - **Configuration & Preflight Tests (Tests 18–20):**
    - `test_18_environment_configuration_loading`
    - `test_19_missing_serial_port_raises_clear_error`
    - `test_20_preflight_checks_offline_detection`

---

## 13. Test Results

### Stage 6L New Suite Execution
```text
======================================================================
Ran 20 tests in 5.189s

OK
```
**Result: 20 / 20 PASS (100%)**

---

## 14. Existing Regression Results

Ran global test runner `scratch/run_all_regressions.py` across all 13 test suites:

| Suite | Tests | Status |
|---|---|---|
| Stage 5A (Fall Event Backend) | 18 | PASS |
| Stage 5B (Dashboard Web API) | 32 | PASS |
| Stage 5C (End-to-End Persistence) | 12 | PASS |
| Stage 5D (Smoke Tests) | 6 | PASS |
| Stage 5E (Hardware Validation) | 7 | PASS |
| Stage 5F (Live Serial Bridge) | 6 | PASS |
| Stage 6B (MQTT Client & Topics) | 4 | PASS |
| Stage 6F (MQTT Fall Publisher) | 7 | PASS |
| Stage 6G (MQTT Gateway Subscriber) | 10 | PASS |
| Stage 6H (Dashboard GPS Location) | 9 | PASS |
| Stage 6I (WhatsApp Notifications) | 22 | PASS |
| Stage 6J (Reliable Dispatch & Retry) | 26 | PASS |
| Stage 6K (Monitoring Dashboard) | 22 | PASS |
| **Stage 6L (Hardware Integration)** | **20** | **PASS** |
| **TOTAL** | **201** | **ALL PASS** |

- **Existing Regression Baseline:** **181 / 181 PASS**
- **New Stage 6L Tests:** **20 / 20 PASS**
- **Total Combined Tests:** **201 / 201 PASS**

---

## 15. Protected Workspace Verification

- **Protected Workspace:** `C:\Users\ROHIT\Desktop\major project`
- **Files Modified during Stage 6L Phase 2:** **0 files modified**
- Verification confirmed via filesystem audit, modification timestamps, and git status.

---

## 16. Remaining Hardware Setup Requirements

Before executing physical hardware validation (Phase 3):
1. **Physical Wiring Check:**
   - ESP32 3V3 / GND $\to$ MPU6050 VCC / GND
   - ESP32 GPIO 21 (SDA) $\to$ MPU6050 SDA
   - ESP32 GPIO 22 (SCL) $\to$ MPU6050 SCL
   - ESP32 3V3 / GND $\to$ NEO-6M VCC / GND
   - ESP32 GPIO 16 (RX2) $\to$ NEO-6M TX
   - ESP32 GPIO 17 (TX2) $\to$ NEO-6M RX
2. **Firmware Flashing Preparation:**
   - Configure local Wi-Fi SSID and Password in `esp32_fall_gps_unified.ino`.
   - Set broker IP address to local PC LAN address (e.g., `192.168.1.10`) or connect via USB.
   - Flash sketch to ESP32 using Arduino IDE or `esptool.py`.
3. **GPS Antenna Exposure:**
   - Place NEO-6M ceramic patch antenna near an open window or outdoor line-of-sight to acquire satellite fix (indicated by blinking PPS LED).

---

## 17. Phase 3 Test Procedure (Recommended)

When approved for Phase 3:
1. **Test A: Stationary Baseline (30 seconds):**
   - Wearable placed static on a flat desk.
   - Verify 100 Hz IMU telemetry in terminal without disconnections.
   - Verify probability stays $< 0.10$ and `confirmed_falls == 0`.
2. **Test B: Controlled Motion / ADL (60 seconds):**
   - Normal walking, sitting on a chair, turning.
   - Verify dynamic acceleration stays within normal ADL bounds and no false alarms trigger.
3. **Test C: Controlled Soft Impact Fall (Mattress / Cushion):**
   - Safe drop onto a soft mattress or cushion from standing height.
   - Observe 3 consecutive windows $P \ge 0.60$ and kinematic thresholds.
   - Verify MQTT publication to `fall_detection/stage6/fall`.
   - Verify Stage 6G gateway forwards event to `/api/fall-events`.
   - Verify GPS coordinates are attached and visible on the dashboard.
   - Verify WhatsApp dispatch queue registers event and dispatches notification.

---

## Required Final Status

- **Stage 6L Phase 2:** `COMPLETE`
- **Existing Regression:** `181/181 PASS`
- **New Stage 6L Tests:** `20/20 PASS`
- **Total Tests:** `201/201 PASS`
- **Protected Original (`major project`):** `0 files modified`
- **Physical Hardware Test:** `NOT YET EXECUTED`
- **Ready for Stage 6L Phase 3?:** `YES`
