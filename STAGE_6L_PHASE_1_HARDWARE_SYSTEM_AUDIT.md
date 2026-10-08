# STAGE 6L — PHASE 1: END-TO-END HARDWARE + SYSTEM VALIDATION AUDIT

**Project:** Standalone Wearable Fall Detection System  
**Stage:** Stage 6L (Phase 1: Read-Only System Audit)  
**Active Development Folder:** `C:\Users\ROHIT\Desktop\major project - Copy`  
**Protected Original Folder:** `C:\Users\ROHIT\Desktop\major project` (Pristine / Untouched)  
**Audit Mode:** READ-ONLY (Zero code, firmware, configuration, or model modifications)  
**Date / Timestamp:** 2026-10-07T06:15:00Z  

---

## 1. Executive Summary

A comprehensive, non-destructive, read-only architectural and hardware audit was conducted across all software, firmware, model, pipeline, network, database, notification, and dashboard components of the Standalone Fall Detection System.

The audit verified that while individual subsystems (Stage 4D Random Forest classifier, Stage 4E temporal confirmation, Stage 5A–5D Flask/SQLite backend, Stage 6G MQTT gateway, Stage 6H GPS dashboard display, Stage 6I/6J reliable WhatsApp delivery, and Stage 6K monitoring dashboard) function and pass unit and simulated integration tests (181 / 181 automated tests passing), **the system as currently composed is NOT ready for live, hardware-in-the-loop end-to-end testing.**

Five critical blockers currently prevent complete physical chain execution:
1. **Missing Unified ESP32 Firmware:** No single firmware exists in the repository that samples MPU6050 accelerometer/gyroscope data and NEO-6M GPS data simultaneously.
2. **Disconnected Live GPS Pipeline:** The ESP32 publishes GPS telemetry to `fall_detection/stage6/gps`, but no backend subscriber or live bridge consumes this topic to attach real-time coordinates to fall events.
3. **Inference Bridge Bypasses MQTT:** The serial inference bridge (`stage_5F_live_bridge.py`) POSTs directly to the HTTP backend, bypassing the Stage 6F MQTT publisher and Stage 6G MQTT gateway.
4. **Hardcoded IP & Broker Binding Constraints:** Firmware and Mosquitto configuration are tied to a historical hotspot IP (`10.163.182.136`), whereas the active LAN interface is currently `192.168.1.10`.
5. **No Unified Hardware Orchestrator:** No launch sequence script coordinates the concurrent operation of Mosquitto, the Flask backend, the MQTT gateway, the serial IMU inference bridge, and the GPS telemetry subscriber.

---

## 2. Workspace Safety Verification

| Parameter | Value | Status |
|:---|:---|:---:|
| **ACTIVE_WORKSPACE** | `C:\Users\ROHIT\Desktop\major project - Copy` | **VERIFIED** |
| **PROTECTED_WORKSPACE** | `C:\Users\ROHIT\Desktop\major project` | **VERIFIED** |
| **PROTECTED_WORKSPACE_STATUS** | Pristine (0 files modified) | **VERIFIED** |
| **FROZEN_COMPONENT_STATUS** | Stage 4, 5A–5F, 6B, 6F–6K intact | **VERIFIED** |
| **PHASE_1_MUTATIONS** | Zero code, firmware, or schema changes | **VERIFIED** |

---

## 3. End-to-End Architecture Audit

The intended real-world system flow was audited against active source code:

```
[Physical ESP32 + MPU6050 + NEO-6M]
       │
       ├── (1) 100 Hz IMU Stream via USB UART (COM22 @ 115200 baud)
       │         ↓
       │    [PC: stage_5F_live_bridge.py / inference engine]
       │         ↓ Feature Extraction (100 features from 50 samples)
       │         ↓ Random Forest Model (stage_4D_random_forest.joblib)
       │         ↓ Temporal Confirmation Adapter (Rule N3, threshold 0.60, cooldown 2.0s)
       │         ↓
       │    [CONFIRMED_FALL Event]
       │         ↓
       │    [Stage 6F MqttFallPublisher] ─── (GAP: Not invoked by stage_5F_live_bridge)
       │         ↓ Topic: fall_detection/stage6/fall
       │
       ├── (2) GPS NMEA via UART2 (GPIO 16/17 @ 9600 baud)
       │         ↓
       │    [ESP32: TinyGPSPlus parser]
       │         ↓ Topic: fall_detection/stage6/gps ─── (GAP: No consumer in src/)
       │
       ↓
[Mosquitto MQTT Broker] (Port 1883)
       ↓
[Stage 6G MQTT Gateway] (Subscribes to fall_detection/stage6/fall)
       ↓ HTTP POST /api/fall-events
[Flask Backend] (NotifyingFallEventService)
       ↓
[SQLite Database] (fall_events committed)
       ↓ Post-Commit Hook (Database-First Guarantee)
[Stage 6J WhatsApp Service] (queue -> worker -> retries -> provider)
       ↓
[Stage 6K Monitoring Dashboard] (/notifications & /api/notifications/*)
```

---

## 4. Sensor Audit (MPU6050)

| Attribute | Trained Model Assumption | Firmware / Ingestion Status | Verdict |
|:---|:---|:---|:---:|
| **Sensor Type** | InvenSense MPU6050 | Documented in `docs/ESP32_MPU6050_SETUP.md` | **PARTIALLY VERIFIED** |
| **Bus / Pins** | I2C Fast Mode (400 kHz) | SDA=GPIO 21, SCL=GPIO 22, AD0=GND (0x68) | **VERIFIED** (Design) |
| **Accelerometer Units** | Standard gravity ($g$) | Expected: $1g \approx 9.80665\text{ m/s}^2$ | **VERIFIED** |
| **Gyroscope Units** | Degrees per second ($\text{deg/s}$) | Expected: $\text{deg/s}$ | **VERIFIED** |
| **Sampling Frequency** | 100.0 Hz ($\Delta t = 10.0\text{ ms}$) | Expected in `stage_4G`/`stage_5E`/`stage_5F` | **VERIFIED** |
| **Window Parameters** | 50 samples ($0.50\text{s}$), 50% overlap (step 25) | Hardcoded in `stage_4G_live_validator.py` | **VERIFIED** |
| **Feature Extraction** | 100 features across 10 signals | Exact parity with `feature_order.json` | **VERIFIED** |
| **Firmware in Repo** | Streaming 100 Hz MPU6050 sketch | **MISSING** in `firmware/` directory | **CRITICAL** |

---

## 5. ESP32 Firmware Audit

Three Arduino `.ino` sketches were inventoried in `firmware/`:

### 5.1 `firmware/esp32_mqtt/esp32_mqtt.ino` (Stage 6C)
- **Role:** Isolated Wi-Fi & MQTT basic ping test.
- **Sensors:** NONE (no MPU6050, no GPS).
- **Wi-Fi SSID / Password:** `<REDACTED>` (hardcoded).
- **MQTT Broker:** `10.163.182.136:1883`.
- **Topic:** `fall_detection/stage6/test`.
- **Status:** **VERIFIED** (Isolated test only).

### 5.2 `firmware/esp32_gps_test/esp32_gps_test.ino` (Stage 6D)
- **Role:** Dedicated NEO-6M GPS ingestion test over UART2.
- **Sensors:** NEO-6M only (no MPU6050).
- **Network:** NONE (no Wi-Fi, no MQTT).
- **Pins:** GPIO 16 (RX), GPIO 17 (TX), 9600 baud.
- **Status:** **VERIFIED** (Isolated test only).

### 5.3 `firmware/esp32_gps_mqtt_test/esp32_gps_mqtt_test.ino` (Stage 6E)
- **Role:** GPS + Wi-Fi + MQTT combined telemetry.
- **Sensors:** NEO-6M only. Line 19 states: *"Completely isolated from MPU6050 fall detection ML pipeline and models."*
- **MQTT Broker:** `10.163.182.136:1883`.
- **Topic:** `fall_detection/stage6/gps`.
- **Payload:** JSON with `fix_valid`, `latitude`, `longitude`, `satellites`, `hdop`.
- **Status:** **VERIFIED** (GPS-only telemetry).

### Firmware Summary:
**No firmware currently in the repository samples the MPU6050 and GPS concurrently.**

---

## 6. GPS Subsystem Audit

| Component | Specification | Status | Notes |
|:---|:---|:---:|:---|
| **Module** | u-blox NEO-6M Breakout | **VERIFIED** | Default 9600 baud, 8-N-1 |
| **Serial Bus** | ESP32 Hardware UART2 | **VERIFIED** | GPIO 16 (RX2), GPIO 17 (TX2) |
| **Parser** | TinyGPSPlus v1.0.3 | **VERIFIED** | Continuous non-blocking parsing |
| **Fix Stale Threshold** | `MAX_FIX_AGE_MS = 2500` (2.5s) | **VERIFIED** | Transitions to `STATE_FIX_LOST` |
| **MQTT Telemetry** | Published to `fall_detection/stage6/gps` | **VERIFIED** | Verified in Stage 6E (150 msgs) |
| **Live Fall Link** | Link between GPS MQTT and Fall Event | **MISSING** | No consumer for GPS topic in `src/` |

---

## 7. MQTT Broker & Networking Audit

- **Broker Software:** Eclipse Mosquitto v2.1.2 (Running as Windows Service, PID 15304).
- **Port:** 1883.
- **Active Bindings (`mosquitto.conf`):**
  - `listener 1883 127.0.0.1` (allow_anonymous true)
  - `listener 1883 10.163.182.136` (allow_anonymous true)
- **Active Host IPv4 Interfaces:**
  - Wi-Fi: `192.168.1.10`
  - Loopback: `127.0.0.1`
- **Network Gap Identified:**
  Mosquitto is bound to `10.163.182.136`, but the PC is currently connected to a Wi-Fi network with IP `192.168.1.10`. An external ESP32 attempting to connect to `10.163.182.136` will time out unless the PC and ESP32 are connected to the OnePlus mobile hotspot.

---

## 8. MQTT Gateway Audit (Stage 6G)

- **File:** `src/fall_confirmation/stage_6G_mqtt_gateway.py`.
- **Class:** `MqttBackendGateway`.
- **Subscribed Topic:** `fall_detection/stage6/fall` (QoS 1).
- **JSON Validation:** Validates schema, requires `event == "CONFIRMED_FALL"`.
- **Backend Invocation:** HTTP `POST http://127.0.0.1:5000/api/fall-events` (timeout 3.0s).
- **Data Preservation:** Preserves `event_id`, `detected_at`, `confidence`, `location` verbatim.
- **Status:** **VERIFIED** (10 / 10 automated tests pass).

---

## 9. Fall Confirmation & Inference Audit (Stage 4 & 5)

- **Production Model Files (Verified Intact):**
  - `fall_detection_dataset/models/stage_4D/stage_4D_random_forest.joblib` (6,054,393 bytes)
  - `fall_detection_dataset/models/stage_4D/scaler.joblib` (3,015 bytes)
  - `fall_detection_dataset/models/stage_4D/feature_order.json` (1,733 bytes)
- **Temporal Confirmation Adapter:**
  - File: `src/fall_confirmation/temporal_adapter.py`.
  - Parameters: Threshold $0.60$, Rule N3 ($3\times 0.25\text{s} = 0.75\text{s}$ consecutive), Cooldown $2.0\text{s}$.
- **Live Bridge Execution:**
  - File: `src/fall_confirmation/stage_5F_live_bridge.py`.
  - Dispatches directly via `post_fall_event_to_backend()`. It does **not** route through `MqttFallPublisher` (`stage_6F_mqtt_fall_publisher.py`).
- **Status:** **PARTIALLY VERIFIED** (Inference logic verified; MQTT publishing decoupled).

---

## 10. Flask Backend Audit (Stage 5A)

- **Application Factory:** `create_notifying_app` in `src/notifications/notification_listener.py`.
- **Base REST API:** `create_app` in `src/fall_event_backend/api.py`.
- **Endpoints:**
  - `GET /api/health` — HTTP 200
  - `POST /api/fall-events` — HTTP 201
  - `GET /api/fall-events` — HTTP 200
  - `GET /api/fall-events/<event_id>` — HTTP 200 / 404
  - `PATCH /api/fall-events/<event_id>/acknowledge` — HTTP 200
  - `GET /dashboard` — HTTP 200
  - `GET /notifications` — HTTP 200
  - `GET /api/notifications/monitoring` — HTTP 200
  - `GET /api/notifications/deliveries` — HTTP 200
  - `GET /api/notifications/events/<event_id>` — HTTP 200 / 404
- **Status:** **VERIFIED** (All endpoints functional).

---

## 11. SQLite Database Audit

- **File Path:** `data/fall_events.db`.
- **Tables:**
  - `fall_events` (Stage 5A): Core fall event store.
  - `notification_deliveries` (Stage 6J): Notification delivery state and retry audit trail.
- **Safety Enforcement:**
  - Stage 6K monitoring queries execute `PRAGMA query_only = ON;`.
  - Zero schema migrations needed or introduced in Stage 6K/6L.
- **Status:** **VERIFIED**.

---

## 12. Stage 6J Notification Subsystem Audit

- **File:** `src/notifications/whatsapp_service.py`.
- **Queue & Persistence:** In-memory worker thread synchronized with SQLite `notification_deliveries`.
- **Retry Mechanism:** Exponential backoff ($5\text{s}, 15\text{s}, 60\text{s}, 300\text{s}$, max 5 attempts).
- **Dead-Letter Handling:** Non-retryable errors or exhausted attempts marked `DEAD_LETTER`.
- **Privacy & Security:** Phone numbers strictly masked (`whatsapp:+91*****3210`); zero secrets exposed.
- **Active Environment State:** Twilio credentials currently `<MISSING>` in active shell environment; service starts safely in `DISABLED` mode without crashing.
- **Status:** **VERIFIED** (26 / 26 automated tests pass).

---

## 13. Stage 6K Monitoring Dashboard Audit

- **Files:**
  - `src/notifications/notification_monitor.py`
  - `src/fall_event_backend/templates/notifications.html`
  - `src/fall_event_backend/static/js/notifications_monitor.js`
- **Features:** 2.5s polling, lifecycle metrics (SENT, QUEUED, RETRY_WAIT, DEAD_LETTER, etc.), queue health, audit table, per-event modal diagnostics.
- **Status:** **VERIFIED** (22 / 22 automated tests pass).

---

## 14. Network & Process Requirements for Live Hardware Testing

To perform a physical test, the following 6 processes must run concurrently:

| # | Process / Component | Host / Port | Dependency |
|:---:|:---|:---|:---|
| 1 | **Wi-Fi AP / Hotspot** | 2.4 GHz LAN | Provides IP subnet to PC & ESP32 |
| 2 | **Mosquitto Broker** | Port 1883 | Bound to reachable LAN IP |
| 3 | **Flask Backend** | `127.0.0.1:5000` | Requires SQLite access |
| 4 | **Stage 6G MQTT Gateway** | Background Daemon | Connects to Broker & Backend |
| 5 | **ESP32 Firmware** | Hardware DevKit | Connected to MPU6050 + NEO-6M |
| 6 | **Serial Bridge / Inference** | `COM22` @ 115200 | Reads MPU6050, runs RF model |

---

## 15. Existing Test Inventory

Executed across the entire project lifecycle via `scratch/run_all_regressions.py`:

| Suite | File | Tests | Classification | Status |
|:---|:---|:---:|:---|:---:|
| **Stage 5A** | `scripts/stage_5A/test_fall_event_backend.py` | 18 | Automated Backend / SQLite | **PASS** |
| **Stage 5B** | `scripts/stage_5B/test_stage_5B_dashboard.py` | 32 | Automated Dashboard UI | **PASS** |
| **Stage 5C** | `scripts/stage_5C/test_stage_5C_end_to_end.py` | 12 | Automated End-to-End Pipeline | **PASS** |
| **Stage 5D** | `scripts/stage_5D/test_stage_5D_smoke.py` | 6 | Automated Smoke & Recovery | **PASS** |
| **Stage 5E** | `scripts/stage_5E/test_stage_5E_hardware_validation.py` | 7 | Automated Hardware Ingestion | **PASS** |
| **Stage 5F** | `scripts/stage_5F/test_stage_5F_live_bridge.py` | 6 | Automated Live Bridge Logic | **PASS** |
| **Stage 6B** | `scripts/stage_6/test_stage_6B_mqtt.py` | 4 | Network / Mosquitto Broker | **PASS** |
| **Stage 6F** | `scripts/stage_6/test_stage_6F_mqtt_fall.py` | 7 | Network / MQTT Publisher | **PASS** |
| **Stage 6G** | `scripts/stage_6/test_stage_6G_mqtt_gateway.py` | 10 | Network / Gateway Forwarding | **PASS** |
| **Stage 6H** | `scripts/stage_6/test_stage_6H_dashboard_location.py` | 9 | Automated GPS Dashboard Display | **PASS** |
| **Stage 6I** | `scripts/stage_6/test_stage_6I_whatsapp.py` | 22 | Automated WhatsApp Service | **PASS** |
| **Stage 6J** | `scripts/stage_6/test_stage_6J_notification_reliability.py` | 26 | Automated Retry & Reliability | **PASS** |
| **Stage 6K** | `scripts/stage_6/test_stage_6K_notification_monitoring.py` | 22 | Automated Monitoring & Auditing | **PASS** |
| **TOTAL** | **13 Test Suites** | **181** | **Comprehensive Regression** | **181 / 181 PASS** |

---

## 16. Hardware-in-the-Loop Readiness Evaluation

| Subsystem Chain | Implementation Status | Readiness |
|:---|:---|:---:|
| **MPU6050 Acquisition** | Documented, but missing combined firmware | **NOT READY** |
| **NEO-6M GPS Acquisition** | Verified in Stage 6D (`esp32_gps_test.ino`) | **READY** |
| **ESP32 Wi-Fi + MQTT** | Verified in Stage 6E (`esp32_gps_mqtt_test.ino`) | **READY** |
| **Combined MPU6050 + GPS Firmware** | Not implemented | **NOT READY** |
| **Live Serial Inference Bridge** | Implemented in `stage_5F_live_bridge.py` | **READY** |
| **Bridge → MQTT Integration** | `publish_stage_5f_event` exists but unlinked in 5F | **NOT READY** |
| **Live GPS → Fall Event Link** | No subscriber for `fall_detection/stage6/gps` | **NOT READY** |
| **MQTT Broker & Gateway** | Verified in Stage 6G | **READY** |
| **Flask Backend & SQLite** | Verified in Stage 5A–5D | **READY** |
| **Stage 6J Reliability & Retries** | Verified in Stage 6J | **READY** |
| **Stage 6K Monitoring Dashboard** | Verified in Stage 6K | **READY** |

---

## 17. Critical Issues (Blockers)

1. **[CRITICAL 1] Absence of Unified MPU6050 + GPS Firmware:**  
   The ESP32 currently has no firmware sketch that concurrently reads the MPU6050 IMU on I2C and the NEO-6M GPS on UART2.
2. **[CRITICAL 2] Disconnected GPS Ingestion Stream:**  
   The ESP32 streams GPS data to `fall_detection/stage6/gps`, but no backend service subscribes to this topic or maintains a cached "latest valid coordinates" state to inject into fall event payloads.
3. **[CRITICAL 3] Live Bridge Decoupled from MQTT:**  
   `stage_5F_live_bridge.py` posts directly to HTTP `/api/fall-events` rather than publishing through `MqttFallPublisher` (`fall_detection/stage6/fall`).
4. **[CRITICAL 4] Network IP Binding Mismatch:**  
   Firmware and `mosquitto.conf` hardcode IP `10.163.182.136`. Active LAN IP is `192.168.1.10`. The ESP32 will fail to connect unless the OnePlus hotspot network is active or broker bindings are updated.

---

## 18. Non-Critical Issues

1. **Serial Port Disconnection on COM22:** If the ESP32 is reconnected to a different USB port, `--port` must be passed dynamically.
2. **GPS Fix Acquisition Delay:** Indoors, NEO-6M GPS can take 60–180 seconds to acquire a cold satellite lock.
3. **Twilio Credentials Missing in Local Env:** Currently starts in `DISABLED` mode unless environment variables are populated.

---

## 19. Missing Components for Complete Real-World Chain

1. **Unified ESP32 Firmware:** A sketch that samples MPU6050 at 100 Hz over USB Serial while simultaneously reading GPS on UART2 and publishing GPS coordinates via MQTT.
2. **GPS Telemetry Gateway / Cacher:** A lightweight background service that subscribes to `fall_detection/stage6/gps` and maintains the most recent valid GPS coordinates in memory.
3. **End-to-End Live Orchestrator:** An integration runner that wires the serial inference engine with the MQTT publisher and the GPS cache.

---

## 20. Proposed Test Procedure (for Future Phase 2)

### Test A — Sensor Only (MPU6050)
- Verify 100 Hz serial output on COM22.
- Check stationary baseline: $AccZ \approx 1.0g$, $AccX \approx 0.0g$, $AccY \approx 0.0g$, $|Gyr| < 1.0\text{ deg/s}$.

### Test B — GPS Only (NEO-6M)
- Verify NMEA stream on UART2 (GPIO 16/17 @ 9600 baud).
- Wait for valid 3D satellite fix (`satellites >= 4`, `hdop < 2.5`).

### Test C — MQTT Only
- Verify ESP32 connects to Mosquitto broker over LAN Wi-Fi.
- Verify heartbeat/telemetry reception on test topic.

### Test D — Gateway Only
- Publish synthetic `CONFIRMED_FALL` to `fall_detection/stage6/fall`.
- Verify Stage 6G gateway parses and forwards to backend.

### Test E — Backend & Persistence
- Verify backend accepts event, assigns UUID, and commits to SQLite.

### Test F — Notification Subsystem
- Verify event transitions through Stage 6J queue (`QUEUED` -> `SENT`).
- Verify masked number and delivery record in `notification_deliveries`.

### Test G — Full Chain Physical Test
- Perform controlled, safe physical tilt/drop onto a safety mat.
- Observe: IMU stream -> Random Forest inference -> Rule N3 confirmation -> MQTT publish -> Gateway -> Backend -> SQLite -> Notification -> Stage 6K Dashboard.

---

## 21. Pass/Fail Criteria

- **Sensor:** Sample interval $\Delta t = 10 \pm 1\text{ ms}$; static error $< 0.1g$.
- **GPS:** Valid coordinates ($lat \ne 0.0$, $lon \ne 0.0$), age $< 2.5\text{s}$.
- **MQTT:** Zero packet drops on QoS 1; reconnects within 5s of network disruption.
- **Gateway:** Correct JSON mapping into backend schema; invalid/unconfirmed payloads rejected.
- **Backend:** HTTP 201 Created; event visible via `GET /api/fall-events`.
- **Notification:** Status reaches `SENT` (or `QUEUED`/`RETRY_WAIT` under test mock); no credentials or raw numbers leaked.
- **Dashboard:** Event appears on `/dashboard`; delivery appears on `/notifications`.

---

## 22. Security Findings

- [x] Zero Wi-Fi passwords printed in reports (`<REDACTED>`).
- [x] Zero MQTT credentials exposed (`<REDACTED>`).
- [x] Zero Twilio Account SIDs or Auth Tokens exposed (`<REDACTED>`).
- [x] Zero raw phone numbers printed; all masked as `whatsapp:+91*****3210`.
- [x] Read-only SQLite enforcement active on monitoring endpoints (`PRAGMA query_only = ON;`).

---

## 23. Recommended Stage 6L Phase 2 Actions

1. Author a unified ESP32 firmware sketch (`firmware/esp32_fall_gps_unified/`) that supports concurrent MPU6050 reading and NEO-6M GPS ingestion.
2. Implement a live GPS telemetry cacher subscribing to `fall_detection/stage6/gps` to supply real-time coordinates to `MqttFallPublisher`.
3. Provide an integrated hardware bridge runner (`scripts/stage_6/run_stage_6L_hardware_bridge.py`) connecting COM22 serial inference with `stage_6F_mqtt_fall_publisher.py`.
4. Ensure Mosquitto broker LAN binding and Wi-Fi network configuration are aligned with the active router/hotspot subnet.
5. Execute the staged Tests A through G sequentially.

---

## 24. Most Important Final Question

### Is the system ready for controlled hardware-in-the-loop testing?

# **NO**

### Exact Explanation & Priority Blockers:
1. **BLOCKER 1 (Firmware):** The repository lacks a unified ESP32 firmware combining MPU6050 IMU sampling with NEO-6M GPS. Currently, uploading the existing Stage 6E firmware leaves the MPU6050 completely unread.
2. **BLOCKER 2 (Data Flow):** The GPS stream published by the ESP32 (`fall_detection/stage6/gps`) has no active subscriber in the backend or live bridge; live GPS coordinates cannot reach the fall event payload.
3. **BLOCKER 3 (Inference Bridge):** The live inference script (`stage_5F_live_bridge.py`) bypasses the MQTT publisher and gateway entirely, posting directly to HTTP.
4. **BLOCKER 4 (Network Configuration):** The firmware and Mosquitto configuration are tied to `10.163.182.136`, while the active workstation IP is `192.168.1.10`.
