# Standalone Wearable Fall Detection System
## Production-Hardened Real-Time Edge-to-Cloud Cyber-Physical System

**Active Workspace**: `C:\Users\ROHIT\Desktop\major project - Copy`  
**Current Milestone**: Stage 9 — Production Hardening & Final Demonstration Readiness  
**System Status**: Production Ready | Detection Core Cryptographically Frozen  

---

## 1. Project Title
**Standalone Wearable Fall Detection System with Edge Inertial Telemetry, Multi-Gate Temporal Confirmation, MQTT Microservices, and Incident Command Center Dashboard.**

---

## 2. Project Objective
To develop, validate, and production-harden an end-to-end wearable cyber-physical fall detection system that:
- Captures high-frequency body motion kinematics ($100\text{ Hz}$) from lumbar-mounted inertial sensors (ESP32 + MPU6050).
- Discriminates true catastrophic fall impacts from aggressive daily living activities (ADLs) using a feature-engineered Random Forest classifier with dynamic kinematic acceleration and gyro gates.
- Enforces strict temporal multi-window confirmation ($N=3$, $P \ge 0.60$) and an incident latch guaranteeing that **one physical fall produces exactly one logical incident record**.
- Dispatches emergency telemetry with real-time GPS geolocation over an asynchronous MQTT broker to a unified Flask/SQLite backend and a responsive operator Command Center Dashboard.
- Provides one-click startup, robust health diagnostics, graceful shutdown, and zero-risk simulated demonstration capabilities for academic and engineering evaluation.

---

## 3. System Architecture
The system employs a layered decoupled architecture:

```
[ Physical Sensors: MPU6050 (±16g, ±2000 dps) + NEO-6M GPS ]
                           │
                           ▼ (I2C @ 400kHz + UART2 @ 9600 baud)
           [ ESP32 Wearable Edge Node ]
                           │
                           ▼ (USB Serial @ 460800 baud to COM22)
          [ Stage 6L Host Hardware Bridge ]
           - 50-sample sliding window (0.50s), 25-stride (0.25s)
                           │
                           ▼
          [ FROZEN Detection & Confirmation Core ]
           - 100-Feature Statistical & Kinematic Extraction
           - StandardScaler (scaler.joblib)
           - Random Forest Classifier (stage_4D_random_forest.joblib)
           - Dynamic Accel Gate (|Acc| >= 2.0g) & Gyro Gate (|Gyr| >= 180 dps)
           - Temporal Confirmation Adapter (N=3 consecutive windows >= 0.60)
           - Stage 7B Incident Latch & Re-arm State Machine
                           │
                           ▼ (JSON: CONFIRMED_FALL payload)
          [ Mosquitto MQTT Broker (127.0.0.1:1883) ]
           - Topic: fall_detection/stage6/fall
                           │
                           ▼
          [ Stage 6G Ingestion Gateway ]
           - Asynchronous MQTT Subscriber Thread
                           │
                           ▼ (Internal HTTP POST /api/fall-events)
          [ Unified Flask Backend & SQLite Database (fall_events.db) ]
           - Port: 5000 (Localhost only)
           - Health probes & Incident State Management
                           │
                           ▼ (HTTP Polling & Leaflet GIS Engine)
          [ Command Center Web Dashboard (Stage 8B) ]
           - Real-time alert banner, countdown, acknowledgment, resolution & map
```

---

## 4. Hardware Components
- **Microcontroller**: ESP32 Dev Module (ESP-WROOM-32, dual-core Tensilica Xtensa 32-bit LX6 @ 240 MHz).
- **IMU Sensor**: InvenSense MPU6050 6-DOF (Triaxial Accelerometer + Triaxial Gyroscope).
  - Accelerometer Range: **$\pm 16g$** ($2048\text{ LSB}/g$).
  - Gyroscope Range: **$\pm 2000^\circ/\text{s}$** ($16.4\text{ LSB}/(^\circ/\text{s})$).
  - I2C Bus: SDA on **GPIO 21**, SCL on **GPIO 22** @ 400 kHz Fast Mode.
- **GPS Receiver**: u-blox NEO-6M GPS module.
  - UART Interface: Hardware UART2 (RX on **GPIO 16**, TX on **GPIO 17**) @ 9600 baud.
- **Serial Interface**: High-throughput USB UART running at **460800 baud** on `COM22`.
- Detailed pinout available in [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md).

---

## 5. Software Stack
- **Firmware**: C++ / Arduino ESP32 Core.
- **Backend**: Python 3.10+ with Flask REST framework.
- **Persistence**: SQLite 3 with WAL journal mode (`fall_events.db`).
- **Messaging**: Eclipse Mosquitto MQTT Broker (`paho-mqtt` client library).
- **Machine Learning**: `scikit-learn` (v1.5.1), `joblib`, `numpy`, `scipy`.
- **Frontend Dashboard**: Vanilla HTML5, Modern CSS Design System, Vanilla ES6 JavaScript, and Leaflet.js / OpenStreetMap GIS.
- **Operating Environment**: Microsoft Windows 10/11 (AMD64).

---

## 6. Dataset Information
The machine learning pipeline was trained and benchmarked across standardized benchmark datasets:
- **SisFall**: A benchmark dataset for fall detection and activities of daily living across 38 subjects (15 elderly, 23 young adults).
- **MobiFall & SmartFall**: Complementary IMU datasets used to evaluate diverse postures, slips, trips, forward falls, backward falls, lateral falls, and intense ADLs (running, jumping, stairs).
- Total balanced training and validation splits were processed into 50-sample windows at 100 Hz.

---

## 7. Machine-Learning Pipeline
1. **Kinematic Feature Extraction**: 100 features computed per 50-sample sliding window:
   - Time-domain statistics: Mean, standard deviation, variance, skewness, kurtosis, min, max, peak-to-peak amplitude.
   - Dynamic acceleration separation via high-pass filtering ($\alpha = 0.969$).
   - Jerk dynamics ($\Delta a / \Delta t$, $\Delta t = 0.01\text{ s}$).
   - Signal magnitude vectors ($|Acc|$, $|Gyr|$, $|DynAcc|$).
   - Frequency-domain spectral energy and power spectral density.
2. **Standardization**: Feature matrix scaled via frozen `StandardScaler` (`scaler.joblib`).
3. **Classification**: 100-tree Random Forest Classifier (`stage_4D_random_forest.joblib`) producing continuous fall probabilities $P(\text{fall}) \in [0.0, 1.0]$.
4. **Physical Dynamic Gates**:
   - Dynamic acceleration magnitude gate: $\max(|Acc|) \ge 2.0g$.
   - Angular velocity threshold gate: $\max(|Gyr|) \ge 180^\circ/\text{s}$.

---

## 8. Fall-Confirmation Logic (Rule N3)
To eliminate false alarms caused by transient shocks (e.g., dropping a phone or slamming a door):
- **Decision Window**: Temporal confirmation requires **$N = 3$ consecutive sliding windows** (0.75 seconds total persistence) where $P(\text{fall}) \ge 0.60$ and kinematic gates are satisfied.
- **Temporal Adapter**: Encapsulated in `temporal_adapter.py`.
- **Cryptographic Core Freeze**: The detection model and parameters are strictly frozen and verified against cryptographic hashes prior to every run.

---

## 9. Incident Lifecycle & Anti-Duplicate Latch
The Stage 7B incident latch guarantees **One-Physical-Fall = One-Logical-Incident**:
1. **TRIGGERED / CONFIRMED**: On $N=3$ confirmation, an incident record is created with state `CONFIRMED`.
2. **ACTIVE LATCH**: Latch closes; all subsequent fall windows within the cooldown and post-fall stillness period are suppressed.
3. **ACKNOWLEDGED**: First responder or operator marks the event `ACKNOWLEDGED`.
4. **RE-ARMING**: System re-arms only after verified upright vertical posture ($Acc_Z \approx 1g$, $|Acc| \approx 1g$, $|Gyr| \le 30^\circ/\text{s}$) is maintained continuously for 5.0 seconds.
5. **RESOLVED**: Incident marked `RESOLVED` in database with notes.

---

## 10. MQTT Architecture
- **Broker**: Mosquitto daemon running locally on `127.0.0.1:1883`.
- **Publication Topic**: `fall_detection/stage6/fall`.
- **Telemetry Topic**: `fall_detection/stage6/telemetry`.
- **Payload Schema**:
  ```json
  {
    "event": "CONFIRMED_FALL",
    "event_id": "FALL-20261007T180000Z-01",
    "timestamp": 1728324000.0,
    "confidence": 0.94,
    "source": "ESP32_MPU6050",
    "location": {
      "latitude": 28.6139,
      "longitude": 77.2090,
      "fix": 1
    }
  }
  ```

---

## 11. Backend Architecture
The backend is unified in `scripts/stage_6/run_full_backend.py`:
- Spawns Flask application on `http://127.0.0.1:5000`.
- Hosts embedded `Stage 6G MQTT Gateway` in a daemon thread.
- Exposes REST API endpoints:
  - `GET /api/health` — Microservice liveness probe.
  - `GET /api/system/health` — Full composite health probe (Backend, Database, MQTT, ESP32, GPS).
  - `POST /api/fall-events` — Idempotent incident ingestion.
  - `GET /api/fall-events` — Incident listing and querying.
  - `POST /api/incidents/<id>/acknowledge` — Operator acknowledgment.
  - `POST /api/incidents/<id>/resolve` — Incident resolution.

---

## 12. Database
- **File**: `fall_detection_dataset/fall_events.db`.
- **Schema**:
  - `event_id` (TEXT PRIMARY KEY)
  - `created_at` (TEXT)
  - `detected_at` (TEXT)
  - `confidence` (REAL)
  - `status` (TEXT: `CONFIRMED`, `ACKNOWLEDGED`, `RESOLVED`, `CANCELLED`)
  - `source` (TEXT)
  - `latitude` (REAL), `longitude` (REAL)
  - `notes` (TEXT), `acknowledged_at` (TEXT), `resolved_at` (TEXT)
- **Integrity**: Verified via SQLite `PRAGMA integrity_check` on every launch. Non-destructive; historic records are permanently preserved.

---

## 13. GPS Geolocation
- High-accuracy NMEA sentences ingested via ESP32 UART2.
- Telemetry carries WGS84 decimal latitude and longitude.
- Automatically injected into incident records.
- Dashboard Leaflet engine dynamically pans to victim coordinates and renders interactive emergency map pins.

---

## 14. Dashboard (Stage 8B Command Center)
Access at `http://127.0.0.1:5000/dashboard`:
- **Real-time Status Banner**: Pulsing alert on active falls; calm green in standby.
- **Emergency GIS Map**: Leaflet.js interactive OpenStreetMap interface with geolocation markers.
- **Action Station**: Instant 1-click Acknowledge and Resolve modal controls.
- **Audit History**: Searchable, filterable table showing past incidents, coordinates, and resolution notes.

---

## 15. Health Monitoring (Stage 8A)
Monitored via `GET /api/system/health`:
- Reports composite status: `SYSTEM_READY` (all systems active) or `SYSTEM_DEGRADED` (missing optional hardware).
- Tracks individual subcomponent health:
  - `backend`: `READY`
  - `database`: `READY`
  - `mqtt`: `CONNECTED` / `OFFLINE`
  - `gateway`: `READY` / `DEGRADED`
  - `esp32`: `READY` / `STALE` / `OFFLINE`
  - `gps`: `READY` / `UNAVAILABLE`

---

## 16. Installation
1. Clone or extract the repository to your host machine:
   ```cmd
   cd "C:\Users\ROHIT\Desktop\major project - Copy"
   ```
2. Activate Python environment:
   ```cmd
   conda activate base
   ```
3. Verify dependencies:
   ```cmd
   python scripts\stage_9\preflight_check.py --dependencies
   ```

---

## 17. Startup (One-Click Launcher)
Double-click `run_fall_detection.bat` or run:
```cmd
run_fall_detection.bat
```
The Stage 9 launcher automatically executes:
```
[1] Environment check
[2] Database check
[3] MQTT broker check
[4] Backend startup
[5] Dashboard availability
[6] ESP32 detection
[7] Hardware bridge startup
[8] System health verification
```
Once verified, the Command Center Dashboard automatically opens in your default web browser.

---

## 18. Shutdown
- Press **`Ctrl+C`** in the launcher terminal window.
- The launcher triggers SIGINT, stops background hardware bridges, shuts down the Flask backend cleanly, releases serial ports, and prevents orphaned Python processes.

---

## 19. Testing
Execute the complete project test suite:
```cmd
python scratch/run_full_regression.py
```
Expected output:
```
Total Test Cases Discovered: 264+
Result: 0 FAILURES, 0 ERRORS, 0 SKIPPED
```

To run Stage 9 production hardening tests specifically:
```cmd
python -m unittest fall_detection_dataset/scripts/stage_6/test_stage_9_production_hardening.py
```

---

## 20. Safety Considerations
- **NO PHYSICAL HUMAN FALLS**: Presenters and operators must never drop to the floor to trigger alarms.
- **Certified Drop Tests Only**: Hardware sensor tests must be performed onto landing mats or cushioned surfaces.
- **Safe Demonstration Mode**: Use the built-in simulation generator to verify alarms without physical risks.

---

## 21. Known Limitations & Operational Boundaries
- **Academic Research Prototype**: Not certified as a medical device (no FDA 510(k) or CE-MDR clearance).
- **Indoor GPS Fix**: GPS accuracy indoors depends on satellite visibility; indoors may report `null` coordinates until an outdoor fix is achieved.
- **Cellular / GSM / SMS**: External telephony and cellular networks are disabled by design; all telemetry resides on the local network.

---

## 22. Demonstration Procedure
Follow the step-by-step 11-stage script in [`docs/DEMONSTRATION_GUIDE.md`](docs/DEMONSTRATION_GUIDE.md):
1. Launch system via `run_fall_detection.bat`.
2. Inspect terminal preflight checks.
3. Open dashboard at `http://127.0.0.1:5000/dashboard`.
4. Trigger safe demo event: `python scripts\stage_9\run_safe_demo.py`.
5. Observe active alarm, sound, and map pin on dashboard.
6. Acknowledge the incident.
7. Resolve the incident.
8. Inspect incident history table.
9. Verify system health panel.
10. Shut down cleanly via `Ctrl+C`.
