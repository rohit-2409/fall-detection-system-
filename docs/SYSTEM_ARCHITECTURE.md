# Fall Detection System — End-to-End System Architecture
## Stage 9 Production Architecture & Dataflow Specification

**Workspace**: `C:\Users\ROHIT\Desktop\major project - Copy`  
**Classification**: Engineering System Specification & Data Pipeline Design

---

## 1. System Overview Architecture Diagram

```
+-----------------------------------------------------------------------------------------+
|                                    PHYSICAL SENSORS                                     |
+-----------------------------------------------------------------------------------------+
|  [ InvenSense MPU6050 ]                             [ u-blox NEO-6M GPS ]               |
|  - Accel: ±16g range                                - NMEA Sentences                    |
|  - Gyro:  ±2000 dps range                           - Hardware UART2 (RX16 / TX17)      |
|  - I2C Bus (SDA 21, SCL 22) @ 400 kHz               - Baud: 9600 bps                    |
+---------------------------+-------------------------------------+-----------------------+
                            |                                     |
                            v                                     v
+-----------------------------------------------------------------------------------------+
|                              EDGE SENSOR NODE (ESP32)                                   |
+-----------------------------------------------------------------------------------------+
|  - 100 Hz Hardware Timer Sampling Loop                                                  |
|  - Accelerometer scaling to standard gravity (g)                                        |
|  - Gyroscope scaling to angular rate (deg/s)                                            |
|  - High-Speed USB Serial UART Stream @ 460800 baud                                       |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v (USB Virtual COM Port: COM22 @ 460800 baud)
+-----------------------------------------------------------------------------------------+
|                         HOST HARDWARE BRIDGE (Stage 6L)                                 |
+-----------------------------------------------------------------------------------------+
|  - Serial Ingestion & Line Framing Buffer                                               |
|  - GPS Coordinate Extraction (Latitude, Longitude, Satellites, Fix Quality)             |
|  - Rolling Time Window: 50 Samples (0.50s at 100 Hz), Step: 25 Samples (0.25s)          |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                     FROZEN DETECTION PIPELINE (Stages 4D - 7B)                          |
+-----------------------------------------------------------------------------------------+
|  [ 100-Feature Extraction ]            (FROZEN)                                         |
|    - Statistical, peak, energy, jerk, & frequency domain across 6 axes                 |
|                                                                                         |
|  [ Feature Scaler ]                    (FROZEN: scaler.joblib)                          |
|    - StandardScaler matching verified 100-feature order                                 |
|                                                                                         |
|  [ Machine Learning Inference ]        (FROZEN: stage_4D_random_forest.joblib)          |
|    - Random Forest Binary Classifier -> P(Fall) probability estimate                   |
|                                                                                         |
|  [ Dynamic Physical Gates ]            (FROZEN)                                         |
|    - Dynamic Acceleration Gate: Peak |Acc| >= 2.0g                                      |
|    - Gyroscope Velocity Gate:   Peak |Gyr| >= 180 deg/s                                 |
|                                                                                         |
|  [ Temporal Confirmation ]             (FROZEN: temporal_adapter.py)                    |
|    - N=3 Consecutive Windows Confirmation (Threshold >= 0.60)                           |
|                                                                                         |
|  [ Incident Latch & Anti-Duplicate ]   (FROZEN: Stage 7B Latch)                         |
|    - One-Physical-Fall = One-Logical-Incident Guarantee                                 |
|    - Duplicate Suppression during event & Post-fall Inactivity Window                   |
|    - Upright Posture Recovery & Re-Arming Logic                                         |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v Event Payload JSON (CONFIRMED_FALL)
+-----------------------------------------------------------------------------------------+
|                        MESSAGE BROKER (Mosquitto MQTT)                                  |
+-----------------------------------------------------------------------------------------+
|  - TCP Socket: 127.0.0.1:1883                                                           |
|  - Production Topic: fall_detection/stage6/fall                                         |
|  - Heartbeat & Telemetry Topic: fall_detection/stage6/telemetry                         |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------------+
|                     INGESTION GATEWAY & BACKEND (Stage 6G + 8A)                         |
+-----------------------------------------------------------------------------------------+
|  [ Stage 6G MQTT Gateway ]                                                              |
|    - Background MQTT subscriber thread                                                  |
|    - Validates payload contract & extracts GPS coordinates                              |
|    - Forwards verified events via internal HTTP POST /api/fall-events                    |
|                                                                                         |
|  [ Flask REST API Server ] (Port 5000)                                                  |
|    - Health Probes: GET /api/health, GET /api/system/health                             |
|    - Incident Endpoints: POST /api/fall-events, GET /api/fall-events                    |
|    - Lifecycle Actions: POST /api/incidents/<id>/acknowledge, .../resolve               |
|                                                                                         |
|  [ SQLite Database ] (fall_events.db)                                                   |
|    - Acid-compliant transactional persistent storage                                    |
|    - Records: event_id, detected_at, confidence, status, latitude, longitude, notes    |
+-------------------------------------------+---------------------------------------------+
                                            |
                                            v HTTP / JSON polling & Leaflet GIS
+-----------------------------------------------------------------------------------------+
|                    COMMAND CENTER DASHBOARD (Stage 8B)                                  |
+-----------------------------------------------------------------------------------------+
|  - Static Web UI: http://127.0.0.1:5000/dashboard                                       |
|  - Live Status Banner: Normal Monitoring vs ACTIVE FALL INCIDENT (Alarm Mode)          |
|  - Leaflet / OpenStreetMap GIS Integration: Exact victim pin & GPS coordinates         |
|  - Operator Actions: Single-click Incident Acknowledgment & Incident Resolution        |
|  - Incident Audit History: Filterable, persistent historical incident log              |
|  - System Health Panel: Honest reporting of Backend, Database, MQTT, ESP32, and GPS    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. GPS Metadata Subsystem Dataflow

The GPS telemetry follows an integrated sideband channel to ensure geolocation is bundled with the fall incident:

```
[ NEO-6M GPS Receiver ]
         │
         ▼ (NMEA Sentences over UART2 RX16/TX17 @ 9600 bps)
[ ESP32 Firmware Parser ]
         │
         ▼ (JSON Telemetry packet with "gps": {"lat": ..., "lon": ..., "fix": 1})
[ Stage 6L Bridge Ingestion ]
         │
         ▼ (Packaged into CONFIRMED_FALL MQTT JSON event)
[ MQTT Gateway (Stage 6G) ]
         │
         ▼ (HTTP POST to /api/fall-events payload with latitude and longitude)
[ SQLite Database & REST API ]
         │
         ▼ (GET /api/fall-events active incident polling)
[ Dashboard Leaflet Map Renderer ]
         - Auto-centers map at [latitude, longitude]
         - Drops high-visibility red victim marker with popup metadata
```

---

## 3. Strict Machine Learning & Detection Core Freeze

The following components are cryptographically locked and must **NEVER** be modified, retrained, or altered:

1. **Random Forest Classifier**: `stage_4D_random_forest.joblib`  
   `SHA-256: a499a0dd0107d9b25de01012a2efe8293f1b1e2092d1af51b08237efd3a8584c`
2. **Feature Scaler**: `scaler.joblib`  
   `SHA-256: ea73df7b4756911221b670a69d7195baae65492bfea43cdf80c3d6dfcde346af`
3. **Feature Column Order**: `feature_order.json`  
   `SHA-256: 995941391e4fe95a2899131583a30cf3be895d1c78401ff66bfc61333262d65b`
4. **Live Validator Reference**: `stage_4G_live_validator.py`  
   `SHA-256: 64bfac3347f3bb87dfd7c7967488cf4359f0319c63cd17583c681aace28d2728`
5. **Temporal Confirmation Adapter**: `temporal_adapter.py`  
   `SHA-256: 389ae97f4e6c5bcd6994bf1ed7dbe6bed7414298542e2c90459767073a5b60b8`

Any variance in the above cryptographic hashes halts system startup immediately.
