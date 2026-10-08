# Fall Detection System — Final Demonstration & Presentation Guide
## Step-by-Step Operator & Examiner Demonstration Workflow

**System**: Standalone Wearable Fall Detection System  
**Stage**: Stage 9 Production Hardened & Demonstration Ready  
**Audience**: Final-Year Project Presentation Panel, Evaluators, Engineering Demonstrators

---

## Safety Notice & Ethical Compliance

> **CRITICAL DEMONSTRATION SAFETY RULE**:
> Evaluators and presenters must **NEVER** perform an uncontrolled physical fall onto the floor during a demonstration.
> Physical impact tests must only be conducted onto certified gymnasium landing mats or using sensor drop packages. For live presentations, use the **Safe Demo Mode** (`scripts/stage_9/run_safe_demo.py`), which generates verifiable simulated fall events without endangering human health.

---

## 11-Step Demonstration Sequence

### STEP 1: One-Click System Startup
In the project root directory, launch the complete production system with:
```cmd
run_fall_detection.bat
```
or run with python directly:
```cmd
python scripts\stage_9\launcher.py
```
*(Optionally add `--start-bridge` if physical ESP32 on COM22 is plugged in, or `--mock-bridge` to run full background simulated inference).*

---

### STEP 2: Verify Subsystem Readiness Banner
Observe the startup preflight diagnostic banner in the terminal:
```
==================================================
FALL DETECTION SYSTEM
Production Startup
==================================================

[1] Environment check       -> [OK] Python runtime & frozen ML hashes verified
[2] Database check          -> [OK] SQLite integrity ok, table ready
[3] MQTT broker check       -> [OK] Broker online at 127.0.0.1:1883
[4] Backend startup         -> [OK] Unified backend active on port 5000
[5] Dashboard availability  -> [OK] Dashboard ready at /dashboard
[6] ESP32 detection         -> [OK] COM22 detected (or Degraded Mode if absent)
[7] Hardware bridge startup -> Status displayed
[8] System health check     -> [SYSTEM_READY] or [SYSTEM_DEGRADED] (honest status)
```

---

### STEP 3: Open the Command Center Dashboard
If the browser did not automatically open, navigate to:
```
http://127.0.0.1:5000/dashboard
```
Explain the layout to examiners:
1. **Header**: Live system health badges (Backend, Database, MQTT, ESP32, GPS).
2. **Main Status Panel**: Green "NORMAL MONITORING" banner with resting heart indicator.
3. **Emergency Geolocation Map**: Leaflet / OpenStreetMap viewport centered on current coordinates.
4. **Action Center**: Interactive buttons for incident acknowledgment and resolution.
5. **Historical Incident Log**: Real-time audit table tracking all past events.

---

### STEP 4: Demonstrate Normal Monitoring State
Show that while no fall is occurring:
- Status banner displays: `ALL SYSTEMS NORMAL - MONITORING ACTIVE`.
- No alarms sound or flash.
- Incident history table is populated with past recorded events.
- System health shows real-time subcomponent telemetry.

---

### STEP 5: Trigger a Controlled Demonstration Event
Open a secondary terminal window and execute the safe simulation script:
```cmd
python scripts\stage_9\run_safe_demo.py
```
*(To customize demo coordinates, use `--lat 28.6139 --lon 77.2090 --confidence 0.94`)*.

Notice the terminal output:
```
==================================================
STAGE 9 SAFE DEMO EVENT DISPATCHER
==================================================
[SIMULATION NOTICE] Dispatched event is explicitly marked:
  - Event ID Prefix: DEMO-FALL-...
  - Source Tag:      SIMULATED_DEMO_STAGE_9
[OK] Published to MQTT: fall_detection/stage6/fall
```

---

### STEP 6: Verify Active Fall Incident Alarm on Dashboard
Instantly on the dashboard (within 1 second):
1. **Status Banner Changes**: Shifts to high-visibility pulsing red banner:
   `ACTIVE FALL DETECTED - IMMEDIATE ATTENTION REQUIRED`
2. **Metadata Card Appears**:
   - Incident ID (e.g. `DEMO-FALL-20261007T...`)
   - ML Model Confidence (e.g. `94.0%`)
   - Exact Timestamp (UTC / Local)
   - Coordinates (`28.613900° N, 77.209000° E`)
3. **Map Marker Drops**: Map automatically centers on coordinates and drops a red emergency marker with an interactive popup.

---

### STEP 7: Acknowledge the Incident
Click the **"Acknowledge Incident"** button on the dashboard:
- The system prompts for operator initials/notes (or auto-acknowledges).
- The banner shifts to amber/yellow: `INCIDENT ACKNOWLEDGED - DISPATCH IN PROGRESS`.
- Incident state updates to `ACKNOWLEDGED` in the database.

---

### STEP 8: Resolve the Incident
Click the **"Resolve Incident"** button on the dashboard:
- System prompts for resolution notes (e.g., `"Demonstration completed safely"`).
- The alarm clears immediately.
- The banner returns to green: `ALL SYSTEMS NORMAL - MONITORING ACTIVE`.
- Incident state in the database updates to `RESOLVED`.

---

### STEP 9: Inspect Incident Audit History
Scroll down to the **Incident History Table**:
- Point out the newly recorded incident.
- Show that status reflects `RESOLVED`.
- Point out that all metadata (timestamps, GPS, confidence, resolution notes) is permanently preserved in SQLite.

---

### STEP 10: Show System Health Diagnostics
Point out the System Health indicators on the top panel:
- Backend: `READY`
- Database: `READY`
- MQTT: `CONNECTED`
- ESP32: Reports `READY` if streaming or `STALE` if offline (never faked).
- Point to `http://127.0.0.1:5000/api/system/health` JSON payload for automated health monitoring.

---

### STEP 11: Stop the System Safely
Return to the primary terminal running `run_fall_detection.bat`:
- Press **`Ctrl+C`**.
- Notice graceful shutdown logs:
  ```
  [SHUTDOWN] Terminating system processes cleanly...
    Stopping Hardware Bridge...
    Stopping Unified Backend...
  [SHUTDOWN] System stopped cleanly. Goodbye.
  ```
- Verify that SQLite database closed cleanly and no orphan Python processes remain running.
