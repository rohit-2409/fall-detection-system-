# ESP32 + MPU6050 Hardware Setup & Live Ingestion Guide
## Standalone Wearable Fall Detection System

**Version**: Stage 5D Hardened Deployment  
**Project Path**: `C:\Users\ROHIT\Desktop\major project`

---

## 1. Required Hardware Components

1. **Microcontroller**: ESP32 Dev Module (Espressif ESP32-WROOM-32 or compatible 30-pin / 38-pin DevKit).
2. **Inertial Measurement Unit (IMU)**: InvenSense MPU6050 breakout board (triaxial accelerometer + triaxial gyroscope).
3. **Interconnects**: 4 DuPont jumper wires (female-to-female or male-to-female).
4. **Host Connection**: Micro-USB or USB-C data cable connecting ESP32 to PC USB port.
5. **Sensor Mounting**: Elastic waist belt or adhesive clip for secure, low-chatter placement at lumbar/waist position.

---

## 2. Hardware Wiring Diagram

The MPU6050 communicates with the ESP32 via standard hardware I2C bus at **400 kHz Fast Mode**.

| MPU6050 Pin | ESP32 GPIO Pin | Function / Description |
| :--- | :--- | :--- |
| **VCC** | **3.3V** | Regulated logic power supply (Do NOT connect to 5V unless board has onboard LDO) |
| **GND** | **GND** | Common ground reference |
| **SCL** | **GPIO 22** | Hardware I2C Clock Line (ESP32 default SCL) |
| **SDA** | **GPIO 21** | Hardware I2C Data Line (ESP32 default SDA) |
| **AD0** | **GND** | Sets I2C peripheral address to **`0x68`** |
| **INT** | *Unconnected* | Not required (Synchronous 100 Hz hardware timer handles polling) |

---

## 3. Sensor Orientation & Baseline Verification

- **Placement**: Worn at the waist / belt line (approx. center of mass near L3–L5 lumbar vertebrae).
- **Coordinate Convention**:
  - **Z-Axis**: Points upwards (parallel to gravity during upright standing).
  - **X-Axis**: Lateral (pointing sideways / left-to-right).
  - **Y-Axis**: Anterior-posterior (pointing forward/backward).
- **Verified Stationary Rest Baseline**:
  When resting completely stationary on a flat horizontal surface:
  - $\text{AccX} \approx 0.010 \pm 0.005\,g$
  - $\text{AccY} \approx -0.020 \pm 0.005\,g$
  - $\text{AccZ} \approx 0.981 \pm 0.005\,g$ (capturing $1g$ static Earth gravity)
  - $\text{GyrX} \approx 0.20 \pm 0.05\,^\circ/\text{s}$
  - $\text{GyrY} \approx -0.10 \pm 0.05\,^\circ/\text{s}$
  - $\text{GyrZ} \approx 0.05 \pm 0.05\,^\circ/\text{s}$

---

## 4. Telemetry Stream Specifications

The ESP32 firmware streams comma-separated ASCII records over USB UART at **100.0 Hz** ($\Delta t = 10.0\text{ ms}$ interval):

```text
timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ
```

### Format Details:
- **`timestamp`**: Monotonically increasing relative time in milliseconds (e.g. `10, 20, 30...`).
- **`AccX, AccY, AccZ`**: Triaxial acceleration scaled in standard gravity units ($g$, where $1g = 9.80665\text{ m/s}^2$).
- **`GyrX, GyrY, GyrZ`**: Triaxial angular rate scaled in degrees per second ($^\circ/\text{s}$).

### Example Sample Stream:
```text
10,0.010254,-0.020142,0.981445,0.206107,-0.106870,0.053435
20,0.010010,-0.019897,0.980957,0.198473,-0.099237,0.045802
30,0.010498,-0.020386,0.981934,0.213740,-0.114504,0.061069
```

---

## 5. Serial Connection Settings

- **Port**: `COM22` (default on Windows; configurable via `--port` argument).
- **Baud Rate**: **`115200` bps**.
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1 (8-N-1)
- **Flow Control**: None

---

## 6. How to Start Live Ingestion

### Option A: Physical ESP32 Hardware
1. Connect ESP32 to PC USB port.
2. Verify assigned COM port in Windows Device Manager (e.g., `COM22`).
3. Run the live diagnostic runner:
   ```bash
   python src/fall_confirmation/live_diagnostic_runner.py --port COM22 --baudrate 115200
   ```
4. The script buffers 50 samples, extracts 100 features every 25 samples (0.25s), runs model inference, and prints real-time probability and temporal confirmation state.

### Option B: Software Mock / Replay Mode (No Hardware Needed)
If physical hardware is disconnected, test the live ingestion pipeline using simulated sensor data:
```bash
python src/fall_confirmation/live_diagnostic_runner.py --mock-mode
```

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `SerialException: could not open port 'COM22'` | Port is locked or incorrect | Close any serial monitor (e.g. Arduino IDE Serial Monitor, PuTTY). Check Device Manager for correct COM port number. |
| `[WARNING] Malformed sample line` | Incomplete baud sync or header | The ingestion parser automatically ignores headers and malformed lines. If persistent, verify baud rate is set to `115200`. |
| Gyro values spike excessively while resting | High noise or floating GND | Ensure GND wire is firmly connected. Keep sensor away from high-frequency electromagnetic interference. |
| Fall events not triggering during test | Impact too gentle or duration $< 0.75$s | Stage 4E Rule N3 requires 3 consecutive windows ($0.75$s) with $P(\text{fall}) \ge 0.60$. Perform deliberate, supervised fall motions onto a gym safety mat. |
