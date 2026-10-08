# Fall Detection System — Hardware Setup & Configuration Guide
## Stage 9 Production & Demonstration Readiness

**Active Workspace**: `C:\Users\ROHIT\Desktop\major project - Copy`  
**Target Hardware**: ESP32 Dev Module (WROOM-32) + InvenSense MPU6050 6-DOF IMU + u-blox NEO-6M GPS Module

---

## 1. Verified Hardware Components & Specifications

| Component | Verified Specification | Verification Source |
| :--- | :--- | :--- |
| **Microcontroller** | ESP32-WROOM-32 (Dual Core 240 MHz, 520 KB SRAM) | Physical Hardware Validation & Firmware |
| **IMU Sensor** | InvenSense MPU6050 (Triaxial Accel + Triaxial Gyro) | `firmware/esp32_sensor_node/` |
| **Accelerometer Range** | **$\pm 16g$** (Full scale range register configured) | Stage 7B Phase 3 Hardware Validation |
| **Gyroscope Range** | **$\pm 2000^\circ/\text{s}$** (Full scale range register configured) | Stage 7B Phase 3 Hardware Validation |
| **IMU Sampling Rate** | **100.0 Hz** ($\Delta t = 10.0\text{ ms}$) via hardware timer | Firmware timer interrupt / loop |
| **GPS Module** | u-blox NEO-6M GPS receiver | Stage 6K / 7B GPS validation |
| **GPS UART Baud** | **9600 bps** (Hardware UART2) | ESP32 HardwareSerial config |
| **Host PC Serial Bridge** | **460800 bps** (High-throughput USB-UART bridge) | Stage 6L / 7B Phase 3 validated |
| **Default Host Port** | `COM22` (Windows device path) | Production default |

---

## 2. Verified Pinout & Wiring Connections

### 2.1 MPU6050 to ESP32 (Hardware I2C)

| MPU6050 Pin | ESP32 GPIO | Description / Signal | Verification Status |
| :--- | :--- | :--- | :--- |
| **VCC** | **3.3V** | Regulated DC logic power | VERIFIED |
| **GND** | **GND** | Common ground plane | VERIFIED |
| **SDA** | **GPIO 21** | I2C Data Line (ESP32 default SDA) | VERIFIED |
| **SCL** | **GPIO 22** | I2C Clock Line (ESP32 default SCL) | VERIFIED |
| **AD0** | **GND** | Sets I2C peripheral address to `0x68` | VERIFIED |
| **INT** | *Not Connected* | Interrupt line unused (100 Hz timer polling) | VERIFIED |

### 2.2 NEO-6M GPS to ESP32 (Hardware UART2)

| NEO-6M GPS Pin | ESP32 GPIO | Description / Signal | Verification Status |
| :--- | :--- | :--- | :--- |
| **VCC** | **3.3V / 5V** | DC power supply | VERIFIED |
| **GND** | **GND** | Common ground plane | VERIFIED |
| **TX** | **GPIO 16** (ESP32 RX2) | GPS Serial Data Out into ESP32 UART2 RX | VERIFIED |
| **RX** | **GPIO 17** (ESP32 TX2) | GPS Serial Data In from ESP32 UART2 TX | VERIFIED |
| **PPS** | *Not Connected* | Pulse-per-second precision sync | NOT VERIFIED |

*Note: Any pin or auxiliary line not explicitly listed above is NOT VERIFIED in the project repository.*

---

## 3. High-Speed Serial Protocol & Ingestion Format

The ESP32 firmware streams JSON or structured telemetry frames at **460800 baud** over USB serial.

### Sensor Data Packet Format:
```json
{
  "accel": [ax, ay, az],
  "gyro": [gx, gy, gz],
  "timestamp": 1234567,
  "gps": {
    "lat": 28.6139,
    "lon": 77.2090,
    "fix": 1,
    "satellites": 6
  }
}
```
- **Units**:
  - `ax, ay, az`: Acceleration in $g$ ($1g = 9.80665\text{ m/s}^2$), scaled across full $\pm 16g$ range.
  - `gx, gy, gz`: Angular rate in degrees per second ($^\circ/\text{s}$), scaled across full $\pm 2000^\circ/\text{s}$ range.
  - `lat, lon`: Decimal degrees WGS84 coordinates.

---

## 4. Hardware Verification & Calibration Check

### 4.1 Stationary Rest Baseline
When the sensor sits undisturbed on a level table:
- $|Acc| = \sqrt{a_x^2 + a_y^2 + a_z^2} \approx 0.98 \text{ to } 1.02\,g$ (Earth gravity vector)
- $|Gyr| = \sqrt{g_x^2 + g_y^2 + g_z^2} \le 1.5^\circ/\text{s}$ (Sensor thermal noise floor)

### 4.2 Impact Headroom Verification
The transition from $\pm 2g$ to $\pm 16g$ allows true human impact peaks ($3g$ to $12g$) to be measured accurately without sensor saturation or clipping, ensuring high-fidelity extraction of impact-stage features (e.g., peak acceleration, acceleration jerk, dynamic energy).

---

## 5. Serial Port Troubleshooting

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| Port COM22 not found | USB unplugged or driver missing | Reconnect USB cable. Install Silicon Labs CP210x or CH340 USB-to-UART bridge drivers. |
| Permission Denied on COM22 | Port held by another process | Close Serial Monitor, PuTTY, or zombie Python bridge processes. |
| Garbage characters in stream | Baud rate mismatch | Ensure host and bridge baud are set to exactly `460800`. |
| GPS Fix is `0` / Coordinates `None` | Indoor satellite occlusion | Move antenna near window or outdoors. Allow 45-60 seconds for GPS cold-start TTFF. |
