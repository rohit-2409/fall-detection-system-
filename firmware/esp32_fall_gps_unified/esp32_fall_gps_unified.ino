/*
  Stage 6L: Unified ESP32 Firmware for Wearable Fall Detection System
  Standalone Fall Detection Project (Active Working Copy)

  Integrates:
    1. InvenSense MPU6050 6-Axis IMU (I2C: SDA=GPIO 21, SCL=GPIO 22)
       - High-speed I2C fast mode (400 kHz)
       - Nominal 100 Hz sampling (10 ms interval, non-blocking)
       - Calibrated output in standard units: Accelerometer in g, Gyroscope in deg/s
       - High-throughput USB Serial stream in structured JSON Lines format
    2. u-blox NEO-6M GPS (Hardware UART2: RX2=GPIO 16, TX2=GPIO 17 @ 9600 baud)
       - Continuous non-blocking NMEA parsing via TinyGPSPlus
       - Tracks fix validity, coordinates, satellites, HDOP, and fix age
    3. Wi-Fi & MQTT Telemetry (PubSubClient)
       - Fully configurable broker host, port, Wi-Fi credentials via defines
       - Publishes live GPS telemetry to topic: fall_detection/stage6/gps (every 2000 ms)
       - Non-blocking Wi-Fi and MQTT reconnection without interrupting 100 Hz IMU acquisition
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <HardwareSerial.h>
#include <TinyGPSPlus.h>

// ==============================================================================
// 1. HARDWARE PIN & BUS CONFIGURATION
// ==============================================================================

// MPU6050 I2C Pins
#define MPU_SDA_PIN         21
#define MPU_SCL_PIN         22
#define MPU_I2C_ADDR        0x68
#define MPU_I2C_CLOCK_HZ    400000

// NEO-6M GPS UART2 Pins
#ifndef GPS_RX_PIN
#define GPS_RX_PIN          16  // ESP32 GPIO 16 connects to NEO-6M TX
#endif

#ifndef GPS_TX_PIN
#define GPS_TX_PIN          17  // ESP32 GPIO 17 connects to NEO-6M RX
#endif

#ifndef GPS_BAUD
#define GPS_BAUD            9600
#endif

// USB Serial Monitor (460800 baud provides 72% headroom for 100 Hz JSON telemetry)
#ifndef SERIAL_BAUD
#define SERIAL_BAUD         460800
#endif

// ==============================================================================
// 2. NETWORK & MQTT CONFIGURATION (CONFIGURABLE PLACEHOLDERS)
// ==============================================================================

#ifndef WIFI_SSID
#define WIFI_SSID           "OnePlus 11R 5G"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD       "rohit@02"
#endif

#ifndef MQTT_BROKER_HOST
#define MQTT_BROKER_HOST    "10.163.182.136"  // Hotspot broker IP
#endif

#ifndef MQTT_BROKER_PORT
#define MQTT_BROKER_PORT    1883
#endif

#ifndef MQTT_TOPIC_GPS
#define MQTT_TOPIC_GPS      "fall_detection/stage6/gps"
#endif

#ifndef MQTT_DEVICE_ID
#define MQTT_DEVICE_ID      "ESP32-STAGE6L-UNIFIED"
#endif

// ==============================================================================
// 3. TIMING CONSTANTS (NON-BLOCKING SCHEDULING)
// ==============================================================================

const unsigned long IMU_INTERVAL_US       = 10000;   // 10,000 us = 100.0 Hz
const unsigned long GPS_SERIAL_INTERVAL_MS = 1000;   // 1 Hz serial diagnostic
const unsigned long MQTT_PUB_INTERVAL_MS   = 2000;   // 0.5 Hz MQTT telemetry
const unsigned long RECONNECT_INTERVAL_MS = 5000;   // Network retry interval
const unsigned long MAX_FIX_AGE_MS        = 2500;   // GPS fix staleness cutoff

// ==============================================================================
// 4. GLOBAL OBJECTS & STATE
// ==============================================================================

TinyGPSPlus gps;
HardwareSerial SerialGPS(2); // ESP32 UART2

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// Timing variables
unsigned long lastImuMicros = 0;
unsigned long lastGpsSerialMillis = 0;
unsigned long lastMqttPubMillis = 0;
unsigned long lastWifiAttemptMillis = 0;
unsigned long lastMqttAttemptMillis = 0;

int publishSequenceCounter = 0;
bool mpuInitialized = false;

// Calibration scale factors for MPU6050
// Full-scale: Accel +/- 16g (2048.0 LSB/g), Gyro +/- 2000 deg/s (16.4 LSB/(deg/s))
// Matches Stage 4D frozen model training distribution (ADXL345 +/- 16g, ITG3200 +/- 2000 deg/s)
const float ACCEL_SCALE = 2048.0f;
const float GYRO_SCALE  = 16.4f;

// ==============================================================================
// 5. MPU6050 DRIVER (DIRECT I2C BURST READ - ZERO EXTERNAL LIB DEPENDENCY)
// ==============================================================================

bool initMPU6050() {
  Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN, MPU_I2C_CLOCK_HZ);
  delay(50);

  // Wake up MPU6050 by writing 0x00 to Power Management 1 (0x6B)
  Wire.beginTransmission(MPU_I2C_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    return false;
  }
  delay(10);

  // Configure Accelerometer to +/- 16g (Register 0x1C = 0x18, AFS_SEL = 3)
  Wire.beginTransmission(MPU_I2C_ADDR);
  Wire.write(0x1C);
  Wire.write(0x18);
  Wire.endTransmission();

  // Configure Gyroscope to +/- 2000 deg/s (Register 0x1B = 0x18, FS_SEL = 3)
  Wire.beginTransmission(MPU_I2C_ADDR);
  Wire.write(0x1B);
  Wire.write(0x18);
  Wire.endTransmission();

  // Configure DLPF (Digital Low Pass Filter) ~44 Hz (Register 0x1A = 0x03)
  Wire.beginTransmission(MPU_I2C_ADDR);
  Wire.write(0x1A);
  Wire.write(0x03);
  Wire.endTransmission();

  return true;
}

bool readMPU6050(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_I2C_ADDR);
  Wire.write(0x3B); // Starting register for accelerometer data
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  // Request 14 bytes: 6 for accel, 2 for temp, 6 for gyro
  uint8_t bytesReceived = Wire.requestFrom((uint8_t)MPU_I2C_ADDR, (uint8_t)14, (uint8_t)true);
  if (bytesReceived < 14) {
    return false;
  }

  int16_t rawAx = (Wire.read() << 8) | Wire.read();
  int16_t rawAy = (Wire.read() << 8) | Wire.read();
  int16_t rawAz = (Wire.read() << 8) | Wire.read();
  int16_t rawTemp = (Wire.read() << 8) | Wire.read(); // Read and discard temp
  (void)rawTemp;
  int16_t rawGx = (Wire.read() << 8) | Wire.read();
  int16_t rawGy = (Wire.read() << 8) | Wire.read();
  int16_t rawGz = (Wire.read() << 8) | Wire.read();

  // Convert raw 16-bit integer readings to physical units
  ax = (float)rawAx / ACCEL_SCALE;
  ay = (float)rawAy / ACCEL_SCALE;
  az = (float)rawAz / ACCEL_SCALE;
  gx = (float)rawGx / GYRO_SCALE;
  gy = (float)rawGy / GYRO_SCALE;
  gz = (float)rawGz / GYRO_SCALE;

  return true;
}

// ==============================================================================
// 6. NETWORK & MQTT MANAGEMENT (NON-BLOCKING)
// ==============================================================================

void maintainNetwork() {
  unsigned long now = millis();

  // Check Wi-Fi
  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastWifiAttemptMillis >= RECONNECT_INTERVAL_MS) {
      lastWifiAttemptMillis = now;
      if (String(WIFI_SSID) != "REPLACE_WITH_WIFI_SSID") {
        WiFi.disconnect();
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      }
    }
    return;
  }

  // Check MQTT if Wi-Fi connected
  if (!mqttClient.connected()) {
    if (now - lastMqttAttemptMillis >= RECONNECT_INTERVAL_MS) {
      lastMqttAttemptMillis = now;
      mqttClient.connect(MQTT_DEVICE_ID);
    }
  } else {
    mqttClient.loop();
  }
}

void publishGpsTelemetry() {
  if (!mqttClient.connected()) return;

  bool fixValid = gps.location.isValid() && (gps.location.age() <= MAX_FIX_AGE_MS);
  double lat = fixValid ? gps.location.lat() : 0.0;
  double lng = fixValid ? gps.location.lng() : 0.0;
  int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
  float hdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.9f;
  unsigned long age = gps.location.isValid() ? gps.location.age() : 9999;

  publishSequenceCounter++;

  char buffer[256];
  snprintf(buffer, sizeof(buffer),
    "{\"device_id\":\"%s\",\"stage\":\"6L\",\"status\":\"GPS\",\"fix_valid\":%s,"
    "\"latitude\":%.6f,\"longitude\":%.6f,\"satellites\":%d,\"hdop\":%.2f,"
    "\"gps_age_ms\":%lu,\"seq\":%d}",
    MQTT_DEVICE_ID,
    fixValid ? "true" : "false",
    lat, lng, sats, hdop, age, publishSequenceCounter
  );

  mqttClient.publish(MQTT_TOPIC_GPS, buffer);
}

// ==============================================================================
// 7. SETUP
// ==============================================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 2000) { ; }

  // 1. Initialize MPU6050
  mpuInitialized = initMPU6050();

  // 2. Initialize Hardware UART2 for GPS
  SerialGPS.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  // 3. Initialize Wi-Fi & MQTT
  WiFi.mode(WIFI_STA);
  if (String(WIFI_SSID) != "REPLACE_WITH_WIFI_SSID") {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
  mqttClient.setServer(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  mqttClient.setBufferSize(384);

  // Send system boot JSON line
  Serial.print("{\"type\":\"system\",\"status\":\"READY\",\"mpu_ok\":");
  Serial.print(mpuInitialized ? "true" : "false");
  Serial.print(",\"baud\":");
  Serial.print(SERIAL_BAUD);
  Serial.println("}");
}

// ==============================================================================
// 8. MAIN NON-BLOCKING EXECUTION LOOP
// ==============================================================================

void loop() {
  // A. Process incoming GPS serial bytes continuously
  while (SerialGPS.available() > 0) {
    gps.encode(SerialGPS.read());
  }

  unsigned long currentMicros = micros();
  unsigned long currentMillis = millis();

  // B. 100 Hz MPU6050 IMU Sample Task (every 10,000 microseconds)
  if (currentMicros - lastImuMicros >= IMU_INTERVAL_US) {
    lastImuMicros = currentMicros;
    float ax = 0.0f, ay = 0.0f, az = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;

    if (mpuInitialized && readMPU6050(ax, ay, az, gx, gy, gz)) {
      // Deterministic JSON Lines stream for PC inference bridge
      Serial.print("{\"type\":\"imu\",\"timestamp_ms\":");
      Serial.print(currentMillis);
      Serial.print(",\"ax_g\":");
      Serial.print(ax, 4);
      Serial.print(",\"ay_g\":");
      Serial.print(ay, 4);
      Serial.print(",\"az_g\":");
      Serial.print(az, 4);
      Serial.print(",\"gx_dps\":");
      Serial.print(gx, 4);
      Serial.print(",\"gy_dps\":");
      Serial.print(gy, 4);
      Serial.print(",\"gz_dps\":");
      Serial.print(gz, 4);
      Serial.println("}");
    }
  }

  // C. Periodic GPS Serial Diagnostic Output (every 1000 ms)
  if (currentMillis - lastGpsSerialMillis >= GPS_SERIAL_INTERVAL_MS) {
    lastGpsSerialMillis = currentMillis;
    bool fixValid = gps.location.isValid() && (gps.location.age() <= MAX_FIX_AGE_MS);

    Serial.print("{\"type\":\"gps\",\"timestamp_ms\":");
    Serial.print(currentMillis);
    Serial.print(",\"valid\":");
    Serial.print(fixValid ? "true" : "false");
    Serial.print(",\"latitude\":");
    Serial.print(fixValid ? gps.location.lat() : 0.0, 6);
    Serial.print(",\"longitude\":");
    Serial.print(fixValid ? gps.location.lng() : 0.0, 6);
    Serial.print(",\"satellites\":");
    Serial.print(gps.satellites.isValid() ? gps.satellites.value() : 0);
    Serial.println("}");
  }

  // D. Periodic Wi-Fi & MQTT Telemetry Task (every 2000 ms)
  if (currentMillis - lastMqttPubMillis >= MQTT_PUB_INTERVAL_MS) {
    lastMqttPubMillis = currentMillis;
    publishGpsTelemetry();
  }

  // E. Non-blocking Network Health & Keepalive
  maintainNetwork();
}
