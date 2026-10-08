/*
  Stage 6E: u-blox NEO-6M GPS + ESP32 Wi-Fi + MQTT Combined Telemetry Test
  Standalone Fall Detection Project (Active Working Copy)

  Hardware Configuration:
    - Microcontroller: ESP32 Dev Module
    - GPS Module: u-blox NEO-6M Breakout Board
    - GPS UART: Hardware UART2 (GPIO 16 RX2, GPIO 17 TX2, 9600 baud)
    - PC MQTT Broker: Mosquitto on 10.163.182.136:1883
    - Wi-Fi: 2.4 GHz Hotspot (OnePlus 11R 5G)
    - MQTT Topic: fall_detection/stage6/gps
    - Device ID: ESP32-STAGE6-GPS

  Functionality:
    - Continuous non-blocking byte-by-byte GPS ingestion via TinyGPSPlus.
    - Concurrent Wi-Fi and MQTT connectivity maintenance without blocking.
    - Periodic telemetry publication to MQTT (~2000 ms interval).
    - Serial monitor diagnostic status reporting (~1000 ms interval).
    - Completely isolated from MPU6050 fall detection ML pipeline and models.
*/

#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <HardwareSerial.h>
#include <TinyGPSPlus.h>

// ==============================================================================
// 1. HARDWARE & NETWORK CONFIGURATION
// ==============================================================================
#ifndef WIFI_SSID
#define WIFI_SSID "OnePlus 11R 5G"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "rohit@02"
#endif

// PC LAN Broker (Mosquitto on OnePlus Hotspot IP)
const char* mqtt_broker = "10.163.182.136";
const int mqtt_port = 1883;

// Topics and Identifiers
const char* mqtt_topic_gps = "fall_detection/stage6/gps";
const char* device_id = "ESP32-STAGE6-GPS";

// GPS Hardware UART2
#define GPS_RX_PIN 16 // ESP32 GPIO 16 connects to NEO-6M TX
#define GPS_TX_PIN 17 // ESP32 GPIO 17 connects to NEO-6M RX
#define GPS_BAUD   9600

#define SERIAL_MONITOR_BAUD 115200

// Timing intervals
const unsigned long PUBLISH_INTERVAL_MS = 2000; // Publish MQTT every 2 seconds
const unsigned long REPORT_INTERVAL_MS  = 1000; // Serial monitor output every 1 second
const unsigned long RECONNECT_INTERVAL_MS = 5000; // Non-blocking retry interval
const unsigned long MAX_FIX_AGE_MS      = 2500; // Staleness threshold for GPS fix

// ==============================================================================
// 2. GLOBAL HARDWARE & PROTOCOL OBJECTS
// ==============================================================================
TinyGPSPlus gps;
HardwareSerial SerialGPS(2);

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ==============================================================================
// 3. STATE TRACKING
// ==============================================================================
enum GpsState {
  STATE_ACQUIRING,
  STATE_FIX_VALID,
  STATE_FIX_LOST
};

GpsState currentGpsState = STATE_ACQUIRING;
bool hasEverHadValidFix = false;
double lastKnownLatitude = 0.0;
double lastKnownLongitude = 0.0;
unsigned long lastValidFixMillis = 0;

unsigned long lastMqttAttemptMillis = 0;
unsigned long lastWifiAttemptMillis = 0;
unsigned long lastPublishMillis = 0;
unsigned long lastReportMillis = 0;

int publishSequenceCounter = 0;
bool lastPublishSuccess = false;

// ==============================================================================
// 4. NON-BLOCKING NETWORK MANAGEMENT
// ==============================================================================
void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  unsigned long now = millis();
  if (now - lastWifiAttemptMillis >= RECONNECT_INTERVAL_MS) {
    lastWifiAttemptMillis = now;
    Serial.println("[Wi-Fi] Attempting connection to SSID...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

void maintainMqtt() {
  if (!WiFi.isConnected() || mqttClient.connected()) {
    return;
  }

  unsigned long now = millis();
  if (now - lastMqttAttemptMillis >= RECONNECT_INTERVAL_MS) {
    lastMqttAttemptMillis = now;
    Serial.print("[MQTT] Connecting to broker ");
    Serial.print(mqtt_broker);
    Serial.print(":");
    Serial.print(mqtt_port);
    Serial.print(" as ");
    Serial.print(device_id);
    Serial.println("...");

    if (mqttClient.connect(device_id)) {
      Serial.println("[MQTT] Connected successfully to Mosquitto broker.");
    } else {
      Serial.print("[MQTT] Connection failed, rc=");
      Serial.println(mqttClient.state());
    }
  }
}

// ==============================================================================
// 5. GPS TELEMETRY PUBLISHER (~2000 ms)
// ==============================================================================
void publishGpsTelemetry() {
  if (!mqttClient.connected()) {
    lastPublishSuccess = false;
    return;
  }

  publishSequenceCounter++;
  bool isFixCurrentlyValid = gps.location.isValid() && (gps.location.age() <= MAX_FIX_AGE_MS);
  char payload[384];

  if (isFixCurrentlyValid) {
    double lat = gps.location.lat();
    double lng = gps.location.lng();
    int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    double hdop = gps.hdop.isValid() ? gps.hdop.hdop() : 99.99;
    unsigned long age = gps.location.age();

    snprintf(payload, sizeof(payload),
             "{\"device_id\":\"%s\",\"stage\":\"6E\",\"status\":\"GPS\",\"fix_valid\":true,\"latitude\":%.6f,\"longitude\":%.6f,\"satellites\":%d,\"hdop\":%.2f,\"gps_age_ms\":%lu,\"seq\":%d}",
             device_id, lat, lng, sats, hdop, age, publishSequenceCounter);
  } else {
    int sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
    snprintf(payload, sizeof(payload),
             "{\"device_id\":\"%s\",\"stage\":\"6E\",\"status\":\"GPS\",\"fix_valid\":false,\"latitude\":0.0,\"longitude\":0.0,\"satellites\":%d,\"hdop\":null,\"gps_age_ms\":null,\"seq\":%d}",
             device_id, sats, publishSequenceCounter);
  }

  lastPublishSuccess = mqttClient.publish(mqtt_topic_gps, payload);
}

// ==============================================================================
// 6. SERIAL MONITOR DIAGNOSTIC REPORTER (~1000 ms)
// ==============================================================================
void printStatusReport() {
  bool isFixCurrentlyValid = gps.location.isValid() && (gps.location.age() <= MAX_FIX_AGE_MS);

  if (isFixCurrentlyValid) {
    currentGpsState = STATE_FIX_VALID;
    hasEverHadValidFix = true;
    lastKnownLatitude = gps.location.lat();
    lastKnownLongitude = gps.location.lng();
    lastValidFixMillis = millis();
  } else if (hasEverHadValidFix) {
    currentGpsState = STATE_FIX_LOST;
  } else {
    currentGpsState = STATE_ACQUIRING;
  }

  Serial.println("==================================================");
  Serial.println("STAGE 6E — GPS + MQTT");
  Serial.println("==================================================");

  // Network Status
  Serial.print("Wi-Fi:        ");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("CONNECTED");
    Serial.print("ESP32 IP:     ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("DISCONNECTED / RECONNECTING");
    Serial.println("ESP32 IP:     0.0.0.0");
  }

  Serial.print("MQTT:         ");
  if (mqttClient.connected()) {
    Serial.println("CONNECTED");
  } else {
    Serial.print("DISCONNECTED (rc=");
    Serial.print(mqttClient.state());
    Serial.println(")");
  }
  Serial.print("Broker:       ");
  Serial.print(mqtt_broker);
  Serial.print(":");
  Serial.println(mqtt_port);
  Serial.println();

  // GPS Status
  Serial.print("GPS Status:   ");
  switch (currentGpsState) {
    case STATE_FIX_VALID:
      Serial.println("FIX_VALID");
      break;
    case STATE_FIX_LOST:
      Serial.println("FIX_LOST");
      break;
    case STATE_ACQUIRING:
    default:
      Serial.println("ACQUIRING");
      break;
  }

  Serial.print("Satellites:   ");
  if (gps.satellites.isValid()) {
    Serial.println(gps.satellites.value());
  } else {
    Serial.println("0");
  }

  Serial.print("Latitude:     ");
  if (currentGpsState == STATE_FIX_VALID) {
    Serial.println(gps.location.lat(), 6);
  } else if (currentGpsState == STATE_FIX_LOST) {
    Serial.print(lastKnownLatitude, 6);
    Serial.println(" (STALE)");
  } else {
    Serial.println("0.000000");
  }

  Serial.print("Longitude:    ");
  if (currentGpsState == STATE_FIX_VALID) {
    Serial.println(gps.location.lng(), 6);
  } else if (currentGpsState == STATE_FIX_LOST) {
    Serial.print(lastKnownLongitude, 6);
    Serial.println(" (STALE)");
  } else {
    Serial.println("0.000000");
  }

  Serial.print("HDOP:         ");
  if (gps.hdop.isValid()) {
    Serial.println(gps.hdop.hdop());
  } else {
    Serial.println("N/A");
  }

  Serial.print("GPS Age:      ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.age());
    Serial.println(" ms");
  } else {
    Serial.println("N/A");
  }
  Serial.println();

  // MQTT Publication Status
  Serial.println("MQTT Topic:");
  Serial.println(mqtt_topic_gps);
  Serial.print("MQTT Publish: ");
  if (publishSequenceCounter == 0) {
    Serial.println("PENDING FIRST TICK");
  } else if (lastPublishSuccess) {
    Serial.println("SUCCESS");
  } else {
    Serial.println("FAILED / NOT CONNECTED");
  }
  Serial.print("Sequence:     ");
  Serial.println(publishSequenceCounter);

  // Diagnostic Counters
  Serial.print("Diagnostics:  Chars: ");
  Serial.print(gps.charsProcessed());
  Serial.print(" | Sentences: ");
  Serial.print(gps.sentencesWithFix());
  Serial.print(" | Passed: ");
  Serial.print(gps.passedChecksum());
  Serial.print(" | Failed: ");
  Serial.println(gps.failedChecksum());
  Serial.println("--------------------------------------------------");
  Serial.println();
}

// ==============================================================================
// 7. SETUP
// ==============================================================================
void setup() {
  Serial.begin(SERIAL_MONITOR_BAUD);
  while (!Serial && millis() < 2500) { ; }

  Serial.println();
  Serial.println("==================================================");
  Serial.println("STAGE 6E: ESP32 + NEO-6M GPS + Wi-Fi + MQTT");
  Serial.println("==================================================");
  Serial.print("[INIT] Serial Monitor:  ");
  Serial.print(SERIAL_MONITOR_BAUD);
  Serial.println(" baud");

  // Initialize Hardware UART2 for GPS
  SerialGPS.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.print("[INIT] GPS UART2:       GPIO ");
  Serial.print(GPS_RX_PIN);
  Serial.print(" RX, GPIO ");
  Serial.print(GPS_TX_PIN);
  Serial.print(" TX @ ");
  Serial.print(GPS_BAUD);
  Serial.println(" baud");

  // Configure MQTT Client
  mqttClient.setServer(mqtt_broker, mqtt_port);
  mqttClient.setBufferSize(512); // Ensure ample capacity for JSON payload

  // Initial Wi-Fi startup (bounded connection attempt)
  Serial.println("[INIT] Connecting to Wi-Fi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - wifiStart < 8000)) {
    // Keep servicing GPS even during initial Wi-Fi connection
    while (SerialGPS.available() > 0) {
      gps.encode(SerialGPS.read());
    }
    delay(100);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[Wi-Fi] Connected! IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[Wi-Fi] Initial timeout - background reconnection will continue.");
  }

  // Initial MQTT attempt
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("[MQTT] Attempting initial connection to broker...");
    if (mqttClient.connect(device_id)) {
      Serial.println("[MQTT] Connected to broker.");
    } else {
      Serial.print("[MQTT] Initial connection deferred, rc=");
      Serial.println(mqttClient.state());
    }
  }

  Serial.println("[READY] Running non-blocking GPS + Wi-Fi + MQTT loop...");
  Serial.println();
}

// ==============================================================================
// 8. MAIN LOOP (Continuous & Non-Blocking)
// ==============================================================================
void loop() {
  // 1. Ingest all available GPS bytes continuously
  while (SerialGPS.available() > 0) {
    char c = SerialGPS.read();
    gps.encode(c);
  }

  // 2. Non-blocking Wi-Fi and MQTT connectivity maintenance
  maintainWifi();
  maintainMqtt();

  // 3. Process MQTT client loop for keep-alives and incoming packets
  if (mqttClient.connected()) {
    mqttClient.loop();
  }

  unsigned long now = millis();

  // 4. Periodic MQTT Telemetry Publish (~2000 ms)
  if (now - lastPublishMillis >= PUBLISH_INTERVAL_MS) {
    lastPublishMillis = now;
    publishGpsTelemetry();
  }

  // 5. Periodic Serial Monitor Diagnostic Report (~1000 ms)
  if (now - lastReportMillis >= REPORT_INTERVAL_MS) {
    lastReportMillis = now;
    printStatusReport();
  }
}
