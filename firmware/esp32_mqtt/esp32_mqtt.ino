/*
  Stage 6C: ESP32 + Wi-Fi + MQTT Minimal Connectivity Baseline
  Standalone Fall Detection Project (Active Working Copy)
  
  Target Board: ESP32 Dev Module (Espressif ESP32)
  Architecture: ESP32 -> Wi-Fi -> MQTT Broker (Mosquitto on PC LAN: 10.163.182.136:1883)
  
  Description:
    1. Connects ESP32 to local 2.4 GHz Wi-Fi network.
    2. Connects to local PC MQTT broker over LAN IPv4 (10.163.182.136:1883).
    3. Publishes structured JSON test message to topic: fall_detection/stage6/test
    4. Automatically reconnects Wi-Fi and MQTT non-blockingly if disconnected.
    5. Includes incrementing sequence counter and Wi-Fi RSSI.
    6. Completely isolated from MPU6050 fall detection ML pipeline and GPS.
*/

#include <WiFi.h>
#include <PubSubClient.h>

// ==============================================================================
// 1. NETWORK & BROKER CONFIGURATION
// Replace placeholders with your 2.4 GHz Wi-Fi credentials
// ==============================================================================
#ifndef WIFI_SSID
#define WIFI_SSID "OnePlus 11R 5G"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "rohit@02"
#endif

// PC LAN IPv4 Address running Mosquitto (OnePlus Hotspot IP: 10.163.182.136)
const char* mqtt_broker = "10.163.182.136";
const int mqtt_port = 1883;

// Topics and Identifiers
const char* mqtt_topic_test = "fall_detection/stage6/test";
const char* device_id = "ESP32-STAGE6-TEST";

// ==============================================================================
// 2. CLIENT & STATE OBJECTS
// ==============================================================================
WiFiClient espClient;
PubSubClient mqttClient(espClient);

unsigned long lastMqttReconnectAttempt = 0;
unsigned long lastPublishTime = 0;
const unsigned long publishIntervalMs = 10000; // Periodic publish every 10 seconds
int publishSequenceCounter = 0;

// ==============================================================================
// 3. WI-FI INITIALIZATION
// ==============================================================================
void setupWifi() {
  delay(10);
  Serial.println();
  Serial.println("==================================================");
  Serial.println("STAGE 6C: ESP32 Wi-Fi & MQTT Client Starting");
  Serial.println("==================================================");
  Serial.print("Connecting to Wi-Fi SSID: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startAttemptTime < 15000)) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[Wi-Fi] Connected successfully!");
    Serial.print("[Wi-Fi] Assigned IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[Wi-Fi] Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("\n[Wi-Fi WARNING] Initial connection timeout. Will auto-retry in loop.");
  }
}

// ==============================================================================
// 4. MQTT RECONNECTION (NON-BLOCKING)
// ==============================================================================
boolean reconnectMqtt() {
  Serial.print("[MQTT] Attempting connection to broker ");
  Serial.print(mqtt_broker);
  Serial.print(":");
  Serial.print(mqtt_port);
  Serial.print(" as ");
  Serial.print(device_id);
  Serial.print("...");

  // Connect with clean session
  if (mqttClient.connect(device_id)) {
    Serial.println(" CONNECTED!");
    // Send an immediate connection greeting JSON
    publishTestMessage("INITIAL_CONNECT");
    return true;
  } else {
    Serial.print(" FAILED, rc=");
    Serial.print(mqttClient.state());
    Serial.println(" (Will retry in 5s)");
    return false;
  }
}

// ==============================================================================
// 5. TEST JSON PUBLISHER
// ==============================================================================
void publishTestMessage(const char* eventReason) {
  publishSequenceCounter++;

  // Construct structured test JSON payload
  char jsonBuffer[256];
  snprintf(jsonBuffer, sizeof(jsonBuffer),
    "{\"device_id\":\"%s\",\"stage\":\"6C\",\"status\":\"TEST\",\"seq\":%d,\"event\":\"%s\",\"rssi\":%d,\"message\":\"ESP32 MQTT connection successful\"}",
    device_id,
    publishSequenceCounter,
    eventReason,
    WiFi.RSSI()
  );

  Serial.print("[MQTT PUBLISH] Topic: ");
  Serial.println(mqtt_topic_test);
  Serial.print("[MQTT PUBLISH] Payload: ");
  Serial.println(jsonBuffer);

  boolean pubSuccess = mqttClient.publish(mqtt_topic_test, jsonBuffer);
  if (pubSuccess) {
    Serial.println("[MQTT PUBLISH] Status: SUCCESS");
  } else {
    Serial.println("[MQTT PUBLISH] Status: FAILED (Buffer or Network Error)");
  }
}

// ==============================================================================
// 6. SETUP & MAIN LOOP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { ; } // Wait for serial port

  setupWifi();
  mqttClient.setServer(mqtt_broker, mqtt_port);
  mqttClient.setBufferSize(512); // Ensure buffer fits JSON payloads
}

void loop() {
  unsigned long now = millis();

  // 1. Maintain Wi-Fi Connection
  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastMqttReconnectAttempt > 5000) {
      Serial.println("[Wi-Fi] Disconnected, reconnecting...");
      WiFi.reconnect();
      lastMqttReconnectAttempt = now;
    }
    return;
  }

  // 2. Maintain MQTT Connection (Non-blocking retry every 5s)
  if (!mqttClient.connected()) {
    if (now - lastMqttReconnectAttempt > 5000) {
      lastMqttReconnectAttempt = now;
      if (reconnectMqtt()) {
        lastMqttReconnectAttempt = 0;
      }
    }
  } else {
    // 3. Process MQTT incoming loop & keepalive
    mqttClient.loop();

    // 4. Periodic Test Publish
    if (now - lastPublishTime >= publishIntervalMs) {
      lastPublishTime = now;
      publishTestMessage("PERIODIC_HEARTBEAT");
    }
  }
}
