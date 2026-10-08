/*
  Stage 6D: u-blox NEO-6M GPS -> ESP32 Dedicated Hardware Ingestion Test
  Standalone Fall Detection Project (Active Working Copy)

  Hardware Configuration:
    - Microcontroller: ESP32 Dev Module
    - GPS Module: u-blox NEO-6M Breakout Board
    - Serial Peripheral: Hardware UART2 (SerialGPS)
    - ESP32 RX Pin: GPIO 16 (connected to NEO-6M TX)
    - ESP32 TX Pin: GPIO 17 (connected to NEO-6M RX)
    - GPS Baud Rate: 9600 bps (standard NEO-6M factory default)
    - Power Connections: NEO-6M GND -> ESP32 GND, NEO-6M VCC -> 3.3V or 5V (per breakout labeling)

  Functionality:
    - Continuous non-blocking byte-by-byte NMEA parsing via TinyGPSPlus.
    - Decoupled from MPU6050, Random Forest, TemporalConfirmationAdapter, and MQTT.
    - Tracks states: ACQUIRING, FIX_VALID, FIX_LOST.
    - Displays latitude, longitude, fix validity, satellites, HDOP, UTC time, and NMEA diagnostic counters.
    - Periodic report output at approximately 1000 ms intervals (zero blocking delays).
*/

#include <Arduino.h>
#include <HardwareSerial.h>
#include <TinyGPSPlus.h>

// ==============================================================================
// 1. PIN & SERIAL CONFIGURATION
// ==============================================================================
#define GPS_RX_PIN 16 // ESP32 GPIO 16 connects to NEO-6M TX
#define GPS_TX_PIN 17 // ESP32 GPIO 17 connects to NEO-6M RX
#define GPS_BAUD   9600

#define SERIAL_MONITOR_BAUD 115200
#define REPORT_INTERVAL_MS  1000

// ==============================================================================
// 2. GLOBAL HARDWARE & PARSER OBJECTS
// ==============================================================================
TinyGPSPlus gps;
HardwareSerial SerialGPS(2); // ESP32 Hardware UART2

// ==============================================================================
// 3. GPS STATE TRACKING & TIMING
// ==============================================================================
enum GpsState {
  STATE_ACQUIRING,
  STATE_FIX_VALID,
  STATE_FIX_LOST
};

GpsState currentState = STATE_ACQUIRING;
bool hasEverHadValidFix = false;
double lastKnownLatitude = 0.0;
double lastKnownLongitude = 0.0;
unsigned long lastValidFixMillis = 0;
unsigned long lastReportMillis = 0;

// Maximum acceptable age for an active fix before transitioning to FIX_LOST (2.5 seconds)
const unsigned long MAX_FIX_AGE_MS = 2500;

// ==============================================================================
// 4. SETUP
// ==============================================================================
void setup() {
  Serial.begin(SERIAL_MONITOR_BAUD);
  while (!Serial && millis() < 3000) { ; } // Allow serial monitor to open

  Serial.println();
  Serial.println("==================================================");
  Serial.println("STAGE 6D: ESP32 + NEO-6M GPS HARDWARE TEST");
  Serial.println("==================================================");
  Serial.print("[CONFIG] Hardware Serial: UART2 (GPIO ");
  Serial.print(GPS_RX_PIN);
  Serial.print(" RX, GPIO ");
  Serial.print(GPS_TX_PIN);
  Serial.println(" TX)");
  Serial.print("[CONFIG] Baud Rate:       ");
  Serial.print(GPS_BAUD);
  Serial.println(" bps");
  Serial.println("[CONFIG] Parser:          TinyGPSPlus v1.0.3");
  Serial.println("==================================================");
  Serial.println("Initializing Hardware UART2 for GPS communication...");

  // Initialize Hardware UART2: 8 data bits, no parity, 1 stop bit
  SerialGPS.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Serial.println("[READY] Listening for continuous NMEA data stream...");
  Serial.println();
}

// ==============================================================================
// 5. PERIODIC STATUS REPORTER (~1000 ms, Non-Blocking)
// ==============================================================================
void printGpsStatusReport() {
  bool isFixCurrentlyValid = gps.location.isValid() && (gps.location.age() <= MAX_FIX_AGE_MS);

  // Determine current lifecycle state
  if (isFixCurrentlyValid) {
    currentState = STATE_FIX_VALID;
    hasEverHadValidFix = true;
    lastKnownLatitude = gps.location.lat();
    lastKnownLongitude = gps.location.lng();
    lastValidFixMillis = millis();
  } else if (hasEverHadValidFix) {
    currentState = STATE_FIX_LOST;
  } else {
    currentState = STATE_ACQUIRING;
  }

  Serial.println("==================================================");
  Serial.println("STAGE 6D — NEO-6M GPS TEST");
  Serial.println("==================================================");

  // 1. State & Validity
  switch (currentState) {
    case STATE_FIX_VALID:
      Serial.println("GPS STATUS: FIX_VALID");
      Serial.println("Fix: VALID");
      break;
    case STATE_FIX_LOST:
      Serial.println("GPS STATUS: FIX_LOST (Signal Temporarily Lost / Obstructed)");
      Serial.println("Fix: INVALID (Stale Position Cached)");
      break;
    case STATE_ACQUIRING:
    default:
      Serial.println("GPS STATUS: ACQUIRING (Searching for Satellites...)");
      Serial.println("Fix: INVALID");
      break;
  }

  // 2. Satellite Count
  Serial.print("Satellites: ");
  if (gps.satellites.isValid()) {
    Serial.println(gps.satellites.value());
  } else {
    Serial.println("0 (Acquiring)");
  }

  // 3. Location (Current vs Stale vs None)
  if (currentState == STATE_FIX_VALID) {
    Serial.print("Latitude:   ");
    Serial.println(gps.location.lat(), 6);
    Serial.print("Longitude:  ");
    Serial.println(gps.location.lng(), 6);
  } else if (currentState == STATE_FIX_LOST) {
    Serial.print("Latitude:   ");
    Serial.print(lastKnownLatitude, 6);
    Serial.println(" [STALE]");
    Serial.print("Longitude:  ");
    Serial.print(lastKnownLongitude, 6);
    Serial.println(" [STALE]");
  } else {
    Serial.println("Latitude:   0.000000");
    Serial.println("Longitude:  0.000000");
  }

  // 4. HDOP (Horizontal Dilution of Precision)
  Serial.print("HDOP:       ");
  if (gps.hdop.isValid()) {
    Serial.println(gps.hdop.hdop());
  } else {
    Serial.println("N/A");
  }

  // 5. UTC Date & Time
  Serial.print("UTC:        ");
  if (gps.date.isValid() && gps.time.isValid()) {
    char timeBuffer[32];
    snprintf(timeBuffer, sizeof(timeBuffer), "%04d-%02d-%02d %02d:%02d:%02d",
             gps.date.year(), gps.date.month(), gps.date.day(),
             gps.time.hour(), gps.time.minute(), gps.time.second());
    Serial.println(timeBuffer);
  } else {
    Serial.println("N/A");
  }

  // 6. Fix Age
  Serial.print("GPS Age:    ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.age());
    Serial.println(" ms");
  } else if (hasEverHadValidFix) {
    Serial.print(millis() - lastValidFixMillis);
    Serial.println(" ms (Since Last Valid Fix)");
  } else {
    Serial.println("N/A");
  }

  // 7. Diagnostic NMEA Stream Counters
  Serial.print("Diagnostics: Chars: ");
  Serial.print(gps.charsProcessed());
  Serial.print(" | Sentences with Fix: ");
  Serial.print(gps.sentencesWithFix());
  Serial.print(" | Passed Checksum: ");
  Serial.print(gps.passedChecksum());
  Serial.print(" | Failed Checksum: ");
  Serial.println(gps.failedChecksum());

  // Troubleshooting warning if no bytes arrive from GPS hardware
  if (gps.charsProcessed() == 0 && millis() > 5000) {
    Serial.println("[WARNING] No NMEA data received yet. Check wiring: NEO-6M TX -> ESP32 GPIO 16, GND -> GND, VCC.");
  }

  Serial.println("--------------------------------------------------");
  Serial.println();
}

// ==============================================================================
// 6. MAIN LOOP (Continuous Byte-by-Byte Ingestion, Zero Blocking Delay)
// ==============================================================================
void loop() {
  // Continuously ingest available characters from Hardware UART2
  while (SerialGPS.available() > 0) {
    char c = SerialGPS.read();
    gps.encode(c);
  }

  // Non-blocking periodic status report (~1000 ms)
  unsigned long now = millis();
  if (now - lastReportMillis >= REPORT_INTERVAL_MS) {
    lastReportMillis = now;
    printGpsStatusReport();
  }
}
