"""
Stage 8A System Health Monitor.
Standalone Fall Detection System.

Central diagnostics aggregator tracking subsystem readiness:
  - Backend REST API
  - SQLite Database
  - Eclipse Mosquitto MQTT Broker
  - Stage 6G MQTT Gateway
  - ESP32 Hardware Serial Connection & Heartbeat
  - NEO-6M GPS Telemetry Availability
"""

import datetime
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple

from src.system_health.logger import get_operational_logger
from src.system_health.models import (
    BACKEND_ERROR,
    BACKEND_READY,
    DATABASE_ERROR,
    DATABASE_READY,
    ESP32_CONNECTED,
    ESP32_DISCONNECTED,
    ESP32_STALE,
    ESP32_UNKNOWN,
    EVENT_BACKEND_READY,
    EVENT_DATABASE_READY,
    EVENT_ESP32_CONNECTED,
    EVENT_ESP32_DISCONNECTED,
    EVENT_GPS_AVAILABLE,
    EVENT_GPS_UNAVAILABLE,
    EVENT_MQTT_CONNECTED,
    EVENT_MQTT_DISCONNECTED,
    EVENT_SYSTEM_DEGRADED,
    EVENT_SYSTEM_ERROR,
    EVENT_SYSTEM_READY,
    GATEWAY_DEGRADED,
    GATEWAY_DISCONNECTED,
    GATEWAY_READY,
    GPS_AVAILABLE,
    GPS_UNAVAILABLE,
    MQTT_CONNECTED,
    MQTT_DISCONNECTED,
    MQTT_RECONNECTING,
    SYSTEM_DEGRADED,
    SYSTEM_ERROR,
    SYSTEM_READY,
    SYSTEM_STARTING,
)


class SystemHealthMonitor:
    """
    Thread-safe health state aggregator providing real-time diagnostics
    and generating the response for GET /api/system/health.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.RLock()
        self.start_time = time.time()
        self.op_logger = get_operational_logger()

        # Database Configuration
        self.db_path = db_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "fall_detection_dataset", "fall_events.db")
        )

        # Subsystem States
        self._system_state = SYSTEM_STARTING
        self._backend_state = BACKEND_READY
        self._database_state = DATABASE_READY
        self._mqtt_state = MQTT_DISCONNECTED
        self._gateway_state = GATEWAY_DISCONNECTED
        self._esp32_state = ESP32_UNKNOWN
        self._gps_state = GPS_UNAVAILABLE

        # Subsystem Metadata
        self._esp32_port: Optional[str] = None
        self._esp32_baud: Optional[int] = None
        self._last_esp32_heartbeat: Optional[float] = None
        self._esp32_stale_threshold_s: float = 3.0

        self._mqtt_broker_host: str = "127.0.0.1"
        self._mqtt_broker_port: int = 1883
        self._mqtt_topic: str = "fall_detection/stage6/fall"

        self._gateway_forwarded_count: int = 0
        self._last_gps_metadata: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # State Updates
    # -------------------------------------------------------------------------
    def set_system_starting(self):
        """Marks initial startup in progress."""
        with self._lock:
            self._system_state = SYSTEM_STARTING

    def set_system_ready(self):
        """Marks system startup complete."""
        with self._lock:
            self._system_state = SYSTEM_READY
            self.op_logger.log_event(EVENT_SYSTEM_READY, "All core services operational")

    def update_backend_state(self, state: str, detail: Optional[str] = None):
        with self._lock:
            self._backend_state = state
            if state == BACKEND_READY:
                self.op_logger.log_event(EVENT_BACKEND_READY, detail or "Flask HTTP server active")

    def update_database_state(self, state: str, detail: Optional[str] = None):
        with self._lock:
            self._database_state = state
            if state == DATABASE_READY:
                self.op_logger.log_event(EVENT_DATABASE_READY, detail or "SQLite schema verified")
            elif state == DATABASE_ERROR:
                self.op_logger.log_event(EVENT_SYSTEM_ERROR, detail or "Database access failure")

    def update_mqtt_state(
        self,
        state: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        topic: Optional[str] = None,
    ):
        with self._lock:
            self._mqtt_state = state
            if host:
                self._mqtt_broker_host = host
            if port:
                self._mqtt_broker_port = port
            if topic:
                self._mqtt_topic = topic

            if state == MQTT_CONNECTED:
                self.op_logger.log_event(EVENT_MQTT_CONNECTED, f"Broker at {self._mqtt_broker_host}:{self._mqtt_broker_port}")
            elif state in (MQTT_DISCONNECTED, MQTT_RECONNECTING):
                self.op_logger.log_event(EVENT_MQTT_DISCONNECTED, f"State: {state}")

    def update_gateway_state(self, state: str, forwarded_count: Optional[int] = None):
        with self._lock:
            self._gateway_state = state
            if forwarded_count is not None:
                self._gateway_forwarded_count = forwarded_count

    def update_esp32_heartbeat(self, port: Optional[str] = None, baud: Optional[int] = None):
        """Called upon successful arrival of physical or simulated IMU telemetry."""
        with self._lock:
            self._last_esp32_heartbeat = time.time()
            if port:
                self._esp32_port = port
            if baud:
                self._esp32_baud = baud

            if self._esp32_state != ESP32_CONNECTED:
                self._esp32_state = ESP32_CONNECTED
                self.op_logger.log_event(EVENT_ESP32_CONNECTED, f"Port {self._esp32_port} @ {self._esp32_baud} baud")

    def mark_esp32_disconnected(self, reason: Optional[str] = None):
        with self._lock:
            self._esp32_state = ESP32_DISCONNECTED
            self.op_logger.log_event(EVENT_ESP32_DISCONNECTED, reason or "Serial port closed or unavailable")

    def update_gps_state(self, available: bool, metadata: Optional[Dict[str, Any]] = None):
        with self._lock:
            prev = self._gps_state
            self._gps_state = GPS_AVAILABLE if available else GPS_UNAVAILABLE
            if metadata:
                self._last_gps_metadata = metadata

            if self._gps_state != prev:
                if self._gps_state == GPS_AVAILABLE:
                    sats = metadata.get("satellites") if metadata else None
                    self.op_logger.log_event(EVENT_GPS_AVAILABLE, f"Fix valid (Satellites: {sats})")
                else:
                    self.op_logger.log_event(EVENT_GPS_UNAVAILABLE, "No satellite fix")

    # -------------------------------------------------------------------------
    # Health Probes & Dynamic Assessment
    # -------------------------------------------------------------------------
    def check_database_health(self) -> Tuple[str, Optional[str]]:
        """
        Executes a non-destructive, read-only SQLite schema and connectivity probe.
        Does NOT alter records or perform vacuum/migrations.
        """
        if not os.path.exists(self.db_path):
            self.update_database_state(DATABASE_ERROR, f"Database file missing: {self.db_path}")
            return DATABASE_ERROR, f"Database file not found at {self.db_path}"

        try:
            conn = sqlite3.connect(self.db_path, timeout=1.5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM fall_events LIMIT 1")
            cursor.fetchone()
            conn.close()
            self.update_database_state(DATABASE_READY)
            return DATABASE_READY, None
        except Exception as ex:
            self.update_database_state(DATABASE_ERROR, str(ex))
            return DATABASE_ERROR, str(ex)

    def assess_esp32_status(self) -> str:
        """Evaluates whether serial connection is CONNECTED, STALE, or DISCONNECTED."""
        with self._lock:
            if self._esp32_state == ESP32_DISCONNECTED:
                return ESP32_DISCONNECTED
            if self._last_esp32_heartbeat is None:
                return self._esp32_state

            elapsed = time.time() - self._last_esp32_heartbeat
            if elapsed > self._esp32_stale_threshold_s:
                self._esp32_state = ESP32_STALE
            else:
                self._esp32_state = ESP32_CONNECTED
            return self._esp32_state

    def compute_overall_status(self) -> str:
        """
        Computes composite status:
          SYSTEM_ERROR    if DB or Backend is in ERROR
          SYSTEM_DEGRADED if MQTT disconnected, ESP32 disconnected/stale, or GPS unavailable
          SYSTEM_READY    if all subsystems are operational
        """
        with self._lock:
            # Check critical failure
            if self._backend_state == BACKEND_ERROR or self._database_state == DATABASE_ERROR:
                return SYSTEM_ERROR

            # Check degradation
            esp_status = self.assess_esp32_status()
            if (
                self._mqtt_state != MQTT_CONNECTED
                or esp_status in (ESP32_DISCONNECTED, ESP32_STALE, ESP32_UNKNOWN)
                or self._gps_state == GPS_UNAVAILABLE
                or self._gateway_state != GATEWAY_READY
            ):
                return SYSTEM_DEGRADED

            return SYSTEM_READY

    # -------------------------------------------------------------------------
    # Diagnostic Representation
    # -------------------------------------------------------------------------
    def get_status_dict(self, service: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generates the standard Stage 8A JSON dictionary response for GET /api/system/health.
        """
        with self._lock:
            # Proactively probe database
            db_status, db_err = self.check_database_health()

            overall = self.compute_overall_status()
            uptime = round(time.time() - self.start_time, 2)
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            esp_status = self.assess_esp32_status()

            # Heartbeat latency
            heartbeat_latency = None
            if self._last_esp32_heartbeat is not None:
                heartbeat_latency = round(time.time() - self._last_esp32_heartbeat, 2)

            data = {
                "status": overall,
                "backend": {
                    "status": self._backend_state,
                    "service": "fall_event_backend",
                    "stage": "8A",
                },
                "database": {
                    "status": db_status,
                    "readable": (db_status == DATABASE_READY),
                    "error": db_err,
                },
                "mqtt": {
                    "status": self._mqtt_state,
                    "broker": f"{self._mqtt_broker_host}:{self._mqtt_broker_port}",
                    "topic": self._mqtt_topic,
                },
                "gateway": {
                    "status": self._gateway_state,
                    "forwarded_count": self._gateway_forwarded_count,
                },
                "esp32": {
                    "status": esp_status,
                    "port": self._esp32_port or os.environ.get("SERIAL_PORT", "COM22"),
                    "baud": self._esp32_baud or int(os.environ.get("SERIAL_BAUD", 460800)),
                    "last_heartbeat_s": heartbeat_latency,
                },
                "gps": {
                    "status": self._gps_state,
                    "fix_valid": bool(self._last_gps_metadata and self._last_gps_metadata.get("fix_valid")),
                    "satellites": self._last_gps_metadata.get("satellites") if self._last_gps_metadata else None,
                },
                "uptime_seconds": uptime,
                "timestamp": now_iso,
            }

            return data


# Global singleton instance
_GLOBAL_HEALTH_MONITOR = SystemHealthMonitor()


def get_global_health_monitor() -> SystemHealthMonitor:
    """Returns the process-wide SystemHealthMonitor instance."""
    return _GLOBAL_HEALTH_MONITOR
