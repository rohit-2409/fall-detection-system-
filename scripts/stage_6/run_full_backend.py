"""
Starts the integrated Standalone Fall Detection System Unified Backend on port 5000.
Coordinates in a single lifecycle:
  1. Flask REST API (Stage 5A/5B, Stage 6I/6J/6K Notifications & Monitoring)
  2. Stage 6G MQTT Gateway (Subscribing to fall_detection/stage6/fall -> POST /api/fall-events)
"""
import argparse
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from werkzeug.serving import make_server

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_6G_mqtt_gateway import (
    DEFAULT_BROKER_HOST,
    DEFAULT_BROKER_PORT,
    DEFAULT_FALL_TOPIC,
    MqttBackendGateway,
)
from src.notifications.notification_listener import create_notifying_app
from src.system_health.models import (
    BACKEND_READY,
    DATABASE_READY,
    DATABASE_ERROR,
    EVENT_SYSTEM_START,
    EVENT_SYSTEM_SHUTDOWN,
    GATEWAY_READY,
    GATEWAY_DISCONNECTED,
    MQTT_CONNECTED,
    MQTT_DISCONNECTED,
)
from src.system_health.monitor import get_global_health_monitor

logger = logging.getLogger("UnifiedBackend")


class UnifiedBackendServer:
    """
    Coordinates Flask REST backend and Stage 6G MQTT Gateway within a single
    process lifecycle, ensuring clean startup synchronization and zero orphaned processes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        broker_host: str = DEFAULT_BROKER_HOST,
        broker_port: int = DEFAULT_BROKER_PORT,
        topic: str = DEFAULT_FALL_TOPIC,
        enable_gateway: bool = True,
    ):
        self.host = host
        self.port = port
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.enable_gateway = enable_gateway
        self.backend_url = f"http://{self.host}:{self.port}"

        self.app = create_notifying_app()
        self.http_server = None
        self.server_thread: Optional[threading.Thread] = None
        self.gateway: Optional[MqttBackendGateway] = None
        self._stop_event = threading.Event()
        self._is_running = False

        # Stage 8A Health Monitoring
        self.health_monitor = get_global_health_monitor()
        self.health_monitor.set_system_starting()
        self.health_monitor.op_logger.log_event(EVENT_SYSTEM_START, "Unified Backend starting")

    def start(self, timeout: float = 10.0) -> bool:
        """Starts Flask HTTP server, verifies health, and starts MQTT Gateway."""
        if self._is_running:
            return True

        # 1. Start Flask HTTP Server in background thread
        try:
            self.http_server = make_server(self.host, self.port, self.app, threaded=True)
            self.server_thread = threading.Thread(
                target=self.http_server.serve_forever,
                name="UnifiedBackend-FlaskThread",
                daemon=True,
            )
            self.server_thread.start()
        except Exception as e:
            print(f"[ERROR] Failed to start Flask HTTP server on {self.host}:{self.port}: {e}")
            return False

        # 2. Wait until backend is available
        start_time = time.time()
        backend_ready = False
        health_url = f"{self.backend_url}/api/health"
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(
                    health_url,
                    headers={"User-Agent": "UnifiedBackend-HealthCheck/1.0"},
                )
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        backend_ready = True
                        break
            except Exception:
                time.sleep(0.1)

        if not backend_ready:
            print(f"[ERROR] Backend did not respond at {health_url} within {timeout}s.")
            self.stop()
            return False

        self.health_monitor.update_backend_state(BACKEND_READY, f"Flask active on {self.backend_url}")
        db_stat, db_err = self.health_monitor.check_database_health()
        print(f"[OK] Flask backend started")
        print(f"[OK] Backend target: {self.backend_url}/api/fall-events")
        print(f"[OK] System health:  {self.backend_url}/api/system/health")
        if db_stat == "READY":
            print(f"[OK] SQLite database verified: READY")
        else:
            print(f"[WARNING] SQLite database check: {db_err}")

        # 3. Start MQTT Gateway if enabled
        if self.enable_gateway:
            self.gateway = MqttBackendGateway(
                broker_host=self.broker_host,
                broker_port=self.broker_port,
                topic=self.topic,
                client_id="FALL-STAGE6G-UNIFIED-BACKEND",
                backend_url=self.backend_url,
            )
            gw_connected = self.gateway.start(timeout=5.0)
            if gw_connected:
                self.health_monitor.update_mqtt_state(
                    MQTT_CONNECTED,
                    host=self.broker_host,
                    port=self.broker_port,
                    topic=self.topic,
                )
                self.health_monitor.update_gateway_state(GATEWAY_READY)
                print(f"[OK] MQTT gateway started")
                print(f"[OK] MQTT broker target: {self.broker_host}:{self.broker_port}")
                print(f"[OK] MQTT topic: {self.topic}")
            else:
                self.health_monitor.update_mqtt_state(
                    MQTT_DISCONNECTED,
                    host=self.broker_host,
                    port=self.broker_port,
                    topic=self.topic,
                )
                self.health_monitor.update_gateway_state(GATEWAY_DISCONNECTED)
                print(f"[WARNING] MQTT gateway could not connect to broker at {self.broker_host}:{self.broker_port}. Gateway will retry in background.")
                print(f"[OK] MQTT broker target: {self.broker_host}:{self.broker_port}")
                print(f"[OK] MQTT topic: {self.topic}")

        # 4. Check hardware port availability (non-blocking)
        serial_port = os.environ.get("SERIAL_PORT", "COM22")
        serial_baud = int(os.environ.get("SERIAL_BAUD", 460800))
        try:
            import serial.tools.list_ports
            detected_ports = [p.device for p in serial.tools.list_ports.comports()]
            if serial_port in detected_ports:
                self.health_monitor.update_esp32_heartbeat(port=serial_port, baud=serial_baud)
                print(f"[OK] ESP32 hardware detected on {serial_port}")
            else:
                self.health_monitor.mark_esp32_disconnected(f"Port {serial_port} not connected")
                print(f"[INFO] ESP32 not connected on {serial_port} (Backend operational in decoupled mode)")
        except Exception:
            self.health_monitor.mark_esp32_disconnected("Serial tools unavailable")

        self.health_monitor.set_system_ready()
        self._is_running = True
        return True

    def stop(self):
        """Stops gateway and HTTP server cleanly."""
        self._stop_event.set()
        if hasattr(self, "health_monitor") and self.health_monitor:
            from src.system_health.models import EVENT_SYSTEM_SHUTDOWN
            self.health_monitor.op_logger.log_event(EVENT_SYSTEM_SHUTDOWN, "Unified Backend stopping")

        if self.gateway:
            try:
                self.gateway.stop()
            except Exception:
                pass
            self.gateway = None

        if self.http_server:
            try:
                self.http_server.shutdown()
                self.http_server.server_close()
            except Exception:
                pass
            self.http_server = None

        self._is_running = False

    def serve_forever(self):
        """Blocks main thread until KeyboardInterrupt or shutdown."""
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[INFO] Shutdown requested. Stopping Unified Backend...")
        finally:
            self.stop()
            print("[INFO] Unified Backend stopped cleanly.")


def run_unified_backend(
    host: Optional[str] = None,
    port: Optional[int] = None,
    broker_host: Optional[str] = None,
    broker_port: Optional[int] = None,
    topic: Optional[str] = None,
    enable_gateway: bool = True,
):
    """Entry point for CLI or batch script execution."""
    host = host or os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = port or int(os.environ.get("BACKEND_PORT", 5000))
    broker_host = broker_host or os.environ.get("MQTT_BROKER_HOST", DEFAULT_BROKER_HOST)
    broker_port = broker_port or int(os.environ.get("MQTT_BROKER_PORT", DEFAULT_BROKER_PORT))
    topic = topic or os.environ.get("MQTT_FALL_TOPIC", DEFAULT_FALL_TOPIC)

    print("=" * 70)
    print("STANDALONE FALL DETECTION SYSTEM: UNIFIED BACKEND & MQTT GATEWAY")
    print("=" * 70)
    print(f"Backend Host:       {host}:{port}")
    print(f"MQTT Broker:        {broker_host}:{broker_port}")
    print(f"MQTT Fall Topic:    {topic}")
    print(f"Gateway Enabled:    {enable_gateway}")
    print("=" * 70)

    server = UnifiedBackendServer(
        host=host,
        port=port,
        broker_host=broker_host,
        broker_port=broker_port,
        topic=topic,
        enable_gateway=enable_gateway,
    )

    started = server.start()
    if not started:
        print("[ABORT] Failed to start Unified Backend.")
        sys.exit(1)

    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Fall Detection Unified Backend")
    parser.add_argument("--host", type=str, default=None, help="Backend bind host")
    parser.add_argument("--port", type=int, default=None, help="Backend bind port")
    parser.add_argument("--broker", type=str, default=None, help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=None, help="MQTT broker port")
    parser.add_argument("--topic", type=str, default=None, help="MQTT fall topic")
    parser.add_argument("--no-gateway", action="store_true", help="Disable embedded MQTT gateway")

    args = parser.parse_args()
    run_unified_backend(
        host=args.host,
        port=args.port,
        broker_host=args.broker,
        broker_port=args.broker_port,
        topic=args.topic,
        enable_gateway=(not args.no_gateway),
    )
