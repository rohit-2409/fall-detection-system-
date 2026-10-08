"""
Stage 6L: Integrated Hardware Inference Bridge.
Standalone Fall Detection Project.

Architecture:
  ESP32 (USB Serial COM22 @ 460800 baud)
    ↓ JSON Lines (or CSV / Pipe format)
  Rolling 50-sample FIFO window (25-sample step, 50% overlap @ 100 Hz)
    ↓
  100-feature extraction (extract_100_features from Stage 4G/5E)
    ↓
  Frozen Stage 4D Random Forest model inference (predict_proba)
    ↓
  Stage 4E TemporalConfirmationAdapter (threshold=0.60, rule=N3, cooldown=2.0s)
    ↓
  Gating on confirmed_fall == True
    ↓
  Query Stage 6L GPS Cache for latest valid, fresh coordinates
    ↓
  Dispatch via Stage 6F MqttFallPublisher (Topic: fall_detection/stage6/fall)
    ↓
  Stage 6G MQTT Gateway -> Flask Backend -> SQLite -> Stage 6J -> Stage 6K
"""

import argparse
import collections
import datetime
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_4G_live_validator import (
    DEFAULT_FEATURE_ORDER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    STEP_SIZE,
    WINDOW_SIZE,
    compute_dynamic_acceleration,
    compute_jerk_magnitude,
    extract_100_features,
)
from src.fall_confirmation.stage_6F_mqtt_fall_publisher import MqttFallPublisher
from src.fall_confirmation.stage_6L_gps_cache import GpsTelemetryCache, get_global_gps_cache
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter

logger = logging.getLogger("Stage6L_HardwareBridge")

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


class HardwareBridgeHeartbeatSender:
    """
    Asynchronously and non-blockingly dispatches throttled HTTP heartbeats to
    Flask backend (POST /api/system/heartbeat).
    Ensures that network latency, timeouts, or backend downtime NEVER interrupt
    serial reading, MPU6050 ingestion, feature extraction, inference, temporal
    confirmation, or MQTT publishing.
    """

    def __init__(
        self,
        backend_url: str = DEFAULT_BACKEND_URL,
        interval_seconds: float = 1.5,
        timeout_seconds: float = 1.0,
    ):
        self.backend_url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self._last_sent_monotonic = 0.0
        self._lock = threading.Lock()
        self._pending_payload: Optional[Dict[str, Any]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._is_active = False

    def start(self) -> None:
        if self._is_active:
            return
        self._is_active = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="Stage6L-HeartbeatWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._is_active = False
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        self._worker_thread = None

    def queue_heartbeat(
        self,
        port: str,
        baud: int,
        gps_available: bool,
        gps_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Throttled non-blocking enqueue.
        Returns immediately (< 1 microsecond execution).
        """
        now = time.monotonic()
        if now - self._last_sent_monotonic < self.interval_seconds:
            return

        with self._lock:
            self._last_sent_monotonic = now
            self._pending_payload = {
                "port": str(port),
                "baud": int(baud),
                "gps_available": bool(gps_available),
                "gps_metadata": gps_metadata or {},
            }
        self._wake_event.set()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.5)
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            payload = None
            with self._lock:
                if self._pending_payload is not None:
                    payload = self._pending_payload
                    self._pending_payload = None

            if payload is not None:
                self._dispatch_http_request(payload)

    def _dispatch_http_request(self, payload: Dict[str, Any]) -> bool:
        endpoint = f"{self.backend_url}/api/system/heartbeat"
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Stage6L-HardwareBridge/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return resp.status == 200
        except Exception as ex:
            # Silently catch network errors or backend downtime without interrupting hardware
            logger.debug("Heartbeat dispatch to %s failed: %s", endpoint, ex)
            return False

# Parsing Regexes for Legacy Formats
RE_PIPE_FORMAT = re.compile(
    r"ACC(?:\(g\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"\s*\|\s*GYRO?(?:\(deg/s\))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
RE_FLOAT = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_serial_stream_line(
    line: str,
    gps_cache: Optional[GpsTelemetryCache] = None,
) -> Optional[Tuple[Optional[float], float, float, float, float, float, float]]:
    """
    Parses a single raw serial line into:
      (timestamp_ms_or_None, ax, ay, az, gx, gy, gz) in units of (g, g, g, deg/s, deg/s, deg/s).
    Supports:
      1. Stage 6L JSON Lines: {"type":"imu","timestamp_ms":...,"ax_g":...,"ay_g":...,"az_g":...,"gx_dps":...,"gy_dps":...,"gz_dps":...}
      2. Stage 6L GPS JSON Lines: {"type":"gps", ...} -> forwards directly to gps_cache.
      3. Stage 5F Pipe format: ACC(g): ax, ay, az | GYRO(deg/s): gx, gy, gz
      4. Standard 7-column CSV: timestamp_ms,ax,ay,az,gx,gy,gz
      5. Standard 6-column CSV: ax,ay,az,gx,gy,gz
    """
    line = line.strip()
    if not line:
        return None

    # Format 1: JSON Lines
    if line.startswith("{") and line.endswith("}"):
        try:
            data = json.loads(line)
            msg_type = data.get("type", "").lower()
            if msg_type == "imu":
                ts = data.get("timestamp_ms") or data.get("timestamp")
                ax = float(data.get("ax_g", data.get("ax", 0.0)))
                ay = float(data.get("ay_g", data.get("ay", 0.0)))
                az = float(data.get("az_g", data.get("az", 0.0)))
                gx = float(data.get("gx_dps", data.get("gx", 0.0)))
                gy = float(data.get("gy_dps", data.get("gy", 0.0)))
                gz = float(data.get("gz_dps", data.get("gz", 0.0)))
                return (float(ts) if ts is not None else None, ax, ay, az, gx, gy, gz)
            elif msg_type == "gps" and gps_cache is not None:
                gps_cache.process_payload(data)
                return None
            elif "ax_g" in data or "ax" in data:
                # Direct dict without type tag
                ax = float(data.get("ax_g", data.get("ax", 0.0)))
                ay = float(data.get("ay_g", data.get("ay", 0.0)))
                az = float(data.get("az_g", data.get("az", 0.0)))
                gx = float(data.get("gx_dps", data.get("gx", 0.0)))
                gy = float(data.get("gy_dps", data.get("gy", 0.0)))
                gz = float(data.get("gz_dps", data.get("gz", 0.0)))
                ts = data.get("timestamp_ms") or data.get("timestamp")
                return (float(ts) if ts is not None else None, ax, ay, az, gx, gy, gz)
        except Exception:
            return None

    lower = line.lower()
    if any(h in lower for h in ["starting", "i2c", "ready", "setup", "baud", "timestamp", "accx"]):
        return None

    # Format 2: Pipe format
    m_pipe = RE_PIPE_FORMAT.search(line)
    if m_pipe:
        try:
            return (
                None,
                float(m_pipe.group(1)),
                float(m_pipe.group(2)),
                float(m_pipe.group(3)),
                float(m_pipe.group(4)),
                float(m_pipe.group(5)),
                float(m_pipe.group(6)),
            )
        except (ValueError, IndexError):
            pass

    # Format 3: Comma-separated CSV
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7:
            try:
                return (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                )
            except (ValueError, IndexError):
                pass
        elif len(parts) == 6:
            try:
                return (
                    None,
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                )
            except (ValueError, IndexError):
                pass

    return None


class MockSensorStream:
    """Simulates a 100 Hz sensor stream for hermetic testing or offline validation."""

    def __init__(self, mode: str = "stationary", total_samples: int = 250):
        self.mode = mode
        self.total_samples = total_samples
        self.current = 0
        self.replay_samples = []

        if self.mode == "fall":
            # Replay authentic fall dynamics from Stage 4G validation dataset (session 12 forward slip)
            replay_csv = os.path.join(
                PROJECT_ROOT,
                "fall_detection_dataset",
                "live_validation",
                "stage_4G",
                "session_012_fall_forward_slip",
                "raw_sensor_data.csv"
            )
            if os.path.exists(replay_csv):
                import csv
                try:
                    with open(replay_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        # Slice starting at sample 550 (the onset of the confirmed fall sequence)
                        target_rows = rows[550:550 + max(total_samples, 200)]
                        for r in target_rows:
                            self.replay_samples.append((
                                float(r["AccX"]),
                                float(r["AccY"]),
                                float(r["AccZ"]),
                                float(r["GyrX"]),
                                float(r["GyrY"]),
                                float(r["GyrZ"]),
                            ))
                except Exception as e:
                    self.replay_samples = []

    def readline(self) -> bytes:
        if self.current >= self.total_samples:
            return b""
        idx = self.current
        self.current += 1
        ts = self.current * 10

        if self.mode == "stationary":
            # Normal resting: 1g on Z, near 0 on others
            ax = 0.01 + 0.005 * math.sin(self.current)
            ay = -0.02 + 0.005 * math.cos(self.current)
            az = 0.98 + 0.005 * math.sin(self.current * 0.5)
            gx = 0.2
            gy = -0.1
            gz = 0.05
        elif self.mode == "fall":
            if self.replay_samples and idx < len(self.replay_samples):
                ax, ay, az, gx, gy, gz = self.replay_samples[idx]
            else:
                # Fallback high dynamic motion if dataset file not found
                if 25 <= self.current <= 85:
                    ax = 2.5 + 0.5 * math.sin(self.current)
                    ay = -2.8 + 0.5 * math.cos(self.current)
                    az = 3.2 + 0.8 * math.sin(self.current)
                    gx = 180.0
                    gy = -210.0
                    gz = 95.0
                else:
                    ax, ay, az, gx, gy, gz = 0.01, -0.02, 0.98, 0.2, -0.1, 0.05
        else:
            ax, ay, az, gx, gy, gz = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0

        json_line = json.dumps({
            "type": "imu",
            "timestamp_ms": ts,
            "ax_g": round(ax, 4),
            "ay_g": round(ay, 4),
            "az_g": round(az, 4),
            "gx_dps": round(gx, 4),
            "gy_dps": round(gy, 4),
            "gz_dps": round(gz, 4),
        }) + "\n"
        return json_line.encode("utf-8")

    @property
    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        pass


def run_stage_6L_hardware_bridge(
    port: str = "COM22",
    baudrate: Optional[int] = None,
    broker_host: str = "127.0.0.1",
    broker_port: int = 1883,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    feature_order_path: str = DEFAULT_FEATURE_ORDER_PATH,
    session_id: Optional[str] = None,
    max_windows: Optional[int] = None,
    dry_run: bool = False,
    mock_mode: bool = False,
    mock_scenario: str = "stationary",
    gps_cache: Optional[GpsTelemetryCache] = None,
    mqtt_publisher: Optional[MqttFallPublisher] = None,
    backend_url: Optional[str] = None,
    heartbeat_sender: Optional[HardwareBridgeHeartbeatSender] = None,
) -> Dict[str, Any]:
    """
    Main hardware bridge execution loop.
    Extracts features from serial IMU stream, runs model inference,
    applies temporal confirmation, attaches live GPS, and publishes to MQTT.
    """
    if baudrate is None:
        baudrate = int(os.environ.get("SERIAL_BAUD", 460800))
    session_id = session_id or f"STAGE6L-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    print("=" * 70)
    print("STAGE 6L INTEGRATED HARDWARE INFERENCE & MQTT BRIDGE")
    print("=" * 70)
    print(f"Session ID:         {session_id}")
    print(f"Serial Port:        {port} @ {baudrate} baud (Mock: {mock_mode})")
    print(f"MQTT Broker:        {broker_host}:{broker_port}")
    print(f"Dry Run Mode:       {dry_run}")
    print("=" * 70)

    # 1. Load ML Artifacts
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file missing: {scaler_path}")
    if not os.path.exists(feature_order_path):
        raise FileNotFoundError(f"Feature order file missing: {feature_order_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_order_path, "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # 2. Initialize Temporal Confirmation Adapter
    adapter = TemporalConfirmationAdapter()

    # 3. Initialize GPS Cache
    cache = gps_cache or get_global_gps_cache()
    if not cache.is_connected and not mock_mode:
        cache.start(timeout=2.0)

    # 4. Initialize Stage 6F MQTT Publisher
    publisher = mqtt_publisher or MqttFallPublisher(
        broker_host=broker_host,
        broker_port=broker_port,
    )
    if not publisher.is_connected and not dry_run and not mock_mode:
        publisher.connect(timeout=2.0)

    # 4b. Initialize Asynchronous Heartbeat Sender
    target_backend_url = backend_url or os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)
    hb_sender = heartbeat_sender or HardwareBridgeHeartbeatSender(backend_url=target_backend_url)
    hb_sender.start()

    # 5. Open Serial Stream
    ser = None
    if mock_mode:
        total_samples = (max_windows * 25 + 50) if max_windows else 200
        ser = MockSensorStream(mode=mock_scenario, total_samples=total_samples)
    else:
        try:
            import serial
            ser = serial.Serial(port, baudrate, timeout=1.0)
            print(f"[OK] Connected to hardware serial port {port}")
        except Exception as e:
            print(f"[ERROR] Could not open serial port {port}: {e}")
            raise

    # 6. Stream Processing State
    fifo_acc: Deque[Tuple[float, float, float]] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_gyr: Deque[Tuple[float, float, float]] = collections.deque(maxlen=WINDOW_SIZE)
    fifo_ts: Deque[float] = collections.deque(maxlen=WINDOW_SIZE)

    total_samples = 0
    window_count = 0
    confirmed_falls = 0
    mqtt_published = 0
    mqtt_failures = 0

    try:
        samples_since_step = 0

        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                if mock_mode:
                    break
                continue

            try:
                line_str = line_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            parsed = parse_serial_stream_line(line_str, gps_cache=cache)
            if parsed is None:
                continue

            ts_ms, ax, ay, az, gx, gy, gz = parsed
            t_sec = (ts_ms / 1000.0) if ts_ms is not None else time.time()

            fifo_acc.append((ax, ay, az))
            fifo_gyr.append((gx, gy, gz))
            fifo_ts.append(t_sec)

            total_samples += 1
            samples_since_step += 1

            # Asynchronous IPC Heartbeat dispatch to Flask Backend (non-blocking)
            cur_loc = cache.get_latest_location() if cache else {}
            gps_avail = bool(cur_loc.get("fix_valid", False))
            hb_sender.queue_heartbeat(
                port=port,
                baud=baudrate,
                gps_available=gps_avail,
                gps_metadata=cur_loc,
            )

            # Window processing when buffer is full and step size is reached
            if len(fifo_acc) == WINDOW_SIZE and samples_since_step >= STEP_SIZE:
                samples_since_step = 0
                window_count += 1

                acc_3d = np.array(fifo_acc, dtype=np.float64)
                gyr_3d = np.array(fifo_gyr, dtype=np.float64)

                win_dict = {
                    "AccX": acc_3d[:, 0],
                    "AccY": acc_3d[:, 1],
                    "AccZ": acc_3d[:, 2],
                    "GyrX": gyr_3d[:, 0],
                    "GyrY": gyr_3d[:, 1],
                    "GyrZ": gyr_3d[:, 2],
                }

                # Extract 100 features
                feat_vec = extract_100_features(win_dict)
                feat_input = np.nan_to_num(feat_vec.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

                # Scale and predict
                if scaler is not None:
                    feat_input = scaler.transform(feat_input)
                prob = float(model.predict_proba(feat_input)[0, 1])

                cur_gyr_mag = float(np.max(np.linalg.norm(gyr_3d, axis=1)))
                cur_dyn_acc = float(np.max(compute_dynamic_acceleration(acc_3d)))
                cur_jerk = float(np.max(compute_jerk_magnitude(acc_3d)))
                w_start = float(fifo_ts[0])
                acc_z_mean = float(np.mean(acc_3d[:, 2]))
                acc_mean = (float(np.mean(acc_3d[:, 0])), float(np.mean(acc_3d[:, 1])), acc_z_mean)

                # Temporal Confirmation & Incident Latching
                temp_res = adapter.process_prediction(
                    probability=prob,
                    window_start_time=w_start,
                    dyn_acc_max=cur_dyn_acc,
                    gyr_mag_max=cur_gyr_mag,
                    jerk_mag_max=cur_jerk,
                    acc_z_mean=acc_z_mean,
                    acc_mean=acc_mean,
                )

                is_confirmed = bool(temp_res["confirmed_fall"])

                # Stage 8A Health Monitoring Heartbeat (In-process and cross-process)
                try:
                    from src.system_health.monitor import get_global_health_monitor
                    h_mon = get_global_health_monitor()
                    h_mon.update_esp32_heartbeat(port=port, baud=baudrate)
                    if cache:
                        h_mon.update_gps_state(available=gps_avail, metadata=cur_loc)
                except Exception:
                    pass

                try:
                    hb_sender.queue_heartbeat(
                        port=port,
                        baud=baudrate,
                        gps_available=gps_avail,
                        gps_metadata=cur_loc,
                    )
                except Exception:
                    pass

                if window_count % 4 == 0 or temp_res["positive_count"] > 0 or is_confirmed or temp_res.get("temporal_state") in ("LOCKED_ALARM", "WAIT_UPRIGHT_STABLE", "REARMED"):
                    print(
                        f"  [Window {window_count:04d}] prob={prob:.4f} | dyn_acc={cur_dyn_acc:.2f}g | "
                        f"gyr_mag={cur_gyr_mag:.1f}dps | state={temp_res['temporal_state']} (pos={temp_res['positive_count']})",
                        flush=True
                    )

                if is_confirmed:
                    confirmed_falls += 1
                    event_id = temp_res.get("incident_id") or f"STAGE6L-FALL-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{confirmed_falls:02d}"
                    detected_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

                    # Query GPS cache for live coordinates
                    latest_loc = cache.get_latest_location()

                    event_data = {
                        "event": "CONFIRMED_FALL",
                        "stage": "6L",
                        "event_id": event_id,
                        "detected_at": detected_at,
                        "confidence": round(prob, 4),
                        "source": "ESP32_MPU6050",
                        "session_id": session_id,
                        "device_id": "ESP32-STAGE6L-UNIFIED",
                        "confirmed_fall": True,
                    }

                    print(f"\n>>> [CONFIRMED FALL DETECTED (Window {window_count})] <<<", flush=True)
                    print(f"  Confidence: {prob:.4f} | Event ID: {event_id}", flush=True)
                    print(f"  Location:   FixValid={latest_loc.get('fix_valid')} Lat={latest_loc.get('latitude')} Lon={latest_loc.get('longitude')}", flush=True)

                    if not dry_run:
                        success, st_msg, payload = publisher.publish_confirmed_fall(
                            event_data=event_data,
                            location=latest_loc,
                        )
                        if success:
                            mqtt_published += 1
                            print(f"  MQTT Publish: SUCCESS ({st_msg})", flush=True)
                        else:
                            mqtt_failures += 1
                            print(f"  MQTT Publish: FAILED ({st_msg})", flush=True)
                    else:
                        print("  MQTT Publish: SKIPPED (Dry-Run Active)", flush=True)

                if max_windows is not None and window_count >= max_windows:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Bridge stopped by user.")
    finally:
        if hb_sender is not None:
            hb_sender.stop()
        if ser is not None and hasattr(ser, "close"):
            ser.close()

    summary = {
        "session_id": session_id,
        "total_samples": total_samples,
        "total_windows": window_count,
        "confirmed_falls": confirmed_falls,
        "mqtt_published": mqtt_published,
        "mqtt_failures": mqtt_failures,
        "dry_run": dry_run,
        "mock_mode": mock_mode,
    }
    print("\n" + "=" * 70)
    print("STAGE 6L BRIDGE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Session ID:         {summary['session_id']}")
    print(f"Total IMU Samples:  {summary['total_samples']}")
    print(f"Total Windows:      {summary['total_windows']}")
    print(f"Confirmed Falls:    {summary['confirmed_falls']}")
    print(f"MQTT Published:     {summary['mqtt_published']}")
    print(f"MQTT Failures:      {summary['mqtt_failures']}")
    print(f"Dry Run:            {summary['dry_run']}")
    print(f"Mock Mode:          {summary['mock_mode']}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6L Integrated Hardware Inference Bridge")
    parser.add_argument("--port", type=str, default=os.environ.get("SERIAL_PORT", "COM22"), help="Serial port (default: COM22)")
    parser.add_argument("--baud", type=int, default=int(os.environ.get("SERIAL_BAUD", 460800)), help="Baud rate (default: 460800 or SERIAL_BAUD)")
    parser.add_argument("--broker", type=str, default=os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"), help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to process")
    parser.add_argument("--dry-run", action="store_true", help="Skip MQTT publishing")
    parser.add_argument("--mock", action="store_true", help="Run with mock sensor stream")
    parser.add_argument("--mock-scenario", type=str, default="stationary", choices=["stationary", "fall"], help="Mock data scenario")
    parser.add_argument("--backend", type=str, default=os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL), help="Backend URL for health heartbeats")

    args = parser.parse_args()

    run_stage_6L_hardware_bridge(
        port=args.port,
        baudrate=args.baud,
        broker_host=args.broker,
        broker_port=args.broker_port,
        session_id=args.session_id,
        max_windows=args.max_windows,
        dry_run=args.dry_run,
        mock_mode=args.mock,
        mock_scenario=args.mock_scenario,
        backend_url=args.backend,
    )
