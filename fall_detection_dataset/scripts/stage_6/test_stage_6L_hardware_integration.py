"""
Stage 6L Automated Test Suite: Hardware Integration & End-to-End Components.
Standalone Fall Detection System.

Hermetic, non-destructive test suite validating:
  1. GPS Cache: Valid fix ingestion and storage.
  2. GPS Cache: Invalid coordinate rejection (-90..90, -180..180, non-zero).
  3. GPS Cache: Malformed JSON and payload error resilience.
  4. GPS Cache: Stale GPS fix detection based on age threshold.
  5. GPS Cache: Latest valid fix query schema compliance.
  6. GPS Cache: Direct programmatic injection (update_location).
  7. Serial Parser: Structured JSON Lines IMU format parsing.
  8. Serial Parser: Structured JSON Lines GPS format forwarding to cache.
  9. Serial Parser: Legacy pipe format parsing (Stage 5F backward compatibility).
 10. Serial Parser: Legacy 7-column CSV parsing.
 11. Serial Parser: Legacy 6-column CSV parsing.
 12. Serial Parser: Malformed, corrupt, and header line rejection.
 13. Hardware Bridge: Mock sensor stationary stream produces ZERO confirmed falls.
 14. Hardware Bridge: Mock sensor dynamic fall stream produces confirmed fall.
 15. Hardware Bridge: Live GPS coordinates attached to confirmed fall event.
 16. Hardware Bridge: Safe null location attached when GPS fix is missing/stale.
 17. Hardware Bridge: Gating on confirmed_fall == True enforced before MQTT publish.
 18. Hardware Bridge: Dry-run mode processes stream without MQTT network publish.
 19. Configuration: Environment variables override broker host and port.
 20. Orchestrator: Preflight TCP check, backend health check, and port enumeration.
"""

import json
import os
import sys
import time
import unittest
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_bridge import (
    MockSensorStream,
    parse_serial_stream_line,
    run_stage_6L_hardware_bridge,
)
from scripts.stage_6.run_stage_6L_hardware_test import (
    check_backend_health,
    check_serial_port_available,
    check_tcp_port,
    run_orchestration_preflight,
)
from src.fall_confirmation.stage_6F_mqtt_fall_publisher import MqttFallPublisher
from src.fall_confirmation.stage_6L_gps_cache import (
    GpsTelemetryCache,
    get_global_gps_cache,
    reset_global_gps_cache,
)


class MockMqttPublisher:
    """Mock publisher capturing dispatched payloads for testing without network broker."""

    def __init__(self):
        self.published_events = []
        self.is_connected = True

    def publish_confirmed_fall(
        self,
        event_data: Dict[str, Any],
        location: Optional[Dict[str, Any]] = None,
        device_id: Optional[str] = None,
    ):
        payload = dict(event_data)
        payload["location"] = location
        self.published_events.append(payload)
        return True, "MOCK_PUBLISHED", payload


class TestStage6LHardwareIntegration(unittest.TestCase):
    """Hermetic unit and integration tests for Stage 6L components."""

    def setUp(self):
        self.gps_cache = GpsTelemetryCache(max_age_seconds=1.0)
        reset_global_gps_cache(self.gps_cache)

    def tearDown(self):
        self.gps_cache.stop()
        reset_global_gps_cache(None)

    # -------------------------------------------------------------------------
    # 1. GPS Cache Tests
    # -------------------------------------------------------------------------

    def test_01_gps_cache_valid_fix_ingestion(self):
        """TEST 01: Valid GPS payload is parsed and stored."""
        payload = {
            "valid": True,
            "latitude": 22.728455,
            "longitude": 75.869210,
            "satellites": 9,
            "hdop": 0.95,
        }
        res = self.gps_cache.process_payload(payload)
        self.assertTrue(res)
        loc = self.gps_cache.get_latest_location()
        self.assertTrue(loc["fix_valid"])
        self.assertEqual(loc["latitude"], 22.728455)
        self.assertEqual(loc["longitude"], 75.869210)
        self.assertEqual(loc["satellites"], 9)
        self.assertEqual(loc["hdop"], 0.95)

    def test_02_gps_cache_invalid_coordinates_rejected(self):
        """TEST 02: Out-of-bounds or zero coordinates are rejected."""
        self.assertFalse(self.gps_cache.process_payload({"valid": True, "latitude": 95.0, "longitude": 75.0}))
        self.assertFalse(self.gps_cache.process_payload({"valid": True, "latitude": 22.0, "longitude": 185.0}))
        self.assertFalse(self.gps_cache.process_payload({"valid": True, "latitude": 0.0, "longitude": 0.0}))
        loc = self.gps_cache.get_latest_location()
        self.assertFalse(loc["fix_valid"])
        self.assertIsNone(loc["latitude"])

    def test_03_gps_cache_malformed_payload_resilience(self):
        """TEST 03: Non-dictionary or missing coordinate payloads do not raise errors."""
        self.assertFalse(self.gps_cache.process_payload({}))
        self.assertFalse(self.gps_cache.process_payload({"valid": True, "latitude": "not_a_number"}))
        loc = self.gps_cache.get_latest_location()
        self.assertFalse(loc["fix_valid"])

    def test_04_gps_cache_stale_fix_detection(self):
        """TEST 04: GPS fix older than max_age_seconds is marked stale."""
        cache_short = GpsTelemetryCache(max_age_seconds=0.1)
        cache_short.update_location(22.72, 75.86, satellites=6, valid=True)
        loc1 = cache_short.get_latest_location()
        self.assertTrue(loc1["fix_valid"])

        # Wait past max_age_seconds
        time.sleep(0.15)
        loc2 = cache_short.get_latest_location()
        self.assertFalse(loc2["fix_valid"])
        self.assertIsNone(loc2["latitude"])
        cache_short.stop()

    def test_05_gps_cache_latest_valid_fix_schema(self):
        """TEST 05: Location format matches Stage 6F schema."""
        loc_empty = self.gps_cache.get_latest_location()
        expected_keys = {"fix_valid", "latitude", "longitude", "satellites", "hdop", "timestamp", "age_seconds"}
        self.assertEqual(set(loc_empty.keys()), expected_keys)
        self.assertFalse(loc_empty["fix_valid"])

    def test_06_gps_cache_direct_injection(self):
        """TEST 06: Programmatic update_location updates internal state."""
        ok = self.gps_cache.update_location(12.9716, 77.5946, satellites=12, hdop=0.8, valid=True)
        self.assertTrue(ok)
        loc = self.gps_cache.get_latest_location()
        self.assertTrue(loc["fix_valid"])
        self.assertEqual(loc["latitude"], 12.9716)

    # -------------------------------------------------------------------------
    # 2. Serial Parser Tests
    # -------------------------------------------------------------------------

    def test_07_parse_json_lines_imu(self):
        """TEST 07: JSON Lines IMU format is correctly parsed."""
        line = '{"type":"imu","timestamp_ms":12345,"ax_g":0.012,"ay_g":-0.024,"az_g":0.985,"gx_dps":0.31,"gy_dps":-0.12,"gz_dps":0.08}'
        res = parse_serial_stream_line(line)
        self.assertIsNotNone(res)
        ts, ax, ay, az, gx, gy, gz = res
        self.assertEqual(ts, 12345.0)
        self.assertAlmostEqual(ax, 0.012)
        self.assertAlmostEqual(ay, -0.024)
        self.assertAlmostEqual(az, 0.985)

    def test_08_parse_json_lines_gps(self):
        """TEST 08: JSON Lines GPS line forwards to GPS cache and returns None."""
        line = '{"type":"gps","timestamp_ms":54321,"valid":true,"latitude":22.728,"longitude":75.869,"satellites":7}'
        res = parse_serial_stream_line(line, gps_cache=self.gps_cache)
        self.assertIsNone(res)
        loc = self.gps_cache.get_latest_location()
        self.assertTrue(loc["fix_valid"])
        self.assertEqual(loc["latitude"], 22.728)

    def test_09_parse_legacy_pipe_format(self):
        """TEST 09: Legacy Stage 5F pipe format is parsed accurately."""
        line = "ACC(g): 0.010, -0.020, 0.980 | GYRO(deg/s): 0.20, -0.10, 0.05"
        res = parse_serial_stream_line(line)
        self.assertIsNotNone(res)
        ts, ax, ay, az, gx, gy, gz = res
        self.assertIsNone(ts)
        self.assertAlmostEqual(ax, 0.010)
        self.assertAlmostEqual(az, 0.980)

    def test_10_parse_legacy_7_column_csv(self):
        """TEST 10: Standard 7-column CSV (timestamp,ax,ay,az,gx,gy,gz) is parsed."""
        line = "1000, 0.015, -0.025, 0.982, 0.22, -0.11, 0.06"
        res = parse_serial_stream_line(line)
        self.assertIsNotNone(res)
        ts, ax, ay, az, gx, gy, gz = res
        self.assertEqual(ts, 1000.0)
        self.assertAlmostEqual(az, 0.982)

    def test_11_parse_legacy_6_column_csv(self):
        """TEST 11: Standard 6-column CSV (ax,ay,az,gx,gy,gz) is parsed."""
        line = "0.015, -0.025, 0.982, 0.22, -0.11, 0.06"
        res = parse_serial_stream_line(line)
        self.assertIsNotNone(res)
        ts, ax, ay, az, gx, gy, gz = res
        self.assertIsNone(ts)
        self.assertAlmostEqual(az, 0.982)

    def test_12_parse_rejects_malformed_and_headers(self):
        """TEST 12: Headers, boot logs, and invalid text are ignored."""
        self.assertIsNone(parse_serial_stream_line(""))
        self.assertIsNone(parse_serial_stream_line("   "))
        self.assertIsNone(parse_serial_stream_line("STAGE 6C: ESP32 Wi-Fi & MQTT Client Starting"))
        self.assertIsNone(parse_serial_stream_line("timestamp,AccX,AccY,AccZ,GyrX,GyrY,GyrZ"))
        self.assertIsNone(parse_serial_stream_line("{broken_json: true}"))

    # -------------------------------------------------------------------------
    # 3. Hardware Bridge Tests
    # -------------------------------------------------------------------------

    def test_13_hardware_bridge_stationary_zero_falls(self):
        """TEST 13: Stationary sensor stream produces ZERO fall events."""
        mock_pub = MockMqttPublisher()
        res = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="stationary",
            max_windows=4,
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertEqual(res["confirmed_falls"], 0)
        self.assertEqual(len(mock_pub.published_events), 0)

    def test_14_hardware_bridge_dynamic_fall_detected(self):
        """TEST 14: Dynamic fall sensor stream detects confirmed fall and invokes publisher."""
        mock_pub = MockMqttPublisher()
        res = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=6,
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertGreaterEqual(res["confirmed_falls"], 1)
        self.assertGreaterEqual(len(mock_pub.published_events), 1)
        evt = mock_pub.published_events[0]
        self.assertEqual(evt["event"], "CONFIRMED_FALL")
        self.assertEqual(evt["stage"], "6L")
        self.assertTrue(evt["confirmed_fall"])

    def test_15_hardware_bridge_live_gps_attached(self):
        """TEST 15: Valid GPS in cache is cleanly attached to confirmed fall event."""
        self.gps_cache.update_location(22.728455, 75.869210, satellites=9, hdop=0.9, valid=True)
        mock_pub = MockMqttPublisher()
        run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=5,
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertGreaterEqual(len(mock_pub.published_events), 1)
        loc = mock_pub.published_events[0]["location"]
        self.assertTrue(loc["fix_valid"])
        self.assertEqual(loc["latitude"], 22.728455)
        self.assertEqual(loc["longitude"], 75.869210)

    def test_16_hardware_bridge_missing_gps_safe_null(self):
        """TEST 16: When GPS is missing, safe explicit null location schema is attached."""
        mock_pub = MockMqttPublisher()
        run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=5,
            dry_run=False,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertGreaterEqual(len(mock_pub.published_events), 1)
        loc = mock_pub.published_events[0]["location"]
        self.assertFalse(loc["fix_valid"])
        self.assertIsNone(loc["latitude"])
        self.assertIsNone(loc["longitude"])

    def test_17_hardware_bridge_dry_run_skips_publishing(self):
        """TEST 17: Dry run processes stream but skips MQTT publish calls."""
        mock_pub = MockMqttPublisher()
        res = run_stage_6L_hardware_bridge(
            mock_mode=True,
            mock_scenario="fall",
            max_windows=5,
            dry_run=True,
            gps_cache=self.gps_cache,
            mqtt_publisher=mock_pub,
        )
        self.assertGreaterEqual(res["confirmed_falls"], 1)
        self.assertEqual(len(mock_pub.published_events), 0)

    # -------------------------------------------------------------------------
    # 4. Configuration & Orchestrator Tests
    # -------------------------------------------------------------------------

    def test_18_environment_driven_configuration(self):
        """TEST 18: Environment variables set custom broker host and port."""
        os.environ["MQTT_BROKER_HOST"] = "192.168.1.50"
        os.environ["MQTT_BROKER_PORT"] = "1884"
        custom_cache = GpsTelemetryCache()
        self.assertEqual(custom_cache.broker_host, "192.168.1.50")
        self.assertEqual(custom_cache.broker_port, 1884)
        custom_cache.stop()
        # Reset env
        os.environ["MQTT_BROKER_HOST"] = "127.0.0.1"
        os.environ["MQTT_BROKER_PORT"] = "1883"

    def test_19_orchestrator_preflight_checks(self):
        """TEST 19: Orchestrator preflight evaluates TCP ports and backend status."""
        # Check closed port
        port_open = check_tcp_port("127.0.0.1", 59999, timeout=0.1)
        self.assertFalse(port_open)

        # Check backend unreachable on dummy port
        be_ok, msg, _ = check_backend_health("http://127.0.0.1:59999", timeout=0.1)
        self.assertFalse(be_ok)
        self.assertIn("UNREACHABLE", msg)

        # Check port listing
        _, avail = check_serial_port_available("COM_NONEXISTENT_99")
        self.assertIsInstance(avail, list)

    def test_20_orchestrator_mock_mode_preflight(self):
        """TEST 20: Preflight completes in mock mode without physical serial port."""
        preflight = run_orchestration_preflight(
            broker_host="127.0.0.1",
            broker_port=1883,
            backend_url="http://127.0.0.1:5000",
            serial_port="COM22",
            mock_mode=True,
        )
        self.assertIn("serial_port", preflight)
        self.assertTrue(preflight["serial_port"])


if __name__ == "__main__":
    unittest.main()
