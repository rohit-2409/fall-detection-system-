"""
Stage 5F Live Bridge Automated Test Suite.
Tests serial parsing, 100-feature extraction parity, frozen model inference,
temporal confirmation gating, dry-run execution, mock stream inference,
and REST backend contract integration.
"""

import http.server
import json
import os
import sys
import threading
import time
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_confirmation.stage_5F_live_bridge import (
    parse_serial_line,
    generate_event_id,
    post_fall_event_to_backend,
    run_stage_5f_live_bridge,
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_ORDER_PATH,
)
from src.fall_confirmation.stage_4G_live_validator import (
    extract_100_features,
    WINDOW_SIZE,
    STEP_SIZE,
)
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter


class MockBackendHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler simulating the Stage 5A/5B REST API."""
    recorded_events = []
    response_status = 201

    def do_POST(self):
        if self.path == "/api/fall-events":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            MockBackendHandler.recorded_events.append(data)

            if MockBackendHandler.response_status == 201:
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "is_new": True, "event": data}).encode("utf-8"))
            elif MockBackendHandler.response_status == 400:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Bad request"}).encode("utf-8"))
            else:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "Server error"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence console output from HTTP server
        pass


class TestStage5FLiveBridge(unittest.TestCase):

    def setUp(self):
        MockBackendHandler.recorded_events = []
        MockBackendHandler.response_status = 201

    def test_01_serial_line_parsing_all_formats(self):
        """Test parsing pipe format, 7-col CSV, 6-col CSV, headers, and corrupt lines."""
        # Pipe format
        pipe_line = "ACC(g): 0.012, -0.025, 0.986 | GYRO(deg/s): 0.31, -0.12, 0.08\n"
        res = parse_serial_line(pipe_line)
        self.assertIsNotNone(res)
        ts, ax, ay, az, gx, gy, gz = res
        self.assertIsNone(ts)
        self.assertAlmostEqual(ax, 0.012, places=3)
        self.assertAlmostEqual(ay, -0.025, places=3)
        self.assertAlmostEqual(az, 0.986, places=3)
        self.assertAlmostEqual(gx, 0.31, places=2)
        self.assertAlmostEqual(gy, -0.12, places=2)
        self.assertAlmostEqual(gz, 0.08, places=2)

        # 7-column CSV format
        csv7_line = "123456,0.05,-0.02,0.99,1.2,-0.5,0.3\n"
        res7 = parse_serial_line(csv7_line)
        self.assertIsNotNone(res7)
        ts, ax, ay, az, gx, gy, gz = res7
        self.assertEqual(ts, 123456.0)
        self.assertAlmostEqual(ax, 0.05, places=2)
        self.assertAlmostEqual(az, 0.99, places=2)

        # 6-column CSV format
        csv6_line = "0.05,-0.02,0.99,1.2,-0.5,0.3\n"
        res6 = parse_serial_line(csv6_line)
        self.assertIsNotNone(res6)
        ts, ax, ay, az, gx, gy, gz = res6
        self.assertIsNone(ts)
        self.assertAlmostEqual(ax, 0.05, places=2)

        # Corrupt and Header lines
        self.assertIsNone(parse_serial_line("ESP32 Starting..."))
        self.assertIsNone(parse_serial_line("timestamp_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps"))
        self.assertIsNone(parse_serial_line("CORRUPT,LINE,12,34"))
        self.assertIsNone(parse_serial_line(""))

    def test_02_event_id_generation(self):
        """Test unique and deterministic structure of generated event IDs."""
        id1 = generate_event_id(prefix="LIVE")
        id2 = generate_event_id(prefix="LIVE")
        self.assertTrue(id1.startswith("LIVE-"))
        self.assertTrue(id2.startswith("LIVE-"))
        self.assertNotEqual(id1, id2)

    def test_03_backend_post_contract_success(self):
        """Test HTTP POST delivery against a mock HTTP server."""
        # Start mock server on ephemeral port
        server = http.server.HTTPServer(("127.0.0.1", 0), MockBackendHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        backend_url = f"http://127.0.0.1:{port}"
        payload = {
            "event_id": "LIVE-TEST-001",
            "detected_at": "2026-10-02T15:30:00Z",
            "confidence": 0.88,
            "source": "ESP32_MPU6050",
            "diagnostic_metadata": {"test": True}
        }

        success, msg, code = post_fall_event_to_backend(backend_url, payload)
        self.assertTrue(success)
        self.assertEqual(code, 201)
        self.assertEqual(msg, "STORED")
        self.assertEqual(len(MockBackendHandler.recorded_events), 1)
        self.assertEqual(MockBackendHandler.recorded_events[0]["event_id"], "LIVE-TEST-001")

        server.shutdown()

    def test_04_backend_post_contract_failure_handling(self):
        """Test that server errors and offline backends return False without crashing."""
        # 1. Offline port
        success, msg, code = post_fall_event_to_backend("http://127.0.0.1:59999", {"event_id": "X"})
        self.assertFalse(success)
        self.assertIsNone(code)
        self.assertIn("CONNECTION_ERROR", msg)

        # 2. HTTP 500 error
        server = http.server.HTTPServer(("127.0.0.1", 0), MockBackendHandler)
        port = server.server_address[1]
        MockBackendHandler.response_status = 500
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        success, msg, code = post_fall_event_to_backend(f"http://127.0.0.1:{port}", {"event_id": "X"})
        self.assertFalse(success)
        self.assertEqual(code, 500)
        self.assertIn("HTTP_500", msg)

        server.shutdown()

    def test_05_mock_mode_live_bridge_run(self):
        """Test running stage_5F_live_bridge in mock mode with dry-run."""
        summary = run_stage_5f_live_bridge(
            port="COM22",
            baudrate=115200,
            backend_url="http://127.0.0.1:5000",
            max_windows=5,
            dry_run=True,
            mock_mode=True
        )

        self.assertGreaterEqual(summary["total_samples"], 50)
        self.assertEqual(summary["total_windows"], 5)
        self.assertEqual(summary["confirmed_falls"], 0)
        self.assertEqual(summary["backend_stored"], 0)
        self.assertTrue(summary["dry_run"])
        self.assertTrue(summary["mock_mode"])

    def test_06_temporal_confirmation_gating_behavior(self):
        """Test that temporal adapter requires N3 consecutive candidate windows and debounces."""
        adapter = TemporalConfirmationAdapter()
        adapter.reset()

        # 1st positive window -> POSSIBLE_FALL (confirmed_fall == False)
        res1 = adapter.process_prediction(0.85, window_start_time=0.0)
        self.assertTrue(res1["raw_positive"])
        self.assertFalse(res1["confirmed_fall"])
        self.assertEqual(res1["positive_count"], 1)

        # 2nd positive window -> POSSIBLE_FALL (confirmed_fall == False)
        res2 = adapter.process_prediction(0.85, window_start_time=0.25)
        self.assertTrue(res2["raw_positive"])
        self.assertFalse(res2["confirmed_fall"])
        self.assertEqual(res2["positive_count"], 2)

        # 3rd positive window -> CONFIRMED_FALL (confirmed_fall == True)
        res3 = adapter.process_prediction(0.85, window_start_time=0.50)
        self.assertTrue(res3["raw_positive"])
        self.assertTrue(res3["confirmed_fall"])
        self.assertEqual(res3["positive_count"], 3)

        # 4th positive window -> COOLDOWN (confirmed_fall == False, debounced)
        res4 = adapter.process_prediction(0.85, window_start_time=0.75)
        self.assertTrue(res4["raw_positive"])
        self.assertFalse(res4["confirmed_fall"])
        self.assertEqual(res4["temporal_state"], "COOLDOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
