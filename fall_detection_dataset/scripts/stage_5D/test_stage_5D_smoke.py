"""
Stage 5D Startup & Health Smoke Test Suite.

Verifies:
1. Backend /api/health endpoint
2. Dashboard web routes (/ and /dashboard)
3. Static assets (/static/css/dashboard.css, /static/js/dashboard.js)
4. Isolated SQLite database read/write/WAL operations
5. Frozen model deserialization and 100-feature signature check
6. Phase 6 End-to-end smoke cycle: Event -> Persistence -> API -> Acknowledge -> Restart Persistence
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from typing import Dict, Any

import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.models import FallEvent, FallEventStatus, get_utc_iso_now
from src.fall_event_backend.storage import FallEventRepository
from src.fall_event_backend.service import FallEventService
from src.fall_event_backend.integration_adapter import FallEventIntegrationAdapter
from src.fall_event_backend.api import create_app
from src.fall_confirmation.stage_4G_live_validator import (
    DEFAULT_MODEL_PATH,
    DEFAULT_FEATURE_ORDER_PATH
)


class TestStage5DSmoke(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "smoke_test_events.db")
        self.repo = FallEventRepository(self.db_path)
        self.service = FallEventService(self.repo)
        self.adapter = FallEventIntegrationAdapter(self.service)
        self.app = create_app(self.service)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_backend_health_check(self):
        """Smoke 1: GET /api/health returns healthy and expected metadata."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "fall_event_backend")
        self.assertEqual(data["emergency_notifications"], "DISABLED")

    def test_02_dashboard_routes(self):
        """Smoke 2: Root and /dashboard routes return HTTP 200 with HTML template."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        html_root = res_root.get_data(as_text=True)
        self.assertIn("Wearable Fall Detection System", html_root)
        self.assertIn("Active Event Telemetry", html_root)

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)

    def test_03_static_assets_accessible(self):
        """Smoke 3: Static CSS and JavaScript files load cleanly."""
        res_css = self.client.get("/static/css/dashboard.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("--color-confirmed", res_css.get_data(as_text=True))

        res_js = self.client.get("/static/js/dashboard.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("POLLING_INTERVAL_MS", res_js.get_data(as_text=True))

    def test_04_isolated_database_integrity(self):
        """Smoke 4: SQLite opens, initializes schema, and verifies WAL journal mode."""
        self.assertTrue(os.path.exists(self.db_path))
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fall_events'")
        table = cur.fetchone()
        self.assertIsNotNone(table)
        conn.close()

    def test_05_frozen_model_signature(self):
        """Smoke 5: Machine learning model loads cleanly and verifies 100-feature order."""
        self.assertTrue(os.path.exists(DEFAULT_MODEL_PATH))
        model = joblib.load(DEFAULT_MODEL_PATH)
        self.assertEqual(getattr(model, "n_features_in_", 100), 100)

        with open(DEFAULT_FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            features = json.load(f)
        self.assertEqual(len(features), 100)

    def test_06_final_end_to_end_smoke_workflow(self):
        """
        Smoke 6 (Phase 6): Complete end-to-end operational cycle:
        1. Emits controlled confirmed-fall event via adapter
        2. Verifies SQLite persistence
        3. Verifies API retrieval & details
        4. Acknowledges event via PATCH
        5. Verifies ACKNOWLEDGED status in DB
        6. Restarts backend (new repo/service)
        7. Verifies persisted event survives restart
        """
        # Step 1: Emit event
        sim_event = {
            "event_id": "EVT_SMOKE_PHASE6_01",
            "session_id": "session_smoke_demo",
            "timestamp": 8.25,
            "probability": 0.9542,
            "state": "CONFIRMED_FALL",
            "positive_count": 3,
            "pre_event_activity": "CONTROLLED_FALL_REPLAY"
        }
        evt, is_new = self.adapter.process_detector_event(sim_event)
        self.assertTrue(is_new)
        self.assertEqual(evt.status, "CONFIRMED")

        # Step 2 & 3: API retrieval
        res_get = self.client.get("/api/fall-events/EVT_SMOKE_PHASE6_01")
        self.assertEqual(res_get.status_code, 200)
        api_data = res_get.get_json()["event"]
        self.assertEqual(api_data["event_id"], "EVT_SMOKE_PHASE6_01")
        self.assertEqual(api_data["status"], "CONFIRMED")
        self.assertAlmostEqual(api_data["confidence"], 0.9542, places=4)

        # Step 4 & 5: Acknowledge
        res_ack = self.client.patch("/api/fall-events/EVT_SMOKE_PHASE6_01/acknowledge", json={
            "acknowledged_by": "Smoke_Operator"
        })
        self.assertEqual(res_ack.status_code, 200)
        ack_data = res_ack.get_json()["event"]
        self.assertEqual(ack_data["status"], "ACKNOWLEDGED")
        self.assertEqual(ack_data["acknowledged_by"], "Smoke_Operator")
        self.assertIsNotNone(ack_data["acknowledged_at"])

        # Step 6 & 7: Restart backend simulation
        restarted_repo = FallEventRepository(self.db_path)
        restarted_service = FallEventService(restarted_repo)
        restarted_app = create_app(restarted_service)
        restarted_client = restarted_app.test_client()

        res_post_restart = restarted_client.get("/api/fall-events/EVT_SMOKE_PHASE6_01")
        self.assertEqual(res_post_restart.status_code, 200)
        survived_data = res_post_restart.get_json()["event"]
        self.assertEqual(survived_data["event_id"], "EVT_SMOKE_PHASE6_01")
        self.assertEqual(survived_data["status"], "ACKNOWLEDGED")
        self.assertEqual(survived_data["acknowledged_by"], "Smoke_Operator")


def run_smoke_tests() -> Dict[str, Any]:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStage5DSmoke)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {
        "total_tests": result.testsRun,
        "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "was_successful": result.wasSuccessful()
    }


if __name__ == "__main__":
    res = run_smoke_tests()
    print("\nSTAGE 5D SMOKE TEST SUMMARY:", res)
    sys.exit(0 if res["was_successful"] else 1)
