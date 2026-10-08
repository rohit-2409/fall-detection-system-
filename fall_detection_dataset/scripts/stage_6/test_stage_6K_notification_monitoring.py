"""
Stage 6K Test Suite: Notification Monitoring & Reliability Dashboard.
Standalone Fall Detection System.

Hermetic, non-destructive test suite validating:
1. Monitoring endpoint returns HTTP 200.
2. Empty database returns zero counts.
3. SENT count is correct.
4. QUEUED count is correct.
5. RETRY_WAIT count is correct.
6. SENDING count is correct.
7. FAILED count is correct.
8. DEAD_LETTER count is correct.
9. DUPLICATE_SUPPRESSED count is correct.
10. DISABLED count is correct.
11. Total count is correct.
12. Recent deliveries are returned.
13. Status filtering works.
14. Event_id filtering works.
15. Limit and offset are respected.
16. Masked recipient remains masked.
17. Secrets never appear in API responses.
18. Monitoring queries do NOT modify database (read-only enforcement).
19. Malformed notification data / missing tables handled safely.
20. Notification detail endpoint GET /api/notifications/events/<event_id> works.
21. Notification dashboard HTML route GET /notifications works.
22. Full pipeline persistence & monitoring telemetry integration.

All tests run on isolated temporary databases with fake/mock providers only.
"""

import datetime
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.models import FallEvent, FallEventStatus
from src.fall_event_backend.storage import FallEventRepository
from src.notifications.notification_config import NotificationConfig, mask_phone_number
from src.notifications.notification_listener import (
    NotifyingFallEventService,
    create_notifying_app,
    register_notification_monitoring_routes,
)
from src.notifications.notification_monitor import NotificationMonitorService
from src.notifications.whatsapp_provider import FakeWhatsAppProvider
from src.notifications.whatsapp_service import (
    NotificationStatus,
    WhatsAppNotificationService,
    reset_global_notification_service,
)


class TestStage6KNotificationMonitoring(unittest.TestCase):
    """Hermetic test suite for Stage 6K notification observability & dashboard telemetry."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_monitoring.db")

        self.config = NotificationConfig(
            provider="fake",
            account_sid="AC_test_account_sid_secret",
            auth_token="auth_token_super_secret_xyz123",
            whatsapp_from="whatsapp:+14155238886",
            whatsapp_recipient="whatsapp:+919876543210",
            timeout_seconds=2.0,
            db_path=self.db_path,
            max_attempts=3,
            retry_base_seconds=0.1,
            retry_max_seconds=1.0,
        )
        self.provider = FakeWhatsAppProvider(mode="success")
        self.notif_service = WhatsAppNotificationService(
            config=self.config,
            provider=self.provider,
            db_path=self.db_path,
        )
        self.repo = FallEventRepository(self.db_path)
        self.notifying_service = NotifyingFallEventService(
            repository=self.repo,
            notification_service=self.notif_service,
        )
        self.monitor_service = NotificationMonitorService(
            db_path=self.db_path,
            notification_service=self.notif_service,
        )
        self.app = create_notifying_app(
            service=self.notifying_service,
            notification_service=self.notif_service,
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.notif_service.shutdown()
        reset_global_notification_service(None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_raw_delivery(
        self,
        event_id: str,
        status: str,
        provider: str = "fake",
        recipient: str = "whatsapp:+919876543210",
        attempt_count: int = 1,
        max_attempts: int = 5,
        last_attempt_at: str = "2026-10-07T04:00:00Z",
        next_attempt_at: str = None,
        dispatched_at: str = None,
        error_message: str = None,
        last_error: str = None,
        created_at: str = "2026-10-07T04:00:00Z",
        updated_at: str = "2026-10-07T04:00:00Z",
    ):
        """Helper to insert raw records into notification_deliveries for monitoring tests."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notification_deliveries (
                    event_id, status, provider, recipient_masked,
                    attempt_count, max_attempts, last_attempt_at, next_attempt_at,
                    dispatched_at, error_message, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
                (
                    event_id,
                    status,
                    provider,
                    recipient,
                    attempt_count,
                    max_attempts,
                    last_attempt_at,
                    next_attempt_at,
                    dispatched_at,
                    error_message,
                    last_error,
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()

    def test_01_monitoring_endpoint_returns_200(self):
        """TEST 01: Monitoring endpoint returns HTTP 200 with required structure."""
        resp = self.client.get("/api/notifications/monitoring")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("summary", data)
        self.assertIn("queue", data)
        self.assertIn("provider", data)
        self.assertIn("recent_activity", data)

    def test_02_empty_database_returns_zero_counts(self):
        """TEST 02: Empty database returns zero counts for all statuses."""
        resp = self.client.get("/api/notifications/monitoring")
        self.assertEqual(resp.status_code, 200)
        summary = resp.get_json()["summary"]
        expected_keys = [
            "total", "sent", "queued", "sending", "retry_wait",
            "failed", "dead_letter", "duplicate_suppressed", "disabled"
        ]
        for key in expected_keys:
            self.assertEqual(summary.get(key), 0, f"Expected {key} to be 0")

    def test_03_sent_count_is_correct(self):
        """TEST 03: SENT count is accurate in monitoring summary."""
        for i in range(3):
            self._insert_raw_delivery(f"evt_sent_{i}", "SENT", dispatched_at="2026-10-07T04:01:00Z")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["sent"], 3)
        self.assertEqual(summary["total"], 3)

    def test_04_queued_count_is_correct(self):
        """TEST 04: QUEUED count is accurate in monitoring summary."""
        for i in range(2):
            self._insert_raw_delivery(f"evt_queued_{i}", "QUEUED")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["queued"], 2)

    def test_05_retry_wait_count_is_correct(self):
        """TEST 05: RETRY_WAIT count is accurate in monitoring summary."""
        for i in range(4):
            self._insert_raw_delivery(
                f"evt_retry_{i}",
                "RETRY_WAIT",
                next_attempt_at="2026-10-07T04:05:00Z",
                attempt_count=2,
            )
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["retry_wait"], 4)

    def test_06_sending_count_is_correct(self):
        """TEST 06: SENDING count is accurate in monitoring summary."""
        self._insert_raw_delivery("evt_sending_1", "SENDING", attempt_count=1)
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["sending"], 1)

    def test_07_failed_count_is_correct(self):
        """TEST 07: FAILED count is accurate in monitoring summary."""
        for i in range(2):
            self._insert_raw_delivery(f"evt_failed_{i}", "FAILED", error_message="HTTP 500")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["failed"], 2)

    def test_08_dead_letter_count_is_correct(self):
        """TEST 08: DEAD_LETTER count is accurate in monitoring summary."""
        for i in range(3):
            self._insert_raw_delivery(f"evt_dlq_{i}", "DEAD_LETTER", attempt_count=5)
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["dead_letter"], 3)

    def test_09_duplicate_suppressed_count_is_correct(self):
        """TEST 09: DUPLICATE_SUPPRESSED count is accurate in monitoring summary."""
        for i in range(5):
            self._insert_raw_delivery(f"evt_dup_{i}", "DUPLICATE_SUPPRESSED")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["duplicate_suppressed"], 5)

    def test_10_disabled_count_is_correct(self):
        """TEST 10: DISABLED count is accurate in monitoring summary."""
        for i in range(2):
            self._insert_raw_delivery(f"evt_dis_{i}", "DISABLED")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["disabled"], 2)

    def test_11_total_count_is_correct(self):
        """TEST 11: Total count is the correct sum of distinct statuses."""
        self._insert_raw_delivery("evt_t1", "SENT")
        self._insert_raw_delivery("evt_t2", "SENT")
        self._insert_raw_delivery("evt_t3", "RETRY_WAIT")
        self._insert_raw_delivery("evt_t4", "FAILED")
        self._insert_raw_delivery("evt_t5", "DEAD_LETTER")
        summary = self.monitor_service.get_summary_counts()
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["sent"], 2)
        self.assertEqual(summary["retry_wait"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["dead_letter"], 1)

    def test_12_recent_deliveries_are_returned(self):
        """TEST 12: Recent deliveries returned in reverse chronological order with limit."""
        for i in range(15):
            ts = f"2026-10-07T04:{i:02d}:00Z"
            self._insert_raw_delivery(f"evt_recent_{i}", "SENT", created_at=ts)

        resp = self.client.get("/api/notifications/monitoring")
        self.assertEqual(resp.status_code, 200)
        recent = resp.get_json()["recent_activity"]
        self.assertEqual(len(recent), 10)
        # Most recent timestamp should be first
        self.assertEqual(recent[0]["event_id"], "evt_recent_14")

    def test_13_status_filtering_works(self):
        """TEST 13: Delivery list filters accurately by status."""
        self._insert_raw_delivery("evt_s1", "SENT")
        self._insert_raw_delivery("evt_s2", "RETRY_WAIT")
        self._insert_raw_delivery("evt_s3", "SENT")
        self._insert_raw_delivery("evt_s4", "DEAD_LETTER")

        resp = self.client.get("/api/notifications/deliveries?status=SENT")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["items"]), 2)
        for item in data["items"]:
            self.assertEqual(item["status"], "SENT")

    def test_14_event_id_filtering_works(self):
        """TEST 14: Delivery list filters accurately by event_id."""
        self._insert_raw_delivery("evt_target_100", "SENT")
        self._insert_raw_delivery("evt_other_200", "SENT")

        resp = self.client.get("/api/notifications/deliveries?event_id=evt_target_100")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(len(data["items"]), 1)
        self.assertEqual(data["items"][0]["event_id"], "evt_target_100")

    def test_15_limit_and_offset_are_respected(self):
        """TEST 15: Pagination parameters limit and offset are respected."""
        for i in range(25):
            self._insert_raw_delivery(f"evt_page_{i:02d}", "SENT", created_at=f"2026-10-07T04:{i:02d}:00Z")

        resp = self.client.get("/api/notifications/deliveries?limit=5&offset=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["total"], 25)
        self.assertEqual(len(data["items"]), 5)
        self.assertEqual(data["limit"], 5)
        self.assertEqual(data["offset"], 10)

    def test_16_masked_recipient_remains_masked(self):
        """TEST 16: Masked recipient remains masked and raw numbers are never exposed."""
        raw_number = "whatsapp:+919876543210"
        self._insert_raw_delivery("evt_mask_check", "SENT", recipient=raw_number)

        resp = self.client.get("/api/notifications/deliveries?event_id=evt_mask_check")
        self.assertEqual(resp.status_code, 200)
        item = resp.get_json()["items"][0]
        # Should be masked (e.g. whatsapp:+91*****3210)
        self.assertNotIn("987654", item["recipient_masked"])
        self.assertIn("*****", item["recipient_masked"])

    def test_17_secrets_never_appear_in_api_responses(self):
        """TEST 17: Secrets and credentials never appear anywhere in API responses."""
        self._insert_raw_delivery("evt_sec_test", "FAILED", error_message="Internal Error")

        secret_strings = [
            "AC_test_account_sid_secret",
            "auth_token_super_secret_xyz123",
            "super_secret",
            "account_sid",
            "auth_token",
        ]

        # Check monitoring snapshot
        resp_mon = self.client.get("/api/notifications/monitoring")
        text_mon = resp_mon.get_data(as_text=True)
        for s in secret_strings:
            self.assertNotIn(s, text_mon)

        # Check deliveries endpoint
        resp_del = self.client.get("/api/notifications/deliveries")
        text_del = resp_del.get_data(as_text=True)
        for s in secret_strings:
            self.assertNotIn(s, text_del)

        # Check detail endpoint
        resp_evt = self.client.get("/api/notifications/events/evt_sec_test")
        text_evt = resp_evt.get_data(as_text=True)
        for s in secret_strings:
            self.assertNotIn(s, text_evt)

    def test_18_monitoring_queries_do_not_modify_database(self):
        """TEST 18: Monitoring queries are strictly read-only and cannot mutate state."""
        self._insert_raw_delivery("evt_ro_test", "SENT")

        # Snapshot table state before queries
        with sqlite3.connect(self.db_path) as conn:
            before_deliveries = conn.execute("SELECT * FROM notification_deliveries").fetchall()
            before_events = conn.execute("SELECT * FROM fall_events").fetchall()

        # Run multiple read-only queries
        self.client.get("/api/notifications/monitoring")
        self.client.get("/api/notifications/deliveries")
        self.client.get("/api/notifications/events/evt_ro_test")

        # Snapshot table state after queries
        with sqlite3.connect(self.db_path) as conn:
            after_deliveries = conn.execute("SELECT * FROM notification_deliveries").fetchall()
            after_events = conn.execute("SELECT * FROM fall_events").fetchall()

        self.assertEqual(before_deliveries, after_deliveries)
        self.assertEqual(before_events, after_events)

        # Verify PRAGMA query_only = ON prevents writes on monitoring connection
        mon_conn = self.monitor_service._get_connection()
        with self.assertRaises(sqlite3.OperationalError):
            mon_conn.execute("INSERT INTO notification_deliveries (event_id, status) VALUES ('x', 'y');")
        mon_conn.close()

    def test_19_malformed_notification_data_or_missing_tables_handled_safely(self):
        """TEST 19: Missing tables or non-existent databases do not raise 500 errors."""
        empty_db_path = os.path.join(self.temp_dir, "nonexistent.db")
        standalone_monitor = NotificationMonitorService(db_path=empty_db_path)

        counts = standalone_monitor.get_summary_counts()
        self.assertEqual(counts["total"], 0)
        self.assertEqual(counts["sent"], 0)

        deliveries, total = standalone_monitor.get_deliveries()
        self.assertEqual(deliveries, [])
        self.assertEqual(total, 0)

        notif = standalone_monitor.get_event_notification("nonexistent_id")
        self.assertIsNone(notif)

    def test_20_notification_detail_endpoint_works(self):
        """TEST 20: Notification detail endpoint returns delivery status or 404 cleanly."""
        self._insert_raw_delivery(
            "evt_detail_found",
            "RETRY_WAIT",
            attempt_count=2,
            next_attempt_at="2026-10-07T04:10:00Z",
            last_error="HTTP 429",
        )

        # Existing record returns 200 with data
        resp = self.client.get("/api/notifications/events/evt_detail_found")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["event_id"], "evt_detail_found")
        notif = data["notification"]
        self.assertEqual(notif["status"], "RETRY_WAIT")
        self.assertEqual(notif["attempt_count"], 2)
        self.assertEqual(notif["last_error"], "HTTP 429")

        # Non-existing record returns 404
        resp_404 = self.client.get("/api/notifications/events/evt_detail_missing")
        self.assertEqual(resp_404.status_code, 404)
        data_404 = resp_404.get_json()
        self.assertEqual(data_404["status"], "not_found")

    def test_21_dashboard_ui_endpoint_returns_200(self):
        """TEST 21: Notification dashboard UI route /notifications returns HTTP 200."""
        resp = self.client.get("/notifications")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Notification Reliability Dashboard", html)
        self.assertIn("notifications_monitor.js", html)
        self.assertIn("/dashboard", html)

    def test_22_full_pipeline_persistence_and_monitoring_integration(self):
        """TEST 22: Persisting fall event via NotifyingFallEventService updates monitoring."""
        event_id = "evt_pipeline_integration_01"
        event, is_new = self.notifying_service.record_confirmed_fall(
            event_id=event_id,
            detected_at="2026-10-07T04:20:00Z",
            confidence=0.96,
            source="ESP32_MPU6050",
            diagnostic_metadata={"location": {"lat": 12.9716, "lon": 77.5946, "valid": True}},
        )
        self.assertTrue(is_new)

        # Allow worker thread to process task
        import time
        for _ in range(20):
            time.sleep(0.1)
            rec = self.notif_service.get_delivery_record(event_id)
            if rec and rec["status"] == NotificationStatus.SENT.value:
                break

        # Check monitoring snapshot reflects the newly generated delivery
        resp = self.client.get("/api/notifications/monitoring")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreaterEqual(data["summary"]["sent"], 1)

        # Check detail endpoint retrieves event telemetry
        resp_detail = self.client.get(f"/api/notifications/events/{event_id}")
        self.assertEqual(resp_detail.status_code, 200)
        notif = resp_detail.get_json()["notification"]
        self.assertEqual(notif["event_id"], event_id)
        self.assertEqual(notif["status"], "SENT")


if __name__ == "__main__":
    unittest.main()
