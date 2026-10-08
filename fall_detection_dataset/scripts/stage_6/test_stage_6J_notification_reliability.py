"""
Stage 6J Automated Test Suite: Offline / Retry / Notification Reliability.
Standalone Fall Detection Project.

Safety Guarantee:
- Absolutely NO real WhatsApp messages are dispatched during testing.
- Uses FakeWhatsAppProvider and isolated temporary SQLite databases.
- Real credentials are never loaded or touched.

Validates all 26 Stage 6J requirements:
 1. successful notification -> SENT
 2. temporary timeout -> RETRY_WAIT
 3. HTTP 500 -> retry
 4. HTTP 502 -> retry
 5. HTTP 503 -> retry
 6. HTTP 429 -> retry
 7. HTTP 400 -> no endless retry
 8. HTTP 401 -> no endless retry
 9. missing credentials -> DISABLED
10. max attempts reached -> DEAD_LETTER or equivalent terminal state
11. exponential backoff calculation
12. retry scheduling persistence
13. restart recovery of QUEUED notification
14. restart recovery of RETRY_WAIT notification
15. SENT notification is never requeued
16. duplicate event_id remains suppressed
17. concurrent duplicate submissions remain single-delivery
18. queue-full condition does not lose persisted notification
19. notification failure does not modify fall event
20. notification failure does not break backend health
21. secrets do not appear in logs
22. recipient number remains masked
23. malformed provider response handled safely
24. worker survives provider exception
25. no infinite retry loop
26. existing Stage 6I tests still pass
"""

import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fall_event_backend.models import FallEvent, FallEventStatus
from src.fall_event_backend.storage import FallEventRepository
from src.notifications.notification_config import (
    NotificationConfig,
    mask_phone_number,
    mask_secret
)
from src.notifications.whatsapp_formatter import format_fall_message
from src.notifications.whatsapp_provider import (
    FakeWhatsAppProvider,
    TwilioWhatsAppProvider
)
from src.notifications.whatsapp_service import (
    NotificationStatus,
    WhatsAppNotificationService,
    calculate_backoff,
    is_retryable_error,
    reset_global_notification_service
)
from src.notifications.notification_listener import (
    NotifyingFallEventService,
    create_notifying_app
)


class TestStage6JNotificationReliability(unittest.TestCase):
    """Hermetic test suite for Stage 6J offline & retry reliability."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reliability.db")
        
        # Test configuration with retries enabled (max_attempts=3)
        self.config = NotificationConfig(
            provider="fake",
            whatsapp_from="whatsapp:+14155238886",
            whatsapp_recipient="whatsapp:+919876543210",
            timeout_seconds=2.0,
            db_path=self.db_path,
            max_attempts=3,
            retry_base_seconds=0.1,  # Short backoff for fast tests
            retry_max_seconds=1.0
        )
        self.provider = FakeWhatsAppProvider(mode="success")
        self.notif_service = WhatsAppNotificationService(
            config=self.config,
            provider=self.provider,
            db_path=self.db_path
        )
        self.repo = FallEventRepository(self.db_path)
        self.notifying_service = NotifyingFallEventService(
            repository=self.repo,
            notification_service=self.notif_service
        )
        self.app = create_notifying_app(service=self.notifying_service)
        self.client = self.app.test_client()

    def tearDown(self):
        self.notif_service.shutdown()
        reset_global_notification_service(None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_successful_notification_reaches_sent(self):
        """TEST 1: Successful notification transitions to SENT."""
        event = FallEvent(
            event_id="evt_success_01",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        status = self.notif_service.enqueue_event(event)
        self.assertEqual(status, NotificationStatus.QUEUED)

        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_success_01")
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], NotificationStatus.SENT.value)
        self.assertEqual(record["attempt_count"], 1)
        self.assertIsNotNone(record["dispatched_at"])
        self.assertEqual(self.provider.get_sent_count(), 1)

    def test_02_temporary_timeout_transitions_to_retry_wait(self):
        """TEST 2: Temporary timeout transitions to RETRY_WAIT."""
        self.provider.set_mode("timeout")
        event = FallEvent(
            event_id="evt_timeout_02",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_timeout_02")
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], NotificationStatus.RETRY_WAIT.value)
        self.assertEqual(record["attempt_count"], 1)
        self.assertEqual(record["error_message"], "Timeout")
        self.assertIsNotNone(record["next_attempt_at"])

    def test_03_http_500_transitions_to_retry_wait(self):
        """TEST 3: HTTP 500 error transitions to RETRY_WAIT."""
        self.provider.set_mode("http_500")
        event = FallEvent(
            event_id="evt_500_03",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_500_03")
        self.assertEqual(record["status"], NotificationStatus.RETRY_WAIT.value)
        self.assertEqual(record["error_message"], "HTTP_500")

    def test_04_http_502_transitions_to_retry_wait(self):
        """TEST 4: HTTP 502 Bad Gateway transitions to RETRY_WAIT."""
        self.provider.set_mode("http_502")
        event = FallEvent(
            event_id="evt_502_04",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_502_04")
        self.assertEqual(record["status"], NotificationStatus.RETRY_WAIT.value)
        self.assertEqual(record["error_message"], "HTTP_502")

    def test_05_http_503_transitions_to_retry_wait(self):
        """TEST 5: HTTP 503 Service Unavailable transitions to RETRY_WAIT."""
        self.provider.set_mode("http_503")
        event = FallEvent(
            event_id="evt_503_05",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_503_05")
        self.assertEqual(record["status"], NotificationStatus.RETRY_WAIT.value)
        self.assertEqual(record["error_message"], "HTTP_503")

    def test_06_http_429_transitions_to_retry_wait_with_retry_after(self):
        """TEST 6: HTTP 429 Rate Limiting respects Retry-After."""
        self.provider.set_mode("http_429")
        self.provider.set_retry_after(0.5)
        event = FallEvent(
            event_id="evt_429_06",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_429_06")
        self.assertEqual(record["status"], NotificationStatus.RETRY_WAIT.value)
        self.assertEqual(record["error_message"], "HTTP_429")
        self.assertIsNotNone(record["next_attempt_at"])

    def test_07_http_400_no_endless_retry(self):
        """TEST 7: HTTP 400 Bad Request is non-retryable and transitions to FAILED."""
        self.provider.set_mode("http_400")
        event = FallEvent(
            event_id="evt_400_07",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_400_07")
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "HTTP_400")
        self.assertIsNone(record["next_attempt_at"])

    def test_08_http_401_no_endless_retry(self):
        """TEST 8: HTTP 401 Unauthorized is non-retryable and transitions to FAILED."""
        self.provider.set_mode("http_401")
        event = FallEvent(
            event_id="evt_401_08",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_401_08")
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "HTTP_401")
        self.assertIsNone(record["next_attempt_at"])

    def test_09_missing_credentials_transitions_to_disabled(self):
        """TEST 9: Missing credentials disable notifications without retrying."""
        disabled_config = NotificationConfig(
            provider="twilio",
            account_sid=None,
            db_path=self.db_path
        )
        disabled_service = WhatsAppNotificationService(config=disabled_config, db_path=self.db_path)
        try:
            event = FallEvent(
                event_id="evt_disabled_09",
                detected_at="2026-10-07T04:00:00Z",
                status=FallEventStatus.CONFIRMED
            )
            status = disabled_service.enqueue_event(event)
            self.assertEqual(status, NotificationStatus.DISABLED)
        finally:
            disabled_service.shutdown()

    def test_10_max_attempts_reached_transitions_to_dead_letter(self):
        """TEST 10: Exhausting maximum retry attempts transitions to DEAD_LETTER."""
        self.provider.set_mode("http_500")
        event = FallEvent(
            event_id="evt_dead_letter_10",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        
        # Enqueue with max_attempts=3
        self.notif_service.enqueue_event(event)

        # Wait for worker scan loop to exhaust retries (short backoff configured)
        for _ in range(15):
            time.sleep(0.2)
            rec = self.notif_service.get_delivery_record("evt_dead_letter_10")
            if rec and rec["status"] == NotificationStatus.DEAD_LETTER.value:
                break

        final_rec = self.notif_service.get_delivery_record("evt_dead_letter_10")
        self.assertIsNotNone(final_rec)
        self.assertEqual(final_rec["status"], NotificationStatus.DEAD_LETTER.value)
        self.assertGreaterEqual(final_rec["attempt_count"], 3)

    def test_11_exponential_backoff_calculation(self):
        """TEST 11: Validates conservative exponential and staged backoff calculations."""
        # Standard staged backoff
        self.assertEqual(calculate_backoff(1, base_seconds=5.0), 5.0)
        self.assertEqual(calculate_backoff(2, base_seconds=5.0), 15.0)
        self.assertEqual(calculate_backoff(3, base_seconds=5.0), 60.0)
        self.assertEqual(calculate_backoff(4, base_seconds=5.0, max_seconds=300.0), 300.0)

        # Retry-After header overrides
        self.assertEqual(calculate_backoff(1, retry_after=42.0), 42.0)
        # Bounded by max_seconds
        self.assertEqual(calculate_backoff(1, max_seconds=300.0, retry_after=9999.0), 300.0)

    def test_12_retry_scheduling_persistence(self):
        """TEST 12: Verify retry scheduling columns are persisted in SQLite."""
        self.provider.set_mode("timeout")
        event = FallEvent(
            event_id="evt_persist_12",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_persist_12")
        self.assertIn("attempt_count", record)
        self.assertIn("max_attempts", record)
        self.assertIn("last_attempt_at", record)
        self.assertIn("next_attempt_at", record)
        self.assertEqual(record["attempt_count"], 1)
        self.assertEqual(record["max_attempts"], 3)

    def test_13_restart_recovery_of_queued_notification(self):
        """TEST 13: Verifies un-dispatched QUEUED notification survives process restart."""
        # Stop existing worker
        self.notif_service.shutdown()

        # Seed database directly with QUEUED record
        event = FallEvent(
            event_id="evt_restart_queued_13",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)

        with self.notif_service._get_connection() as conn:
            conn.execute("""
                INSERT INTO notification_deliveries (
                    event_id, status, provider, recipient_masked,
                    attempt_count, max_attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("evt_restart_queued_13", "QUEUED", "fake", "+91*****3210", 0, 3, "now", "now"))
            conn.commit()

        # Start a brand new service simulating process restart
        self.provider.clear()
        restarted_service = WhatsAppNotificationService(
            config=self.config,
            provider=self.provider,
            db_path=self.db_path
        )
        try:
            restarted_service.join_queue(timeout=3.0)
            record = restarted_service.get_delivery_record("evt_restart_queued_13")
            self.assertEqual(record["status"], NotificationStatus.SENT.value)
            self.assertEqual(self.provider.get_sent_count(), 1)
        finally:
            restarted_service.shutdown()

    def test_14_restart_recovery_of_retry_wait_notification(self):
        """TEST 14: Verifies due RETRY_WAIT notification survives process restart."""
        self.notif_service.shutdown()

        event = FallEvent(
            event_id="evt_restart_retry_14",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)

        # Seed database with RETRY_WAIT due in the past
        past_ts = "2020-01-01T00:00:00Z"
        with self.notif_service._get_connection() as conn:
            conn.execute("""
                INSERT INTO notification_deliveries (
                    event_id, status, provider, recipient_masked,
                    attempt_count, max_attempts, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("evt_restart_retry_14", "RETRY_WAIT", "fake", "+91*****3210", 1, 3, past_ts, "now", "now"))
            conn.commit()

        self.provider.clear()
        restarted_service = WhatsAppNotificationService(
            config=self.config,
            provider=self.provider,
            db_path=self.db_path
        )
        try:
            restarted_service.join_queue(timeout=3.0)
            record = restarted_service.get_delivery_record("evt_restart_retry_14")
            self.assertEqual(record["status"], NotificationStatus.SENT.value)
            self.assertEqual(self.provider.get_sent_count(), 1)
        finally:
            restarted_service.shutdown()

    def test_15_sent_notification_is_never_requeued(self):
        """TEST 15: Verifies SENT notifications are never re-queued on restart."""
        self.notif_service.shutdown()

        with self.notif_service._get_connection() as conn:
            conn.execute("""
                INSERT INTO notification_deliveries (
                    event_id, status, provider, recipient_masked,
                    attempt_count, max_attempts, dispatched_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("evt_sent_15", "SENT", "fake", "+91*****3210", 1, 3, "2026-10-07T04:00:00Z", "now", "now"))
            conn.commit()

        self.provider.clear()
        restarted_service = WhatsAppNotificationService(
            config=self.config,
            provider=self.provider,
            db_path=self.db_path
        )
        try:
            time.sleep(0.5)
            self.assertEqual(self.provider.get_sent_count(), 0)
        finally:
            restarted_service.shutdown()

    def test_16_duplicate_event_id_remains_suppressed(self):
        """TEST 16: Duplicate event_id remains suppressed."""
        payload = {
            "event_id": "evt_dup_16",
            "detected_at": "2026-10-07T04:00:00Z",
            "confidence": 0.95
        }
        res1 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res1.status_code, 201)

        res2 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.get_json()["is_new"])

        self.notif_service.join_queue(timeout=3.0)
        self.assertEqual(self.provider.get_sent_count(), 1)

    def test_17_concurrent_duplicate_submissions_remain_single_delivery(self):
        """TEST 17: Concurrent duplicate submissions produce exactly 1 notification delivery."""
        event_id = "evt_concurrent_17"
        num_threads = 15
        statuses = []
        barrier = threading.Barrier(num_threads)

        def worker_fn():
            event = FallEvent(
                event_id=event_id,
                detected_at="2026-10-07T04:00:00Z",
                status=FallEventStatus.CONFIRMED
            )
            barrier.wait()
            st = self.notif_service.enqueue_event(event)
            statuses.append(st)

        threads = [threading.Thread(target=worker_fn) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.notif_service.join_queue(timeout=3.0)

        queued_count = sum(1 for s in statuses if s == NotificationStatus.QUEUED)
        dup_count = sum(1 for s in statuses if s == NotificationStatus.DUPLICATE_SUPPRESSED)
        self.assertEqual(queued_count, 1)
        self.assertEqual(dup_count, num_threads - 1)
        self.assertEqual(self.provider.get_sent_count(), 1)

    def test_18_queue_full_condition_does_not_lose_persisted_notification(self):
        """TEST 18: Queue-full condition preserves notification record in SQLite."""
        # Create a tiny queue config
        tiny_config = NotificationConfig(
            provider="fake",
            queue_maxsize=1,
            db_path=self.db_path,
            max_attempts=3
        )
        tiny_service = WhatsAppNotificationService(config=tiny_config, provider=self.provider, db_path=self.db_path)

        try:
            # Fill queue
            tiny_service._queue.put_nowait(FallEvent("evt_fill_1", "now", status=FallEventStatus.CONFIRMED))

            # Next enqueue causes queue.Full
            overflow_event = FallEvent("evt_overflow_18", "now", status=FallEventStatus.CONFIRMED)
            self.repo.create(overflow_event)
            status = tiny_service.enqueue_event(overflow_event)

            # Returns QUEUED and state is safely in SQLite
            self.assertEqual(status, NotificationStatus.QUEUED)
            record = tiny_service.get_delivery_record("evt_overflow_18")
            self.assertIsNotNone(record)
            self.assertEqual(record["status"], NotificationStatus.QUEUED.value)
        finally:
            tiny_service.shutdown()

    def test_19_notification_failure_does_not_modify_fall_event(self):
        """TEST 19: Notification failure does not affect persisted fall event."""
        self.provider.set_mode("http_500")
        event, is_new = self.notifying_service.record_confirmed_fall(
            event_id="evt_persist_intact_19",
            detected_at="2026-10-07T04:00:00Z",
            confidence=0.89
        )
        self.assertTrue(is_new)
        self.notif_service.join_queue(timeout=3.0)

        persisted = self.repo.get("evt_persist_intact_19")
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, FallEventStatus.CONFIRMED)
        self.assertEqual(persisted.confidence, 0.89)

    def test_20_notification_failure_does_not_break_backend_health(self):
        """TEST 20: Notification failure does not degrade backend health endpoint."""
        self.provider.set_mode("http_500")
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "healthy")

    def test_21_secrets_do_not_appear_in_logs(self):
        """TEST 21: Authentication tokens and secrets never appear in logs."""
        secret_token = "CRITICAL_SECRET_AUTH_TOKEN_777"
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("Stage6J_WhatsAppService")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        cfg = NotificationConfig(
            provider="twilio",
            account_sid="AC1234567890",
            auth_token=secret_token,
            whatsapp_from="+14155238886",
            whatsapp_recipient="+919876543210"
        )
        str_repr = str(cfg)
        logger.removeHandler(handler)

        self.assertNotIn(secret_token, str_repr)
        self.assertNotIn(secret_token, log_stream.getvalue())

    def test_22_recipient_number_remains_masked(self):
        """TEST 22: Recipient phone number is masked in persistent delivery records."""
        event = FallEvent(
            event_id="evt_mask_22",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_mask_22")
        self.assertEqual(record["recipient_masked"], "whatsapp:+91*****3210")
        self.assertNotIn("987654", record["recipient_masked"])

    def test_23_malformed_provider_response_handled_safely(self):
        """TEST 23: Malformed provider response is caught safely."""
        self.provider.set_mode("malformed")
        event = FallEvent(
            event_id="evt_malformed_23",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)

        record = self.notif_service.get_delivery_record("evt_malformed_23")
        self.assertIsNotNone(record)
        self.assertEqual(record["error_message"], "MalformedResponse")

    def test_24_worker_survives_provider_exception(self):
        """TEST 24: Background worker survives unexpected provider exceptions."""
        class ExplodingProvider(FakeWhatsAppProvider):
            def send_message(self, recipient, message, event_id):
                raise RuntimeError("Unexpected physical link crash")

        exploding_service = WhatsAppNotificationService(
            config=self.config,
            provider=ExplodingProvider(),
            db_path=self.db_path
        )
        try:
            event = FallEvent(
                event_id="evt_explode_24",
                detected_at="2026-10-07T04:00:00Z",
                status=FallEventStatus.CONFIRMED
            )
            self.repo.create(event)
            exploding_service.enqueue_event(event)
            exploding_service.join_queue(timeout=3.0)

            # Worker thread remains alive
            self.assertTrue(exploding_service._worker_thread.is_alive())
            rec = exploding_service.get_delivery_record("evt_explode_24")
            self.assertEqual(rec["error_message"], "RuntimeError")
        finally:
            exploding_service.shutdown()

    def test_25_no_infinite_retry_loop(self):
        """TEST 25: Delivery transitions to DEAD_LETTER after max attempts without looping infinitely."""
        self.provider.set_mode("http_500")
        event = FallEvent(
            event_id="evt_no_infinite_25",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.repo.create(event)
        self.notif_service.enqueue_event(event)

        # Allow worker scanner to run through max_attempts (3)
        for _ in range(20):
            time.sleep(0.15)
            rec = self.notif_service.get_delivery_record("evt_no_infinite_25")
            if rec and rec["status"] == NotificationStatus.DEAD_LETTER.value:
                break

        rec = self.notif_service.get_delivery_record("evt_no_infinite_25")
        self.assertEqual(rec["status"], NotificationStatus.DEAD_LETTER.value)
        self.assertLessEqual(rec["attempt_count"], 4)

    def test_26_existing_stage_6i_tests_still_pass(self):
        """TEST 26: Complete Stage 6I test suite passes without modification."""
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "fall_detection_dataset", "scripts", "stage_6", "test_stage_6I_whatsapp.py")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Stage 6I suite failed:\n{res.stdout}\n{res.stderr}")
        self.assertIn("Ran 22 tests", res.stderr + res.stdout)
        self.assertIn("OK", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
