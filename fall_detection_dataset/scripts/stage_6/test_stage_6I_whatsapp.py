"""
Stage 6I Automated Test Suite: WhatsApp Notification Layer.
Standalone Fall Detection Project.

Safety Guarantee:
- Absolutely NO real WhatsApp messages are dispatched during testing.
- Uses FakeWhatsAppProvider and hermetic isolated SQLite databases.
- Real credentials are never loaded or touched.

Validates all 20 Stage 6I requirements:
 1. Confirmed persisted event triggers notification.
 2. Unconfirmed event does not trigger notification.
 3. Duplicate event_id is suppressed.
 4. Valid GPS coordinates are included.
 5. No GPS fix does not include coordinates.
 6. Missing location does not crash.
 7. Invalid latitude is rejected.
 8. Invalid longitude is rejected.
 9. Google Maps URL is generated only for valid coordinates.
10. Provider HTTP 500 is handled safely.
11. Provider HTTP 401 is handled safely.
12. Provider timeout is handled safely.
13. Malformed provider response is handled safely.
14. Missing credentials disable notifications safely.
15. Secret values do not appear in logs.
16. Recipient phone number is masked in logs.
17. Notification failure does not affect persisted fall event.
18. Backend request does not wait for provider completion.
19. Duplicate concurrent submissions for the same event_id do not produce duplicate notifications.
20. Existing fall event lifecycle remains unchanged.
"""

import io
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

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
from src.notifications.whatsapp_formatter import (
    format_fall_message,
    validate_gps_coordinates
)
from src.notifications.whatsapp_provider import (
    FakeWhatsAppProvider,
    TwilioWhatsAppProvider
)
from src.notifications.whatsapp_service import (
    NotificationStatus,
    WhatsAppNotificationService,
    reset_global_notification_service
)
from src.notifications.notification_listener import (
    NotifyingFallEventService,
    create_notifying_app
)


class TestStage6IWhatsApp(unittest.TestCase):
    """Hermetic test suite for Stage 6I WhatsApp notifications."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_notifications.db")
        
        # Test configuration using fake provider (NEVER real credentials)
        self.config = NotificationConfig(
            provider="fake",
            whatsapp_from="whatsapp:+14155238886",
            whatsapp_recipient="whatsapp:+919876543210",
            timeout_seconds=2.0,
            db_path=self.db_path
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

    def test_01_confirmed_persisted_event_triggers_notification(self):
        """TEST 1: Confirmed persisted event triggers notification."""
        payload = {
            "event_id": "test_evt_001",
            "detected_at": "2026-10-07T04:00:00Z",
            "confidence": 0.95,
            "source": "ESP32_MPU6050",
            "diagnostic_metadata": {
                "location": {
                    "fix_valid": True,
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "satellites": 8,
                    "hdop": 1.2
                }
            }
        }
        res = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertTrue(data["is_new"])

        # Wait for async background worker
        self.notif_service.join_queue(timeout=3.0)

        self.assertEqual(self.provider.get_sent_count(), 1)
        sent = self.provider.get_last_message()
        self.assertIsNotNone(sent)
        self.assertEqual(sent["event_id"], "test_evt_001")
        self.assertIn("EMERGENCY: Confirmed Fall Detected", sent["message"])

        # Check SQLite delivery record
        delivery = self.notif_service.get_delivery_record("test_evt_001")
        self.assertIsNotNone(delivery)
        self.assertEqual(delivery["status"], NotificationStatus.SENT.value)

    def test_02_unconfirmed_event_does_not_trigger_notification(self):
        """TEST 2: Unconfirmed event does not trigger notification."""
        unconfirmed_event = FallEvent(
            event_id="test_unconfirmed_002",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CANCELLED  # Not CONFIRMED
        )
        status = self.notif_service.enqueue_event(unconfirmed_event)
        self.assertEqual(status, NotificationStatus.REJECTED)
        self.notif_service.join_queue(timeout=1.0)
        self.assertEqual(self.provider.get_sent_count(), 0)

    def test_03_duplicate_event_id_is_suppressed(self):
        """TEST 3: Duplicate event_id is suppressed."""
        payload = {
            "event_id": "test_dup_003",
            "detected_at": "2026-10-07T04:00:00Z",
            "confidence": 0.92
        }
        # First attempt: 201 Created -> Dispatched
        res1 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res1.status_code, 201)
        self.assertTrue(res1.get_json()["is_new"])

        # Second attempt: 200 OK -> Idempotent, suppressed
        res2 = self.client.post("/api/fall-events", json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.get_json()["is_new"])

        self.notif_service.join_queue(timeout=3.0)
        # Exactly 1 message sent
        self.assertEqual(self.provider.get_sent_count(), 1)

        # Enqueueing directly also yields DUPLICATE_SUPPRESSED
        evt = self.repo.get("test_dup_003")
        self.assertIsNotNone(evt)
        status = self.notif_service.enqueue_event(evt)
        self.assertEqual(status, NotificationStatus.DUPLICATE_SUPPRESSED)

    def test_04_valid_gps_coordinates_are_included(self):
        """TEST 4: Valid GPS coordinates are included in message."""
        event = FallEvent(
            event_id="test_gps_valid_004",
            detected_at="2026-10-07T04:00:00Z",
            confidence=0.88,
            source="ESP32_NEO6M",
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata={
                "location": {
                    "fix_valid": True,
                    "latitude": 12.971598,
                    "longitude": 77.594566,
                    "satellites": 7,
                    "hdop": 1.1
                }
            }
        )
        msg = format_fall_message(event)
        self.assertIn("GPS Status: FIX VALID", msg)
        self.assertIn("Location: 12.971598, 77.594566", msg)
        self.assertIn("Satellites: 7", msg)
        self.assertIn("HDOP: 1.1", msg)
        self.assertIn("https://www.google.com/maps?q=12.971598,77.594566", msg)

    def test_05_no_gps_fix_does_not_include_coordinates(self):
        """TEST 5: No GPS fix does not include coordinates."""
        event = FallEvent(
            event_id="test_no_fix_005",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata={
                "location": {
                    "fix_valid": False,
                    "latitude": 12.9716,
                    "longitude": 77.5946
                }
            }
        )
        msg = format_fall_message(event)
        self.assertIn("GPS Status: NO VALID FIX", msg)
        self.assertIn("Location: Coordinates Not Available", msg)
        self.assertNotIn("https://www.google.com/maps", msg)
        self.assertNotIn("12.9716", msg)

    def test_06_missing_location_does_not_crash(self):
        """TEST 6: Missing location does not crash."""
        event = FallEvent(
            event_id="test_missing_loc_006",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata=None
        )
        msg = format_fall_message(event)
        self.assertIn("GPS Status: LOCATION UNAVAILABLE", msg)
        self.assertIn("Location: Not available", msg)
        self.assertNotIn("https://www.google.com/maps", msg)

    def test_07_invalid_latitude_is_rejected(self):
        """TEST 7: Invalid latitude is rejected."""
        valid, lat, lon = validate_gps_coordinates(95.0, 77.0)
        self.assertFalse(valid)
        self.assertIsNone(lat)

        valid_neg, lat_neg, lon_neg = validate_gps_coordinates(-91.0, 77.0)
        self.assertFalse(valid_neg)

        event = FallEvent(
            event_id="test_inv_lat_007",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata={
                "location": {
                    "fix_valid": True,
                    "latitude": 95.0,
                    "longitude": 77.0
                }
            }
        )
        msg = format_fall_message(event)
        self.assertIn("GPS Status: NO VALID FIX", msg)
        self.assertNotIn("https://www.google.com/maps", msg)

    def test_08_invalid_longitude_is_rejected(self):
        """TEST 8: Invalid longitude is rejected, and 0,0 is rejected."""
        valid, lat, lon = validate_gps_coordinates(12.0, 185.0)
        self.assertFalse(valid)

        # Critical rule: Never send 0,0
        valid_zero, lat_zero, lon_zero = validate_gps_coordinates(0.0, 0.0)
        self.assertFalse(valid_zero)

        event_zero = FallEvent(
            event_id="test_zero_coords_008",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata={
                "location": {
                    "fix_valid": True,
                    "latitude": 0.0,
                    "longitude": 0.0
                }
            }
        )
        msg_zero = format_fall_message(event_zero)
        self.assertIn("GPS Status: NO VALID FIX", msg_zero)
        self.assertNotIn("https://www.google.com/maps", msg_zero)

    def test_09_google_maps_url_only_for_valid_coordinates(self):
        """TEST 9: Google Maps URL is generated ONLY for valid coordinates."""
        # Valid
        msg_valid = format_fall_message({
            "event_id": "test_e1",
            "detected_at": "now",
            "diagnostic_metadata": {"location": {"fix_valid": True, "latitude": 28.6139, "longitude": 77.2090}}
        })
        self.assertIn("https://www.google.com/maps?q=28.6139,77.209", msg_valid)

        # NaN / Inf
        valid_nan, _, _ = validate_gps_coordinates(float("nan"), 77.0)
        self.assertFalse(valid_nan)
        valid_inf, _, _ = validate_gps_coordinates(28.0, float("inf"))
        self.assertFalse(valid_inf)

    def test_10_provider_http_500_handled_safely(self):
        """TEST 10: Provider HTTP 500 is handled safely."""
        self.provider.set_mode("http_500")
        
        event = FallEvent(
            event_id="test_err_500",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        # Enqueue should not raise
        status = self.notif_service.enqueue_event(event)
        self.assertEqual(status, NotificationStatus.QUEUED)
        
        self.notif_service.join_queue(timeout=3.0)
        
        # Delivery record should reflect FAILED without crashing
        record = self.notif_service.get_delivery_record("test_err_500")
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "HTTP_500")

    def test_11_provider_http_401_handled_safely(self):
        """TEST 11: Provider HTTP 401 is handled safely."""
        self.provider.set_mode("http_401")
        
        event = FallEvent(
            event_id="test_err_401",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        status = self.notif_service.enqueue_event(event)
        self.assertEqual(status, NotificationStatus.QUEUED)
        
        self.notif_service.join_queue(timeout=3.0)
        
        record = self.notif_service.get_delivery_record("test_err_401")
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "HTTP_401")

    def test_12_provider_timeout_handled_safely(self):
        """TEST 12: Provider timeout is handled safely."""
        self.provider.set_mode("timeout")
        
        event = FallEvent(
            event_id="test_err_timeout",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)
        
        record = self.notif_service.get_delivery_record("test_err_timeout")
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "Timeout")

    def test_13_malformed_provider_response_handled_safely(self):
        """TEST 13: Malformed provider response is handled safely."""
        self.provider.set_mode("malformed")
        
        event = FallEvent(
            event_id="test_err_malformed",
            detected_at="2026-10-07T04:00:00Z",
            status=FallEventStatus.CONFIRMED
        )
        self.notif_service.enqueue_event(event)
        self.notif_service.join_queue(timeout=3.0)
        
        record = self.notif_service.get_delivery_record("test_err_malformed")
        self.assertEqual(record["status"], NotificationStatus.FAILED.value)
        self.assertEqual(record["error_message"], "MalformedResponse")

    def test_14_missing_credentials_disable_notifications_safely(self):
        """TEST 14: Missing credentials disable notifications safely."""
        disabled_config = NotificationConfig(
            provider="twilio",
            account_sid=None,
            auth_token=None,
            whatsapp_from=None,
            whatsapp_recipient=None,
            db_path=self.db_path
        )
        self.assertFalse(disabled_config.enabled)
        self.assertEqual(disabled_config.status, "DISABLED")

        disabled_service = WhatsAppNotificationService(config=disabled_config, db_path=self.db_path)
        try:
            event = FallEvent(
                event_id="test_disabled_014",
                detected_at="2026-10-07T04:00:00Z",
                status=FallEventStatus.CONFIRMED
            )
            status = disabled_service.enqueue_event(event)
            self.assertEqual(status, NotificationStatus.DISABLED)
        finally:
            disabled_service.shutdown()

    def test_15_secret_values_do_not_appear_in_logs(self):
        """TEST 15: Secret values do not appear in logs or string representations."""
        secret_token = "SUPER_SECRET_TOKEN_XYZ_123"
        secret_sid = "AC_TEST_ACCOUNT_SID_999"

        # Capture log records
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("Stage6I_NotificationConfig")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        cfg = NotificationConfig(
            provider="twilio",
            account_sid=secret_sid,
            auth_token=secret_token,
            whatsapp_from="whatsapp:+14155238886",
            whatsapp_recipient="whatsapp:+919876543210"
        )
        cfg_str = str(cfg)
        
        logger.removeHandler(handler)

        self.assertNotIn(secret_token, cfg_str)
        self.assertNotIn(secret_token, log_stream.getvalue())
        self.assertEqual(mask_secret(secret_token), "[REDACTED]")

    def test_16_recipient_phone_number_masked_in_logs(self):
        """TEST 16: Recipient phone number is masked in logs."""
        raw_number = "+919876543210"
        masked = mask_phone_number(raw_number)
        self.assertEqual(masked, "+91*****3210")
        self.assertNotIn("987654", masked)

        with_prefix = "whatsapp:+919876543210"
        masked_prefix = mask_phone_number(with_prefix)
        self.assertEqual(masked_prefix, "whatsapp:+91*****3210")
        self.assertNotIn("987654", masked_prefix)

    def test_17_notification_failure_does_not_affect_persisted_fall_event(self):
        """TEST 17: Notification failure does not affect persisted fall event."""
        # Cause provider to fail
        self.provider.set_mode("http_500")

        event, is_new = self.notifying_service.record_confirmed_fall(
            event_id="test_persist_intact_017",
            detected_at="2026-10-07T04:00:00Z",
            confidence=0.91,
            source="ESP32_MPU6050"
        )
        self.assertTrue(is_new)
        self.assertEqual(event.event_id, "test_persist_intact_017")
        self.assertEqual(event.status, FallEventStatus.CONFIRMED)

        self.notif_service.join_queue(timeout=3.0)

        # Database record must still exist and be intact
        persisted = self.repo.get("test_persist_intact_017")
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, FallEventStatus.CONFIRMED)
        self.assertEqual(persisted.confidence, 0.91)

    def test_18_backend_request_does_not_wait_for_provider_completion(self):
        """TEST 18: Backend request does not wait for provider completion (non-blocking)."""
        # Create a provider with a simulated slow delivery (e.g. 0.5s network latency)
        class SlowFakeProvider(FakeWhatsAppProvider):
            def send_message(self, recipient, message, event_id):
                time.sleep(0.5)
                return super().send_message(recipient, message, event_id)

        slow_provider = SlowFakeProvider()
        slow_service = WhatsAppNotificationService(
            config=self.config,
            provider=slow_provider,
            db_path=self.db_path
        )
        slow_notifying_svc = NotifyingFallEventService(
            repository=self.repo,
            notification_service=slow_service
        )

        try:
            start_t = time.time()
            event, is_new = slow_notifying_svc.record_confirmed_fall(
                event_id="test_nonblocking_018",
                detected_at="2026-10-07T04:00:00Z"
            )
            elapsed = time.time() - start_t

            # Main ingestion must return essentially instantaneously (< 0.15s), NOT waiting 0.5s
            self.assertLess(elapsed, 0.25)
            self.assertTrue(is_new)
        finally:
            slow_service.shutdown()

    def test_19_duplicate_concurrent_submissions_no_duplicate_notifications(self):
        """TEST 19: Duplicate concurrent submissions for the same event_id do not duplicate."""
        event_id = "test_concurrent_dup_019"
        num_threads = 10
        statuses = []
        barrier = threading.Barrier(num_threads)

        def worker_fn():
            event = FallEvent(
                event_id=event_id,
                detected_at="2026-10-07T04:00:00Z",
                status=FallEventStatus.CONFIRMED
            )
            barrier.wait()  # Synchronize starts for highest concurrency
            status = self.notif_service.enqueue_event(event)
            statuses.append(status)

        threads = [threading.Thread(target=worker_fn) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.notif_service.join_queue(timeout=3.0)

        # Exactly 1 thread must have succeeded with QUEUED
        queued_count = sum(1 for s in statuses if s == NotificationStatus.QUEUED)
        dup_count = sum(1 for s in statuses if s == NotificationStatus.DUPLICATE_SUPPRESSED)

        self.assertEqual(queued_count, 1)
        self.assertEqual(dup_count, num_threads - 1)
        self.assertEqual(self.provider.get_sent_count(), 1)

    def test_20_existing_fall_event_lifecycle_remains_unchanged(self):
        """TEST 20: Existing fall event lifecycle remains unchanged."""
        event, is_new = self.notifying_service.record_confirmed_fall(
            event_id="test_lifecycle_020",
            detected_at="2026-10-07T04:00:00Z"
        )
        self.assertEqual(event.status, FallEventStatus.CONFIRMED)

        # Acknowledge
        ack_event = self.notifying_service.acknowledge_event("test_lifecycle_020", acknowledged_by="nurse_1")
        self.assertEqual(ack_event.status, FallEventStatus.ACKNOWLEDGED)
        self.assertEqual(ack_event.acknowledged_by, "nurse_1")

        # Resolve
        res_event = self.notifying_service.resolve_event("test_lifecycle_020", resolved_by="dr_smith")
        self.assertEqual(res_event.status, FallEventStatus.RESOLVED)
        self.assertEqual(res_event.resolved_by, "dr_smith")

    def test_21_twilio_provider_endpoint_and_auth_headers_construction(self):
        """TEST 21: Verify Twilio provider constructs HTTPS endpoint, auth headers, and body correctly."""
        provider = TwilioWhatsAppProvider(
            account_sid="AC1234567890abcdef",
            auth_token="test_auth_token_secret",
            from_number="+14155238886",
            timeout_seconds=5.0
        )
        self.assertEqual(
            provider._endpoint_url,
            "https://api.twilio.com/2010-04-01/Accounts/AC1234567890abcdef/Messages.json"
        )
        self.assertEqual(provider.from_number, "whatsapp:+14155238886")

        # Mock urllib.request.urlopen returning 201 Created
        mock_response = MagicMock()
        mock_response.getcode.return_value = 201
        mock_response.read.return_value = b'{"sid": "SMmock12345", "status": "queued"}'
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            success, sid, diag = provider.send_message(
                recipient="+919876543210",
                message="Test Message",
                event_id="test_evt_twilio_021"
            )
            self.assertTrue(success)
            self.assertEqual(sid, "SMmock12345")
            self.assertEqual(diag["twilio_status"], "queued")

            # Verify request properties
            req_called = mock_urlopen.call_args[0][0]
            self.assertEqual(req_called.get_full_url(), "https://api.twilio.com/2010-04-01/Accounts/AC1234567890abcdef/Messages.json")
            self.assertEqual(req_called.get_method(), "POST")
            # Authorization header must be Basic auth with base64 encoded credentials
            auth_header = req_called.get_header("Authorization")
            self.assertTrue(auth_header.startswith("Basic "))
            # Body must contain To, From, Body
            body_data = req_called.data.decode("utf-8")
            self.assertIn("To=whatsapp%3A%2B919876543210", body_data)
            self.assertIn("From=whatsapp%3A%2B14155238886", body_data)
            self.assertIn("Body=Test+Message", body_data)

    def test_22_twilio_provider_http_error_handling(self):
        """TEST 22: Verify Twilio provider safely handles HTTP 401, URLError, and timeout without crashing."""
        provider = TwilioWhatsAppProvider(
            account_sid="AC1234567890abcdef",
            auth_token="test_auth_token_secret",
            from_number="+14155238886",
            timeout_seconds=5.0
        )

        # HTTP 401
        http_err = urllib.error.HTTPError(
            url="https://api.twilio.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"message": "Authenticate"}')
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            success, err, diag = provider.send_message("+919876543210", "Test", "evt_401")
            self.assertFalse(success)
            self.assertEqual(err, "HTTP_401")
            self.assertNotIn("test_auth_token_secret", str(diag))

        # URLError (e.g. connection refused)
        url_err = urllib.error.URLError(reason="Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            success, err, diag = provider.send_message("+919876543210", "Test", "evt_url_err")
            self.assertFalse(success)
            self.assertEqual(err, "ConnectionError")

        # Timeout
        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            success, err, diag = provider.send_message("+919876543210", "Test", "evt_timeout")
            self.assertFalse(success)
            self.assertEqual(err, "Timeout")


if __name__ == "__main__":
    unittest.main()

