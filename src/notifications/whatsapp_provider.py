"""
Stage 6I WhatsApp Provider Abstraction and Twilio HTTPS Implementation.
Standalone Fall Detection Project.

Strict Constraints:
- NO Twilio Python SDK.
- NO external HTTP dependencies (no requests package).
- Standard-library HTTPS only (urllib.request).
- Twilio endpoint constructed strictly internally from account_sid.
- No authorization headers or secret tokens leaked to logs or exceptions.
- Masked phone numbers in all logging outputs.
"""

import abc
import base64
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from src.notifications.notification_config import mask_phone_number

logger = logging.getLogger("Stage6I_WhatsAppProvider")


class WhatsAppProvider(abc.ABC):
    """Abstract interface for internet HTTPS WhatsApp messaging providers."""

    @abc.abstractmethod
    def send_message(
        self,
        recipient: str,
        message: str,
        event_id: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sends a message to the specified recipient.

        Args:
            recipient: E.164 phone number, optionally prefixed with 'whatsapp:'.
            message: Formatted text payload.
            event_id: Associated fall event ID for tracing.

        Returns:
            Tuple[bool, str, Dict[str, Any]]:
                - success: True if provider accepted the message.
                - message_id_or_error: Provider message SID or safe error summary.
                - diagnostic_info: Safe diagnostic metadata (no secrets/headers).
        """
        pass


class TwilioWhatsAppProvider(WhatsAppProvider):
    """
    Standard-library HTTPS REST implementation for Twilio WhatsApp API.
    Zero third-party dependencies.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        timeout_seconds: float = 10.0
    ):
        if not account_sid or not auth_token:
            raise ValueError("account_sid and auth_token must be provided.")
        if not from_number:
            raise ValueError("from_number must be provided.")

        self.account_sid = str(account_sid).strip()
        self.auth_token = str(auth_token).strip()
        
        # Ensure 'whatsapp:' prefix for Twilio WhatsApp channel
        raw_from = str(from_number).strip()
        self.from_number = raw_from if raw_from.startswith("whatsapp:") else f"whatsapp:{raw_from}"
        self.timeout_seconds = max(1.0, float(timeout_seconds))

        # Endpoint constructed strictly internally from configured Account SID
        self._endpoint_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"

    def send_message(
        self,
        recipient: str,
        message: str,
        event_id: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sends a WhatsApp message via official Twilio Messages HTTPS REST endpoint.
        """
        masked_to = mask_phone_number(recipient)
        
        # Ensure recipient has 'whatsapp:' prefix
        clean_to = str(recipient).strip()
        to_param = clean_to if clean_to.startswith("whatsapp:") else f"whatsapp:{clean_to}"

        # Twilio REST API expects application/x-www-form-urlencoded
        payload_dict = {
            "To": to_param,
            "From": self.from_number,
            "Body": message
        }
        encoded_payload = urllib.parse.urlencode(payload_dict).encode("utf-8")

        # Construct HTTP Basic Auth header safely
        credentials = f"{self.account_sid}:{self.auth_token}"
        encoded_auth = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
        auth_header = f"Basic {encoded_auth}"

        req = urllib.request.Request(
            url=self._endpoint_url,
            data=encoded_payload,
            method="POST"
        )
        req.add_header("Authorization", auth_header)
        req.add_header("Content-Type", "application/x-www-form-urlencoded;charset=utf-8")

        logger.debug(
            "Dispatching Twilio WhatsApp notification for event=%s to %s",
            event_id,
            masked_to
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                status_code = response.getcode()
                response_bytes = response.read()
                
            try:
                response_json = json.loads(response_bytes.decode("utf-8"))
            except Exception:
                response_json = {}

            # Twilio returns 201 Created on successful message queueing
            if status_code in (200, 201):
                sid = response_json.get("sid", "UNKNOWN_SID")
                twilio_status = response_json.get("status", "queued")
                logger.info(
                    "Twilio WhatsApp dispatch SUCCESS for event=%s [SID=%s, status=%s, to=%s]",
                    event_id,
                    sid,
                    twilio_status,
                    masked_to
                )
                return True, sid, {
                    "provider": "twilio",
                    "status_code": status_code,
                    "sid": sid,
                    "twilio_status": twilio_status
                }
            else:
                err_msg = f"Unexpected HTTP status {status_code}"
                logger.warning(
                    "Twilio WhatsApp dispatch failed for event=%s [status=%s, to=%s]",
                    event_id,
                    status_code,
                    masked_to
                )
                return False, f"HTTP_{status_code}", {
                    "provider": "twilio",
                    "status_code": status_code,
                    "error": err_msg
                }

        except urllib.error.HTTPError as e:
            # Capture HTTP error codes (e.g., 401 Unauthorized, 400 Bad Request, 429, 500)
            status_code = e.code
            safe_reason = e.reason if hasattr(e, "reason") else "HTTPError"
            retry_after = None
            if hasattr(e, "headers") and e.headers:
                raw_ra = e.headers.get("Retry-After")
                if raw_ra:
                    try:
                        retry_after = float(raw_ra)
                    except (ValueError, TypeError):
                        pass

            logger.error(
                "Twilio HTTP error for event=%s: status=%s, reason=%s [to=%s]",
                event_id,
                status_code,
                safe_reason,
                masked_to
            )
            return False, f"HTTP_{status_code}", {
                "provider": "twilio",
                "status_code": status_code,
                "retry_after": retry_after,
                "error": "HTTPError"
            }

        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            # Catch network timeout or DNS failure without crashing
            is_timeout = isinstance(e, (socket.timeout, TimeoutError)) or "timed out" in str(e).lower()
            err_type = "Timeout" if is_timeout else "ConnectionError"
            logger.error(
                "Twilio network failure for event=%s: %s [to=%s]",
                event_id,
                err_type,
                masked_to
            )
            return False, err_type, {
                "provider": "twilio",
                "error": err_type
            }

        except Exception as e:
            # Safe catch-all protecting against any unexpected socket or OS error
            logger.error(
                "Twilio unexpected error for event=%s: %s",
                event_id,
                type(e).__name__
            )
            return False, "ProviderError", {
                "provider": "twilio",
                "error": type(e).__name__
            }


class FakeWhatsAppProvider(WhatsAppProvider):
    """
    Mock / Synthetic WhatsApp provider for hermetic testing.
    Records dispatches in-memory with zero network traffic.
    Allows simulating various failure scenarios (HTTP 500, 502, 503, 429, 401, 400, timeout, malformed, sequences).
    """

    def __init__(self, mode: str = "success"):
        self.mode = mode
        self.sequence: List[str] = []
        self.fake_retry_after: Optional[float] = None
        self.sent_messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._counter = 0

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.mode = mode
            self.sequence.clear()

    def set_sequence(self, sequence: List[str]) -> None:
        with self._lock:
            self.sequence = list(sequence)

    def set_retry_after(self, seconds: Optional[float]) -> None:
        with self._lock:
            self.fake_retry_after = seconds

    def clear(self) -> None:
        with self._lock:
            self.sent_messages.clear()
            self.sequence.clear()
            self._counter = 0
            self.fake_retry_after = None

    def get_sent_count(self) -> int:
        with self._lock:
            return len(self.sent_messages)

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.sent_messages[-1] if self.sent_messages else None

    def send_message(
        self,
        recipient: str,
        message: str,
        event_id: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        masked_to = mask_phone_number(recipient)
        
        with self._lock:
            self._counter += 1
            call_id = f"FAKE_MSG_{self._counter}"

            current_mode = self.sequence.pop(0) if self.sequence else self.mode

            if current_mode == "success":
                record = {
                    "id": call_id,
                    "recipient": recipient,
                    "recipient_masked": masked_to,
                    "message": message,
                    "event_id": event_id,
                    "timestamp": time.time()
                }
                self.sent_messages.append(record)
                return True, call_id, {"provider": "fake", "status": "sent", "id": call_id}

            elif current_mode == "http_500":
                return False, "HTTP_500", {"provider": "fake", "status_code": 500, "error": "Internal Server Error"}

            elif current_mode == "http_502":
                return False, "HTTP_502", {"provider": "fake", "status_code": 502, "error": "Bad Gateway"}

            elif current_mode == "http_503":
                return False, "HTTP_503", {"provider": "fake", "status_code": 503, "error": "Service Unavailable"}

            elif current_mode == "http_429":
                return False, "HTTP_429", {
                    "provider": "fake",
                    "status_code": 429,
                    "retry_after": self.fake_retry_after,
                    "error": "Too Many Requests"
                }

            elif current_mode == "http_400":
                return False, "HTTP_400", {"provider": "fake", "status_code": 400, "error": "Bad Request"}

            elif current_mode == "http_401":
                return False, "HTTP_401", {"provider": "fake", "status_code": 401, "error": "Unauthorized"}

            elif current_mode == "timeout":
                return False, "Timeout", {"provider": "fake", "error": "Timeout"}

            elif current_mode == "malformed":
                return False, "MalformedResponse", {"provider": "fake", "error": "MalformedResponse"}

            else:
                return False, "UnknownError", {"provider": "fake", "error": current_mode}
