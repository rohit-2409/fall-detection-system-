"""
Stage 6J WhatsApp Notification Service and Async Background Worker.
Standalone Fall Detection Project.

Reliability Features (Stage 6J):
- Gating: Dispatches ONLY for confirmed, persisted FallEvents.
- Database-First: Enqueued strictly AFTER successful SQLite fall event commit.
- Non-blocking: Enqueue operation returns immediately via bounded queue.
- Persistent Retry: Tracks attempt_count, max_attempts, next_attempt_at, last_attempt_at, last_error.
- Bounded Exponential Backoff: Staged backoff (immediate -> 5s -> 15s -> 60s -> 5m, max 5 attempts).
- Error Classification: Distinguishes transient (HTTP 429, 5xx, timeouts) from permanent errors (HTTP 400, 401, 403).
- Rate Limiting: Respects HTTP 429 Retry-After headers up to configured maximum delay.
- Process Restart Recovery: Un-sent records (QUEUED, SENDING, RETRY_WAIT) automatically recovered on startup.
- Queue Overflow Safety: Queue-full conditions preserve state in SQLite rather than dropping alerts.
- Duplicate Protection: Thread-safe in-memory cache + persistent SQLite delivery state table.
- Graceful Degradation: Missing credentials result in DISABLED state without crashing.
- Failure Isolation: Network, HTTP, or provider failures never propagate to backend or caller.

Distributed Systems Note:
If the application crashes after provider dispatch acceptance but before SQLite records SENT,
restart recovery will re-attempt delivery upon reboot. This provides at-least-once delivery with
strong local event_id deduplication.
"""

import datetime
import enum
import logging
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.fall_event_backend.models import FallEvent, FallEventStatus, get_utc_iso_now
from src.fall_event_backend.storage import DEFAULT_DB_PATH, FallEventRepository
from src.notifications.notification_config import (
    NotificationConfig,
    mask_phone_number
)
from src.notifications.whatsapp_formatter import format_fall_message
from src.notifications.whatsapp_provider import (
    FakeWhatsAppProvider,
    TwilioWhatsAppProvider,
    WhatsAppProvider
)

logger = logging.getLogger("Stage6J_WhatsAppService")


class NotificationStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RETRY_WAIT = "RETRY_WAIT"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"
    DISABLED = "DISABLED"
    REJECTED = "REJECTED"


RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUSES = {400, 401, 403, 404}


def is_retryable_error(error_code_or_msg: str, status_code: Optional[int] = None) -> bool:
    """
    Classifies provider errors into retryable vs non-retryable.
    
    Retryable:
      - Network / transport errors: ConnectionError, URLError, Timeout
      - HTTP 429 (Rate limiting), 500, 502, 503, 504
    
    Non-retryable:
      - HTTP 400, 401, 403, 404
      - Permanent configuration, authentication, or malformed request errors
    """
    if status_code is not None:
        if status_code in RETRYABLE_HTTP_STATUSES:
            return True
        if status_code in NON_RETRYABLE_HTTP_STATUSES:
            return False

    err_str = str(error_code_or_msg).upper()
    
    for code in RETRYABLE_HTTP_STATUSES:
        if f"HTTP_{code}" in err_str:
            return True
            
    for code in NON_RETRYABLE_HTTP_STATUSES:
        if f"HTTP_{code}" in err_str:
            return False
            
    if any(k in err_str for k in ("TIMEOUT", "CONNECTION", "NETWORK", "DNS", "502", "503", "504", "429")):
        return True
        
    return False


def calculate_backoff(
    attempt_count: int,
    base_seconds: float = 5.0,
    max_seconds: float = 300.0,
    retry_after: Optional[float] = None
) -> float:
    """
    Calculates backoff delay in seconds for the next attempt.
    If retry_after is provided (e.g. from HTTP 429), respects it up to max_seconds.
    Otherwise uses conservative exponential/staged backoff:
      attempt 1 failed -> attempt 2: 5s
      attempt 2 failed -> attempt 3: 15s
      attempt 3 failed -> attempt 4: 60s
      attempt 4 failed -> attempt 5: 300s
    """
    if retry_after is not None:
        try:
            val = float(retry_after)
            if val > 0:
                return min(val, max_seconds)
        except (ValueError, TypeError):
            pass

    schedule = {
        1: base_seconds,          # 5s
        2: base_seconds * 3.0,    # 15s
        3: base_seconds * 12.0,   # 60s
        4: max_seconds,           # 300s
    }
    delay = schedule.get(attempt_count, min(base_seconds * (2 ** max(0, attempt_count - 1)), max_seconds))
    return min(delay, max_seconds)


def get_iso_after(seconds: float) -> str:
    """Returns an ISO 8601 UTC timestamp offset by seconds from now."""
    dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return dt.isoformat()


class WhatsAppNotificationService:
    """
    Reliable asynchronous notification management service for WhatsApp alerts.
    Provides persistent retries, exponential backoff, restart recovery, and offline safety.
    """

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
        provider: Optional[WhatsAppProvider] = None,
        db_path: Optional[str] = None
    ):
        self.config = config or NotificationConfig.from_env()
        self.db_path = db_path or self.config.db_path or DEFAULT_DB_PATH
        self.event_repo = FallEventRepository(self.db_path)
        
        # Determine provider instance
        if provider is not None:
            self.provider = provider
        elif not self.config.enabled:
            self.provider = None
        elif self.config.provider in ("fake", "mock"):
            self.provider = FakeWhatsAppProvider()
        else:
            self.provider = TwilioWhatsAppProvider(
                account_sid=self.config.account_sid or "",
                auth_token=self.config.auth_token or "",
                from_number=self.config.whatsapp_from or "",
                timeout_seconds=self.config.timeout_seconds
            )

        # In-memory thread safety and deduplication
        self._lock = threading.Lock()
        self._in_memory_seen: Set[str] = set()
        self._in_flight_events: Dict[str, FallEvent] = {}
        
        # Async bounded work queue
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)
        self._stop_event = threading.Event()
        
        # Persistent storage initialization and schema migration
        self._init_db()
        self._load_existing_deliveries()

        # Recover pending / queued / interrupted notifications from previous process run
        self._recover_on_startup()

        # Start daemon background worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="WhatsAppNotificationWorker",
            daemon=True
        )
        self._worker_thread.start()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection with WAL journal mode."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """
        Creates notification_deliveries table additively and runs schema migration.
        Leaves existing tables (fall_events) and schemas 100% untouched.
        """
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notification_deliveries (
                        event_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        recipient_masked TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        max_attempts INTEGER NOT NULL DEFAULT 5,
                        last_attempt_at TEXT,
                        next_attempt_at TEXT,
                        dispatched_at TEXT,
                        error_message TEXT,
                        last_error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_status ON notification_deliveries(status);")

                # Inspect columns for safe additive migration from Stage 6I schema
                existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(notification_deliveries)").fetchall()}
                if "attempt_count" not in existing_cols:
                    conn.execute("ALTER TABLE notification_deliveries ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0;")
                if "max_attempts" not in existing_cols:
                    conn.execute("ALTER TABLE notification_deliveries ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 5;")
                if "last_attempt_at" not in existing_cols:
                    conn.execute("ALTER TABLE notification_deliveries ADD COLUMN last_attempt_at TEXT;")
                if "next_attempt_at" not in existing_cols:
                    conn.execute("ALTER TABLE notification_deliveries ADD COLUMN next_attempt_at TEXT;")
                if "last_error" not in existing_cols:
                    conn.execute("ALTER TABLE notification_deliveries ADD COLUMN last_error TEXT;")
                conn.commit()
        except Exception as e:
            logger.error("Failed to initialize notification_deliveries table: %s", e)

    def _load_existing_deliveries(self) -> None:
        """Populates in-memory deduplication cache from persisted database records."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT event_id FROM notification_deliveries").fetchall()
                with self._lock:
                    for r in rows:
                        self._in_memory_seen.add(r["event_id"])
        except Exception as e:
            logger.warning("Could not pre-load notification deliveries into cache: %s", e)

    def _recover_on_startup(self) -> None:
        """
        Scans SQLite for uncompleted notifications across process restarts:
        - QUEUED: Enqueue for delivery.
        - SENDING: Crashed mid-attempt; treat as unconfirmed attempt and re-evaluate.
        - RETRY_WAIT: Check eligibility for retry.
        Notifications marked SENT are NEVER re-enqueued.
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT event_id, status, attempt_count, max_attempts, next_attempt_at
                    FROM notification_deliveries
                    WHERE status IN ('QUEUED', 'SENDING', 'RETRY_WAIT')
                """).fetchall()

            for r in rows:
                event_id = r["event_id"]
                status = r["status"]
                attempt_count = r["attempt_count"]
                max_attempts = r["max_attempts"]

                if attempt_count >= max_attempts:
                    logger.warning("Restart recovery: event %s exceeded max attempts (%d); marking DEAD_LETTER", event_id, max_attempts)
                    self._update_delivery_status(event_id, NotificationStatus.DEAD_LETTER, error_msg="Max attempts exceeded before restart")
                    continue

                event = self.event_repo.get(event_id)
                if event is None:
                    logger.warning("Restart recovery: Could not find FallEvent record for event_id=%s", event_id)
                    continue

                with self._lock:
                    self._in_flight_events[event_id] = event

                if status == NotificationStatus.SENDING.value:
                    # Crashed during send; treat as interrupted attempt
                    logger.info("Restart recovery: Recovering interrupted SENDING event=%s", event_id)
                    if self.config.max_attempts <= 1:
                        self._update_delivery_status(event_id, NotificationStatus.FAILED, error_msg="Interrupted by restart")
                    else:
                        delay = calculate_backoff(attempt_count, self.config.retry_base_seconds, self.config.retry_max_seconds)
                        self._update_delivery_status(
                            event_id,
                            NotificationStatus.RETRY_WAIT,
                            next_attempt_at=get_iso_after(delay),
                            error_msg="Interrupted by restart"
                        )
                elif status == NotificationStatus.QUEUED.value:
                    logger.info("Restart recovery: Re-queueing QUEUED event=%s", event_id)
                    try:
                        self._queue.put_nowait(event)
                    except queue.Full:
                        pass
                elif status == NotificationStatus.RETRY_WAIT.value:
                    now_ts = get_utc_iso_now()
                    next_at = r["next_attempt_at"]
                    if not next_at or next_at <= now_ts:
                        logger.info("Restart recovery: Re-queueing eligible RETRY_WAIT event=%s", event_id)
                        self._update_delivery_status(event_id, NotificationStatus.QUEUED)
                        try:
                            self._queue.put_nowait(event)
                        except queue.Full:
                            pass
        except Exception as e:
            logger.error("Error during restart recovery: %s", e)

    def is_event_processed(self, event_id: str) -> bool:
        """Thread-safe check whether event_id has already been registered."""
        with self._lock:
            if event_id in self._in_memory_seen:
                return True

        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM notification_deliveries WHERE event_id = ?",
                    (event_id,)
                ).fetchone()
                if row:
                    with self._lock:
                        self._in_memory_seen.add(event_id)
                    return True
        except Exception as e:
            logger.error("Database check failed during deduplication query: %s", e)

        return False

    def enqueue_event(self, event: FallEvent) -> NotificationStatus:
        """
        Schedules a WhatsApp notification for a confirmed FallEvent.
        Non-blocking: Enqueues and returns immediately.
        
        Guarantees:
        - Only confirmed falls are accepted.
        - Idempotent: repeated calls for the same event_id return DUPLICATE_SUPPRESSED.
        - If notifications are disabled, returns DISABLED without error.
        - Bounded queue overflow: State safely persisted in SQLite, never dropped.
        - Failure to enqueue never raises an exception to the caller.
        """
        # Gating check: Must be a confirmed fall event
        if event.status != FallEventStatus.CONFIRMED:
            logger.debug("Rejecting notification: event %s has status %s", event.event_id, event.status)
            return NotificationStatus.REJECTED

        # Configuration check: Must be enabled
        if not self.config.enabled or self.provider is None:
            logger.debug("Notification skipped: WhatsApp service is DISABLED.")
            return NotificationStatus.DISABLED

        event_id = event.event_id

        # Thread-safe duplicate check and atomic registration
        with self._lock:
            if event_id in self._in_memory_seen:
                logger.info("Notification suppressed: Duplicate event_id=%s", event_id)
                return NotificationStatus.DUPLICATE_SUPPRESSED
            # Mark as seen in-memory immediately to prevent race conditions
            self._in_memory_seen.add(event_id)
            self._in_flight_events[event_id] = event

        # Record initial QUEUED state in SQLite
        now_ts = get_utc_iso_now()
        masked_to = mask_phone_number(self.config.whatsapp_recipient)
        provider_name = self.config.provider
        max_attempts = self.config.max_attempts

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO notification_deliveries (
                        event_id, status, provider, recipient_masked,
                        attempt_count, max_attempts, last_attempt_at, next_attempt_at,
                        dispatched_at, error_message, last_error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    NotificationStatus.QUEUED.value,
                    provider_name,
                    masked_to,
                    0,
                    max_attempts,
                    None,
                    None,
                    None,
                    None,
                    None,
                    now_ts,
                    now_ts
                ))
                conn.commit()
        except sqlite3.IntegrityError:
            # Another worker or process already persisted this event_id
            logger.info("Notification suppressed by database constraint: Duplicate event_id=%s", event_id)
            return NotificationStatus.DUPLICATE_SUPPRESSED
        except Exception as e:
            logger.error("Failed to persist notification QUEUED state for event=%s: %s", event_id, e)

        # Place task in bounded queue (non-blocking)
        try:
            self._queue.put_nowait(event)
            logger.info("Notification QUEUED for event_id=%s to recipient=%s", event_id, masked_to)
            return NotificationStatus.QUEUED
        except queue.Full:
            # Queue Overflow Safety: Keep in SQLite as QUEUED; background loop will pick it up
            logger.warning(
                "Notification queue is full! Event_id=%s retained in persistent SQLite storage for scanner pickup.",
                event_id
            )
            return NotificationStatus.QUEUED

    def _worker_loop(self) -> None:
        """Background daemon worker consuming queued notifications and managing retries."""
        logger.debug("Notification background worker started.")
        last_scan_time = 0.0

        while not self._stop_event.is_set():
            # 1. Consume from in-memory queue
            try:
                item = self._queue.get(timeout=0.25)
                if isinstance(item, str):
                    event = self.event_repo.get(item) or self._in_flight_events.get(item)
                else:
                    event = item

                if event:
                    try:
                        self._process_notification(event)
                    except Exception as e:
                        logger.error("Unexpected worker error processing event=%s: %s", event.event_id, e)
                self._queue.task_done()
            except queue.Empty:
                pass

            # 2. Periodic SQLite scan for due retries or queue overflow recovery (every 0.5s)
            now_t = time.time()
            if now_t - last_scan_time >= 0.5:
                last_scan_time = now_t
                try:
                    self._scan_and_requeue_pending()
                except Exception as e:
                    logger.error("Error during retry scan loop: %s", e)

        logger.debug("Notification background worker stopped.")

    def _scan_and_requeue_pending(self) -> None:
        """Queries SQLite for RETRY_WAIT items whose next_attempt_at is due, or overflow QUEUED items."""
        now_ts = get_utc_iso_now()
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT event_id, status, attempt_count, max_attempts, next_attempt_at
                FROM notification_deliveries
                WHERE (status = 'RETRY_WAIT' AND next_attempt_at <= ?)
                   OR (status = 'QUEUED')
                LIMIT 20
            """, (now_ts,)).fetchall()

        for r in rows:
            event_id = r["event_id"]
            attempt_count = r["attempt_count"]
            max_attempts = r["max_attempts"]

            if attempt_count >= max_attempts:
                self._update_delivery_status(event_id, NotificationStatus.DEAD_LETTER, error_msg="Max attempts reached")
                continue

            event = self._in_flight_events.get(event_id) or self.event_repo.get(event_id)
            if not event:
                continue

            try:
                self._queue.put_nowait(event)
                # If was RETRY_WAIT, mark as QUEUED
                if r["status"] == NotificationStatus.RETRY_WAIT.value:
                    self._update_delivery_status(event_id, NotificationStatus.QUEUED)
            except queue.Full:
                break

    def _process_notification(self, event: FallEvent) -> None:
        """Executes message formatting, provider dispatch, error classification, and retry scheduling."""
        event_id = event.event_id

        # Inspect current persistent record
        record = self.get_delivery_record(event_id)
        if not record:
            return

        # Do not re-send if already completed
        if record["status"] == NotificationStatus.SENT.value:
            return

        attempt_count = record.get("attempt_count", 0) + 1
        max_attempts = record.get("max_attempts", self.config.max_attempts)
        now_ts = get_utc_iso_now()

        # Update to SENDING status and record attempt
        self._update_delivery_status(
            event_id=event_id,
            status=NotificationStatus.SENDING,
            attempt_count=attempt_count,
            last_attempt_at=now_ts
        )

        # Format deterministic message payload
        try:
            body = format_fall_message(event)
        except Exception as e:
            logger.error("Message formatting failed for event=%s: %s", event_id, e)
            self._update_delivery_status(
                event_id=event_id,
                status=NotificationStatus.FAILED,
                error_msg=f"FormattingError: {e}"
            )
            return

        recipient = self.config.whatsapp_recipient or ""
        masked_to = mask_phone_number(recipient)

        # Provider invocation with isolated failure handling
        try:
            success, msg_id_or_err, diag = self.provider.send_message(
                recipient=recipient,
                message=body,
                event_id=event_id
            )
            status_code = diag.get("status_code")
            retry_after = diag.get("retry_after")

            if success:
                logger.info(
                    "WhatsApp message delivered successfully for event=%s (ref=%s, attempt=%d/%d)",
                    event_id,
                    msg_id_or_err,
                    attempt_count,
                    max_attempts
                )
                self._update_delivery_status(
                    event_id=event_id,
                    status=NotificationStatus.SENT,
                    dispatched_at=get_utc_iso_now(),
                    error_msg=None
                )
            else:
                retryable = is_retryable_error(msg_id_or_err, status_code)
                logger.warning(
                    "WhatsApp dispatch failed for event=%s [error=%s, retryable=%s, attempt=%d/%d, to=%s]",
                    event_id,
                    msg_id_or_err,
                    retryable,
                    attempt_count,
                    max_attempts,
                    masked_to
                )

                if self.config.max_attempts <= 1:
                    # Single-attempt mode (Stage 6I compatibility)
                    self._update_delivery_status(
                        event_id=event_id,
                        status=NotificationStatus.FAILED,
                        error_msg=msg_id_or_err
                    )
                elif not retryable:
                    # Non-retryable permanent error (HTTP 400, 401, 403, 404)
                    self._update_delivery_status(
                        event_id=event_id,
                        status=NotificationStatus.FAILED,
                        error_msg=msg_id_or_err
                    )
                elif attempt_count >= max_attempts:
                    # Retryable error, but exhausted all attempts -> DEAD_LETTER
                    logger.error(
                        "WhatsApp delivery reached DEAD_LETTER for event=%s after %d attempts",
                        event_id,
                        attempt_count
                    )
                    self._update_delivery_status(
                        event_id=event_id,
                        status=NotificationStatus.DEAD_LETTER,
                        error_msg=f"Exhausted {max_attempts} attempts: {msg_id_or_err}"
                    )
                else:
                    # Retryable error and attempts remain -> RETRY_WAIT
                    backoff = calculate_backoff(
                        attempt_count=attempt_count,
                        base_seconds=self.config.retry_base_seconds,
                        max_seconds=self.config.retry_max_seconds,
                        retry_after=retry_after
                    )
                    next_attempt_at = get_iso_after(backoff)
                    logger.info(
                        "Scheduling retry for event=%s at %s (+%.1fs)",
                        event_id,
                        next_attempt_at,
                        backoff
                    )
                    self._update_delivery_status(
                        event_id=event_id,
                        status=NotificationStatus.RETRY_WAIT,
                        next_attempt_at=next_attempt_at,
                        error_msg=msg_id_or_err
                    )

        except Exception as e:
            err_name = type(e).__name__
            logger.error("Unhandled exception calling WhatsApp provider for event=%s: %s", event_id, err_name)
            
            if self.config.max_attempts <= 1:
                self._update_delivery_status(
                    event_id=event_id,
                    status=NotificationStatus.FAILED,
                    error_msg=err_name
                )
            elif attempt_count >= max_attempts:
                self._update_delivery_status(
                    event_id=event_id,
                    status=NotificationStatus.DEAD_LETTER,
                    error_msg=f"Exhausted {max_attempts} attempts: {err_name}"
                )
            else:
                backoff = calculate_backoff(
                    attempt_count=attempt_count,
                    base_seconds=self.config.retry_base_seconds,
                    max_seconds=self.config.retry_max_seconds
                )
                self._update_delivery_status(
                    event_id=event_id,
                    status=NotificationStatus.RETRY_WAIT,
                    next_attempt_at=get_iso_after(backoff),
                    error_msg=err_name
                )

    def _update_delivery_status(
        self,
        event_id: str,
        status: NotificationStatus,
        attempt_count: Optional[int] = None,
        last_attempt_at: Optional[str] = None,
        next_attempt_at: Optional[str] = None,
        dispatched_at: Optional[str] = None,
        error_msg: Optional[str] = None
    ) -> None:
        """Updates delivery status and audit columns in the SQLite notification_deliveries table."""
        now_ts = get_utc_iso_now()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE notification_deliveries
                    SET status = ?,
                        attempt_count = COALESCE(?, attempt_count),
                        last_attempt_at = COALESCE(?, last_attempt_at),
                        next_attempt_at = ?,
                        dispatched_at = COALESCE(?, dispatched_at),
                        error_message = ?,
                        last_error = ?,
                        updated_at = ?
                    WHERE event_id = ?
                """, (
                    status.value,
                    attempt_count,
                    last_attempt_at,
                    next_attempt_at,
                    dispatched_at,
                    error_msg,
                    error_msg,
                    now_ts,
                    event_id
                ))
                conn.commit()
        except Exception as e:
            logger.error("Failed to update notification status for event=%s: %s", event_id, e)

    def get_delivery_record(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Queries delivery status record for testing and diagnostics."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM notification_deliveries WHERE event_id = ?",
                    (event_id,)
                ).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error("Failed to query notification delivery record for event=%s: %s", event_id, e)
            return None

    def join_queue(self, timeout: float = 5.0) -> None:
        """Blocks until all items currently in the in-memory queue have been processed."""
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._queue.join()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Gracefully signals the worker thread to exit."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)


_global_service: Optional[WhatsAppNotificationService] = None
_global_lock = threading.Lock()


def get_global_notification_service() -> WhatsAppNotificationService:
    """Returns singleton WhatsAppNotificationService instance."""
    global _global_service
    with _global_lock:
        if _global_service is None:
            _global_service = WhatsAppNotificationService()
        return _global_service


def reset_global_notification_service(new_service: Optional[WhatsAppNotificationService] = None) -> None:
    """Resets global service singleton (used primarily in test teardown)."""
    global _global_service
    with _global_lock:
        if _global_service is not None:
            _global_service.shutdown()
        _global_service = new_service
