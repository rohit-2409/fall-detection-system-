"""
Stage 6K Notification Monitoring & Observability Service.
Standalone Fall Detection Project.

Strict Safety & Performance Rules:
- READ-ONLY SELECT and SQL aggregations only. Zero mutations.
- Uses existing notification_deliveries table as single source of truth.
- Zero secrets, zero auth tokens, zero headers, zero unmasked phone numbers exposed.
- Lightweight queries with LIMIT clauses; will not overload SQLite.
- Safe fallback behavior if database or notification service is temporarily unavailable.
"""

import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from src.fall_event_backend.storage import DEFAULT_DB_PATH
from src.notifications.notification_config import mask_phone_number
from src.notifications.whatsapp_service import (
    NotificationStatus,
    WhatsAppNotificationService,
    get_global_notification_service
)

logger = logging.getLogger("Stage6K_NotificationMonitor")


class NotificationMonitorService:
    """
    Provides read-only operational metrics and delivery telemetry
    for the notification reliability dashboard.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        notification_service: Optional[WhatsAppNotificationService] = None
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.notification_service = notification_service

    def _get_connection(self) -> sqlite3.Connection:
        """Returns read-only configured SQLite connection with row factory."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def get_summary_counts(self) -> Dict[str, int]:
        """
        Executes lightweight SQL aggregation to calculate notification lifecycle counts.
        """
        counts = {
            "total": 0,
            "sent": 0,
            "queued": 0,
            "sending": 0,
            "retry_wait": 0,
            "failed": 0,
            "dead_letter": 0,
            "duplicate_suppressed": 0,
            "disabled": 0
        }
        try:
            with self._get_connection() as conn:
                # Check if table exists
                tbl = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='notification_deliveries';"
                ).fetchone()
                if not tbl:
                    return counts

                rows = conn.execute(
                    "SELECT status, COUNT(*) as cnt FROM notification_deliveries GROUP BY status;"
                ).fetchall()

                total = 0
                for r in rows:
                    st = (r["status"] or "").lower()
                    cnt = int(r["cnt"])
                    total += cnt
                    if st in counts:
                        counts[st] = cnt

                counts["total"] = total
        except Exception as e:
            logger.error("Failed to query notification summary counts: %s", e)

        return counts

    def get_queue_health(self) -> Dict[str, Any]:
        """
        Calculates queue capacity, in-flight pending count, and operational health rating.
        """
        pending = 0
        capacity = 100

        svc = self.notification_service or get_global_notification_service()
        if svc and hasattr(svc, "_queue"):
            try:
                pending = svc._queue.qsize()
                capacity = svc.config.queue_maxsize
            except Exception:
                pass
        else:
            # Fallback estimation from database
            try:
                with self._get_connection() as conn:
                    row = conn.execute("""
                        SELECT COUNT(*) as cnt FROM notification_deliveries
                        WHERE status IN ('QUEUED', 'SENDING')
                    """).fetchone()
                    if row:
                        pending = int(row["cnt"])
            except Exception:
                pass

        if pending >= capacity:
            health = "CRITICAL (OVERFLOW)"
        elif pending >= int(capacity * 0.8):
            health = "ATTENTION REQUIRED"
        else:
            health = "HEALTHY"

        return {
            "pending": pending,
            "capacity": capacity,
            "health": health
        }

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Returns sanitized, non-sensitive provider operational metadata.
        Never exposes auth tokens, API keys, or raw configurations.
        """
        provider_name = "twilio"
        configured = False
        status_text = "Disabled"

        svc = self.notification_service or get_global_notification_service()
        if svc and hasattr(svc, "config"):
            provider_name = svc.config.provider
            configured = bool(svc.config.enabled)
            status_text = "Healthy" if configured else "Disabled"
            if svc.provider is None and configured:
                status_text = "Attention Required"

        return {
            "name": provider_name,
            "configured": configured,
            "status": status_text
        }

    def get_deliveries(
        self,
        status: Optional[str] = None,
        event_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Fetches paginated, filterable notification deliveries.
        Guarantees phone numbers remain masked and zero secrets are returned.
        """
        limit = max(1, min(200, int(limit)))
        offset = max(0, int(offset))

        clauses = []
        params = []

        if status:
            clauses.append("UPPER(status) = ?")
            params.append(status.strip().upper())

        if event_id:
            clauses.append("event_id = ?")
            params.append(event_id.strip())

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        items: List[Dict[str, Any]] = []
        total_count = 0

        try:
            with self._get_connection() as conn:
                tbl = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='notification_deliveries';"
                ).fetchone()
                if not tbl:
                    return [], 0

                count_row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM notification_deliveries {where_sql};",
                    params
                ).fetchone()
                total_count = int(count_row["cnt"]) if count_row else 0

                query_params = list(params) + [limit, offset]
                rows = conn.execute(f"""
                    SELECT event_id, status, provider, recipient_masked,
                           attempt_count, max_attempts, last_attempt_at, next_attempt_at,
                           dispatched_at, error_message, last_error, created_at, updated_at
                    FROM notification_deliveries
                    {where_sql}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?;
                """, query_params).fetchall()

                for r in rows:
                    row_dict = dict(r)
                    # Enforce strict phone number masking on all outputs
                    row_dict["recipient_masked"] = mask_phone_number(row_dict.get("recipient_masked"))
                    # Map error message cleanly
                    if not row_dict.get("last_error"):
                        row_dict["last_error"] = row_dict.get("error_message")
                    items.append(row_dict)

        except Exception as e:
            logger.error("Failed to query notification deliveries: %s", e)

        return items, total_count

    def get_event_notification(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves notification delivery state for a specific fall event ID.
        """
        if not event_id:
            return None

        try:
            with self._get_connection() as conn:
                tbl = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='notification_deliveries';"
                ).fetchone()
                if not tbl:
                    return None

                row = conn.execute("""
                    SELECT event_id, status, provider, recipient_masked,
                           attempt_count, max_attempts, last_attempt_at, next_attempt_at,
                           dispatched_at, error_message, last_error, created_at, updated_at
                    FROM notification_deliveries
                    WHERE event_id = ?;
                """, (event_id.strip(),)).fetchone()

                if row:
                    row_dict = dict(row)
                    row_dict["recipient_masked"] = mask_phone_number(row_dict.get("recipient_masked"))
                    if not row_dict.get("last_error"):
                        row_dict["last_error"] = row_dict.get("error_message")
                    return row_dict
        except Exception as e:
            logger.error("Failed to query event notification for event_id=%s: %s", event_id, e)

        return None

    def get_monitoring_snapshot(self) -> Dict[str, Any]:
        """
        Aggregates complete operational monitoring telemetry in a single payload.
        """
        summary = self.get_summary_counts()
        queue_info = self.get_queue_health()
        provider_info = self.get_provider_info()
        recent_items, _ = self.get_deliveries(limit=10, offset=0)

        return {
            "status": "ok",
            "summary": summary,
            "queue": queue_info,
            "provider": provider_info,
            "recent_activity": recent_items
        }
