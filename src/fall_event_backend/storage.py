"""
Stage 5A SQLite Persistence Engine for Fall Events.
Standalone Fall Detection Project.
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple, Any

from src.fall_event_backend.models import (
    FallEvent,
    FallEventStatus,
    EventNotFoundError,
    get_utc_iso_now
)

DEFAULT_DB_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "fall_detection_dataset",
    "fall_events.db"
)


class FallEventRepository:
    """
    Thread-safe SQLite repository managing persistent FallEvent records with ACID
    transactions and unique constraint-based idempotency.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns an SQLite connection configured with Row factory."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Initializes database schema if not present."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fall_events (
                    event_id TEXT PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    occurred_at TEXT,
                    session_id TEXT,
                    confidence REAL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    diagnostic_metadata TEXT,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    cancelled_at TEXT,
                    cancelled_by TEXT,
                    cancellation_reason TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON fall_events(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON fall_events(session_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON fall_events(created_at);")
            conn.commit()

    def _row_to_event(self, row: sqlite3.Row) -> FallEvent:
        """Constructs a FallEvent from an SQLite row."""
        meta_raw = row["diagnostic_metadata"]
        meta_dict = json.loads(meta_raw) if meta_raw else None
        return FallEvent(
            event_id=row["event_id"],
            detected_at=row["detected_at"],
            occurred_at=row["occurred_at"],
            session_id=row["session_id"],
            confidence=row["confidence"],
            source=row["source"],
            status=FallEventStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            diagnostic_metadata=meta_dict,
            acknowledged_at=row["acknowledged_at"],
            acknowledged_by=row["acknowledged_by"],
            cancelled_at=row["cancelled_at"],
            cancelled_by=row["cancelled_by"],
            cancellation_reason=row["cancellation_reason"],
            resolved_at=row["resolved_at"],
            resolved_by=row["resolved_by"]
        )

    def create(self, event: FallEvent) -> Tuple[FallEvent, bool]:
        """
        Persists a FallEvent.
        Enforces idempotency: If event_id already exists, returns the existing record
        with is_created=False without altering existing timestamps or state.
        
        Returns:
            Tuple[FallEvent, bool]: (event_instance, is_newly_created)
        """
        existing = self.get(event.event_id)
        if existing is not None:
            return existing, False

        meta_json = json.dumps(event.diagnostic_metadata) if event.diagnostic_metadata else None
        now_ts = get_utc_iso_now()
        created_at = event.created_at or now_ts
        updated_at = event.updated_at or now_ts

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO fall_events (
                        event_id, detected_at, occurred_at, session_id,
                        confidence, source, status, created_at, updated_at,
                        diagnostic_metadata, acknowledged_at, acknowledged_by,
                        cancelled_at, cancelled_by, cancellation_reason,
                        resolved_at, resolved_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.detected_at,
                    event.occurred_at,
                    event.session_id,
                    event.confidence,
                    event.source,
                    event.status.value,
                    created_at,
                    updated_at,
                    meta_json,
                    event.acknowledged_at,
                    event.acknowledged_by,
                    event.cancelled_at,
                    event.cancelled_by,
                    event.cancellation_reason,
                    event.resolved_at,
                    event.resolved_by
                ))
                conn.commit()
                return self.get(event.event_id), True
            except sqlite3.IntegrityError:
                # Concurrent insertion race condition safeguard
                return self.get(event.event_id), False

    def get(self, event_id: str) -> Optional[FallEvent]:
        """Retrieves a FallEvent by its unique event_id, or None if not found."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM fall_events WHERE event_id = ?", (event_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_event(row)

    def list_all(
        self,
        status: Optional[FallEventStatus] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FallEvent]:
        """Queries FallEvents with optional status and session filtering."""
        query = "SELECT * FROM fall_events WHERE 1=1"
        params: List[Any] = []

        if status is not None:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, FallEventStatus) else str(status))

        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_event(r) for r in cur.fetchall()]

    def update(self, event: FallEvent) -> FallEvent:
        """
        Updates an existing FallEvent with new status or resolution fields.
        Raises EventNotFoundError if the event does not exist.
        """
        existing = self.get(event.event_id)
        if existing is None:
            raise EventNotFoundError(f"Event with id '{event.event_id}' not found.")

        now_ts = get_utc_iso_now()
        meta_json = json.dumps(event.diagnostic_metadata) if event.diagnostic_metadata else None

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE fall_events SET
                    status = ?,
                    updated_at = ?,
                    diagnostic_metadata = ?,
                    acknowledged_at = ?,
                    acknowledged_by = ?,
                    cancelled_at = ?,
                    cancelled_by = ?,
                    cancellation_reason = ?,
                    resolved_at = ?,
                    resolved_by = ?
                WHERE event_id = ?
            """, (
                event.status.value,
                now_ts,
                meta_json,
                event.acknowledged_at,
                event.acknowledged_by,
                event.cancelled_at,
                event.cancelled_by,
                event.cancellation_reason,
                event.resolved_at,
                event.resolved_by,
                event.event_id
            ))
            conn.commit()

        return self.get(event.event_id)
