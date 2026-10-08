"""
Stage 5A Fall Event Business Logic and Lifecycle Service.
Standalone Fall Detection Project.
"""

from typing import Dict, List, Optional, Tuple, Any

from src.fall_event_backend.models import (
    FallEvent,
    FallEventStatus,
    InvalidStateTransitionError,
    EventNotFoundError,
    get_utc_iso_now
)
from src.fall_event_backend.storage import FallEventRepository


class FallEventService:
    """
    Manages the lifecycle, idempotency, and audit trails of confirmed fall events.
    """

    def __init__(self, repository: Optional[FallEventRepository] = None):
        self.repository = repository or FallEventRepository()

    def record_confirmed_fall(
        self,
        event_id: str,
        detected_at: str,
        occurred_at: Optional[str] = None,
        session_id: Optional[str] = None,
        confidence: Optional[float] = None,
        source: str = "ESP32_MPU6050",
        diagnostic_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[FallEvent, bool]:
        """
        Ingests a confirmed fall event from the detector pipeline.
        
        Idempotency: If event_id already exists, returns the existing record
        with is_created=False, avoiding duplicate database records from network retries,
        duplicate stream callbacks, or client reconnects.
        
        Returns:
            Tuple[FallEvent, bool]: (event, is_newly_created)
        """
        if not event_id or not isinstance(event_id, str):
            raise ValueError("event_id must be a non-empty string.")
        if not detected_at or not isinstance(detected_at, str):
            raise ValueError("detected_at must be a valid timestamp string.")

        event = FallEvent(
            event_id=event_id,
            detected_at=detected_at,
            occurred_at=occurred_at,
            session_id=session_id,
            confidence=confidence,
            source=source,
            status=FallEventStatus.CONFIRMED,
            diagnostic_metadata=diagnostic_metadata
        )

        return self.repository.create(event)

    def acknowledge_event(
        self,
        event_id: str,
        acknowledged_by: Optional[str] = None
    ) -> FallEvent:
        """
        Transitions event from CONFIRMED -> ACKNOWLEDGED.
        Idempotent: If already ACKNOWLEDGED, returns current record without error.
        Raises InvalidStateTransitionError if event is in CANCELLED or RESOLVED state.
        """
        event = self.repository.get(event_id)
        if event is None:
            raise EventNotFoundError(f"Event with id '{event_id}' not found.")

        # Idempotent re-acknowledgement
        if event.status == FallEventStatus.ACKNOWLEDGED:
            return event

        if not event.can_transition_to(FallEventStatus.ACKNOWLEDGED):
            raise InvalidStateTransitionError(
                f"Cannot transition event '{event_id}' from '{event.status.value}' to 'ACKNOWLEDGED'."
            )

        event.status = FallEventStatus.ACKNOWLEDGED
        event.acknowledged_at = get_utc_iso_now()
        event.acknowledged_by = acknowledged_by or "operator"

        return self.repository.update(event)

    def cancel_event(
        self,
        event_id: str,
        cancelled_by: Optional[str] = None,
        reason: Optional[str] = None
    ) -> FallEvent:
        """
        Transitions event from CONFIRMED -> CANCELLED (e.g. false alarm / verified safe).
        Idempotent: If already CANCELLED, returns current record without error.
        Raises InvalidStateTransitionError if event is in ACKNOWLEDGED or RESOLVED state.
        """
        event = self.repository.get(event_id)
        if event is None:
            raise EventNotFoundError(f"Event with id '{event_id}' not found.")

        # Idempotent re-cancellation
        if event.status == FallEventStatus.CANCELLED:
            return event

        if not event.can_transition_to(FallEventStatus.CANCELLED):
            raise InvalidStateTransitionError(
                f"Cannot transition event '{event_id}' from '{event.status.value}' to 'CANCELLED'."
            )

        event.status = FallEventStatus.CANCELLED
        event.cancelled_at = get_utc_iso_now()
        event.cancelled_by = cancelled_by or "operator"
        event.cancellation_reason = reason or "User / Operator cancellation"

        return self.repository.update(event)

    def resolve_event(
        self,
        event_id: str,
        resolved_by: Optional[str] = None
    ) -> FallEvent:
        """
        Transitions event from ACKNOWLEDGED -> RESOLVED.
        Idempotent: If already RESOLVED, returns current record without error.
        Raises InvalidStateTransitionError if event is in CONFIRMED (not yet acknowledged) or CANCELLED state.
        """
        event = self.repository.get(event_id)
        if event is None:
            raise EventNotFoundError(f"Event with id '{event_id}' not found.")

        # Idempotent re-resolution
        if event.status == FallEventStatus.RESOLVED:
            return event

        if not event.can_transition_to(FallEventStatus.RESOLVED):
            raise InvalidStateTransitionError(
                f"Cannot transition event '{event_id}' from '{event.status.value}' to 'RESOLVED'. Event must be ACKNOWLEDGED first."
            )

        event.status = FallEventStatus.RESOLVED
        event.resolved_at = get_utc_iso_now()
        event.resolved_by = resolved_by or "operator"

        return self.repository.update(event)

    def get_event(self, event_id: str) -> Optional[FallEvent]:
        """Fetches a FallEvent by its unique event_id."""
        return self.repository.get(event_id)

    def list_events(
        self,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[FallEvent]:
        """Lists FallEvents with optional filters."""
        status_enum = FallEventStatus(status) if status else None
        return self.repository.list_all(status=status_enum, session_id=session_id, limit=limit, offset=offset)
