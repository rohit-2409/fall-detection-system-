"""
Stage 5A Fall Event Data Models and Lifecycle Constants.
Standalone Fall Detection Project.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional


class FallEventStatus(str, Enum):
    """Permitted lifecycle status values for a FallEvent."""
    CONFIRMED = "CONFIRMED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CANCELLED = "CANCELLED"
    RESOLVED = "RESOLVED"


# Allowed state transition graph
ALLOWED_TRANSITIONS = {
    FallEventStatus.CONFIRMED: {FallEventStatus.ACKNOWLEDGED, FallEventStatus.CANCELLED},
    FallEventStatus.ACKNOWLEDGED: {FallEventStatus.RESOLVED},
    FallEventStatus.CANCELLED: set(),  # Terminal state
    FallEventStatus.RESOLVED: set(),   # Terminal state
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal lifecycle transition is requested."""
    pass


class DuplicateEventError(ValueError):
    """Raised when attempting to create an event with a conflicting event_id."""
    pass


class EventNotFoundError(KeyError):
    """Raised when an event_id is not found in the persistence store."""
    pass


def get_utc_iso_now() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FallEvent:
    """
    Core FallEvent entity representing a detected and confirmed physical fall.
    """
    event_id: str
    detected_at: str
    occurred_at: Optional[str] = None
    session_id: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "ESP32_MPU6050"
    status: FallEventStatus = FallEventStatus.CONFIRMED
    created_at: str = field(default_factory=get_utc_iso_now)
    updated_at: str = field(default_factory=get_utc_iso_now)
    diagnostic_metadata: Optional[Dict[str, Any]] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    cancelled_at: Optional[str] = None
    cancelled_by: Optional[str] = None
    cancellation_reason: Optional[str] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None

    def can_transition_to(self, target_status: FallEventStatus) -> bool:
        """Checks whether the requested status transition is legally permitted."""
        return target_status in ALLOWED_TRANSITIONS.get(self.status, set())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the FallEvent to a standard JSON-compatible dictionary."""
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, FallEventStatus) else str(self.status)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FallEvent":
        """Instantiates a FallEvent from a dictionary with validation."""
        raw_status = data.get("status", FallEventStatus.CONFIRMED)
        status = FallEventStatus(raw_status) if isinstance(raw_status, str) else raw_status
        return cls(
            event_id=data["event_id"],
            detected_at=data["detected_at"],
            occurred_at=data.get("occurred_at"),
            session_id=data.get("session_id"),
            confidence=data.get("confidence"),
            source=data.get("source", "ESP32_MPU6050"),
            status=status,
            created_at=data.get("created_at", get_utc_iso_now()),
            updated_at=data.get("updated_at", get_utc_iso_now()),
            diagnostic_metadata=data.get("diagnostic_metadata"),
            acknowledged_at=data.get("acknowledged_at"),
            acknowledged_by=data.get("acknowledged_by"),
            cancelled_at=data.get("cancelled_at"),
            cancelled_by=data.get("cancelled_by"),
            cancellation_reason=data.get("cancellation_reason"),
            resolved_at=data.get("resolved_at"),
            resolved_by=data.get("resolved_by"),
        )
