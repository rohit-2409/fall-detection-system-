"""
Stage 5A Fall Event Backend Package.
Standalone Fall Detection Project.
"""

from src.fall_event_backend.models import (
    FallEvent,
    FallEventStatus,
    InvalidStateTransitionError,
    DuplicateEventError,
    EventNotFoundError,
    ALLOWED_TRANSITIONS
)
from src.fall_event_backend.storage import FallEventRepository
from src.fall_event_backend.service import FallEventService
from src.fall_event_backend.integration_adapter import FallEventIntegrationAdapter
from src.fall_event_backend.api import create_app

__all__ = [
    "FallEvent",
    "FallEventStatus",
    "InvalidStateTransitionError",
    "DuplicateEventError",
    "EventNotFoundError",
    "ALLOWED_TRANSITIONS",
    "FallEventRepository",
    "FallEventService",
    "FallEventIntegrationAdapter",
    "create_app"
]
