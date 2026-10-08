"""
Stage 5A Integration Adapter: Connecting Detector CONFIRMED_FALL to FallEvent Backend.
Standalone Fall Detection Project.
"""

from datetime import datetime, timezone
import json
from typing import Dict, Any, Optional, Tuple, Callable

from src.fall_event_backend.models import FallEvent, get_utc_iso_now
from src.fall_event_backend.service import FallEventService


class FallEventIntegrationAdapter:
    """
    Lightweight, non-intrusive adapter that receives CONFIRMED_FALL notifications
    from the Stage 4F/4G detector engine and dispatches them to the FallEvent backend
    while strictly preserving the frozen detector architecture.
    """

    def __init__(self, service: Optional[FallEventService] = None):
        self.service = service or FallEventService()

    def process_detector_event(
        self,
        event_dict: Dict[str, Any],
        session_id: Optional[str] = None,
        source: str = "ESP32_MPU6050",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[FallEvent, bool]:
        """
        Translates raw detector confirmation output into a persistent FallEvent.

        Expected detector dictionary keys:
            - 'event_id': Optional string (defaults to deterministic synthesis)
            - 'timestamp': Float seconds or string
            - 'probability': Float (0.0 to 1.0)
            - 'state': String ("CONFIRMED_FALL")
            - 'positive_count': Integer (e.g. 3)

        Returns:
            Tuple[FallEvent, bool]: (event, is_newly_created)
        """
        now_iso = get_utc_iso_now()
        sid = session_id or event_dict.get("session_id", "LIVE_SESSION")
        
        # Explicit deterministic event_id for idempotency
        evt_id = event_dict.get("event_id")
        if not evt_id:
            ts_val = event_dict.get("timestamp", 0.0)
            evt_id = f"EVT_{sid}_{int(round(float(ts_val) * 1000))}"

        occurred_ts = event_dict.get("timestamp")
        occurred_str = f"{float(occurred_ts):.4f}s" if occurred_ts is not None else None

        prob = event_dict.get("probability")
        confidence = float(prob) if prob is not None else None

        diag_meta = {
            "temporal_state": event_dict.get("state", "CONFIRMED_FALL"),
            "positive_count": event_dict.get("positive_count", 3),
            "pre_event_activity": event_dict.get("pre_event_activity", "UNKNOWN")
        }
        if additional_metadata:
            diag_meta.update(additional_metadata)

        return self.service.record_confirmed_fall(
            event_id=evt_id,
            detected_at=now_iso,
            occurred_at=occurred_str,
            session_id=sid,
            confidence=confidence,
            source=source,
            diagnostic_metadata=diag_meta
        )

    def create_detector_callback(self) -> Callable[[Dict[str, Any]], Tuple[FallEvent, bool]]:
        """
        Returns a callable callback function that can be plugged into any live
        stream runner without modifying its internal loops.
        """
        def callback(detector_output: Dict[str, Any]):
            return self.process_detector_event(detector_output)
        return callback
