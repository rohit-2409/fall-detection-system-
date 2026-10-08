"""
Stage 8A Structured Operational Event Logger.
Standalone Fall Detection System.

Logs production operational transitions cleanly (e.g., SYSTEM_READY, ESP32_CONNECTED)
without flooding console logs during 100 Hz continuous sensor streaming.
"""

import datetime
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("SystemHealth")


class OperationalLogger:
    """
    Lightweight, thread-safe operational event logger with rate-limiting
    and deduplication for high-frequency state updates.
    """

    def __init__(self, log_level: int = logging.INFO):
        self._last_events: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._logger = logger

    def log_event(self, event_name: str, detail: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """
        Logs a structured system event if the state or detail has changed,
        preventing console flooding.
        """
        with self._lock:
            state_key = f"{event_name}:{detail or ''}"
            last = self._last_events.get(event_name)
            if last == state_key:
                return  # Suppress duplicate rapid transitions
            self._last_events[event_name] = state_key

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        msg = f"[SYSTEM_EVENT] {event_name}"
        if detail:
            msg += f" | {detail}"
        if extra:
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            msg += f" ({extra_str})"

        print(msg, flush=True)
        self._logger.info("%s - %s", event_name, detail or "")


_GLOBAL_LOGGER = OperationalLogger()


def get_operational_logger() -> OperationalLogger:
    """Returns the singleton operational logger instance."""
    return _GLOBAL_LOGGER
