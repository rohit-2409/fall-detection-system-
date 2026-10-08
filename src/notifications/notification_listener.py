"""
Stage 6I Notification Listener and Backend Integration Hook.
Standalone Fall Detection Project.

Ensures the Database-First Guarantee:
1. Backend receives event.
2. Backend persists event successfully in SQLite.
3. Persistence transaction commits.
4. Only AFTER successful NEW event persistence, WhatsApp notification is scheduled.

Failure Isolation:
- If notification enqueue or delivery fails, the persisted database record remains intact.
- The HTTP backend request completes normally without failing.
- Frozen files (api.py, service.py, storage.py) require ZERO modifications.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from flask import Flask, jsonify, render_template, request

from src.fall_event_backend.api import create_app
from src.fall_event_backend.models import FallEvent
from src.fall_event_backend.service import FallEventService
from src.fall_event_backend.storage import FallEventRepository
from src.notifications.notification_monitor import NotificationMonitorService
from src.notifications.whatsapp_service import (
    WhatsAppNotificationService,
    get_global_notification_service
)

logger = logging.getLogger("Stage6K_NotificationListener")


class NotifyingFallEventService(FallEventService):
    """
    Subclasses FallEventService to inject a clean post-persistence hook
    without altering existing business logic or database transaction integrity.
    """

    def __init__(
        self,
        repository: Optional[FallEventRepository] = None,
        notification_service: Optional[WhatsAppNotificationService] = None
    ):
        super().__init__(repository=repository)
        self.notification_service = (
            notification_service if notification_service is not None
            else get_global_notification_service()
        )

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
        Ingests fall event into database first, then triggers notification if and only if
        the event is newly persisted.
        """
        # STEP 1: Persist into SQLite via parent business logic
        event, is_new = super().record_confirmed_fall(
            event_id=event_id,
            detected_at=detected_at,
            occurred_at=occurred_at,
            session_id=session_id,
            confidence=confidence,
            source=source,
            diagnostic_metadata=diagnostic_metadata
        )

        # STEP 2: Only after successful commit and ONLY if is_new == True
        if is_new and self.notification_service is not None:
            try:
                self.notification_service.enqueue_event(event)
            except Exception as e:
                # Failure isolation: logging only, never fail the backend operation
                logger.error(
                    "Notification enqueue failed for newly persisted event=%s: %s",
                    event_id,
                    e
                )

        return event, is_new


def register_notification_monitoring_routes(
    app: Flask,
    notification_service: Optional[WhatsAppNotificationService] = None,
    db_path: Optional[str] = None
) -> Flask:
    """
    Registers Stage 6K read-only notification reliability & monitoring endpoints.
    Safe, idempotent, and non-destructive.
    """
    monitor_svc = NotificationMonitorService(
        db_path=db_path,
        notification_service=notification_service
    )

    @app.route("/notifications", methods=["GET"])
    def notification_dashboard():
        """Serves the Stage 6K notification reliability monitoring dashboard UI."""
        return render_template("notifications.html")

    @app.route("/api/notifications/monitoring", methods=["GET"])
    def get_notification_monitoring():
        """Aggregated operational statistics and queue/provider health."""
        snapshot = monitor_svc.get_monitoring_snapshot()
        return jsonify(snapshot), 200

    @app.route("/api/notifications/deliveries", methods=["GET"])
    def list_notification_deliveries():
        """Paginated, filterable delivery records."""
        status = request.args.get("status")
        event_id = request.args.get("event_id")
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "limit and offset must be integers."}), 400

        items, total = monitor_svc.get_deliveries(status=status, event_id=event_id, limit=limit, offset=offset)
        return jsonify({
            "status": "ok",
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200

    @app.route("/api/notifications/events/<string:event_id>", methods=["GET"])
    def get_event_notification_status(event_id: str):
        """Per-fall notification delivery status lookup."""
        item = monitor_svc.get_event_notification(event_id)
        if item is None:
            return jsonify({
                "status": "not_found",
                "message": f"No notification delivery record found for event '{event_id}'."
            }), 404
        return jsonify({
            "status": "ok",
            "event_id": event_id,
            "notification": item
        }), 200

    return app


def create_notifying_app(
    service: Optional[NotifyingFallEventService] = None,
    notification_service: Optional[WhatsAppNotificationService] = None,
    repository: Optional[FallEventRepository] = None
) -> Flask:
    """
    Creates a Flask application instance pre-configured with the NotifyingFallEventService
    and Stage 6K notification monitoring endpoints.
    """
    if service is None:
        service = NotifyingFallEventService(
            repository=repository,
            notification_service=notification_service
        )
    app = create_app(service=service)
    db_path = getattr(service.repository, "db_path", None)
    notif_svc = notification_service or getattr(service, "notification_service", None)
    register_notification_monitoring_routes(app, notif_svc, db_path=db_path)
    return app

