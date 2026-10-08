"""
Stage 5A REST API Application for Fall Event Management.
Standalone Fall Detection Project.
"""

import os
from typing import Optional
from flask import Flask, request, jsonify, Response, render_template

from src.fall_event_backend.models import (
    FallEventStatus,
    InvalidStateTransitionError,
    EventNotFoundError
)
from src.fall_event_backend.service import FallEventService


def create_app(service: Optional[FallEventService] = None) -> Flask:
    """Application factory for the Fall Event Backend REST service."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    event_service = service or FallEventService()

    @app.route("/", methods=["GET"])
    @app.route("/dashboard", methods=["GET"])
    def dashboard():
        """Serves the standalone monitoring dashboard UI."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "fall_event_backend",
            "stage": "5B",
            "emergency_notifications": "DISABLED"
        }), 200

    @app.route("/api/system/health", methods=["GET"])
    def system_health_check():
        """Stage 8A Comprehensive System Health and Diagnostics Endpoint."""
        from src.system_health.monitor import get_global_health_monitor
        monitor = get_global_health_monitor()
        return jsonify(monitor.get_status_dict(service=event_service)), 200

    @app.route("/api/system/heartbeat", methods=["POST"])
    def record_system_heartbeat():
        """Stage 8A System Heartbeat Ingestion Endpoint for External Hardware Bridge."""
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Malformed JSON payload or empty request body."
            }), 400

        port = data.get("port")
        baud = data.get("baud")
        gps_available = data.get("gps_available")
        gps_metadata = data.get("gps_metadata")

        if port is not None and not isinstance(port, str):
            return jsonify({"status": "error", "message": "Invalid 'port' format; expected string."}), 400
        if baud is not None:
            try:
                baud = int(baud)
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Invalid 'baud' format; expected integer."}), 400
        if gps_available is not None and not isinstance(gps_available, bool):
            return jsonify({"status": "error", "message": "Invalid 'gps_available' format; expected boolean."}), 400
        if gps_metadata is not None and not isinstance(gps_metadata, dict):
            return jsonify({"status": "error", "message": "Invalid 'gps_metadata' format; expected dictionary."}), 400

        from src.system_health.monitor import get_global_health_monitor
        monitor = get_global_health_monitor()

        monitor.update_esp32_heartbeat(port=port, baud=baud)
        if gps_available is not None:
            monitor.update_gps_state(available=gps_available, metadata=gps_metadata)

        return jsonify({
            "status": "ok",
            "message": "Heartbeat recorded successfully."
        }), 200

    @app.route("/api/fall-events", methods=["POST"])
    def create_fall_event():
        """
        Creates a new FallEvent or idempotently returns the existing record.
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Malformed JSON payload or empty request body."
            }), 400

        event_id = data.get("event_id")
        detected_at = data.get("detected_at")

        if not event_id or not isinstance(event_id, str):
            return jsonify({
                "status": "error",
                "message": "Missing or invalid 'event_id'."
            }), 400

        if not detected_at or not isinstance(detected_at, str):
            return jsonify({
                "status": "error",
                "message": "Missing or invalid 'detected_at'."
            }), 400

        try:
            event, is_new = event_service.record_confirmed_fall(
                event_id=event_id,
                detected_at=detected_at,
                occurred_at=data.get("occurred_at"),
                session_id=data.get("session_id"),
                confidence=data.get("confidence"),
                source=data.get("source", "ESP32_MPU6050"),
                diagnostic_metadata=data.get("diagnostic_metadata")
            )

            status_code = 201 if is_new else 200
            return jsonify({
                "status": "success",
                "is_new": is_new,
                "event": event.to_dict()
            }), status_code

        except ValueError as e:
            return jsonify({"status": "error", "message": str(e)}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": f"Internal server error: {str(e)}"}), 500

    @app.route("/api/fall-events", methods=["GET"])
    def list_fall_events():
        """Lists FallEvents with optional status and session filters."""
        status_param = request.args.get("status")
        session_param = request.args.get("session_id")
        
        try:
            limit = int(request.args.get("limit", 50))
            offset = int(request.args.get("offset", 0))
        except ValueError:
            return jsonify({"status": "error", "message": "limit and offset must be integers."}), 400

        events = event_service.list_events(
            status=status_param,
            session_id=session_param,
            limit=limit,
            offset=offset
        )

        return jsonify({
            "status": "success",
            "count": len(events),
            "events": [e.to_dict() for e in events]
        }), 200

    @app.route("/api/fall-events/<string:event_id>", methods=["GET"])
    def get_fall_event(event_id: str):
        """Fetches an individual FallEvent by event_id."""
        event = event_service.get_event(event_id)
        if event is None:
            return jsonify({
                "status": "error",
                "message": f"Fall event with id '{event_id}' not found."
            }), 404

        return jsonify({
            "status": "success",
            "event": event.to_dict()
        }), 200

    @app.route("/api/fall-events/<string:event_id>/acknowledge", methods=["PATCH"])
    def acknowledge_fall_event(event_id: str):
        """Transitions event from CONFIRMED -> ACKNOWLEDGED."""
        data = request.get_json(silent=True) or {}
        acknowledged_by = data.get("acknowledged_by", "operator")

        try:
            event = event_service.acknowledge_event(
                event_id=event_id,
                acknowledged_by=acknowledged_by
            )
            return jsonify({
                "status": "success",
                "event": event.to_dict()
            }), 200
        except EventNotFoundError:
            return jsonify({"status": "error", "message": f"Event '{event_id}' not found."}), 404
        except InvalidStateTransitionError as e:
            return jsonify({"status": "error", "message": str(e)}), 409

    @app.route("/api/fall-events/<string:event_id>/cancel", methods=["PATCH"])
    def cancel_fall_event(event_id: str):
        """Transitions event from CONFIRMED -> CANCELLED."""
        data = request.get_json(silent=True) or {}
        cancelled_by = data.get("cancelled_by", "operator")
        reason = data.get("reason", "Cancelled by user/operator")

        try:
            event = event_service.cancel_event(
                event_id=event_id,
                cancelled_by=cancelled_by,
                reason=reason
            )
            return jsonify({
                "status": "success",
                "event": event.to_dict()
            }), 200
        except EventNotFoundError:
            return jsonify({"status": "error", "message": f"Event '{event_id}' not found."}), 404
        except InvalidStateTransitionError as e:
            return jsonify({"status": "error", "message": str(e)}), 409

    @app.route("/api/fall-events/<string:event_id>/resolve", methods=["PATCH"])
    def resolve_fall_event(event_id: str):
        """Transitions event from ACKNOWLEDGED -> RESOLVED."""
        data = request.get_json(silent=True) or {}
        resolved_by = data.get("resolved_by", "operator")

        try:
            event = event_service.resolve_event(
                event_id=event_id,
                resolved_by=resolved_by
            )
            return jsonify({
                "status": "success",
                "event": event.to_dict()
            }), 200
        except EventNotFoundError:
            return jsonify({"status": "error", "message": f"Event '{event_id}' not found."}), 404
        except InvalidStateTransitionError as e:
            return jsonify({"status": "error", "message": str(e)}), 409

    return app


if __name__ == "__main__":
    app = create_app()
    print("Starting Standalone Fall Event REST Backend on port 5000...")
    app.run(host="127.0.0.1", port=5000, debug=False)
