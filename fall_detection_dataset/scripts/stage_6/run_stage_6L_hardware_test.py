"""
Stage 6L Hardware Orchestrator Wrapper.
Invokes the root scripts/stage_6/run_stage_6L_hardware_test.py.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_test import (
    check_tcp_port,
    check_backend_health,
    check_serial_port_available,
    run_orchestration_preflight,
    run_staged_hardware_test,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 6L Hardware-in-the-Loop Orchestrator")
    parser.add_argument("--broker", type=str, default=os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"), help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--backend", type=str, default=os.environ.get("BACKEND_URL", "http://127.0.0.1:5000"), help="Backend URL")
    parser.add_argument("--port", type=str, default=os.environ.get("SERIAL_PORT", "COM22"), help="Serial port")
    parser.add_argument("--mock", action="store_true", help="Run with mock sensor stream")
    parser.add_argument("--mock-scenario", type=str, default="stationary", choices=["stationary", "fall"], help="Mock scenario")
    parser.add_argument("--windows", type=int, default=4, help="Max windows to process")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without publishing")

    args = parser.parse_args()
    run_staged_hardware_test(
        broker_host=args.broker,
        broker_port=args.broker_port,
        backend_url=args.backend,
        serial_port=args.port,
        mock_mode=args.mock,
        mock_scenario=args.mock_scenario,
        max_windows=args.windows,
        dry_run=args.dry_run,
    )
