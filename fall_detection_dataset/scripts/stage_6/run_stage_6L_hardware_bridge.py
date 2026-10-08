"""
Stage 6L Hardware Bridge Wrapper.
Invokes the root scripts/stage_6/run_stage_6L_hardware_bridge.py.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.stage_6.run_stage_6L_hardware_bridge import (
    parse_serial_stream_line,
    MockSensorStream,
    run_stage_6L_hardware_bridge,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 6L Integrated Hardware Inference Bridge")
    parser.add_argument("--port", type=str, default=os.environ.get("SERIAL_PORT", "COM22"), help="Serial port")
    parser.add_argument("--baud", type=int, default=int(os.environ.get("SERIAL_BAUD", 460800)), help="Baud rate (default: 460800 or SERIAL_BAUD)")
    parser.add_argument("--broker", type=str, default=os.environ.get("MQTT_BROKER_HOST", "127.0.0.1"), help="MQTT broker host")
    parser.add_argument("--broker-port", type=int, default=int(os.environ.get("MQTT_BROKER_PORT", 1883)), help="MQTT broker port")
    parser.add_argument("--session-id", type=str, default=None, help="Session ID")
    parser.add_argument("--max-windows", type=int, default=None, help="Max windows to process")
    parser.add_argument("--dry-run", action="store_true", help="Skip MQTT publishing")
    parser.add_argument("--mock", action="store_true", help="Run with mock sensor stream")
    parser.add_argument("--mock-scenario", type=str, default="stationary", choices=["stationary", "fall"], help="Mock data scenario")

    args = parser.parse_args()
    run_stage_6L_hardware_bridge(
        port=args.port,
        baudrate=args.baud,
        broker_host=args.broker,
        broker_port=args.broker_port,
        session_id=args.session_id,
        max_windows=args.max_windows,
        dry_run=args.dry_run,
        mock_mode=args.mock,
        mock_scenario=args.mock_scenario,
    )
