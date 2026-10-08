"""
Configuration Loader for Fall Confirmation Engine (Stage 4F).

Loads validated configuration from confirmation_engine_config.json while
providing immutable fallback defaults matching the Stage 4E baseline:
  Threshold: 0.60
  Rule: N3 (3 consecutive positive windows)
  Event Gating: False
"""

import json
import os
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "stage": "4E_baseline",
    "model_threshold": 0.60,
    "temporal_rule": "N3",
    "require_event_confirmation": False,
    "min_dyn_acc": 1.80,
    "min_gyr_mag": 100.0,
    "cooldown_windows": 8,            # 8 * 0.25s = 2.0s technical post-fall debounce
    "stream_gap_threshold_s": 0.50,   # Reset candidate buffer if stream gap > 500 ms
    "lock_duration_s": 60.0,          # 60s post-fall locked alarm period
    "min_stable_duration_s": 5.0,     # >= 5.0s continuous upright + stable duration to re-arm
    "min_upright_acc_g": 0.65,        # |a_z_mean| >= 0.65g indicates upright posture
    "max_stable_dyn_acc_g": 0.35,     # dyn_acc <= 0.35g indicates low motion
    "max_stable_gyr_dps": 40.0,       # gyr_mag <= 40 dps indicates minimal rotational motion
    "max_stable_prob": 0.40,          # model prob < 0.40 indicates non-fall conditions
    "fast_path_enabled": False,       # Opt-in fast confirmation path (Candidate 3B)
    "fast_path_threshold": 0.85,      # Fast path probability threshold (P >= 0.85)
    "fast_path_accel_threshold_g": 2.50,   # Fast path dynamic acceleration threshold (>= 2.50 g)
    "fast_path_gyro_threshold_dps": 200.0, # Fast path gyroscope magnitude threshold (>= 200.0 deg/s)
}


def load_confirmation_config(custom_path: str = None) -> Dict[str, Any]:
    """
    Loads confirmation engine configuration.
    Prefers Stage 4E JSON artifact; falls back safely to verified baseline.
    """
    config = DEFAULT_CONFIG.copy()

    if custom_path and os.path.isfile(custom_path):
        target_path = custom_path
    else:
        # Standard relative location from workspace
        target_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "fall_detection_dataset",
                "models",
                "stage_4E",
                "confirmation_engine_config.json",
            )
        )

    if os.path.isfile(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                config.update(loaded)
        except Exception as err:
            print(f"[WARNING] Failed to load confirmation config from {target_path}: {err}. Using defaults.")

    return config
