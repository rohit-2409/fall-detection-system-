"""
Feature Extraction Module for KFall Fall Detection.

Computes 48 deterministic temporal and statistical features over 50-sample windows
(8 signals x 6 statistics: mean, std, min, max, ptp, rms).
"""

import json
import os
from typing import Dict, List, OrderedDict
import numpy as np
import pandas as pd

# The 8 signals
FEATURE_SIGNALS = [
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "AccMag",
    "GyrMag",
]

# The 6 statistics
STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "ptp",
    "rms",
]

# 48 Feature Names in deterministic order
FEATURE_COLUMNS = [f"{sig}_{stat}" for sig in FEATURE_SIGNALS for stat in STATISTICS]


def extract_features_from_window(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extracts 48 statistical features from a 50-sample window DataFrame.

    Parameters
    ----------
    window_df : pd.DataFrame
        DataFrame with at least 50 samples containing the 8 signals.

    Returns
    -------
    dict mapping feature_name -> float value
    """
    feats = {}
    for sig in FEATURE_SIGNALS:
        val = window_df[sig].values.astype(np.float64)
        mean_val = float(np.mean(val))
        std_val = float(np.std(val, ddof=1)) if len(val) > 1 else 0.0
        min_val = float(np.min(val))
        max_val = float(np.max(val))
        ptp_val = float(max_val - min_val)
        rms_val = float(np.sqrt(np.mean(val ** 2)))

        feats[f"{sig}_mean"] = mean_val
        feats[f"{sig}_std"] = std_val
        feats[f"{sig}_min"] = min_val
        feats[f"{sig}_max"] = max_val
        feats[f"{sig}_ptp"] = ptp_val
        feats[f"{sig}_rms"] = rms_val

    return feats


def export_feature_definition(output_path: str = "reports/feature_definition.json"):
    """
    Exports the formal feature catalog and ordering to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    catalog = {
        "total_features": len(FEATURE_COLUMNS),
        "signals": FEATURE_SIGNALS,
        "statistics": STATISTICS,
        "feature_order": FEATURE_COLUMNS,
        "descriptions": {
            "mean": "Arithmetic mean of the 50-sample temporal window",
            "std": "Sample standard deviation (ddof=1) of the window",
            "min": "Minimum value observed in the window",
            "max": "Maximum value observed in the window",
            "ptp": "Peak-to-Peak amplitude (max - min) in the window",
            "rms": "Root-Mean-Square: sqrt(mean(x^2)) measuring total signal energy",
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)


if __name__ == "__main__":
    export_feature_definition()
    print(f"Exported {len(FEATURE_COLUMNS)} feature definitions to reports/feature_definition.json")
