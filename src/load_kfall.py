"""
KFall Data Loading and Quality Audit Module.

Loads 100 Hz motion recordings, performs strict quality verification,
removes invalid data, and computes derived signals (AccMag, GyrMag).
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# 8 Signals for Feature Extraction
RAW_SIGNALS = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
DERIVED_SIGNALS = ["AccMag", "GyrMag"]
ALL_SIGNALS = RAW_SIGNALS + DERIVED_SIGNALS


def parse_motion_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parses KFall motion filename into components.
    Example: 'SA06T01R01.csv' -> {'subject_id': 'SA06', 'task_id': 'T01', 'trial_id': 'R01'}
    """
    basename = os.path.basename(filename)
    match = re.match(r"(SA\d+)(T\d+)(R\d+)", basename, re.IGNORECASE)
    if not match:
        return None
    sub, task, trial = match.groups()
    return {
        "subject_id": sub.upper(),
        "task_id": task.upper(),
        "trial_id": trial.upper(),
        "filename": basename,
    }


def is_fall_task(task_id: str) -> bool:
    """
    Determines if task is a Fall (T22-T36) or ADL (T01-T21).
    """
    m = re.search(r"(\d+)", task_id)
    if m:
        task_num = int(m.group(1))
        return task_num >= 22
    return False


def load_and_audit_motion_file(
    file_path: str,
    nominal_dt: float = 0.01,
    gap_threshold: float = 0.05,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Loads a motion CSV, audits quality, removes invalid rows,
    and calculates derived signals AccMag and GyrMag.

    Parameters
    ----------
    file_path : str
        Path to the KFall CSV file.
    nominal_dt : float
        Nominal sample period in seconds (0.01s = 100 Hz).
    gap_threshold : float
        Threshold in seconds above which a time difference is flagged as an abnormal gap.

    Returns
    -------
    tuple of (DataFrame or None, audit_dict)
    """
    audit = {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "initial_rows": 0,
        "clean_rows": 0,
        "dropped_rows": 0,
        "duplicate_frames": 0,
        "out_of_order_frames": 0,
        "sampling_gaps_count": 0,
        "max_gap_sec": 0.0,
        "mean_gap_sec": 0.0,
        "is_valid": False,
        "rejection_reason": None,
    }

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        audit["rejection_reason"] = "File missing or empty"
        return None, audit

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        audit["rejection_reason"] = f"CSV read error: {str(e)}"
        return None, audit

    audit["initial_rows"] = len(df)
    if len(df) < 50:
        audit["rejection_reason"] = f"Insufficient rows ({len(df)} < 50 window size)"
        return None, audit

    # Column name normalization
    clean_cols = {}
    for col in df.columns:
        c_low = col.strip().lower().replace(" ", "").replace("_", "")
        if c_low in ["timestamp(s)", "timestamp", "time(s)", "time"]:
            clean_cols[col] = "TimeStamp(s)"
        elif c_low in ["framecounter", "frame", "framecount"]:
            clean_cols[col] = "FrameCounter"
        elif c_low in ["accx", "accelerationx", "ax"]:
            clean_cols[col] = "AccX"
        elif c_low in ["accy", "accelerationy", "ay"]:
            clean_cols[col] = "AccY"
        elif c_low in ["accz", "accelerationz", "az"]:
            clean_cols[col] = "AccZ"
        elif c_low in ["gyrx", "gyrox", "gyroscopex", "gx"]:
            clean_cols[col] = "GyrX"
        elif c_low in ["gyry", "gyroy", "gyroscopey", "gy"]:
            clean_cols[col] = "GyrY"
        elif c_low in ["gyrz", "gyroz", "gyroscopez", "gz"]:
            clean_cols[col] = "GyrZ"

    df = df.rename(columns=clean_cols)
    required_cols = ["TimeStamp(s)", "FrameCounter"] + RAW_SIGNALS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        audit["rejection_reason"] = f"Missing required columns: {missing_cols}"
        return None, audit

    # Keep only required columns
    df = df[required_cols].copy()

    # Convert to numeric, dropping non-numeric values
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    nan_mask = df.isna().any(axis=1)
    n_nans = int(nan_mask.sum())
    if n_nans > 0:
        df = df[~nan_mask].copy()

    if len(df) < 50:
        audit["rejection_reason"] = f"Too many NaNs: rows after cleaning ({len(df)} < 50)"
        return None, audit

    # Check FrameCounter duplicate and ordering
    frames = df["FrameCounter"].values
    diffs = np.diff(frames)

    dup_count = int(np.sum(diffs == 0))
    neg_count = int(np.sum(diffs < 0))
    audit["duplicate_frames"] = dup_count
    audit["out_of_order_frames"] = neg_count

    if dup_count > 0:
        # Drop duplicate frame rows keeping first
        df = df.drop_duplicates(subset=["FrameCounter"], keep="first").copy()

    if neg_count > 0:
        # Sort by FrameCounter
        df = df.sort_values(by="FrameCounter").reset_index(drop=True)

    # Check sampling gaps
    timestamps = df["TimeStamp(s)"].values
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        # Handle zero or negative time jumps if any
        positive_diffs = time_diffs[time_diffs > 0]
        if len(positive_diffs) > 0:
            audit["mean_gap_sec"] = float(np.mean(positive_diffs))
            audit["max_gap_sec"] = float(np.max(positive_diffs))
            audit["sampling_gaps_count"] = int(np.sum(positive_diffs > gap_threshold))

    audit["clean_rows"] = len(df)
    audit["dropped_rows"] = audit["initial_rows"] - audit["clean_rows"]

    # Calculate derived magnitude signals
    df["AccMag"] = np.sqrt(df["AccX"] ** 2 + df["AccY"] ** 2 + df["AccZ"] ** 2)
    df["GyrMag"] = np.sqrt(df["GyrX"] ** 2 + df["GyrY"] ** 2 + df["GyrZ"] ** 2)

    audit["is_valid"] = True
    return df, audit


def find_all_motion_files(sensor_dir: str = "data/kfall/sensor_data") -> List[str]:
    """
    Recursively scans sensor_dir for all SAxxTyyRzz.csv files.
    """
    found_files = []
    if not os.path.exists(sensor_dir):
        return found_files

    for root, _, files in os.walk(sensor_dir):
        for f in files:
            if f.lower().endswith(".csv") and parse_motion_filename(f):
                found_files.append(os.path.join(root, f))

    return sorted(found_files)
