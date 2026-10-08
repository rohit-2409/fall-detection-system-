"""
SisFall Windowing and Label Assignment Module (Stage B2).

Generates fixed-size sliding temporal windows from 100 Hz resampled SisFall recordings:
- Sampling rate: 100 Hz (dt = 0.01s)
- Window size: 50 samples (0.50 seconds)
- Step size: 25 samples (50% overlap, 0.25 seconds stride)
- Channels: 6 baseline channels (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
- Labels: Inherited from authoritative trial-level identity:
    Dxx -> ADL (label=0, activity_type="ADL")
    Fxx -> FALL (label=1, activity_type="FALL")
- Streaming batch Parquet storage to ensure memory safety across all 4,505 trials.

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

import math
import os
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.sisfall.sisfall_loader import (
    SisFallMetadata,
    discover_sisfall_files,
    load_sisfall_file,
)
from src.sisfall.sisfall_parser import BASELINE_COLUMNS

# Window configuration constants
SAMPLING_RATE_HZ: float = 100.0
WINDOW_SIZE_SAMPLES: int = 50       # 50 samples = 0.50s at 100 Hz
STEP_SIZE_SAMPLES: int = 25         # 25 samples = 50% overlap (0.25s stride)
WINDOW_DURATION_SEC: float = WINDOW_SIZE_SAMPLES / SAMPLING_RATE_HZ  # 0.50s
STEP_DURATION_SEC: float = STEP_SIZE_SAMPLES / SAMPLING_RATE_HZ      # 0.25s

# 6 Baseline channel names stored inside each window
WINDOW_CHANNELS: List[str] = [
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
]

# Source DataFrame column mapping to clean window channel names
CHANNEL_SOURCE_MAP: Dict[str, str] = {
    "AccX": "AccX_g",
    "AccY": "AccY_g",
    "AccZ": "AccZ_g",
    "GyrX": "GyrX_dps",
    "GyrY": "GyrY_dps",
    "GyrZ": "GyrZ_dps",
}

# Explicit PyArrow Schema for SisFall Window Storage
SISFALL_WINDOW_ARROW_SCHEMA = pa.schema([
    ("window_id", pa.string()),
    ("subject_id", pa.string()),
    ("subject_group", pa.string()),
    ("activity_id", pa.string()),
    ("activity_type", pa.string()),
    ("trial_id", pa.string()),
    ("source_filename", pa.string()),
    ("label", pa.int32()),
    ("window_start_sample", pa.int64()),
    ("window_end_sample", pa.int64()),
    ("window_start_seconds", pa.float64()),
    ("window_end_seconds", pa.float64()),
    ("sampling_rate_hz", pa.float64()),
    ("window_size_samples", pa.int32()),
    ("step_size_samples", pa.int32()),
    ("AccX", pa.list_(pa.float64())),
    ("AccY", pa.list_(pa.float64())),
    ("AccZ", pa.list_(pa.float64())),
    ("GyrX", pa.list_(pa.float64())),
    ("GyrY", pa.list_(pa.float64())),
    ("GyrZ", pa.list_(pa.float64())),
])


def compute_expected_window_count(
    signal_length: int,
    window_size: int = WINDOW_SIZE_SAMPLES,
    step_size: int = STEP_SIZE_SAMPLES,
) -> int:
    """
    Computes the exact number of complete sliding windows generated from a signal.

    Formula:
        number_of_windows = floor((N - window_size) / step_size) + 1  for N >= window_size
        number_of_windows = 0                                          for N < window_size

    Parameters
    ----------
    signal_length : int
        Length of the discrete signal in samples.
    window_size : int, default=50
        Window size in samples.
    step_size : int, default=25
        Step / stride size in samples.

    Returns
    -------
    int
        Expected number of complete non-padded windows.
    """
    if signal_length < window_size:
        return 0
    return math.floor((signal_length - window_size) / step_size) + 1


def generate_windows_from_trial(
    df: pd.DataFrame,
    meta: SisFallMetadata,
    window_size: int = WINDOW_SIZE_SAMPLES,
    step_size: int = STEP_SIZE_SAMPLES,
) -> List[Dict[str, Any]]:
    """
    Slices a continuous 100 Hz trial DataFrame into 50-sample sliding windows (50% overlap).

    Labels are assigned directly from authoritative trial identity:
    - ADL (Dxx) -> label = 0, activity_type = 'ADL'
    - FALL (Fxx) -> label = 1, activity_type = 'FALL'

    Parameters
    ----------
    df : pd.DataFrame
        Continuous trial DataFrame at 100 Hz containing columns:
        ['AccX_g', 'AccY_g', 'AccZ_g', 'GyrX_dps', 'GyrY_dps', 'GyrZ_dps'].
    meta : SisFallMetadata
        Trial metadata object.
    window_size : int, default=50
        Window length in samples.
    step_size : int, default=25
        Step size in samples.

    Returns
    -------
    List[Dict[str, Any]]
        List of window dictionaries matching SISFALL_WINDOW_ARROW_SCHEMA.
    """
    num_samples = len(df)
    expected_count = compute_expected_window_count(num_samples, window_size, step_size)

    if expected_count == 0:
        return []

    # Extract numpy float64 arrays for maximum performance and precision
    acc_x = df[CHANNEL_SOURCE_MAP["AccX"]].to_numpy(dtype=np.float64)
    acc_y = df[CHANNEL_SOURCE_MAP["AccY"]].to_numpy(dtype=np.float64)
    acc_z = df[CHANNEL_SOURCE_MAP["AccZ"]].to_numpy(dtype=np.float64)
    gyr_x = df[CHANNEL_SOURCE_MAP["GyrX"]].to_numpy(dtype=np.float64)
    gyr_y = df[CHANNEL_SOURCE_MAP["GyrY"]].to_numpy(dtype=np.float64)
    gyr_z = df[CHANNEL_SOURCE_MAP["GyrZ"]].to_numpy(dtype=np.float64)

    # Subject group: young adult (SA) vs elderly (SE)
    subject_group = "SA" if meta.subject_id.startswith("SA") else "SE"

    # Trial-level ground truth label
    if meta.activity_type == "ADL":
        label = 0
    elif meta.activity_type == "FALL":
        label = 1
    else:
        raise ValueError(f"Invalid activity_type '{meta.activity_type}' in {meta.source_file}")

    windows: List[Dict[str, Any]] = []
    window_idx = 0
    start = 0

    while start + window_size <= num_samples:
        end = start + window_size

        # Format standardized, globally unique window identifier
        win_id = f"{meta.subject_id}_{meta.activity_id}_{meta.trial_id}_W{window_idx:04d}"

        start_sec = start / SAMPLING_RATE_HZ
        end_sec = end / SAMPLING_RATE_HZ

        # Extract 50 samples for each of the 6 channels
        w_dict: Dict[str, Any] = {
            "window_id": win_id,
            "subject_id": meta.subject_id,
            "subject_group": subject_group,
            "activity_id": meta.activity_id,
            "activity_type": meta.activity_type,
            "trial_id": meta.trial_id,
            "source_filename": meta.source_file,
            "label": label,
            "window_start_sample": start,
            "window_end_sample": end,
            "window_start_seconds": start_sec,
            "window_end_seconds": end_sec,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "window_size_samples": window_size,
            "step_size_samples": step_size,
            "AccX": acc_x[start:end].tolist(),
            "AccY": acc_y[start:end].tolist(),
            "AccZ": acc_z[start:end].tolist(),
            "GyrX": gyr_x[start:end].tolist(),
            "GyrY": gyr_y[start:end].tolist(),
            "GyrZ": gyr_z[start:end].tolist(),
        }
        windows.append(w_dict)
        window_idx += 1
        start += step_size

    if len(windows) != expected_count:
        raise RuntimeError(
            f"Window count anomaly in {meta.source_file}: generated {len(windows)}, expected {expected_count}"
        )

    return windows


def build_sisfall_window_dataset(
    dataset_root: str = r"data/SisFall_dataset/SisFall_dataset",
    output_parquet_path: str = r"data/processed/sisfall/sisfall_windows.parquet",
    debug_sample_csv_path: Optional[str] = r"data/processed/sisfall/sisfall_debug_sample_windows.csv",
    batch_flush_trials: int = 50,
    debug_sample_size: int = 100,
    verbose_every: int = 500,
) -> Dict[str, Any]:
    """
    Builds the complete SisFall 100 Hz window dataset using a memory-conscious streaming pipeline.

    Processes trials one at a time, batches windows into Arrow RecordBatches, and writes
    directly to Parquet, avoiding storing the entire dataset in RAM.

    Parameters
    ----------
    dataset_root : str
        Path to SisFall dataset root.
    output_parquet_path : str
        Target destination for output Parquet dataset.
    debug_sample_csv_path : Optional[str]
        Optional path for a small human-readable debug CSV sample.
    batch_flush_trials : int, default=50
        Number of trials to accumulate before flushing batch to disk.
    debug_sample_size : int, default=100
        Number of representative windows to save to debug CSV.
    verbose_every : int, default=500
        Logging interval.

    Returns
    -------
    Dict[str, Any]
        Complete audit and accounting statistics dictionary.
    """
    all_files = discover_sisfall_files(dataset_root)
    total_files = len(all_files)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_parquet_path))
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"BUILDING SISFALL WINDOW DATASET ({total_files} TRIALS)")
    print(f"Output Parquet: {output_parquet_path}")
    print("=" * 80)

    # Accounting counters
    files_discovered = total_files
    files_processed = 0
    files_skipped = 0
    skip_reasons: Dict[str, str] = {}

    adl_trials_processed = 0
    fall_trials_processed = 0
    adl_windows_generated = 0
    fall_windows_generated = 0
    total_windows_generated = 0

    subjects_seen = set()
    activities_seen = set()

    # Debug sample collector
    debug_samples: List[Dict[str, Any]] = []

    # Initialize ParquetWriter
    writer = pq.ParquetWriter(
        output_parquet_path,
        schema=SISFALL_WINDOW_ARROW_SCHEMA,
        compression="snappy",
    )

    window_batch: List[Dict[str, Any]] = []

    try:
        for idx, fpath in enumerate(all_files, start=1):
            try:
                df, meta = load_sisfall_file(fpath, resample_to_100hz=True)
                windows = generate_windows_from_trial(df, meta)

                if len(windows) == 0:
                    files_skipped += 1
                    skip_reasons[fpath] = f"Signal length ({len(df)}) < window size ({WINDOW_SIZE_SAMPLES})"
                    continue

                files_processed += 1
                subjects_seen.add(meta.subject_id)
                activities_seen.add(meta.activity_id)

                num_w = len(windows)
                total_windows_generated += num_w

                if meta.activity_type == "ADL":
                    adl_trials_processed += 1
                    adl_windows_generated += num_w
                else:
                    fall_trials_processed += 1
                    fall_windows_generated += num_w

                window_batch.extend(windows)

                # Collect a small subset for debug CSV
                if len(debug_samples) < debug_sample_size:
                    # Collect first 2 windows of early trials
                    for w in windows[:2]:
                        if len(debug_samples) < debug_sample_size:
                            # Flatten channel lists into first/mean representation for compact debug CSV
                            debug_entry = {
                                "window_id": w["window_id"],
                                "subject_id": w["subject_id"],
                                "subject_group": w["subject_group"],
                                "activity_id": w["activity_id"],
                                "activity_type": w["activity_type"],
                                "trial_id": w["trial_id"],
                                "source_filename": w["source_filename"],
                                "label": w["label"],
                                "window_start_sample": w["window_start_sample"],
                                "window_end_sample": w["window_end_sample"],
                                "window_start_seconds": w["window_start_seconds"],
                                "window_end_seconds": w["window_end_seconds"],
                                "sampling_rate_hz": w["sampling_rate_hz"],
                                "window_size_samples": w["window_size_samples"],
                                "step_size_samples": w["step_size_samples"],
                                "AccX_first": w["AccX"][0],
                                "AccX_mean": float(np.mean(w["AccX"])),
                                "AccY_first": w["AccY"][0],
                                "AccY_mean": float(np.mean(w["AccY"])),
                                "AccZ_first": w["AccZ"][0],
                                "AccZ_mean": float(np.mean(w["AccZ"])),
                                "GyrX_first": w["GyrX"][0],
                                "GyrX_mean": float(np.mean(w["GyrX"])),
                                "GyrY_first": w["GyrY"][0],
                                "GyrY_mean": float(np.mean(w["GyrY"])),
                                "GyrZ_first": w["GyrZ"][0],
                                "GyrZ_mean": float(np.mean(w["GyrZ"])),
                            }
                            debug_samples.append(debug_entry)

                # Flush batch to Parquet
                if idx % batch_flush_trials == 0:
                    if window_batch:
                        table = pa.Table.from_pylist(window_batch, schema=SISFALL_WINDOW_ARROW_SCHEMA)
                        writer.write_table(table)
                        window_batch.clear()

                if idx % verbose_every == 0 or idx == total_files:
                    print(
                        f"  Processed {idx}/{total_files} trials | "
                        f"Windows: {total_windows_generated:,} (ADL: {adl_windows_generated:,}, Fall: {fall_windows_generated:,})"
                    )

            except Exception as err:
                files_skipped += 1
                skip_reasons[fpath] = str(err)
                print(f"ERROR on trial {fpath}: {err}")

        # Flush remaining windows
        if window_batch:
            table = pa.Table.from_pylist(window_batch, schema=SISFALL_WINDOW_ARROW_SCHEMA)
            writer.write_table(table)
            window_batch.clear()

    finally:
        writer.close()

    # Write debug sample CSV if requested
    if debug_sample_csv_path and debug_samples:
        debug_csv_dir = os.path.dirname(os.path.abspath(debug_sample_csv_path))
        os.makedirs(debug_csv_dir, exist_ok=True)
        debug_df = pd.DataFrame(debug_samples)
        debug_df.to_csv(debug_sample_csv_path, index=False)
        print(f"Debug sample CSV saved ({len(debug_samples)} rows): {debug_sample_csv_path}")

    file_size_bytes = os.path.getsize(output_parquet_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    summary: Dict[str, Any] = {
        "dataset_root": os.path.normpath(dataset_root),
        "output_parquet_path": os.path.normpath(output_parquet_path),
        "parquet_size_mb": round(file_size_mb, 2),
        "files_discovered": files_discovered,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "skip_reasons": skip_reasons,
        "adl_trials_processed": adl_trials_processed,
        "fall_trials_processed": fall_trials_processed,
        "subjects_count": len(subjects_seen),
        "subjects": sorted(list(subjects_seen)),
        "activities_count": len(activities_seen),
        "total_windows_generated": total_windows_generated,
        "adl_windows_generated": adl_windows_generated,
        "fall_windows_generated": fall_windows_generated,
        "adl_window_ratio": round(adl_windows_generated / total_windows_generated, 4),
        "fall_window_ratio": round(fall_windows_generated / total_windows_generated, 4),
    }

    print("\n" + "=" * 80)
    print("SISFALL WINDOW DATASET GENERATION COMPLETE")
    print(f"  Total Windows Generated: {total_windows_generated:,}")
    print(f"  ADL Windows (label=0): {adl_windows_generated:,} ({summary['adl_window_ratio']*100:.1f}%)")
    print(f"  FALL Windows (label=1): {fall_windows_generated:,} ({summary['fall_window_ratio']*100:.1f}%)")
    print(f"  Parquet File Size: {file_size_mb:.2f} MB")
    print("=" * 80)

    return summary
