"""
SisFall Feature Extraction Module (Stage B3).

Extracts deterministic time-domain statistical features from 50-sample sliding windows:
- 8 Signals:
    6 Raw channels: AccX, AccY, AccZ, GyrX, GyrY, GyrZ
    2 Derived magnitude channels:
        AccMag = sqrt(AccX^2 + AccY^2 + AccZ^2)
        GyrMag = sqrt(GyrX^2 + GyrY^2 + GyrZ^2)
- 6 Statistics per signal:
    mean:       mean(x)
    std:        std(x, ddof=0)  (population standard deviation)
    min:        min(x)
    max:        max(x)
    ptp:        max(x) - min(x)
    rms:        sqrt(mean(x^2))
- Total Features: 8 signals * 6 statistics = 48 numerical features.
- Preserves all 15 metadata fields to support subject-wise splitting in Stage B4.
- High-performance, memory-safe streaming Parquet processing.

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The 8 signals used for feature extraction
BASE_SIGNALS: List[str] = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
DERIVED_SIGNALS: List[str] = ["AccMag", "GyrMag"]
ALL_SIGNALS: List[str] = BASE_SIGNALS + DERIVED_SIGNALS

# The 6 summary statistics per signal
STAT_NAMES: List[str] = ["mean", "std", "min", "max", "ptp", "rms"]

# Ordered list of all 48 feature names
FEATURE_NAMES: List[str] = [
    f"{sig}_{stat}" for sig in ALL_SIGNALS for stat in STAT_NAMES
]

# The 15 metadata columns preserved from Stage B2
METADATA_COLUMNS: List[str] = [
    "window_id",
    "subject_id",
    "subject_group",
    "activity_id",
    "activity_type",
    "trial_id",
    "source_filename",
    "label",
    "window_start_sample",
    "window_end_sample",
    "window_start_seconds",
    "window_end_seconds",
    "sampling_rate_hz",
    "window_size_samples",
    "step_size_samples",
]

# Full 63-column Arrow schema for sisfall_features.parquet
SISFALL_FEATURE_ARROW_SCHEMA = pa.schema([
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
] + [(feat, pa.float64()) for feat in FEATURE_NAMES])


def compute_signal_statistics_vectorized(
    signals_2d: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Computes all 48 statistical features across a batch of windows using NumPy vectorization.

    Parameters
    ----------
    signals_2d : Dict[str, np.ndarray]
        Dictionary mapping each of the 8 signal names to a 2D array of shape (B, 50).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each of the 48 feature names to a 1D array of length B.
    """
    features: Dict[str, np.ndarray] = {}

    for sig_name in ALL_SIGNALS:
        sig_arr = signals_2d[sig_name]  # Shape: (B, 50)

        # 1. Mean: mean(x)
        features[f"{sig_name}_mean"] = np.mean(sig_arr, axis=1)

        # 2. Standard deviation: std(x, ddof=0)
        features[f"{sig_name}_std"] = np.std(sig_arr, axis=1, ddof=0)

        # 3. Minimum: min(x)
        features[f"{sig_name}_min"] = np.min(sig_arr, axis=1)

        # 4. Maximum: max(x)
        features[f"{sig_name}_max"] = np.max(sig_arr, axis=1)

        # 5. Peak-to-peak: max(x) - min(x)
        features[f"{sig_name}_ptp"] = np.ptp(sig_arr, axis=1)

        # 6. RMS: sqrt(mean(x^2))
        features[f"{sig_name}_rms"] = np.sqrt(np.mean(sig_arr**2, axis=1))

    return features


def extract_features_from_batch(
    df_batch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extracts features for a batch DataFrame loaded from sisfall_windows.parquet.

    Parameters
    ----------
    df_batch : pd.DataFrame
        DataFrame chunk containing metadata columns and channel list columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 15 metadata columns and 48 feature columns (63 columns total).
    """
    # 1. Stack channel list values into 2D arrays: shape (B, 50)
    ax = np.vstack(df_batch["AccX"].values).astype(np.float64)
    ay = np.vstack(df_batch["AccY"].values).astype(np.float64)
    az = np.vstack(df_batch["AccZ"].values).astype(np.float64)
    gx = np.vstack(df_batch["GyrX"].values).astype(np.float64)
    gy = np.vstack(df_batch["GyrY"].values).astype(np.float64)
    gz = np.vstack(df_batch["GyrZ"].values).astype(np.float64)

    # 2. Compute magnitude signals: shape (B, 50)
    amag = np.sqrt(ax**2 + ay**2 + az**2)
    gmag = np.sqrt(gx**2 + gy**2 + gz**2)

    signals_dict = {
        "AccX": ax,
        "AccY": ay,
        "AccZ": az,
        "GyrX": gx,
        "GyrY": gy,
        "GyrZ": gz,
        "AccMag": amag,
        "GyrMag": gmag,
    }

    # 3. Compute 48 features
    feat_dict = compute_signal_statistics_vectorized(signals_dict)

    # 4. Construct output DataFrame preserving metadata
    meta_df = df_batch[METADATA_COLUMNS].copy().reset_index(drop=True)
    feat_df = pd.DataFrame(feat_dict)
    combined_df = pd.concat([meta_df, feat_df], axis=1)

    return combined_df


def build_sisfall_feature_dataset(
    input_parquet_path: str = r"data/processed/sisfall/sisfall_windows.parquet",
    output_parquet_path: str = r"data/processed/sisfall/sisfall_features.parquet",
    debug_sample_csv_path: Optional[str] = r"data/processed/sisfall/sisfall_debug_sample_features.csv",
    batch_size: int = 10000,
    debug_sample_size: int = 100,
    verbose_every: int = 50000,
) -> Dict[str, Any]:
    """
    Streams through the SisFall window dataset, extracts all 48 statistical features,
    and writes the master feature dataset to Parquet.

    Parameters
    ----------
    input_parquet_path : str
        Path to sisfall_windows.parquet.
    output_parquet_path : str
        Path for sisfall_features.parquet.
    debug_sample_csv_path : Optional[str]
        Path for human-readable debug CSV sample.
    batch_size : int, default=10000
        Number of rows to process per chunk.
    debug_sample_size : int, default=100
        Number of rows to store in debug CSV.
    verbose_every : int, default=50000
        Logging interval.

    Returns
    -------
    Dict[str, Any]
        Audit summary statistics.
    """
    if not os.path.isfile(input_parquet_path):
        raise FileNotFoundError(f"Input Parquet file not found: {input_parquet_path}")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_parquet_path))
    os.makedirs(out_dir, exist_ok=True)

    parquet_file = pq.ParquetFile(input_parquet_path)
    total_input_windows = parquet_file.metadata.num_rows

    print("\n" + "=" * 80)
    print(f"STARTING SISFALL FEATURE EXTRACTION ({total_input_windows:,} WINDOWS)")
    print(f"Input:  {input_parquet_path}")
    print(f"Output: {output_parquet_path}")
    print("=" * 80)

    writer = pq.ParquetWriter(
        output_parquet_path,
        schema=SISFALL_FEATURE_ARROW_SCHEMA,
        compression="snappy",
    )

    total_rows_processed = 0
    debug_sample_rows: List[pd.DataFrame] = []

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df_chunk = batch.to_pandas()
            features_chunk = extract_features_from_batch(df_chunk)

            # Collect initial rows for debug CSV
            if len(debug_sample_rows) < debug_sample_size:
                needed = debug_sample_size - sum(len(d) for d in debug_sample_rows)
                debug_sample_rows.append(features_chunk.iloc[:needed])

            # Write chunk to Parquet
            table_chunk = pa.Table.from_pandas(
                features_chunk,
                schema=SISFALL_FEATURE_ARROW_SCHEMA,
                preserve_index=False,
            )
            writer.write_table(table_chunk)

            total_rows_processed += len(features_chunk)

            if total_rows_processed % verbose_every == 0 or total_rows_processed == total_input_windows:
                print(f"  Extracted features for {total_rows_processed:,} / {total_input_windows:,} windows...")

    finally:
        writer.close()

    # Save debug CSV sample
    if debug_sample_csv_path and debug_sample_rows:
        debug_csv_df = pd.concat(debug_sample_rows, axis=0).reset_index(drop=True)
        debug_csv_df.to_csv(debug_sample_csv_path, index=False)
        print(f"Debug sample CSV saved ({len(debug_csv_df)} rows): {debug_sample_csv_path}")

    file_size_bytes = os.path.getsize(output_parquet_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    summary: Dict[str, Any] = {
        "input_parquet_path": os.path.normpath(input_parquet_path),
        "output_parquet_path": os.path.normpath(output_parquet_path),
        "parquet_size_mb": round(file_size_mb, 2),
        "total_input_windows": total_input_windows,
        "total_feature_rows": total_rows_processed,
        "metadata_column_count": len(METADATA_COLUMNS),
        "feature_column_count": len(FEATURE_NAMES),
        "total_column_count": len(SISFALL_FEATURE_ARROW_SCHEMA.names),
        "one_to_one_match": total_input_windows == total_rows_processed,
    }

    print("\n" + "=" * 80)
    print("SISFALL FEATURE EXTRACTION COMPLETE")
    print(f"  Total Windows Processed: {total_rows_processed:,}")
    print(f"  Feature Columns: {len(FEATURE_NAMES)} (8 signals * 6 stats)")
    print(f"  Parquet File Size: {file_size_mb:.2f} MB")
    print("=" * 80)

    return summary
