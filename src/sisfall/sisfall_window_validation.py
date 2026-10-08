"""
SisFall Window Validation Module (Stage B2).

Executes comprehensive validation of the generated SisFall window dataset:
1. Representative trial testing (ADL SA, ADL SE, Fall SA, Fall SE, SA15 anomaly, etc.).
2. Mathematical window-count formula validation.
3. Exhaustive structural integrity verification on the generated Parquet dataset:
   - Exact 50-sample length per channel
   - 6 baseline signal channels
   - Zero NaN / Inf / nulls
   - Strict label correspondence (0=ADL, 1=FALL)
   - Step size and index continuity (advance by 25)
   - Global window ID uniqueness
   - Strict preservation of parent-folder subject identity (SA15 anomaly check)
"""

import math
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.sisfall.sisfall_loader import load_sisfall_file
from src.sisfall.sisfall_windowing import (
    SAMPLING_RATE_HZ,
    STEP_SIZE_SAMPLES,
    WINDOW_CHANNELS,
    WINDOW_SIZE_SAMPLES,
    compute_expected_window_count,
    generate_windows_from_trial,
)


def run_representative_window_tests(
    dataset_root: str = r"data/SisFall_dataset/SisFall_dataset",
) -> List[Dict[str, Any]]:
    """
    Executes windowing validation on required representative trials:
    - ADL SA: SA01/D01_SA01_R01.txt
    - ADL SE: SE01/D01_SE01_R01.txt
    - ADL Car: SA02/D17_SA02_R01.txt
    - FALL SA: SA01/F01_SA01_R01.txt
    - FALL SE: SE06/F01_SE06_R01.txt (Judo expert)
    - FALL Sitting: SA01/F15_SA01_R01.txt
    - Anomaly: SA15/D17_SE15_R01.txt

    Returns
    -------
    List[Dict[str, Any]]
        List of validation records.
    """
    targets = [
        ("SA01", "D01_SA01_R01.txt", "ADL Young Adult Walking (D01)"),
        ("SE01", "D01_SE01_R01.txt", "ADL Elderly Adult Walking (D01)"),
        ("SA02", "D17_SA02_R01.txt", "ADL Car Activity (D17)"),
        ("SA01", "F01_SA01_R01.txt", "FALL Young Adult Slip (F01)"),
        ("SE06", "F01_SE06_R01.txt", "FALL Elderly Judo Expert (F01)"),
        ("SA01", "F15_SA01_R01.txt", "FALL Fainting while Sitting (F15)"),
        ("SA15", "D17_SE15_R01.txt", "CRITICAL ANOMALY: SA15/D17_SE15_R01.txt"),
    ]

    records: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("RUNNING REPRESENTATIVE WINDOWING TESTS")
    print("=" * 80)

    for subj_dir, filename, desc in targets:
        fpath = os.path.join(dataset_root, subj_dir, filename)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Representative trial not found: {fpath}")

        df, meta = load_sisfall_file(fpath, resample_to_100hz=True)
        N = len(df)
        expected_w = compute_expected_window_count(N, WINDOW_SIZE_SAMPLES, STEP_SIZE_SAMPLES)
        windows = generate_windows_from_trial(df, meta, WINDOW_SIZE_SAMPLES, STEP_SIZE_SAMPLES)

        # Basic assertions
        assert len(windows) == expected_w, f"Count mismatch: {len(windows)} vs {expected_w}"
        assert len(windows) > 0, f"Expected > 0 windows for {filename}"

        # Anomaly assertion
        if subj_dir == "SA15":
            assert meta.subject_id == "SA15", f"Expected SA15, got {meta.subject_id}"
            assert windows[0]["subject_id"] == "SA15"
            assert windows[0]["subject_group"] == "SA"

        first_w = windows[0]
        last_w = windows[-1]

        # Verify continuity across all windows
        for i in range(len(windows)):
            w = windows[i]
            assert w["window_end_sample"] - w["window_start_sample"] == WINDOW_SIZE_SAMPLES
            assert len(w["AccX"]) == WINDOW_SIZE_SAMPLES
            assert len(w["AccY"]) == WINDOW_SIZE_SAMPLES
            assert len(w["AccZ"]) == WINDOW_SIZE_SAMPLES
            assert len(w["GyrX"]) == WINDOW_SIZE_SAMPLES
            assert len(w["GyrY"]) == WINDOW_SIZE_SAMPLES
            assert len(w["GyrZ"]) == WINDOW_SIZE_SAMPLES
            if i > 0:
                prev_w = windows[i - 1]
                assert w["window_start_sample"] - prev_w["window_start_sample"] == STEP_SIZE_SAMPLES

        rec = {
            "description": desc,
            "subject_id": meta.subject_id,
            "subject_group": first_w["subject_group"],
            "trial_id": meta.trial_id,
            "activity_id": meta.activity_id,
            "activity_type": meta.activity_type,
            "source_filename": meta.source_file,
            "original_samples_200hz": meta.original_samples,
            "resampled_samples_100hz": N,
            "expected_window_count": expected_w,
            "actual_window_count": len(windows),
            "first_window_range": f"[{first_w['window_start_sample']}, {first_w['window_end_sample']}]",
            "last_window_range": f"[{last_w['window_start_sample']}, {last_w['window_end_sample']}]",
            "first_window_seconds": f"[{first_w['window_start_seconds']:.2f}s, {first_w['window_end_seconds']:.2f}s]",
            "last_window_seconds": f"[{last_w['window_start_seconds']:.2f}s, {last_w['window_end_seconds']:.2f}s]",
            "label": first_w["label"],
            "status": "PASS",
        }
        records.append(rec)

        print(f"\n[PASS] {desc}")
        print(f"  File: {meta.source_file} (Folder: {subj_dir})")
        print(f"  Resolved: Subject={rec['subject_id']} ({rec['subject_group']}), Label={rec['label']} ({rec['activity_type']})")
        print(f"  Resampled: {N} samples @ 100 Hz")
        print(f"  Windows: Expected={expected_w}, Generated={len(windows)}")
        print(f"  First Window: sample {rec['first_window_range']} ({rec['first_window_seconds']})")
        print(f"  Last Window:  sample {rec['last_window_range']} ({rec['last_window_seconds']})")

    return records


def validate_parquet_window_dataset(
    parquet_path: str = r"data/processed/sisfall/sisfall_windows.parquet",
    batch_size: int = 10000,
) -> Dict[str, Any]:
    """
    Exhaustively scans the output Parquet dataset chunk-by-chunk to verify
    data integrity, consistency, and adherence to all specifications.

    Parameters
    ----------
    parquet_path : str
        Path to sisfall_windows.parquet.
    batch_size : int, default=10000
        Number of rows per validation chunk.

    Returns
    -------
    Dict[str, Any]
        Comprehensive validation statistics and check statuses.
    """
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow

    print("\n" + "=" * 80)
    print(f"VALIDATING GENERATED PARQUET DATASET: {parquet_path}")
    print(f"Total Rows (Windows): {num_rows:,} | Columns: {len(schema.names)}")
    print("=" * 80)

    # Verification accumulators
    seen_window_ids = set()
    label_counts = {0: 0, 1: 0}
    activity_type_counts = {"ADL": 0, "FALL": 0}
    subject_counts = set()
    subject_group_counts = {"SA": 0, "SE": 0}

    nan_count = 0
    inf_count = 0
    length_mismatches = 0
    label_mismatches = 0
    boundary_mismatches = 0
    sa15_anomaly_count = 0

    rows_read = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        rows_read += len(df_batch)

        for _, row in df_batch.iterrows():
            win_id = row["window_id"]
            if win_id in seen_window_ids:
                raise ValueError(f"Duplicate window_id detected: {win_id}")
            seen_window_ids.add(win_id)

            subj = row["subject_id"]
            grp = row["subject_group"]
            act_type = row["activity_type"]
            lbl = row["label"]
            start_s = row["window_start_sample"]
            end_s = row["window_end_sample"]
            src_fn = row["source_filename"]

            subject_counts.add(subj)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            activity_type_counts[act_type] = activity_type_counts.get(act_type, 0) + 1
            subject_group_counts[grp] = subject_group_counts.get(grp, 0) + 1

            # Label / Activity correspondence
            if lbl == 0 and act_type != "ADL":
                label_mismatches += 1
            elif lbl == 1 and act_type != "FALL":
                label_mismatches += 1

            # Window length
            if end_s - start_s != WINDOW_SIZE_SAMPLES:
                boundary_mismatches += 1

            # SA15 anomaly verification
            if subj == "SA15" and "D17_SE15" in src_fn:
                sa15_anomaly_count += 1
                if grp != "SA":
                    label_mismatches += 1

            # Verify arrays
            for ch in WINDOW_CHANNELS:
                arr = row[ch]
                if len(arr) != WINDOW_SIZE_SAMPLES:
                    length_mismatches += 1
                narr = np.array(arr, dtype=np.float64)
                if np.isnan(narr).any():
                    nan_count += 1
                if np.isinf(narr).any():
                    inf_count += 1

        print(f"  Validated {rows_read:,} / {num_rows:,} windows...")

    checks = {
        "total_windows": num_rows,
        "unique_window_ids": len(seen_window_ids),
        "duplicate_window_ids": num_rows - len(seen_window_ids),
        "subjects_count": len(subject_counts),
        "adl_windows": label_counts.get(0, 0),
        "fall_windows": label_counts.get(1, 0),
        "subject_groups": subject_group_counts,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "length_mismatches": length_mismatches,
        "label_mismatches": label_mismatches,
        "boundary_mismatches": boundary_mismatches,
        "sa15_anomaly_windows_count": sa15_anomaly_count,
        "all_checks_passed": (
            nan_count == 0
            and inf_count == 0
            and length_mismatches == 0
            and label_mismatches == 0
            and boundary_mismatches == 0
            and (num_rows - len(seen_window_ids)) == 0
            and sa15_anomaly_count > 0
        ),
    }

    print("\n" + "=" * 80)
    print("PARQUET VALIDATION SUMMARY")
    print(f"  Total Windows: {checks['total_windows']:,}")
    print(f"  ADL Windows (0): {checks['adl_windows']:,} | FALL Windows (1): {checks['fall_windows']:,}")
    print(f"  Unique Subjects: {checks['subjects_count']} (SA Windows: {subject_group_counts['SA']:,}, SE Windows: {subject_group_counts['SE']:,})")
    print(f"  SA15 Anomaly Windows Successfully Attributed: {checks['sa15_anomaly_windows_count']}")
    print(f"  NaN / Inf Occurrences: {nan_count + inf_count}")
    print(f"  All Integrity Checks Passed: {checks['all_checks_passed']}")
    print("=" * 80)

    return checks


if __name__ == "__main__":
    rep_results = run_representative_window_tests()
    pq_results = validate_parquet_window_dataset()
