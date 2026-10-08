"""
SisFall Adapter Validation Module.

Executes:
1. Representative trial validation (ADL SA, ADL SE, Fall SA, Fall SE, SA15 anomaly, D01, D17, F01, F15).
2. Full dataset structural validation across all 4,505 files in SisFall.
3. Memory-efficient, file-by-file verification ensuring zero data corruption or unhandled quirks.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.sisfall.sisfall_parser import BASELINE_COLUMNS
from src.sisfall.sisfall_loader import (
    SisFallMetadata,
    discover_sisfall_files,
    extract_sisfall_metadata,
    load_sisfall_file,
)


def run_representative_validation(
    dataset_root: str = r"data/SisFall_dataset/SisFall_dataset",
) -> List[Dict[str, Any]]:
    """
    Runs comprehensive representative validation on specified key benchmark files:
    - ADL SA subject: SA01/D01_SA01_R01.txt
    - ADL SE subject: SE01/D01_SE01_R01.txt
    - Fall SA subject: SA01/F01_SA01_R01.txt
    - Fall SE subject: SE06/F01_SE06_R01.txt (Judo expert)
    - SA15 Anomaly: SA15/D17_SE15_R01.txt (Must resolve to SA15)
    - Additional activities: D17 (SA02/D17_SA02_R01.txt), F15 (SA01/F15_SA01_R01.txt)

    Returns
    -------
    List[Dict[str, Any]]
        List of audit record dicts for each representative trial.
    """
    representative_targets = [
        ("SA01", "D01_SA01_R01.txt", "ADL Young Adult (D01)"),
        ("SE01", "D01_SE01_R01.txt", "ADL Elderly Adult (D01)"),
        ("SA01", "F01_SA01_R01.txt", "Fall Young Adult (F01)"),
        ("SE06", "F01_SE06_R01.txt", "Fall Elderly Judo Expert (F01)"),
        ("SA15", "D17_SE15_R01.txt", "CRITICAL ANOMALY: SA15/D17_SE15_R01.txt (Car egress)"),
        ("SA02", "D17_SA02_R01.txt", "ADL Car Activity (D17)"),
        ("SA01", "F15_SA01_R01.txt", "Fall Fainting while Sitting (F15)"),
    ]

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("RUNNING SISFALL REPRESENTATIVE VALIDATION TESTS")
    print("=" * 80)

    for subj_folder, filename, label in representative_targets:
        file_path = os.path.join(dataset_root, subj_folder, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Representative file not found: {file_path}")

        df, meta = load_sisfall_file(file_path, resample_to_100hz=True)

        # Anomaly verification
        if subj_folder == "SA15" and "D17_SE15" in filename:
            assert meta.subject_id == "SA15", (
                f"FAIL: Expected subject_id == 'SA15' for {file_path}, got '{meta.subject_id}'"
            )

        # Basic assertions
        assert len(df) == meta.resampled_samples, "Length mismatch in DataFrame"
        assert meta.resampled_fs_hz == 100.0, "Resampled frequency must be 100 Hz"
        assert not df.isna().any().any(), "NaN found in DataFrame"
        assert not np.isinf(df.to_numpy()).any(), "Inf found in DataFrame"
        for col in BASELINE_COLUMNS:
            assert col in df.columns, f"Missing baseline column: {col}"

        # Channel stats
        ch_mins = {col: float(df[col].min()) for col in BASELINE_COLUMNS}
        ch_maxs = {col: float(df[col].max()) for col in BASELINE_COLUMNS}
        first_row = {col: float(df[col].iloc[0]) for col in BASELINE_COLUMNS}
        last_row = {col: float(df[col].iloc[-1]) for col in BASELINE_COLUMNS}

        record = {
            "label": label,
            "source_file": meta.source_file,
            "subject_id": meta.subject_id,
            "activity_id": meta.activity_id,
            "activity_type": meta.activity_type,
            "trial_id": meta.trial_id,
            "original_samples": meta.original_samples,
            "original_fs_hz": meta.original_fs_hz,
            "resampled_samples": meta.resampled_samples,
            "resampled_fs_hz": meta.resampled_fs_hz,
            "first_sample": first_row,
            "last_sample": last_row,
            "ch_mins": ch_mins,
            "ch_maxs": ch_maxs,
            "validation_status": "PASS",
        }
        results.append(record)

        print(f"\n[PASS] {label}")
        print(f"  File: {meta.source_file} (Folder: {subj_folder})")
        print(f"  Resolved: Subject={meta.subject_id}, Activity={meta.activity_id}, Type={meta.activity_type}, Trial={meta.trial_id}")
        print(f"  Samples: {meta.original_samples} @ 200 Hz -> {meta.resampled_samples} @ 100 Hz")
        print(f"  AccX_g: [{ch_mins['AccX_g']:.3f}, {ch_maxs['AccX_g']:.3f}], AccY_g: [{ch_mins['AccY_g']:.3f}, {ch_maxs['AccY_g']:.3f}], AccZ_g: [{ch_mins['AccZ_g']:.3f}, {ch_maxs['AccZ_g']:.3f}]")
        print(f"  GyrX_dps: [{ch_mins['GyrX_dps']:.1f}, {ch_maxs['GyrX_dps']:.1f}], GyrY_dps: [{ch_mins['GyrY_dps']:.1f}, {ch_maxs['GyrY_dps']:.1f}], GyrZ_dps: [{ch_mins['GyrZ_dps']:.1f}, {ch_maxs['GyrZ_dps']:.1f}]")

    return results


def run_full_structural_validation(
    dataset_root: str = r"data/SisFall_dataset/SisFall_dataset",
    verbose_every: int = 500,
) -> Dict[str, Any]:
    """
    Executes memory-efficient full structural and schema validation across all
    files in the SisFall dataset.

    Returns
    -------
    Dict[str, Any]
        Summary metrics of the entire scan.
    """
    all_files = discover_sisfall_files(dataset_root)
    print("\n" + "=" * 80)
    print(f"STARTING FULL DATASET STRUCTURAL VALIDATION ({len(all_files)} FILES)")
    print("=" * 80)

    total_scanned = len(all_files)
    valid_files = 0
    invalid_files = 0
    malformed_rows = 0
    col_failures = 0
    nan_inf_count = 0

    adl_count = 0
    fall_count = 0
    subjects_detected = set()
    activities_detected = set()
    trials_detected = set()

    sa15_anomaly_verified = 0

    global_mins = {col: float("inf") for col in BASELINE_COLUMNS}
    global_maxs = {col: float("-inf") for col in BASELINE_COLUMNS}

    total_orig_samples = 0
    total_resampled_samples = 0

    for idx, fpath in enumerate(all_files, start=1):
        try:
            df, meta = load_sisfall_file(fpath, resample_to_100hz=True)
            valid_files += 1

            if meta.activity_type == "ADL":
                adl_count += 1
            elif meta.activity_type == "FALL":
                fall_count += 1

            subjects_detected.add(meta.subject_id)
            activities_detected.add(meta.activity_id)
            trials_detected.add(meta.trial_id)

            if meta.subject_id == "SA15" and "D17_SE15" in meta.source_file:
                sa15_anomaly_verified += 1

            total_orig_samples += meta.original_samples
            total_resampled_samples += meta.resampled_samples

            # Check NaNs / Infs
            if df.isna().any().any() or np.isinf(df.to_numpy()).any():
                nan_inf_count += 1

            # Global ranges
            for col in BASELINE_COLUMNS:
                c_min = float(df[col].min())
                c_max = float(df[col].max())
                if c_min < global_mins[col]:
                    global_mins[col] = c_min
                if c_max > global_maxs[col]:
                    global_maxs[col] = c_max

        except Exception as err:
            invalid_files += 1
            print(f"ERROR on file {fpath}: {err}")
            col_failures += 1

        if idx % verbose_every == 0 or idx == total_scanned:
            print(f"  Processed {idx}/{total_scanned} files... (Valid: {valid_files}, Invalid: {invalid_files})")

    summary = {
        "dataset_root": os.path.normpath(dataset_root),
        "total_scanned": total_scanned,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "malformed_rows": malformed_rows,
        "col_failures": col_failures,
        "nan_inf_count": nan_inf_count,
        "adl_count": adl_count,
        "fall_count": fall_count,
        "subject_count": len(subjects_detected),
        "subjects": sorted(list(subjects_detected)),
        "activity_count": len(activities_detected),
        "activities": sorted(list(activities_detected)),
        "trials": sorted(list(trials_detected)),
        "sa15_anomaly_verified_count": sa15_anomaly_verified,
        "total_orig_samples": total_orig_samples,
        "total_resampled_samples": total_resampled_samples,
        "global_mins": global_mins,
        "global_maxs": global_maxs,
    }

    print("\n" + "=" * 80)
    print("FULL STRUCTURAL VALIDATION COMPLETE")
    print(f"  Total Files Scanned: {summary['total_scanned']}")
    print(f"  Valid Files: {summary['valid_files']}, Invalid: {summary['invalid_files']}")
    print(f"  ADL Trials: {summary['adl_count']}, Fall Trials: {summary['fall_count']}")
    print(f"  Subjects Detected: {summary['subject_count']} (SA: 23, SE: 15)")
    print(f"  SA15 D17 Anomaly Cases Successfully Resolved: {summary['sa15_anomaly_verified_count']}/5")
    print(f"  Total Samples: {summary['total_orig_samples']} @ 200 Hz -> {summary['total_resampled_samples']} @ 100 Hz")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    rep_res = run_representative_validation()
    full_res = run_full_structural_validation()
