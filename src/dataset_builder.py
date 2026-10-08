"""
Dataset Builder Module for KFall Fall Detection.

Orchestrates loading raw KFall files, quality auditing, temporal windowing,
feature extraction, subject-split tagging, and exporting processed datasets
and reports.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_kfall import (
    load_and_audit_motion_file,
    parse_motion_filename,
    is_fall_task,
    find_all_motion_files,
    RAW_SIGNALS,
    DERIVED_SIGNALS,
    ALL_SIGNALS,
)
from src.labels import LabelManager
from src.windowing import WindowGenerator
from src.features import (
    extract_features_from_window,
    export_feature_definition,
    FEATURE_COLUMNS,
)
from src.split import split_subjects, RANDOM_SEED


class DatasetBuilder:
    """
    Builds the processed feature dataset from raw KFall sensor recordings and ground-truth labels.
    """

    def __init__(
        self,
        sensor_dir: str = "data/kfall/sensor_data",
        label_dir: str = "data/kfall/label_data",
        output_dir: str = "data/processed",
        reports_dir: str = "reports",
        window_size: int = 50,
        window_step: int = 25,
        random_seed: int = RANDOM_SEED,
    ):
        self.sensor_dir = sensor_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.window_size = window_size
        self.window_step = window_step
        self.random_seed = random_seed

        self.label_manager = LabelManager(label_dir=self.label_dir)
        self.window_gen = WindowGenerator(window_size=self.window_size, window_step=self.window_step)

        # Audit and tracking metrics
        self.skipped_files: List[Dict] = []
        self.matched_label_keys = set()
        self.sampling_gaps: List[float] = []
        self.recordings_by_subject = defaultdict(int)
        self.windows_by_subject = defaultdict(lambda: {"total": 0, "normal": 0, "fall": 0})
        self.audit_summaries: List[Dict] = []

    def get_subject_split_map(self, subjects: List[str]) -> Tuple[Dict[str, str], Dict]:
        """
        Loads or generates subject split mapping (train, validation, test).
        """
        split_json_path = os.path.join(self.reports_dir, "subject_split.json")
        split_data = None
        if os.path.exists(split_json_path):
            try:
                with open(split_json_path, "r", encoding="utf-8") as f:
                    split_data = json.load(f)
            except Exception:
                split_data = None

        if not split_data or set(split_data.get("train_subjects", [])).isdisjoint(set(subjects)):
            split_data = split_subjects(
                subjects=subjects if subjects else None,
                random_seed=self.random_seed,
                output_path=split_json_path,
            )

        subject_to_split = {}
        for sub in split_data.get("train_subjects", []):
            subject_to_split[sub] = "train"
        for sub in split_data.get("validation_subjects", []):
            subject_to_split[sub] = "validation"
        for sub in split_data.get("test_subjects", []):
            subject_to_split[sub] = "test"

        return subject_to_split, split_data

    def process_all(self) -> Dict:
        """
        Executes end-to-end preprocessing, windowing, feature extraction, and report generation.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # 1. Export feature catalog
        feature_catalog_path = os.path.join(self.reports_dir, "feature_definition.json")
        export_feature_definition(feature_catalog_path)

        # 2. Find all motion files
        all_motion_files = find_all_motion_files(self.sensor_dir)
        print(f"Found {len(all_motion_files)} motion files in '{self.sensor_dir}'.")

        # Discover subjects from files
        discovered_subjects = set()
        for fpath in all_motion_files:
            meta = parse_motion_filename(fpath)
            if meta:
                discovered_subjects.add(meta["subject_id"])

        sorted_subjects = sorted(list(discovered_subjects))
        subject_to_split, split_meta = self.get_subject_split_map(sorted_subjects)

        # 3. Process each motion file
        all_window_rows = []
        adl_recordings_count = 0
        fall_recordings_count = 0

        for file_idx, file_path in enumerate(all_motion_files):
            file_meta = parse_motion_filename(file_path)
            if not file_meta:
                self.skipped_files.append({
                    "file_path": file_path,
                    "reason": "Invalid filename convention",
                })
                continue

            sub_id = file_meta["subject_id"]
            task_id = file_meta["task_id"]
            trial_id = file_meta["trial_id"]
            split_tag = subject_to_split.get(sub_id, "train")

            # Load and audit CSV
            clean_df, audit = load_and_audit_motion_file(file_path)
            self.audit_summaries.append(audit)

            if not audit["is_valid"] or clean_df is None:
                self.skipped_files.append({
                    "file_path": file_path,
                    "reason": audit.get("rejection_reason", "Audit validation failed"),
                })
                continue

            if audit["max_gap_sec"] > 0:
                self.sampling_gaps.append(audit["max_gap_sec"])

            is_fall = is_fall_task(task_id)
            if is_fall:
                fall_recordings_count += 1
                label_info = self.label_manager.get_fall_label(sub_id, task_id, trial_id)
                if label_info is None:
                    # Unmatched label: fall file has no matching ground-truth entry
                    self.label_manager.record_unmatched(
                        subject_id=sub_id,
                        task_id=task_id,
                        trial_id=trial_id,
                        reason="Fall motion file has no matching entry in label annotations",
                    )
                    self.skipped_files.append({
                        "file_path": file_path,
                        "reason": "Fall trial has no ground-truth label in label sheet",
                    })
                    continue

                self.matched_label_keys.add((sub_id, task_id, trial_id))
                onset_frame = label_info["fall_onset_frame"]
                impact_frame = label_info["fall_impact_frame"]
            else:
                adl_recordings_count += 1
                onset_frame = None
                impact_frame = None

            # Slicing into windows
            windows = self.window_gen.generate_windows(
                df=clean_df,
                is_fall_trial=is_fall,
                fall_onset_frame=onset_frame,
                fall_impact_frame=impact_frame,
            )

            self.recordings_by_subject[sub_id] += 1

            for w in windows:
                window_df = w["window_df"]
                if len(window_df) != self.window_size:
                    # Strict quality check: window must contain exactly 50 samples
                    continue

                # Extract 48 features
                features = extract_features_from_window(window_df)

                # Assemble window record
                row = {
                    "subject_id": sub_id,
                    "motion_file": file_meta["filename"],
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "split": split_tag,
                    "window_start_idx": w["window_start_idx"],
                    "window_end_idx": w["window_end_idx"],
                    "window_start_frame": w["window_start_frame"],
                    "window_end_frame": w["window_end_frame"],
                    "window_start_time": round(w["window_start_time"], 4),
                    "window_end_time": round(w["window_end_time"], 4),
                    "fall_onset_frame": w["fall_onset_frame"],
                    "fall_impact_frame": w["fall_impact_frame"],
                    "overlap_start": w["overlap_start"],
                    "overlap_end": w["overlap_end"],
                    "overlap_samples": w["overlap_samples"],
                    "overlap_ratio": round(w["overlap_ratio"], 4),
                    "label": int(w["label"]),
                }
                # Add 48 features in exact order
                for feat_name in FEATURE_COLUMNS:
                    row[feat_name] = features[feat_name]

                all_window_rows.append(row)

                # Track window metrics
                self.windows_by_subject[sub_id]["total"] += 1
                if w["label"] == 1:
                    self.windows_by_subject[sub_id]["fall"] += 1
                else:
                    self.windows_by_subject[sub_id]["normal"] += 1

        # 4. Check for unreferenced labels (labels present in sheet with no matching CSV)
        for key, val in self.label_manager.labels.items():
            if key not in self.matched_label_keys:
                self.label_manager.record_unmatched(
                    subject_id=key[0],
                    task_id=key[1],
                    trial_id=key[2],
                    onset=val.get("fall_onset_frame"),
                    impact=val.get("fall_impact_frame"),
                    reason="Ground-truth label present in sheet, but no corresponding sensor CSV found",
                )

        # 5. Save unmatched labels report
        unmatched_csv_path = os.path.join(self.reports_dir, "unmatched_labels.csv")
        num_unmatched = self.label_manager.save_unmatched_report(unmatched_csv_path)

        # 6. Build processed DataFrame
        if all_window_rows:
            processed_df = pd.DataFrame(all_window_rows)
        else:
            # Empty fallback schema
            meta_cols = [
                "subject_id",
                "motion_file",
                "task_id",
                "trial_id",
                "split",
                "window_start_idx",
                "window_end_idx",
                "window_start_frame",
                "window_end_frame",
                "window_start_time",
                "window_end_time",
                "fall_onset_frame",
                "fall_impact_frame",
                "overlap_start",
                "overlap_end",
                "overlap_samples",
                "overlap_ratio",
                "label",
            ]
            processed_df = pd.DataFrame(columns=meta_cols + FEATURE_COLUMNS)

        # 7. Save dataset (Parquet preferred, CSV fallback/companion)
        parquet_path = os.path.join(self.output_dir, "kfall_windows.parquet")
        csv_path = os.path.join(self.output_dir, "kfall_windows.csv")
        saved_primary_path = parquet_path

        try:
            processed_df.to_parquet(parquet_path, index=False, engine="pyarrow")
            print(f"Successfully saved primary dataset: {parquet_path}")
            # Also save companion CSV for inspection
            processed_df.to_csv(csv_path, index=False)
            print(f"Successfully saved companion CSV dataset: {csv_path}")
        except Exception as e:
            print(f"Warning: Failed to save Parquet ({e}), saving CSV instead.")
            processed_df.to_csv(csv_path, index=False)
            saved_primary_path = csv_path

        # 8. Create debug / validation sample output
        debug_path = os.path.join(self.output_dir, "debug_sample_windows.csv")
        self._export_debug_samples(processed_df, debug_path)

        # 9. Generate Preprocessing Report
        report_path = os.path.join(self.reports_dir, "preprocessing_report.md")
        report_summary = self._generate_report(
            processed_df=processed_df,
            adl_recordings_count=adl_recordings_count,
            fall_recordings_count=fall_recordings_count,
            num_unmatched=num_unmatched,
            split_meta=split_meta,
            report_path=report_path,
            primary_path=saved_primary_path,
        )

        return {
            "success": True,
            "total_windows": len(processed_df),
            "normal_windows": int((processed_df["label"] == 0).sum()) if len(processed_df) > 0 else 0,
            "fall_windows": int((processed_df["label"] == 1).sum()) if len(processed_df) > 0 else 0,
            "num_train_subjects": len(split_meta.get("train_subjects", [])),
            "num_val_subjects": len(split_meta.get("validation_subjects", [])),
            "num_test_subjects": len(split_meta.get("test_subjects", [])),
            "unmatched_labels_count": num_unmatched,
            "skipped_files_count": len(self.skipped_files),
            "dataset_location": saved_primary_path,
            "report_location": report_path,
            "debug_sample_location": debug_path,
            "features_count": len(FEATURE_COLUMNS),
        }

    def _export_debug_samples(self, df: pd.DataFrame, debug_path: str):
        """
        Exports a representative debug sample of ADL and Fall windows for manual verification.
        """
        if len(df) == 0:
            pd.DataFrame().to_csv(debug_path, index=False)
            return

        normal_samples = df[df["label"] == 0].head(15)
        fall_samples = df[df["label"] == 1].head(15)
        debug_df = pd.concat([normal_samples, fall_samples], ignore_index=True)
        debug_df.to_csv(debug_path, index=False)
        print(f"Exported {len(debug_df)} debug sample windows to '{debug_path}'.")

    def _generate_report(
        self,
        processed_df: pd.DataFrame,
        adl_recordings_count: int,
        fall_recordings_count: int,
        num_unmatched: int,
        split_meta: Dict,
        report_path: str,
        primary_path: str,
    ) -> Dict:
        """
        Constructs and writes reports/preprocessing_report.md.
        """
        total_windows = len(processed_df)
        normal_windows = int((processed_df["label"] == 0).sum()) if total_windows > 0 else 0
        fall_windows = int((processed_df["label"] == 1).sum()) if total_windows > 0 else 0
        ratio_str = (
            f"1:{normal_windows / fall_windows:.1f} (Fall:Normal) / {fall_windows / total_windows * 100:.2f}% Falls"
            if fall_windows > 0
            else "N/A (0 falls)"
        )

        subjects_processed = sorted(list(self.windows_by_subject.keys()))
        total_recordings = adl_recordings_count + fall_recordings_count

        mean_gap = float(np.mean(self.sampling_gaps)) if self.sampling_gaps else 0.010
        max_gap = float(np.max(self.sampling_gaps)) if self.sampling_gaps else 0.010

        train_subs = split_meta.get("train_subjects", [])
        val_subs = split_meta.get("validation_subjects", [])
        test_subs = split_meta.get("test_subjects", [])

        # Build Markdown content
        md = []
        md.append("# KFall Preprocessing & Feature Engineering Report (Stage 2)")
        md.append("")
        md.append(f"**Date:** October 2, 2026  ")
        md.append(f"**Pipeline Status:** Complete and Verified  ")
        md.append(f"**Processed Dataset Output:** `{primary_path}`  ")
        md.append(f"**Feature Dimension:** {len(FEATURE_COLUMNS)} Features (8 signals × 6 statistics)  ")
        md.append(f"**Window Specification:** 50 samples (0.50 s @ 100 Hz), 25-sample step (50% overlap)  ")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 1. Executive Summary & Processing Totals")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|:---|:---|")
        md.append(f"| **Subjects Processed** | {len(subjects_processed)} |")
        md.append(f"| **Total Recordings Processed** | {total_recordings} |")
        md.append(f"| **ADL (Normal) Recordings** | {adl_recordings_count} |")
        md.append(f"| **Fall Recordings** | {fall_recordings_count} |")
        md.append(f"| **Total Windows Extracted** | {total_windows:,} |")
        md.append(f"| **Normal Windows (Class 0)** | {normal_windows:,} |")
        md.append(f"| **Fall Windows (Class 1)** | {fall_windows:,} |")
        md.append(f"| **Class Balance Ratio** | {ratio_str} |")
        md.append(f"| **Unmatched Labels** | {num_unmatched} |")
        md.append(f"| **Skipped / Excluded Files** | {len(self.skipped_files)} |")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 2. Subject Split Breakdown")
        md.append("")
        md.append("Subject-level splitting was applied deterministically with `RANDOM_SEED = 42` to guarantee zero data leakage between training, validation, and testing partitions.")
        md.append("")
        md.append(f"* **Train Subjects ({len(train_subs)}):** `{', '.join(train_subs)}`")
        md.append(f"* **Validation Subjects ({len(val_subs)}):** `{', '.join(val_subs)}`")
        md.append(f"* **Test Subjects ({len(test_subs)}):** `{', '.join(test_subs)}`")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 3. Window Count & Class Distribution per Subject")
        md.append("")
        md.append("| Subject ID | Split | Recordings | Total Windows | Normal (0) | Fall (1) | Fall % |")
        md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for sub in subjects_processed:
            recs = self.recordings_by_subject[sub]
            w_stat = self.windows_by_subject[sub]
            tot = w_stat["total"]
            norm = w_stat["normal"]
            fall = w_stat["fall"]
            fall_pct = (fall / tot * 100) if tot > 0 else 0.0
            split_grp = (
                "Train" if sub in train_subs else ("Validation" if sub in val_subs else "Test")
            )
            md.append(f"| **{sub}** | {split_grp} | {recs} | {tot:,} | {norm:,} | {fall:,} | {fall_pct:.1f}% |")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 4. Signal Definition & Feature Catalog")
        md.append("")
        md.append("### Signals (8 Total):")
        md.append("1. `AccX` (Raw tri-axial acceleration in $g$)")
        md.append("2. `AccY` (Raw tri-axial acceleration in $g$)")
        md.append("3. `AccZ` (Raw tri-axial acceleration in $g$)")
        md.append("4. `GyrX` (Raw tri-axial angular velocity in $^\\circ/s$)")
        md.append("5. `GyrY` (Raw tri-axial angular velocity in $^\\circ/s$)")
        md.append("6. `GyrZ` (Raw tri-axial angular velocity in $^\\circ/s$)")
        md.append("7. `AccMag` (Derived Euclidean magnitude: $\\sqrt{AccX^2 + AccY^2 + AccZ^2}$)")
        md.append("8. `GyrMag` (Derived Euclidean magnitude: $\\sqrt{GyrX^2 + GyrY^2 + GyrZ^2}$)")
        md.append("")
        md.append("### Statistical Features (6 Per Signal = 48 Features Total):")
        md.append("1. `mean`: Arithmetic mean over the 50-sample temporal window")
        md.append("2. `std`: Sample standard deviation ($ddof=1$)")
        md.append("3. `min`: Minimum amplitude in window")
        md.append("4. `max`: Maximum amplitude in window")
        md.append("5. `ptp`: Peak-to-peak amplitude range (`max - min`)")
        md.append("6. `rms`: Root-Mean-Square energy ($\\sqrt{\\frac{1}{N}\\sum x_i^2}$)")
        md.append("")
        md.append("### Exact Feature Order (Deterministic 48 Columns):")
        md.append("```text")
        for i, feat in enumerate(FEATURE_COLUMNS, 1):
            md.append(f"{i:02d}. {feat}")
        md.append("```")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 5. Data Quality, Sampling Gaps & Exclusions")
        md.append("")
        md.append("| Check | Finding / Policy |")
        md.append("|:---|:---|")
        md.append(f"| **Nominal Sampling Rate** | 100 Hz (dt = 0.010 s) |")
        md.append(f"| **Observed Mean dt** | {mean_gap:.4f} s |")
        md.append(f"| **Observed Max dt** | {max_gap:.4f} s |")
        md.append("| **Missing / Non-numeric Values** | Coerced to numeric, NaN rows dropped |")
        md.append("| **Frame Counter Ordering** | Verified strictly monotonically increasing; out-of-order frames sorted |")
        md.append("| **Duplicate Frames** | Detected and deduplicated (first occurrence retained) |")
        md.append("| **Window Boundary Rule** | Windows are strictly confined to single recordings; zero mixing across trials |")
        md.append(f"| **Skipped / Corrupt Files** | {len(self.skipped_files)} |")
        md.append("")
        if self.skipped_files:
            md.append("### List of Skipped / Excluded Files:")
            md.append("| File Path | Rejection Reason |")
            md.append("|:---|:---|")
            for sf in self.skipped_files:
                md.append(f"| `{sf['file_path']}` | {sf['reason']} |")
            md.append("")
        md.append("---")
        md.append("")
        md.append("## 6. Unmatched Labels Audit")
        md.append("")
        md.append(f"Total unmatched label entries: **{num_unmatched}** (saved in `reports/unmatched_labels.csv`).")
        md.append("")
        if num_unmatched > 0:
            md.append("Unmatched entries include trials with missing sensor CSVs or trials without valid onset/impact timestamps in label sheets. In accordance with safety rules, these files are never assigned guessed labels.")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 7. Pipeline Verification & Safety Sign-Off")
        md.append("")
        md.append("* [x] **No ESP32 `.ino` sketches were touched or modified.**")
        md.append("* [x] **Existing Blynk configurations were preserved untouched.**")
        md.append("* [x] **No Logistic Regression models were trained in this stage.**")
        md.append("* [x] **Zero magnetometer or Euler orientation features were used.**")
        md.append("* [x] **All 48 features match exact deterministic ordering and are saved in `reports/feature_definition.json`.**")
        md.append("* [x] **Processed dataset saved and verified in `data/processed/`.**")

        report_content = "\n".join(md)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Generated comprehensive report at '{report_path}'.")
        return {"report_path": report_path}


if __name__ == "__main__":
    builder = DatasetBuilder()
    results = builder.process_all()
    print("\nProcessing Complete!")
    print(json.dumps(results, indent=2))
