"""
Labels Module for KFall Fall Detection.

Loads and standardizes ground-truth fall annotations from SAxx_label.xlsx/.csv,
mapping subject, task, and trial identifiers to onset and impact frame counts.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd


class LabelManager:
    """
    Manages loading and querying of KFall ground-truth label annotations.
    """

    def __init__(self, label_dir: str = "data/kfall/label_data"):
        self.label_dir = label_dir
        self.labels: Dict[Tuple[str, str, str], Dict] = {}
        self.unmatched_records: List[Dict] = []
        self._load_all_labels()

    @staticmethod
    def normalize_task_id(task_val) -> str:
        """Normalize task identifier to Txx format."""
        s = str(task_val).strip()
        m = re.search(r"(\d+)", s)
        if m:
            return f"T{int(m.group(1)):02d}"
        return s.upper()

    @staticmethod
    def normalize_trial_id(trial_val) -> str:
        """Normalize trial identifier to Rxx format."""
        s = str(trial_val).strip()
        m = re.search(r"(\d+)", s)
        if m:
            return f"R{int(m.group(1)):02d}"
        return s.upper()

    @staticmethod
    def normalize_subject_id(subject_val) -> str:
        """Normalize subject identifier to SAxx format."""
        s = str(subject_val).strip().upper()
        m = re.search(r"(\d+)", s)
        if m:
            return f"SA{int(m.group(1)):02d}"
        return s

    def _load_all_labels(self):
        """Scans label_dir and parses all Excel/CSV label files."""
        if not os.path.exists(self.label_dir):
            return

        for fname in os.listdir(self.label_dir):
            fpath = os.path.join(self.label_dir, fname)
            if not os.path.isfile(fpath):
                continue

            # Extract subject ID from filename e.g. SA06_label.xlsx
            sub_match = re.search(r"(SA\d+)", fname, re.IGNORECASE)
            if not sub_match:
                continue
            subject_id = self.normalize_subject_id(sub_match.group(1))

            try:
                if fname.lower().endswith(".xlsx"):
                    df = pd.read_excel(fpath)
                elif fname.lower().endswith(".csv"):
                    df = pd.read_csv(fpath)
                else:
                    continue
            except Exception as e:
                self.record_unmatched(
                    subject_id=subject_id,
                    task_id="ALL",
                    trial_id="ALL",
                    onset=None,
                    impact=None,
                    reason=f"Failed to read label file: {str(e)}",
                )
                continue

            self._parse_label_df(subject_id, df, fname)

    def _parse_label_df(self, subject_id: str, df: pd.DataFrame, source_file: str):
        """Standardizes and extracts rows from a participant's label sheet."""
        # Normalize column names for flexible matching
        col_map = {}
        for col in df.columns:
            clean = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if "task" in clean:
                col_map[col] = "task_code"
            elif "trial" in clean:
                col_map[col] = "trial_id"
            elif "onset" in clean or "init" in clean:
                col_map[col] = "fall_onset_frame"
            elif "impact" in clean:
                col_map[col] = "fall_impact_frame"
            elif "desc" in clean:
                col_map[col] = "description"

        df_renamed = df.rename(columns=col_map)
        required = ["task_code", "trial_id", "fall_onset_frame", "fall_impact_frame"]
        missing_req = [c for c in required if c not in df_renamed.columns]
        if missing_req:
            self.record_unmatched(
                subject_id=subject_id,
                task_id="ALL",
                trial_id="ALL",
                onset=None,
                impact=None,
                reason=f"Missing required columns {missing_req} in {source_file}",
            )
            return

        for _, row in df_renamed.iterrows():
            raw_task = row.get("task_code")
            raw_trial = row.get("trial_id")
            if pd.isna(raw_task) or pd.isna(raw_trial):
                continue

            task_id = self.normalize_task_id(raw_task)
            trial_id = self.normalize_trial_id(raw_trial)
            desc = str(row.get("description", ""))

            raw_onset = row.get("fall_onset_frame")
            raw_impact = row.get("fall_impact_frame")

            try:
                onset_frame = int(round(float(raw_onset))) if not pd.isna(raw_onset) else None
                impact_frame = int(round(float(raw_impact))) if not pd.isna(raw_impact) else None
            except (ValueError, TypeError):
                onset_frame = None
                impact_frame = None

            if onset_frame is None or impact_frame is None:
                self.record_unmatched(
                    subject_id=subject_id,
                    task_id=task_id,
                    trial_id=trial_id,
                    onset=raw_onset,
                    impact=raw_impact,
                    reason=f"Non-numeric or NaN onset/impact frames in {source_file}",
                )
                continue

            key = (subject_id, task_id, trial_id)
            self.labels[key] = {
                "subject_id": subject_id,
                "task_id": task_id,
                "trial_id": trial_id,
                "fall_onset_frame": onset_frame,
                "fall_impact_frame": impact_frame,
                "description": desc,
                "source_file": source_file,
            }

    def get_fall_label(self, subject_id: str, task_id: str, trial_id: str) -> Optional[Dict]:
        """
        Retrieves the fall onset and impact annotations for a specific trial.
        Returns dict or None if not found.
        """
        key = (
            self.normalize_subject_id(subject_id),
            self.normalize_task_id(task_id),
            self.normalize_trial_id(trial_id),
        )
        return self.labels.get(key)

    def record_unmatched(
        self,
        subject_id: str,
        task_id: str,
        trial_id: str,
        onset=None,
        impact=None,
        reason: str = "Unmatched label",
    ):
        """Records an unmatched or anomalous label entry."""
        self.unmatched_records.append({
            "subject_id": subject_id,
            "task_id": task_id,
            "trial_id": trial_id,
            "fall_onset_frame": onset,
            "fall_impact_frame": impact,
            "reason": reason,
        })

    def save_unmatched_report(self, output_path: str = "reports/unmatched_labels.csv"):
        """Saves unmatched records to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.unmatched_records:
            df = pd.DataFrame(self.unmatched_records)
        else:
            df = pd.DataFrame(
                columns=[
                    "subject_id",
                    "task_id",
                    "trial_id",
                    "fall_onset_frame",
                    "fall_impact_frame",
                    "reason",
                ]
            )
        df.to_csv(output_path, index=False)
        return len(self.unmatched_records)
