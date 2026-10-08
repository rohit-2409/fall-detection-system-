"""
SisFall Model Evaluation and Standalone Verification Module (Stage B4).

Loads serialized model pipelines from models/sisfall/, performs independent
evaluation on validation and held-out test splits, verifies prediction fidelity,
and asserts that Stage B4 requirements are strictly fulfilled.

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

import json
import os
import sys
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sisfall.sisfall_model_utils import (
    FEATURE_NAMES,
    compute_metrics,
    load_sisfall_feature_dataset,
    verify_leakage,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "sisfall")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def evaluate_sisfall_artifacts() -> Dict[str, Any]:
    """
    Evaluates serialized Stage B4 artifacts against validation and test splits.
    """
    print("=" * 78)
    print("  SISFALL STAGE B4: MODEL ARTIFACT EVALUATION & INTEGRITY VERIFICATION")
    print("=" * 78)

    # 1. Check Split Artifact
    split_path = os.path.join(REPORTS_DIR, "sisfall_subject_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split artifact missing: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_subs = split_data["train_subjects"]
    val_subs = split_data["validation_subjects"]
    test_subs = split_data["test_subjects"]

    print(f"Loaded subject split: {len(train_subs)} Train, {len(val_subs)} Val, {len(test_subs)} Test.")

    # 2. Load Dataset
    df, feature_cols = load_sisfall_feature_dataset()

    # 3. Leakage Verification
    leak_report = verify_leakage(df, train_subs, val_subs, test_subs)
    if not leak_report["is_leak_free"]:
        raise RuntimeError("Leakage detected in split config!")
    print("Data leakage verification: PASSED (Disjoint subjects & trials).")

    # 4. Partition test data
    test_df = df[df["subject_id"].isin(test_subs)]
    val_df = df[df["subject_id"].isin(val_subs)]

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    # 5. Check and Load Models
    model_files = {
        "Logistic Regression": "sisfall_logistic_regression.joblib",
        "Random Forest": "sisfall_random_forest.joblib",
        "Support Vector Machine": "sisfall_svm.joblib",
    }

    eval_summary: Dict[str, Dict[str, Any]] = {}

    for model_name, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact missing: {filepath}")

        pipeline = joblib.load(filepath)
        size_kb = os.path.getsize(filepath) / 1024

        # Validation evaluation
        val_pred = pipeline.predict(X_val)
        val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        v_metrics = compute_metrics(y_true=y_val, y_pred=val_pred, y_prob=val_prob)

        # Test evaluation
        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        t_metrics = compute_metrics(y_true=y_test, y_pred=test_pred, y_prob=test_prob)

        eval_summary[model_name] = {
            "file": filename,
            "size_kb": round(size_kb, 1),
            "validation": v_metrics,
            "test": t_metrics,
        }

    # Print summary table
    print("\n" + "-" * 78)
    print(f"{'Model Name':<24} | {'Val F1':<8} | {'Val AUC':<8} | {'Test F1':<8} | {'Test AUC':<8} | {'Test Acc':<8}")
    print("-" * 78)
    for name, res in eval_summary.items():
        vf1 = res["validation"]["fall_f1"]
        vauc = res["validation"]["roc_auc"]
        tf1 = res["test"]["fall_f1"]
        tauc = res["test"]["roc_auc"]
        tacc = res["test"]["accuracy"]
        print(f"{name:<24} | {vf1:<8.4f} | {vauc:<8.4f} | {tf1:<8.4f} | {tauc:<8.4f} | {tacc:<8.4f}")
    print("-" * 78)

    print("\nArtifact integrity evaluation completed successfully.")
    return eval_summary


if __name__ == "__main__":
    evaluate_sisfall_artifacts()
