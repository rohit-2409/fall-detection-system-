"""
Model Utilities Module for KFall Fall Detection (Stage 3).

Provides dataset loading, strict programmatic data leakage audits,
metric calculation, and visualization utilities for confusion matrix and ROC curve.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_audit_dataset(
    parquet_path: str = os.path.join(PROJECT_ROOT, "data", "processed", "kfall_windows.parquet"),
    feature_def_path: str = os.path.join(PROJECT_ROOT, "reports", "feature_definition.json"),
    split_json_path: str = os.path.join(PROJECT_ROOT, "reports", "subject_split.json"),
) -> Tuple[pd.DataFrame, List[str], Dict, Dict]:
    """
    Loads processed dataset, validates features and targets, and executes strict
    programmatic data leakage verification across subject-level splits.

    Fails loudly on any integrity or leakage violation.
    """
    # 1. Verify existence of prerequisite artifacts
    for p, desc in [
        (parquet_path, "Processed dataset parquet"),
        (feature_def_path, "Feature definition JSON"),
        (split_json_path, "Subject split JSON"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Critical Stage 3 requirement missing: {desc} at '{p}'.")

    # 2. Load dataset
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise ValueError("Processed dataset is empty. Cannot train models.")

    # 3. Load feature definitions
    with open(feature_def_path, "r", encoding="utf-8") as f:
        feat_def = json.load(f)

    feature_cols = feat_def.get("feature_order", [])
    if len(feature_cols) != 48:
        raise ValueError(
            f"Expected exactly 48 feature definitions, found {len(feature_cols)} in {feature_def_path}."
        )

    # Check for duplicate feature definitions
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("Duplicate feature names detected in feature definition JSON.")

    # Verify all 48 features exist in DataFrame
    missing_feats = [col for col in feature_cols if col not in df.columns]
    if missing_feats:
        raise KeyError(f"Dataset is missing required features: {missing_feats}")

    # Check for NaNs or Infinities in features
    X_subset = df[feature_cols]
    if X_subset.isna().any().any():
        raise ValueError("NaN values detected in feature columns.")
    if np.isinf(X_subset.values).any():
        raise ValueError("Infinite values detected in feature columns.")

    # 4. Validate Target Column
    if "label" not in df.columns:
        raise KeyError("Target column 'label' not found in dataset.")

    unique_labels = sorted(df["label"].unique().tolist())
    if unique_labels != [0, 1] and unique_labels != [0] and unique_labels != [1]:
        raise ValueError(f"Unexpected target classes {unique_labels}. Expected binary labels [0, 1].")

    if df["label"].isna().any():
        raise ValueError("Target column contains null/NaN values.")

    # 5. Programmatic Data Leakage Audit
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    cohort_train_subs = set(split_data.get("train_subjects", []))
    cohort_val_subs = set(split_data.get("validation_subjects", []))
    cohort_test_subs = set(split_data.get("test_subjects", []))

    # Audit cohort split disjointness
    if not cohort_train_subs.isdisjoint(cohort_val_subs):
        overlap = cohort_train_subs.intersection(cohort_val_subs)
        raise RuntimeError(f"DATA LEAKAGE DETECTED in split config: Train & Validation overlap on {overlap}")
    if not cohort_train_subs.isdisjoint(cohort_test_subs):
        overlap = cohort_train_subs.intersection(cohort_test_subs)
        raise RuntimeError(f"DATA LEAKAGE DETECTED in split config: Train & Test overlap on {overlap}")
    if not cohort_val_subs.isdisjoint(cohort_test_subs):
        overlap = cohort_val_subs.intersection(cohort_test_subs)
        raise RuntimeError(f"DATA LEAKAGE DETECTED in split config: Validation & Test overlap on {overlap}")

    # Audit actual subjects appearing in dataset partitions
    if "split" not in df.columns or "subject_id" not in df.columns:
        raise KeyError("Dataset missing 'split' or 'subject_id' columns.")

    ds_train = df[df["split"] == "train"]
    ds_val = df[df["split"] == "validation"]
    ds_test = df[df["split"] == "test"]

    if len(ds_train) == 0:
        raise ValueError("Training split contains 0 rows.")
    if len(ds_val) == 0:
        raise ValueError("Validation split contains 0 rows.")
    if len(ds_test) == 0:
        raise ValueError("Test split contains 0 rows.")

    actual_train_subs = set(ds_train["subject_id"].unique())
    actual_val_subs = set(ds_val["subject_id"].unique())
    actual_test_subs = set(ds_test["subject_id"].unique())

    # Direct disjointness checks on actual partition rows
    if not actual_train_subs.isdisjoint(actual_val_subs):
        overlap = actual_train_subs.intersection(actual_val_subs)
        raise RuntimeError(f"CRITICAL LEAKAGE: Subject(s) {overlap} present in BOTH Train and Validation data!")
    if not actual_train_subs.isdisjoint(actual_test_subs):
        overlap = actual_train_subs.intersection(actual_test_subs)
        raise RuntimeError(f"CRITICAL LEAKAGE: Subject(s) {overlap} present in BOTH Train and Test data!")
    if not actual_val_subs.isdisjoint(actual_test_subs):
        overlap = actual_val_subs.intersection(actual_test_subs)
        raise RuntimeError(f"CRITICAL LEAKAGE: Subject(s) {overlap} present in BOTH Validation and Test data!")

    # Verify subjects map strictly to configured cohorts
    for sub in actual_train_subs:
        if sub not in cohort_train_subs:
            raise RuntimeError(f"Subject {sub} in Train data is not an approved Train subject in split config.")
    for sub in actual_val_subs:
        if sub not in cohort_val_subs:
            raise RuntimeError(f"Subject {sub} in Validation data is not an approved Validation subject.")
    for sub in actual_test_subs:
        if sub not in cohort_test_subs:
            raise RuntimeError(f"Subject {sub} in Test data is not an approved Test subject.")

    leakage_audit = {
        "passed": True,
        "cohort_split": {
            "train_subjects": sorted(list(cohort_train_subs)),
            "val_subjects": sorted(list(cohort_val_subs)),
            "test_subjects": sorted(list(cohort_test_subs)),
        },
        "active_subjects": {
            "train": sorted(list(actual_train_subs)),
            "validation": sorted(list(actual_val_subs)),
            "test": sorted(list(actual_test_subs)),
        },
    }

    split_counts = {
        "train": {
            "total_windows": len(ds_train),
            "normal_windows": int((ds_train["label"] == 0).sum()),
            "fall_windows": int((ds_train["label"] == 1).sum()),
            "subjects": sorted(list(actual_train_subs)),
        },
        "validation": {
            "total_windows": len(ds_val),
            "normal_windows": int((ds_val["label"] == 0).sum()),
            "fall_windows": int((ds_val["label"] == 1).sum()),
            "subjects": sorted(list(actual_val_subs)),
        },
        "test": {
            "total_windows": len(ds_test),
            "normal_windows": int((ds_test["label"] == 0).sum()),
            "fall_windows": int((ds_test["label"] == 1).sum()),
            "subjects": sorted(list(actual_test_subs)),
        },
    }

    return df, feature_cols, leakage_audit, split_counts


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """
    Computes comprehensive classification metrics with particular focus on the Fall class (Class 1).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Binary metrics specifically for Class 1 (Fall)
    fall_prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    fall_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    fall_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # Binary metrics for Class 0 (Normal)
    norm_prec = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    norm_rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    norm_f1 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC calculation
    roc_auc = None
    if y_prob is not None:
        try:
            # Check if both classes are present in y_true
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = None

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "fall_precision": round(fall_prec, 4),
        "fall_recall": round(fall_rec, 4),
        "fall_f1": round(fall_f1, 4),
        "normal_precision": round(norm_prec, 4),
        "normal_recall": round(norm_rec, 4),
        "normal_f1": round(norm_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def plot_confusion_matrix_figure(
    cm_dict: Dict[str, int],
    output_path: str,
    model_name: str = "Selected Model",
):
    """
    Renders and saves a high-quality annotated confusion matrix plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Color map
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Normal (0)", "Fall (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    thresh = matrix.max() / 2.0
    labels_grid = [
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                labels_grid[i][j],
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold",
            )

    ax.set_ylabel("True Ground Truth Label", fontsize=12, labelpad=10)
    ax.set_xlabel("Predicted Model Label", fontsize=12, labelpad=10)
    ax.set_title(f"Test Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot: {output_path}")


def plot_roc_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    model_name: str = "Selected Model",
):
    """
    Renders and saves the ROC Curve plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        ax.plot(
            fpr,
            tpr,
            color="#1f77b4",
            lw=2.5,
            label=f"{model_name} (AUC = {auc_val:.3f})",
        )
    else:
        ax.text(0.5, 0.5, "Single-class partition: ROC curve unavailable", ha="center", va="center")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    ax.set_title(f"Receiver Operating Characteristic (ROC) — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved ROC curve plot: {output_path}")
