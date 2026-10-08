"""
SisFall Model Utilities and Leakage-Safe Subject Splitting Module (Stage B4).

Provides:
- Deterministic subject-wise stratified train / validation / test partitioning (SA & SE cohorts)
- Zero-leakage verification (subject boundary & trial boundary checks)
- Candidate model pipeline builders (StandardScaler -> LogisticRegression, RandomForest, StandardScaler -> SVC)
- Comprehensive classification metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Cohort and class distribution profiling

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Feature list and metadata list
BASE_SIGNALS: List[str] = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
DERIVED_SIGNALS: List[str] = ["AccMag", "GyrMag"]
ALL_SIGNALS: List[str] = BASE_SIGNALS + DERIVED_SIGNALS
STAT_NAMES: List[str] = ["mean", "std", "min", "max", "ptp", "rms"]
FEATURE_NAMES: List[str] = [f"{sig}_{stat}" for sig in ALL_SIGNALS for stat in STAT_NAMES]

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

# Predefined transparent selection criteria (Section 13)
PRIMARY_SELECTION_METRIC = "fall_f1"      # F1 score for positive class (1 = FALL)
SECONDARY_SELECTION_METRIC = "roc_auc"    # Receiver Operating Characteristic Area Under Curve


def create_subject_split(
    all_subjects: List[str],
    random_state: int = 42,
    split_artifact_path: Optional[str] = r"reports/sisfall_subject_split.json",
) -> Dict[str, Any]:
    """
    Creates a deterministic subject-level split stratified by age cohort (SA young adults, SE elderly).

    Target distribution:
    - ~70% subjects -> TRAIN (16 SA, 11 SE = 27 subjects)
    - ~15% subjects -> VALIDATION (4 SA, 2 SE = 6 subjects)
    - ~15% subjects -> TEST (3 SA, 2 SE = 5 subjects)
    Total: 38 subjects.

    Parameters
    ----------
    all_subjects : List[str]
        List of all 38 unique subject IDs.
    random_state : int, default=42
        Random seed for reproducibility.
    split_artifact_path : Optional[str]
        Path to save the split JSON artifact.

    Returns
    -------
    Dict[str, Any]
        Dictionary with split metadata and subject lists.
    """
    sa_subs = sorted([s for s in all_subjects if s.startswith("SA")])
    se_subs = sorted([s for s in all_subjects if s.startswith("SE")])

    if len(sa_subs) != 23 or len(se_subs) != 15:
        raise ValueError(
            f"Expected 23 SA and 15 SE subjects, got {len(sa_subs)} SA and {len(se_subs)} SE."
        )

    # Deterministic shuffling with seed
    rng_sa = random.Random(random_state)
    shuffled_sa = list(sa_subs)
    rng_sa.shuffle(shuffled_sa)
    sa_train = sorted(shuffled_sa[:16])
    sa_val = sorted(shuffled_sa[16:20])
    sa_test = sorted(shuffled_sa[20:])

    rng_se = random.Random(random_state)
    shuffled_se = list(se_subs)
    rng_se.shuffle(shuffled_se)
    se_train = sorted(shuffled_se[:11])
    se_val = sorted(shuffled_se[11:13])
    se_test = sorted(shuffled_se[13:])

    train_subjects = sorted(sa_train + se_train)
    val_subjects = sorted(sa_val + se_val)
    test_subjects = sorted(sa_test + se_test)

    # Verification of disjoint sets
    s_tr, s_va, s_te = set(train_subjects), set(val_subjects), set(test_subjects)
    assert len(s_tr & s_va) == 0, "Overlap between train and validation subjects!"
    assert len(s_tr & s_te) == 0, "Overlap between train and test subjects!"
    assert len(s_va & s_te) == 0, "Overlap between validation and test subjects!"
    assert len(s_tr | s_va | s_te) == 38, "Subject count mismatch!"

    split_dict = {
        "dataset": "SisFall",
        "seed": random_state,
        "strategy": "subject-wise stratified by cohort (SA & SE)",
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
        "cohort_summary": {
            "train": {
                "total_subjects": len(train_subjects),
                "sa_subjects": sa_train,
                "se_subjects": se_train,
            },
            "validation": {
                "total_subjects": len(val_subjects),
                "sa_subjects": sa_val,
                "se_subjects": se_val,
            },
            "test": {
                "total_subjects": len(test_subjects),
                "sa_subjects": sa_test,
                "se_subjects": se_test,
            },
        },
    }

    if split_artifact_path:
        out_dir = os.path.dirname(os.path.abspath(split_artifact_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(split_artifact_path, "w") as f:
            json.dump(split_dict, f, indent=2)
        print(f"Subject split artifact saved to: {split_artifact_path}")

    return split_dict


def verify_leakage(
    df: pd.DataFrame,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
) -> Dict[str, Any]:
    """
    Validates that there is zero subject overlap and zero trial overlap across partitions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'subject_id' and 'source_filename' columns.
    train_subjects, val_subjects, test_subjects : List[str]
        Lists of subject IDs in each partition.

    Returns
    -------
    Dict[str, Any]
        Leakage audit report.
    """
    tr_subs = set(train_subjects)
    va_subs = set(val_subjects)
    te_subs = set(test_subjects)

    subj_overlap_tr_va = tr_subs & va_subs
    subj_overlap_tr_te = tr_subs & te_subs
    subj_overlap_va_te = va_subs & te_subs

    # Extract unique trials per partition using vectorized operations
    tr_df = df[df["subject_id"].isin(tr_subs)]
    va_df = df[df["subject_id"].isin(va_subs)]
    te_df = df[df["subject_id"].isin(te_subs)]

    # Raw filename sets
    tr_raw_trials = set(tr_df["source_filename"].unique())
    va_raw_trials = set(va_df["source_filename"].unique())
    te_raw_trials = set(te_df["source_filename"].unique())

    # Full unique trial identifier: subject_id/source_filename
    tr_full_trials = set((tr_df["subject_id"] + "/" + tr_df["source_filename"]).unique())
    va_full_trials = set((va_df["subject_id"] + "/" + va_df["source_filename"]).unique())
    te_full_trials = set((te_df["subject_id"] + "/" + te_df["source_filename"]).unique())

    trial_overlap_tr_va = tr_full_trials & va_full_trials
    trial_overlap_tr_te = tr_full_trials & te_full_trials
    trial_overlap_va_te = va_full_trials & te_full_trials

    is_leak_free = (
        len(subj_overlap_tr_va) == 0
        and len(subj_overlap_tr_te) == 0
        and len(subj_overlap_va_te) == 0
        and len(trial_overlap_tr_va) == 0
        and len(trial_overlap_tr_te) == 0
        and len(trial_overlap_va_te) == 0
    )

    return {
        "is_leak_free": is_leak_free,
        "subject_overlap_train_val": sorted(list(subj_overlap_tr_va)),
        "subject_overlap_train_test": sorted(list(subj_overlap_tr_te)),
        "subject_overlap_val_test": sorted(list(subj_overlap_va_te)),
        "train_trials_count": len(tr_full_trials),
        "val_trials_count": len(va_full_trials),
        "test_trials_count": len(te_full_trials),
        "total_trials_accounted": len(tr_full_trials) + len(va_full_trials) + len(te_full_trials),
        "trial_overlap_train_val_count": len(trial_overlap_tr_va),
        "trial_overlap_train_test_count": len(trial_overlap_tr_te),
        "trial_overlap_val_test_count": len(trial_overlap_va_te),
    }


def compute_cohort_distribution(
    df: pd.DataFrame,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
) -> Dict[str, Any]:
    """
    Computes class and cohort distribution across partitions.
    """
    dist: Dict[str, Any] = {}
    partitions = [
        ("train", train_subjects),
        ("validation", val_subjects),
        ("test", test_subjects),
    ]

    for part_name, subs in partitions:
        sub_df = df[df["subject_id"].isin(subs)]
        total = len(sub_df)
        adl = int((sub_df["label"] == 0).sum())
        fall = int((sub_df["label"] == 1).sum())
        sa_cnt = int(sub_df["subject_id"].apply(lambda s: s.startswith("SA")).sum())
        se_cnt = int(sub_df["subject_id"].apply(lambda s: s.startswith("SE")).sum())
        sa_subs = [s for s in subs if s.startswith("SA")]
        se_subs = [s for s in subs if s.startswith("SE")]

        dist[part_name] = {
            "total_rows": total,
            "adl_count": adl,
            "fall_count": fall,
            "fall_percentage": round((fall / total) * 100, 2) if total > 0 else 0.0,
            "total_subjects": len(subs),
            "sa_subject_count": len(sa_subs),
            "se_subject_count": len(se_subs),
            "sa_rows": sa_cnt,
            "se_rows": se_cnt,
        }

    return dist


def build_candidate_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Constructs the three baseline candidate pipelines:
    1. Logistic Regression: StandardScaler -> LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
    2. Random Forest: RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', n_jobs=-1)
    3. SVM: StandardScaler -> SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', max_iter=2000)

    The StandardScaler is bundled inside the pipeline to ensure leak-free transformation.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=2000,
                ),
            ),
        ]),
    }
    return models


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics.

    Target Class 1 = FALL (positive class).
    Class 0 = ADL (negative class).
    """
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    acc = float(accuracy_score(y_true_int, y_pred_int))
    f1_macro = float(f1_score(y_true_int, y_pred_int, average="macro", zero_division=0))
    fall_f1 = float(f1_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))
    fall_prec = float(precision_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))
    fall_rec = float(recall_score(y_true_int, y_pred_int, pos_label=1, zero_division=0))

    adl_f1 = float(f1_score(y_true_int, y_pred_int, pos_label=0, zero_division=0))
    adl_prec = float(precision_score(y_true_int, y_pred_int, pos_label=0, zero_division=0))
    adl_rec = float(recall_score(y_true_int, y_pred_int, pos_label=0, zero_division=0))

    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true_int, y_prob))
        except Exception:
            roc_auc = 0.0
    else:
        roc_auc = 0.0

    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": round(acc, 4),
        "fall_f1": round(fall_f1, 4),
        "fall_precision": round(fall_prec, 4),
        "fall_recall": round(fall_rec, 4),
        "adl_f1": round(adl_f1, 4),
        "adl_precision": round(adl_prec, 4),
        "adl_recall": round(adl_rec, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def load_sisfall_feature_dataset(
    parquet_path: str = r"data/processed/sisfall/sisfall_features.parquet",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads and rigorously audits the SisFall feature dataset.

    Verifies:
    - File existence
    - 312,649 rows
    - 15 metadata columns
    - 48 feature columns
    - 38 subjects (23 SA, 15 SE)
    - 206,579 ADL, 106,070 FALL
    - No NaN/nulls or infinite values
    - All window_id values unique

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Loaded DataFrame and ordered list of 48 feature column names.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"SisFall feature dataset not found at: {parquet_path}")

    print(f"Loading SisFall feature dataset from: {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # 1. Row count validation
    if len(df) != 312649:
        raise ValueError(f"Expected 312,649 rows, got {len(df)}")

    # 2. Metadata columns validation
    missing_meta = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing_meta:
        raise KeyError(f"Missing metadata columns in dataset: {missing_meta}")

    # 3. Feature columns validation
    missing_feats = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing feature columns in dataset: {missing_feats}")

    # 4. Total columns validation
    if len(df.columns) != 63:
        raise ValueError(f"Expected 63 total columns (48 feats + 15 meta), got {len(df.columns)}")

    # 5. Null and Inf validation
    if df[FEATURE_NAMES].isna().any().any():
        raise ValueError("Detected NaN values in feature matrix!")
    if np.isinf(df[FEATURE_NAMES].values).any():
        raise ValueError("Detected infinite values in feature matrix!")

    # 6. Window uniqueness
    if df["window_id"].nunique() != len(df):
        raise ValueError("Duplicate window_id values detected!")

    # 7. Subject count
    subjects = df["subject_id"].unique()
    if len(subjects) != 38:
        raise ValueError(f"Expected 38 subjects, got {len(subjects)}")

    sa_count = sum(1 for s in subjects if s.startswith("SA"))
    se_count = sum(1 for s in subjects if s.startswith("SE"))
    if sa_count != 23 or se_count != 15:
        raise ValueError(f"Expected 23 SA and 15 SE subjects, got {sa_count} SA and {se_count} SE")

    # 8. Class distribution
    label_counts = df["label"].value_counts().to_dict()
    if label_counts.get(0, 0) != 206579 or label_counts.get(1, 0) != 106070:
        raise ValueError(
            f"Class count mismatch! Expected ADL=206579, FALL=106070, got {label_counts}"
        )

    print(f"Dataset audit PASSED: {len(df)} rows, 48 features, 38 subjects, 0 NaN/Inf values.")
    return df, FEATURE_NAMES


def plot_confusion_matrix_figure(
    cm_dict: Dict[str, int],
    output_path: str,
    model_name: str = "Random Forest",
):
    """
    Renders and saves a publication-quality annotated confusion matrix plot for SisFall.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["ADL (0)", "FALL (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    thresh = matrix.max() / 2.0
    labels_grid = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
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
    ax.set_title(f"Test Confusion Matrix — SisFall ({model_name})", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved SisFall confusion matrix plot: {output_path}")


def plot_roc_curve_figure(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
):
    """
    Renders and saves a multi-model ROC Curve comparison plot for held-out SisFall subjects.

    Parameters
    ----------
    roc_data : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Mapping of model_name -> (y_true, y_prob)
    output_path : str
        Output file path for PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    colors = {
        "Logistic Regression": "#2ca02c",      # green
        "Random Forest": "#1f77b4",            # blue
        "Support Vector Machine": "#d62728",   # red
    }

    for model_name, (y_true, y_prob) in roc_data.items():
        if y_prob is not None and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            c = colors.get(model_name, "#333333")
            ax.plot(
                fpr,
                tpr,
                color=c,
                lw=2.2,
                label=f"{model_name} (AUC = {auc_val:.4f})",
            )

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.5000)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Held-Out Test ROC Curves — SisFall Baseline Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved SisFall ROC curves plot: {output_path}")
