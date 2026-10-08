"""
SisFall Model Training, Validation, and Serialization Module (Stage B4).

Trains three baseline classifiers:
1. Logistic Regression: StandardScaler -> LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
2. Random Forest: RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
3. SVM: StandardScaler -> SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42, max_iter=2000)

Protocol:
- Subject-wise stratified split (27 Train, 6 Val, 5 Test) with 0 subject / trial overlap.
- Preprocessing (StandardScaler) strictly fitted on TRAIN data within Pipeline.
- Model selection decided strictly on validation FALL F1 (with validation ROC-AUC tie-breaker).
- Test set strictly held out until after selection; evaluated once for transparent comparison.
- Models serialized to models/sisfall/*.joblib.
- Metadata and reports saved to models/sisfall/ and reports/.

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sisfall.sisfall_model_utils import (
    FEATURE_NAMES,
    METADATA_COLUMNS,
    PRIMARY_SELECTION_METRIC,
    SECONDARY_SELECTION_METRIC,
    build_candidate_models,
    compute_cohort_distribution,
    compute_metrics,
    create_subject_split,
    load_sisfall_feature_dataset,
    plot_confusion_matrix_figure,
    plot_roc_curve_figure,
    verify_leakage,
)

RANDOM_SEED = 42
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "sisfall")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def run_sisfall_training_pipeline() -> Dict[str, Any]:
    """
    Executes the full Stage B4 model training, validation, selection, evaluation,
    serialization, and verification pipeline.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 78)
    print("  SISFALL STAGE B4: LEAKAGE-SAFE SUBJECT-WISE MODEL TRAINING")
    print("=" * 78)

    # 1. Load and audit the B3 feature dataset
    df, feature_cols = load_sisfall_feature_dataset(
        parquet_path=os.path.join(PROJECT_ROOT, "data", "processed", "sisfall", "sisfall_features.parquet")
    )
    all_subjects = sorted(df["subject_id"].unique().tolist())

    # 2. Deterministic Subject-Level Split (Stratified by Cohort)
    print("\n--- [Step 1] Creating Deterministic Subject-Level Split ---")
    split_artifact_path = os.path.join(REPORTS_DIR, "sisfall_subject_split.json")
    split_info = create_subject_split(
        all_subjects=all_subjects,
        random_state=RANDOM_SEED,
        split_artifact_path=split_artifact_path,
    )
    train_subs = split_info["train_subjects"]
    val_subs = split_info["validation_subjects"]
    test_subs = split_info["test_subjects"]

    print(f"  Train Subjects ({len(train_subs)}): {train_subs}")
    print(f"  Validation Subjects ({len(val_subs)}): {val_subs}")
    print(f"  Test Subjects ({len(test_subs)}): {test_subs}")

    # 3. Rigorous Data Leakage Verification
    print("\n--- [Step 2] Auditing Data Leakage ---")
    leak_report = verify_leakage(df, train_subs, val_subs, test_subs)
    if not leak_report["is_leak_free"]:
        raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: {leak_report}")
    print("  Data Leakage Audit: PASSED (Zero subject overlap, zero trial overlap).")
    print(f"    Train Trials: {leak_report['train_trials_count']}")
    print(f"    Val Trials:   {leak_report['val_trials_count']}")
    print(f"    Test Trials:  {leak_report['test_trials_count']}")
    print(f"    Total Trials Accounted: {leak_report['total_trials_accounted']} / 4505")

    # 4. Cohort & Class Distribution
    print("\n--- [Step 3] Cohort & Class Distribution ---")
    cohort_dist = compute_cohort_distribution(df, train_subs, val_subs, test_subs)
    for p_name, p_info in cohort_dist.items():
        print(
            f"  {p_name.upper():10s}: {p_info['total_rows']:,} rows | "
            f"ADL: {p_info['adl_count']:,} | FALL: {p_info['fall_count']:,} ({p_info['fall_percentage']}%) | "
            f"Subs: {p_info['total_subjects']} (SA: {p_info['sa_subject_count']}, SE: {p_info['se_subject_count']})"
        )

    # 5. Partition Data Matrices
    print("\n--- [Step 4] Partitioning Data Matrices ---")
    train_df = df[df["subject_id"].isin(train_subs)]
    val_df = df[df["subject_id"].isin(val_subs)]
    test_df = df[df["subject_id"].isin(test_subs)]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test: {y_test.shape}")

    # 6. Build Candidate Pipelines
    candidates = build_candidate_models(random_state=RANDOM_SEED)

    # 7. Model Training & Validation Evaluation
    print("\n--- [Step 5] Training Baseline Models on Train Partition ---")
    val_metrics: Dict[str, Dict[str, Any]] = {}
    train_times: Dict[str, float] = {}

    for name, pipeline in candidates.items():
        print(f"\n  Fitting {name}...")
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        t_fit = time.time() - t0
        train_times[name] = round(t_fit, 2)
        print(f"    Completed fit in {t_fit:.2f}s")

        # Evaluate on Validation Partition
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
        metrics = compute_metrics(y_true=y_val, y_pred=y_val_pred, y_prob=y_val_prob)
        metrics["training_time_seconds"] = round(t_fit, 2)
        val_metrics[name] = metrics

        print(
            f"    Validation Results -> Acc: {metrics['accuracy']:.4f} | "
            f"Fall F1: {metrics['fall_f1']:.4f} | Fall Prec: {metrics['fall_precision']:.4f} | "
            f"Fall Rec: {metrics['fall_recall']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
        )
        cm = metrics["confusion_matrix"]
        print(f"    Confusion Matrix: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

    # 8. Model Selection (Strictly on Validation Set)
    print("\n--- [Step 6] Model Selection via Validation Performance ---")
    def selection_key(item):
        m = item[1]
        f1 = m.get(PRIMARY_SELECTION_METRIC, 0.0) or 0.0
        auc = m.get(SECONDARY_SELECTION_METRIC, 0.0) or 0.0
        return (f1, auc)

    selected_model_name = max(val_metrics.items(), key=selection_key)[0]
    print(f"  Primary selection criterion:   Validation {PRIMARY_SELECTION_METRIC}")
    print(f"  Secondary selection criterion: Validation {SECONDARY_SELECTION_METRIC}")
    print(f"  Model selected_by_validation_rule: '{selected_model_name}'")
    print(f"    Validation {PRIMARY_SELECTION_METRIC}: {val_metrics[selected_model_name][PRIMARY_SELECTION_METRIC]:.4f}")
    print(f"    Validation {SECONDARY_SELECTION_METRIC}: {val_metrics[selected_model_name][SECONDARY_SELECTION_METRIC]:.4f}")

    # 9. Final Test Evaluation (Transparent Held-Out Comparison)
    print("\n--- [Step 7] Evaluating All Baseline Models on Held-Out Test Split ---")
    print("  NOTICE: Test set is strictly held out and was NOT used for fitting or model selection.")
    test_metrics: Dict[str, Dict[str, Any]] = {}
    roc_plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, pipeline in candidates.items():
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        t_metrics = compute_metrics(y_true=y_test, y_pred=y_test_pred, y_prob=y_test_prob)
        test_metrics[name] = t_metrics
        roc_plot_data[name] = (y_test, y_test_prob)

        is_selected = " [SELECTED BY VALIDATION RULE]" if name == selected_model_name else ""
        print(f"\n  Test Results for '{name}'{is_selected}:")
        print(
            f"    Acc: {t_metrics['accuracy']:.4f} | Fall F1: {t_metrics['fall_f1']:.4f} | "
            f"Fall Prec: {t_metrics['fall_precision']:.4f} | Fall Rec: {t_metrics['fall_recall']:.4f} | "
            f"ROC-AUC: {t_metrics['roc_auc']:.4f}"
        )
        cm = t_metrics["confusion_matrix"]
        print(f"    Confusion Matrix: TN={cm['tn']:,}, FP={cm['fp']:,}, FN={cm['fn']:,}, TP={cm['tp']:,}")

    # 10. Generate Visualizations for Held-Out Test Set
    print("\n--- [Step 8] Generating Visualizations ---")
    cm_path = os.path.join(REPORTS_DIR, "sisfall_confusion_matrix.png")
    plot_confusion_matrix_figure(
        cm_dict=test_metrics[selected_model_name]["confusion_matrix"],
        output_path=cm_path,
        model_name=selected_model_name,
    )

    roc_path = os.path.join(REPORTS_DIR, "sisfall_roc_curve.png")
    plot_roc_curve_figure(roc_data=roc_plot_data, output_path=roc_path)

    # 11. Serialize Models and Metadata
    print("\n--- [Step 9] Serializing Model Artifacts ---")
    filename_map = {
        "Logistic Regression": "sisfall_logistic_regression.joblib",
        "Random Forest": "sisfall_random_forest.joblib",
        "Support Vector Machine": "sisfall_svm.joblib",
    }

    serialized_paths = {}
    for name, pipeline in candidates.items():
        fn = filename_map[name]
        art_path = os.path.join(MODELS_DIR, fn)
        joblib.dump(pipeline, art_path)
        serialized_paths[name] = art_path
        size_kb = os.path.getsize(art_path) / 1024
        print(f"  Saved model: {art_path} ({size_kb:.1f} KB)")

    # Save Model Metadata
    metadata_path = os.path.join(MODELS_DIR, "sisfall_model_metadata.json")
    metadata = {
        "dataset": "SisFall",
        "stage": "B4",
        "status": "READY_FOR_SISFALL_STAGE_B5",
        "model_selected_by_validation_rule": selected_model_name,
        "selection_rule": {
            "primary": PRIMARY_SELECTION_METRIC,
            "secondary": SECONDARY_SELECTION_METRIC,
            "description": "Validation-set Fall F1 score with validation ROC-AUC tie-breaker",
        },
        "random_seed": RANDOM_SEED,
        "system_info": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
            "platform": platform.platform(),
        },
        "feature_order": feature_cols,
        "feature_count": len(feature_cols),
        "split_summary": {
            "train_subjects": train_subs,
            "validation_subjects": val_subs,
            "test_subjects": test_subs,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "models": {
            name: {
                "artifact_file": filename_map[name],
                "pipeline_steps": [step[0] for step in candidates[name].steps],
                "training_time_seconds": train_times[name],
                "validation_metrics": val_metrics[name],
                "test_metrics": test_metrics[name],
            }
            for name in candidates
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved SisFall model metadata: {metadata_path}")

    # 12. Save Comparison Reports
    print("\n--- [Step 10] Writing Model Comparison Reports ---")
    comp_path = os.path.join(REPORTS_DIR, "sisfall_model_comparison.json")
    comp_data = {
        "dataset": "SisFall",
        "selection_criterion": {
            "primary": PRIMARY_SELECTION_METRIC,
            "secondary": SECONDARY_SELECTION_METRIC,
            "description": "Validation FALL F1-score with validation ROC-AUC tie-breaker",
        },
        "selected_by_validation_rule": selected_model_name,
        "validation_comparison": val_metrics,
        "test_comparison": test_metrics,
    }
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"  Saved model comparison JSON: {comp_path}")

    metrics_path = os.path.join(REPORTS_DIR, "sisfall_stageB4_metrics.json")
    all_metrics_payload = {
        "dataset": "SisFall",
        "stage": "B4",
        "split_distribution": cohort_dist,
        "validation_results": val_metrics,
        "test_results": test_metrics,
        "selected_model": selected_model_name,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_payload, f, indent=2)
    print(f"  Saved Stage B4 metrics JSON: {metrics_path}")

    # 13. Reload Verification
    print("\n--- [Step 11] Verifying Model Deserialization & Prediction Identity ---")
    for name, art_path in serialized_paths.items():
        reloaded = joblib.load(art_path)
        # Predict on first 100 test samples
        orig_preds = candidates[name].predict(X_test[:100])
        reload_preds = reloaded.predict(X_test[:100])
        if not np.array_equal(orig_preds, reload_preds):
            raise AssertionError(f"Model reload prediction mismatch for {name}!")

        if hasattr(candidates[name], "predict_proba"):
            orig_probs = candidates[name].predict_proba(X_test[:100])
            reload_probs = reloaded.predict_proba(X_test[:100])
            if not np.allclose(orig_probs, reload_probs, atol=1e-6):
                raise AssertionError(f"Model reload probability mismatch for {name}!")

        print(f"  Reload verification PASSED for: {name}")

    print("\n" + "=" * 78)
    print("  SISFALL STAGE B4 COMPLETE — ALL CHECKS PASSED")
    print("  Final Status: READY_FOR_SISFALL_STAGE_B5")
    print("=" * 78)

    return {
        "status": "READY_FOR_SISFALL_STAGE_B5",
        "selected_model": selected_model_name,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cohort_distribution": cohort_dist,
    }


if __name__ == "__main__":
    run_sisfall_training_pipeline()
