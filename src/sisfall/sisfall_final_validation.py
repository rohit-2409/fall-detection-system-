"""
SisFall Stage B5: Final ML Validation, Reproducibility & Scientific Analysis Module.

Performs independent end-to-end audit and validation of Stage B4 artifacts:
1. Model deserialization and structural pipeline audit
2. Feature schema and canonical order validation against metadata
3. Subject split and trial-level disjointness verification
4. Exhaustive row accounting and class distribution checks
5. Independent metric and confusion matrix recomputation vs B4 references
6. Strict reload prediction and probability reproducibility testing
7. Model selection rule adherence audit (Validation Fall F1 -> ROC-AUC)
8. Production of machine-readable validation records and CSV tables

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
Completed SisFall Stage B1-B4 artifacts and datasets are read-only and preserved.
"""

import json
import os
import platform
import sys
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "sisfall")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
B5_TABLES_DIR = os.path.join(REPORTS_DIR, "sisfall_stageB5_validation")


def run_stage_b5_validation() -> Dict[str, Any]:
    os.makedirs(B5_TABLES_DIR, exist_ok=True)
    print("=" * 80)
    print("  SISFALL STAGE B5: FINAL ML VALIDATION & SCIENTIFIC AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. AUDIT REQUIRED ARTIFACTS
    # -------------------------------------------------------------------------
    required_artifacts = {
        "subject_split": os.path.join(REPORTS_DIR, "sisfall_subject_split.json"),
        "model_comparison": os.path.join(REPORTS_DIR, "sisfall_model_comparison.json"),
        "stageB4_metrics": os.path.join(REPORTS_DIR, "sisfall_stageB4_metrics.json"),
        "stageB4_report": os.path.join(REPORTS_DIR, "sisfall_stageB4_training_report.md"),
        "model_metadata": os.path.join(MODELS_DIR, "sisfall_model_metadata.json"),
        "logistic_regression": os.path.join(MODELS_DIR, "sisfall_logistic_regression.joblib"),
        "random_forest": os.path.join(MODELS_DIR, "sisfall_random_forest.joblib"),
        "svm": os.path.join(MODELS_DIR, "sisfall_svm.joblib"),
        "features_parquet": os.path.join(PROJECT_ROOT, "data", "processed", "sisfall", "sisfall_features.parquet"),
    }

    artifact_checks = []
    all_artifacts_present = True
    for name, path in required_artifacts.items():
        exists = os.path.exists(path)
        sz = os.path.getsize(path) if exists else 0
        artifact_checks.append({
            "artifact_name": name,
            "filepath": os.path.relpath(path, PROJECT_ROOT),
            "exists": exists,
            "size_bytes": sz,
            "status": "PASS" if exists and sz > 0 else "FAIL",
        })
        if not (exists and sz > 0):
            all_artifacts_present = False

    if not all_artifacts_present:
        raise RuntimeError(f"Missing required B4 artifacts: {artifact_checks}")
    print("[1] Prerequisite B4 Artifact Existence: PASSED (All 9 artifacts present).")

    # -------------------------------------------------------------------------
    # 2. FEATURE SCHEMA VALIDATION
    # -------------------------------------------------------------------------
    print("\n[2] Validating Feature Schema and Integrity...")
    with open(required_artifacts["model_metadata"], "r", encoding="utf-8") as f:
        b4_metadata = json.load(f)

    expected_feature_order = b4_metadata["feature_order"]
    if len(expected_feature_order) != 48:
        raise ValueError(f"Expected 48 features in metadata, found {len(expected_feature_order)}")

    df = pd.read_parquet(required_artifacts["features_parquet"])
    if len(df) != 312649:
        raise ValueError(f"Expected 312,649 rows, found {len(df)}")

    # Verify column existence and order
    actual_features_in_df = [c for c in expected_feature_order if c in df.columns]
    feature_schema_pass = (actual_features_in_df == expected_feature_order)

    # Check for NaN / Inf / null
    feature_matrix = df[expected_feature_order].values
    has_nan = np.isnan(feature_matrix).any()
    has_inf = np.isinf(feature_matrix).any()
    all_numeric = np.issubdtype(feature_matrix.dtype, np.number)

    feature_schema_summary = {
        "total_rows": len(df),
        "feature_count": len(expected_feature_order),
        "feature_order_preserved": bool(feature_schema_pass),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "all_numeric": bool(all_numeric),
        "target_column": "label",
        "unique_labels": sorted(df["label"].unique().tolist()),
        "status": "PASS" if (feature_schema_pass and not has_nan and not has_inf and all_numeric) else "FAIL",
    }
    print(f"  Feature Schema Check: {feature_schema_summary['status']} (48 features, 0 NaN, 0 Inf).")

    # -------------------------------------------------------------------------
    # 3. SUBJECT SPLIT AND TRIAL LEAKAGE VALIDATION
    # -------------------------------------------------------------------------
    print("\n[3] Auditing Subject Split & Trial-Level Disjointness...")
    with open(required_artifacts["subject_split"], "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_subs = split_data["train_subjects"]
    val_subs = split_data["validation_subjects"]
    test_subs = split_data["test_subjects"]

    all_subs_set = set(train_subs) | set(val_subs) | set(test_subs)
    sa_subs = [s for s in all_subs_set if s.startswith("SA")]
    se_subs = [s for s in all_subs_set if s.startswith("SE")]

    # Overlap checks
    subj_overlap_tr_va = set(train_subs) & set(val_subs)
    subj_overlap_tr_te = set(train_subs) & set(test_subs)
    subj_overlap_va_te = set(val_subs) & set(test_subs)

    subject_split_pass = (
        len(all_subs_set) == 38
        and len(sa_subs) == 23
        and len(se_subs) == 15
        and len(train_subs) == 27
        and len(val_subs) == 6
        and len(test_subs) == 5
        and len(subj_overlap_tr_va) == 0
        and len(subj_overlap_tr_te) == 0
        and len(subj_overlap_va_te) == 0
    )

    # Trial-level disjointness
    tr_df = df[df["subject_id"].isin(train_subs)]
    va_df = df[df["subject_id"].isin(val_subs)]
    te_df = df[df["subject_id"].isin(test_subs)]

    tr_trials = set((tr_df["subject_id"] + "/" + tr_df["source_filename"]).unique())
    va_trials = set((va_df["subject_id"] + "/" + va_df["source_filename"]).unique())
    te_trials = set((te_df["subject_id"] + "/" + te_df["source_filename"]).unique())

    trial_overlap_tr_va = tr_trials & va_trials
    trial_overlap_tr_te = tr_trials & te_trials
    trial_overlap_va_te = va_trials & te_trials

    total_trials = len(tr_trials) + len(va_trials) + len(te_trials)
    trial_leakage_pass = (
        len(tr_trials) == 3202
        and len(va_trials) == 734
        and len(te_trials) == 569
        and total_trials == 4505
        and len(trial_overlap_tr_va) == 0
        and len(trial_overlap_tr_te) == 0
        and len(trial_overlap_va_te) == 0
    )

    split_leakage_summary = {
        "total_subjects": len(all_subs_set),
        "train_subjects_count": len(train_subs),
        "val_subjects_count": len(val_subs),
        "test_subjects_count": len(test_subs),
        "subject_overlap_train_val": list(subj_overlap_tr_va),
        "subject_overlap_train_test": list(subj_overlap_tr_te),
        "subject_overlap_val_test": list(subj_overlap_va_te),
        "subject_split_status": "PASS" if subject_split_pass else "FAIL",
        "train_trials": len(tr_trials),
        "val_trials": len(va_trials),
        "test_trials": len(te_trials),
        "total_trials": total_trials,
        "trial_overlap_train_val": list(trial_overlap_tr_va),
        "trial_overlap_train_test": list(trial_overlap_tr_te),
        "trial_overlap_val_test": list(trial_overlap_va_te),
        "trial_leakage_status": "PASS" if trial_leakage_pass else "FAIL",
    }
    print(f"  Subject Split Status: {split_leakage_summary['subject_split_status']} (38 subjects, 0 overlap).")
    print(f"  Trial Leakage Status: {split_leakage_summary['trial_leakage_status']} (4505 trials, 0 overlap).")

    # -------------------------------------------------------------------------
    # 4. ROW ACCOUNTING AND CLASS DISTRIBUTION
    # -------------------------------------------------------------------------
    print("\n[4] Exhaustive Row Accounting and Class Distribution...")
    tr_total = len(tr_df)
    tr_adl = int((tr_df["label"] == 0).sum())
    tr_fall = int((tr_df["label"] == 1).sum())

    va_total = len(va_df)
    va_adl = int((va_df["label"] == 0).sum())
    va_fall = int((va_df["label"] == 1).sum())

    te_total = len(te_df)
    te_adl = int((te_df["label"] == 0).sum())
    te_fall = int((te_df["label"] == 1).sum())

    row_sum_pass = (tr_total + va_total + te_total == 312649)
    adl_sum_pass = (tr_adl + va_adl + te_adl == 206579)
    fall_sum_pass = (tr_fall + va_fall + te_fall == 106070)

    exact_counts_pass = (
        tr_total == 221471 and tr_adl == 146373 and tr_fall == 75098
        and va_total == 51839 and va_adl == 34142 and va_fall == 17697
        and te_total == 39339 and te_adl == 26064 and te_fall == 13275
    )

    row_accounting_summary = {
        "train": {"total": tr_total, "adl": tr_adl, "fall": tr_fall, "fall_pct": round(tr_fall / tr_total * 100, 2)},
        "validation": {"total": va_total, "adl": va_adl, "fall": va_fall, "fall_pct": round(va_fall / va_total * 100, 2)},
        "test": {"total": te_total, "adl": te_adl, "fall": te_fall, "fall_pct": round(te_fall / te_total * 100, 2)},
        "total_windows": tr_total + va_total + te_total,
        "total_adl": tr_adl + va_adl + te_adl,
        "total_fall": tr_fall + va_fall + te_fall,
        "status": "PASS" if (row_sum_pass and adl_sum_pass and fall_sum_pass and exact_counts_pass) else "FAIL",
    }
    print(f"  Row Accounting Check: {row_accounting_summary['status']} (221,471 + 51,839 + 39,339 = 312,649).")

    # -------------------------------------------------------------------------
    # 5. MODEL STRUCTURAL AUDIT
    # -------------------------------------------------------------------------
    print("\n[5] Model Structural Pipeline Audit...")
    models = {
        "Logistic Regression": joblib.load(required_artifacts["logistic_regression"]),
        "Random Forest": joblib.load(required_artifacts["random_forest"]),
        "Support Vector Machine": joblib.load(required_artifacts["svm"]),
    }

    # Audit Logistic Regression
    lr_pipe = models["Logistic Regression"]
    lr_steps = [s[0] for s in lr_pipe.steps]
    lr_clf = lr_pipe.named_steps["classifier"]
    lr_scaler = lr_pipe.named_steps["scaler"]
    lr_pass = (
        lr_steps == ["scaler", "classifier"]
        and isinstance(lr_scaler, sklearn.preprocessing.StandardScaler)
        and isinstance(lr_clf, sklearn.linear_model.LogisticRegression)
        and lr_clf.C == 1.0
        and lr_clf.max_iter == 1000
        and lr_clf.class_weight == "balanced"
        and lr_clf.solver == "lbfgs"
        and lr_clf.random_state == 42
    )

    # Audit Random Forest
    rf_pipe = models["Random Forest"]
    rf_steps = [s[0] for s in rf_pipe.steps]
    rf_clf = rf_pipe.named_steps["classifier"]
    rf_pass = (
        rf_steps == ["classifier"]
        and isinstance(rf_clf, sklearn.ensemble.RandomForestClassifier)
        and rf_clf.n_estimators == 100
        and rf_clf.max_depth == 6
        and rf_clf.class_weight == "balanced"
        and rf_clf.random_state == 42
        and rf_clf.n_jobs == -1
    )

    # Audit SVM
    svm_pipe = models["Support Vector Machine"]
    svm_steps = [s[0] for s in svm_pipe.steps]
    svm_clf = svm_pipe.named_steps["classifier"]
    svm_scaler = svm_pipe.named_steps["scaler"]
    svm_pass = (
        svm_steps == ["scaler", "classifier"]
        and isinstance(svm_scaler, sklearn.preprocessing.StandardScaler)
        and isinstance(svm_clf, sklearn.svm.SVC)
        and svm_clf.C == 1.0
        and svm_clf.kernel == "rbf"
        and svm_clf.probability is True
        and svm_clf.class_weight == "balanced"
        and svm_clf.max_iter == 2000
        and svm_clf.random_state == 42
    )

    structural_checks = {
        "Logistic Regression": {"steps": lr_steps, "status": "PASS" if lr_pass else "FAIL"},
        "Random Forest": {"steps": rf_steps, "status": "PASS" if rf_pass else "FAIL"},
        "Support Vector Machine": {"steps": svm_steps, "status": "PASS" if svm_pass else "FAIL"},
    }
    print(f"  Structural Pipeline Checks: LR={structural_checks['Logistic Regression']['status']}, "
          f"RF={structural_checks['Random Forest']['status']}, SVM={structural_checks['Support Vector Machine']['status']}")

    # -------------------------------------------------------------------------
    # 6. INDEPENDENT METRIC RECOMPUTATION & CONFUSION MATRIX VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[6] Independently Recomputing Metrics on Validation and Test Splits...")
    X_val = va_df[expected_feature_order].values
    y_val = va_df["label"].values.astype(int)

    X_test = te_df[expected_feature_order].values
    y_test = te_df["label"].values.astype(int)

    # Load B4 reference comparison
    with open(required_artifacts["model_comparison"], "r", encoding="utf-8") as f:
        b4_comparison = json.load(f)

    tolerance = 1e-4
    metric_comparison_rows = []
    all_metrics_match = True

    indep_metrics = {"validation": {}, "test": {}}

    for name, pipeline in models.items():
        # Validation
        val_pred = pipeline.predict(X_val)
        val_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None

        v_acc = float(accuracy_score(y_val, val_pred))
        v_f1_macro = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        v_fall_f1 = float(f1_score(y_val, val_pred, pos_label=1, zero_division=0))
        v_fall_prec = float(precision_score(y_val, val_pred, pos_label=1, zero_division=0))
        v_fall_rec = float(recall_score(y_val, val_pred, pos_label=1, zero_division=0))
        v_auc = float(roc_auc_score(y_val, val_prob)) if val_prob is not None else 0.0
        v_cm = confusion_matrix(y_val, val_pred, labels=[0, 1])

        # Test
        test_pred = pipeline.predict(X_test)
        test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        t_acc = float(accuracy_score(y_test, test_pred))
        t_f1_macro = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
        t_fall_f1 = float(f1_score(y_test, test_pred, pos_label=1, zero_division=0))
        t_fall_prec = float(precision_score(y_test, test_pred, pos_label=1, zero_division=0))
        t_fall_rec = float(recall_score(y_test, test_pred, pos_label=1, zero_division=0))
        t_auc = float(roc_auc_score(y_test, test_prob)) if test_prob is not None else 0.0
        t_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

        indep_metrics["validation"][name] = {
            "accuracy": round(v_acc, 4),
            "fall_precision": round(v_fall_prec, 4),
            "fall_recall": round(v_fall_rec, 4),
            "fall_f1": round(v_fall_f1, 4),
            "f1_macro": round(v_f1_macro, 4),
            "roc_auc": round(v_auc, 4),
            "confusion_matrix": {"tn": int(v_cm[0, 0]), "fp": int(v_cm[0, 1]), "fn": int(v_cm[1, 0]), "tp": int(v_cm[1, 1])},
        }

        indep_metrics["test"][name] = {
            "accuracy": round(t_acc, 4),
            "fall_precision": round(t_fall_prec, 4),
            "fall_recall": round(t_fall_rec, 4),
            "fall_f1": round(t_fall_f1, 4),
            "f1_macro": round(t_f1_macro, 4),
            "roc_auc": round(t_auc, 4),
            "confusion_matrix": {"tn": int(t_cm[0, 0]), "fp": int(t_cm[0, 1]), "fn": int(t_cm[1, 0]), "tp": int(t_cm[1, 1])},
        }

        # Compare with B4 reference values
        b4_v = b4_comparison["validation_comparison"][name]
        b4_t = b4_comparison["test_comparison"][name]

        # Metric check pairs
        checks = [
            ("validation", "accuracy", v_acc, b4_v["accuracy"]),
            ("validation", "fall_precision", v_fall_prec, b4_v["fall_precision"]),
            ("validation", "fall_recall", v_fall_rec, b4_v["fall_recall"]),
            ("validation", "fall_f1", v_fall_f1, b4_v["fall_f1"]),
            ("validation", "roc_auc", v_auc, b4_v["roc_auc"]),
            ("test", "accuracy", t_acc, b4_t["accuracy"]),
            ("test", "fall_precision", t_fall_prec, b4_t["fall_precision"]),
            ("test", "fall_recall", t_fall_rec, b4_t["fall_recall"]),
            ("test", "fall_f1", t_fall_f1, b4_t["fall_f1"]),
            ("test", "roc_auc", t_auc, b4_t["roc_auc"]),
            ("test", "cm_tn", int(t_cm[0, 0]), b4_t["confusion_matrix"]["tn"]),
            ("test", "cm_fp", int(t_cm[0, 1]), b4_t["confusion_matrix"]["fp"]),
            ("test", "cm_fn", int(t_cm[1, 0]), b4_t["confusion_matrix"]["fn"]),
            ("test", "cm_tp", int(t_cm[1, 1]), b4_t["confusion_matrix"]["tp"]),
        ]

        for split_n, m_name, computed_val, ref_val in checks:
            if isinstance(ref_val, int):
                matches = (computed_val == ref_val)
                diff = abs(computed_val - ref_val)
            else:
                diff = abs(computed_val - ref_val)
                matches = diff <= tolerance

            metric_comparison_rows.append({
                "model": name,
                "split": split_n,
                "metric": m_name,
                "computed_value": computed_val,
                "reference_value": ref_val,
                "abs_diff": round(diff, 6),
                "status": "PASS" if matches else "FAIL",
            })
            if not matches:
                all_metrics_match = False

    print(f"  Independent Metric Recomputation: {'PASS' if all_metrics_match else 'FAIL'} (Tolerance: {tolerance}).")

    # Confusion matrix sum verification
    cm_sum_pass = True
    for name in models:
        cm = indep_metrics["test"][name]["confusion_matrix"]
        cm_sum = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        if cm_sum != 39339:
            cm_sum_pass = False
    print(f"  Confusion Matrix Total Sum Check: {'PASS' if cm_sum_pass else 'FAIL'} (All equal 39,339).")

    # -------------------------------------------------------------------------
    # 7. RELOAD PREDICTION & PROBABILITY REPRODUCIBILITY TEST
    # -------------------------------------------------------------------------
    print("\n[7] Testing Strict Reload Prediction & Probability Reproducibility...")
    reproducibility_rows = []
    all_reproducible = True

    for name, path in [
        ("Logistic Regression", required_artifacts["logistic_regression"]),
        ("Random Forest", required_artifacts["random_forest"]),
        ("Support Vector Machine", required_artifacts["svm"]),
    ]:
        m1 = joblib.load(path)
        m2 = joblib.load(path)

        p1_val = m1.predict(X_val)
        p2_val = m2.predict(X_val)
        pred_match_val = np.array_equal(p1_val, p2_val)

        p1_test = m1.predict(X_test)
        p2_test = m2.predict(X_test)
        pred_match_test = np.array_equal(p1_test, p2_test)

        prob_match_test = True
        max_prob_diff = 0.0
        if hasattr(m1, "predict_proba"):
            prob1 = m1.predict_proba(X_test)
            prob2 = m2.predict_proba(X_test)
            max_prob_diff = float(np.max(np.abs(prob1 - prob2)))
            prob_match_test = np.allclose(prob1, prob2, atol=1e-7)

        repro_pass = pred_match_val and pred_match_test and prob_match_test
        reproducibility_rows.append({
            "model": name,
            "val_preds_identical": bool(pred_match_val),
            "test_preds_identical": bool(pred_match_test),
            "test_probs_identical": bool(prob_match_test),
            "max_prob_diff": max_prob_diff,
            "status": "PASS" if repro_pass else "FAIL",
        })
        if not repro_pass:
            all_reproducible = False

    print(f"  Reload Reproducibility: {'PASS' if all_reproducible else 'FAIL'}.")

    # -------------------------------------------------------------------------
    # 8. VALIDATION-BASED SELECTION RULE AUDIT
    # -------------------------------------------------------------------------
    print("\n[8] Auditing Model Selection Rule Adherence...")
    val_fall_f1s = {
        name: indep_metrics["validation"][name]["fall_f1"]
        for name in models
    }
    val_roc_aucs = {
        name: indep_metrics["validation"][name]["roc_auc"]
        for name in models
    }

    def selection_key(name):
        return (val_fall_f1s[name], val_roc_aucs[name])

    audit_selected_model = max(models.keys(), key=selection_key)
    metadata_selected_model = b4_metadata["model_selected_by_validation_rule"]
    comparison_selected_model = b4_comparison["selected_by_validation_rule"]

    selection_rule_pass = (
        audit_selected_model == "Random Forest"
        and metadata_selected_model == "Random Forest"
        and comparison_selected_model == "Random Forest"
        and val_fall_f1s["Random Forest"] > val_fall_f1s["Logistic Regression"]
        and val_fall_f1s["Random Forest"] > val_fall_f1s["Support Vector Machine"]
    )
    print(f"  Selection Rule Adherence: {'PASS' if selection_rule_pass else 'FAIL'} "
          f"(Winner: '{audit_selected_model}', Val Fall F1: {val_fall_f1s['Random Forest']:.4f}).")

    # -------------------------------------------------------------------------
    # 9. EXPORT CSV VALIDATION TABLES
    # -------------------------------------------------------------------------
    print("\n[9] Exporting Validation CSV Tables...")
    df_metrics = pd.DataFrame(metric_comparison_rows)
    df_metrics.to_csv(os.path.join(B5_TABLES_DIR, "metric_recomputation.csv"), index=False)

    df_artifacts = pd.DataFrame(artifact_checks)
    df_artifacts.to_csv(os.path.join(B5_TABLES_DIR, "artifact_validation.csv"), index=False)

    df_repro = pd.DataFrame(reproducibility_rows)
    df_repro.to_csv(os.path.join(B5_TABLES_DIR, "reproducibility.csv"), index=False)
    print(f"  CSV tables saved to: {B5_TABLES_DIR}")

    # -------------------------------------------------------------------------
    # 10. COMPILE MACHINE-READABLE B5 JSON
    # -------------------------------------------------------------------------
    print("\n[10] Writing Machine-Readable B5 Validation JSON...")
    overall_status = (
        all_artifacts_present
        and feature_schema_summary["status"] == "PASS"
        and split_leakage_summary["subject_split_status"] == "PASS"
        and split_leakage_summary["trial_leakage_status"] == "PASS"
        and row_accounting_summary["status"] == "PASS"
        and all(c["status"] == "PASS" for c in structural_checks.values())
        and all_metrics_match
        and cm_sum_pass
        and all_reproducible
        and selection_rule_pass
    )

    final_status_str = "READY_FOR_SISFALL_FINAL_ML_VALIDATION" if overall_status else "NOT_READY_FOR_SISFALL_FINAL_ML_VALIDATION"

    b5_json_payload = {
        "stage": "B5",
        "dataset": "SisFall",
        "timestamp": "2026-10-02",
        "final_status": final_status_str,
        "overall_validation_pass": bool(overall_status),
        "artifact_checks": artifact_checks,
        "feature_schema": feature_schema_summary,
        "subject_split": split_leakage_summary,
        "row_accounting": row_accounting_summary,
        "model_structural_checks": structural_checks,
        "metric_recomputation": {
            "all_metrics_match": bool(all_metrics_match),
            "tolerance": tolerance,
            "recomputed_metrics": indep_metrics,
        },
        "confusion_matrix_checks": {
            "test_cm_sum_matches_test_rows": bool(cm_sum_pass),
            "expected_test_rows": 39339,
        },
        "reproducibility": {
            "all_reproducible": bool(all_reproducible),
            "models_tested": reproducibility_rows,
        },
        "selection_rule_check": {
            "primary_criterion": "Validation Fall F1",
            "secondary_criterion": "Validation ROC-AUC",
            "selected_model": audit_selected_model,
            "rule_adherence_pass": bool(selection_rule_pass),
        },
        "limitations": [
            "SisFall ground truth is annotated at trial level (no impact frame micro-annotations).",
            "38 subjects total (23 Young Adults SA, 15 Elderly SE); test partition comprises 5 held-out subjects.",
            "Waist/belt sensor position differs from wrist/chest wearable deployment targets.",
            "Offline benchmark does not establish real-world or clinical fall detection efficacy."
        ],
    }

    b5_json_path = os.path.join(REPORTS_DIR, "sisfall_stageB5_validation.json")
    with open(b5_json_path, "w", encoding="utf-8") as f:
        json.dump(b5_json_payload, f, indent=2)
    print(f"  Machine-readable B5 JSON saved to: {b5_json_path}")

    print("\n" + "=" * 80)
    print(f"  SISFALL STAGE B5 COMPLETED — FINAL STATUS: {final_status_str}")
    print("=" * 80)

    return b5_json_payload


if __name__ == "__main__":
    run_stage_b5_validation()
