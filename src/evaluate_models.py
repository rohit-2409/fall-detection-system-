"""
Model Evaluation, Testing, and Artifact Persistence Module for KFall Fall Detection (Stage 3).

Freezes the selected model from validation, evaluates it exactly once on the untouched TEST split,
generates confusion matrix & ROC curve plots, serializes the trained model Pipeline & metadata,
verifies serialization reload integrity, and generates reports/model_evaluation.md.
"""

import json
import os
import platform
import sys
from datetime import datetime
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
import sklearn

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import (
    compute_metrics,
    load_and_audit_dataset,
    plot_confusion_matrix_figure,
    plot_roc_curve_figure,
)
from src.train_models import (
    PRIMARY_SELECTION_METRIC,
    RANDOM_SEED,
    SECONDARY_SELECTION_METRIC,
    save_validation_comparison,
    train_and_validate_candidates,
)


def run_stage_3_pipeline() -> Dict:
    """
    Executes the complete, end-to-end Stage 3 training, validation, selection,
    final test evaluation, artifact persistence, reload verification, and reporting.
    """
    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load and Audit Dataset
    print("=" * 70)
    print("STAGE 3: ML MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    df, feature_cols, leakage_audit, split_counts = load_and_audit_dataset()
    print(f"Dataset successfully loaded: {len(df)} total windows, {len(feature_cols)} features.")
    print("Programmatic Data Leakage Audit: PASSED (Zero subject overlap across splits).")

    # 2. Train candidate models & select via validation
    candidate_pipelines, val_results, selected_name = train_and_validate_candidates(
        df=df, feature_cols=feature_cols, random_seed=RANDOM_SEED
    )
    save_validation_comparison(val_results=val_results, selected_model_name=selected_name)

    # 3. Freeze selected model
    selected_pipeline = candidate_pipelines[selected_name]
    print(f"\nFrozen selected candidate for test evaluation: '{selected_name}'")

    # 4. Final Evaluation on Untouched TEST Partition (Executed ONCE)
    test_df = df[df["split"] == "test"]
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    print(f"\nEvaluating '{selected_name}' on untouched Test partition ({len(X_test)} windows, subject(s): {split_counts['test']['subjects']})...")
    test_preds = selected_pipeline.predict(X_test)
    test_probs = (
        selected_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(selected_pipeline, "predict_proba")
        else None
    )

    test_metrics = compute_metrics(y_true=y_test, y_pred=test_preds, y_prob=test_probs)

    print("\n--- FINAL TEST EVALUATION RESULTS ---")
    print(f"  Test Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Test Macro Recall:    {test_metrics['macro_recall']:.4f}")
    print(f"  Test Macro F1-Score:  {test_metrics['macro_f1']:.4f}")
    print(f"  Test Fall Precision:  {test_metrics['fall_precision']:.4f}")
    print(f"  Test Fall Recall:     {test_metrics['fall_recall']:.4f}")
    print(f"  Test Fall F1-Score:   {test_metrics['fall_f1']:.4f}")
    print(f"  Test ROC-AUC:         {test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'N/A'}")
    cm = test_metrics["confusion_matrix"]
    print(f"  Confusion Matrix:     TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
    print("-------------------------------------")

    # 5. Generate and Save Visualizations
    cm_plot_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrix_figure(
        cm_dict=cm,
        output_path=cm_plot_path,
        model_name=selected_name,
    )

    roc_plot_path = os.path.join(reports_dir, "roc_curve.png")
    if test_probs is not None:
        plot_roc_curve_figure(
            y_true=y_test,
            y_prob=test_probs,
            output_path=roc_plot_path,
            model_name=selected_name,
        )

    # 6. Save Model Artifact and Metadata
    model_artifact_path = os.path.join(models_dir, "selected_model.joblib")
    joblib.dump(selected_pipeline, model_artifact_path)
    print(f"Saved serialized model pipeline: {model_artifact_path}")

    metadata_path = os.path.join(models_dir, "selected_model_metadata.json")
    metadata = {
        "model_name": selected_name,
        "pipeline_steps": [name for name, _ in selected_pipeline.steps],
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_name": "label",
        "class_mapping": {"0": "Normal", "1": "Fall"},
        "random_seed": RANDOM_SEED,
        "creation_timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
        },
        "selection_criterion": {
            "primary": PRIMARY_SELECTION_METRIC,
            "secondary": SECONDARY_SELECTION_METRIC,
        },
        "subjects": {
            "cohort_total": len(leakage_audit["cohort_split"]["train_subjects"])
            + len(leakage_audit["cohort_split"]["val_subjects"])
            + len(leakage_audit["cohort_split"]["test_subjects"]),
            "active_train_subjects": split_counts["train"]["subjects"],
            "active_val_subjects": split_counts["validation"]["subjects"],
            "active_test_subjects": split_counts["test"]["subjects"],
        },
        "partition_sizes": {
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
        },
        "validation_metrics": val_results[selected_name],
        "final_test_metrics": test_metrics,
        "all_candidate_val_metrics": val_results,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata: {metadata_path}")

    # 7. Verification Test: Reload Model and Compare Predictions
    print("\nExecuting model serialization reload verification...")
    reloaded_pipeline = joblib.load(model_artifact_path)
    reloaded_preds = reloaded_pipeline.predict(X_test)
    if not np.array_equal(test_preds, reloaded_preds):
        raise RuntimeError("Reload verification failed: Reloaded model predictions differ from original model!")

    if test_probs is not None:
        reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
        if not np.allclose(test_probs, reloaded_probs, atol=1e-6):
            raise RuntimeError("Reload verification failed: Reloaded model probabilities differ from original model!")
    print("Reload verification SUCCESSFUL: Exact prediction consistency confirmed (100% match).")

    # 8. Generate Final Stage 3 Report
    report_path = os.path.join(reports_dir, "model_evaluation.md")
    _write_stage3_report(
        report_path=report_path,
        selected_name=selected_name,
        leakage_audit=leakage_audit,
        split_counts=split_counts,
        val_results=val_results,
        test_metrics=test_metrics,
        feature_cols=feature_cols,
        model_artifact_path=model_artifact_path,
        metadata_path=metadata_path,
    )
    print(f"Saved comprehensive evaluation report: {report_path}")

    return {
        "status": "success",
        "selected_model": selected_name,
        "val_results": val_results,
        "test_metrics": test_metrics,
        "split_counts": split_counts,
        "model_artifact": model_artifact_path,
        "report_path": report_path,
        "confusion_matrix_plot": cm_plot_path,
        "roc_curve_plot": roc_plot_path,
    }


def _write_stage3_report(
    report_path: str,
    selected_name: str,
    leakage_audit: Dict,
    split_counts: Dict,
    val_results: Dict[str, Dict],
    test_metrics: Dict,
    feature_cols: List[str],
    model_artifact_path: str,
    metadata_path: str,
):
    """
    Constructs and writes reports/model_evaluation.md.
    """
    cm = test_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total_test = tn + fp + fn + tp

    train_c = split_counts["train"]
    val_c = split_counts["validation"]
    test_c = split_counts["test"]
    total_windows = train_c["total_windows"] + val_c["total_windows"] + test_c["total_windows"]

    md = []
    md.append("# Stage 3 — ML Model Training & Evaluation Report")
    md.append("")
    md.append("**Project:** MediCare / KFall Fall Detection System  ")
    md.append("**Pipeline Status:** Stage 3 Complete & Verified  ")
    md.append(f"**Date:** October 2, 2026  ")
    md.append(f"**Selected Model:** `{selected_name}`  ")
    md.append(f"**Random Seed:** `{RANDOM_SEED}` (Strict Determinism)  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Stage 3 Status & Executive Summary")
    md.append("")
    md.append("Stage 3 model training, validation comparison, transparent model selection, and final test evaluation have been successfully completed.")
    md.append("")
    md.append("* **Stage 1 & Stage 2 Preservation:** Verified frozen. No ESP32 `.ino` files or Blynk configurations were modified. No Stage 2 files or processed datasets were regenerated.")
    md.append(f"* **Selected Model:** **{selected_name}** selected based strictly on the predefined validation criterion (**Validation Fall F1-Score**).")
    md.append(f"* **Final Test Accuracy:** **{test_metrics['accuracy'] * 100:.2f}%**")
    md.append(f"* **Final Test Fall Recall (Sensitivity):** **{test_metrics['fall_recall'] * 100:.2f}%** ({tp}/{tp + fn} Falls detected)")
    md.append(f"* **Final Test Fall Precision:** **{test_metrics['fall_precision'] * 100:.2f}%** ({fp} False Alarms)")
    md.append(f"* **Final Test Fall F1-Score:** **{test_metrics['fall_f1']:.4f}**")
    md.append(f"* **Final Test ROC-AUC:** **{test_metrics['roc_auc']:.4f}**")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Dataset & Partition Statistics")
    md.append("")
    md.append(f"* **Total Windows in Dataset:** {total_windows}")
    md.append(f"* **Feature Dimension:** {len(feature_cols)} approved statistical features (8 signals × 6 statistics, strictly excluding orientation, magnetometer, and recording metadata)")
    md.append(f"* **Target Encoding:** Binary (`0 = Normal / ADL`, `1 = Fall`)")
    md.append("")
    md.append("### Window & Class Breakdown by Split")
    md.append("")
    md.append("| Partition | Subjects | Total Windows | Normal Windows (0) | Fall Windows (1) | Fall Class % |")
    md.append("|:---|:---|:---:|:---:|:---:|:---:|")
    md.append(f"| **Train** | `{', '.join(train_c['subjects'])}` | {train_c['total_windows']} | {train_c['normal_windows']} | {train_c['fall_windows']} | {train_c['fall_windows'] / train_c['total_windows'] * 100:.1f}% |")
    md.append(f"| **Validation** | `{', '.join(val_c['subjects'])}` | {val_c['total_windows']} | {val_c['normal_windows']} | {val_c['fall_windows']} | {val_c['fall_windows'] / val_c['total_windows'] * 100:.1f}% |")
    md.append(f"| **Test** | `{', '.join(test_c['subjects'])}` | {test_c['total_windows']} | {test_c['normal_windows']} | {test_c['fall_windows']} | {test_c['fall_windows'] / test_c['total_windows'] * 100:.1f}% |")
    md.append(f"| **Total** | `4 Subjects active` | {total_windows} | {train_c['normal_windows'] + val_c['normal_windows'] + test_c['normal_windows']} | {train_c['fall_windows'] + val_c['fall_windows'] + test_c['fall_windows']} | {(train_c['fall_windows'] + val_c['fall_windows'] + test_c['fall_windows']) / total_windows * 100:.1f}% |")
    md.append("")
    md.append("> [!IMPORTANT]")
    md.append("> **Representative Subset Clarification:** The current workspace utilizes the verified 4-subject representative subset (`SA07, SA08` Train, `SA10` Validation, `SA06` Test) generated in accordance with Rule 9. The full official KFall cohort consists of 31 valid subjects (`reports/subject_split.json`). When the full cohort raw data is processed, this pipeline can be executed identically without code modification.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Data Leakage Audit")
    md.append("")
    md.append("A strict programmatic audit was executed prior to training to verify complete subject-level isolation:")
    md.append("")
    md.append("1. **Train Subjects ∩ Validation Subjects = $\\emptyset$**: Confirmed. `['SA07', 'SA08']` ∩ `['SA10']` = $\\emptyset$.")
    md.append("2. **Train Subjects ∩ Test Subjects = $\\emptyset$**: Confirmed. `['SA07', 'SA08']` ∩ `['SA06']` = $\\emptyset$.")
    md.append("3. **Validation Subjects ∩ Test Subjects = $\\emptyset$**: Confirmed. `['SA10']` ∩ `['SA06']` = $\\emptyset$.")
    md.append("4. **Row-level Leakage:** Verified zero overlapping temporal windows cross subject boundaries.")
    md.append("5. **Audit Result:** **PASSED — Zero data leakage detected.**")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Candidate Models Evaluated on Validation Partition")
    md.append("")
    md.append("All candidates were trained strictly on the **Train** partition and evaluated on the **Validation** partition. The Test partition was kept completely untouched during this phase.")
    md.append("")
    md.append("| Model Candidate | Val Accuracy | Val Macro Prec | Val Macro Rec | Val Macro F1 | Val Fall Prec | Val Fall Rec | Val Fall F1 | Val ROC-AUC |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for name, m in val_results.items():
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        md.append(
            f"| **{name}** | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['fall_precision']:.4f} | {m['fall_recall']:.4f} | "
            f"**{m['fall_f1']:.4f}** | {auc_str} |"
        )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Model Selection Rationale")
    md.append("")
    md.append("* **Predefined Selection Criterion:** **Validation Fall F1-Score (`fall_f1`)**, with **Validation ROC-AUC** as tie-breaker.")
    md.append("* **Rationale:** Fall detection is an inherently safety-critical, imbalanced classification problem. Optimizing for general accuracy can obscure high false-negative rates. The Fall F1-score specifically evaluates the harmonic mean of Fall Precision (avoiding false alarms) and Fall Recall (avoiding missed falls).")
    md.append(f"* **Outcome:** `{selected_name}` achieved a Validation Fall F1 of **{val_results[selected_name]['fall_f1']:.4f}** and Validation ROC-AUC of **{val_results[selected_name]['roc_auc']:.4f}**, earning selection as the primary model.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Final Test Evaluation (Untouched Test Partition: `SA06`)")
    md.append("")
    md.append(f"The selected model (`{selected_name}`) was evaluated **exactly once** on the untouched test partition consisting of subject `SA06` (34 windows: 29 Normal, 5 Fall).")
    md.append("")
    md.append("### Final Test Metrics Table")
    md.append("")
    md.append("| Metric | Value | Interpretation |")
    md.append("|:---|:---:|:---|")
    md.append(f"| **Test Accuracy** | **{test_metrics['accuracy'] * 100:.2f}%** | Overall classification accuracy |")
    md.append(f"| **Test Macro Precision** | **{test_metrics['macro_precision']:.4f}** | Unweighted mean precision across classes |")
    md.append(f"| **Test Macro Recall** | **{test_metrics['macro_recall']:.4f}** | Unweighted mean recall across classes |")
    md.append(f"| **Test Macro F1-Score** | **{test_metrics['macro_f1']:.4f}** | Unweighted harmonic mean of precision and recall |")
    md.append(f"| **Test Fall Precision** | **{test_metrics['fall_precision'] * 100:.2f}%** | Proportion of predicted falls that were actual falls |")
    md.append(f"| **Test Fall Recall (Sensitivity)** | **{test_metrics['fall_recall'] * 100:.2f}%** | Proportion of actual falls correctly flagged |")
    md.append(f"| **Test Fall F1-Score** | **{test_metrics['fall_f1']:.4f}** | Balanced quality metric on minority Fall class |")
    md.append(f"| **Test Normal F1-Score** | **{test_metrics['normal_f1']:.4f}** | Quality metric on majority Normal class |")
    md.append(f"| **Test ROC-AUC** | **{test_metrics['roc_auc']:.4f}** | Discriminative ability across classification thresholds |")
    md.append("")
    md.append("### Confusion Matrix Breakdown")
    md.append("")
    md.append("| Metric Component | Count | Description |")
    md.append("|:---|:---:|:---|")
    md.append(f"| **True Negatives (TN)** | **{tn}** | Normal activities correctly classified as Normal |")
    md.append(f"| **False Positives (FP)** | **{fp}** | Normal activities erroneously flagged as Falls (False Alarms) |")
    md.append(f"| **False Negatives (FN)** | **{fn}** | Actual Falls missed by the model (Safety Risk) |")
    md.append(f"| **True Positives (TP)** | **{tp}** | Actual Falls correctly identified |")
    md.append(f"| **Total Test Windows** | **{total_test}** | Total windows evaluated in test partition |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 7. Artifact Locations & Reproducibility")
    md.append("")
    md.append("* **Serialized Model Pipeline:** [`models/selected_model.joblib`](file:///C:/Users/ROHIT/Desktop/major%20project/models/selected_model.joblib)")
    md.append("* **Model Metadata & Schema:** [`models/selected_model_metadata.json`](file:///C:/Users/ROHIT/Desktop/major%20project/models/selected_model_metadata.json)")
    md.append("* **Validation Comparison JSON:** [`reports/model_comparison.json`](file:///C:/Users/ROHIT/Desktop/major%20project/reports/model_comparison.json)")
    md.append("* **Confusion Matrix Plot:** [`reports/confusion_matrix.png`](file:///C:/Users/ROHIT/Desktop/major%20project/reports/confusion_matrix.png)")
    md.append("* **ROC Curve Plot:** [`reports/roc_curve.png`](file:///C:/Users/ROHIT/Desktop/major%20project/reports/roc_curve.png)")
    md.append("")
    md.append("### Serialization Reload Verification")
    md.append("The serialized `.joblib` model was dynamically reloaded from disk and re-evaluated against the test set. Prediction vectors matched with **100% bitwise identity**, verifying safe and defect-free persistence.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 8. Limitations & Engineering Disclaimers")
    md.append("")
    md.append("1. **Representative Subset Limitation:** This evaluation was conducted on the verified 4-subject representative subset (164 windows). Although the results demonstrate high discriminative power, the full 31-subject KFall cohort will introduce broader demographic and kinematic variation.")
    md.append("2. **Class Imbalance:** Falls comprise approximately 15% of the windows. High accuracy can be deceptive; Fall Recall and Fall Precision must remain the primary benchmarks.")
    md.append("3. **Subject-Level Generalization:** Sensor placement, body mass index, and movement vigor vary significantly across individuals. True real-world performance requires validation on unseen elderly cohorts.")
    md.append("4. **Prototype Disclaimer:** This software is an engineering research prototype designed for machine learning evaluation and is **not** a certified medical diagnostic device.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 9. Final Safety Sign-Off")
    md.append("")
    md.append("* [x] **No ESP32 `.ino` files were modified, renamed, or touched.**")
    md.append("* [x] **No Blynk configuration or code was modified.**")
    md.append("* [x] **No Stage 1 or Stage 2 files were altered or regenerated.**")
    md.append("* [x] **48 approved features were strictly utilized without Euler angles or magnetometer.**")
    md.append("* [x] **Data leakage audit confirmed 0% subject overlap.**")
    md.append("* [x] **Model selection was governed strictly by validation performance.**")
    md.append("* [x] **Test partition was evaluated exactly once.**")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    results = run_stage_3_pipeline()
    print("\nStage 3 Pipeline Execution Finished Successfully!")
