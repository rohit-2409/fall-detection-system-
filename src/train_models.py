"""
Model Training & Validation Module for KFall Fall Detection (Stage 3).

Trains baseline classifiers (Logistic Regression, Random Forest, SVM)
strictly on TRAIN subjects, evaluates them on VALIDATION subjects,
and selects the final model based on a transparent predefined validation criterion.
"""

import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import compute_metrics, load_and_audit_dataset

RANDOM_SEED = 42

# Predefined transparent selection criterion
PRIMARY_SELECTION_METRIC = "fall_f1"  # F1-score specifically for Class 1 (Fall)
SECONDARY_SELECTION_METRIC = "roc_auc"


def build_candidate_models(random_seed: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Constructs the three baseline candidate pipelines.
    Feature scaling is bundled inside the pipeline to ensure leak-free transformation.
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
                    random_state=random_seed,
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
                    random_state=random_seed,
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
                    random_state=random_seed,
                ),
            ),
        ]),
    }
    return models


def train_and_validate_candidates(
    df: pd.DataFrame,
    feature_cols: List[str],
    random_seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict], str]:
    """
    Trains all candidate models strictly on TRAIN subjects and evaluates on VALIDATION subjects.
    Selects the winning model based on PRIMARY_SELECTION_METRIC.
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(int)

    candidate_pipelines = build_candidate_models(random_seed=random_seed)
    val_results = {}

    print(f"\nTraining {len(candidate_pipelines)} baseline models on Train partition ({len(X_train)} windows)...")

    for name, pipeline in candidate_pipelines.items():
        print(f"  Training {name}...")
        pipeline.fit(X_train, y_train)

        # Validation prediction
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_metrics(y_true=y_val, y_pred=val_preds, y_prob=val_probs)
        val_results[name] = metrics
        print(
            f"    -> Val Acc: {metrics['accuracy']:.4f} | Fall Prec: {metrics['fall_precision']:.4f} | "
            f"Fall Rec: {metrics['fall_recall']:.4f} | Fall F1: {metrics['fall_f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}"
        )

    # Model selection based strictly on validation performance
    # Primary: fall_f1, Secondary: roc_auc
    def selection_key(item):
        m = item[1]
        f1 = m.get(PRIMARY_SELECTION_METRIC, 0.0) or 0.0
        auc = m.get(SECONDARY_SELECTION_METRIC, 0.0) or 0.0
        return (f1, auc)

    selected_model_name = max(val_results.items(), key=selection_key)[0]
    print(f"\nModel Selection Outcome (Criterion: {PRIMARY_SELECTION_METRIC}):")
    print(f"  Selected Model: '{selected_model_name}'")
    print(
        f"  Validation {PRIMARY_SELECTION_METRIC}: {val_results[selected_model_name][PRIMARY_SELECTION_METRIC]:.4f}"
    )

    return candidate_pipelines, val_results, selected_model_name


def save_validation_comparison(
    val_results: Dict[str, Dict],
    selected_model_name: str,
    output_path: str = os.path.join(PROJECT_ROOT, "reports", "model_comparison.json"),
):
    """
    Saves the comparative validation metrics to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "selection_criterion": {
            "primary": PRIMARY_SELECTION_METRIC,
            "secondary": SECONDARY_SELECTION_METRIC,
            "description": "Validation Fall F1-Score (harmonic mean of Fall Precision and Fall Recall) with ROC-AUC tie-breaker.",
        },
        "selected_model": selected_model_name,
        "models": val_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved model comparison report: {output_path}")


if __name__ == "__main__":
    df, feature_cols, leakage_audit, split_counts = load_and_audit_dataset()
    candidate_pipelines, val_results, selected_model_name = train_and_validate_candidates(
        df=df, feature_cols=feature_cols
    )
    save_validation_comparison(val_results=val_results, selected_model_name=selected_model_name)
