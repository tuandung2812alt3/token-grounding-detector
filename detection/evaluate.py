"""Threshold-based evaluation and layer-wise analysis utilities."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from detection.train import build_feature_matrix
from utils.io_utils import load_pkl


def evaluate_ads_threshold(
    features: List[dict],
    tau: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate ADS as a standalone hallucination detector via thresholding."""
    valid = [f for f in features if f.get("label") in (0, 1)]
    scores = np.array([f["ads_score"] for f in valid], dtype=np.float32)
    labels = np.array([f["label"] for f in valid], dtype=np.int32)

    if tau is None:
        tau = _best_threshold(scores, labels, higher_is_hallucinated=True)

    y_pred = (scores > tau).astype(int)
    return _compute_metrics(labels, y_pred, scores)


def evaluate_cgc_threshold(
    features: List[dict],
    tau: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate CGC as a standalone detector."""
    valid = [f for f in features if f.get("label") in (0, 1)]
    scores = np.array([f["cgc_score"] for f in valid], dtype=np.float32)
    labels = np.array([f["label"] for f in valid], dtype=np.int32)

    inv_scores = -scores
    if tau is None:
        tau = _best_threshold(inv_scores, labels, higher_is_hallucinated=True)

    y_pred = (inv_scores > tau).astype(int)
    return _compute_metrics(labels, y_pred, inv_scores)


def _best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    higher_is_hallucinated: bool = True,
) -> float:
    """Grid-search threshold that maximises F1."""
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_tau, best_f1 = thresholds[0], -1.0
    for tau in thresholds:
        y_pred = (scores > tau).astype(int)
        f1 = f1_score(labels, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return float(best_tau)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc":       float(auc),
    }


def evaluate_trained_classifier(
    clf,
    features: List[dict],
) -> Dict[str, float]:
    """Evaluate a trained sklearn/XGB classifier on a list of feature dicts."""
    X, y, _ = build_feature_matrix(features)
    y_pred = clf.predict(X)
    try:
        y_prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y, y_pred)),
        "auc":       float(auc),
        "report":    classification_report(y, y_pred, zero_division=0),
    }


def print_confusion_matrix(clf, features: List[dict]) -> None:
    """Print a labelled confusion matrix to stdout."""
    X, y, _ = build_feature_matrix(features)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (rows=true, cols=pred):")
    print("              Pred: True  Pred: Hall")
    print(f"  True: True      {cm[0,0]:>5}       {cm[0,1]:>5}")
    print(f"  True: Hall      {cm[1,0]:>5}       {cm[1,1]:>5}")


LAYER_RANGES = {
    "1-10":  (0,  10),
    "10-15": (10, 15),
    "15-20": (15, 20),
    "20-25": (20, 25),
    "25-33": (25, 33),
}


def layerwise_analysis(
    features: List[dict],
    feature_key: str = "ads_per_layer",
) -> Dict[str, dict]:
    """Compute mean ± std and Mann-Whitney U p-value for true vs. hallucinated"""
    valid = [f for f in features if f.get("label") in (0, 1)]
    true_feats = [f for f in valid if f["label"] == 0]
    hall_feats = [f for f in valid if f["label"] == 1]

    true_mat = np.array([f[feature_key] for f in true_feats], dtype=np.float32)
    hall_mat = np.array([f[feature_key] for f in hall_feats], dtype=np.float32)

    results = {}
    for range_label, (start, end) in LAYER_RANGES.items():
        end = min(end, true_mat.shape[1], hall_mat.shape[1])
        true_vals = true_mat[:, start:end].mean(axis=1)
        hall_vals = hall_mat[:, start:end].mean(axis=1)

        _, p_val = stats.mannwhitneyu(
            true_vals, hall_vals, alternative="two-sided"
        )
        results[range_label] = {
            "true_mean": float(true_vals.mean()),
            "true_std":  float(true_vals.std()),
            "hall_mean": float(hall_vals.mean()),
            "hall_std":  float(hall_vals.std()),
            "p_value":   float(p_val),
        }
        print(
            f"  Layers {range_label:6s}: "
            f"True={true_vals.mean():.3f}±{true_vals.std():.3f}  "
            f"Hall={hall_vals.mean():.3f}±{hall_vals.std():.3f}  "
            f"p={p_val:.2e}"
        )

    return results


def compute_shap_importance(
    clf,
    features: List[dict],
    num_layers: int,
    max_samples: int = 500,
) -> Optional[Dict[str, np.ndarray]]:
    """Compute mean absolute SHAP values for ADS and CGC features separately."""
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed. Run: pip install shap")
        return None

    X, y, _ = build_feature_matrix(features)
    X_sample = X[:max_samples]

    explainer = shap.TreeExplainer(clf) if hasattr(clf, "get_booster") \
                else shap.KernelExplainer(clf.predict_proba, X_sample[:50])
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = np.abs(shap_values[1])
    else:
        sv = np.abs(shap_values)

    ads_shap = sv[:, :num_layers].mean(axis=0)
    cgc_shap = sv[:, num_layers:].mean(axis=0)

    return {"ads_shap": ads_shap, "cgc_shap": cgc_shap}
