"""Train XGB, RF, and MLP classifiers on ADS + CGC features."""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from utils.io_utils import load_pkl, save_pkl


def build_feature_matrix(
    features: List[dict],
    label_key: str = "label",
    layer_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Convert a list of per-token feature dicts into (X, y) arrays."""
    valid = [f for f in features if f.get(label_key) in (0, 1)]

    if not valid:
        n_cols = 0
        for f in features:
            if "ads_per_layer" in f and "cgc_per_layer" in f:
                n_layers = len(f["ads_per_layer"])
                n_cols = n_layers * 2
                break
        X = np.empty((0, n_cols), dtype=np.float32)
        y = np.empty((0,), dtype=np.int32)
        return X, y, []

    X_rows, y_rows, meta = [], [], []
    for f in valid:
        ads_full = np.array(f["ads_per_layer"],  dtype=np.float32)
        cgc_full = np.array(f["cgc_per_layer"],  dtype=np.float32)

        if layer_range is not None:
            L = len(ads_full)
            start = max(0, int(L * layer_range[0]))
            end   = min(L, int(L * layer_range[1]))
            ads_full  = ads_full[start:end]
            cgc_full  = cgc_full[start:end]

        x = np.concatenate([ads_full, cgc_full])

        X_rows.append(x)
        y_rows.append(int(f[label_key]))
        meta.append(f)

    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, meta


def split_by_image_id(
    features: List[dict],
    train_image_ids: set,
    val_image_ids: set,
    test_image_ids: Optional[set] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split feature list by image id to prevent leakage."""
    train = [f for f in features if f["image_id"] in train_image_ids]
    val   = [f for f in features if f["image_id"] in val_image_ids]
    test  = [f for f in features if f["image_id"] in test_image_ids] \
            if test_image_ids else val
    return train, val, test


def build_classifier(clf_type: str, params: dict):
    """Instantiate a classifier from type string and hyperparameter dict."""
    if clf_type == "xgb":
        return XGBClassifier(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 500),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif clf_type == "rf":
        return RandomForestClassifier(
            max_depth=params.get("max_depth", 10),
            n_estimators=params.get("n_estimators", 400),
            random_state=42,
            n_jobs=-1,
        )
    elif clf_type == "mlp":
        hidden = params.get("hidden_layer_sizes", [128])
        if isinstance(hidden[0], list):
            hidden_tuple = tuple(hidden[0])
        else:
            hidden_tuple = tuple(hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden_tuple,
            learning_rate_init=params.get("learning_rate_init", 0.001),
            solver=params.get("solver", "adam"),
            max_iter=params.get("max_iter", 500),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def grid_search(
    clf_type: str,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scoring: str = "f1",
) -> Tuple[object, dict, float]:
    """Exhaustive grid search over `param_grid`, evaluated by `scoring` on val set."""
    best_clf = None
    best_params = {}
    best_score = -1.0

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError(
            f"grid_search received empty split: "
            f"X_train={X_train.shape}, X_val={X_val.shape}. "
            "Ensure both train and val splits contain labeled object tokens."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Training set contains only one class {np.unique(y_train)}. "
            "Need at least one sample of each class (0=true, 1=hallucinated)."
        )

    for params in ParameterGrid(param_grid):
        clf = build_classifier(clf_type, params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        if scoring == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

        if score > best_score:
            best_score = score
            best_params = params
            best_clf = clf

    return best_clf, best_params, best_score


def evaluate_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Return precision, recall, F1, accuracy, and AUC."""
    y_pred = clf.predict(X)

    try:
        y_prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    except Exception:
        auc = float("nan")

    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y, y_pred),
        "auc":       auc,
    }


def train_and_evaluate(
    feature_path: str,
    train_image_ids: set,
    val_image_ids: set,
    test_image_ids: set,
    clf_configs: dict,
    output_dir: str,
    model_key: str = "model",
) -> Dict[str, dict]:
    """Full train + evaluate pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    all_features = load_pkl(feature_path)
    print(f"[Train] Loaded {len(all_features)} token features from {feature_path}")

    train_feats, val_feats, test_feats = split_by_image_id(
        all_features, train_image_ids, val_image_ids, test_image_ids
    )
    print(
        f"[Train] Split: train={len(train_feats)}, "
        f"val={len(val_feats)}, test={len(test_feats)} tokens"
    )

    X_train, y_train, _ = build_feature_matrix(train_feats)
    X_val,   y_val,   _ = build_feature_matrix(val_feats)
    X_test,  y_test,  _ = build_feature_matrix(test_feats)

    print(
        f"[Train] Feature matrices — "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )
    print(
        f"[Train] Label balance — "
        f"train: {y_train.mean():.2%} hallucinated, "
        f"test: {y_test.mean():.2%} hallucinated"
    )

    if X_train.shape[0] == 0:
        raise ValueError(
            "[Train] train split is empty after filtering to labeled tokens. "
            "Run step2 with more images, or check labeling.json for missing spans."
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"[Train] train split has only one class {np.unique(y_train)}. "
            "Need both class 0 (true) and class 1 (hallucinated) tokens."
        )
    if X_val.shape[0] == 0 or len(np.unique(y_val)) < 2:
        print(
            f"[Train] WARNING: val split has {X_val.shape[0]} samples / "
            f"{len(np.unique(y_val))} class(es). Falling back to train split for "
            "hyperparameter tuning — do not use these results as final numbers."
        )
        X_val, y_val = X_train.copy(), y_train.copy()
    if X_test.shape[0] == 0 or len(np.unique(y_test)) < 2:
        print(
            f"[Train] WARNING: test split has {X_test.shape[0]} samples / "
            f"{len(np.unique(y_test))} class(es). Falling back to val split for "
            "final evaluation."
        )
        X_test, y_test = X_val.copy(), y_val.copy()

    all_results = {}

    print("\n[Train] Grid-searching XGBoost …")
    xgb_grid = clf_configs.get("xgb", {})
    _sanitise_grid(xgb_grid)
    best_xgb, xgb_params, xgb_val_f1 = grid_search(
        "xgb", xgb_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best XGB params: {xgb_params}  (val F1={xgb_val_f1:.4f})")
    xgb_metrics = evaluate_classifier(best_xgb, X_test, y_test)
    xgb_metrics["best_params"] = xgb_params
    all_results["xgb"] = xgb_metrics
    save_pkl(best_xgb, os.path.join(output_dir, f"{model_key}_xgb.pkl"))

    print("\n[Train] Grid-searching Random Forest …")
    rf_grid = clf_configs.get("rf", {})
    _sanitise_grid(rf_grid)
    best_rf, rf_params, rf_val_f1 = grid_search(
        "rf", rf_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best RF params: {rf_params}  (val F1={rf_val_f1:.4f})")
    rf_metrics = evaluate_classifier(best_rf, X_test, y_test)
    rf_metrics["best_params"] = rf_params
    all_results["rf"] = rf_metrics
    save_pkl(best_rf, os.path.join(output_dir, f"{model_key}_rf.pkl"))

    print("\n[Train] Grid-searching MLP …")
    mlp_grid = clf_configs.get("mlp", {})
    _sanitise_grid(mlp_grid)
    best_mlp, mlp_params, mlp_val_f1 = grid_search(
        "mlp", mlp_grid, X_train, y_train, X_val, y_val
    )
    print(f"  Best MLP params: {mlp_params}  (val F1={mlp_val_f1:.4f})")
    mlp_metrics = evaluate_classifier(best_mlp, X_test, y_test)
    mlp_metrics["best_params"] = mlp_params
    all_results["mlp"] = mlp_metrics
    save_pkl(best_mlp, os.path.join(output_dir, f"{model_key}_mlp.pkl"))

    print("\n" + "=" * 60)
    print(f"  Results for {model_key}")
    print("=" * 60)
    print(f"{'Method':<10} {'PR':>6} {'RC':>6} {'F1':>6} {'ACC':>6} {'AUC':>6}")
    print("-" * 60)
    for clf_name, m in all_results.items():
        print(
            f"{clf_name.upper():<10} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['accuracy']:>6.3f} "
            f"{m['auc']:>6.3f}"
        )

    results_path = os.path.join(output_dir, f"{model_key}_results.pkl")
    save_pkl(all_results, results_path)
    print(f"\n[Train] Results saved to {results_path}")

    return all_results


def _sanitise_grid(grid: dict) -> None:
    """sklearn ParameterGrid requires lists, not single values."""
    for k, v in grid.items():
        if not isinstance(v, list):
            grid[k] = [v]

