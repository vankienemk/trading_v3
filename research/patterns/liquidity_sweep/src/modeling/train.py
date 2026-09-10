"""Model training pipeline for liquidity sweep prediction.

Implements guide section 20: train models in order Logistic Regression →
Random Forest → CatBoost → LightGBM/XGBoost.  Calibrates probabilities
(Platt/Isotonic per §20.5) and evaluates with Brier score + calibration
curve.

The pipeline:
1. Build feature matrix X and label vector y from scored events.
2. Time-based split with purge + embargo (src.modeling.split).
3. Train models sequentially; keep the best by validation PR-AUC.
4. Calibrate on validation set.
5. Evaluate on test set ONCE (no peeking).

Deliverables:
- artifacts/models/model.pkl — serialized best model + preprocessor
- artifacts/models/calibrator.pkl — calibrated wrapper
- reports/metrics/test_metrics.json — out-of-sample metrics
- reports/figures/calibration_curve.png — calibration visualization
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.modeling.split import apply_purge_and_embargo, build_train_val_test_splits

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class ModelingError(RuntimeError):
    """Raised when modeling pipeline fails."""


# Feature columns that must NEVER be used as inputs (guide §20.4)
# Prefix-based banning catches all variants (mfe_r_h4, mae_r_h32, etc.)
BANNED_PREFIXES = [
    "mfe_r_", "mae_r_", "close_return_r_", "max_close_return_r_",
    "min_close_return_r_", "outcome_", "target_", "bars_to_", "bars_held",
    "exit_", "gross_", "net_", "cost_", "future_",
]

# Exact-match banned columns (legacy + identity)
BANNED_EXACT = {
    "event_id", "event_time", "direction", "level_id", "level_price",
    "entry_time", "entry_price", "stop_price", "target_1r", "target_1_5r",
    "target_2r", "exit_time", "exit_price", "exit_reason",
    "confirmation_time", "rule_score", "score_level", "score_sweep",
    "score_reclaim", "score_confirmation", "score_context", "score_volume",
}


def _load_registry_features() -> set[str]:
    """Load the 32 canonical feature names from the feature registry."""
    import json
    import os

    # Project root is 3 levels up from src/modeling/train.py
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    registry_path = os.path.join(
        project_root, "artifacts", "feature_schemas", "features.json",
    )
    if not os.path.exists(registry_path):
        raise ModelingError(
            f"Feature registry not found at {registry_path}. "
            "Run Pipeline 3 (xauusd-dataset) first to generate artifacts."
        )

    with open(registry_path) as f:
        data = json.load(f)

    feature_names = {feat["name"] for feat in data.get("features", [])}
    if len(feature_names) != 32:
        raise ModelingError(
            f"Registry has {len(feature_names)} features, expected 32"
        )
    return feature_names


def prepare_features_and_labels(
    events_df: pd.DataFrame,
    target_col: str = "outcome_2r_h16",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Extract feature matrix X, binary label y, and feature names.

    Uses WHITELIST approach: keeps ONLY the 32 registered features from
    artifacts/feature_schemas/features.json. All other columns are dropped,
    including any future/label/excursion columns that might have slipped through.

    Converts multi-class outcome to binary: tp=1, else=0.

    Parameters
    ----------
    events_df : DataFrame
        Labeled event dataset with features + outcomes.
    target_col : str
        Outcome column name (default outcome_2r_h16).

    Returns
    -------
    tuple[DataFrame, Series, list[str]]
        (X, y_binary, feature_names) — exactly 32 features from registry.
    """
    if target_col not in events_df.columns:
        raise ModelingError(f"Target column '{target_col}' not found in events")

    df = events_df.copy()

    # Create binary label: tp=1, everything else=0
    y_binary = (df[target_col] == "tp").astype(int)

    # Load registry features (whitelist)
    registry_features = _load_registry_features()

    if registry_features:
        # Whitelist approach: keep ONLY registered features
        available = registry_features & set(df.columns)
        missing = registry_features - set(df.columns)
        if missing:
            raise ModelingError(
                f"Registry has {len(missing)} features missing from dataset: "
                f"{sorted(missing)[:5]}..."
            )
        if len(available) != 32:
            raise ModelingError(
                f"Expected 32 registry features, found {len(available)}: "
                f"{sorted(available)}"
            )

        # Use ONLY numeric features from registry (categorical excluded for simplicity)
        # This gives ~30 features; 2 categorical (level_type, volatility_regime) are dropped
        numeric_features = sorted([
            name for name in available
            if pd.api.types.is_numeric_dtype(df[name])
        ])

        if len(numeric_features) < 20:
            raise ModelingError(
                f"Only {len(numeric_features)} numeric features from registry; "
                "need at least 20 for modeling."
            )

        feature_names = numeric_features
        X = df[feature_names].copy()

        # NOTE: Imputation is handled by the training pipeline (fit on train only).
        # This function returns raw features with potential NaN values.
    else:
        # Fallback: blacklist approach (less safe, for backwards compat)
        drop_cols = set()
        # Prefix-based banning
        for col in df.columns:
            for prefix in BANNED_PREFIXES:
                if col.startswith(prefix):
                    drop_cols.add(col)
                    break
        # Exact-match banning
        drop_cols |= (BANNED_EXACT & set(df.columns))

        if drop_cols:
            df = df.drop(columns=list(drop_cols))

        # Fill NaN with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        X = df.select_dtypes(include=[np.number])
        feature_names = list(X.columns)

    if len(feature_names) == 0:
        raise ModelingError("No valid features remaining after filtering")

    return X, y_binary.loc[X.index], feature_names


def build_split_masks(
    n_events: int,
    config: dict[str, Any],
    event_positions: np.ndarray | None = None,
    max_horizon: int = 32,
) -> dict[str, np.ndarray]:
    """Build train/val/test index masks with purge + embargo.

    Parameters
    ----------
    n_events : int
        Total number of events.
    config : dict
        Config with splitting section.
    event_positions : np.ndarray, optional
        Positional indices of events (defaults to np.arange(n_events)).
    max_horizon : int
        Maximum label horizon for purging (default 32).

    Returns
    -------
    dict
        {"train": idx_array, "validation": idx_array, "test": idx_array}
        where each array contains positional indices into the original dataset.
    """
    splits = build_train_val_test_splits(n_events, config)
    splitting = config.get("splitting", {})
    embargo_bars = int(splitting.get("embargo_bars", 32))

    # Apply purge + embargo to training set
    train_idx = splits["train"]
    val_start_pos = splits["validation"][0] if len(splits["validation"]) > 0 else n_events

    if len(train_idx) > 0:
        mask = apply_purge_and_embargo(
            train_idx, max_horizon, int(val_start_pos), embargo_bars
        )
        splits["train"] = train_idx[mask]

    return splits


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train models in sequence and select best by validation PR-AUC.

    Order: Logistic Regression → Random Forest → [CatBoost] → [LightGBM]

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data
    config : dict with model settings

    Returns
    -------
    dict
        {"best_model": estimator, "best_name": str, "all_results": dict}
    """
    # Impute NaN using train medians ONLY (no look-ahead)
    impute_medians = {}
    for col in X_train.columns:
        if X_train[col].isna().any():
            impute_medians[col] = float(X_train[col].median())
    
    X_train_imputed = X_train.fillna(impute_medians)
    X_val_imputed = X_val.fillna(impute_medians)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed),
        columns=X_val.columns,
        index=X_val.index,
    )

    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_val_scaled)[:, 1]
    lr_pr_auc = _compute_pr_auc(np.asarray(y_val.values), lr_proba)
    results["logistic_regression"] = {
        "model": lr,
        "scaler": scaler,
        "pr_auc": lr_pr_auc,
    }

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_proba = rf.predict_proba(X_val_scaled)[:, 1]
    rf_pr_auc = _compute_pr_auc(np.asarray(y_val.values), rf_proba)
    results["random_forest"] = {
        "model": rf,
        "scaler": scaler,
        "pr_auc": rf_pr_auc,
    }

    # 3. CatBoost (if available)
    if _HAS_CATBOOST:
        cb = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )
        cb.fit(X_train_scaled, y_train)
        cb_proba = cb.predict_proba(X_val_scaled)[:, 1]
        cb_pr_auc = _compute_pr_auc(np.asarray(y_val.values), cb_proba)
        results["catboost"] = {
            "model": cb,
            "scaler": scaler,
            "pr_auc": cb_pr_auc,
        }

    # 4. LightGBM (if available)
    if _HAS_LIGHTGBM:
        lgbm = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train_scaled, y_train)
        lgbm_proba = lgbm.predict_proba(X_val_scaled)[:, 1]
        lgbm_pr_auc = _compute_pr_auc(np.asarray(y_val.values), lgbm_proba)
        results["lightgbm"] = {
            "model": lgbm,
            "scaler": scaler,
            "pr_auc": lgbm_pr_auc,
        }

    # Select best by PR-AUC
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_entry = results[best_name]

    return {
        "best_model": best_entry["model"],
        "best_scaler": best_entry["scaler"],
        "best_name": best_name,
        "all_results": {k: v["pr_auc"] for k, v in results.items()},
        "impute_medians": impute_medians,
    }


def calibrate_model(
    model: Any,
    scaler: StandardScaler,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "isotonic",
    impute_medians: dict[str, float] | None = None,
) -> CalibratedClassifierCV:
    """Calibrate model probabilities on validation set.

    Parameters
    ----------
    model : trained classifier
    scaler : fitted StandardScaler
    X_val, y_val : validation data for calibration
    method : "isotonic" or "sigmoid" (Platt scaling)
    impute_medians : dict of column -> median for NaN imputation

    Returns
    -------
    CalibratedClassifierCV
        Calibrated wrapper around the model.
    """
    if impute_medians:
        X_val = X_val.fillna(impute_medians)
    X_val_scaled = scaler.transform(X_val)

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit",  # model already trained
    )
    calibrated.fit(X_val_scaled, y_val)

    return calibrated


def evaluate_on_test(
    calibrated_model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scaler: StandardScaler,
    impute_medians: dict[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate calibrated model on test set (ONCE — no tuning after this).

    Parameters
    ----------
    calibrated_model : calibrated classifier
    X_test, y_test : held-out test data
    scaler : fitted scaler
    impute_medians : dict of column -> median for NaN imputation

    Returns
    -------
    dict
        Metrics: brier_score, roc_auc, pr_auc, accuracy, precision, recall, f1
    """
    if impute_medians:
        X_test = X_test.fillna(impute_medians)
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = calibrated_model.predict(X_test_scaled)

    metrics = {
        "brier_score": float(brier_score_loss(y_test.values, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_test.values, y_pred_proba)),
        "pr_auc": float(_compute_pr_auc(np.asarray(y_test.values), y_pred_proba)),
        "accuracy": float(np.mean(y_pred == y_test.values)),
        "precision": float(_safe_precision(np.asarray(y_test.values), y_pred)),
        "recall": float(_safe_recall(np.asarray(y_test.values), y_pred)),
        "f1": float(_safe_f1(np.asarray(y_test.values), y_pred)),
        "test_size": len(y_test),
        "positive_rate": float(y_test.mean()),
    }

    return metrics


def _compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Precision-Recall AUC, handling edge cases.

    Uses :func:`sklearn.metrics.average_precision_score` (the canonical PR-AUC).
    The previous trapezoid implementation integrated over a *decreasing*
    recall axis and could produce impossible negative values (integration fix,
    t16; reported to Agent 6).
    """
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(np.asarray(y_true), np.asarray(y_prob)))
    except ValueError:
        return 0.0


def _safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision with zero-division protection."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def _safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall with zero-division protection."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score with zero-division protection."""
    prec = _safe_precision(y_true, y_pred)
    rec = _safe_recall(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def save_model_artifacts(
    calibrated_model: CalibratedClassifierCV,
    scaler: StandardScaler,
    feature_names: list[str],
    metrics: dict[str, float],
    output_dir: str = "artifacts/models",
    metrics_path: str = "reports/metrics/test_metrics.json",
    impute_medians: dict[str, float] | None = None,
) -> dict[str, str]:
    """Serialize model, calibrator, and metrics to disk.

    Parameters
    ----------
    calibrated_model : trained + calibrated model
    scaler : fitted scaler
    feature_names : list of feature column names
    metrics : evaluation metrics dict
    output_dir : directory for model artifacts
    metrics_path : path for JSON metrics report
    impute_medians : dict of column -> median for NaN imputation (persisted for inference)

    Returns
    -------
    dict
        {"model_path": str, "calibrator_path": str, "metrics_path": str}
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    model_path = os.path.join(output_dir, "model.pkl")
    calibrator_path = os.path.join(output_dir, "calibrator.pkl")

    # Save full pipeline (model + scaler + metadata + imputation)
    pipeline = {
        "model": calibrated_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "impute_medians": impute_medians or {},
    }
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # Save calibrator separately
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrated_model, f)

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": os.path.abspath(model_path),
        "calibrator_path": os.path.abspath(calibrator_path),
        "metrics_path": os.path.abspath(metrics_path),
    }


def walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
    initial_train_size: int | None = None,
    step_size: int | None = None,
    validation_size: int | None = None,
) -> dict[str, Any]:
    """Run expanding-window walk-forward validation per guide §22.3.

    For each fold:
    1. Split into train (expanding) + validation with purge+embargo.
    2. Train models (LR → RF → [CatBoost] → [LightGBM]).
    3. Select best by validation PR-AUC.
    4. Calibrate on validation set.
    5. Evaluate calibrated model on validation set (OOS metric).

    Aggregates metrics across folds with stability statistics (mean/std).

    Parameters
    ----------
    X : DataFrame of features
    y : Series of binary labels
    config : dict with splitting/model settings
    initial_train_size : optional override for initial training window
    step_size : optional override for expansion step
    validation_size : optional override for validation window size

    Returns
    -------
    dict
        {
            "folds": list of per-fold results,
            "aggregate": {metric: {"mean": float, "std": float}},
            "n_folds": int,
        }
    """
    from src.modeling.split import walk_forward_folds

    splitting = config.get("splitting", {})
    embargo_bars = int(splitting.get("embargo_bars", 32))
    horizon_bars = int(splitting.get("embargo_bars", 32))  # same as max horizon

    n_events = len(X)

    # Default fold parameters if not provided
    if initial_train_size is None:
        initial_train_size = max(round(n_events * 0.4), 100)  # at least 40% or 100
    if step_size is None:
        step_size = max(round(n_events * 0.1), 50)  # 10% steps
    if validation_size is None:
        validation_size = max(round(n_events * 0.1), 50)  # 10% val

    # Generate folds
    folds = walk_forward_folds(
        n_events, initial_train_size, step_size, validation_size,
        embargo_bars, horizon_bars,
    )

    fold_results: list[dict[str, object]] = []

    for fold_info in folds:
        fold_idx = fold_info["fold"]
        train_idx = fold_info["train"]
        val_idx = fold_info["validation"]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue  # Skip empty folds

        # Extract fold data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Train models on this fold
        try:
            train_result = train_models(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                config,
            )
        except Exception as e:
            fold_results.append({
                "fold": fold_idx,
                "error": str(e),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            })
            continue

        # Get best model and calibrate
        best_model = train_result["best_model"]
        best_scaler = train_result["best_scaler"]
        best_name = train_result["best_name"]
        # Impute medians fit on TRAIN ONLY (returned by train_models) —
        # must be passed to calibration/evaluation so NaN-safe inference
        # is consistent with the CLI path (F3-WF fix, t32).
        impute_medians_fold = train_result.get("impute_medians", {})

        calibrated = calibrate_model(
            best_model, best_scaler,
            X_val_fold, y_val_fold,
            method=config.get("model", {}).get("probability_calibration", "isotonic"),
            impute_medians=impute_medians_fold,
        )

        # Evaluate on validation (OOS for this fold)
        fold_metrics_raw = evaluate_on_test(
            calibrated, X_val_fold, y_val_fold, best_scaler,
            impute_medians=impute_medians_fold,
        )
        fold_metrics: dict[str, object] = dict(fold_metrics_raw)  # widen type
        fold_metrics["fold"] = fold_idx
        fold_metrics["model"] = best_name
        fold_metrics["train_size"] = len(train_idx)
        fold_metrics["val_size"] = len(val_idx)

        fold_results.append(fold_metrics)

    # Aggregate metrics across folds
    aggregate = _aggregate_fold_metrics(fold_results)

    return {
        "folds": fold_results,
        "aggregate": aggregate,
        "n_folds": len(fold_results),
    }


def _aggregate_fold_metrics(fold_results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean/std of metrics across walk-forward folds.

    Parameters
    ----------
    fold_results : list of per-fold metric dicts

    Returns
    -------
    dict
        {metric_name: {"mean": float, "std": float, "min": float, "max": float}}
    """
    if not fold_results:
        return {}

    # Collect numeric metrics (exclude non-numeric keys like "fold", "model")
    metric_keys = [
        k for k in fold_results[0].keys()
        if k not in ("fold", "model", "error") and isinstance(fold_results[0].get(k), (int, float))
    ]

    aggregate = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r and r[key] is not None]
        if values:
            arr = np.array(values, dtype=float)
            aggregate[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n_folds": len(values),
            }

    return aggregate


def predict_with_model(
    model_path: str,
    raw_row: pd.DataFrame | pd.Series,
) -> dict[str, float]:
    """Inference helper: load model.pkl, impute NaN, scale, predict.

    Parameters
    ----------
    model_path : path to model.pkl
    raw_row : single row DataFrame or Series with raw feature values (may have NaN)

    Returns
    -------
    dict
        {"probability": float, "prediction": int}
    """
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline["model"]
    scaler = pipeline["scaler"]
    feature_names = pipeline["feature_names"]
    impute_medians = pipeline.get("impute_medians", {})

    # Ensure row is DataFrame
    if isinstance(raw_row, pd.Series):
        row_df = raw_row.to_frame().T
    else:
        row_df = raw_row.copy()

    # Select only model features
    row_df = row_df[feature_names]

    # Impute NaN using persisted medians
    if impute_medians:
        row_df = row_df.fillna(impute_medians)

    # Scale
    row_scaled = scaler.transform(row_df)

    # Predict
    proba = model.predict_proba(row_scaled)[0, 1]
    pred = int(model.predict(row_scaled)[0])

    return {"probability": float(proba), "prediction": pred}

