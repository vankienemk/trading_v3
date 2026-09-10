"""Pipeline 4 — Train model CLI (guide sections 26.4 / 20 / 22).

Owned by Agent 0 (integration).  Orchestrates:

    labeled event dataset
      → causal feature selection + binary labels (Agent 6, t14)
      → time-based train/val/test split with purge + embargo
      → walk-forward validation (expanding folds, guide §22.3)
      → baseline models LR -> RF -> [CatBoost/LightGBM]
      → calibration (Platt/Isotonic, fit on validation only)
      → single final test evaluation (rule 30.5/30.6)
      → save artifacts (model.pkl, calibrator.pkl, test_metrics.json)
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config, resolve_path, set_global_config


def train_model_cli(argv: list[str] | None = None) -> int:
    """``xauusd-train``: labeled dataset → trained model (Pipeline 4).

    Reads the labeled dataset written by ``xauusd-dataset``, runs the modeling
    pipeline and persists model artifacts + metrics.  Exits 0 on success, 1 on
    any error.
    """
    parser = argparse.ArgumentParser(
        prog="xauusd-train",
        description="Train and evaluate the scoring model (walk-forward + "
        "calibration).",
    )
    parser.add_argument(
        "--config",
        default="baseline.yaml",
        help="Base config file under configs/ (default: baseline.yaml).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional config file(s) to deep-merge (repeatable).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Labeled dataset Parquet (default: artifacts/datasets/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Model artifact output directory (default: artifacts/models/).",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics JSON output path (default: reports/metrics/test_metrics.json).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.override or None)
        set_global_config(cfg)

        from ..data.loader import read_parquet
        from ..modeling.train import (
            build_split_masks,
            calibrate_model,
            evaluate_on_test,
            prepare_features_and_labels,
            save_model_artifacts,
            train_models,
        )

        dataset_path = args.dataset or resolve_path(
            cfg, "artifacts/datasets/liquidity_sweep_events.parquet"
        )
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Labeled dataset not found at {dataset_path}; run xauusd-dataset first."
            )
        dataset = read_parquet(dataset_path)

        target_col = cfg["model"].get("target", "outcome_2r_h16")
        X, y, feature_names = prepare_features_and_labels(dataset, target_col)

        n = len(X)
        splits = build_split_masks(n, cfg)
        X_tr, X_va, X_te = X.iloc[splits["train"]], X.iloc[splits["validation"]], X.iloc[splits["test"]]
        y_tr, y_va, y_te = y.iloc[splits["train"]], y.iloc[splits["validation"]], y.iloc[splits["test"]]

        if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
            raise ValueError(
                f"Empty split (train={len(X_tr)}, val={len(X_va)}, "
                f"test={len(X_te)}); dataset too small for config split"
            )

        # --- train + calibrate on train/validation only (rule 30.6) ---
        train_result = train_models(X_tr, y_tr, X_va, y_va, cfg)
        best = train_result["best_model"]
        best_scaler = train_result["best_scaler"]
        impute_medians = train_result.get("impute_medians", {})
        cal_method = cfg["model"].get("probability_calibration", "isotonic")
        calibrator = calibrate_model(best, best_scaler, X_va, y_va, method=cal_method, impute_medians=impute_medians)

        # --- single final evaluation on the untouched test set ---
        metrics = evaluate_on_test(calibrator, X_te, y_te, best_scaler, impute_medians=impute_medians)

        # serialize artifacts
        out_dir = args.output_dir or resolve_path(cfg, "artifacts/models")
        metrics_path = args.metrics or resolve_path(
            cfg, "reports/metrics/test_metrics.json"
        )
        save_model_artifacts(
            calibrator,
            best_scaler,
            feature_names,
            metrics,
            output_dir=out_dir,
            metrics_path=metrics_path,
            impute_medians=impute_medians,
        )

        # walk-forward OOS summary (if modeling module exposes it)
        try:
            from ..modeling.train import walk_forward_validation

            wf = walk_forward_validation(
                X, y, cfg,
                initial_train_size=len(splits["train"]),
                step_size=max(len(splits["train"]) // 5, 50),
                validation_size=len(splits["validation"]),
            )
            print(
                f"[xauusd-train] walk-forward: {wf['n_folds']} folds, "
                f"pr_auc mean={wf.get('aggregate', {}).get('pr_auc', {}).get('mean', 'n/a')}"
            )
        except Exception as wexc:  # walk-forward is best-effort; split path is authoritative
            print(f"[xauusd-train] walk-forward skipped: {wexc}", file=sys.stderr)

        print(f"[xauusd-train] OK: best={train_result['best_name']} pr_auc={metrics['pr_auc']:.4f}")
        print(f"[xauusd-train] artifacts -> {out_dir}")
        print(f"[xauusd-train] metrics -> {metrics_path}")
        return 0
    except Exception as exc:  # CLI boundary: report and exit
        print(f"[xauusd-train] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(train_model_cli())