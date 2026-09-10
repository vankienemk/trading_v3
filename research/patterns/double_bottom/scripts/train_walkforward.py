"""
train_walkforward.py — train DB/DT walk-forward models on XAUUSD (+FW/HS explore).

Runs the full §6.2 research → train → artifacts leg for the classical
pattern plugins on the trading_v3 XAUUSD M15 dataset (self-contained: no
trading_live paths):

    detect (plugin detector) → label (forward MFE fixed-horizon, causal)
    → build causal feature matrix (§3.4, bars ≤ confirm_bar)
    → §6.3 gates → walk-forward training (expanding window, purge+embargo)
    → final tier-2 meta-model + isotonic calibration (fit on train only)
    → artifacts (model.pkl / calibrator.pkl / features.json / summary)

Artifacts land inside this task's inScope plugin trees:

    double_bottom          → research/patterns/double_bottom/artifacts/
    double_top             → research/patterns/double_top/artifacts/
    falling_wedge          → research/patterns/falling_wedge/artifacts/
    head_shoulders         → research/patterns/head_shoulders/artifacts/
    inverse_head_shoulders → research/patterns/head_shoulders/artifacts/
                             (IHS shares the HS family root; its own plugin
                             dir is outside t1's inScope)

Usage:
    /tmp/ptv2_venv/bin/python research/patterns/double_bottom/scripts/train_walkforward.py
    /tmp/ptv2_venv/bin/python research/patterns/double_bottom/scripts/train_walkforward.py --only double_bottom
    /tmp/ptv2_venv/bin/python research/patterns/double_bottom/scripts/train_walkforward.py --only falling_wedge --gates-off
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from research.core.config_hash import compute_config_hash
from research.core.walkforward_trainer import (
    TrainFrame,
    TrainSummary,
    WalkForwardReport,
    _profit_factor,
    binary_metrics,
    build_feature_frame,
    fit_final_model,
    render_walk_forward_report,
    run_walk_forward,
    sequential_split,
    utc_now_iso,
    write_model_artifacts,
)
from research.patterns.double_bottom.dataset import (
    label_events,
    load_xauusd_m15,
    render_gate_report,
    run_gates,
)
from research.patterns.double_bottom.detector import DoubleBottomDetector
from research.patterns.double_top.detector import DoubleTopDetector
from research.patterns.falling_wedge.detector import FallingWedgeDetector
from research.patterns.head_shoulders.detector import HeadShouldersDetector
from research.patterns.inverse_head_shoulders.detector import (
    InverseHeadShouldersDetector,
)

#: trading_v3 package root (this file sits one level deeper than dataset.py:
#: .../trading_v3/research/patterns/double_bottom/scripts/)
_TV3_ROOT = Path(__file__).resolve().parents[4]

#: (pattern → detector class, extra attribute features, artifact root,
#:   feature schema version)
_PATTERNS: dict[str, dict[str, Any]] = {
    "double_bottom": {
        "detector": DoubleBottomDetector,
        "extra_attrs": (),
        "artifact_root": Path("research/patterns/double_bottom/artifacts"),
        "feature_schema_version": "double-v1.0",
    },
    "double_top": {
        "detector": DoubleTopDetector,
        "extra_attrs": (),
        "artifact_root": Path("research/patterns/double_top/artifacts"),
        "feature_schema_version": "double-v1.0",
    },
    "falling_wedge": {
        "detector": FallingWedgeDetector,
        "extra_attrs": ("width_atr", "convergence_ratio", "interior_ratio"),
        "artifact_root": Path("research/patterns/falling_wedge/artifacts"),
        "feature_schema_version": "wedge-v1.0",
    },
    "head_shoulders": {
        "detector": HeadShouldersDetector,
        "extra_attrs": ("head_depth_atr", "shoulder_offset_atr", "neckline_offset_atr"),
        "artifact_root": Path("research/patterns/head_shoulders/artifacts"),
        "feature_schema_version": "hs-v1.0",
    },
    "inverse_head_shoulders": {
        "detector": InverseHeadShouldersDetector,
        "extra_attrs": ("head_depth_atr", "shoulder_offset_atr", "neckline_offset_atr"),
        "artifact_root": Path("research/patterns/head_shoulders/artifacts"),
        "feature_schema_version": "hs-v1.0",
    },
}

#: §6.3 sample-size gates (H&S explores below them honestly).
GATE_MIN_TOTAL = 300
GATE_MIN_OOS = 100


def _rel(p: Path) -> str:
    return str(p.relative_to(_TV3_ROOT))


def train_one(
    df: pd.DataFrame,
    pattern: str,
    spec: dict[str, Any],
    run_gates_flag: bool,
) -> tuple[TrainSummary, WalkForwardReport]:
    detector = spec["detector"]()
    events = detector.detect(df, detector.get_default_config())
    labeled = label_events(df, events)
    if not labeled:
        raise RuntimeError(f"{pattern}: no labeled events on XAUUSD M15")
    horizon = int(labeled[0].horizon_bars)
    target_r = 1.25

    X, feature_names = build_feature_frame(
        df, events, extra_attr_features=spec["extra_attrs"]
    )
    # Align events → labeled rows (label reads strictly after entry bar).
    ids = [e.event_id for e in events if e.event_id in {le.event_id for le in labeled}]
    X = X.loc[ids]
    y = pd.Series(
        [int(le.label) for le in labeled if le.event_id in X.index],
        index=[le.event_id for le in labeled if le.event_id in X.index],
        dtype="int64",
    )
    y = y.reindex(X.index)
    entry = pd.Series(
        [int(le.entry_bar) for le in labeled if le.event_id in X.index],
        index=[le.event_id for le in labeled if le.event_id in X.index],
        dtype="int64",
    ).reindex(X.index)
    pnl = pd.Series(
        [float(le.pnl_r(target_r)) for le in labeled if le.event_id in X.index],
        index=[le.event_id for le in labeled if le.event_id in X.index],
        dtype="float64",
    ).reindex(X.index)
    frame = TrainFrame(X=X, y=y, entry_bar=entry, horizon_bars=horizon, pnl_r=pnl)

    # --- §6.3 gates (pattern dataset machinery; logistic fit on train only)
    gate_md = ""
    gate_passed = False
    if run_gates_flag:
        report = run_gates(df, events)
        gate_passed = report.all_passed()
        gate_md = render_gate_report(report)

    # --- walk-forward + final model (fit on train split only) -------------
    train_f, oos_f = sequential_split(frame, oos_frac=0.40, purge_bars=96, embargo_bars=24)
    wf = run_walk_forward(frame, n_folds=5, purge_bars=96, embargo_bars=24, seed=42)
    final = fit_final_model(train_f, seed=42)
    prob_oos = pd.Series(
        final.predict_proba(oos_f.X.to_numpy(dtype=float))[:, 1],
        index=oos_f.y.index,
    )
    oos_metrics = binary_metrics(oos_f.y, prob_oos)

    # Model-gated PF: cutoff = train median probability; apply on OOS pnl.
    if oos_f.pnl_r is not None:
        train_prob = pd.Series(
            final.predict_proba(train_f.X.to_numpy(dtype=float))[:, 1],
            index=train_f.y.index,
        )
        cutoff = float(train_prob.median())
        gated = prob_oos >= cutoff
        if int(gated.sum()) >= 5:
            oos_metrics["oos_pf_rule"] = _profit_factor(oos_f.pnl_r)
            oos_metrics["oos_pf_model_gated"] = _profit_factor(oos_f.pnl_r[gated])
            oos_metrics["model_gate_cutoff"] = cutoff
            oos_metrics["model_gated_n"] = float(gated.sum())

    # --- artifacts --------------------------------------------------------
    root = _TV3_ROOT / spec["artifact_root"]
    model_dir = root / "models" / f"{pattern}_xauusd_m15_v1"
    datasets_dir = root / "datasets"
    reports_dir = root / "reports"

    feat_df = X.copy()
    feat_df["label"] = y
    feat_df["entry_bar"] = entry
    feat_df["pnl_r"] = pnl
    datasets_dir.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(datasets_dir / f"{pattern}_xauusd_m15_features.parquet")

    artifact_paths = {
        "model": _rel(model_dir / "model.pkl"),
        "calibrator": _rel(model_dir / "calibrator.pkl"),
        "features": _rel(model_dir / "features.json"),
        "train_summary": _rel(model_dir / "train_summary.json"),
        "feature_dataset": _rel(datasets_dir / f"{pattern}_xauusd_m15_features.parquet"),
    }

    notes = (
        f"Tier-2 meta-model ({pattern}): RandomForest + isotonic calibration, "
        f"fit on train split only; {len(feature_names)} causal features at "
        f"detect/confirm bars (≤ known_at); label = forward MFE "
        f"{horizon}-bar, {target_r}R target."
    )
    if pattern in ("falling_wedge", "head_shoulders", "inverse_head_shoulders"):
        notes += (
            f" EXPLORE status: sample n={len(labeled)} "
            f"(gate ≥ {GATE_MIN_TOTAL}); gate_passed={gate_passed} — kept "
            f"strict anti-noise detector per Agent 4 DoD; model registered as "
            f"'trained', promotion deferred until t2 verdict."
        )
    cfg_hash = compute_config_hash(detector.get_default_config())
    model_id = f"{pattern}_xauusd_m15_v1"
    summary = TrainSummary(
        model_id=model_id,
        pattern_name=pattern,
        symbol="XAUUSD",
        timeframe="M15",
        feature_schema_version=str(spec["feature_schema_version"]),
        lifecycle_state="validated" if gate_passed else "trained",
        gate_passed=gate_passed,
        calibrated=True,
        config_hash=cfg_hash,
        trained_at=utc_now_iso(),
        horizon_bars=horizon,
        target_r=target_r,
        n_events=len(labeled),
        n_train=len(train_f),
        n_oos=len(oos_f),
        metrics=oos_metrics,
        artifact_paths=artifact_paths,
        notes=notes,
    )
    write_model_artifacts(model_dir, final, feature_names, summary)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rep_md = render_walk_forward_report(summary, wf, oos_metrics, gate_md)
    (reports_dir / f"walk_forward_report_{pattern}_xauusd_m15.md").write_text(
        rep_md + "\n"
    )
    return summary, wf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(_PATTERNS),
        help="train a single pattern (default: all five)",
    )
    parser.add_argument(
        "--gates-off",
        action="store_true",
        help="skip the §6.3 gate report (faster explore runs)",
    )
    args = parser.parse_args(argv)

    df = load_xauusd_m15()
    print(f"[data] XAUUSD M15: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
    targets = [args.only] if args.only else sorted(_PATTERNS)
    summaries: list[TrainSummary] = []
    for pattern in targets:
        print(f"[train] {pattern} ...")
        summary, _wf = train_one(df, pattern, _PATTERNS[pattern], not args.gates_off)
        summaries.append(summary)
        m = summary.metrics
        print(
            f"  -> {summary.model_id}: gate_passed={summary.gate_passed} "
            f"OOS PR-AUC={float(m.get('pr_auc', float('nan'))):.4f} "
            f"ROC-AUC={float(m.get('roc_auc', float('nan'))):.4f} "
            f"n_train={summary.n_train} n_oos={summary.n_oos}"
        )
    print("\n=== train summaries ===")
    print(json.dumps([s.to_dict() for s in summaries], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())