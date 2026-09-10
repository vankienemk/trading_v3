"""
dataset.py -- Dataset builder + §6.3 research gates for the wedge plugins.

Reuses the forward fixed-horizon labeling and the §6.3 gate machinery from
the double-pattern family verbatim (same semantics, same thresholds — the
DoD demands "research gates như Agent 3"):

* ``load_xauusd_m15`` — full-history XAUUSD M15 loader (2018-2026);
* ``label_events`` / ``labeled_to_frame`` — forward MFE/MAE fixed-horizon
  labeling (strictly AFTER entry, no look-ahead);
* ``run_gates`` / ``assert_gates`` / ``render_gate_report`` — the §6.3 gates:
  n ≥ 300, n_oos ≥ 100, balance, ESS ≥ 60 %, OOS PF bootstrap CI > 1.0,
  PR-AUC ≥ 1.05*baseline, ≥ 3 WF folds PF ≥ 0.8.

The gate features are computed on the wedge events' causal attributes
(``pivot_known_at_bar`` / ``left_len`` / ``right_len`` / ``depth_atr`` /
``low_offset_atr`` / ``structure_levels["neckline"]``), all of which are
available at the detect bar (§3.4).
"""

from __future__ import annotations

from research.patterns.double_bottom.dataset import (
    GateReport,
    LabeledEvent,
    assert_gates,
    effective_sample_size,
    gate_features,
    label_events,
    labeled_to_frame,
    load_xauusd_m15,
    profit_factor,
    render_gate_report,
    run_gates,
    split_train_oos,
    walk_forward_pfs,
)

__all__ = [
    "GateReport",
    "LabeledEvent",
    "assert_gates",
    "effective_sample_size",
    "gate_features",
    "label_events",
    "labeled_to_frame",
    "load_xauusd_m15",
    "profit_factor",
    "render_gate_report",
    "run_gates",
    "split_train_oos",
    "walk_forward_pfs",
]