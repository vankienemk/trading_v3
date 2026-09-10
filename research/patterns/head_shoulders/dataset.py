"""
dataset.py -- Dataset builder + §6.3 research gates for the H&S plugins.

Reuses the forward fixed-horizon labeling and the §6.3 gate machinery from
the double-pattern family verbatim (same semantics, same thresholds — the
DoD demands "research gates như Agent 3").

The gate features are computed on the H&S events' causal attributes
(``pivot_known_at_bar`` / ``left_len`` / ``right_len`` / ``head_depth_atr``
/ ``shoulder_offset_atr`` / ``structure_levels["neckline"]``), all of which
are available at the detect bar (§3.4).
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