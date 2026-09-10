"""Equity-curve figure for the final deliverables (guide §33).

Computes the cumulative net result (in R units, cost-adjusted) of the primary
trade (2R / h16) ordered by event time — the honest equity curve of strategy B
(confirmed events, causal group_rule="first").  Written by Agent 0 (t16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_MPL = False


def plot_equity_curve(
    labeled: pd.DataFrame,
    output_path: str = "reports/figures/equity_curve_r.png",
    primary_outcome: str = "outcome_2r_h16",
    net_col: str = "net_result_r",
) -> str:
    """Plot cumulative net R over event time.

    Parameters
    ----------
    labeled :
        Labeled dataset (Pipeline 3 output) with ``event_time`` and a net
        result column (primary trade only; ambiguous rows count as 0).
    output_path :
        Where to save the PNG.
    primary_outcome / net_col :
        Column names to source the primary outcome and net P&L.

    Returns
    -------
    Absolute path of the saved figure.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib required for the equity-curve figure")

    df = labeled.sort_values("event_time").copy()
    if net_col not in df.columns:
        raise ValueError(f"Missing net-result column '{net_col}'")
    # Ambiguous primary outcomes carry NaN/ambiguous; treat as zero P&L.
    net = np.asarray(pd.to_numeric(df[net_col], errors="coerce").fillna(0.0), dtype=float)
    equity = np.cumsum(net)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["event_time"], equity, linewidth=1.2, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Equity curve — primary outcome ({primary_outcome}), net R after costs"
    )
    ax.set_xlabel("Event time (UTC)")
    ax.set_ylabel("Cumulative net result (R)")
    ax.grid(alpha=0.3)

    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return os.path.abspath(output_path)