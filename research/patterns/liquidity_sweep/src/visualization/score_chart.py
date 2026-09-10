"""Visualization for rule-based scoring results (guide section 19.4).

Generates score bucket performance charts showing win rate, profit factor,
and event distribution across score buckets.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from src.scoring.rule_score import compute_bucket_report


class VisualizationError(RuntimeError):
    """Raised when visualization cannot be generated."""


def plot_score_bucket_performance(
    scored_events: pd.DataFrame,
    output_path: str = "reports/figures/score_bucket_performance.png",
    config: dict[str, Any] | None = None,
) -> str:
    """Generate score bucket performance chart.

    Parameters
    ----------
    scored_events : DataFrame
        Event table with rule_score and outcome columns computed.
    output_path : str
        Path to save the PNG figure.
    config : dict, optional
        Configuration dict (used for title/customization).

    Returns
    -------
    str
        Absolute path to the saved figure.

    Raises
    ------
    VisualizationError
        If matplotlib is not installed or required columns are missing.
    """
    if not _HAS_MPL:
        raise VisualizationError(
            "matplotlib is required for visualization. Install with: "
            "pip install matplotlib"
        )

    # Validate required columns
    required = ["rule_score", "outcome_2r_h16", "net_result_r"]
    missing = [c for c in required if c not in scored_events.columns]
    if missing:
        raise VisualizationError(f"Missing required columns: {missing}")

    # Compute bucket report
    report = compute_bucket_report(scored_events)
    if len(report) == 0:
        raise VisualizationError("No events to plot after bucketing")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rule-Based Score Bucket Performance", fontsize=14, fontweight="bold")

    buckets = report["bucket"].tolist()
    x_pos = range(len(buckets))

    # Plot 1: Win Rate by Bucket
    ax1 = axes[0, 0]
    win_rates = report["win_rate"].fillna(0).tolist()
    bars1 = ax1.bar(x_pos, [wr * 100 for wr in win_rates], color="steelblue", alpha=0.8)
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels(buckets, rotation=45, ha="right")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate by Score Bucket")
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax1.legend()
    # Add value labels on bars
    for bar, val in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)

    # Plot 2: Profit Factor by Bucket
    ax2 = axes[0, 1]
    pf_values = report["profit_factor"].fillna(0).tolist()
    colors_pf = ["green" if pf >= 1 else "red" for pf in pf_values]
    bars2 = ax2.bar(x_pos, pf_values, color=colors_pf, alpha=0.8)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(buckets, rotation=45, ha="right")
    ax2.set_ylabel("Profit Factor")
    ax2.set_title("Profit Factor by Score Bucket")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Break-even")
    ax2.legend()
    for bar, val in zip(bars2, pf_values):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Plot 3: Average Net Result (R) by Bucket
    ax3 = axes[1, 0]
    net_results = report["avg_net_result_r"].fillna(0).tolist()
    colors_net = ["green" if nr >= 0 else "red" for nr in net_results]
    bars3 = ax3.bar(x_pos, net_results, color=colors_net, alpha=0.8)
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels(buckets, rotation=45, ha="right")
    ax3.set_ylabel("Avg Net Result (R)")
    ax3.set_title("Average Net Result by Score Bucket")
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    for bar, val in zip(bars3, net_results):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.01 if height >= 0 else -0.01
        ax3.text(bar.get_x() + bar.get_width() / 2., height + offset,
                f"{val:.3f}R", ha="center", va=va, fontsize=8)

    # Plot 4: Event Count Distribution
    ax4 = axes[1, 1]
    counts = report["event_count"].tolist()
    bars4 = ax4.bar(x_pos, counts, color="coral", alpha=0.8)
    ax4.set_xticks(list(x_pos))
    ax4.set_xticklabels(buckets, rotation=45, ha="right")
    ax4.set_ylabel("Event Count")
    ax4.set_title("Event Distribution by Score Bucket")
    for bar, val in zip(bars4, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                str(val), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)
