"""Calibration curve visualization for ML models.

Generates reliability diagrams (calibration curves) showing how well
predicted probabilities match observed frequencies.  Evaluates with
Brier score and Expected Calibration Error (ECE).
"""

from __future__ import annotations

import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class VisualizationError(RuntimeError):
    """Raised when visualization cannot be generated."""


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Brier score and Expected Calibration Error.

    Parameters
    ----------
    y_true : array of true labels (0/1)
    y_prob : array of predicted probabilities
    n_bins : number of bins for ECE calculation

    Returns
    -------
    dict
        {"brier_score": float, "ece": float}
    """
    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        bin_count = mask.sum()
        if bin_count > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += (bin_count / total) * abs(avg_pred - avg_true)

    return {"brier_score": brier, "ece": float(ece)}


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    output_path: str = "reports/figures/calibration_curve.png",
    n_bins: int = 10,
) -> str:
    """Generate calibration curve (reliability diagram).

    Parameters
    ----------
    y_true : array of true binary labels
    y_prob : array of predicted probabilities [0, 1]
    model_name : name for the plot title
    output_path : path to save the PNG
    n_bins : number of bins for the histogram

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    if not _HAS_MPL:
        raise VisualizationError("matplotlib is required for visualization")

    if len(y_true) != len(y_prob):
        raise VisualizationError("y_true and y_prob must have same length")

    # Compute calibration metrics
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins)

    # Compute bin statistics for the curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_frac_pos = []
    bin_counts = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        count = mask.sum()
        if count > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_frac_pos.append(y_true[mask].mean())
            bin_counts.append(count)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    if bin_centers:
        ax1.plot(bin_centers, bin_frac_pos, "s-", color="steelblue",
                label=model_name, markersize=8)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Curve ({model_name})")
    ax1.legend(loc="upper left")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Add metrics text
    textstr = f"Brier Score: {metrics['brier_score']:.4f}\nECE: {metrics['ece']:.4f}"
    ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Bottom panel: Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), color="steelblue",
            alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)
