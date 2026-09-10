"""
trendline.py — Trendline fitting infra for classical patterns (§11 P3/P4).

Agent 4 deliverable (REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 P3/P4, §13 Agent 4).

Wedge (P3) and Head & Shoulders (P4) are defined by *lines*: the wedge by two
converging channel lines (upper + lower), Head & Shoulders by a neckline.
This module is the single shared implementation of that geometry:

* :class:`Trendline` — a linear-regression line fit on pivot points
  (``np.polyfit`` degree 1) with a ``.at(bar)`` evaluator and an ATR-normalised
  fit error (:meth:`Trendline.max_error_atr`).
* :func:`fit_trendline` — fit a line through ``(bar, price)`` pivot points.
* :func:`channel_width` / :func:`convergence_ratio` — the wedge's converging
  channel metrics.
* :func:`closes_inside_channel` — how well closes stay inside the channel
  (with an ATR tolerance) — the main anti-noise gate for wedges.

Causality contract (same stance as ``swing_detector.py`` §3.2): a line is only
fit on pivots whose ``known_at_bar`` has passed; the detector sets
``detect_time`` at (or after) the LAST pivot's known bar and never evaluates
``Trendline.at(bar)`` for a bar past the event's ``known_at``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trendline:
    """A straight line ``price = slope * bar + intercept`` fit on pivot points.

    ``support`` keeps the exact ``(bar, price)`` points used for the fit —
    the ATR tolerance of the fit (:meth:`max_error_atr`) and the channel
    anchor checks (wedge hysteresis) read it directly.
    """

    slope: float
    intercept: float
    support: tuple[tuple[int, float], ...]

    def at(self, bar: int) -> float:
        """Line value at *bar* (0-based bar index of the input frame)."""
        return self.slope * bar + self.intercept

    def max_error_atr(self, atr_value: float) -> float:
        """Max |price - line(bar)| over the fit points, in ATR units.

        The "tolerance ATR" gate of the spec: a genuine trendline touches its
        pivot anchors within a small ATR multiple; a fit that needs > 1-2 ATR
        of deviation to cover its points is not a line, it is noise.
        """
        if atr_value <= 0.0:
            return float("inf")
        err = 0.0
        for bar, price in self.support:
            err = max(err, abs(price - self.at(bar)))
        return err / atr_value

    @property
    def r2(self) -> float:
        """R² of the linear fit (1.0 = perfect line)."""
        yy = np.asarray([p for _, p in self.support], dtype=float)
        if len(yy) < 2:
            return 0.0
        resid = sum((p - self.at(b)) ** 2 for b, p in self.support)
        ss = float(np.sum((yy - yy.mean()) ** 2))
        if ss == 0.0:
            return 1.0
        return 1.0 - resid / ss


def fit_trendline(points: Sequence[tuple[int, float]]) -> Trendline:
    """Linear-regression line through pivot (bar, price) points.

    Requires at least 2 distinct bars (raises ``ValueError`` otherwise).
    Uses degree-1 polynomial least squares (``np.polyfit``).
    """
    pts = tuple(points)
    if len(pts) < 2:
        raise ValueError("fit_trendline needs at least 2 pivot points")
    bars = np.asarray([b for b, _ in pts], dtype=float)
    prices = np.asarray([p for _, p in pts], dtype=float)
    slope, intercept = np.polyfit(bars, prices, 1)
    return Trendline(float(slope), float(intercept), pts)


def channel_width(upper: Trendline, lower: Trendline, bar: int) -> float:
    """Vertical distance between *upper* and *lower* lines at *bar*."""
    return upper.at(bar) - lower.at(bar)


def convergence_ratio(
    upper: Trendline, lower: Trendline, bar_a: int, bar_b: int
) -> float:
    """How much the channel narrowed from *bar_a* to *bar_b* (0..1+).

    ``1 - width(bar_b) / width(bar_a)`` — a genuine wedge converges
    (positive ratio); 0 or negative means parallel / diverging.
    """
    wa = channel_width(upper, lower, bar_a)
    if wa <= 0.0:
        return 0.0
    wb = channel_width(upper, lower, bar_b)
    return 1.0 - wb / wa


def closes_inside_channel(
    df: pd.DataFrame,
    upper: Trendline,
    lower: Trendline,
    start: int,
    end: int,
    tol_atr: float,
    atr_at: float,
) -> float:
    """Fraction of closes in [start, end) contained in the channel, with a
    symmetrical ATR tolerance band around both lines.

    Landmark-free gate: structureless noise fails it easily (closes wander
    outside a fitted channel), a planted wedge keeps ≥ 80 % inside.
    """
    if end <= start:
        return 0.0
    closes = df["close"].to_numpy(dtype=float)
    inside = 0
    total = 0
    for j in range(start, end):
        total += 1
        lo = lower.at(j) - tol_atr * atr_at
        hi = upper.at(j) + tol_atr * atr_at
        if lo <= closes[j] <= hi:
            inside += 1
    return inside / total if total else 0.0