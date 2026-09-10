"""
swing_detector.py — Shared swing/pivot infra for classical patterns (§3.2).

Agent 3 deliverable (REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §3.2, §11 P1/P2).

Every classical price pattern (Double Bottom/Top, Head & Shoulders, Wedge…)
is built on swing points.  The spec's **Right-Bar Rule** (§3.2) states that a
pivot high/low at bar ``t`` only *exists causally* at bar ``t + right_bars``
— the bars needed to confirm that neither neighbour is more extreme.  This
module is the single shared implementation of that rule:

* :class:`SwingPoint` — a pivot carrying its causal
  ``known_at_bar = bar + right_bars``.
* :class:`SwingDetector` — left/right bars configurable (spec §11 field
  "swing formula (left/right bars + price source)").

Contract (used by double_bottom / double_top and later Wedge / H&S):

* price source is fixed: pivot highs come from the ``high`` series, pivot
  lows from the ``low`` series (documented in each PATTERN_SPECS.md).
* a pattern detector MUST NOT reference a pivot before ``known_at_bar``;
  the runtime causality validator (``research.core.causal_checks``) backs
  this up at the event level.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    """One confirmed swing pivot.

    ``kind`` is ``"L"`` for a swing low (from the low series) or ``"H"`` for
    a swing high (from the high series).  ``known_at_bar`` is the FIRST bar
    at which the pivot is causally known: ``bar + right_bars`` (§3.2).
    """

    bar: int          # 0-based bar index of the extreme
    price: float      # pivot price (low for L, high for H)
    kind: str         # "L" | "H"
    known_at_bar: int  # = bar + right_bars — causal existence stamp

    def __post_init__(self) -> None:
        if self.kind not in ("L", "H"):
            raise ValueError(f"swing kind must be 'L'|'H', got {self.kind!r}")
        if self.known_at_bar < self.bar:
            raise ValueError(
                f"known_at_bar {self.known_at_bar} < bar {self.bar} "
                f"— pivot cannot be known before it forms"
            )


class SwingDetector:
    """Configurable swing detector honouring the §3.2 Right-Bar Rule.

    A pivot at bar ``t`` is detected when its value is the strict extreme of
    the window ``[t - left_bars, t + right_bars]`` and strictly more extreme
    than the window edges (so plateaus do not produce spurious pivots).
    """

    def __init__(self, left_bars: int = 3, right_bars: int = 3) -> None:
        if left_bars < 1 or right_bars < 1:
            raise ValueError("left_bars/right_bars must be >= 1")
        self.left_bars = int(left_bars)
        self.right_bars = int(right_bars)

    def find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        """Return all swing lows+highs of *df* merged in time order.

        ``df`` must be an OHLCV frame with ``high``/``low`` columns and a
        DatetimeIndex.  Each returned ``SwingPoint`` is causally usable from
        ``known_at_bar`` onward.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("swing_detector requires a DatetimeIndex")
        lows = df["low"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        n = len(df)
        lb = self.left_bars
        rb = self.right_bars

        out: list[SwingPoint] = []
        for i in range(lb, n - rb):
            lo = lows[i]
            hi = highs[i]
            if lo == min(lows[i - lb: i + rb + 1]) and lo < lows[i - lb] and lo < lows[i + rb]:
                out.append(SwingPoint(i, float(lo), "L", i + rb))
            if hi == max(highs[i - lb: i + rb + 1]) and hi > highs[i - lb] and hi > highs[i + rb]:
                out.append(SwingPoint(i, float(hi), "H", i + rb))
        out.sort(key=lambda s: s.bar)
        return out

    def swing_lows(self, df: pd.DataFrame) -> list[SwingPoint]:
        return [s for s in self.find_swings(df) if s.kind == "L"]

    def swing_highs(self, df: pd.DataFrame) -> list[SwingPoint]:
        return [s for s in self.find_swings(df) if s.kind == "H"]


# ---------------------------------------------------------------------------
# Convenience helpers (shared by every pattern plugin)
# ---------------------------------------------------------------------------

def known_at_time(df: pd.DataFrame, swing: SwingPoint) -> pd.Timestamp:
    """The timestamp at which *swing* becomes causally known (§3.2)."""
    return pd.Timestamp(df.index[int(swing.known_at_bar)])


def pairs_sorted(swings: list[SwingPoint]) -> list[tuple[int, float, str]]:
    """Flatten swings into (bar, price, kind) triples sorted by bar."""
    return [(s.bar, s.price, s.kind) for s in sorted(swings, key=lambda s: s.bar)]