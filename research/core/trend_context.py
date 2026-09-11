"""trend_context.py — Trend-context gate for reversal patterns (rework §2.1).

A double bottom is a *reversal* pattern: classically it only means something
after a real downtrend (and a double top after a real uptrend).  The detector
gated only on the pattern's INTERNAL geometry (``max_equal_atr``,
``min_depth_atr``, ``min_separation_bars``), so a long sideways accumulation
range could be emitted as a "double bottom" even though there was no trend to
reverse (rework request §1.1).

This module implements **Phương án B** of rework §2.1: a linear-regression
slope + R² measured over a lookback window that ends strictly BEFORE
``extreme1_bar``.  Compared to Phương án A ("compare two endpoint closes") the
regression is far less sensitive to a single outlier bar, and requiring a
minimum R² rejects a window whose net move is just noise inside a range.

Two design properties matter more than the numbers themselves:

**Causality (highest-risk part).**  The window is
``closes[extreme1_bar - lookback : extreme1_bar]`` — an exclusive upper bound,
so no bar at or after ``extreme1_bar`` is ever read.  Likewise the ATR passed
in by the caller must be the ATR *known at* ``extreme1_bar``, never the ATR at
the later confirm bar: using the confirm-bar ATR would leak information about
how violently price moved after the pattern completed.  The detector honours
both, and :func:`assert_causal_window` exists so tests can prove it
mechanically rather than by inspection.

**Direction sign.**  Bullish patterns need a preceding DOWNTREND (negative
slope), bearish patterns an UPTREND (positive slope).  ``expected_slope_sign``
derives that from the contract's ``DIRECTION_*`` vocabulary.  Getting the sign
backwards silently inverts the strategy (it would keep only the patterns that
have no reversal context at all), so the sign is pinned by tests for BOTH
directions.

Provider seam (rework §2.4): :func:`has_trend_context` is the default
*slope-heuristic* provider.  A state-based provider (HMM trend direction) can
replace it behind :class:`TrendContextProvider` without the detector changing,
so shipping Phương án B now does not block Phương án C later.

MEASURED — THIS GATE HAS NO DEMONSTRATED SELECTION POWER (read before enabling)
-------------------------------------------------------------------------------
Measured on the FULL XAUUSD M15 history (204,117 bars,
2018-01-02..2026-09-03) by three independent members of the `trend-hmm-rework`
team (this engineer, `measurement_analyst` via docs/rework_trend_hmm_measure.py,
and the captain), and independently corroborated by `gate_engineer`.
The gate *works* — it removes events — but it removes signal and noise at the
same rate, so it does NOT separate good events from bad ones::

    pattern        kept   exp(kept)    dropped   exp(dropped)
    double_bottom   118    -0.0101R        236      -0.0084R
    double_top      105    -0.1599R        205      -0.1596R

The kept-vs-dropped expectancy gap is inside the noise.  Baseline with every gate
OFF: DB 376 events, DT 328 events.  So enabling this gate costs about two thirds
of the trading opportunities and changes nothing about their quality.

(Counts below are on the FULL 204,117-bar window; an earlier draft of this
measurement ran on a 195,893-bar frame truncated at 2018-06-01, which omitted
8,224 bars and 22 DB + 18 DT events — see rework/reviewer_adversarial_review.md
R-1.  Every conclusion is unchanged, but the pool sizes differ: 376/328, not
354/310.)

It also discards outright winners.  Verified directly against the detector
output at lookback=30/min_r2=0.3: **66 DB and 52 DT dropped events have
``rule_score >= 0.75`` AND a winning outcome**, e.g.

    DB 2025-11-12  rule 0.9456  -> +1.453R   (dropped)
    DT 2026-01-29  rule 0.8639  -> +1.486R   (dropped)

These are false drops by any reading.

``lb60/r2=0.3`` produces the best-looking single cell (DB exp +0.1187R), but that
is almost certainly noise: n is small and the grid is 6 cells, so picking the
winner after seeing the grid is exactly the overfitting the request §2.1 warns
against.  No cell shows a *statistically* defensible gain.

That is corroborated on an axis measured BEFORE any trade outcome — the
candidates' own ``rule_score``.  At lookback=30/min_r2=0.3 the kept and dropped
groups are indistinguishable::

    pattern   kept n   mean rule_score   dropped n   mean rule_score    diff
    DB          130          0.7611          246           0.7663      -0.0052
    DT          116          0.7758          212           0.7713      +0.0045

A gate with genuine selection power would keep the higher-quality candidates.
This one keeps marginally *lower*-scoring ones, which is consistent with noise.

Root cause: ``min_r2`` decides, not the slope.  On real events the median
``|slope|/ATR`` is ~0.09 and the median R² is ~0.44, so ``min_r2=0.3`` sits at
roughly the 35th percentile and necessarily cuts ~65% of events regardless of
quality.

Consequence for §4.2: the request's OOS >= 100 requirement **FAILS**, and it is
reported as a FAIL rather than met by loosening parameters.  The captain ruled
that the floor is **per-pattern, not combined** (a combined count would let a
strong DB mask a weak DT).  Under the fixed post-2023-10-12 split::

    pattern   baseline OOS   lb20/r2=0.2   lb30/r2=0.3 (request default)
    double_bottom       151             55              43
    double_top           98             46              38

``double_top`` is **already below 100 at baseline (98)**, before any gate runs,
so the threshold disagrees with the pre-existing data.  Every setting fails both
patterns.  The gate is therefore left DISABLED.

Per the captain's decision this gate therefore ships OPT-IN with
``trend_context_enabled = False`` by default, and is kept only because its
causal mechanism is correct and may prove useful once combined with the HMM
trend-direction state (rework §2.4).  It must NOT be enabled on the strength of
the evidence above.

This module deliberately imports nothing from ``live/`` — ``research/`` must
stay independent of the live engine (it is imported by the backtest runner and
by ``live/``, never the other way round).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from research.core.contracts import DIRECTION_BEARISH, DIRECTION_BULLISH

#: Reasons emitted through ``PatternEvent.attributes["discard_reason"]``.
DISCARD_LOW_TREND_CONTEXT = "low_trend_context"

#: Below this many samples a regression slope is not meaningful.
MIN_LOOKBACK_BARS = 2

#: Default gate parameters (rework §2.1 Phương án B).
#:
#: Approved by the captain on 2026-09-11 as the module default value, with
#: ``trend_context_enabled = False``.  They are readable starting points for a
#: future investigation, NOT a tuned optimum and NOT an endorsed setting.
#:
#: These are NOT tuned to the two sample charts the request was written from
#: (the request §2.1 warns explicitly against that).  They are simply the
#: loosest combination that still enforces a genuine trend.
#:
#: They do NOT satisfy §4.2.  The captain ruled that the OOS >= 100 floor is
#: **per-pattern, not combined**: the request §4.2 states it per pattern, the
#: whole point of the gate is that DB and DT behave differently, and a combined
#: count would let a strong DB mask a weak DT.  Under that reading the gate
#: FAILS — no setting reaches 100 OOS for either pattern, and ``double_top`` is
#: already at 98 OOS at baseline before any gate runs.  A combined count
#: (DB+DT) does reach 120 here, and is recorded only to show what the rejected
#: reading would have given.
DEFAULT_TREND_LOOKBACK_BARS = 20
DEFAULT_MIN_SLOPE_ATR = 0.05
DEFAULT_MIN_R2 = 0.2

#: Floor for the R² denominator so a zero-variance window cannot divide by 0.
_VARIANCE_EPS = 1e-12


def expected_slope_sign(direction: str) -> int:
    """Sign the regression slope must have for ``direction``.

    A bullish reversal (double bottom) sits at the END of a DOWNTREND, so the
    context window's slope must be negative (``-1``).  A bearish reversal
    (double top) sits at the end of an UPTREND, so the slope must be positive
    (``+1``).  Unknown directions fail closed with ``0``, which no real slope
    can satisfy, so an unrecognised direction can never accidentally pass.
    """
    if direction == DIRECTION_BULLISH:
        return -1
    if direction == DIRECTION_BEARISH:
        return 1
    return 0


def trend_context_window(
    closes: np.ndarray[Any, Any],
    extreme1_bar: int,
    lookback: int,
) -> np.ndarray[Any, Any] | None:
    """Return the causal context window, or ``None`` when it is unusable.

    The window is ``closes[extreme1_bar - lookback : extreme1_bar]``.  The
    upper bound is EXCLUSIVE: bar ``extreme1_bar`` (the first extreme of the
    pattern) and everything after it is future information relative to the
    setup and must never enter a trend-context decision.

    ``None`` (fail-closed) when the lookback is degenerate or the window would
    start before the beginning of the series, i.e. when there is not enough
    history to judge a trend at all.
    """
    n = int(closes.shape[0])
    bar = int(extreme1_bar)
    lb = int(lookback)
    if lb < MIN_LOOKBACK_BARS:
        return None
    if bar < 0 or bar > n:
        return None
    start = bar - lb
    if start < 0:
        return None
    window = closes[start:bar]
    if window.shape[0] != lb:
        return None
    return np.asarray(window, dtype=float)


def regression_slope_r2(
    window: np.ndarray[Any, Any],
) -> tuple[float, float] | None:
    """Least-squares ``(slope, r2)`` of ``window`` against ``0..len-1``.

    ``slope`` is in price units per bar.  ``r2`` is the coefficient of
    determination of the line fit; a perfectly flat (zero-variance) window has
    an undefined R² and returns ``None`` per the fail-closed contract, as does
    any non-finite input (NaN/inf in the window).
    """
    y = np.asarray(window, dtype=float)
    if y.ndim != 1 or y.shape[0] < MIN_LOOKBACK_BARS:
        return None
    if not np.all(np.isfinite(y)):
        return None
    x = np.arange(y.shape[0], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return None
    residuals = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= _VARIANCE_EPS:
        # Zero-variance window: R² is 0/0 — undefined, so fail closed rather
        # than let a flat range masquerade as a high-R² trend.
        return None
    ss_res = float((residuals**2).sum())
    r2 = 1.0 - ss_res / ss_tot
    if not np.isfinite(r2):
        return None
    return float(slope), float(r2)


def has_trend_context(
    closes: np.ndarray[Any, Any],
    atr_k: float,
    extreme1_bar: int,
    lookback: int,
    min_slope_atr: float = 0.05,
    min_r2: float = 0.3,
    direction: str | None = None,
) -> bool:
    """True when the window before ``extreme1_bar`` holds a real trend.

    rework §2.1 Phương án B::

        window = closes[extreme1_bar - lookback : extreme1_bar]
        slope, intercept = np.polyfit(x, window, 1)
        r2 = 1 - (residuals**2).sum() / ((window - mean)**2).sum()
        return abs(slope) / atr_k >= min_slope_atr and r2 >= min_r2

    Two deliberate tightenings over the request's snippet, both fail-closed:

    * the slope threshold is expressed in ATR-per-bar, so it is
      scale-free across the 2018-2026 price range of the symbol;
    * when ``direction`` is given, the slope's SIGN must match the direction's
      reversal context (:func:`expected_slope_sign`) instead of the snippet's
      ``abs(slope)``.  ``abs()`` would accept an UPTREND before a double
      bottom, which is the exact inverse of what the gate is for.

    ``returns False`` on every degenerate input — lookback too small,
    not enough bars before ``extreme1_bar``, index out of range, NaN/inf ATR,
    non-positive ATR, zero-variance window (undefined R²) or a non-finite
    regression.  Returning False means "no trend context" so the candidate is
    dropped: the gate never fails open into a trade.
    """
    if not np.isfinite(atr_k) or atr_k <= 0.0:
        return False
    if not (np.isfinite(min_slope_atr) and np.isfinite(min_r2)):
        return False
    if min_slope_atr < 0.0 or min_r2 < 0.0:
        return False

    window = trend_context_window(closes, extreme1_bar, lookback)
    if window is None:
        return False

    fit = regression_slope_r2(window)
    if fit is None:
        return False
    slope, r2 = fit

    if r2 < min_r2:
        return False

    if direction is not None:
        sign = expected_slope_sign(direction)
        if sign == 0:
            return False
        if slope * sign <= 0.0:
            return False

    return abs(slope) / float(atr_k) >= min_slope_atr


def assert_causal_window(
    closes: np.ndarray[Any, Any],
    extreme1_bar: int,
    lookback: int,
) -> np.ndarray[Any, Any] | None:
    """Read the window while asserting no bar >= ``extreme1_bar`` is touched.

    Used by the no-lookahead tests: ``closes`` is wrapped so that every
    subscript is recorded, then the returned window indices are checked
    against ``extreme1_bar``.  Any read at or past ``extreme1_bar`` raises
    ``AssertionError``.
    """
    reads: list[int] = []
    bar = int(extreme1_bar)
    lb = int(lookback)
    if lb < 0 or bar - lb < 0:
        return None
    start = bar - lb

    class _RecordingCloses:
        """Proxy that records every index numpy actually reads."""

        def __init__(self, arr: np.ndarray[Any, Any]) -> None:
            self._arr = arr

        def __getitem__(self, key: Any) -> Any:
            idx = np.arange(self._arr.shape[0])[key]
            if np.ndim(idx) == 0:
                reads.append(int(idx))
            else:
                reads.extend(int(i) for i in np.atleast_1d(idx))
            return self._arr[key]

        @property
        def shape(self) -> tuple[int, ...]:
            return self._arr.shape

        def __len__(self) -> int:
            return int(self._arr.shape[0])

    window = trend_context_window(_RecordingCloses(closes), bar, lb)  # type: ignore[arg-type]
    violated = [i for i in reads if i >= bar]
    if violated:
        raise AssertionError(
            f"trend-context read future bar(s) {violated} at/after extreme1_bar={bar}"
        )
    _ = start
    return None if window is None else np.asarray(window, dtype=float)


@runtime_checkable
class TrendContextProvider(Protocol):
    """Seam for swapping the slope heuristic for a state-based provider.

    rework §2.4 replaces Phương án B with an HMM trend-direction state.  The
    detector only needs this one call, so the HMM provider can be dropped in
    without touching detector code::

        class HmmTrendProvider:
            def has_trend_context(self, closes, atr_k, extreme1_bar, **kw):
                state = self.state_at(extreme1_bar)  # known-at that bar
                return state == "downtrend" if kw["direction"] == DIRECTION_BULLISH \\
                    else state == "uptrend"
    """

    def has_trend_context(
        self,
        closes: np.ndarray[Any, Any],
        atr_k: float,
        extreme1_bar: int,
        **kwargs: Any,
    ) -> bool:  # pragma: no cover - protocol declaration
        ...


class SlopeTrendContextProvider:
    """Default provider — rework §2.1 Phương án B regression heuristic."""

    def __init__(self, lookback: int, min_slope_atr: float, min_r2: float) -> None:
        self.lookback = int(lookback)
        self.min_slope_atr = float(min_slope_atr)
        self.min_r2 = float(min_r2)

    def has_trend_context(
        self,
        closes: np.ndarray[Any, Any],
        atr_k: float,
        extreme1_bar: int,
        **kwargs: Any,
    ) -> bool:
        return has_trend_context(
            closes,
            atr_k,
            extreme1_bar,
            self.lookback,
            self.min_slope_atr,
            self.min_r2,
            direction=kwargs.get("direction"),
        )
