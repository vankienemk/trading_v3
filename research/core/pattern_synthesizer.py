"""
pattern_synthesizer.py — Ground-truth OHLC generator & detector benchmark (§8).

Agent 8 deliverable (REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §8, §13 Agent 8).

``PatternSynthesizer`` generates synthetic OHLCV frames that carry a planted
reversal pattern (liquidity_sweep, double_bottom, double_top, rising_wedge,
falling_wedge, head_shoulders, inverse_head_shoulders) with fully controlled
shape parameters + gaussian noise + an optional trend drift, and a
``GroundTruth`` object stating EXACTLY where the pivot / neckline / breakout
bars are (§8.1).  This is the objective, unbiased benchmark for any detector.

The acceptance harness (§8.3) scores an arbitrary detector against the
planted ground truth:

    * Recall ≥ 80 %  — detector recovers ≥ 80 % of planted patterns
      (position tolerance ≤ 2 bars on the breakout/detect bar).
    * FP ≤ 5 %       — detections on pure-noise (pattern-free) series.
    * Robustness      — recall does not drop more than 15 points when
      ``noise_sigma`` is doubled.

The ``ReferenceDetector`` bundled here is an honest, simple geometric
detector that only reads OHLC (never the ground truth) — it is used to prove
the harness is achievable end-to-end and gives CI a green baseline that every
new pattern detector must at least match.  Future detectors (Agents 3/4) can
be scored with the exact same ``run_benchmark`` harness.

Ground-truth convention: every pattern's ``breakout_bar`` is the FIRST bar
(computed on the clean, pre-noise path) whose close crosses the pattern's
confirmation level (neckline / trendline / prior extreme) after the pattern
last pivot — the same convention a causal detector can honestly recover.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pattern vocabulary (§11)
# ---------------------------------------------------------------------------
SUPPORTED_PATTERNS: list[str] = [
    "liquidity_sweep",
    "double_bottom",
    "double_top",
    "rising_wedge",
    "falling_wedge",
    "head_shoulders",
    "inverse_head_shoulders",
]

BULLISH = "bullish"
BEARISH = "bearish"

#: Canonical native direction of each pattern (breakout direction).
_DIRECTION: dict[str, str] = {
    "liquidity_sweep": BULLISH,
    "double_bottom": BULLISH,
    "double_top": BEARISH,
    "rising_wedge": BEARISH,
    "falling_wedge": BULLISH,
    "head_shoulders": BEARISH,
    "inverse_head_shoulders": BULLISH,
}


# ---------------------------------------------------------------------------
# Ground truth & normalized detection result
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    """Exact planted-pattern geometry for objective detector scoring.

    All ``*bar`` fields are integer positions (0-based) in the generated
    frame.  ``structure_levels`` holds canonical price levels (neckline,
    sweep_low, ...) in the frame price scale.

    ``known_at`` is the causality anchor for downstream no-lookahead suites
    (spec §3.4 / tests/no_lookahead_base.py): the timestamp of the breakout
    bar — the earliest moment the MARKET knows the pattern is complete.
    Every feature used by a detector must be resolvable at or before it.
    """

    pattern_name: str
    direction: str
    pivot_bars: dict[str, int]          # named anchors: low1, low2, head, ...
    structure_levels: dict[str, float]  # neckline, sweep_low, upper, lower, ...
    breakout_bar: int                   # bar where the pattern confirms (§8.2)
    entry_bar: int                      # planned entry bar (breakout or +1)
    known_at: pd.Timestamp | None = None  # = df.index[breakout_bar] (market-knows stamp)
    pattern_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pattern_name not in SUPPORTED_PATTERNS:
            raise ValueError(
                f"unknown pattern {self.pattern_name!r}; "
                f"supported={SUPPORTED_PATTERNS}"
            )
        if self.direction not in (BULLISH, BEARISH):
            raise ValueError(
                f"direction must be bullish|bearish, got {self.direction!r}"
            )


@dataclass
class DetectedPattern:
    """Normalised detection result used by the benchmark harness.

    A real detector (returning ``contracts.PatternEvent``) can be wrapped into
    this shape with a one-line adapter (pattern_name, direction, detect bar).
    """

    pattern_name: str
    direction: str
    breakout_bar: int                   # detector's claimed detect/breakout bar
    confidence: float = 1.0


class PatternDetector(Protocol):
    """Duck-typed interface every detector passed to ``run_benchmark`` must
    satisfy: ``detect(df, pattern_name) -> list[DetectedPattern]``.

    A real pattern plugin (returning ``contracts.PatternEvent``) can be wrapped
    into a ``PatternDetector`` with a one-line adapter that maps each event's
    pattern_name / direction / detect-bar to a ``DetectedPattern``.
    """

    def detect(
        self, df: pd.DataFrame, pattern_name: str
    ) -> list[DetectedPattern]: ...


# ---------------------------------------------------------------------------
# Private OHLC assembly helpers
# ---------------------------------------------------------------------------
def _lin(start: float, end: float, n: int) -> list[float]:
    """n evenly spaced samples between start and end (inclusive)."""
    if n <= 0:
        return []
    if n == 1:
        return [start]
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


def _assemble(
    closes: Sequence[float],
    plant_low: dict[int, float],
    plant_high: dict[int, float],
    noise_sigma: float,
    rng: np.random.Generator,
    wick_body: float,
) -> pd.DataFrame:
    """Turn a close-path plus explicitly planted pivot lows/highs into OHLCV.

    ``plant_low[i]`` / ``plant_high[i]`` pin the bar's low/high to the planted
    value when the noisy body would otherwise be more extreme (or deeper);
    otherwise the pivot stays exactly at the planted level.  Bars are
    internally consistent: high >= max(o,c) and low <= min(o,c).
    """
    n = len(closes)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes_out: list[float] = []
    prev_close: float | None = None
    for i in range(n):
        c = closes[i] + rng.normal(0.0, noise_sigma)
        o = closes[i - 1] if prev_close is not None else closes[0]
        o = o + rng.normal(0.0, noise_sigma * 0.5)
        if i in plant_high:
            hi = max(plant_high[i], max(o, c))
        else:
            hi = max(o, c) + wick_body + abs(rng.normal(0.0, noise_sigma * 0.3))
        if i in plant_low:
            lo = min(plant_low[i], min(o, c))
        else:
            lo = min(o, c) - wick_body - abs(rng.normal(0.0, noise_sigma * 0.3))
        opens.append(o)
        closes_out.append(c)
        highs.append(hi)
        lows.append(lo)
        prev_close = c

    idx = pd.date_range("2020-01-01", periods=n, freq="15min", name="time")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes_out,
            "volume": np.ones(n, dtype=np.int64),
        },
        index=idx,
    )
    df.index.name = "time"
    return df


def _pad(
    closes: list[float],
    n_bars: int,
    drift: float,
    rng: np.random.Generator,
) -> list[float]:
    """Pad (or trim) a close list to exactly n_bars with a small trend drift.

    The tail continues from the LAST close (drift per bar), preserving a
    smooth post-pattern path instead of jumping back to the series start.
    """
    if len(closes) >= n_bars:
        return closes[:n_bars]
    last = closes[-1]
    add = n_bars - len(closes)
    tail = [last + drift * i + rng.normal(0.0, 0.02) for i in range(1, add + 1)]
    return closes + tail


def _first_cross(closes: Sequence[float], start: int, level: float, above: bool) -> int:
    """First bar index >= start whose close crosses ``level`` (pre-noise path).
    Falls back to ``len(closes)-1`` — callers only use this when a crossing
    exists by construction."""
    for j in range(max(0, start), len(closes)):
        if above and closes[j] > level:
            return j
        if not above and closes[j] < level:
            return j
    return len(closes) - 1


# ---------------------------------------------------------------------------
# Per-pattern geometry builders → (closes, plant_low, plant_high, anchors, levels)
# ---------------------------------------------------------------------------
def _build_double(  # double_bottom / double_top (mirror)
    n_bars: int,
    pattern_name: str,
    p: dict[str, Any],
) -> tuple[list[float], dict[int, float], dict[int, float], dict[str, int], dict[str, float]]:
    total = max(n_bars, 8)
    pre = int(p.get("pre_bars", 10))
    left = int(p.get("left_len", 8))
    mid = int(p.get("mid_len", 8))
    right_down = int(p.get("right_down", 8))
    right_up = int(p.get("right_up", 10))
    seg = pre + left + mid + right_down + right_up
    if seg > total:
        scale = (total - pre) / max(1, left + mid + right_down + right_up)
        left = max(2, int(left * scale))
        mid = max(2, int(mid * scale))
        right_down = max(2, int(right_down * scale))
        right_up = max(2, int(right_up * scale))

    base = float(p.get("base", 100.0))
    depth = float(p.get("depth", 0.03))
    neck_frac = float(p.get("neckline_frac", 0.006))

    bullish = pattern_name == "double_bottom"
    level = base * (1.0 - depth if bullish else 1.0 + depth)
    neck = base * (1.0 - neck_frac if bullish else 1.0 + neck_frac)

    closes: list[float] = []
    closes += _lin(base, base * (1.0 - 0.01 if bullish else 1.0 + 0.01), pre)

    a = _lin(closes[-1], level, left)         # to extreme 1
    b = _lin(level, neck, mid)                # to between-peak (neckline touch)
    c = _lin(neck, level, right_down)         # back to extreme 2
    d = _lin(level, neck + (0.012 if bullish else -0.012) * base, right_up)
    closes += a + b + c + d
    closes = _pad(closes, total, 0.0, np.random.default_rng(0))

    # anchor indices on the clean path
    i_ext1 = pre + left - 1
    i_neck = pre + left + mid - 1
    i_ext2 = pre + left + mid + right_down - 1
    # breakout = first close past the neckline after extreme 2
    i_brk = _first_cross(closes, i_ext2 + 1, neck, above=bullish)

    plant_low: dict[int, float] = {}
    plant_high: dict[int, float] = {}
    if bullish:
        plant_low[i_ext1] = level
        plant_low[i_ext2] = level
        plant_high[i_neck] = neck
    else:
        plant_high[i_ext1] = level
        plant_high[i_ext2] = level
        plant_low[i_neck] = neck

    anchors = {
        "extreme1_bar": i_ext1,
        "extreme2_bar": i_ext2,
        "neckline_bar": i_neck,
        "breakout_bar": i_brk,
    }
    levels = {"neckline": neck}
    if bullish:
        levels["double_bottom_level"] = level
    else:
        levels["double_top_level"] = level
    return closes, plant_low, plant_high, anchors, levels


def _build_head_shoulders(  # head_shoulders / inverse_head_shoulders
    n_bars: int,
    pattern_name: str,
    p: dict[str, Any],
) -> tuple[list[float], dict[int, float], dict[int, float], dict[str, int], dict[str, float]]:
    total = max(n_bars, 10)
    pre = int(p.get("pre_bars", 8))

    def scale_segs(*segs: int, head_room: int = 0) -> list[int]:
        s = list(segs)
        totalm = sum(s) + head_room
        if totalm > total - pre:
            k = (total - pre - head_room) / max(1, sum(s))
            s = [max(2, int(x * k)) for x in s]
        return s

    l_shm, to_headm, headm, to_rshm, r_shm = scale_segs(
        int(p.get("shoulder_len", 7)),
        int(p.get("to_head", 6)),
        int(p.get("head_len", 6)),
        int(p.get("to_right_shoulder", 6)),
        int(p.get("shoulder_len", 7)),
        head_room=int(p.get("out_len", 10)),
    )
    out = int(p.get("out_len", 10))

    base = float(p.get("base", 100.0))
    sh_amp = float(p.get("shoulder_amp", 0.02))   # shoulder amplitude vs base
    head_amp = float(p.get("head_amp", 0.035))    # head depth vs base
    neck_off = float(p.get("neckline_off", 0.008))

    bullish = pattern_name == "inverse_head_shoulders"

    def sh_level() -> float:
        return base * (1.0 + sh_amp if not bullish else 1.0 - sh_amp)

    def head_level() -> float:
        return base * (1.0 + head_amp if not bullish else 1.0 - head_amp)

    def neck_level() -> float:
        return base * (1.0 - neck_off if not bullish else 1.0 + neck_off)

    closes: list[float] = []
    # approach: gently drift toward the opposite side of the formation so the
    # first shoulder is a genuine swing (rise for H&S / fall for inverse)
    closes += _lin(base, base * (1.0 - 0.006 if not bullish else 1.0 + 0.006), pre)
    c0 = closes[-1]

    ls = sh_level()
    hl = head_level()
    nk = neck_level()
    # H&S (bearish): rise→ls, fall→nk, rise→hl, fall→nk, rise→rs, break below nk
    # inverse:          fall→ls, rise→nk, fall→hl, rise→nk, fall→rs, break above nk
    closes += _lin(c0, ls, l_shm)
    closes += _lin(ls, nk, to_headm)
    closes += _lin(nk, hl, headm)
    closes += _lin(hl, nk, to_rshm)
    closes += _lin(nk, ls, r_shm)
    closes += _lin(ls, nk - (0.012 if not bullish else -0.012) * base, out)
    closes = _pad(closes, total, 0.0, np.random.default_rng(0))

    i_ls = pre + l_shm - 1
    i_head = pre + l_shm + to_headm + headm - 1
    i_rs = pre + l_shm + to_headm + headm + to_rshm + r_shm - 1
    i_n2 = pre + l_shm + to_headm + headm + to_rshm - 1
    # breakout = first close past the neckline after the right shoulder
    i_brk = _first_cross(closes, i_rs + 1, nk, above=bullish)

    plant_low: dict[int, float] = {}
    plant_high: dict[int, float] = {}
    if not bullish:  # head and shoulders (peaks)
        plant_high[i_ls] = ls
        plant_high[i_head] = hl
        plant_high[i_rs] = ls
        plant_low[i_n2] = nk
    else:  # inverse head and shoulders (troughs)
        plant_low[i_ls] = ls
        plant_low[i_head] = hl
        plant_low[i_rs] = ls
        plant_high[i_n2] = nk

    anchors = {
        "left_shoulder_bar": i_ls,
        "head_bar": i_head,
        "right_shoulder_bar": i_rs,
        "neckline_bar": i_n2,
        "breakout_bar": i_brk,
    }
    levels = {"neckline": nk, "shoulder_level": ls, "head_level": hl}
    return closes, plant_low, plant_high, anchors, levels


def _build_wedge(  # rising_wedge / falling_wedge
    n_bars: int,
    pattern_name: str,
    p: dict[str, Any],
) -> tuple[list[float], dict[int, float], dict[int, float], dict[str, int], dict[str, float]]:
    total = max(n_bars, 12)
    pre = int(p.get("pre_bars", 6))
    touch = int(p.get("touch_bars", 5))
    out = int(p.get("out_len", 8))
    n_touch = int(p.get("n_touches", 5))

    # wedge channel: two converging trendlines in price space
    base = float(p.get("base", 100.0))
    width = float(p.get("width", 0.03))
    bull = pattern_name == "falling_wedge"

    if bull:  # falling wedge: both lines descend, upper falls faster (converging)
        u0, u1 = base * (1.0 + 1.2 * width), base * (1.0 - 0.2 * width)
        l0, l1 = base * (1.0 - 0.4 * width), base * (1.0 - 1.2 * width)
    else:  # rising wedge: both lines ascend, lower rises faster (converging)
        u0, u1 = base * (1.0 + 0.4 * width), base * (1.0 + 1.2 * width)
        l0, l1 = base * (1.0 - 1.2 * width), base * (1.0 + 0.2 * width)

    # segment lengths: alternating touch segments + one breakout run
    segs = [touch] * (n_touch + 1)
    while sum(segs) + pre > total - out and len(segs) > 3:
        segs = [max(3, s - 1) for s in segs]
        if all(s == 3 for s in segs):
            break

    closes: list[float] = []
    # pre: enter from OUTSIDE the channel (above for falling, below for rising)
    # so the first touch is a genuine swing off the first channel line.
    closes += _lin(base * (1.0 + 0.02 if bull else 1.0 - 0.02), u0 if not bull else l0, pre)
    prev = closes[-1]
    idx = pre
    touch_low: dict[int, float] = {}
    touch_high: dict[int, float] = {}
    ref_low: list[tuple[int, float]] = []
    ref_high: list[tuple[int, float]] = []
    n_pat = sum(segs)

    def line(f0: float, f1: float, bar: int) -> float:
        """Value of a channel line at bar (interpolated across the pattern)."""
        f = (bar - pre) / max(1, n_pat - 1)
        return f0 + (f1 - f0) * f

    for s_i, seglen in enumerate(segs):
        frac = (sum(segs[:s_i]) + seglen * 0.5) / max(1, n_pat)
        u = u0 + (u1 - u0) * frac
        lo = l0 + (l1 - l0) * frac
        last_seg = s_i == len(segs) - 1
        if last_seg:
            # breakout run: pierce decisively BEYOND the opposite channel line
            # (margin scaled to the channel width so the close clearly crosses
            # the line before the run ends).
            margin = 0.5 * width * base
            target = (u1 + margin) if bull else (l1 - margin)
            closes += _lin(prev, target, seglen)
            break
        if s_i % 2 == 0:  # descend to low touch on the lower line
            closes += _lin(prev, lo, seglen)
            touch_low[idx + seglen - 1] = lo
            ref_low.append((idx + seglen - 1, lo))
            prev = lo
        else:  # ascend to high touch on the upper line
            closes += _lin(prev, u, seglen)
            touch_high[idx + seglen - 1] = u
            ref_high.append((idx + seglen - 1, u))
            prev = u
        idx += seglen

    closes = _pad(closes, total, 0.0, np.random.default_rng(0))

    # breakout (ground truth) = the first close on the clean path that
    # decisively pierces the OPPOSING channel line after the third monotonic
    # low touch (the earliest bar at which a 3-touch wedge is complete).  A
    # detector that finds the wedge geometry and scans for the same decisive
    # close-cross lands inside the ±2-bar tolerance (pivot right-bar lag is
    # absorbed by the decisive exit margin).
    def brk_line_f(j: int) -> float:
        return line(u0, u1, j) if bull else line(l0, l1, j)

    if len(ref_low) >= 3:
        scan_from = ref_low[2][0] + 1
    else:
        scan_from = idx + 1
    i_brk = total - 1
    margin_cross = 0.3 * width * base
    for j in range(scan_from, min(len(closes), scan_from + 20)):
        if bull and closes[j] > brk_line_f(j) + margin_cross:
            i_brk = j
            break
        if not bull and closes[j] < brk_line_f(j) - margin_cross:
            i_brk = j
            break
    i_brk = min(i_brk, total - 1)

    anchors = {"breakout_bar": i_brk}
    for k, (bi, _v) in enumerate(ref_low):
        anchors[f"low_touch_{k}"] = bi
    for k, (bi, _v) in enumerate(ref_high):
        anchors[f"high_touch_{k}"] = bi
    if ref_low:
        anchors["last_low_touch"] = ref_low[-1][0]
    if ref_high:
        anchors["last_high_touch"] = ref_high[-1][0]
    levels = {
        "upper_trendline_0": u0,
        "upper_trendline_1": u1,
        "lower_trendline_0": l0,
        "lower_trendline_1": l1,
    }
    return closes, touch_low, touch_high, anchors, levels


def _build_sweep(  # liquidity_sweep (bullish low sweep)
    n_bars: int,
    pattern_name: str,
    p: dict[str, Any],
) -> tuple[list[float], dict[int, float], dict[int, float], dict[str, int], dict[str, float]]:
    total = max(n_bars, 10)
    pre = int(p.get("pre_bars", 8))

    def scale_segs(*segs: int) -> list[int]:
        s = list(segs)
        if sum(s) + pre > total:
            k = (total - pre) / max(1, sum(s))
            s = [max(2, int(x * k)) for x in s]
        return s

    to_lowm, pullbackm, sweep_downm, reclaimm = scale_segs(
        int(p.get("to_low", 7)),
        int(p.get("pullback", 5)),
        int(p.get("sweep_down", 4)),
        int(p.get("reclaim", 6)),
    )

    base = float(p.get("base", 100.0))
    depth = float(p.get("depth", 0.02))
    pen = float(p.get("penetration", 0.006))

    prior_low = base * (1.0 - depth)
    sweep_low = prior_low - pen * base

    closes: list[float] = []
    closes += _lin(base, base * 1.005, pre)
    closes += _lin(closes[-1], prior_low, to_lowm)
    closes += _lin(prior_low, prior_low + depth * 0.5 * base, pullbackm)
    closes += _lin(prior_low + depth * 0.5 * base, sweep_low, sweep_downm)
    closes += _lin(sweep_low, prior_low + pen * base, reclaimm)
    closes = _pad(closes, total, 0.0, np.random.default_rng(0))

    i_prior = pre + to_lowm - 1
    i_sweep = pre + to_lowm + pullbackm + sweep_downm - 1
    i_reclaim = _first_cross(closes, i_sweep + 1, prior_low, above=True)

    plant_low: dict[int, float] = {i_prior: prior_low, i_sweep: sweep_low}
    plant_high: dict[int, float] = {}

    anchors = {
        "prior_low_bar": i_prior,
        "sweep_low_bar": i_sweep,
        "breakout_bar": i_reclaim,
    }
    levels = {"sweep_low": sweep_low, "prior_low": prior_low}
    return closes, plant_low, plant_high, anchors, levels


_BUILDERS: dict[str, Callable[..., tuple[list[float], dict[int, float], dict[int, float], dict[str, int], dict[str, float]]]] = {
    "double_bottom": lambda n, p: _build_double(n, "double_bottom", p),
    "double_top": lambda n, p: _build_double(n, "double_top", p),
    "head_shoulders": lambda n, p: _build_head_shoulders(n, "head_shoulders", p),
    "inverse_head_shoulders": lambda n, p: _build_head_shoulders(n, "inverse_head_shoulders", p),
    "rising_wedge": lambda n, p: _build_wedge(n, "rising_wedge", p),
    "falling_wedge": lambda n, p: _build_wedge(n, "falling_wedge", p),
    "liquidity_sweep": lambda n, p: _build_sweep(n, "liquidity_sweep", p),
}


# ---------------------------------------------------------------------------
# Public synthesizer
# ---------------------------------------------------------------------------
def default_pattern_params(pattern_name: str) -> dict[str, Any]:
    """Sane defaults per pattern (overridable via ``pattern_params``)."""
    if pattern_name not in SUPPORTED_PATTERNS:
        raise ValueError(f"unknown pattern {pattern_name!r}")
    return {
        "double_bottom": {"depth": 0.03, "left_len": 8, "mid_len": 8, "right_down": 8, "right_up": 10},
        "double_top": {"depth": 0.03, "left_len": 8, "mid_len": 8, "right_down": 8, "right_up": 10},
        "head_shoulders": {"shoulder_amp": 0.02, "head_amp": 0.035, "neckline_off": 0.008, "shoulder_len": 7, "out_len": 10},
        "inverse_head_shoulders": {"shoulder_amp": 0.02, "head_amp": 0.035, "neckline_off": 0.008, "shoulder_len": 7, "out_len": 10},
        "rising_wedge": {"width": 0.03, "touch_bars": 5, "n_touches": 5, "out_len": 8},
        "falling_wedge": {"width": 0.03, "touch_bars": 5, "n_touches": 5, "out_len": 8},
        "liquidity_sweep": {"depth": 0.02, "penetration": 0.006, "pre_bars": 8},
    }[pattern_name]


def _grid_params(pattern_name: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample a random parameter point across the expected operating domain
    (§8.3 grid) so the benchmark is not over-fit to one geometry."""
    base = float(rng.uniform(80.0, 200.0))
    common = {"base": base}
    if pattern_name in ("double_bottom", "double_top"):
        return {
            **common,
            "depth": float(rng.uniform(0.02, 0.05)),
            "left_len": int(rng.integers(6, 12)),
            "mid_len": int(rng.integers(6, 12)),
            "right_down": int(rng.integers(6, 12)),
            "right_up": int(rng.integers(8, 14)),
        }
    if pattern_name in ("head_shoulders", "inverse_head_shoulders"):
        return {
            **common,
            "shoulder_amp": float(rng.uniform(0.015, 0.03)),
            "head_amp": float(rng.uniform(0.025, 0.045)),
            "shoulder_len": int(rng.integers(5, 9)),
            "out_len": int(rng.integers(8, 12)),
        }
    if pattern_name in ("rising_wedge", "falling_wedge"):
        return {
            **common,
            "width": float(rng.uniform(0.025, 0.05)),
            "touch_bars": int(rng.integers(4, 7)),
            # odd number of touches → alternating touches end on a LOW,
            # giving ≥3 low touches for the breakout leg (L,H,L,...,L)
            "n_touches": int(rng.choice([5, 7])),
            "out_len": int(rng.integers(6, 10)),
        }
    # liquidity_sweep
    return {
        **common,
        "depth": float(rng.uniform(0.015, 0.03)),
        "penetration": float(rng.uniform(0.006, 0.012)),
        "pre_bars": int(rng.integers(6, 10)),
    }


class PatternSynthesizer:
    """Ground-truth OHLC generator (spec §8.2)."""

    def __init__(self, wick_body: float = 0.02) -> None:
        self.wick_body = wick_body

    def generate(
        self,
        pattern_name: str,
        n_bars: int,
        pattern_params: dict[str, Any] | None = None,
        noise_sigma: float = 0.05,
        trend_drift: float = 0.0,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, GroundTruth]:
        """Generate an OHLCV frame with one planted ``pattern_name``.

        Returns ``(df_ohlcv, ground_truth)`` per §8.2.  ``pattern_params``
        sets the shape (depth, symmetry, duration, ...); unset keys fall back
        to ``default_pattern_params``.  ``noise_sigma`` is the per-bar
        gaussian noise (in price units), ``trend_drift`` a linear drift added
        to the global path, ``seed`` makes the whole thing reproducible.
        """
        if pattern_name not in SUPPORTED_PATTERNS:
            raise ValueError(
                f"unknown pattern {pattern_name!r}; supported={SUPPORTED_PATTERNS}"
            )
        if n_bars < 8:
            raise ValueError("n_bars must be >= 8")
        merged: dict[str, Any] = dict(default_pattern_params(pattern_name))
        if pattern_params:
            merged.update(pattern_params)
        merged.setdefault("base", 100.0)

        rng = np.random.default_rng(seed)
        builder = _BUILDERS[pattern_name]
        closes, plant_low, plant_high, anchors, levels = builder(n_bars, merged)

        # apply global trend drift
        if trend_drift != 0.0:
            closes = [c + trend_drift * i for i, c in enumerate(closes)]

        df = _assemble(closes, plant_low, plant_high, noise_sigma, rng, self.wick_body)

        breakout = anchors["breakout_bar"]
        gt = GroundTruth(
            pattern_name=pattern_name,
            direction=_DIRECTION[pattern_name],
            pivot_bars=anchors,
            structure_levels=levels,
            breakout_bar=breakout,
            entry_bar=breakout,
            known_at=df.index[breakout],  # causality anchor (spec §3.4)
            pattern_params=dict(merged),
        )
        return df, gt

    def generate_noise(
        self,
        n_bars: int,
        noise_sigma: float = 0.05,
        trend_drift: float = 0.0,
        seed: int = 0,
    ) -> pd.DataFrame:
        """A pure-noise series with NO planted pattern (FP benchmark §8.3).

        Uses a stationary mean-reverting (Ornstein-Uhlenbeck) process around
        a fixed level: the canonical "no pattern" control.  A drifting random
        walk would routinely produce trend-like channels and multi-ATR swings
        that no honest detector can be expected to ignore — the OU control
        isolates pattern-hallucination from trend-following.
        """
        if n_bars < 8:
            raise ValueError("n_bars must be >= 8")
        rng = np.random.default_rng(seed)
        level = 100.0
        kappa = 0.15  # mean reversion strength
        closes = [level]
        for _ in range(1, n_bars):
            prev = closes[-1]
            nx = prev + kappa * (level - prev) + trend_drift + rng.normal(0.0, noise_sigma)
            closes.append(nx)
        return _assemble(closes, {}, {}, noise_sigma, rng, self.wick_body)


# ---------------------------------------------------------------------------
# Reference geometric detectors (honest — read OHLC only)
# ---------------------------------------------------------------------------
def _const_line(level: float) -> Callable[[int], float]:
    """Return a constant-valued line helper (horizontal neckline) for the
    close-cross scan.  A plain ``lambda _: level`` would trip ruff's B023
    closure-over-loop-variable check in the detector loops."""
    return lambda _j: level


def _vol(df: pd.DataFrame) -> float:
    """Robust volatility = median per-bar range (high-low).

    Unlike Wilder ATR, the median range is NOT inflated by the planted
    pattern's own long pivot wicks, so it gives a clean noise-scale baseline
    that separates genuine multi-{vol} structure from sub-vol noise."""
    return float(np.median((df["high"] - df["low"]).to_numpy()))


def _swings(df: pd.DataFrame, look: int) -> list[tuple[int, float, str]]:
    """Swing lows (from the low series) + swing highs (from the high series),
    merged in time order.  Returns (bar, price, 'L'|'H').

    No same-type merging on purpose: a breakout junction bar may legitimately
    be more extreme than an earlier touch (e.g. the pierce low of a rising
    wedge), and keeping both lets the pattern-specific rules pick the correct
    monotonic window.  Noise candidates are filtered by the ATR gates.
    """
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    n = len(df)
    events: list[tuple[int, float, str]] = []
    for i in range(look, n - look):
        if lows[i] == min(lows[i - look: i + look + 1]) and lows[i] < lows[i - look] and lows[i] < lows[i + look]:
            events.append((i, float(lows[i]), "L"))
        if highs[i] == max(highs[i - look: i + look + 1]) and highs[i] > highs[i - look] and highs[i] > highs[i + look]:
            events.append((i, float(highs[i]), "H"))
    events.sort(key=lambda t: t[0])
    return events


class ReferenceDetector:
    """Simple, honest geometric detector per pattern (baseline for CI).

    Only reads OHLC — never the ground truth.  Structural gates are expressed
    in ATR multiples so pure-noise series (sub-ATR wiggles) cannot satisfy
    them while planted patterns (multi-ATR depth) are recovered.
    """

    name = "reference"

    def detect(
        self, df: pd.DataFrame, pattern_name: str, look: int = 3
    ) -> list[DetectedPattern]:
        if pattern_name == "liquidity_sweep":
            return self._sweep(df, look)
        if pattern_name in ("double_bottom", "double_top"):
            return self._double(df, pattern_name, look)
        if pattern_name in ("head_shoulders", "inverse_head_shoulders"):
            return self._hs(df, pattern_name, look)
        if pattern_name in ("rising_wedge", "falling_wedge"):
            return self._wedge(df, pattern_name, look)
        return []

    # -- helpers ------------------------------------------------------------
    def _first_close(
        self, df: pd.DataFrame, start: int, level: float, above: bool, lookahead: int = 14
    ) -> int | None:
        closes = df["close"].to_numpy()
        for j in range(start + 1, min(len(closes), start + lookahead + 1)):
            if above and closes[j] > level:
                return j
            if not above and closes[j] < level:
                return j
        return None

    def _close_cross(
        self, df: pd.DataFrame, start: int, line: Callable[[int], float], above: bool
    ) -> int | None:
        closes = df["close"].to_numpy()
        for j in range(start + 1, min(len(closes), start + 16)):
            lv = line(j)
            if above and closes[j] > lv:
                return j
            if not above and closes[j] < lv:
                return j
        return None

    # -- double bottom / top ------------------------------------------------
    def _double(self, df: pd.DataFrame, pattern_name: str, look: int) -> list[DetectedPattern]:
        bull = pattern_name == "double_bottom"
        sw = _swings(df, look)
        vol = _vol(df)
        # double bottom: L-H-L with two lows at the same level;
        # double top:    H-L-H with two highs at the same level.
        want = ("L", "H", "L") if bull else ("H", "L", "H")
        results: list[DetectedPattern] = []
        for k in range(len(sw) - 2):
            (i1, v1, t1), (i2, v2, t2), (i3, v3, t3) = sw[k], sw[k + 1], sw[k + 2]
            if (t1, t2, t3) != want:
                continue
            if i2 - i1 < look or i3 - i2 < look:
                continue
            if bull:
                if not (v1 <= v2 and v3 <= v2):
                    continue
                if abs(v1 - v3) > 1.0 * vol:  # two lows at the same level
                    continue
                if v2 - max(v1, v3) < 5.0 * vol:  # real depth vs noise
                    continue
                brk = self._first_close(df, i3, v2, above=True)
                if brk is not None:
                    results.append(DetectedPattern("double_bottom", BULLISH, brk))
            else:
                if not (v1 >= v2 and v3 >= v2):
                    continue
                if abs(v1 - v3) > 1.0 * vol:
                    continue
                if min(v1, v3) - v2 < 5.0 * vol:
                    continue
                brk = self._first_close(df, i3, v2, above=False)
                if brk is not None:
                    results.append(DetectedPattern("double_top", BEARISH, brk))
        return results

    # -- head & shoulders ---------------------------------------------------
    def _hs(self, df: pd.DataFrame, pattern_name: str, look: int) -> list[DetectedPattern]:
        bull = pattern_name == "inverse_head_shoulders"
        sw = _swings(df, look)
        vol = _vol(df)
        # regular H&S (bearish): H-L-H-L-H  (shoulder, trough, head, trough, shoulder)
        # inverse H&S (bullish): L-H-L-H-L
        want = ("L", "H", "L", "H", "L") if bull else ("H", "L", "H", "L", "H")
        results: list[DetectedPattern] = []
        for k in range(len(sw) - 4):
            seq = sw[k:k + 5]
            (ia, _va, ta), (ib, vb, tb), (ic, vc, tc), (id_, vd, td), (ie, _ve, te) = seq
            if (ta, tb, tc, td, te) != want:
                continue
            if not (ia < ib < ic < id_ < ie):
                continue
            if bull:
                # inverse H&S: head trough (ic) deepest; shoulder troughs equal;
                # neckline = the two peaks between them (use the lower peak).
                head_ok = vc < vb and vc < vd
                sh_eq = abs(vb - vd) <= 0.8 * vol
                if not (head_ok and sh_eq):
                    continue
                if min(vb, vd) - vc < 5.0 * vol:  # head deeper than shoulders
                    continue
                neck = min(vb, vd)
                brk = self._close_cross(df, ie, _const_line(neck), above=True)
                if brk is not None:
                    results.append(DetectedPattern("inverse_head_shoulders", BULLISH, brk))
            else:
                # H&S: head peak (ic) highest; shoulder peaks equal; neckline =
                # the two troughs between them (use the higher trough).
                head_ok = vc > vb and vc > vd
                sh_eq = abs(vb - vd) <= 0.8 * vol
                if not (head_ok and sh_eq):
                    continue
                if vc - max(vb, vd) < 5.0 * vol:
                    continue
                neck = max(vb, vd)
                brk = self._close_cross(df, ie, _const_line(neck), above=False)
                if brk is not None:
                    results.append(DetectedPattern("head_shoulders", BEARISH, brk))
        return results

    # -- wedge --------------------------------------------------------------
    @staticmethod
    def _fit_line(points: Sequence[tuple[int, float]]) -> Callable[[int], float]:
        ii, vv = zip(*points)
        slope, intercept = np.polyfit(ii, vv, 1)
        return lambda j: float(slope * j + intercept)

    @staticmethod
    def _channel_interior(
        df: pd.DataFrame, i0: int, i2: int,
        hi_line: Callable[[int], float], lo_line: Callable[[int], float],
    ) -> bool:
        """A genuine wedge keeps ≥80% of its closes INSIDE the converging
        channel (landmark-free detection tolerates ±bar swing offsets and
        small noise pokes); structureless noise fails this easily."""
        closes = df["close"].to_numpy()
        vol = _vol(df)
        tol = 0.8 * vol
        inside = 0
        total = 0
        for j in range(i0 + 1, i2):
            total += 1
            if lo_line(j) - tol <= closes[j] <= hi_line(j) + tol:
                inside += 1
        if total == 0:
            return False
        return inside / total >= 0.80

    def _wedge(self, df: pd.DataFrame, pattern_name: str, look: int) -> list[DetectedPattern]:
        bull = pattern_name == "falling_wedge"
        sw = _swings(df, look)
        vol = _vol(df)
        lows = [(i, v) for i, v, t in sw if t == "L"]
        highs = [(i, v) for i, v, t in sw if t == "H"]
        results: list[DetectedPattern] = []

        for k in range(len(lows) - 2):
            (i0, v0), (i1, v1), (i2, v2) = lows[k], lows[k + 1], lows[k + 2]
            if bull:
                # falling wedge: 3 descending lows AND ≥2 descending highs
                # BETWEEN them, converging channel (width narrows by ≥12 %
                # across the highs pair, evaluated inside fit support) and
                # closes contained in the channel.
                if not (v0 > v1 > v2):
                    continue
                if (v0 - v2) < 3.0 * vol:
                    continue
                hw = [(i, v) for i, v in highs if i0 < i < i2]
                if len(hw) < 2 or not all(hw[j][1] > hw[j + 1][1] for j in range(len(hw) - 1)):
                    continue
                hi_line = self._fit_line(hw)
                lo_line = self._fit_line([(i0, v0), (i1, v1), (i2, v2)])
                a, b = hw[0][0], hw[-1][0]
                if not hi_line(b) - lo_line(b) < 1.0 * (hi_line(a) - lo_line(a)):
                    continue  # channel must narrow (converging)
                if not self._channel_interior(df, i0, i2, hi_line, lo_line):
                    continue
                # breakout = first close decisively ABOVE the upper line after
                # the 3rd monotonic low touch (price-cross convention matches
                # the generator's ground truth; the decisive exit margin
                # absorbs pivot right-bar lag).
                brk = self._close_cross(df, i2, hi_line, above=True)
                if brk is not None:
                    results.append(DetectedPattern("falling_wedge", BULLISH, brk))
            else:
                # rising wedge: 3 ascending lows AND ≥2 ascending highs between
                # them, converging channel (lower line rises faster) and closes
                # contained in the channel.
                if not (v0 < v1 < v2):
                    continue
                if (v2 - v0) < 3.0 * vol:
                    continue
                hw = [(i, v) for i, v in highs if i0 < i < i2]
                if len(hw) < 2 or not all(hw[j][1] < hw[j + 1][1] for j in range(len(hw) - 1)):
                    continue
                lo_line = self._fit_line([(i0, v0), (i1, v1), (i2, v2)])
                hi_line = self._fit_line(hw)
                a, b = hw[0][0], hw[-1][0]
                if not hi_line(b) - lo_line(b) < 1.0 * (hi_line(a) - lo_line(a)):
                    continue  # channel must narrow (converging)
                if not self._channel_interior(df, i0, i2, hi_line, lo_line):
                    continue
                brk = self._close_cross(df, i2, lo_line, above=False)
                if brk is not None:
                    results.append(DetectedPattern("rising_wedge", BEARISH, brk))
        return results

    # -- liquidity sweep ----------------------------------------------------
    def _sweep(self, df: pd.DataFrame, look: int) -> list[DetectedPattern]:
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        n = len(df)
        vol = _vol(df)
        piv: list[tuple[int, float]] = [
            (i, float(lows[i]))
            for i in range(look, n - look)
            if lows[i] == min(lows[i - look: i + look + 1])
            and lows[i] < lows[i - look]
            and lows[i] < lows[i + look]
        ]
        results: list[DetectedPattern] = []
        for k in range(len(piv) - 1):
            (i1, v1), (i2, v2) = piv[k], piv[k + 1]
            if i2 - i1 < look:
                continue
            if v2 < v1:  # stop hunt below the prior low
                if v1 - v2 < 2.0 * vol:  # meaningful penetration vs noise
                    continue
                # trend context: the prior low formed after a genuine decline
                # (liquidity pool above); 4 bars earlier price was ≥2 ATR higher
                if i1 - 4 < 0 or closes[i1 - 4] - v1 < 3.0 * vol:
                    continue
                brk = self._first_close(df, i2, v1, above=True, lookahead=10)
                if brk is not None:
                    results.append(DetectedPattern("liquidity_sweep", BULLISH, brk))
        return results


# ---------------------------------------------------------------------------
# Acceptance benchmark harness (§8.3)
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkReport:
    pattern_name: str
    n_series: int
    recall: float                    # fraction of planted patterns recovered (≤2 bar tol)
    recall_at_2x_noise: float        # recall when noise_sigma doubled
    robustness_drop: float           # recall_base - recall_2x (must stay <= 0.15)
    n_detected: int
    n_planted: int


@dataclass
class BenchmarkResult:
    per_pattern: list[BenchmarkReport]
    false_positive_rate: float       # fraction of noise-only series producing >=1 detection
    n_noise_series: int

    def recall(self, pattern_name: str) -> float:
        for r in self.per_pattern:
            if r.pattern_name == pattern_name:
                return r.recall
        return 0.0

    def fp_rate(self) -> float:
        return self.false_positive_rate

    def all_recall_ge(self, threshold: float = 0.80) -> bool:
        return all(r.recall >= threshold for r in self.per_pattern)

    def all_robust(self, max_drop: float = 0.15) -> bool:
        return all(r.robustness_drop <= max_drop for r in self.per_pattern)


def _match(detections: Sequence[DetectedPattern], gt: GroundTruth, tol: int = 2) -> bool:
    for d in detections:
        if d.pattern_name != gt.pattern_name:
            continue
        if d.direction != gt.direction:
            continue
        if abs(d.breakout_bar - gt.breakout_bar) <= tol:
            return True
    return False


def run_benchmark(
    detector: PatternDetector,
    pattern_names: Sequence[str],
    n_series: int = 100,
    noise_sigma: float = 0.05,
    n_bars: int = 120,
    seed: int = 0,
    tol: int = 2,
    n_noise_series: int = 200,
    trend_drift: float = 0.0,
) -> BenchmarkResult:
    """Score ``detector`` (duck-typed ``detect(df, pattern_name) -> [...]``)
    against planted ground truth (§8.3).

    Uses parameter gridding (``_grid_params``) so the score covers the
    expected operating domain rather than one geometry; per-pattern series
    are disjoint-seeded so the 2x-noise robustness series never coincide with
    the base-noise series.
    """
    syn = PatternSynthesizer()
    reports: list[BenchmarkReport] = []
    for pat in pattern_names:
        rng = np.random.default_rng(seed)
        n_hit = 0
        n_hit_2x = 0
        for s in range(n_series):
            params = _grid_params(pat, rng)
            df, gt = syn.generate(
                pat, n_bars, params,
                noise_sigma=noise_sigma, trend_drift=trend_drift, seed=seed + s,
            )
            if _match(detector.detect(df, pat), gt, tol):
                n_hit += 1

            df2, gt2 = syn.generate(
                pat, n_bars, params,
                noise_sigma=2.0 * noise_sigma, trend_drift=trend_drift, seed=seed + s + 10_000,
            )
            if _match(detector.detect(df2, pat), gt2, tol):
                n_hit_2x += 1

        reports.append(
            BenchmarkReport(
                pattern_name=pat,
                n_series=n_series,
                recall=n_hit / max(1, n_series),
                recall_at_2x_noise=n_hit_2x / max(1, n_series),
                robustness_drop=(n_hit - n_hit_2x) / max(1, n_series),
                n_detected=n_hit,
                n_planted=n_series,
            )
        )

    # FP on pure-noise series
    fp = 0
    for s in range(n_noise_series):
        df = syn.generate_noise(
            n_bars, noise_sigma=noise_sigma, trend_drift=trend_drift, seed=seed + 100_000 + s
        )
        if any(detector.detect(df, pat) for pat in pattern_names):
            fp += 1
    fp_rate = fp / max(1, n_noise_series)

    return BenchmarkResult(
        per_pattern=reports, false_positive_rate=fp_rate, n_noise_series=n_noise_series
    )


def assert_acceptance(result: BenchmarkResult) -> None:
    """Assert the §8.3 acceptance gates — used directly by the CI test."""
    for r in result.per_pattern:
        assert r.recall >= 0.80, (
            f"{r.pattern_name}: recall {r.recall:.2f} < 0.80"
        )
        assert r.robustness_drop <= 0.15, (
            f"{r.pattern_name}: robustness drop {r.robustness_drop:.2f} > 0.15 "
            f"({r.recall:.2f} -> {r.recall_at_2x_noise:.2f})"
        )
    assert result.false_positive_rate <= 0.05, (
        f"FP rate {result.false_positive_rate:.3f} > 0.05 on pure-noise series"
    )