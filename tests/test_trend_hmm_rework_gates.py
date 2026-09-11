"""Detector-level gates from the trend/HMM rework request — t2 (gate_engineer).

Covers the two cheap, independent detector gates of
``rework/trading_v3_rework_request_trend_hmm.md``:

  * **§2.2** ``max_pattern_length_bars`` — drop a candidate whose two extremes
    are further apart than the bound (``pattern_length = extreme2_bar -
    extreme1_bar``), ``discard_reason = "pattern_too_long"``.
  * **§2.3** (detector side) ``min_rule_score`` — independent floor on this
    detector's own ``rule_score``, ``discard_reason = "low_rule_score"``.

Both live in ``DoublePatternDetectorBase`` and therefore apply to
``double_bottom`` AND ``double_top`` (double_top overrides no method).

MEASURED HONESTY NOTE (read before "improving" a default)
---------------------------------------------------------
§2.2 is a **NO-OP on XAUUSD M15 and is NOT an improvement**. Measured over
195,893 bars (2018-06-01..2026-09-03) and over the full 204,117-bar parquet:
``double_bottom`` max span 29 bars (n=354 / 376), ``double_top`` max 26
(n=310 / 328). **Zero** events exceed 30, so the request's proposed 40-60
would remove 0 events. The default 30 is simply the tightest bound provably
removing nothing. It is defence-in-depth, not a fix for the sideways-range
symptom (the request's §1.2 premise is false for this detector:
``min_separation_bars`` + the staleness window already cap the span).

§2.3 defaults to 0.0 (OFF): at 0.60 the keep-rate is 95.5% (DB) / 96.8% (DT)
and the expectancy delta sits inside noise, so 0.60 is not a proven
improvement and therefore cannot be a default.  The tests below pin the
*mechanism* and the *default-preserving* property, not a claimed edge.

``pattern_length`` was declared ``AVAILABLE_AT_DETECT`` in ``feature_schema``
but never populated before this change (t1 verification finding F-01); the
tests also pin that it is now actually emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.core.config_hash import compute_config_hash, config_hash_for_detector
from research.patterns.double_bottom.detector import (
    FEATURE_SCHEMA_VERSION,
    DoubleBottomDetector,
)
from research.patterns.double_top.detector import DoubleTopDetector

# --------------------------------------------------------------------------
# Config-hash contract (§5.7 regression): recorded from the live code.
# --------------------------------------------------------------------------
#: DB/DT default config hash BEFORE this change added the two new keys.
HASH_BEFORE_NEW_KEYS = "d7d4c40092ee"
#: DB/DT default config hash with ONLY the §2.2 + §2.3 keys added (t2 alone).
HASH_AFTER_T2_KEYS_ONLY = "01de20eca41a"
#: DB/DT default config hash with t2 keys AND trend_gate_engineer's §2.1
#: trend-context keys (trend_context_enabled, trend_lookback_bars,
#: min_slope_atr, min_r2) both present.  Recorded so the hash chain is
#: attributable: d7d4c40092ee -> 01de20eca41a -> c82311b35696.
HASH_WITH_T2_AND_TREND_KEYS = "c82311b35696"

_DETECTORS = (DoubleBottomDetector, DoubleTopDetector)


def _frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    steps = rng.normal(0.0, 1.0, n).cumsum()
    close = 1900.0 + steps
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def _two_low_frame(span_bars: int, n: int = 500) -> pd.DataFrame:
    """Synthetic frame containing a genuine, detectable L-H-L double bottom
    whose two lows are ``span_bars`` bars apart.

    Built as a piecewise-linear path THROUGH the intended pivots (rather than a
    flat lead-in) so that :class:`SwingDetector` sees true local extremes at
    exactly ``i1`` / ``i2`` / ``i3`` with ``(L, H, L)`` kinds, the neckline
    cross fires a few bars after ``i3``, and the geometry clears
    ``max_equal_atr`` / ``min_depth_atr`` / ``min_separation_bars``.

    Verified pivot output for span=45: L@101 (1869.7), H@122 (1900.3),
    L@145 (1869.7) -> pattern_length 44.
    """
    base = 1900.0
    depth = 30.0
    prom = 15.0
    i1 = 100
    i3 = i1 + span_bars
    i2 = (i1 + i3) // 2
    if i3 + 20 >= n:
        raise ValueError("frame too short for the requested span")
    pts = [
        (0, base),
        (i1, base - depth * 0.1),
        (i1 + 1, base - depth),
        (i2, base),
        (i3, base - depth),
        (i3 + 1, base - depth),
        (i3 + 10, base + prom),
        (n - 1, base + prom),
    ]
    idx = np.arange(n, dtype=float)
    close = np.interp(
        idx,
        np.array([p[0] for p in pts], dtype=float),
        np.array([p[1] for p in pts], dtype=float),
    )
    high = close + 0.3
    low = close - 0.3
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=pd.date_range("2020-01-01", periods=n, freq="15min"),
    )


# --------------------------------------------------------------------------
# 1. Gate OFFSET semantics: both gates are opt-in safe
# --------------------------------------------------------------------------

@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_defaults_preserve_current_behaviour(det_cls) -> None:
    """The shipped defaults must be no-ops on ordinary data.

    §2.2 default 30 removes nothing on the measured history; §2.3 default 0.0
    turns the floor off entirely.  So a default run must equal an explicit
    `all gates off` run, bit for bit.
    """
    df = _frame()
    det = det_cls()
    default_events = det.detect(df, dict(det.config))
    default_discards = list(det.last_discards)

    off_cfg = dict(det.config)
    off_cfg["max_pattern_length_bars"] = 0
    off_cfg["min_rule_score"] = 0.0
    off_events = det.detect(df, off_cfg)

    assert [e.event_id for e in default_events] == [e.event_id for e in off_events]
    # §2.3 default is 0.0 => gate cannot fire; §2.2 default 30 => must not fire
    # on this frame either (nothing here exceeds 30 bars).
    assert default_discards == []


@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_gate_defaults_are_the_documented_values(det_cls) -> None:
    cfg = det_cls().get_default_config()
    assert cfg["max_pattern_length_bars"] == 30
    assert cfg["min_rule_score"] == 0.0
    # the "off" sentinels
    assert cfg["min_rule_score"] == 0.0


# --------------------------------------------------------------------------
# 2. §2.2 max_pattern_length_bars
# --------------------------------------------------------------------------

def test_too_long_pattern_is_dropped_with_pattern_too_long() -> None:
    """A structure whose extremes are ~44 bars apart must be rejected by a
    30-bar bound, with discard_reason == "pattern_too_long"."""
    df = _two_low_frame(span_bars=45)
    det = DoubleBottomDetector()
    cfg = dict(det.config)
    cfg["max_pattern_length_bars"] = 30
    events = det.detect(df, cfg)

    reasons = [d["discard_reason"] for d in det.last_discards]
    assert "pattern_too_long" in reasons, (
        f"expected a pattern_too_long rejection; got {reasons}"
    )
    too_long = [d for d in det.last_discards if d["discard_reason"] == "pattern_too_long"]
    assert all(d["pattern_length"] > 30 for d in too_long)
    assert all(d["max_pattern_length_bars"] == 30 for d in too_long)
    # the rejected structure is the one we planted
    assert too_long[0]["extreme1_bar"] == 101
    assert too_long[0]["extreme2_bar"] == 145
    assert too_long[0]["pattern_length"] == 44
    # and nothing survives that violates the bound
    assert len(events) < 1 or all(
        int(ev.attributes["extreme2_bar"]) - int(ev.attributes["extreme1_bar"]) <= 30
        for ev in events
    )


def test_short_valid_pattern_survives_the_length_gate() -> None:
    """The same synthetic structure at ~19 bars must pass a 30-bar bound."""
    df = _two_low_frame(span_bars=20)
    det = DoubleBottomDetector()
    cfg = dict(det.config)
    cfg["max_pattern_length_bars"] = 30
    events = det.detect(df, cfg)

    assert [d for d in det.last_discards if d["discard_reason"] == "pattern_too_long"] == []
    assert events, "a ~19-bar structure should survive a 30-bar bound"
    for ev in events:
        pl = int(ev.attributes["extreme2_bar"]) - int(ev.attributes["extreme1_bar"])
        assert pl <= 30


def test_length_gate_boundary_is_inclusive() -> None:
    """The gate fires only on strict `pattern_length > bound`.

    Pinned at the value the fixture actually produces, so the boundary is
    expressed in terms of real detected lengths rather than the requested span
    (the detected span can differ by a bar from the planted span).
    """
    df = _two_low_frame(span_bars=45)  # detected pattern_length == 44

    # bound == detected length -> must SURVIVE
    det = DoubleBottomDetector()
    ev_equal = det.detect(df, {**det.config, "max_pattern_length_bars": 44})
    assert [d for d in det.last_discards if d["discard_reason"] == "pattern_too_long"] == []
    assert ev_equal

    # bound == detected length - 1 -> must DROP
    det2 = DoubleBottomDetector()
    ev_below = det2.detect(df, {**det2.config, "max_pattern_length_bars": 43})
    assert [d for d in det2.last_discards if d["discard_reason"] == "pattern_too_long"]
    assert len(ev_below) == 0


def test_zero_disables_the_length_gate() -> None:
    df = _two_low_frame(span_bars=45)
    det = DoubleBottomDetector()
    cfg = dict(det.config)
    cfg["max_pattern_length_bars"] = 0
    events = det.detect(df, cfg)
    assert [d for d in det.last_discards if d["discard_reason"] == "pattern_too_long"] == []
    assert events, "with the gate off a 44-bar structure must be emitted"


def test_default_30_drops_a_structurally_too_long_candidate() -> None:
    """Defence-in-depth proof: the DEFAULT bound (30) does reject a genuinely
    over-long structure, even though no such structure exists in 8.3 years of
    XAUUSD M15 (which is why the gate is a measured no-op on real data)."""
    df = _two_low_frame(span_bars=45)
    det = DoubleBottomDetector()
    assert det.get_default_config()["max_pattern_length_bars"] == 30
    det.detect(df, dict(det.config))
    assert [d["discard_reason"] for d in det.last_discards] == ["pattern_too_long"]


def test_shared_bound_30_is_0_drop_for_both_patterns() -> None:
    """measurement_analyst's sweep: 30 is 0-drop for BOTH patterns (DB max 29,
    DT max 26), and 26 would drop 2 real DB events.  Pin that the ONE shared
    base-class default is the smallest value that is 0-drop for the union."""
    df = load_xauusd_or_skip()

    for det_cls in _DETECTORS:
        det = det_cls()
        # bound 30 => nothing dropped
        det.detect(df, {**det.config, "max_pattern_length_bars": 30})
        assert [d for d in det.last_discards if d["discard_reason"] == "pattern_too_long"] == [], (
            f"{det_cls.__name__}: bound 30 must drop nothing"
        )

    # DB binds: bound 26 drops real events, proving 30 is the tightest shared bound
    det_db = DoubleBottomDetector()
    det_db.detect(df, {**det_db.config, "max_pattern_length_bars": 26})
    assert [d for d in det_db.last_discards if d["discard_reason"] == "pattern_too_long"], (
        "bound 26 should drop 2 real double_bottom events (max span 29)"
    )

    # DT never binds at its own max
    det_dt = DoubleTopDetector()
    det_dt.detect(df, {**det_dt.config, "max_pattern_length_bars": 26})
    assert [d for d in det_dt.last_discards if d["discard_reason"] == "pattern_too_long"] == [], (
        "bound 26 should be 0-drop for double_top (max span 26)"
    )


# --------------------------------------------------------------------------
# 3. §2.3 min_rule_score  (detector side)
# --------------------------------------------------------------------------

def test_none_and_zero_are_behaviourally_identical_off_states() -> None:
    """The "off" state must be unambiguous: None and 0.0 must both mean
    "no drop".  (The code does `float(cfg.get(...) or 0.0)`, so None collapses
    to 0.0 -- pinned here so a future refactor cannot split them.)"""
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    ev_zero = det.detect(df, {**det.config, "min_rule_score": 0.0})
    d_zero = list(det.last_discards)
    ev_none = det.detect(df, {**det.config, "min_rule_score": None})
    d_none = list(det.last_discards)

    assert [e.event_id for e in ev_zero] == [e.event_id for e in ev_none]
    assert d_zero == d_none == []


def test_floor_only_bites_once_a_nonzero_value_is_configured() -> None:
    """Fail-preserving semantics of the DEFAULT.

    An event with an unknown/zero rule_score still passes a 0.0 floor
    (`0.0 < 0.0` is False), which is the correct conservatively-off behaviour
    for a default.  The floor therefore only ever removes events once a
    non-zero value is set.  (The live-side §2.3 gate documents the opposite
    intent for CONFIGURED floors in a different module -- deliberately, so that
    a misconfigured floor fails closed there while the detector default here
    fails open.)"""
    df = _two_low_frame(span_bars=20)
    det = DoubleBottomDetector()
    # default/off floor: everything passes
    ev_off = det.detect(df, {**det.config, "min_rule_score": 0.0})
    assert ev_off, "with the floor off the emitter must still produce events"
    assert [d for d in det.last_discards if d["discard_reason"] == "low_rule_score"] == []

    # a floor above this event's score must remove it -> proves the floor bites
    score = ev_off[0].rule_score
    det2 = DoubleBottomDetector()
    ev_on = det2.detect(df, {**det2.config, "min_rule_score": score + 0.01})
    assert ev_on == []
    assert [d["discard_reason"] for d in det2.last_discards] == ["low_rule_score"]


def test_rule_score_floor_drops_low_scores_and_keeps_high_scores() -> None:
    """With a floor of 0.6: every surviving event has rule_score >= 0.6 and
    every rejection carries a sub-floor score and the right reason."""
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    cfg = dict(det.config)
    cfg["min_rule_score"] = 0.6
    events = det.detect(df, cfg)

    assert events, "the 0.6 floor must not wipe out every event"
    for ev in events:
        assert ev.rule_score >= 0.6

    rejected = [d for d in det.last_discards if d["discard_reason"] == "low_rule_score"]
    assert rejected, "a 0.6 floor is expected to reject some candidates"
    for d in rejected:
        assert d["rule_score"] < 0.6
        assert d["min_rule_score"] == 0.6


def test_rule_score_floor_zero_is_a_true_no_op() -> None:
    """0.0 must disable the gate: observed rule_score has a positive minimum,
    so a strict `<` against 0.0 can never fire."""
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    det.detect(df, {**det.config, "min_rule_score": 0.0})
    assert [d for d in det.last_discards if d["discard_reason"] == "low_rule_score"] == []

    # and the observed minimum really is far above zero (measured ~0.51),
    # which is WHY 0.0 is a safe "off" sentinel for this formula.
    det2 = DoubleBottomDetector()
    evs = det2.detect(df, {**det2.config, "min_rule_score": 0.0})
    assert min(e.rule_score for e in evs) > 0.4


def test_rule_score_floor_is_monotone() -> None:
    """Raising the floor may only remove events, never add them."""
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    counts = []
    for thr in (0.0, 0.55, 0.6, 0.65, 0.7, 0.8):
        evs = det.detect(df, {**det.config, "min_rule_score": thr})
        counts.append(len(evs))
    assert counts == sorted(counts, reverse=True), counts


def test_gates_are_applied_before_nms_so_no_new_events_appear() -> None:
    """The gates feed NMS (they run before `_nms_structure_overlap`), so
    enabling them can only shrink the emitted set -- never surface a structure
    that the ungated run had suppressed."""
    df = load_xauusd_or_skip()

    def windows(events):
        return {
            (int(e.attributes["extreme1_bar"]), int(e.attributes["extreme2_bar"]))
            for e in events
        }

    for det_cls in _DETECTORS:
        det = det_cls()
        off = det.detect(df, {**det.config, "max_pattern_length_bars": 0, "min_rule_score": 0.0})
        on = det.detect(df, {**det.config, "max_pattern_length_bars": 30, "min_rule_score": 0.6})
        assert not (windows(on) - windows(off)), "gate must not create new structures"
        assert len(on) <= len(off)


# --------------------------------------------------------------------------
# 4. Determinism
# --------------------------------------------------------------------------

@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_gates_are_deterministic(det_cls) -> None:
    df = _frame()
    cfg_over = {"max_pattern_length_bars": 30, "min_rule_score": 0.0}
    runs = []
    for _ in range(3):
        det = det_cls()
        evs = det.detect(df, {**det.config, **cfg_over})
        runs.append(
            [
                (e.event_id, e.detect_time, float(e.rule_score), int(e.attributes["pattern_length"]))
                for e in evs
            ]
        )
    assert runs[0] == runs[1] == runs[2]


def test_discard_reasons_are_deterministic() -> None:
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    cfg = {**det.config, "min_rule_score": 0.6}
    det.detect(df, cfg)
    first = [d["discard_reason"] for d in det.last_discards]
    det.detect(df, cfg)
    second = [d["discard_reason"] for d in det.last_discards]
    assert first == second


# --------------------------------------------------------------------------
# 5. F-01 defect: pattern_length is now actually populated
# --------------------------------------------------------------------------

@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_pattern_length_feature_is_actually_emitted(det_cls) -> None:
    """`pattern_length` is declared AVAILABLE_AT_DETECT in feature_schema; the
    t1 finding F-01 was that it was never populated.  Pin that it now is, and
    that it equals extreme2_bar - extreme1_bar."""
    df = load_xauusd_or_skip()
    det = det_cls()
    events = det.detect(df, dict(det.config))
    assert events
    for ev in events:
        assert "pattern_length" in ev.attributes, "F-01 regression: feature not emitted"
        expected = int(ev.attributes["extreme2_bar"]) - int(ev.attributes["extreme1_bar"])
        assert ev.attributes["pattern_length"] == expected
        assert isinstance(ev.attributes["pattern_length"], int)

    declared = {f.name for f in det_cls.feature_schema}
    assert "pattern_length" in declared


@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_every_detect_feature_is_populated(det_cls) -> None:
    """Scope guard for the F-01 defect class.

    ``pattern_length`` is fixed by this change.  ``atr`` and ``symmetry_ratio``
    are STILL declared AVAILABLE_AT_DETECT and still never written to
    ``attributes`` — the same defect class as F-01, but outside the §2.2/§2.3
    scope of this task, so they are pinned here explicitly rather than
    silently tolerated.  When someone fixes them, this test fails and points
    at the fix.
    """
    from research.core.contracts import AVAILABLE_AT_DETECT

    df = load_xauusd_or_skip()
    det = det_cls()
    events = det.detect(df, dict(det.config))
    assert events
    detect_features = {
        f.name for f in det_cls.feature_schema if f.available_at == AVAILABLE_AT_DETECT
    }
    # known-unpopulated, pre-existing, OUT OF SCOPE for t2
    known_gaps = {"atr", "symmetry_ratio"}
    for ev in events:
        missing = detect_features - set(ev.attributes)
        assert missing <= known_gaps, (
            f"new AVAILABLE_AT_DETECT feature(s) not populated: {missing - known_gaps}"
        )
        # the one we DO own must always be there
        assert "pattern_length" in ev.attributes


def test_atr_and_symmetry_ratio_gaps_are_still_open() -> None:
    """Documents the two pre-existing F-01-class defects found while testing.

    If this test starts FAILING, someone populated the features — good news;
    update ``known_gaps`` in the test above and remove this one.
    """
    df = load_xauusd_or_skip()
    det = DoubleBottomDetector()
    events = det.detect(df, dict(det.config))
    assert events
    assert "atr" not in events[0].attributes
    assert "symmetry_ratio" not in events[0].attributes


def load_xauusd_or_skip() -> pd.DataFrame:
    """Real XAUUSD M15 frame (window A, matching the measurement window).

    Skips — loudly, never silently passing — if the parquet is unavailable.
    """
    try:
        from research.multi_backtest.runner import load_symbol_frame

        df = load_symbol_frame("XAUUSD", start="2018-06-01", end="2026-09-03")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"XAUUSD M15 frame unavailable: {exc}")
    if df is None or len(df) == 0:  # pragma: no cover
        pytest.skip("XAUUSD M15 frame empty")
    return df


# --------------------------------------------------------------------------
# 6. config_hash regression (pinned, so a future edit cannot move it silently)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_default_config_hash_is_pinned(det_cls) -> None:
    """Adding the §2.2/§2.3 keys MUST change the hash exactly once.

    `compute_config_hash` hashes the whole canonical config dict
    (research/core/config_hash.py:71-72), so any new key moves it for every
    DB/DT event.  This test records the before/after values so a future
    silent change is caught instead of corrupting artifact lineage.

    The expected value is DERIVED from which key-sets are currently present,
    so a concurrent landing by another owner (e.g. trend_gate_engineer's §2.1
    keys) is attributable rather than an opaque failure.
    """
    det = det_cls()
    current = config_hash_for_detector(det, det.config)
    cfg = det.get_default_config()

    t2_keys = {"max_pattern_length_bars", "min_rule_score"}
    trend_keys = {
        "trend_context_enabled",
        "trend_lookback_bars",
        "min_slope_atr",
        "min_r2",
    }
    has_t2 = t2_keys <= set(cfg)
    has_trend = trend_keys <= set(cfg)

    if has_t2 and has_trend:
        expected = HASH_WITH_T2_AND_TREND_KEYS
    elif has_t2:
        expected = HASH_AFTER_T2_KEYS_ONLY
    else:  # pragma: no cover - t2 keys are always present after this change
        expected = HASH_BEFORE_NEW_KEYS

    assert current != HASH_BEFORE_NEW_KEYS, (
        "t2 keys are present but the hash did not move — compute_config_hash "
        "is no longer hashing the whole config dict"
    )
    assert current == expected, (
        f"config_hash drifted to {current}; expected {expected} for key-sets "
        f"(t2={has_t2}, trend={has_trend}). If you intentionally changed the "
        "default config, update these pins AND regenerate DB/DT artifacts "
        "(train_summary.json / features.json)."
    )
    assert len(current) == 12


@pytest.mark.parametrize("det_cls", _DETECTORS)
def test_each_new_gate_key_individually_participates_in_the_hash(det_cls) -> None:
    """Each new key must be hash-participating, so drift is attributable."""
    det = det_cls()
    base = dict(det.get_default_config())
    base_hash = compute_config_hash(base)

    for key, new_value in (
        ("max_pattern_length_bars", 55),
        ("min_rule_score", 0.6),
    ):
        assert key in base, f"{key} must be present in get_default_config"
        mutated = dict(base)
        mutated[key] = new_value
        assert compute_config_hash(mutated) != base_hash, f"{key} did not move the hash"


def test_gate_engineer_and_trend_gate_key_sets_are_disjoint() -> None:
    """Attribution guard shared with trend_gate_engineer (t4).

    Their set and mine must be disjoint, so a hash move can be traced to one
    owner.  If either side ever takes over the other's key, fail loudly.
    """
    mine = {"max_pattern_length_bars", "min_rule_score"}
    theirs = {"trend_context_enabled", "trend_lookback_bars", "min_slope_atr", "min_r2"}
    assert mine & theirs == set()

    cfg = DoubleBottomDetector().get_default_config()
    # whichever set is present must be present as a whole (no half-wired state)
    for name, keys in (("gate_engineer", mine), ("trend_gate_engineer", theirs)):
        present = {k for k in keys if k in cfg}
        assert present in (set(), keys), f"half-wired {name} keys: {present}"
    # after both land, BOTH sets must be complete
    assert mine <= set(cfg), f"gate_engineer keys incomplete: {mine - set(cfg)}"


def test_config_hash_requires_version_key() -> None:
    """§6.2 contract: a config without "version" must raise."""
    with pytest.raises(ValueError, match="version"):
        compute_config_hash({"min_rule_score": 0.6})


def test_feature_schema_version_unchanged_by_gates() -> None:
    """§2.2/§2.3 add no FEATURES (they only gate existing ones), so the
    feature-schema version must stay at the frozen value.  Bumping it would
    invalidate every trained artifact for no semantic reason."""
    assert FEATURE_SCHEMA_VERSION == "double-v1.0"


# --------------------------------------------------------------------------
# 7. Defaults do not move the golden LSW path
# --------------------------------------------------------------------------

def test_golden_lsw_config_untouched_by_db_dt_keys() -> None:
    """The §2.2/§2.3 keys belong to the DOUBLE detector's config only.

    LiquiditySweep keeps its own get_default_config, so its hash must not
    contain (or be affected by) the DB/DT gate keys — the golden LSW test
    asserts self-consistency and must stay bit-identical.
    """
    from research.patterns.liquidity_sweep.detector import LiquiditySweepDetector

    lsw_cfg = LiquiditySweepDetector().get_default_config()
    assert "max_pattern_length_bars" not in lsw_cfg
    assert "min_rule_score" not in lsw_cfg
