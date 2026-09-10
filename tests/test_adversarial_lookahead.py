"""
test_adversarial_lookahead.py — CI-enforced adversarial look-ahead & bias audit
(Agent 7, REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §3, §13 A7, §15.6).

Every detector PR is reviewed against the §3.1-3.5 checklist; this suite is
the *automated* half of that review and is **permanently wired into CI**
(``testpaths = ["tests"]`` picks it up every run; the module carries the
registered ``no_lookahead`` marker so it also runs under ``-m no_lookahead``).

The suite proves the CI net catches real look-ahead injections:

1. **shift(-1) data leak** — a "detect" feature computed from ``close[t+1]``
   (the classic ``df.close.shift(-1)`` slip): caught by the value-provenance
   check "reported feature == recompute from bars <= known_at", which fails
   for a poisoned lab detector at every leaked bar and passes for the honest
   implementation; the real plugins are held to the truncation-invariance
   form (events on ``df[:m]`` must not change when the frame is extended).
2. **confirm-bar close used in a detect feature** — an event that reads the
   confirmation candle's close but stamps ``known_at`` one bar earlier
   (i.e. "I knew the confirm close at detect time"): caught by the runtime
   validator (:class:`research.core.causal_checks.CausalityViolation`) and by
   the exact gate ``NoLookaheadTestBase.test_all_feature_stamps_within_known_at``
   executes.
3. **timeline / stamp attacks** — confirm before detect, ``known_at`` before
   confirm, a confirm-available feature on an unconfirmed event, a
   ``uses_future_data=True`` declaration: every one must raise.
4. **staleness (§3.3)** — the configurable ``max_bars_between_detect_and_confirm``
   window hard-discards: a 0-bar window yields zero events; every emitted
   event honours ``confirm - detect <= max_wait``.

Then the three audits demanded by §13 A7:

* **labeling audit** — the forward label window is strictly AFTER the entry
  bar (``s = entry_bar + 1``), the features zone ends before entry (never
  touches the label window), purge/embargo keep train/OOS disjoint, and ESS
  non-overlap holds.
* **backtest cost audit** — the exact sweep cost model
  (``cost_r = 2*(half_spread+slippage)/risk + commission_r``,
  ``net = gross - cost``) is asserted against the real labeling module, and
  the guard that rejects nonzero legacy ``entry.spread_price`` /
  ``entry.slippage_price`` (silent cost understatement) is proven live.
* **backtest ≡ live path audit** — the classes the benchmark harness wraps
  are the exact classes exported for the live registry
  (``DETECTOR_CLASS`` / ``PATTERN_ENTRY``), and the sweep golden test
  (plugin bit-identical to the legacy ``signal_engine_v2`` chain) is present
  and wired into the no-lookahead template.

Run::

    cd trading_v3 && /tmp/ptv2_venv/bin/python -m pytest tests/test_adversarial_lookahead.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from research.core.causal_checks import (
    CausalityViolation,
    resolve_feature_stamp,
    validate_causality,
    validate_event_timeline,
)
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    AVAILABLE_AT_DETECT,
    AVAILABLE_AT_ENTRY,
    BasePatternDetector,
    PatternEvent,
    PatternFeature,
)
from research.core.pattern_synthesizer import PatternSynthesizer
from research.patterns.double_bottom.dataset import (
    LabeledEvent,
    effective_sample_size,
    label_events,
    split_train_oos,
)
from research.patterns.double_bottom.detector import DoubleBottomDetector
from research.patterns.double_top.detector import DoubleTopDetector
from research.patterns.falling_wedge.detector import FallingWedgeDetector
from research.patterns.head_shoulders.detector import HeadShouldersDetector
from research.patterns.inverse_head_shoulders.detector import (
    InverseHeadShouldersDetector,
)
from research.patterns.liquidity_sweep.detector import LiquiditySweepDetector
from research.patterns.rising_wedge.detector import RisingWedgeDetector
from tests.no_lookahead_base import NoLookaheadTestBase

#: Registered CI marker (pyproject [tool.pytest.ini_options].markers).
pytestmark = pytest.mark.no_lookahead

_HERE = Path(__file__).resolve().parent
_TV3 = _HERE.parent
_XAUUSD_PARQUET = (
    _TV3
    / "research" / "patterns" / "liquidity_sweep"
    / "data" / "processed" / "xauusd_m15.parquet"
)

#: Seeds that reliably yield >= 1 event per classical plugin (probed).
_CLASSICAL_CASES: list[tuple[type[BasePatternDetector], str]] = [
    (DoubleBottomDetector, "double_bottom"),
    (DoubleTopDetector, "double_top"),
    (FallingWedgeDetector, "falling_wedge"),
    (HeadShouldersDetector, "head_shoulders"),
    (InverseHeadShouldersDetector, "inverse_head_shoulders"),
]

_ALL_PLUGINS: list[type[BasePatternDetector]] = [
    LiquiditySweepDetector,
    DoubleBottomDetector,
    DoubleTopDetector,
    RisingWedgeDetector,
    FallingWedgeDetector,
    HeadShouldersDetector,
    InverseHeadShouldersDetector,
]


def _synth_frame(pattern: str, seed: int, n_bars: int = 170) -> pd.DataFrame:
    return PatternSynthesizer().generate(pattern, n_bars, noise_sigma=0.05, seed=seed)[0]


# ---------------------------------------------------------------------------
# 1. Injected look-ahead must be caught (§13 A7 item 2)
# ---------------------------------------------------------------------------

class _BarShiftDetector(BasePatternDetector):
    """Adversarial lab detector with a tunable per-bar data shift.

    ``shift == 0`` is the honest implementation: the scoring feature at bar
    ``t`` reads ``close[t]``.  ``shift == -1`` is the injected bug: the
    feature reads ``close[t+1]`` — exactly ``df.close.shift(-1)`` — while the
    event timeline still claims the bar is known at ``t`` (the classic leak,
    invisible to any per-event stamp check because the value provenance is
    hidden inside ``attributes``).

    ``lab_max_t`` bounds the emitted bars; ``recompute_from_causal_data`` is
    the value-provenance oracle the CI check uses: it re-computes the feature
    from the bars *causally available at the event's known_at* — the honest
    detector reproduces its own value, the poisoned one cannot.
    """

    name = "bar_shift_lab"
    version = "0.1"
    short_name = "BS"

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(
            "leak_close_shift",
            "float",
            AVAILABLE_AT_DETECT,
            False,
            "close[t+shift] exposed as a detect-time feature (adversarial lab)",
        ),
    ]

    def __init__(self, shift: int = 0, lab_max_t: int = 95) -> None:
        self.shift = int(shift)
        self.lab_max_t = int(lab_max_t)

    def get_default_config(self) -> dict[str, Any]:
        return {"version": self.version}

    def recompute_from_causal_data(self, df: pd.DataFrame, t: int) -> float:
        """The feature value computable from the bars causally available at
        bar ``t`` (i.e. a prefix ending at ``t``)."""
        prefix = df.iloc[: t + 1]
        leaked = prefix["close"].shift(self.shift).to_numpy(dtype=float)
        return float(leaked[t])

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        leaked = df["close"].shift(self.shift).to_numpy(dtype=float)
        last = min(self.lab_max_t, len(df) - 2)
        return [
            PatternEvent(
                event_id=f"BS-{t:06d}",
                pattern_name=self.name,
                pattern_version=self.version,
                symbol="X",
                timeframe="M15",
                direction="bullish",
                detect_time=df.index[t],
                confirm_time=df.index[t],
                entry_time=df.index[t + 1],
                entry_price=float(df["open"].iloc[t + 1]),
                stop_price=float(df["low"].iloc[t]) - 1.0,
                target_price=float(df["open"].iloc[t + 1]) + 2.0,
                rule_score=0.5,
                attributes={"leak_close_shift": float(leaked[t])},
            )
            for t in range(5, last + 1)
        ]


class _ConfirmCloseInDetectDetector(BasePatternDetector):
    """Adversarial lab detector: uses the CONFIRM bar's close in a feature but
    stamps ``known_at`` at DETECT time (one bar earlier than the data it
    read).  This is the §13 A7 attack "dùng close của confirm bar trong detect
    features" — the runtime validator must reject it."""

    name = "confirm_close_in_detect_lab"
    version = "0.1"
    short_name = "CD"

    feature_schema: ClassVar[list[PatternFeature]] = [
        PatternFeature(
            "confirm_close_leak",
            "float",
            AVAILABLE_AT_CONFIRM,
            False,
            "honest declaration: only known at the confirm bar",
        ),
    ]

    def get_default_config(self) -> dict[str, Any]:
        return {"version": self.version}

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        t = 5
        return [
            PatternEvent(
                event_id="CD-000001",
                pattern_name=self.name,
                pattern_version=self.version,
                symbol="X",
                timeframe="M15",
                direction="bullish",
                detect_time=df.index[t],
                confirm_time=df.index[t + 1],
                entry_time=df.index[t + 2],
                known_at=df.index[t],  # ← the leak: claims detect-time knowledge
                entry_price=float(df["open"].iloc[t + 2]),
                stop_price=float(df["low"].iloc[t]) - 1.0,
                target_price=float(df["open"].iloc[t + 2]) + 2.0,
                attributes={"confirm_close_leak": float(df["close"].iloc[t + 1])},
            )
        ]


def _features_reproducible_under_known_at(det: _BarShiftDetector, frame: pd.DataFrame) -> None:
    """Assert every event's features are reproducible from bars <= known_at.

    Re-computes each event's scoring feature from the data causally available
    at its known_at bar and requires it to equal the feature the detector
    reported.  A causal detector's feature at bar ``t`` is a function of bars
    <= t, so this must hold; a ``shift(-1)`` leak makes the reported feature
    read ``close[t+1]`` which NO prefix ending at ``t`` can reproduce — the
    mismatch is caught at every leaked bar, not just at a truncation boundary.
    """
    events = sorted(det.detect(frame, det.get_default_config()), key=lambda e: e.detect_time)
    assert events, "lab detector must emit events"
    for ev in events:
        t = int(frame.index.get_loc(ev.detect_time))
        causal_value = det.recompute_from_causal_data(frame, t)
        reported = float(ev.attributes["leak_close_shift"])
        assert np.isclose(causal_value, reported, equal_nan=True), (
            f"event {ev.event_id}: reported feature {reported} is NOT "
            f"reproducible from bars <= known_at bar {t} (causal recompute "
            f"gives {causal_value}) — the feature reads future data"
        )


def test_shift_minus_one_injection_is_caught_by_reproducibility_check() -> None:
    """The exact ``shift(-1)`` leak (feature at bar t reads close[t+1]) must
    fail the value-provenance check "reported feature == recompute from bars
    <= known_at" at every bar — the same check the real detectors' event sets
    are held to via truncation invariance (CI bắt được)."""
    frame = _synth_frame("double_bottom", seed=0, n_bars=160)
    honest = _BarShiftDetector(shift=0, lab_max_t=95)
    poisoned = _BarShiftDetector(shift=-1, lab_max_t=95)

    # Positive control: the honest (shift=0) detector reproduces its features
    # from the causally available bars (bars <= known_at).
    _features_reproducible_under_known_at(honest, frame)

    # The injected shift(-1) leak fails the identical check → CI catches it.
    with pytest.raises(AssertionError):
        _features_reproducible_under_known_at(poisoned, frame)


def test_confirm_bar_close_in_detect_feature_is_caught_by_validator() -> None:
    """The §13 A7 attack — a feature reading the confirm bar's close while the
    event claims detect-time knowledge — must raise CausalityViolation, and
    the NoLookaheadTestBase stamp gate must fail identically."""
    frame = _synth_frame("double_bottom", seed=0)
    det = _ConfirmCloseInDetectDetector()
    events = det.detect(frame, det.get_default_config())

    with pytest.raises(CausalityViolation):
        validate_causality(events, feature_schema=det.feature_schema)

    # The exact check NoLookaheadTestBase.test_all_feature_stamps_within_known_at
    # runs (resolve_feature_stamp <= known_at) must fail on the poisoned event.
    with pytest.raises(AssertionError):
        for ev in events:
            for feat in det.feature_schema:
                stamp = resolve_feature_stamp(feat, ev)
                assert stamp <= ev.known_at_ts, (
                    f"{ev.event_id}: feature '{feat.name}' available_at {stamp} "
                    f"> known_at {ev.known_at_ts} — would read future data"
                )


def _single_event_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 40
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.1, n),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )


def test_confirm_before_detect_rejected() -> None:
    """Timeline attack: confirm_time < detect_time must raise."""
    df = _single_event_frame()
    ev = PatternEvent(
        event_id="TS-1",
        pattern_name="liquidity_sweep",
        pattern_version="2.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=df.index[10],
        confirm_time=df.index[9],  # ← impossible: confirmation before setup
        entry_time=df.index[11],
        entry_price=100.0,
        stop_price=99.0,
        target_price=103.0,
    )
    with pytest.raises(CausalityViolation):
        validate_event_timeline(ev)


def test_known_at_before_confirm_rejected() -> None:
    """Stamp attack: known_at shifted one bar before confirm must raise."""
    df = _single_event_frame()
    ev = PatternEvent(
        event_id="TS-2",
        pattern_name="liquidity_sweep",
        pattern_version="2.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=df.index[10],
        confirm_time=df.index[12],
        entry_time=df.index[13],
        known_at=df.index[11],  # ← claims the confirm bar is unknowable yet
        entry_price=100.0,
        stop_price=99.0,
        target_price=103.0,
    )
    with pytest.raises(CausalityViolation):
        validate_causality([ev], feature_schema=None)


def test_feature_availability_stamp_missing_rejected() -> None:
    """A feature declared available_at=confirm/entry on an event without that
    stamp must raise (detector mis-declaration is caught)."""
    df = _single_event_frame()
    schema_confirm = [
        PatternFeature("f", "float", AVAILABLE_AT_CONFIRM, False, "needs confirm")
    ]
    ev_no_confirm = PatternEvent(
        event_id="TS-3",
        pattern_name="double_bottom",
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=df.index[10],
        confirm_time=None,  # ← unconfirmed event claiming a confirm feature
        entry_price=100.0,
        stop_price=99.0,
    )
    with pytest.raises(CausalityViolation):
        validate_causality([ev_no_confirm], feature_schema=schema_confirm)

    schema_entry = [
        PatternFeature("g", "float", AVAILABLE_AT_ENTRY, False, "needs entry")
    ]
    ev_no_entry = PatternEvent(
        event_id="TS-4",
        pattern_name="double_bottom",
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=df.index[10],
        confirm_time=df.index[11],
        entry_time=None,  # ← no entry planned but feature needs entry data
        entry_price=100.0,
        stop_price=99.0,
    )
    with pytest.raises(CausalityViolation):
        validate_causality([ev_no_entry], feature_schema=schema_entry)


def test_future_data_feature_declaration_rejected() -> None:
    """uses_future_data=True is rejected at declaration — the schema cannot
    even instantiate such a feature, so no detector can smuggle one into CI.
    (Every plugin schema is additionally verified by
    test_no_plugin_schema_declares_future_data.)"""
    with pytest.raises(ValueError):
        PatternFeature("bad", "float", AVAILABLE_AT_DETECT, True, "forbidden")
    # even an invalid available_at is rejected at declaration
    with pytest.raises(ValueError):
        PatternFeature("bad_stamp", "float", "after_known_at", False, "")


# ---------------------------------------------------------------------------
# 2. Real detectors must be clean (§3.4 / §13 A7 item 2, acceptance 6)
# ---------------------------------------------------------------------------

_XAUUSD_SLICE = 8000


@pytest.fixture(scope="module")
def sweep_real_events() -> tuple[list[PatternEvent], pd.DataFrame]:
    """Sweep plugin on a real XAUUSD M15 slice (fast, ~1s)."""
    assert _XAUUSD_PARQUET.exists(), f"XAUUSD parquet missing: {_XAUUSD_PARQUET}"
    df = pd.read_parquet(_XAUUSD_PARQUET)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"])
        df = df.drop(columns=["timestamp"])
    df = df.sort_index().iloc[-_XAUUSD_SLICE:]
    return LiquiditySweepDetector().detect(df, {}), df


def test_sweep_zero_causality_violation_on_real_data(
    sweep_real_events: tuple[list[PatternEvent], pd.DataFrame],
) -> None:
    """Acceptance 6: no latent CausalityViolation in the sweep detector —
    runtime validator passes, timeline coherent, entry == confirm + 1."""
    events, df = sweep_real_events
    assert len(events) >= 1, "XAUUSD slice must yield sweep events"
    det = LiquiditySweepDetector()
    validate_causality(events, feature_schema=det.feature_schema)  # must not raise
    for ev in events:
        assert ev.confirm_time is not None
        assert df.index.get_loc(ev.entry_time) == df.index.get_loc(ev.confirm_time) + 1
        assert ev.known_at_ts == ev.confirm_time  # confirm dominates


def test_truncation_invariance_classical_detectors() -> None:
    """Appending bars must never change events whose confirm bar lies inside
    the prefix (catches any shift(-1)-style future dependence)."""
    for det_class, pattern in _CLASSICAL_CASES:
        det = det_class()
        frame = _synth_frame(pattern, seed=0)
        m = 90
        assert _prefix_events(det, frame, m) == _prefix_events(det, frame, m + 25), (
            f"{det_class.__name__}: events inside the prefix changed when the "
            "frame was extended — future-bar dependence (look-ahead)"
        )


def _prefix_events(det: BasePatternDetector, frame: pd.DataFrame, m: int) -> tuple:
    """Events fully inside the prefix ``df[:m]`` (confirm bar <= m - margin)."""
    margin = 15
    df = frame.iloc[:m]
    events = sorted(det.detect(df, det.get_default_config()), key=lambda e: e.detect_time)
    stable: list[tuple] = []
    for ev in events:
        conf_bar = df.index.get_loc(ev.confirm_time)
        if conf_bar <= m - margin:
            stable.append(
                (
                    ev.event_id,
                    ev.detect_time,
                    ev.confirm_time,
                    ev.entry_time,
                    round(ev.entry_price, 9),
                    round(ev.stop_price, 9),
                    round(ev.target_price if ev.target_price is not None else float("nan"), 9),
                    round(ev.rule_score, 9),
                )
            )
    return tuple(stable)


def test_no_plugin_schema_declares_future_data() -> None:
    """Every plugin's feature schema is causal by declaration (§3.4)."""
    for det_class in _ALL_PLUGINS:
        det = det_class()
        for feat in det.feature_schema:
            assert not feat.uses_future_data, (
                f"{det_class.__name__}: feature '{feat.name}' uses_future_data"
            )
            assert feat.available_at in (AVAILABLE_AT_DETECT, AVAILABLE_AT_CONFIRM, AVAILABLE_AT_ENTRY)


# ---------------------------------------------------------------------------
# 3. §3.3 Staleness window (configurable, hard discard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("det_class", "pattern"),
    [
        (DoubleBottomDetector, "double_bottom"),
        (HeadShouldersDetector, "head_shoulders"),
        (FallingWedgeDetector, "falling_wedge"),
    ],
)
def test_staleness_window_hard_discards(det_class: type, pattern: str) -> None:
    """§3.3: a 0-bar window discards everything; every emitted event honours
    confirm - detect <= max_bars_between_detect_and_confirm."""
    det = det_class()
    cfg = det.get_default_config()
    max_wait = int(cfg["max_bars_between_detect_and_confirm"])
    frame = _synth_frame(pattern, seed=0)

    zero_cfg = dict(cfg)
    zero_cfg["max_bars_between_detect_and_confirm"] = 0
    assert det.detect(frame, zero_cfg) == [], (
        f"{det_class.__name__}: 0-bar staleness window still emitted events"
    )

    events = det.detect(frame, cfg)
    assert len(events) >= 1, f"{det_class.__name__}: default cfg on seed 0 must fire"
    for ev in events:
        conf_bar = frame.index.get_loc(ev.confirm_time)
        det_bar = frame.index.get_loc(ev.detect_time)
        assert conf_bar - det_bar <= max_wait, (
            f"{det_class.__name__}: confirm {conf_bar} - detect {det_bar} "
            f"exceeds staleness window {max_wait}"
        )


# ---------------------------------------------------------------------------
# 4. Labeling audit — no leakage between forward window and features (§13 A7
#    item 3)
# ---------------------------------------------------------------------------

def _labeled_double_events(n_bars: int = 240) -> tuple[pd.DataFrame, list]:
    """Double-bottom events + their forward labels on a synthetic frame."""
    det = DoubleBottomDetector()
    labeled: list = []
    frame = None
    for seed in range(20):
        candidate = _synth_frame("double_bottom", seed=seed, n_bars=n_bars)
        events = det.detect(candidate, det.get_default_config())
        labeled = label_events(candidate, events, horizon_bars=48, target_r=1.5)
        if labeled:
            frame = candidate
            break
    assert frame is not None and labeled, "no seed produced a labelable event"
    return frame, labeled


def test_label_window_strictly_after_entry() -> None:
    """Labels read ONLY bars (entry_bar, entry_bar + H] — never bar <= entry —
    and the features zone (<= detect < entry) never touches the label zone."""
    frame, labeled = _labeled_double_events()
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)

    for le in labeled:
        eb = le.entry_bar
        H = le.horizon_bars
        assert H == 48
        s, e = eb + 1, eb + H  # strictly after entry — no bar <= eb is read
        assert s > eb
        mfe = float(np.max(high[s : e + 1])) - le.entry_price
        mae = le.entry_price - float(np.min(low[s : e + 1]))
        expected_mfe = mfe / le.risk
        expected_mae = mae / le.risk
        expected_close = (float(close[e]) - le.entry_price) / le.risk
        assert np.isclose(le.mfe_ratio, expected_mfe, atol=1e-9)
        assert np.isclose(le.mae_ratio, expected_mae, atol=1e-9)
        assert np.isclose(le.close_ratio, expected_close, atol=1e-9)
        # features end before the label window begins
        det_bar = frame.index.get_loc(le.detect_time)
        assert det_bar < eb, (
            f"event {le.event_id}: detect {det_bar} >= entry {eb} — features "
            "would overlap the forward label window"
        )


def test_purge_and_embargo_keep_train_oos_disjoint() -> None:
    """Every train event's forward window must end purge+embargo bars before
    the OOS boundary; events too close to the boundary are purged."""
    labeled = [
        LabeledEvent(
            event_id=f"SYN-{i:04d}",
            pattern_name="double_bottom",
            direction="bullish",
            detect_time=pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=15 * (10 + 20 * i)),
            confirm_time=pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=15 * (11 + 20 * i)),
            entry_time=pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=15 * (12 + 20 * i)),
            entry_bar=10 + 20 * i,
            entry_price=100.0 + 0.1 * i,
            stop_price=99.0 + 0.1 * i,
            risk=1.0,
            rule_score=0.5,
            config_hash="0123456789ab",
            horizon_bars=72,
            mfe_ratio=0.0,
            mae_ratio=0.0,
            close_ratio=0.0,
            label=0,
        )
        for i in range(200)
    ]
    train, oos = split_train_oos(labeled, oos_frac=0.40, purge_bars=96, embargo_bars=24)
    assert train and oos
    boundary_bar = oos[0].entry_bar
    for le in train:
        assert (
            le.entry_bar + le.horizon_bars + 96 + 24 <= boundary_bar
        ), f"{le.event_id}: forward window overlaps the OOS boundary (no purge/embargo)"
    # the purge gate is actually active: a candidate close to the boundary must
    # have been removed from train
    oos_start_index = len(labeled) - len(oos)
    near_boundary = labeled[oos_start_index - 1]
    assert near_boundary.entry_bar + near_boundary.horizon_bars + 96 + 24 > boundary_bar
    assert near_boundary not in train


def test_effective_sample_size_non_overlap() -> None:
    """ESS keeps events at least ``window_bars`` apart on the entry timeline."""
    labeled = [
        LabeledEvent(
            event_id=f"E{i:04d}",
            pattern_name="double_bottom",
            direction="bullish",
            detect_time=pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i),
            confirm_time=pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i) + pd.Timedelta(minutes=15),
            entry_time=pd.Timestamp("2023-01-01") + pd.Timedelta(hours=i) + pd.Timedelta(minutes=30),
            entry_bar=10 + 7 * i,
            entry_price=100.0,
            stop_price=99.0,
            risk=1.0,
            rule_score=0.5,
            config_hash="0123456789ab",
            horizon_bars=72,
            mfe_ratio=0.0,
            mae_ratio=0.0,
            close_ratio=0.0,
            label=0,
        )
        for i in range(80)
    ]
    ess = effective_sample_size(labeled, window_bars=20)
    assert 0 < ess <= len(labeled)
    ordered = sorted(labeled, key=lambda le: le.entry_bar)
    bars = [le.entry_bar for le in ordered]
    assert all(bars[i + 1] - bars[i] < 20 for i in range(len(bars) - 1))  # dense input
    # greedy kept events are >= 20 bars apart
    kept: list[int] = []
    for le in ordered:
        if not kept or le.entry_bar - kept[-1] >= 20:
            kept.append(le.entry_bar)
    assert len(kept) == ess
    assert all(kept[i + 1] - kept[i] >= 20 for i in range(len(kept) - 1))


# ---------------------------------------------------------------------------
# 5. Backtest audit — costs complete, path backtest ≡ live (§13 A7 item 4)
# ---------------------------------------------------------------------------

def _import_src_labeling(module: str) -> Any:
    """Import a module from the shared sweep research package (same sys.path
    injection the golden test uses)."""
    sweep_pkg = _TV3 / "research" / "patterns" / "liquidity_sweep"
    if str(sweep_pkg) not in sys.path:
        sys.path.insert(0, str(sweep_pkg))
    import importlib

    return importlib.import_module(f"src.labeling.{module}")


def test_backtest_cost_model_exact() -> None:
    """§6.5 cost model: cost_r = 2*(half_spread+slippage)/risk + commission_r,
    net = gross - cost; legacy price-unit keys are rejected loudly so costs
    can never be silently understated."""
    ob = _import_src_labeling("outcome_builder")

    cost = ob.cost_in_r(risk=10.0, half_spread_price=1.0, slippage_price=0.5, commission_r=0.2)
    assert cost == pytest.approx(2.0 * (1.0 + 0.5) / 10.0 + 0.2)

    gross_long = ob.gross_result_in_r("long", entry=100.0, exit_price=102.0, risk=10.0)
    assert gross_long == pytest.approx(0.2)
    gross_short = ob.gross_result_in_r("short", entry=100.0, exit_price=98.0, risk=10.0)
    assert gross_short == pytest.approx(0.2)

    # net = gross - cost (contract; also asserted structurally below)
    assert ob._PRIMARY_COLUMNS[:3] == ["exit_time", "exit_price", "exit_reason"]
    primary = set(ob._PRIMARY_COLUMNS)
    assert {"gross_result_r", "cost_r", "net_result_r"} <= primary

    # zero-cost default is explicit (documented placeholder), not silent
    defaults = ob.labeling_defaults()
    assert defaults["half_spread_price"] == 0.0
    assert defaults["slippage_price"] == 0.0
    assert defaults["commission_r"] == 0.0

    # guard: nonzero legacy entry.* price keys (points vs price confusion) fail
    with pytest.raises(ValueError):
        ob.resolve_labeling_config({"entry": {"spread_price": 0.5}})
    with pytest.raises(ValueError):
        ob.resolve_labeling_config({"entry": {"slippage_price": 0.3}})


def test_backtest_and_live_use_the_same_detector_classes() -> None:
    """The benchmark harness wraps the exact classes exported for the live
    registry (DETECTOR_CLASS / PATTERN_ENTRY) — backtest code path ≡ live."""
    from research.patterns.double_bottom import benchmark as db_bench
    from research.patterns.double_bottom.detector import DETECTOR_CLASS as DB_DC
    from research.patterns.double_top.detector import DETECTOR_CLASS as DT_DC
    from research.patterns.falling_wedge.detector import DETECTOR_CLASS as FW_DC
    from research.patterns.head_shoulders.detector import DETECTOR_CLASS as HS_DC
    from research.patterns.inverse_head_shoulders.detector import (
        DETECTOR_CLASS as IHS_DC,
    )
    from research.patterns.liquidity_sweep.detector import (
        DETECTOR_CLASS as LSW_DC,
    )
    from research.patterns.liquidity_sweep.detector import (
        PATTERN_ENTRY,
    )
    from research.patterns.rising_wedge.detector import DETECTOR_CLASS as RW_DC

    assert DB_DC is DoubleBottomDetector
    assert DT_DC is DoubleTopDetector
    assert RW_DC is RisingWedgeDetector
    assert FW_DC is FallingWedgeDetector
    assert HS_DC is HeadShouldersDetector
    assert IHS_DC is InverseHeadShouldersDetector
    assert LSW_DC is LiquiditySweepDetector
    assert PATTERN_ENTRY["liquidity_sweep"] is LiquiditySweepDetector

    # benchmarks wrap the registry classes, not copies
    assert db_bench.DoubleBottomDetector is DoubleBottomDetector
    assert db_bench.DoubleTopDetector is DoubleTopDetector

    # sweep default config feeds the causal labeling path (group_rule "first"
    # = first bar of the sweep run — required by the labeling module's F1
    # causality contract, outcome_builder docstring)
    assert LiquiditySweepDetector().get_default_config()["group_rule"] == "first"


def test_golden_regression_wired_into_no_lookahead_ci() -> None:
    """The sweep golden test (plugin bit-identical to the legacy live chain)
    exists, is collected, and inherits the no-lookahead template — proving
    backtest (plugin) ≡ live (legacy _check_new_bar) is CI-pinned."""
    import tests.test_liquidity_sweep_golden as golden

    assert hasattr(golden, "test_plugin_events_identical_to_legacy")
    assert hasattr(golden, "test_plugin_causality_clean")
    assert issubclass(golden.TestLiquiditySweepNoLookahead, NoLookaheadTestBase)
    assert golden._LEGACY_CFG["group_rule"] == "first"


# ---------------------------------------------------------------------------
# 6. Suite wiring (acceptance 5: adversarial suite permanently in CI)
# ---------------------------------------------------------------------------

def test_adversarial_suite_wired_into_ci() -> None:
    """The suite lives under the configured testpaths and carries the
    registered no_lookahead marker — every `pytest tests/` run executes it."""
    import tests.test_adversarial_lookahead as this_module

    assert Path(this_module.__file__).resolve().parent == (_TV3 / "tests")
    assert this_module.pytestmark.mark.name == "no_lookahead"