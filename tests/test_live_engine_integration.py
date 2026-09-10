"""MultiPatternEngine integration tests — §9.1 live flow (Agent 6 t9 DoD).

End-to-end on a stubbed candle source + detector plugins, exercising the
exact §9.1 flow the MultiPatternEngine orchestrates:

  1. DoD #1 — **two patterns confirming on the same candle produce exactly ONE
     order** (dedup mode keeps the best event; §4.3/§9.1).
  2. DoD #2 — **a shadow assignment never creates an order** (§5.2): its
     events join the correlation grouping but produce no SignalCandidate.
  3. Bias-auditor §3.3 — **entry-drift guard**: when the live market price has
     moved beyond ``max_entry_drift_atr`` ATR from the event's entry_price,
     the event is dropped as stale (no order, no chasing price).
  4. §9.2 — the candidate carries the standardised order comment.
  5. confluence mode — a 2-pattern group produces one boosted signal.

Marker: no_lookahead (stub detectors only emit already-known events; the
engine's causality + drift handling never reads future bars).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from live.engine.correlation_manager import (
    MODE_CONFLUENCE,
    MODE_DEDUP,
    CorrelationConfig,
)
from live.engine.signal_engine_v2 import (
    MultiPatternEngine,
    PatternAssignment,
)
from research.core.contracts import (
    LIFECYCLE_LIVE,
    LIFECYCLE_SHADOW,
    BasePatternDetector,
    PatternEvent,
)

pytestmark = pytest.mark.no_lookahead


# ---------------------------------------------------------------------------
# Stub detector plugin — returns pre-built PatternEvents (no market logic)
# ---------------------------------------------------------------------------

class StubDetector(BasePatternDetector):
    """Fixed-output detector used by the integration tests."""

    name = "stub"
    version = "1.0"
    short_name = "STB"

    def __init__(self, events: list[PatternEvent], name: str | None = None) -> None:
        self._events = events
        if name is not None:
            self.__class__.name = name  # per-instance pattern name override
        super().__init__()

    def get_default_config(self) -> dict[str, Any]:
        return {"version": self.version}

    def detect(self, df: pd.DataFrame, config: dict[str, Any]) -> list[PatternEvent]:
        return [e for e in self._events if e is not None]

    def detect_frame(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
        return pd.DataFrame()


class _NamedDetector(StubDetector):
    """Detector class carrying its pattern identity at class level."""

    def __init__(self, events: list[PatternEvent]) -> None:
        StubDetector.__init__(self, events)


def _make_detector_class(name: str, short: str, version: str = "1.0") -> type[_NamedDetector]:
    return type(
        f"Detector_{name}",
        (_NamedDetector,),
        {"name": name, "version": version, "short_name": short},
    )


def _ts(bar: int) -> pd.Timestamp:
    return pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(minutes=15 * bar)


def _ev(
    event_id: str,
    pattern_name: str,
    entry_price: float = 2650.0,
    rule_score: float = 50.0,
    model_prob: float | None = 0.6,
    atr: float = 2.0,
    bar: int = 10,
    direction: str = "bullish",
) -> PatternEvent:
    return PatternEvent(
        event_id=event_id,
        pattern_name=pattern_name,
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction=direction,
        detect_time=_ts(bar - 2),
        confirm_time=_ts(bar),
        entry_time=_ts(bar + 1),
        known_at=_ts(bar),
        entry_price=entry_price,
        stop_price=entry_price - 10.0 if direction == "bullish" else entry_price + 10.0,
        target_price=entry_price + 30.0 if direction == "bullish" else entry_price - 30.0,
        rule_score=rule_score,
        model_prob=model_prob,
        attributes={"atr_value": atr},
        structure_levels={"level_price": entry_price},
    )


def _candles(close: float = 2650.0, n: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="15min")
    o = [close] * n
    return pd.DataFrame(
        {
            "open": o,
            "high": [close + 0.5] * n,
            "low": [close - 0.5] * n,
            "close": o,
            "volume": [0] * n,
        },
        index=idx,
    )


class _CandleSource:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def get_chart_history(self, symbol: str) -> pd.DataFrame:
        return self.df


def _engine(
    events: list[PatternEvent],
    states: dict[str, str] | None = None,
    cfg: CorrelationConfig | None = None,
    detector_classes: dict[str, type] | None = None,
    max_drift: float = 1.5,
    candle_close: float = 2650.0,
) -> MultiPatternEngine:
    """Build a MultiPatternEngine over per-event detector plugins.

    Every event gets its own assignment (state live unless *states* overrides
    by pattern_name), so the correlation grouping sees all events at once.
    """
    states = states or {}
    detector_classes = detector_classes or {}
    by_name: dict[str, list[PatternEvent]] = {}
    for e in events:
        by_name.setdefault(e.pattern_name, []).append(e)

    assignments: list[PatternAssignment] = []
    for i, (name, evs) in enumerate(sorted(by_name.items())):
        cls = detector_classes.get(
            name,
            _make_detector_class(name, ("STB" + str(i))[:3].upper()),
        )
        det = cls(evs)
        state = states.get(name, LIFECYCLE_LIVE)
        assignments.append(
            PatternAssignment(
                assignment_id=f"assign-{name}",
                pattern_name=name,
                timeframe="M15",
                state=state,
                detector=det,
                max_entry_drift_atr=max_drift,
            )
        )
    return MultiPatternEngine(
        symbol="XAUUSD",
        assignments=assignments,
        candle_fn=_CandleSource(_candles(close=candle_close)).get_chart_history,
        correlation_cfg=cfg,
    )


# ---------------------------------------------------------------------------
# DoD #1 — dedup: 2 patterns on the same candle → exactly 1 order
# ---------------------------------------------------------------------------

def test_two_live_patterns_same_candle_produce_one_order() -> None:
    """§4.1/§4.3 DoD: two live patterns confirming on the SAME candle must
    yield exactly ONE SignalCandidate (the best event)."""
    db = _ev("DB-0001", "double_bottom", entry_price=2650.0,
             rule_score=60.0, model_prob=0.8)
    lsw = _ev("LSW-0001", "liquidity_sweep", entry_price=2650.0,
              rule_score=90.0, model_prob=0.7)
    engine = _engine([db, lsw], cfg=CorrelationConfig(mode=MODE_DEDUP))

    candidates = engine.check_new_bar()

    assert len(candidates) == 1  # dedup → 1 order
    c = candidates[0]
    # LSW wins priority (0.7*90=63 > 0.8*60=48)
    assert c.pattern_name == "liquidity_sweep"
    assert c.event_id == "LSW-0001"
    # §9.2 standard order comment
    assert c.order_comment == "LSW-v1-0001"
    # the discarded DB event is flagged for Event Lake
    db_ev = next(e for e in engine.get_last_events() if e.event_id == "DB-0001")
    assert db_ev.attributes.get("discard_reason") == "correlated"


def test_two_independent_events_still_one_order_when_not_correlated() -> None:
    """Different directions/symbols are not correlated → each produces an order."""
    long = _ev("L1", "a_pat", entry_price=2650.0, direction="bullish")
    short = _ev("S1", "b_pat", entry_price=2650.0, direction="bearish")
    engine = _engine([long, short], cfg=CorrelationConfig(mode=MODE_DEDUP))
    candidates = engine.check_new_bar()
    assert len(candidates) == 2  # separate groups → 2 orders


# ---------------------------------------------------------------------------
# DoD #2 — shadow never creates an order
# ---------------------------------------------------------------------------

def test_shadow_assignment_creates_no_order() -> None:
    """§5.2 DoD: a shadow assignment's event joins the group but yields NO
    SignalCandidate — even when the shadow event is the best one."""
    live_db = _ev("DB-0001", "double_bottom", entry_price=2650.0,
                  rule_score=60.0, model_prob=0.8)
    shadow_lsw = _ev("LSW-0001", "liquidity_sweep", entry_price=2650.0,
                     rule_score=90.0, model_prob=0.9)  # best priority
    engine = _engine(
        [live_db, shadow_lsw],
        states={"liquidity_sweep": LIFECYCLE_SHADOW, "double_bottom": LIFECYCLE_LIVE},
        cfg=CorrelationConfig(mode=MODE_DEDUP),
    )
    candidates = engine.check_new_bar()
    # shadow is the best event → the whole group must NOT emit an order
    assert candidates == []
    # but the shadow event was still recorded for Event Lake
    shadow_ev = next(e for e in engine.get_last_events() if e.event_id == "LSW-0001")
    assert shadow_ev.attributes.get("shadow_mode") is True


def test_pure_shadow_symbol_never_emits() -> None:
    """A symbol whose only assignment is shadow produces zero orders."""
    ev = _ev("X1", "wedge", entry_price=2650.0)
    engine = _engine([ev], states={"wedge": LIFECYCLE_SHADOW})
    assert engine.check_new_bar() == []


# ---------------------------------------------------------------------------
# Bias-auditor §3.3 — entry-drift guard
# ---------------------------------------------------------------------------

def test_entry_drift_beyond_threshold_drops_event() -> None:
    """§3.3: market price moved > max_entry_drift_atr*ATR from entry → the
    event is dropped as stale (no signal chasing price).  max=1.5, ATR=2.0
    → max gap 3.0; live close 2654.0 is 4.0 away → dropped."""
    ev = _ev("E1", "double_bottom", entry_price=2650.0, atr=2.0)
    engine = _engine([ev], max_drift=1.5, candle_close=2654.0)

    candidates = engine.check_new_bar()
    assert candidates == []
    stale = next(e for e in engine.get_last_events() if e.event_id == "E1")
    assert stale.attributes.get("discard_reason") == "stale"


def test_entry_drift_within_threshold_keeps_signal() -> None:
    """§3.3: price drift within max_entry_drift_atr*ATR → signal still fires."""
    ev = _ev("E1", "double_bottom", entry_price=2650.0, atr=2.0)
    engine = _engine([ev], max_drift=1.5, candle_close=2651.0)  # within 3.0

    candidates = engine.check_new_bar()
    assert len(candidates) == 1
    assert candidates[0].event_id == "E1"


def test_entry_drift_guard_disabled_when_zero() -> None:
    """§3.3: a max_entry_drift_atr <= 0 disables the guard (legacy path)."""
    ev = _ev("E1", "double_bottom", entry_price=2650.0, atr=2.0)
    engine = _engine([ev], max_drift=0.0, candle_close=2700.0)
    assert len(engine.check_new_bar()) == 1


# ---------------------------------------------------------------------------
# §4.3 confluence mode — one boosted signal per group
# ---------------------------------------------------------------------------

def test_confluence_two_patterns_one_signal_with_boost() -> None:
    """§4.3 confluence: a 2-pattern group becomes ONE signal carrying the
    §4.5 confluence score (engine sizes up via risk_fraction x boost)."""
    db = _ev("DB-0001", "double_bottom", entry_price=2650.0, rule_score=60.0,
             model_prob=0.8)
    lsw = _ev("LSW-0001", "liquidity_sweep", entry_price=2650.0, rule_score=90.0,
              model_prob=0.7)
    engine = _engine([db, lsw], cfg=CorrelationConfig(mode=MODE_CONFLUENCE))

    candidates = engine.check_new_bar()
    assert len(candidates) == 1
    c = candidates[0]
    assert c.confluence_score > 0.5
    assert c.confluence_group_id is not None
    assert c.metadata.get("n_patterns") == 2


# ---------------------------------------------------------------------------
# §9.2 order comment
# ---------------------------------------------------------------------------

def test_order_comment_schema() -> None:
    """§9.2: comment matches ``{pattern_short}-v{major}-{event_id_short}``."""
    ev = _ev("XAUUSD-DB-000456", "double_bottom", entry_price=2650.0)
    engine = _engine([ev])
    candidates = engine.check_new_bar()
    assert len(candidates) == 1
    assert candidates[0].order_comment == "DB-v1-0456"


def test_multi_pattern_engine_keeps_legacy_symbol_engine_importable() -> None:
    """The old single-pattern interface must still exist (t9 contract:
    'giữ interface cũ ra ngoài')."""
    import live.engine.signal_engine_v2 as m
    assert hasattr(m, "create_symbol_engine")
    assert hasattr(m, "_check_new_bar")