"""CorrelationManager tests — Dedup / Correlation Contract (§4, Agent 6 t9).

Covers REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §4:

  1. §4.2  grouping is union-find & transitive (e1~e2 and e2~e3 ⇒ all 3 in one
     group even when e1/e3 are not directly correlated);
  2. §4.3  the three per-symbol modes:
       * ``dedup``       — one group keeps the best event (priority =
         model_prob x rule_score, fallback rule_score); the others are
         discarded with ``attributes.discard_reason = "correlated"``;
       * ``confluence``  — one merged signal per group with a §4.5 score;
       * ``independent`` — every event passes through;
  3. §4.4  exposure caps (max_total_risk_per_symbol and
     max_direction_cluster_risk);
  4. §4.5  confluence score weights / monotonicity basics.

Marker: no_lookahead (the manager only reads event timestamps that are already
known — nothing here reads future bars).
"""

from __future__ import annotations

import pandas as pd
import pytest

from live.engine.correlation_manager import (
    MODE_CONFLUENCE,
    MODE_DEDUP,
    MODE_INDEPENDENT,
    CorrelationConfig,
    CorrelationManager,
    confluence_score,
    default_atr_resolver,
)
from research.core.contracts import PatternEvent

pytestmark = pytest.mark.no_lookahead


def _ts(bar: int) -> pd.Timestamp:
    """Bar B at 2026-01-01 00:B:00 (M15 cadence)."""
    return pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(minutes=15 * bar)


def _ev(
    event_id: str,
    pattern_name: str,
    symbol: str = "XAUUSD",
    direction: str = "bullish",
    bar: int = 10,
    entry_price: float = 2650.0,
    rule_score: float = 50.0,
    model_prob: float | None = None,
    atr: float | None = 2.0,
    timeframe: str = "M15",
) -> PatternEvent:
    """Minimal causal PatternEvent for correlation tests.

    Same-bar events use identical *bar* so |known_at - known_at| = 0 and the
    price window is the deciding filter (when ATR present) or the same-bar
    fallback (no ATR).
    """
    return PatternEvent(
        event_id=event_id,
        pattern_name=pattern_name,
        pattern_version="1.0",
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        detect_time=_ts(bar - 2),
        confirm_time=_ts(bar),
        entry_time=_ts(bar + 1),
        entry_price=entry_price,
        stop_price=entry_price - 10.0 if direction == "bullish" else entry_price + 10.0,
        target_price=entry_price + 30.0 if direction == "bullish" else entry_price - 30.0,
        rule_score=rule_score,
        model_prob=model_prob,
        attributes={"atr_value": atr} if atr is not None else {},
    )


# ---------------------------------------------------------------------------
# §4.2 union-find grouping + transitivity
# ---------------------------------------------------------------------------

def test_group_same_candle_two_patterns_dedup_keeps_one() -> None:
    """§4.1/§4.3: two patterns confirming on the same candle (same symbol,
    same direction, same known_at, same price) de-duplicate into ONE group and
    only the best event is kept."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_DEDUP))
    a = _ev("DB-0001", "double_bottom", bar=10, entry_price=2650.0,
            rule_score=60.0, model_prob=0.8)
    b = _ev("LSW-0001", "liquidity_sweep", bar=10, entry_price=2650.0,
            rule_score=90.0, model_prob=0.7)
    groups = mgr.group([a, b])

    assert len(groups) == 1  # one correlation group
    g = groups[0]
    assert g.group_id is not None
    assert g.mode == MODE_DEDUP
    assert sorted(e.event_id for e in g.events) == ["DB-0001", "LSW-0001"]
    assert g.representative is not None
    # priority = model_prob*rule_score: 0.8*60=48 vs 0.7*90=63 → LSW wins
    assert g.representative.event_id == "LSW-0001"

    # discarded event gets the §7.1 discard_reason
    discarded = g.discarded
    assert [d.event_id for d in discarded] == ["DB-0001"]
    assert discarded[0].attributes.get("discard_reason") == "correlated"
    # §2.1 confluence_group_id stamped on all group members
    assert a.confluence_group_id == g.group_id
    assert b.confluence_group_id == g.group_id


def test_group_fallback_priority_uses_rule_score() -> None:
    """§4.3 fallback: without model_prob, the best event is max rule_score."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_DEDUP))
    lo = _ev("X1", "a", bar=10, entry_price=2650.0, rule_score=40.0, model_prob=None)
    hi = _ev("X2", "b", bar=10, entry_price=2650.0, rule_score=85.0, model_prob=None)
    groups = mgr.group([lo, hi])
    assert groups[0].representative is not None
    assert groups[0].representative.event_id == "X2"


def test_transitive_union_find() -> None:
    """§4.2 transitivity: e1~e2 (time+price close), e2~e3 (time+price close)
    but e1 far from e3 in price slightly over window → still one group."""
    cfg = CorrelationConfig(mode=MODE_DEDUP, corr_price_window_atr=0.5,
                            bar_seconds=900.0)
    # ATR = 2.0 → price window 1.0; time window = 5 bars * 900s = 4500s
    e1 = _ev("E1", "a", bar=10, entry_price=2650.0, atr=2.0)
    e2 = _ev("E2", "b", bar=11, entry_price=2650.8, atr=2.0)
    e3 = _ev("E3", "c", bar=12, entry_price=2651.6, atr=2.0)
    # direct: |e1-e3| = 1.6 > 1.0 window → not directly correlated,
    # but e1~e2 (0.8) and e2~e3 (0.8) → transitive group of 3.
    mgr = CorrelationManager(cfg)
    groups = mgr.group([e1, e2, e3])
    assert len(groups) == 1
    assert len(groups[0].events) == 3


def test_different_direction_not_grouped() -> None:
    """§4.2 (1): different directions never group even on the same candle."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_DEDUP))
    long = _ev("L1", "a", direction="bullish", bar=10, entry_price=2650.0)
    short = _ev("S1", "b", direction="bearish", bar=10, entry_price=2650.0)
    groups = mgr.group([long, short])
    assert len(groups) == 2  # two separate groups


def test_price_window_atr_boundary() -> None:
    """§4.2 (3): price gap within corr_price_window_atr*ATR groups; beyond it
    does not (even same bar)."""
    cfg = CorrelationConfig(mode=MODE_DEDUP, corr_price_window_atr=0.5, bar_seconds=900.0)
    near = _ev("N1", "a", bar=10, entry_price=2650.0, atr=2.0)
    far = _ev("F1", "b", bar=10, entry_price=2652.0, atr=2.0)  # 2.0 > 1.0 window
    mgr = CorrelationManager(cfg)
    groups = mgr.group([near, far])
    assert len(groups) == 2


def test_price_window_same_bar_fallback_without_atr() -> None:
    """Without ATR data the manager still de-duplicates same-candle events
    (the §4.1 risk the contract exists to prevent)."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_DEDUP, bar_seconds=900.0))
    a = _ev("A1", "a", bar=10, entry_price=2650.0, atr=None)
    b = _ev("B1", "b", bar=10, entry_price=2650.0, atr=None)
    # different timeframes, same minute → same-bar match
    a.timeframe = "M15"
    b.timeframe = "H1"
    groups = mgr.group([a, b])
    assert len(groups) == 1


# ---------------------------------------------------------------------------
# §4.3 modes
# ---------------------------------------------------------------------------

def test_mode_independent_passes_every_event() -> None:
    """§4.3 independent: correlation is computed but every event is kept."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_INDEPENDENT))
    a = _ev("A1", "a", bar=10, entry_price=2650.0)
    b = _ev("B1", "b", bar=10, entry_price=2650.0)
    groups = mgr.group([a, b])
    assert len(groups) == 2
    for g in groups:
        assert len(g.events) == 1
        assert g.mode == MODE_INDEPENDENT
        assert g.kept_event is not None


def test_mode_confluence_one_signal_with_score() -> None:
    """§4.3 confluence: one merged signal per group carrying the §4.5 score,
    the representative event, and n_patterns."""
    mgr = CorrelationManager(CorrelationConfig(mode=MODE_CONFLUENCE))
    a = _ev("A1", "a", bar=10, entry_price=2650.0, rule_score=60.0, model_prob=0.8)
    b = _ev("B1", "b", bar=10, entry_price=2650.2, rule_score=90.0, model_prob=0.7)
    groups = mgr.group([a, b])

    assert len(groups) == 1
    g = groups[0]
    assert g.mode == MODE_CONFLUENCE
    assert g.representative is not None
    assert g.n_patterns == 2
    # n-patterns term alone gives > 0.5 (w1=0.5 * (2-1)/(2-1) = 0.5) plus
    # proximity terms → score strictly positive and <= 1
    assert 0.5 < g.confluence_score <= 1.0
    # no discard in confluence mode
    assert g.discarded == []


def test_confluence_score_monotonic_in_n() -> None:
    """§4.5 — with a fixed n_patterns_max denominator, the n-patterns term
    grows with the number of agreeing patterns."""
    cfg = CorrelationConfig(mode=MODE_CONFLUENCE, n_patterns_max=3)
    base = _ev("B1", "a", bar=10, entry_price=2650.0, rule_score=50.0, model_prob=0.5)
    other = _ev("B2", "b", bar=10, entry_price=2650.0, rule_score=50.0, model_prob=0.5)
    third = _ev("B3", "c", bar=10, entry_price=2650.0, rule_score=50.0, model_prob=0.5)
    s2 = confluence_score([base, other], cfg)      # term_n = (2-1)/(3-1) = 0.5
    s3 = confluence_score([base, other, third], cfg)  # term_n = (3-1)/(3-1) = 1.0
    assert s3 > s2


def test_atr_resolver_prefers_atr_value() -> None:
    events = [
        _ev("A", "a", atr=2.5),
        _ev("B", "b", atr=None),
    ]
    resolved = default_atr_resolver(events)
    assert resolved["A"] == 2.5
    assert resolved["B"] is None


def test_invalid_mode_raises() -> None:
    from live.engine.correlation_manager import CorrelationError

    with pytest.raises(CorrelationError):
        CorrelationConfig(mode="bogus")
    with pytest.raises(CorrelationError):
        CorrelationConfig(w1=0.0, w2=0.0, w3=0.0)
    with pytest.raises(CorrelationError):
        CorrelationConfig(confluence_size_boost=0.5)


# ---------------------------------------------------------------------------
# §4.4 exposure caps
# ---------------------------------------------------------------------------

def test_exposure_cap_total_risk() -> None:
    cfg = CorrelationConfig(mode=MODE_DEDUP, max_total_risk_per_symbol=1.0)
    mgr = CorrelationManager(cfg)

    # simple dict positions (adapter handles dicts too)
    positions = [{"direction": "long", "entry_price": 2640.0, "position_size": 0.4}]
    ok, reason = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.5, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=positions, config=cfg,
    )
    assert ok and reason == ""
    # 0.4 + 0.7 = 1.1 > 1.0 → blocked
    ok2, reason2 = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.7, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=positions, config=cfg,
    )
    assert not ok2
    assert "max_total_risk_per_symbol" in reason2


def test_exposure_cap_direction_cluster() -> None:
    cfg = CorrelationConfig(mode=MODE_DEDUP, max_direction_cluster_risk=0.75)
    mgr = CorrelationManager(cfg)
    # same-direction position within 1 ATR (2.0) of proposed entry
    positions = [{"direction": "long", "entry_price": 2651.0, "position_size": 0.5}]
    ok, _ = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.2, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=positions, config=cfg,
    )
    assert ok  # 0.5+0.2=0.7 <= 0.75
    ok2, reason2 = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.3, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=positions, config=cfg,
    )
    assert not ok2
    assert "max_direction_cluster_risk" in reason2


def test_exposure_cap_ignores_opposite_direction() -> None:
    cfg = CorrelationConfig(mode=MODE_CONFLUENCE, max_direction_cluster_risk=0.75,
                            max_total_risk_per_symbol=5.0)
    mgr = CorrelationManager(cfg)
    # opposite-direction positions never count toward the cluster cap
    positions = [{"direction": "short", "entry_price": 2649.0, "position_size": 0.9}]
    ok, reason = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.5, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=positions, config=cfg,
    )
    assert ok, reason


def test_position_field_aliases() -> None:
    """Adapter reads both OpenPosition-like objects and dicts."""
    class _Pos:
        direction = "long"
        entry_price = 2645.0
        position_size = 0.3

    cfg = CorrelationConfig(mode=MODE_DEDUP, max_total_risk_per_symbol=1.0)
    mgr = CorrelationManager(cfg)
    ok, _ = mgr.check_exposure_caps(
        "XAUUSD", "long", proposed_risk=0.5, proposed_entry=2650.0,
        proposed_atr=2.0, open_positions=[_Pos()], config=cfg,
    )
    assert ok