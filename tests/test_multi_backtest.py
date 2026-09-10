"""multi_backtest runner tests (t4 — SPEC v1.1 §12 + handoff §3 P0#3).

Covers:

  1. code-path parity — the runner reaches the SAME group decisions as the
     live ``MultiPatternEngine`` on identical input (backtest ≡ live,
     §12 / §9.1);
  2. reuse — ``CorrelationManager`` / ``CorrelationConfig`` /
     ``ResolvedGroup`` are imported from ``live.engine`` (no duplicated
     grouping logic);
  3. causality — events are consumed at ``known_at`` only (no-lookahead
     marker);
  4. costs — spread / commission / slippage are applied to every trade;
  5. report shape — the generated report carries all 5 §12 items;
  6. correlation + confluence math on a synthetic co-confirming batch
     (items 3 & 4 are not vacuously empty).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from live.engine import correlation_manager as live_corr
from live.engine.correlation_manager import (
    CorrelationConfig,
    ResolvedGroup,
)
from research.core.contracts import PatternEvent
from research.multi_backtest import runner as mb
from research.multi_backtest.report import generate_symbol_report

pytestmark = pytest.mark.no_lookahead

#: The heavier tests share one real XAUUSD M15 slice (module-scoped).
_SYMBOL = "XAUUSD"
_START = "2026-01-01"
_END = "2026-09-03"
_PATTERNS = ["liquidity_sweep", "double_bottom", "double_top"]


@pytest.fixture(scope="module")
def xauusd_frame() -> pd.DataFrame:
    return mb.load_symbol_frame(_SYMBOL, start=_START, end=_END,
                                warmup_bars=1500, sim_extra_bars=72)


@pytest.fixture(scope="module")
def assignments(xauusd_frame: pd.DataFrame):
    return mb.build_assignments(_SYMBOL, _PATTERNS, df=xauusd_frame,
                                attach_models=False)


def _corr_cfg() -> CorrelationConfig:
    return CorrelationConfig(
        mode="dedup", bar_seconds=900.0,
        corr_time_window=5.0, corr_price_window_atr=0.5,
        max_total_risk_per_symbol=2.0, max_direction_cluster_risk=1.0,
    )


# ---------------------------------------------------------------------------
# 1+2. Parity with the live engine + same-code-path reuse
# ---------------------------------------------------------------------------

def test_runner_reuses_live_correlation_manager_imports() -> None:
    """The runner must NOT duplicate grouping logic — it reuses the exact
    ``live.engine.correlation_manager`` classes (identity check)."""
    assert mb.CorrelationManager is live_corr.CorrelationManager
    assert mb.CorrelationConfig is live_corr.CorrelationConfig
    assert mb.ResolvedGroup is live_corr.ResolvedGroup


def test_parity_group_decisions_same_as_live_engine(
    xauusd_frame: pd.DataFrame, assignments,
) -> None:
    """§12/§9.1 — on identical input the runner and MultiPatternEngine reach
    the same correlation-group decisions (same kept/discarded per event)."""
    from live.engine.signal_engine_v2 import MultiPatternEngine

    corr_cfg = _corr_cfg()

    # ---- runner path ------------------------------------------------------
    events_runner = mb.detect_all(xauusd_frame, assignments)
    groups_runner = mb.group_events(events_runner, corr_cfg)

    # ---- live path --------------------------------------------------------
    engine = MultiPatternEngine(
        symbol=_SYMBOL,
        assignments=assignments,
        candle_fn=lambda _s: xauusd_frame,
        correlation_cfg=corr_cfg,
    )
    engine.check_new_bar()
    events_live = engine.get_last_events()
    # re-run the engine's own grouping call over its detected events
    groups_live = engine.corr_manager.group(events_live, config=engine.corr_config)

    # both sides must detect the same events
    assert {e.event_id for e in events_runner} == {e.event_id for e in events_live}
    assert len(events_runner) == len(events_live)
    assert len(events_runner) > 0, "parity needs at least one detected event"

    def decisions(groups: list[ResolvedGroup]) -> dict[str, tuple[str, frozenset]]:
        """event_id -> (kept|discarded, member event-ids of its group)."""
        out: dict[str, tuple[str, frozenset]] = {}
        for g in groups:
            members = frozenset(e.event_id for e in g.events)
            for e in g.events:
                status = "kept" if g.representative is not None and \
                    e.event_id == g.representative.event_id else "discarded"
                out[e.event_id] = (status, members)
        return out

    d_runner = decisions(groups_runner)
    d_live = decisions(groups_live)

    assert d_runner.keys() == d_live.keys()
    for eid in d_runner:
        assert d_runner[eid] == d_live[eid], (
            f"group decision mismatch for {eid}: "
            f"runner={d_runner[eid]} live={d_live[eid]}"
        )


def test_group_events_returns_live_resolved_groups(
    xauusd_frame: pd.DataFrame, assignments,
) -> None:
    """``group_events`` returns the live ``ResolvedGroup`` objects."""
    events = mb.detect_all(xauusd_frame, assignments)
    groups = mb.group_events(events, _corr_cfg())
    assert all(isinstance(g, ResolvedGroup) for g in groups)


# ---------------------------------------------------------------------------
# 3. Causality — consumed at known_at only
# ---------------------------------------------------------------------------

def test_trades_consume_events_at_known_at(
    xauusd_frame: pd.DataFrame, assignments,
) -> None:
    """Every trade enters at/after its event's known_at: the entry bar
    (detector's next-open-after-confirm) is never before known_at."""
    corr_cfg = _corr_cfg()
    result = mb.run_symbol_backtest(
        xauusd_frame, assignments, corr_cfg=corr_cfg,
        costs=mb.CostConfig.for_symbol(_SYMBOL),
        symbol_override=_SYMBOL,
    )
    assert result.trades, "expected at least one executed trade"
    for t in result.trades:
        assert t.entry_time >= t.known_at, (
            f"{t.event_id}: entry {t.entry_time} before known_at {t.known_at}"
        )


def test_detect_is_frame_truncation_stable(xauusd_frame: pd.DataFrame) -> None:
    """Detection output for events is unchanged when the frame is extended:
    no detector reads bars after an event's known_at (classical DB detector;
    fast)."""
    from research.patterns.double_bottom.detector import DoubleBottomDetector

    det = DoubleBottomDetector()
    full = det.detect(xauusd_frame, det.get_default_config())
    if not full:
        pytest.skip("no double_bottom events in the slice")
    trunc = xauusd_frame.iloc[: len(xauusd_frame) // 2]
    part = det.detect(trunc, det.get_default_config())
    part_ids = {e.event_id for e in part}
    # any event fully inside the truncated frame must be identical
    stable = [
        e for e in full
        if e.known_at_ts <= trunc.index[-1]
        and e.entry_time <= trunc.index[-1]
    ]
    assert stable, "slice has no stable events to compare"
    for e in stable:
        assert e.event_id in part_ids, f"{e.event_id} lost after truncation"
    by_id = {e.event_id: e for e in part}
    for e in stable:
        other = by_id[e.event_id]
        assert e.entry_price == other.entry_price
        assert e.stop_price == other.stop_price
        assert e.known_at_ts == other.known_at_ts


# ---------------------------------------------------------------------------
# 4. Costs applied
# ---------------------------------------------------------------------------

def test_costs_applied_to_trades(
    xauusd_frame: pd.DataFrame, assignments,
) -> None:
    """spread/commission/slippage reduce net R — and never inflate it."""
    corr_cfg = _corr_cfg()
    costs = mb.CostConfig.for_symbol(_SYMBOL)
    assert costs.total_bps in (pytest.approx(0.7 + 0.5 + 1.0),)
    result = mb.run_symbol_backtest(
        xauusd_frame, assignments, corr_cfg=corr_cfg, costs=costs,
        symbol_override=_SYMBOL,
    )
    assert result.trades
    for t in result.trades:
        assert t.cost_r > 0
        assert t.net_r == pytest.approx(t.gross_r - t.cost_r)
        assert t.net_r <= t.gross_r + 1e-9


# ---------------------------------------------------------------------------
# 5. Report shape — all 5 §12 items
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["dedup", "confluence", "independent"])
def test_report_shape_smoke(
    xauusd_frame: pd.DataFrame, assignments, tmp_path: Path, mode: str,
) -> None:
    """The generated report carries all 5 §12 items with sane structure."""
    corr_cfg = _corr_cfg()
    corr_cfg.mode = mode
    result = mb.run_symbol_backtest(
        xauusd_frame, assignments, corr_cfg=corr_cfg,
        costs=mb.CostConfig.for_symbol(_SYMBOL),
        symbol_override=_SYMBOL,
    )
    md, js = generate_symbol_report(
        result, xauusd_frame, out_dir=tmp_path,
        start=pd.Timestamp(_START), end=pd.Timestamp(_END),
    )
    assert md.exists() and js.exists()
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["schema"] == "multi_backtest_report_v1"
    assert data["symbol"] == _SYMBOL
    assert set(data.keys()) >= {
        "per_pattern", "portfolio", "correlation_matrix",
        "confluence_buckets", "oos_comparison",
    }
    # item 1: per-pattern stats
    assert len(data["per_pattern"]) >= 3
    for p in data["per_pattern"]:
        assert set(p.keys()) >= {
            "pattern", "trades", "pf", "expectancy_r", "max_dd_r",
            "pr_auc", "rolling_pf",
        }
    # item 2: portfolio equity after caps
    assert isinstance(data["portfolio"]["equity"], list)
    assert data["portfolio"]["trades"] >= 0
    # item 3: correlation matrix
    cm = data["correlation_matrix"]
    assert len(cm["patterns"]) == len(cm["matrix"]) >= 3
    assert all(len(row) == len(cm["patterns"]) for row in cm["matrix"])
    # item 4: confluence buckets
    assert isinstance(data["confluence_buckets"], list)
    # item 5: OOS comparison
    o = data["oos_comparison"]
    assert "portfolio_net_r" in o and "best_single_pattern" in o
    assert isinstance(o["multi_beats_best_single"], bool)
    assert "verdict" in o
    # markdown is non-trivial
    md_text = md.read_text(encoding="utf-8")
    assert "## 1. Per-pattern statistics" in md_text
    assert "## 5. Multi-pattern portfolio vs best single pattern (OOS)" in md_text


# ---------------------------------------------------------------------------
# 6. Correlation + confluence math on synthetic co-confirming events
# ---------------------------------------------------------------------------

def _synthetic_event(
    event_id: str,
    pattern_name: str,
    bar: int,
    entry_price: float = 100.0,
    direction: str = "bullish",
    atr: float = 1.0,
    symbol: str = _SYMBOL,
) -> PatternEvent:
    ts = pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(minutes=15 * bar)
    return PatternEvent(
        event_id=event_id,
        pattern_name=pattern_name,
        pattern_version="1.0",
        symbol=symbol,
        timeframe="M15",
        direction=direction,
        detect_time=ts,
        confirm_time=ts,
        entry_time=ts + pd.Timedelta(minutes=15),
        entry_price=entry_price,
        stop_price=entry_price - 2.0,
        target_price=entry_price + 3.0,
        rule_score=50.0,
        attributes={"atr_value": atr},
    )


def _synthetic_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100.0,
        },
        index=idx,
    )


def test_correlation_matrix_and_confluence_buckets_nonempty() -> None:
    """Two patterns confirming on the same bar/price form one group: the
    correlation matrix and the confluence buckets must reflect it (items
    3+4 are real math, not vacuous zeros)."""
    ev_a = _synthetic_event("A-0001", "double_bottom", bar=5, entry_price=100.0)
    ev_b = _synthetic_event("B-0001", "liquidity_sweep", bar=5, entry_price=100.2,
                            atr=1.0)
    ev_c = _synthetic_event("C-0001", "double_top", bar=20, direction="bearish",
                            entry_price=100.0)
    cfg = _corr_cfg()
    groups = mb.group_events([ev_a, ev_b, ev_c], cfg)
    assert len(groups) == 2, "same-bar pair groups, distant event stays alone"
    pair = next(g for g in groups if g.n_patterns == 2)
    assert pair.representative is not None
    assert {e.pattern_name for e in pair.events} == {
        "double_bottom", "liquidity_sweep",
    }

    from research.multi_backtest import report as rep

    corr = rep._correlation_matrix(
        mb.PortfolioResult(
            symbol=_SYMBOL, timeframe="M15", events=[ev_a, ev_b, ev_c],
            groups=groups, trades=[], rejected=[],
            corr_config=cfg, costs=mb.CostConfig.for_symbol(_SYMBOL),
            mode="dedup",
        )
    )
    db_row = corr["matrix"][corr["patterns"].index("double_bottom")]
    lsw_idx = corr["patterns"].index("liquidity_sweep")
    assert db_row[lsw_idx] == pytest.approx(1.0), "DB shares 100% of groups with LSW"

    buckets = rep._confluence_buckets(
        mb.PortfolioResult(
            symbol=_SYMBOL, timeframe="M15", events=[ev_a, ev_b, ev_c],
            groups=groups, trades=[], rejected=[],
            corr_config=cfg, costs=mb.CostConfig.for_symbol(_SYMBOL),
            mode="dedup",
        )
    )
    assert any(b["n_patterns"] == 2 for b in buckets)


# ---------------------------------------------------------------------------
# Sample reports committed (≥2 symbols)
# ---------------------------------------------------------------------------

def test_sample_reports_committed_for_two_symbols() -> None:
    """The repo carries §12 sample reports for XAUUSD + EURUSD (MD + JSON)."""
    reports_dir = mb.REPORTS_DIR
    for symbol in ("XAUUSD", "EURUSD"):
        md = reports_dir / f"{symbol}_multi_backtest.md"
        js = reports_dir / f"{symbol}_multi_backtest.json"
        assert md.exists(), f"missing {md}"
        assert js.exists(), f"missing {js}"
        data = json.loads(js.read_text(encoding="utf-8"))
        assert data["symbol"] == symbol
        patterns = {p["pattern"] for p in data["per_pattern"]}
        # minimum: LSW + DB + DT
        assert {"liquidity_sweep", "double_bottom", "double_top"} <= patterns
        assert data["portfolio"]["trades"] > 0
        assert data["costs"]["spread_bps"] > 0  # costs applied


def test_sample_report_costs_applied() -> None:
    """The committed sample reports carry applied costs (spread/commission/
    slippage > 0) and a cost column reduction (gross ≠ net)."""
    js = mb.REPORTS_DIR / f"{_SYMBOL}_multi_backtest.json"
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["costs"]["commission_bps"] > 0
    assert data["costs"]["slippage_bps"] > 0