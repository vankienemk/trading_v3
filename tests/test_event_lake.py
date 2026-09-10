"""Tests for Agent 5 — Event Lake & pattern feature store (§7).

DoD (spec §13 Agent 5):
  * append-only writer on the §7.2 layout
    ``events/{pattern_name}/{symbol}/{yyyy-mm}.parquet``;
  * round-trip write -> read identical (frame AND reconstructed dataclass);
  * outcome join 100% by event_id — every persisted event has exactly one
    outcome row after the tracker job;
  * PSI drift job runs on simulated data and flags retrain/degrade per §5.3;
  * ``rebuild_dataset(pattern, symbol, from, to)`` returns a labeled dataset
    without re-running any detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from research.core.config_hash import canonical_json
from research.core.contracts import (
    LIFECYCLE_LIVE,
    LIFECYCLE_SHADOW,
    PatternEvent,
)
from research.core.event_lake import (
    EXIT_NO_DATA,
    EXIT_STOP_HIT,
    EXIT_TARGET_HIT,
    EXIT_TIMEOUT,
    DriftMonitor,
    EventLake,
    EventLakeError,
    EventOutcome,
    OutcomeTracker,
    events_from_frame,
    events_to_frame,
    population_stability_index,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

P0 = pd.Timestamp("2026-09-07 00:00", tz="UTC")


def ts(hour: int, minute: int = 0) -> pd.Timestamp:
    return P0 + pd.Timedelta(hours=hour, minutes=minute)


def make_event(
    event_id: str,
    *,
    hour: int,
    direction: str = "bullish",
    entry: float = 2650.0,
    stop: float = 2640.0,
    target: float = 2680.0,
    pattern: str = "liquidity_sweep",
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    lifecycle: str = LIFECYCLE_LIVE,
    discard_reason: str | None = None,
    extra_attrs: dict | None = None,
) -> PatternEvent:
    attrs: dict = {}
    if discard_reason is not None:
        attrs["discard_reason"] = discard_reason
    if extra_attrs:
        attrs.update(extra_attrs)
    return PatternEvent(
        event_id=event_id,
        pattern_name=pattern,
        pattern_version="2.0",
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        detect_time=ts(hour),
        confirm_time=ts(hour + 1),
        entry_time=ts(hour + 2),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        structure_levels={"neckline": (entry + stop) / 2},
        rule_score=0.7,
        model_prob=0.6,
        config_hash="deadbeef123",
        feature_schema_version="db_v1.0",
        attributes=attrs,
        lifecycle_state=lifecycle,
    )


def make_candles(
    start_hour: int = 0,
    n_bars: int = 40,
    *,
    phase_high: float = 2685.0,
    phase_low: float = 2590.0,
) -> pd.DataFrame:
    """Deterministic biphasic OHLC: rise to *phase_high*, then fall to
    *phase_low* — enough range to trigger targets/stops on both legs."""
    idx = pd.date_range(ts(start_hour), periods=n_bars, freq="15min", tz="UTC")
    half = n_bars // 2
    closes = np.linspace(2645.0, phase_high, half + 1)
    closes = np.concatenate([closes, np.linspace(phase_high, phase_low, n_bars - half)])[
        1:
    ]
    highs = closes + 3.0
    lows = closes - 3.0
    opens = np.concatenate([[2646.0], closes[:-1]])
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes}, index=idx
    )


@pytest.fixture()
def lake(tmp_path):
    return EventLake(tmp_path / "event_lake")


# ---------------------------------------------------------------------------
# 1. Layout + append-only writer
# ---------------------------------------------------------------------------

def test_append_events_partition_layout(lake) -> None:
    evs = [
        make_event("A-1", hour=1, symbol="XAUUSD"),
        make_event("A-2", hour=2, symbol="XAUUSD"),
        make_event("A-3", hour=1, symbol="EURUSD"),
    ]
    lake.append_events(evs)
    assert lake.events_partition_path(
        "liquidity_sweep", "XAUUSD", "2026-09"
    ).is_file()
    assert lake.events_partition_path(
        "liquidity_sweep", "EURUSD", "2026-09"
    ).is_file()
    # month partition derived from detect_time
    ev_prev = make_event("A-4", hour=1)
    ev_prev.detect_time = ts(1) - pd.Timedelta(days=14)  # previous month
    lake.append_events([ev_prev])
    assert lake.events_partition_path(
        "liquidity_sweep", "XAUUSD", "2026-08"
    ).is_file()


def test_append_only_duplicate_event_raises(lake) -> None:
    ev = make_event("DUP-1", hour=1)
    lake.append_events([ev])
    with pytest.raises(EventLakeError, match="append-only"):
        lake.append_events([ev])
    # unchanged row count after the rejected overwrite
    assert lake.event_ids() == ["DUP-1"]


def test_append_only_two_batches_grow_without_mutation(lake) -> None:
    lake.append_events([make_event("B-1", hour=1)])
    first = lake.read_events()
    lake.append_events([make_event("B-2", hour=2)])
    second = lake.read_events()
    assert len(second) == 2
    pdt.assert_frame_equal(first, second.iloc[:1].reset_index(drop=True))


def test_append_requires_detect_time(lake) -> None:
    ev = make_event("NO-DT", hour=1)
    ev.detect_time = pd.NaT
    with pytest.raises(EventLakeError, match="detect_time"):
        lake.append_events([ev])


def test_rejected_and_shadow_events_persisted(lake) -> None:
    evs = [
        make_event(
            "R-1", hour=1, discard_reason="stale", lifecycle=LIFECYCLE_SHADOW
        ),
        make_event(
            "R-2", hour=2, discard_reason="correlated", lifecycle=LIFECYCLE_LIVE
        ),
        make_event("R-3", hour=3, discard_reason="threshold"),
    ]
    lake.append_events(evs)
    back = {ev.event_id: ev for ev in lake.events()}
    assert len(back) == 3
    for eid, reason in (("R-1", "stale"), ("R-2", "correlated"), ("R-3", "threshold")):
        assert back[eid].attributes["discard_reason"] == reason
        assert back[eid].lifecycle_state in (LIFECYCLE_SHADOW, LIFECYCLE_LIVE)


# ---------------------------------------------------------------------------
# 2. Round-trip write -> read identical
# ---------------------------------------------------------------------------

def test_round_trip_frame_identical(lake) -> None:
    evs = [
        make_event("RT-1", hour=1),
        make_event("RT-2", hour=2, discard_reason="dedup"),
        make_event(
            "RT-3",
            hour=3,
            direction="bearish",
            target=None,
            entry=2650.0,
            stop=2660.0,
        ),
    ]
    lake.append_events(evs)
    written = events_to_frame(evs)
    read = lake.read_events()
    pdt.assert_frame_equal(written, read)
    # reconstructed dataclasses carry every field, including None/NaN edge cases
    back = {ev.event_id: ev for ev in events_from_frame(read)}
    assert back["RT-3"].target_price is None
    assert back["RT-3"].model_prob == pytest.approx(0.6)  # make_event default
    assert back["RT-1"].confluence_group_id is None
    assert back["RT-1"].target_price == 2680.0


def test_round_trip_attributes_and_levels_preserved(lake) -> None:
    ev = make_event(
        "RT-A",
        hour=1,
        discard_reason="stale",
        extra_attrs={"features": {"atr": 1.25, "vol": 0.5}, "session": "london"},
    )
    ev.structure_levels = {"neckline": 2660.0, "sweep_low": 2630.5}
    lake.append_events([ev])
    back = lake.events()[0]
    assert canonical_json(back.attributes) == canonical_json(ev.attributes)
    assert canonical_json(back.structure_levels) == canonical_json(ev.structure_levels)
    assert back.attributes["discard_reason"] == "stale"


def test_round_trip_known_at_and_na_model_prob(lake) -> None:
    ev = make_event("RT-K", hour=1)
    ev.model_prob = None
    ev.known_at = ts(3)
    lake.append_events([ev])
    back = lake.events()[0]
    assert back.model_prob is None
    assert back.known_at == ts(3)


def test_read_events_filters(lake) -> None:
    evs = [
        make_event("F-1", hour=1, symbol="XAUUSD"),
        make_event("F-2", hour=2, symbol="EURUSD"),
        make_event("F-3", hour=3, pattern="double_bottom", symbol="XAUUSD"),
    ]
    lake.append_events(evs)
    by_pattern = lake.read_events(pattern="liquidity_sweep")
    assert set(by_pattern["event_id"]) == {"F-1", "F-2"}
    by_both = lake.read_events(pattern="liquidity_sweep", symbol="XAUUSD")
    assert set(by_both["event_id"]) == {"F-1"}
    with pytest.raises(EventLakeError):
        lake.read_events(symbol="XAUUSD")  # symbol filter requires pattern
    ranged = lake.read_events(from_ts=ts(1, 30))
    assert set(ranged["event_id"]) == {"F-2", "F-3"}


# ---------------------------------------------------------------------------
# 3. Outcome tracker — forward outcomes + hypothetical PnL, 100% join
# ---------------------------------------------------------------------------

def test_tracker_target_hit_pnl_minus_costs(lake) -> None:
    ev = make_event("O-TP", hour=1)  # long 2650 SL 2640 TP 2680, entry 03:00
    lake.append_events([ev])
    candles = make_candles(start_hour=3)  # rises to 2685 -> TP reachable
    tracker = OutcomeTracker(lake, horizon=16)
    ocs = tracker.compute_outcomes(
        [ev],
        candles,
        costs={"XAUUSD": {"spread": 0.3, "commission": 0.1, "slippage": 0.0}},
    )
    oc = ocs[0]
    assert oc.exit_reason == EXIT_TARGET_HIT
    assert oc.exit_price == 2680.0
    assert oc.trade_direction == "long"
    assert oc.costs_total == pytest.approx(0.4)
    assert oc.pnl_net == pytest.approx(2680.0 - 2650.0 - 0.4)
    assert oc.pnl_net_r == pytest.approx((2680.0 - 2650.0 - 0.4) / 10.0)
    assert oc.mfe_r > 0 and oc.window_truncated is False


def test_tracker_stop_hit_conservative(lake) -> None:
    # long, stop 2660 risk 10; next bar tanks below -> stop_hit at 2660
    ev = PatternEvent(
        event_id="O-SL",
        pattern_name="liquidity_sweep",
        pattern_version="2.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=ts(1),
        confirm_time=ts(2),
        entry_time=ts(3),
        entry_price=2650.0,
        stop_price=2660.0,
        target_price=2700.0,
    )
    candles = pd.DataFrame(
        {
            "open": [2650.0, 2652.0, 2640.0, 2655.0],
            "high": [2653.0, 2655.0, 2642.0, 2658.0],
            "low": [2645.0, 2643.0, 2630.0, 2650.0],
            "close": [2652.0, 2644.0, 2635.0, 2655.0],
        },
        index=pd.date_range(ts(3), periods=4, freq="15min", tz="UTC"),
    )
    tracker = OutcomeTracker(lake, horizon=16)
    oc = tracker.compute_outcomes([ev], candles)[0]
    assert oc.exit_reason == EXIT_STOP_HIT
    assert oc.exit_price == 2660.0
    assert oc.gross_pnl == pytest.approx(10.0)  # 2660-2650 in price units
    assert oc.pnl_net == pytest.approx(10.0)


def test_tracker_timeout_at_close(lake) -> None:
    ev = make_event("O-TO", hour=1, target=None)  # no TP -> pure horizon walk
    candles = pd.DataFrame(
        {
            "open": [2650.0, 2651.0, 2652.0, 2653.0],
            "high": [2652.0, 2654.0, 2654.0, 2656.0],
            "low": [2648.0, 2649.0, 2650.0, 2651.0],
            "close": [2651.0, 2652.0, 2653.0, 2655.0],
        },
        index=pd.date_range(ts(3), periods=4, freq="15min", tz="UTC"),
    )
    tracker = OutcomeTracker(lake, horizon=4)
    oc = tracker.compute_outcomes([ev], candles)[0]
    assert oc.exit_reason == EXIT_TIMEOUT
    assert oc.exit_price == pytest.approx(2655.0)
    assert oc.window_truncated is False
    assert oc.forward_return_r == pytest.approx((2655.0 - 2650.0) / 10.0)


def test_tracker_no_data_and_truncation_flags(lake) -> None:
    ev = make_event("O-END", hour=30, target=9999.0)  # entry beyond candle frame
    candles = make_candles(start_hour=0, n_bars=6)  # frame ends at 01:15
    tracker = OutcomeTracker(lake, horizon=16)
    ocs = tracker.compute_outcomes([ev], candles)
    assert ocs[0].exit_reason == EXIT_NO_DATA
    assert ocs[0].window_truncated is True
    # event with few forward bars -> truncated but still labeled
    ev2 = make_event("O-TRUNC", hour=0)
    candles2 = make_candles(start_hour=2, n_bars=4)  # entry at 02:00 -> 4 of 16 bars
    ocs2 = tracker.compute_outcomes([ev2], candles2)
    assert ocs2[0].window_truncated is True
    assert ocs2[0].exit_reason in (EXIT_TIMEOUT, EXIT_TARGET_HIT)


def test_tracker_covers_rejected_and_shadow_events(lake) -> None:
    evs = [
        make_event("JR-1", hour=1, discard_reason="stale", lifecycle=LIFECYCLE_SHADOW),
        make_event("JR-2", hour=2, discard_reason="correlated"),
        make_event("JR-3", hour=3),
    ]
    lake.append_events(evs)
    candles = make_candles(start_hour=2)  # entries at hours 1/2/3 + 2h
    tracker = OutcomeTracker(lake, horizon=16)
    assert tracker.run(lambda p, s, tf: candles) == 3
    out = lake.read_outcomes()
    assert set(out["event_id"]) == {"JR-1", "JR-2", "JR-3"}  # 100% join
    assert len(out) == len(lake.event_ids())
    # second run is a no-op (append-only idempotent job)
    assert tracker.run(lambda p, s, tf: candles) == 0


def test_outcome_write_read_round_trip(lake) -> None:
    oc = EventOutcome(
        event_id="OC-1",
        pattern_name="liquidity_sweep",
        symbol="XAUUSD",
        timeframe="M15",
        entry_time=ts(3),
        horizon=16,
        trade_direction="long",
        exit_reason=EXIT_TIMEOUT,
        exit_price=2655.0,
        forward_return_r=0.5,
        mfe_r=0.8,
        mae_r=0.2,
        mfe=8.0,
        mae=2.0,
        gross_pnl=5.0,
        costs_total=0.4,
        pnl_net=4.6,
        pnl_net_r=0.46,
        window_truncated=False,
        computed_at=ts(10),
    )
    lake.append_outcomes([oc])
    read = lake.outcomes(["OC-1"])[0]
    assert read.event_id == oc.event_id and read.exit_price == oc.exit_price
    assert read.pnl_net == pytest.approx(4.6) and read.window_truncated is False
    with pytest.raises(EventLakeError, match="append-only"):
        lake.append_outcomes([oc])


def test_pending_event_ids(lake) -> None:
    lake.append_events([make_event("P-1", hour=1), make_event("P-2", hour=2)])
    assert lake.pending_event_ids() == ["P-1", "P-2"]
    lake.append_outcomes(
        [
            EventOutcome(
                event_id="P-1",
                pattern_name="liquidity_sweep",
                symbol="XAUUSD",
                timeframe="M15",
                entry_time=ts(3),
                horizon=16,
                trade_direction="long",
                exit_reason=EXIT_TIMEOUT,
                exit_price=2655.0,
                forward_return_r=0.5,
                mfe_r=0.8,
                mae_r=0.2,
                mfe=8.0,
                mae=2.0,
                gross_pnl=5.0,
                costs_total=0.0,
                pnl_net=5.0,
                pnl_net_r=0.5,
                window_truncated=False,
                computed_at=ts(10),
            )
        ]
    )
    assert lake.pending_event_ids() == ["P-2"]


# ---------------------------------------------------------------------------
# 4. rebuild_dataset — retrain without re-running the detector (§7.3.1)
# ---------------------------------------------------------------------------

def test_rebuild_dataset_no_detector(lake) -> None:
    evs = [
        make_event(
            "DS-1",
            hour=1,
            extra_attrs={"features": {"atr": 1.2, "vol": 0.4}},
            discard_reason="stale",
        ),
        make_event(
            "DS-2",
            hour=2,
            extra_attrs={"features": {"atr": 0.8, "vol": 0.6}},
            direction="bearish",
            entry=2650.0,
            stop=2660.0,
            target=2620.0,
        ),
    ]
    lake.append_events(evs)
    candles = make_candles(start_hour=3)
    OutcomeTracker(lake, horizon=16).run(lambda p, s, tf: candles)

    # no detector object is ever constructed — data comes entirely from the lake
    ds = lake.rebuild_dataset("liquidity_sweep", "XAUUSD")
    assert len(ds) == 2
    assert set(ds["event_id"]) == {"DS-1", "DS-2"}
    # event features flattened from attributes + outcome labels + cost columns
    assert ds["features.atr"].tolist() == pytest.approx([1.2, 0.8])
    assert "discard_reason" in ds.columns
    assert ds["discard_reason"].fillna("").tolist() == ["stale", ""]
    assert "pnl_net" in ds.columns and "costs_total" in ds.columns
    assert ds["rule_score"].tolist() == pytest.approx([0.7, 0.7])
    # outcome metrics belong to the right event (join integrity)
    assert ds.loc[ds["event_id"] == "DS-1", "pnl_net"].iloc[0] == pytest.approx(
        2680.0 - 2650.0
    )


def test_rebuild_dataset_time_window_and_incomplete(lake) -> None:
    lake.append_events(
        [make_event("W-1", hour=1), make_event("W-2", hour=40)]  # W-2 far future
    )
    candles = make_candles(start_hour=3, n_bars=20)
    OutcomeTracker(lake, horizon=16).run(lambda p, s, tf: candles)
    windowed = lake.rebuild_dataset(
        "liquidity_sweep",
        "XAUUSD",
        from_ts=ts(1),
        to_ts=ts(42),
        require_completed_outcome=False,
    )
    assert set(windowed["event_id"]) == {"W-1", "W-2"}  # window over events table
    completed = lake.rebuild_dataset("liquidity_sweep", "XAUUSD")
    # W-2 has no completed outcome -> excluded from the training slice
    assert set(completed["event_id"]) == {"W-1"}


# ---------------------------------------------------------------------------
# 5. Metrics store (rolling metrics cho LifecycleManager)
# ---------------------------------------------------------------------------

def test_metrics_append_read_sorted_and_unique_asof(lake) -> None:
    model_id = "double_bottom_xauusd_h1_v1"
    lake.append_metrics(
        model_id,
        pd.DataFrame(
            {
                "asof": [ts(1), ts(2)],
                "pf": [1.1, 1.4],
                "lifecycle_state": ["shadow", "shadow"],
            }
        ),
    )
    lake.append_metrics(
        model_id,
        pd.DataFrame({"asof": [ts(3)], "pf": [1.6], "lifecycle_state": ["live"]}),
    )
    frame = lake.read_metrics(model_id)
    assert frame["pf"].tolist() == pytest.approx([1.1, 1.4, 1.6])
    assert frame["asof"].is_monotonic_increasing
    with pytest.raises(EventLakeError, match="append-only"):
        lake.append_metrics(
            model_id, pd.DataFrame({"asof": [ts(2)], "pf": [9.9]})
        )
    assert lake.metrics_path(model_id).parent.name == "metrics"


# ---------------------------------------------------------------------------
# 6. PSI drift monitor (§5.2 / §5.3)
# ---------------------------------------------------------------------------

def test_psi_identical_vs_shifted() -> None:
    rng = np.random.default_rng(11)
    expected = rng.normal(0.0, 1.0, 5000)
    same = rng.normal(0.0, 1.0, 5000)
    shifted = rng.normal(1.5, 1.0, 5000)
    assert population_stability_index(expected, same) < 0.1
    assert population_stability_index(expected, shifted) > 0.25


def test_psi_degenerate_inputs_return_zero() -> None:
    assert population_stability_index([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert population_stability_index([], []) == 0.0


def test_drift_monitor_report_flags(lake) -> None:
    rng = np.random.default_rng(13)
    n = 4000
    train = pd.DataFrame(
        {"atr": rng.normal(0, 1, n), "vol": rng.normal(0, 1, n), "rsi": rng.normal(0, 1, n)}
    )
    live_same = pd.DataFrame(
        {"atr": rng.normal(0, 1, n), "vol": rng.normal(0, 1, n), "rsi": rng.normal(0, 1, n)}
    )
    live_drift2 = pd.DataFrame(
        {
            "atr": rng.normal(1.5, 1, n),  # >= 0.2 -> drift
            "vol": rng.normal(1.6, 1, n),  # >= 0.2 -> drift (2 features -> retrain)
            "rsi": rng.normal(0, 1, n),
        }
    )
    mon = DriftMonitor()
    ok = mon.check(train, live_same, ["atr", "vol", "rsi"])
    assert not ok.requires_retrain and not ok.demote_to_degraded
    bad = mon.check(train, live_drift2, ["atr", "vol", "rsi"])
    assert bad.n_drift >= 2
    assert bad.requires_retrain is True
    assert bad.demote_to_degraded is True  # some psi >= 0.25
    assert "atr" in bad.drifted_features()
    assert "RETRAIN" in bad.summary()


def test_psi_job_on_simulated_lake_data(lake) -> None:
    """The periodic §7.3.3 job: train vs live windows from the lake, PSI per
    core feature, fed to LifecycleManager.  Runs on simulated data."""
    rng = np.random.default_rng(17)
    events: list[PatternEvent] = []
    for i in range(120):
        hour = i // 4  # detect hours 0..29 — two 60-event halves
        drift = i >= 60  # live half of the simulated period drifts
        events.append(
            make_event(
                f"DR-{i:03d}",
                hour=hour,
                extra_attrs={
                    "features": {
                        "atr": float(rng.normal(0.0 if not drift else 1.8, 1.0)),
                        "vol": float(rng.normal(0.0 if not drift else 1.9, 1.0)),
                    }
                },
            )
        )
    lake.append_events(events)
    candles = make_candles(start_hour=0, n_bars=200)
    OutcomeTracker(lake, horizon=8).run(lambda p, s, tf: candles)

    mon = DriftMonitor()
    report = mon.check_lake_drift(
        lake,
        "liquidity_sweep",
        "XAUUSD",
        ["features.atr", "features.vol"],
        train_from=ts(0),
        train_to=ts(12),
        live_from=ts(15),
        live_to=ts(30),
    )
    assert report.requires_retrain is True
    assert report.demote_to_degraded is True
    assert report.per_feature[0].feature == "features.atr"

    # control: same window for both -> no drift
    ok = mon.check_lake_drift(
        lake,
        "liquidity_sweep",
        "XAUUSD",
        ["features.atr", "features.vol"],
        train_from=ts(0),
        train_to=ts(10),
        live_from=ts(0),
        live_to=ts(10),
    )
    assert not ok.requires_retrain


def test_lake_feature_columns_discovery(lake) -> None:
    lake.append_events(
        [
            make_event("FC-1", hour=1, extra_attrs={"features": {"atr": 1.2}}),
            make_event("FC-2", hour=2, extra_attrs={"user": 5}),
        ]
    )
    cols = lake.feature_columns("liquidity_sweep", "XAUUSD")
    assert "features.atr" in cols
    assert "event_id" not in cols