"""OOS edge test — LiquiditySweep with legacy model filter (hướng a).

Replicates the LIVE inference contract (signal_engine_v2._compute_model_prob):
numeric subset of the 32 registered features (categoricals dropped),
fillna(0.0), calibrated prob via artifacts/xauusd/models/calibrator.pkl;
then gates trades by calibrated-prob threshold and simulates with the SAME
triple-barrier (target 3R / stop 1R / horizon 72 bars) + costs as
multi_backtest/runner.py simulate_trade.

Run:  cd trading_v3 && /tmp/ptv2_venv/bin/python -u research/multi_backtest/scripts/oos_lsw_model_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # trading_v3/ root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/patterns/liquidity_sweep"))  # src.* imports (golden convention)

from live.engine.signal_engine_v2 import SIGNAL_FEATURE_NAMES  # noqa: E402
from research.core.contracts import PatternEvent  # noqa: E402
from research.multi_backtest.runner import CostConfig, simulate_trade  # noqa: E402
from research.patterns.liquidity_sweep.detector import (  # noqa: E402
    LiquiditySweepDetector,
    _bar_index,
    run_anchor_position,
    run_opposite_extreme_at,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_full_frame() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "research/multi_backtest/data/XAUUSD_m15.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if df.index.tz is not None and str(df.index.tz) not in ("UTC", "Etc/UTC"):
        df.index = df.index.tz_convert("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def build_events(det, df, cfg, confirmed, sweep_out, features_by_id):
    """Mirror LiquiditySweepDetector.detect() post-chain event construction."""
    symbol = str(cfg.get("symbol", "XAUUSD"))
    timeframe = str(cfg.get("timeframe", "M15"))
    target_r = float(cfg.get("target_r", 3.0))
    buffer_atr = float(cfg.get("buffer_atr", 0.10))
    atr_series = sweep_out["atr"]
    rule_scores = {
        eid: float(rs) for eid, rs in zip(confirmed["event_id"], confirmed["rule_score"])
    }
    events: list[PatternEvent] = []
    for _, ev in confirmed.iterrows():
        eid = str(ev["event_id"])
        direction = "bullish" if str(ev.get("direction", "bullish")).lower() in ("bullish", "long") else "bearish"
        is_long = direction == "bullish"
        bar_pos = _bar_index(df.index, ev["event_time"])
        anchor = run_anchor_position(sweep_out, bar_pos, str(ev["direction"]))
        conf_time = pd.Timestamp(ev["confirmation_time"])
        conf_bar = _bar_index(df.index, conf_time)
        entry_bar = conf_bar + 1
        if entry_bar >= len(df):
            continue
        entry_price = float(df["open"].iloc[entry_bar])
        atr_sweep = float(atr_series.iloc[anchor])
        extreme = run_opposite_extreme_at(df, anchor, anchor, str(ev["direction"]))
        if is_long:
            stop_price = extreme - buffer_atr * atr_sweep
            target_price = entry_price + target_r * (entry_price - stop_price)
        else:
            stop_price = extreme + buffer_atr * atr_sweep
            target_price = entry_price - target_r * (stop_price - entry_price)
        attrs = {
            "position": int(ev.get("position", -1)),
            "level_id": str(ev.get("level_id", "")),
            "level_price": float(ev.get("level_price", 0.0)),
            "penetration_atr": float(ev.get("penetration_atr", 0.0)),
            "wick_ratio": float(ev.get("wick_ratio", 0.0)),
            "reclaim_atr": float(ev.get("reclaim_atr", 0.0)),
            "h1_trend": int(ev.get("h1_trend", 0)),
            "features": dict(features_by_id.get(eid, {})),
        }
        events.append(
            PatternEvent(
                event_id=eid,
                pattern_name=det.name,
                pattern_version=det.version,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                detect_time=pd.Timestamp(ev["event_time"]),
                confirm_time=conf_time,
                entry_time=pd.Timestamp(df.index[entry_bar]),
                entry_price=round(entry_price, 5),
                stop_price=round(stop_price, 5),
                target_price=round(target_price, 5),
                structure_levels={"level_price": float(ev.get("level_price", float("nan"))), "sweep_extreme": float(extreme)},
                rule_score=round(float(rule_scores.get(eid, 0.0)), 2),
                model_prob=None,
                attributes=attrs,
            )
        )
    return events


def prob_for_events(features_by_id, events):
    """Live inference contract: numeric features (registry order), fillna(0.0), calibrated prob."""
    rows = [dict(feats) | {"event_id": eid} for eid, feats in features_by_id.items()]
    feats_df = pd.DataFrame(rows)
    numeric_names = [
        c for c in SIGNAL_FEATURE_NAMES
        if c in feats_df.columns and pd.api.types.is_numeric_dtype(feats_df[c])
    ]
    X = feats_df[numeric_names].astype("float64").fillna(0.0)
    log(f"model X: {X.shape} (numeric subset {len(numeric_names)} of {len(SIGNAL_FEATURE_NAMES)} registered)")
    calibrator = joblib.load(str(ROOT / "artifacts/xauusd/models/calibrator.pkl"))
    proba = calibrator.predict_proba(X.values)
    prob = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
    return dict(zip(feats_df["event_id"].astype(str), prob)), numeric_names


def main() -> None:
    df = load_full_frame()
    log(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]}")

    det = LiquiditySweepDetector()
    cfg = det.get_default_config()

    log("running legacy chain once (detect->dedup->confirm->features->scores)...")
    sweep_out, confirmed, features_by_id = det._run_legacy_chain(df, cfg)
    log(f"confirmed events: {len(confirmed)}")

    events = build_events(det, df, cfg, confirmed, sweep_out, features_by_id)
    log(f"PatternEvents built: {len(events)}")

    prob_by_id, _numeric_names = prob_for_events(features_by_id, events)
    log(f"calibrated probs: {len(prob_by_id)}")

    costs = CostConfig.for_symbol("XAUUSD")
    windows = {
        "legacy_win_2018_2022H1": ("2018-01-02", "2022-06-30"),
        "oos_recent_2023H2_2026": ("2023-10-12", "2026-09-03"),
    }
    thresholds = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7]
    print("\nwindow | thr | kept | trades | PF | totalR | expR/trade", flush=True)
    for wname, (start, end) in windows.items():
        t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        for thr in thresholds:
            nets = []
            for ev in events:
                p = prob_by_id.get(str(ev.event_id))
                if p is None:
                    continue
                known = pd.Timestamp(ev.known_at_ts or ev.confirm_time or ev.detect_time)
                if not (t0 <= known <= t1):
                    continue
                if thr > 0.0 and p < thr:
                    continue
                t = simulate_trade(df, ev, costs, horizon_bars=72)
                if t.executed:
                    nets.append(t.net_r)
            n = len(nets)
            if n == 0:
                print(f"{wname} | {thr:.2f} | 0 | 0 | n/a | n/a | n/a", flush=True)
                continue
            wins = sum(v for v in nets if v > 0)
            losses = -sum(v for v in nets if v < 0)
            pf = wins / losses if losses > 0 else float("inf")
            # selection-power: split by calibrated prob (top half vs bottom half)
    print("\nselection power (model prob halves, window-filtered):", flush=True)
    for wname, (start, end) in windows.items():
        t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        scored: list[tuple[float, float]] = []
        for ev in events:
            p = prob_by_id.get(str(ev.event_id))
            if p is None:
                continue
            known = pd.Timestamp(ev.known_at_ts or ev.confirm_time or ev.detect_time)
            if not (t0 <= known <= t1):
                continue
            t = simulate_trade(df, ev, costs, horizon_bars=72)
            if t.executed:
                scored.append((p, t.net_r))
        scored.sort(key=lambda x: x[0])
        n = len(scored)
        for label, lo, hi in (("bottom", 0, n // 2), ("top", n // 2, n)):
            part = scored[lo:hi]
            if not part:
                continue
            wins = sum(v for _, v in part if v > 0)
            losses = -sum(v for _, v in part if v < 0)
            pf = wins / losses if losses > 0 else float("inf")
            tot = float(sum(v for _, v in part))
            print(f"{wname} {label}50% n={len(part)} PF={pf:.3f} totalR={tot:+.2f}R pr_min={part[0][0]:.3f} pr_max={part[-1][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()