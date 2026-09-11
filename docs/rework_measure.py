"""rework_measure.py — measure overlap + SL/TP geometry for the pattern rework.

Usage:
    python docs/rework_measure.py --label baseline
    python docs/rework_measure.py --label after

Measures, on every detector in the multi-pattern registry over the full
XAUUSD M15 history:

  * overlap_rate     — fraction of same-pattern event pairs whose
                       [extreme1_bar, extreme2_bar] structure windows
                       intersect (spec §1.1 "chồng lấn").
  * opposite_rate    — fraction of DB/DT event pairs sharing the same middle
                       swing bar with opposite direction (spec §1.3.3).
  * target_atr       — |target - entry| / ATR, the spec §2.1 complaint.
  * risk_atr         — |entry - stop| / ATR.
  * rr               — target_atr / risk_atr (must stay >= 1.0).
  * winrate / expectancy / events_per_year via simulate_trade.

Writes a JSON report next to this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.engine.pattern_registry import get_registry  # noqa: E402
from research.multi_backtest.runner import (  # noqa: E402
    CostConfig,
    load_symbol_frame,
    simulate_trade,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M15"


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _sort_key(ev) -> int:
    s = _span(ev)
    return s[0] if s else 0


def _span(ev) -> tuple[int, int] | None:
    """Structure window of an event in bar units.

    DB/DT expose ``extreme1_bar``/``extreme2_bar``.  Wedges and H&S expose
    geometry-specific bars instead, so the window is derived from their own
    first/last structural anchor — never from a missing key (which would
    silently collapse to bar 0 and manufacture phantom overlaps).
    """
    a = ev.attributes or {}
    if "extreme1_bar" in a and "extreme2_bar" in a:
        return int(a["extreme1_bar"]), int(a["extreme2_bar"])
    if "low1_bar" in a and "low3_bar" in a:          # rising/falling wedge
        bars = [a.get("low1_bar"), a.get("low2_bar"), a.get("low3_bar")]
    elif "left_shoulder_bar" in a:                    # head & shoulders
        bars = [a.get("left_shoulder_bar"), a.get("head_bar"),
                a.get("right_shoulder_bar")]
    else:
        return None
    bars = [int(b) for b in bars if b is not None]
    if len(bars) < 2:
        return None
    return min(bars), max(bars)


def measure(df: pd.DataFrame, atr: np.ndarray, events: list, costs: CostConfig) -> dict:
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    out: dict = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "n_bars": len(df),
                 "years": round(n_years, 2), "patterns": {}}

    # ---- overlap (same pattern, structure windows intersect) -------------
    by_pat: dict[str, list] = {}
    for ev in events:
        by_pat.setdefault(str(ev.pattern_name), []).append(ev)

    for name, evs in sorted(by_pat.items()):
        evs = sorted(evs, key=_sort_key)
        # A pair-count rate is meaningless when windows vary hugely in length
        # (a wedge spans an entire pattern; a DB spans ~20 bars).  Count the
        # EVENTS that intersect at least one other event's window instead —
        # that is the "how many drawn boxes collide" quantity that matters,
        # and it is scale-free across pattern families.  O(n^2) with a break
        # on sorted starts, so long windows stay cheap.
        spans = [_span(e) for e in evs]
        n_ovl_events = 0
        n_pairs = 0
        for i, wi in enumerate(spans):
            if wi is None:
                continue
            hit = False
            for j in range(i + 1, len(spans)):
                wj = spans[j]
                if wj is None:
                    continue
                if wj[0] > wi[1]:
                    break  # starts are sorted; no later window can reach back
                if wj[1] >= wi[0]:
                    hit = True
                    n_pairs += 1
            if hit:
                n_ovl_events += 1

        # ---- geometry + trade outcome -----------------------------------
        tgt_atr, risk_atr, rr = [], [], []
        trades = []
        for ev in evs:
            pos = df.index.get_indexer([pd.Timestamp(ev.entry_time)], method="nearest")[0]
            pos = int(np.clip(pos, 0, len(atr) - 1))
            a = float(atr[pos]) if np.isfinite(atr[pos]) else np.nan
            risk = abs(float(ev.entry_price) - float(ev.stop_price))
            if a and a > 0 and np.isfinite(a):
                risk_atr.append(risk / a)
                if ev.target_price is not None:
                    tgt_atr.append(abs(float(ev.target_price) - float(ev.entry_price)) / a)
                    if risk > 0:
                        rr.append(abs(float(ev.target_price) - float(ev.entry_price)) / risk)
            t = simulate_trade(df, ev, costs)
            if t.executed:
                trades.append(t)

        wins = [t for t in trades if t.net_r > 0]
        reason: dict[str, int] = {}
        for t in trades:
            reason[t.exit_reason] = reason.get(t.exit_reason, 0) + 1
        out["patterns"][name] = {
            "n_events": len(evs),
            "events_per_year": round(len(evs) / n_years, 1) if n_years else None,
            "overlap_events": n_ovl_events,
            "overlap_pairs": n_pairs,
            "overlap_rate": round(n_ovl_events / len(evs), 4) if evs else 0.0,
            "target_atr_median": round(float(np.median(tgt_atr)), 3) if tgt_atr else None,
            "target_atr_p90": round(float(np.percentile(tgt_atr, 90)), 3) if tgt_atr else None,
            "target_atr_max": round(float(np.max(tgt_atr)), 3) if tgt_atr else None,
            "risk_atr_median": round(float(np.median(risk_atr)), 3) if risk_atr else None,
            "rr_median": round(float(np.median(rr)), 3) if rr else None,
            "rr_min": round(float(np.min(rr)), 3) if rr else None,
            "n_trades": len(trades),
            "winrate": round(len(wins) / len(trades), 4) if trades else None,
            "expectancy_R": round(float(np.mean([t.net_r for t in trades])), 4) if trades else None,
            "exit_reasons": reason,
        }

    # ---- opposite-direction shared-middle overlap (§1.3.3) ---------------
    # DB events share the middle swing (the high) with DT events (the low)?
    # True overlap = DB.extreme2_bar region == DT.middle bar, i.e. a DB whose
    # structure window covers a DT's neckline bar.
    opp_pairs = opp_hits = 0
    db = by_pat.get("double_bottom", [])
    dt = by_pat.get("double_top", [])
    for b in db:
        wb = _span(b)
        if wb is None:
            continue
        mb = b.attributes.get("neckline_bar")
        for t in dt:
            wt = _span(t)
            if wt is None:
                continue
            opp_pairs += 1
            mt = t.attributes.get("neckline_bar")
            # shared structure: one pattern's neckline lies inside the
            # other pattern's extreme window AND windows intersect
            shared = (mb is not None and wt[0] <= int(mb) <= wt[1]) or (
                mt is not None and wb[0] <= int(mt) <= wb[1]
            )
            if shared and not (wb[1] < wt[0] or wb[0] > wt[1]):
                opp_hits += 1
    out["opposite"] = {
        "pairs": opp_pairs,
        "hits": opp_hits,
        "rate": round(opp_hits / opp_pairs, 4) if opp_pairs else 0.0,
    }
    out["total_events"] = len(events)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--start", default="2018-06-01")
    ap.add_argument("--end", default="2026-09-03")
    args = ap.parse_args(argv)

    df = load_symbol_frame(SYMBOL, start=args.start, end=args.end)
    atr = _atr(df, 14).to_numpy(dtype=float)
    costs = CostConfig.for_symbol(SYMBOL)

    registry = get_registry()
    events = []
    for name in registry.pattern_names:
        if name in ("liquidity_sweep",):
            continue  # legacy LSW path is measured by its own golden suite
        detector = registry.create(name)
        if detector is None or not hasattr(detector, "detect"):
            continue
        try:
            evs = detector.detect(df, {})
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"  ! {name}: detect failed: {exc}")
            continue
        print(f"  {name}: {len(evs)} events")
        events.extend(evs)

    report = measure(df, atr, events, costs)
    report["label"] = args.label
    out = Path(__file__).resolve().parent / f"rework_measure_{args.label}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")
    for name, s in report["patterns"].items():
        print(
            f"  {name:22s} n={s['n_events']:4d} ovl={s['overlap_rate']:.3f} "
            f"tgtATR={s['target_atr_median']} riskATR={s['risk_atr_median']} "
            f"rr={s['rr_median']} win={s['winrate']} exp={s['expectancy_R']}"
        )
    print(f"  opposite: {report['opposite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
