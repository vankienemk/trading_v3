"""Export EVERY detected PatternEvent of the system's XAUUSD test to the
DebugDrawer input file (MQL5\\Files\\debug_events.txt format).

Replays the EXACT multi_backtest pipeline the report
reports/XAUUSD_multi_backtest.json came from (window 2023-01-01 ->
2026-09-03, patterns LSW/DB/DT, mode dedup, tier-2 model_prob attached),
then serializes all events, marking an event as REJECTED when its group
rep trade was rejected by the regime gate / caps (discard_reason).

Usage (from trading_v3/):
    /tmp/ptv2_venv/bin/python docs/debug_drawer/export_debug_events.py \
        [--start 2023-10-12] [--end 2026-09-03] [--max-events 0] \
        [--utc-offset-hours 0] [--out docs/debug_drawer/debug_events.txt]

Output schema (ASCII, pipe-delimited, one event per line):
    0 pattern | 1 direction(BUY/SELL) | 2 symbol | 3 detect | 4 confirm
  | 5 entry_t | 6 entry_p | 7 stop_p | 8 target_p | 9 rule_score
  | 10 model_prob | 11 discard_reason | 12 level_price | 13 extreme
Times = "YYYY.MM.DD HH:MM" (UTC by default; --utc-offset-hours shifts them
to the MT5 server-time zone for pixel-exact alignment on the chart).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # trading_v3/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402  (import after sys.path bootstrap)

from research.multi_backtest.runner import (  # noqa: E402  (bootstrap import)
    CorrelationConfig,
    CostConfig,
    build_assignments,
    load_symbol_frame,
    run_symbol_backtest,
)

PATTERNS = ["liquidity_sweep", "double_bottom", "double_top"]


def _fmt_time(ts: pd.Timestamp | None, offset_hours: int) -> str:
    if ts is None:
        return ""
    t = pd.Timestamp(ts)
    if offset_hours:
        t = t + pd.Timedelta(hours=offset_hours)
    return t.strftime("%Y.%m.%d %H:%M")


def _fmt_price(v: object) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:  # NaN
        return ""
    return f"{f:.2f}"


def _fmt_num(v: object, digits: int = 3) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:
        return ""
    return f"{f:.{digits}f}"


def _direction(d: str) -> str:
    return "BUY" if str(d).lower() in ("bullish", "long", "buy") else "SELL"


def _structure_box(
    ev: object, df: pd.DataFrame,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float | None, float | None]:
    """Real pattern structure box: first structural bar -> last structural bar.

    DB/DT : extreme1_bar .. extreme2_bar (the two equal extremes + neckline)
    LSW   : sweep_anchor_bar .. sweep_anchor_bar + confirmation_delay_bars
    The price envelope is min(low)/max(high) over that bar span, so the
    yellow wash hugs the pattern instead of the whole trade range.
    """
    a = getattr(ev, "attributes", None) or {}
    try:
        start_bar = a.get("extreme1_bar", a.get("sweep_anchor_bar"))
        end_bar = a.get("extreme2_bar")
        if end_bar is None:
            delay = a.get("confirmation_delay_bars")
            if start_bar is not None and delay is not None:
                end_bar = int(start_bar) + round(float(delay))
            else:
                end_bar = a.get("confirm_bar")
        # A3 (bug summary): the pattern is not complete at the second extreme --
        # the breakout candle (confirm_bar) belongs to the structure. Extend the
        # yellow box to confirm_bar so it covers the neckline break, i.e. the
        # candle that actually triggers the trade.
        confirm_bar = a.get("confirm_bar")
        if confirm_bar is not None:
            end_bar = max(int(end_bar if end_bar is not None else start_bar),
                          int(confirm_bar)) if start_bar is not None else confirm_bar
        if start_bar is None:
            return (None, None, None, None)
        i1, i2 = int(start_bar), int(end_bar if end_bar is not None else start_bar)
        if i2 < i1:
            i1, i2 = i2, i1
        i1 = max(0, i1)
        i2 = min(len(df) - 1, i2)
        if i2 < i1:
            return (None, None, None, None)
        low = float(df["low"].iloc[i1:i2 + 1].min())
        high = float(df["high"].iloc[i1:i2 + 1].max())
        return (pd.Timestamp(df.index[i1]), pd.Timestamp(df.index[i2]), low, high)
    except (KeyError, TypeError, ValueError, IndexError):
        return (None, None, None, None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-03")
    ap.add_argument("--max-events", type=int, default=0,
                    help="0 = unlimited; cap for a lighter chart")
    ap.add_argument("--utc-offset-hours", type=int, default=0,
                    help="shift event times into the MT5 server-time zone")
    ap.add_argument("--out", default=str(ROOT / "docs/debug_drawer/debug_events.txt"))
    args = ap.parse_args(argv)

    print(f"loading {args.symbol} M15 frame {args.start}..{args.end} …")
    df = load_symbol_frame(args.symbol, start=args.start, end=args.end)
    print(f"bars={len(df):,}")

    assignments = build_assignments(
        args.symbol, PATTERNS, timeframe="M15", df=df, attach_models=True,
    )
    corr_cfg = CorrelationConfig(
        mode="dedup", bar_seconds=900.0,
        max_total_risk_per_symbol=2.0, max_direction_cluster_risk=1.0,
    )
    result = run_symbol_backtest(
        df, assignments, corr_cfg=corr_cfg,
        costs=CostConfig.for_symbol(args.symbol),
        symbol_override=args.symbol,
    )

    # event_id -> discard reason (group rep rejected by gate/caps)
    rejected_by_event: dict[str, str] = {}
    for t in result.rejected:
        if t.cap_reason:
            rejected_by_event[t.event_id] = t.cap_reason
    for t in result.executed_trades:
        rejected_by_event.pop(t.event_id, None)  # executed -> not rejected

    lines: list[str] = []
    n_skip = 0
    n_rej = 0
    n_struct = 0
    counts: dict[str, int] = {}
    events = sorted(
        (e for e in result.events if e is not None),
        key=lambda e: (pd.Timestamp(e.known_at_ts), e.event_id),
    )
    for ev in events:
        if args.max_events and len(lines) >= args.max_events:
            break
        # entry price is mandatory; stop/target optional
        try:
            ep = float(ev.entry_price or 0.0)
        except (TypeError, ValueError):
            ep = 0.0
        if ep <= 0.0 or ep != ep:
            n_skip += 1
            continue

        pattern = str(ev.pattern_name or "").lower()
        counts[pattern] = counts.get(pattern, 0) + 1

        discard = str(ev.attributes.get("discard_reason") or "") if ev.attributes else ""
        if not discard:
            discard = rejected_by_event.get(ev.event_id, "")
        if discard:
            n_rej += 1

        level = ""
        extreme = ""
        if ev.attributes:
            level = _fmt_price(ev.attributes.get("level_price"))
            extreme = _fmt_price(ev.attributes.get("extreme"))

        # --- fields 14..17: the REAL pattern structure box -------------
        # (so the drawer can wash yellow over the pattern itself, not over
        #  the detect->confirm decision window)
        s_start, s_end, s_low, s_high = _structure_box(ev, df)
        if s_start is not None:
            n_struct += 1

        fields = [
            pattern,
            _direction(str(ev.direction)),
            str(ev.symbol or args.symbol),
            _fmt_time(ev.detect_time, args.utc_offset_hours),
            _fmt_time(getattr(ev, "confirm_time", None), args.utc_offset_hours),
            _fmt_time(ev.entry_time, args.utc_offset_hours),
            _fmt_price(ev.entry_price),
            _fmt_price(ev.stop_price),
            _fmt_price(ev.target_price),
            _fmt_num(ev.rule_score, 2),
            _fmt_num(ev.model_prob, 3),
            discard,
            level,
            extreme,
            # --- pattern structure box (fields 14..17) ---
            _fmt_time(s_start, args.utc_offset_hours),
            _fmt_time(s_end, args.utc_offset_hours),
            _fmt_price(s_low),
            _fmt_price(s_high),
        ]
        lines.append("|".join(fields))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")

    print(f"events drawn={len(lines)} skipped={n_skip} rejected={n_rej} "
          f"with_structure_box={n_struct}")
    print("per pattern:", json.dumps(counts))
    from collections import Counter
    print("reject reasons:", dict(Counter(rejected_by_event.values())))
    print(f"wrote {out} ({out.stat().st_size:,} bytes, "
          f"{out.read_text(encoding='ascii').count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())