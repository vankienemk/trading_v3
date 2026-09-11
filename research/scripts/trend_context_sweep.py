"""Trend-context gate sweep on the full XAUUSD M15 history (rework §2.1/§4.2).

Measures, for each (lookback, min_r2) combination, how many double_bottom and
double_top events survive the §2.1 gate, and whether the OOS (post-2023) count
still meets the §4.2 requirement of >= 100 events AFTER the semantic gates.

Run::

    PYTHONPATH=<venv>/lib/python3.9/site-packages <venv>/bin/python \
        research/scripts/trend_context_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.core.contracts import DIRECTION_BEARISH, DIRECTION_BULLISH  # noqa: E402
from research.core.trend_context import has_trend_context  # noqa: E402
from research.multi_backtest.runner import load_symbol_frame  # noqa: E402
from research.patterns.double_bottom.detector import (  # noqa: E402
    DoubleBottomDetector,
    atr_series,
)
from research.patterns.double_top.detector import DoubleTopDetector  # noqa: E402

OOS_START = "2023-01-01"

LOOKBACKS = [20, 30, 40, 60]
MIN_R2S = [0.2, 0.3, 0.5]
MIN_SLOPE_ATR = 0.05


def _collect(detector, df, cfg_extra):  # type: ignore[no-untyped-def]
    """Detect with the trend gate DISABLED, then evaluate it post-hoc.

    Detecting once and filtering afterwards keeps the sweep cheap and, more
    importantly, makes every combination comparable: the candidate pool is
    identical and only the gate decision changes.
    """
    cfg = detector.get_default_config()
    cfg.update(cfg_extra)
    cfg["trend_context_enabled"] = False  # measure the pool, not the gate
    events = detector.detect(df, cfg)
    atr = atr_series(df, int(cfg["atr_period"]))
    closes = df["close"].to_numpy(dtype=float)
    rows = []
    for ev in events:
        e1 = int(ev.attributes["extreme1_bar"])
        atr_bar = int(ev.attributes.get("trend_context_atr_bar", e1))
        atr_k = float(atr[atr_bar])
        rows.append(
            {
                "event": ev,
                "extreme1_bar": e1,
                "atr_k": atr_k,
                "confirm_time": ev.confirm_time,
                "rule_score": float(ev.rule_score),
            }
        )
    return rows, closes, events


def main() -> None:
    # FULL history: pass no ``start`` so the whole 204k-bar file is used.
    # An earlier version used start="2018-06-01", which silently dropped 8,224
    # leading bars (4.03%) and 22 DB + 18 DT events — see
    # rework/reviewer_adversarial_review.md R-1.  The §2.1 conclusions are
    # unchanged on the full window, but the counts here are the correct ones.
    df = load_symbol_frame("XAUUSD", end="2026-09-03")
    print(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]}")

    results: dict[str, object] = {"frame_bars": len(df), "sweep": []}

    for name, det, direction in (
        ("double_bottom", DoubleBottomDetector(), DIRECTION_BULLISH),
        ("double_top", DoubleTopDetector(), DIRECTION_BEARISH),
    ):
        rows, closes, _events = _collect(det, df, {})
        oos_rows = [
            r
            for r in rows
            if r["confirm_time"] is not None and str(r["confirm_time"]) >= OOS_START
        ]
        print(
            f"\n=== {name} === pool={len(rows)}  OOS pool={len(oos_rows)} "
            f"(confirm_time >= {OOS_START})"
        )
        for lb in LOOKBACKS:
            for r2 in MIN_R2S:
                kept = [
                    r
                    for r in rows
                    if has_trend_context(
                        closes,
                        r["atr_k"],
                        r["extreme1_bar"],
                        lb,
                        MIN_SLOPE_ATR,
                        r2,
                        direction,
                    )
                ]
                kept_oos = [
                    r
                    for r in kept
                    if r["confirm_time"] is not None
                    and str(r["confirm_time"]) >= OOS_START
                ]
                entry = {
                    "pattern": name,
                    "lookback": lb,
                    "min_r2": r2,
                    "min_slope_atr": MIN_SLOPE_ATR,
                    "pool": len(rows),
                    "kept": len(kept),
                    "kept_pct": round(100.0 * len(kept) / max(len(rows), 1), 1),
                    "kept_oos": len(kept_oos),
                    "oos_ok": len(kept_oos) >= 100,
                }
                results["sweep"].append(entry)
                flag = "OK " if entry["oos_ok"] else "LOW"
                print(
                    f"  lookback={lb:>3} min_r2={r2:<4} "
                    f"kept={len(kept):>4} ({entry['kept_pct']:>5}%)  "
                    f"OOS={len(kept_oos):>4}  {flag}"
                )

    out = ROOT / "research" / "multi_backtest" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "trend_context_sweep.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
