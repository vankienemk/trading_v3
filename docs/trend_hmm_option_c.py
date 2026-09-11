"""trend_hmm_option_c.py -- Phuong an C (HMM trend DIRECTION) measurement.

Companion to ``docs/rework_trend_hmm_measure.py`` for rework request §2.1/§2.4
option comparison: Phuong an B (regression slope heuristic,
``research/core/trend_context.py``) vs Phuong an C (3-state causal trend HMM,
``research/regime/trend_hmm.py``).

Why this is a separate module: fitting the trend HMM is expensive and must obey
the repo's causal fit-split convention, which the main measurement script has no
business doing inline.  The convention is taken from the existing volatility
regime study (``research/multi_backtest/scripts/oos_hmm_regime_filter.py``):
**fit on the 2018-01-02 -> 2023-09-30 train prefix only**, then forward-filter
(strict ``predict``) over the FULL frame.  The gate therefore never sees a state
computed from data after the bar it is attached to, and the HMM emissions are
never fit on the OOS window it is later judged on.

Evidence rule: if the module is missing, unimportable, or fails to fit, this
returns ``status="pending"`` with the reason.  It never invents numbers.

Run::

    PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages \\
        /tmp/ptv2_venv/bin/python docs/trend_hmm_option_c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Same causal fit-split the volatility regime study uses (never trimmed here).
FIT_START = "2018-01-02"
FIT_END = "2023-09-30"


def _log(msg: str) -> None:
    print(msg, flush=True)


def fit_and_predict_trend_hmm(df: pd.DataFrame) -> tuple[list[str], str]:
    """Fit on the train prefix, forward-filter the whole frame (strict causal).

    Returns ``(state_name_per_bar, note)``.  Raises on any failure so the
    caller can report it as ``pending`` rather than silently degrade.
    """
    from research.regime.trend_hmm import CausalTrendHMM

    plugin = CausalTrendHMM()
    cfg = plugin.get_default_config()
    fit_df = df.loc[pd.Timestamp(FIT_START, tz="UTC"): pd.Timestamp(FIT_END, tz="UTC")]
    if len(fit_df) < int(cfg.get("min_fit_bars", 2000)):
        raise RuntimeError(
            f"fit prefix has only {len(fit_df)} bars "
            f"(< min_fit_bars {cfg.get('min_fit_bars')})"
        )
    plugin.fit(fit_df, cfg)
    states = plugin.predict(df, cfg)
    names = [str(s.state_name) for s in states]
    if len(names) != len(df):
        raise RuntimeError(
            f"predict returned {len(names)} states for {len(df)} bars"
        )
    note = (
        f"fit on {len(fit_df)} bars {FIT_START}..{FIT_END}; "
        f"forward-filtered (no lookahead) over {len(df)} bars; "
        f"config_hash={plugin.config_hash}; "
        f"state_names={list(plugin.state_names)}"
    )
    return names, note


def main() -> int:
    from research.multi_backtest.runner import (
        CostConfig,
        load_symbol_frame,
        simulate_trade,
    )
    from research.patterns.double_bottom.detector import (
        DoubleBottomDetector,
    )
    from research.patterns.double_top.detector import DoubleTopDetector

    df = load_symbol_frame("XAUUSD", start="2018-06-01", end="2026-09-03")
    costs = CostConfig.for_symbol("XAUUSD")
    _log(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]}")

    report: dict[str, Any] = {
        "symbol": "XAUUSD",
        "n_bars": len(df),
        "fit_split": [FIT_START, FIT_END],
        "patterns": {},
    }

    try:
        state_names, note = fit_and_predict_trend_hmm(df)
    except Exception as exc:
        report["status"] = "pending"
        report["reason"] = f"Phuong an C not measurable: {type(exc).__name__}: {exc}"
        _log("!! " + report["reason"])
        Path(__file__).resolve().parent.joinpath(
            "trend_hmm_option_c.json"
        ).write_text(json.dumps(report, indent=2, default=str))
        return 0

    report["status"] = "measured"
    report["note"] = note
    state_arr = np.asarray(state_names, dtype=object)
    counts = {s: int((state_arr == s).sum()) for s in sorted(set(state_names))}
    report["state_bar_counts"] = counts
    report["state_bar_share"] = {
        k: round(v / len(df), 4) for k, v in counts.items()
    }
    _log(f"HMM states over bars: {report['state_bar_share']}")

    for pattern, det, wanted in (
        ("double_bottom", DoubleBottomDetector(), "downtrend"),
        ("double_top", DoubleTopDetector(), "uptrend"),
    ):
        cfg = det.get_default_config()
        cfg["trend_context_enabled"] = False
        cfg["max_pattern_length_bars"] = 0
        cfg["min_rule_score"] = 0.0
        events = det.detect(df, cfg)
        rows = []
        for ev in events:
            a = ev.attributes or {}
            i1 = a.get("extreme1_bar")
            keep = False
            state_at = None
            if i1 is not None and 0 <= int(i1) < len(df):
                # Point lookup at extreme1_bar.  NOTE the causality caveat the
                # trend_hmm module itself documents: for DB/DT, extreme1_bar is
                # inside the pattern window, but the state is a *causal forward
                # filter* value at that bar, so it is knowable once bar i1 has
                # closed -- which always precedes confirm_bar.  It is therefore
                # available at confirm time, not future information.
                state_at = state_names[int(i1)]
                keep = state_at == wanted
            rows.append((ev, state_at, keep))

        kept = [r for r in rows if r[2]]
        dropped = [r for r in rows if not r[2]]

        def _stats(evs: list[Any]) -> dict[str, Any]:
            tr = [simulate_trade(df, e, costs) for e in evs]
            tr = [t for t in tr if t.executed]
            if not tr:
                return {"n": 0}
            return {
                "n": len(tr),
                "winrate": round(sum(1 for t in tr if t.net_r > 0) / len(tr), 4),
                "expectancy_R": round(float(np.mean([t.net_r for t in tr])), 4),
                "total_R": round(float(sum(t.net_r for t in tr)), 2),
            }

        state_hist: dict[str, int] = {}
        for _ev, s, _k in rows:
            state_hist[str(s)] = state_hist.get(str(s), 0) + 1

        t_fixed = pd.Timestamp("2023-10-12", tz="UTC")
        kept_known = sorted(pd.Timestamp(e.known_at_ts) for e, _s, k in kept if k)
        oos_fixed = sum(1 for k in kept_known if k >= t_fixed)
        last40 = round(len(kept_known) * 0.40)

        report["patterns"][pattern] = {
            "wanted_state": wanted,
            "n_pool": len(events),
            "state_at_extreme1_histogram": state_hist,
            "kept": len(kept),
            "kept_pct": round(100.0 * len(kept) / max(len(events), 1), 1),
            "kept_outcomes": _stats([e for e, _s, _k in kept]),
            "dropped_outcomes": _stats([e for e, _s, _k in dropped]),
            "oos_fixed_split_n": oos_fixed,
            "oos_fixed_split_pass": oos_fixed >= 100,
            "oos_last40_n": last40,
            "oos_last40_pass": last40 >= 100,
        }
        p = report["patterns"][pattern]
        _log(
            f"\n{pattern}: pool={p['n_pool']} kept={p['kept']} "
            f"({p['kept_pct']}%) wanted={wanted}"
        )
        _log(f"  state histogram at extreme1: {state_hist}")
        _log(f"  kept    : {p['kept_outcomes']}")
        _log(f"  dropped : {p['dropped_outcomes']}")
        _log(
            f"  OOS fixed={oos_fixed} {'PASS' if p['oos_fixed_split_pass'] else 'FAIL'}"
            f" | last40%={last40} {'PASS' if p['oos_last40_pass'] else 'FAIL'}"
        )

    out = Path(__file__).resolve().parent / "trend_hmm_option_c.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    _log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
