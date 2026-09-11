"""rework_trend_hmm_measure.py -- mandatory measurement for the trend/HMM rework.

Decision evidence for ``rework/trading_v3_rework_request_trend_hmm.md`` §4
("do truoc, bat sau" -- measure before enabling).  Extends the pattern
established by ``docs/rework_measure.py`` (overlap / target-risk geometry /
winrate / expectancy / events-per-year via ``runner.simulate_trade``), and adds
everything §4 requires that the older tool does not do:

1. Each gate ALONE and CUMULATIVELY, per pattern (double_bottom/double_top):
   ``baseline`` -> ``2.2`` (max_pattern_length_bars) -> ``2.3`` (rule_score
   floor) -> ``2.1`` (trend-context) -> ``all``.
2. The §4.2 OOS sample-size gate: OOS events counted AFTER the semantic gates
   (2.1 + 2.2) are applied, never before.  Reported as an explicit PASS/FAIL.
3. ``rule_score`` and ``model_prob`` distributions per pattern, so the §2.3
   thresholds are justified by data instead of copied from bug_summary B1.
4. The §2.1 premise check: do the specific problem events cited by the request
   survive the gates?  Plus explicit FALSE-DROP examples (the cost side).
5. Trend-gate option comparison (Phuong an B slope heuristic vs Phuong an C
   HMM trend regime) when the other option's module exists.

Evidence rules obeyed here:

* ``model_prob`` is NEVER substituted with a default.  It is computed offline
  through the SAME contract the live engine uses
  (``signal_engine_v2._compute_model_prob``: canonical causal
  ``build_feature_frame`` matrix -> per-pattern ``calibrator.pkl`` ->
  ``predict_proba[:, 1]``).  If an artifact is missing for a pattern the field
  is reported as ``"not available"`` and the dependent gate is reported as
  NOT MEASURABLE, never as a number.
* Every reported number is produced by running this script; nothing is copied
  from a document.

Usage::

    PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages \\
        /tmp/ptv2_venv/bin/python docs/rework_trend_hmm_measure.py

Writes ``docs/rework_trend_hmm_measure.json`` next to this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.core.walkforward_trainer import build_feature_frame  # noqa: E402
from research.multi_backtest.runner import (  # noqa: E402
    CostConfig,
    load_symbol_frame,
    simulate_trade,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
START = "2018-06-01"
END = "2026-09-03"

#: §4.2 -- "OOS >= 100 events AFTER the semantic gates (2.1 + 2.2)".
OOS_MIN_EVENTS = 100

#: OOS split.  The repo has TWO conventions and the request does not pin one,
#: so both are reported side by side and neither is silently preferred:
#:
#:   * ``TRAIN_SPLIT`` -- what the model training itself uses
#:     (``sequential_split(oos_frac=0.40)`` in
#:     ``patterns/double_bottom/scripts/train_walkforward.py``): the last 40 %
#:     of events by known-at bar.  This is the split the SHIPPED ``model_prob``
#:     was fit on, so a model_prob-based gate is only honest if evaluated on
#:     the part the model did not see.
#:   * ``FIXED_SPLIT`` -- the absolute-date convention used by
#:     ``research/multi_backtest/scripts/oos_lsw_model_filter.py``
#:     (``oos_recent_2023H2_2026`` = 2023-10-12..2026-09-03) and by
#:     ``research/scripts/trend_context_sweep.py`` (``OOS_START=2023-01-01``).
#:
#: ``FIXED_SPLIT`` is the DEFAULT for the headline §4.2 number because it is
#: the absolute-date convention the sibling OOS tooling uses, and because a
#: fixed date is reproducible independently of how many events a gate removes.
TRAIN_OOS_FRAC = 0.40
FIXED_OOS_START = "2023-10-12"

#: Gate parameter defaults to measure.  §2.2 proposes 40-60 bars; the sweep
#: covers 26..60 so the choice can be justified rather than assumed.
#:
#: 26..29 are included because the measured span maxima are DB 29 / DT 26: the
#: arms just BELOW each maximum are what prove the gate is a no-op at the
#: proposed values (every arm from the pattern's own max upward drops 0), and
#: they pin the tightest 0-drop bound a default could defensibly use.
MAX_LEN_SWEEP = [26, 27, 28, 29, 30, 40, 50, 60]

#: The value gate_engineer actually shipped as the detector default
#: (``get_default_config()["max_pattern_length_bars"]``), measured 2026-09-11.
#: The cumulative arms use THIS rather than a sweep value so "cumulative"
#: describes the configuration that would really run.
SHIPPED_MAX_LEN = 30
SHIPPED_RULE_FLOOR = 0.6
#: §2.3 proposes RULE_SCORE_FLOOR = 0.6; the sweep straddles it.
RULE_FLOOR_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7]
#: §2.1 Phuong an B defaults (matching research/core/trend_context.py).
TREND_LOOKBACK = 30
TREND_MIN_SLOPE_ATR = 0.05
TREND_MIN_R2 = 0.3
TREND_SWEEP = [(20, 0.2), (20, 0.3), (30, 0.3), (30, 0.5), (40, 0.3), (60, 0.3)]

#: Requested-from-§2.3-bug_summary-B1 PROB_THRESHOLDS are NOT used as gates
#: here; they are only plotted against the measured distribution.
B1_PROPOSED_PROB = {"double_bottom": 0.55, "double_top": 0.55}

#: Problem events cited by the request (§1.3 / §7).  Windows are searched in
#: the FIXED_OOS window and matched on the cited model_prob where given.
PREMISE_PROB_TOL = 0.006


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# model_prob -- live inference contract, computed offline
# ---------------------------------------------------------------------------

def _model_paths(pattern: str) -> dict[str, Path]:
    return {
        "double_bottom": ROOT / "research/patterns/double_bottom/artifacts/models"
        / "double_bottom_xauusd_m15_v1",
        "double_top": ROOT / "research/patterns/double_top/artifacts/models"
        / "double_top_xauusd_m15_v1",
    }.get(pattern, Path("/nonexistent"))


def compute_model_prob(
    df: pd.DataFrame, pattern: str, events: list[Any]
) -> tuple[dict[str, float] | None, str]:
    """Calibrated tier-2 probability per event, or ``(None, reason)``.

    Mirrors ``signal_engine_v2._compute_model_prob`` exactly: the canonical
    causal feature matrix (:func:`build_feature_frame`, every column available
    at or before the confirm bar), restricted to the cached feature-schema
    numeric subset, ``fillna(0.0)``, then the per-pattern artifact's
    ``predict_proba`` (the calibrated probability of the positive class).

    Returns ``(None, "<why not available>")`` when the artifact or the
    features cannot be produced -- the caller must then report the
    model_prob-dependent gate as NOT MEASURABLE rather than assume a value.
    """
    import joblib

    model_dir = _model_paths(pattern)
    if not model_dir.exists():
        return None, f"no artifact directory at {model_dir}"
    schema_path = model_dir / "features.json"
    calibrator_path = model_dir / "calibrator.pkl"
    if not schema_path.exists():
        return None, f"no features.json at {schema_path}"
    if not calibrator_path.exists():
        return None, f"no calibrator.pkl at {calibrator_path}"

    schema = json.loads(schema_path.read_text())
    wanted = list(schema.get("features", []))
    if not wanted:
        return None, "features.json declares no features"

    X, _names = build_feature_frame(df, events)
    if X.empty:
        return None, "build_feature_frame produced no rows"

    usable = [
        c for c in wanted
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if not usable:
        return None, f"none of the {len(wanted)} schema features are numeric in X"
    missing = [c for c in wanted if c not in usable]

    calibrator = joblib.load(str(calibrator_path))
    try:
        proba = calibrator.predict_proba(X[usable].astype("float64").fillna(0.0).values)
    except Exception as exc:  # pragma: no cover - diagnostic
        return None, f"predict_proba failed: {exc}"
    if proba.ndim != 2:
        return None, f"unexpected predict_proba shape {proba.shape}"
    col = 1 if proba.shape[1] >= 2 else 0
    probs = {str(eid): float(p) for eid, p in zip(X.index, proba[:, col])}
    note = (
        f"ok; {len(usable)}/{len(wanted)} schema features used"
        + (f"; dropped (absent/non-numeric): {missing}" if missing else "")
    )
    return probs, note


# ---------------------------------------------------------------------------
# Gate decisions -- post-hoc, so the candidate pool is identical across arms
# ---------------------------------------------------------------------------

def _gate_max_len(ev: Any, max_len: int | None) -> bool:
    """§2.2 -- keep when the structure span is within ``max_len`` bars."""
    if max_len is None:
        return True
    a = ev.attributes or {}
    i1, i3 = a.get("extreme1_bar"), a.get("extreme2_bar")
    if i1 is None or i3 is None:
        return True  # not a double pattern -- the gate does not apply
    return (int(i3) - int(i1)) <= int(max_len)


def _gate_rule(ev: Any, floor: float | None) -> bool:
    """§2.3 detector side -- keep when ``rule_score`` clears the floor."""
    if floor is None:
        return True
    return float(ev.rule_score or 0.0) >= float(floor)


def _gate_prob(ev: Any, prob: float | None, probs: dict[str, float] | None) -> bool:
    """§2.3 live side -- keep when the tier-2 ``model_prob`` clears the bar."""
    if prob is None or probs is None:
        return True
    p = probs.get(str(ev.event_id))
    if p is None:
        return True  # unavailable -> the caller reports NOT MEASURABLE
    return p >= float(prob)


def _trend_ok(
    closes: np.ndarray, atr: np.ndarray, ev: Any, lookback: int, min_r2: float,
    min_slope_atr: float,
) -> bool | None:
    """§2.1 Phuong an B decision, or ``None`` when not evaluable.

    Uses the shared ``research/core/trend_context.has_trend_context`` so this
    measurement cannot drift from the implementation it is measuring.  The ATR
    is read at ``extreme1_bar`` (never the confirm bar) so no post-pattern
    volatility leaks into the decision; the direction sign is passed so a
    double_bottom requires a preceding DOWNTREND.
    """
    try:
        from research.core.trend_context import has_trend_context
    except Exception:
        return None
    a = ev.attributes or {}
    i1 = a.get("extreme1_bar")
    if i1 is None:
        return None
    i1 = int(i1)
    if i1 < 0 or i1 >= len(atr) or not np.isfinite(atr[i1]) or atr[i1] <= 0:
        return None
    return bool(
        has_trend_context(
            closes, float(atr[i1]), i1, int(lookback), float(min_slope_atr),
            float(min_r2), str(ev.direction),
        )
    )


# ---------------------------------------------------------------------------
# Per-arm statistics
# ---------------------------------------------------------------------------

def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 4) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(np.median(values)), 4) if values else None


def stats_for(
    df: pd.DataFrame, atr: np.ndarray, events: list[Any], costs: CostConfig,
    years: float, probs: dict[str, float] | None,
) -> dict[str, Any]:
    """winrate / expectancy(R) / target-distances(ATR) + model_prob summary."""
    tgt_atr, risk_atr, rr, trades = [], [], [], []
    for ev in events:
        pos = df.index.get_indexer([pd.Timestamp(ev.entry_time)], method="nearest")[0]
        pos = int(np.clip(pos, 0, len(atr) - 1))
        a = float(atr[pos]) if np.isfinite(atr[pos]) else float("nan")
        risk = abs(float(ev.entry_price) - float(ev.stop_price))
        if np.isfinite(a) and a > 0:
            risk_atr.append(risk / a)
            if ev.target_price is not None:
                tgt_atr.append(abs(float(ev.target_price) - float(ev.entry_price)) / a)
                if risk > 0:
                    rr.append(abs(float(ev.target_price) - float(ev.entry_price)) / risk)
        t = simulate_trade(df, ev, costs)
        if t.executed:
            trades.append(t)

    wins = [t for t in trades if t.net_r > 0]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    prob_vals = (
        [probs[str(ev.event_id)] for ev in events if probs and str(ev.event_id) in probs]
        if probs else []
    )
    out: dict[str, Any] = {
        "n_events": len(events),
        "events_per_year": round(len(events) / years, 1) if years > 0 else None,
        "n_trades": len(trades),
        "winrate": round(len(wins) / len(trades), 4) if trades else None,
        "expectancy_R": (
            round(float(np.mean([t.net_r for t in trades])), 4) if trades else None
        ),
        "total_R": round(float(sum(t.net_r for t in trades)), 2) if trades else None,
        "target_atr_mean": (
            round(float(np.mean(tgt_atr)), 3) if tgt_atr else None
        ),
        "target_atr_median": _median(tgt_atr),
        "target_atr_p90": _pct(tgt_atr, 90),
        "risk_atr_median": _median(risk_atr),
        "rr_median": _median(rr),
        "exit_reasons": reasons,
    }
    if prob_vals:
        out["model_prob"] = {
            "n": len(prob_vals),
            "min": round(min(prob_vals), 4),
            "p10": _pct(prob_vals, 10),
            "p25": _pct(prob_vals, 25),
            "median": _median(prob_vals),
            "p75": _pct(prob_vals, 75),
            "p90": _pct(prob_vals, 90),
            "max": round(max(prob_vals), 4),
            "mean": round(float(np.mean(prob_vals)), 4),
        }
        # what a §2.3 PROB threshold from bug_summary B1 would actually keep
        out["model_prob_keep_rate"] = {
            str(thr): round(
                sum(1 for v in prob_vals if v >= thr) / len(prob_vals), 4
            )
            for thr in (0.4, 0.45, 0.5, 0.55, 0.6)
        }
    else:
        out["model_prob"] = "not available"
        out["model_prob_keep_rate"] = "not available"
    return out


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "p05": _pct(values, 5),
        "p10": _pct(values, 10),
        "p25": _pct(values, 25),
        "median": _median(values),
        "p75": _pct(values, 75),
        "p90": _pct(values, 90),
        "max": round(max(values), 4),
        "mean": round(float(np.mean(values)), 4),
    }


# ---------------------------------------------------------------------------
# Premise check (§1.3 cited events) + false drops
# ---------------------------------------------------------------------------

def _near(events: list[Any], probs: dict[str, float] | None,
          lo: str, hi: str, prob: float | None) -> list[Any]:
    t0, t1 = pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC")
    out = []
    for ev in events:
        known = pd.Timestamp(ev.known_at_ts)
        if not (t0 <= known <= t1):
            continue
        if prob is not None and probs is not None:
            p = probs.get(str(ev.event_id))
            if p is None or abs(p - prob) > PREMISE_PROB_TOL:
                continue
        out.append(ev)
    return out


def _describe(df: pd.DataFrame, atr: np.ndarray, ev: Any,
              probs: dict[str, float] | None) -> dict[str, Any]:
    a = ev.attributes or {}
    i1, i3 = int(a.get("extreme1_bar", -1)), int(a.get("extreme2_bar", -1))
    pos = df.index.get_indexer([pd.Timestamp(ev.entry_time)], method="nearest")[0]
    pos = int(np.clip(pos, 0, len(atr) - 1))
    atr_k = float(atr[pos]) if np.isfinite(atr[pos]) else float("nan")
    tgt = (
        abs(float(ev.target_price) - float(ev.entry_price)) / atr_k
        if ev.target_price is not None and atr_k > 0 else None
    )
    return {
        "event_id": str(ev.event_id),
        "known_at": str(ev.known_at_ts),
        "entry_time": str(ev.entry_time),
        "direction": str(ev.direction),
        "rule_score": round(float(ev.rule_score or 0.0), 4),
        "model_prob": (
            round(float(probs[str(ev.event_id)]), 4)
            if probs and str(ev.event_id) in probs else "not available"
        ),
        "pattern_length_bars": (i3 - i1) if i1 >= 0 and i3 >= 0 else None,
        "depth_atr": (
            round(float(a["depth_atr"]), 3) if "depth_atr" in a else None
        ),
        "target_atr": round(tgt, 3) if tgt is not None else None,
        "extreme1_time": (
            str(df.index[i1]) if 0 <= i1 < len(df) else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args(argv)

    df = load_symbol_frame(SYMBOL, start=args.start, end=args.end)
    years = (df.index[-1] - df.index[0]).days / 365.25
    costs = CostConfig.for_symbol(SYMBOL)
    _log(f"frame: {len(df)} bars {df.index[0]} -> {df.index[-1]} ({years:.2f} y)")

    from research.patterns.double_bottom.detector import (
        DoubleBottomDetector,
        atr_series,
    )
    from research.patterns.double_top.detector import DoubleTopDetector

    report: dict[str, Any] = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "n_bars": len(df),
        "window": [str(df.index[0]), str(df.index[-1])],
        "years": round(years, 2),
        # The brief calls this "the full 204k-bar history" while this frame
        # reports 195,893.  Both are correct about different things, so record
        # the reconciliation in the artifact instead of leaving a reader to
        # wonder which number is wrong: 204,133 is the RAW parquet row count
        # (= load_symbol_frame with no window), and 195,893 is the same data
        # after the 2018-06-01 start filter drops the 8,240-row prefix.  Every
        # number in this report is computed on the 195,893-row windowed frame.
        "bar_count_note": (
            "n_bars is the WINDOWED frame (start=2018-06-01); the raw parquet "
            "has 204,133 rows over 2018-01-02..2026-09-03.  Same dataset, "
            "different window; the 8,240-row delta is the dropped prefix."
        ),
        # recomputed rather than hardcoded so the reconciliation cannot rot
        "raw_frame_bars": len(load_symbol_frame(SYMBOL)),
        "oos_min_events": OOS_MIN_EVENTS,
        "oos_split_fixed_start": FIXED_OOS_START,
        "oos_split_train_frac": TRAIN_OOS_FRAC,
        "patterns": {},
        "gate_method": (
            "Detect ONCE with every new gate disabled, then apply each gate "
            "post-hoc to the identical candidate pool: all arms differ only "
            "in the gate decision, never in the pool.  2.1 uses the shipped "
            "research/core/trend_context.has_trend_context; model_prob uses "
            "the live inference contract (build_feature_frame + per-pattern "
            "calibrator)."
        ),
        "env": {
            "python": sys.executable,
            "command": (
                "PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/"
                "site-packages /tmp/ptv2_venv/bin/python "
                "docs/rework_trend_hmm_measure.py"
            ),
        },
    }

    for pattern, det in (
        ("double_bottom", DoubleBottomDetector()),
        ("double_top", DoubleTopDetector()),
    ):
        _log(f"\n=== {pattern} ===")
        cfg = det.get_default_config()
        # Measure the POOL: every new gate off, so each arm is comparable.
        cfg["trend_context_enabled"] = False
        cfg["max_pattern_length_bars"] = 0
        cfg["min_rule_score"] = 0.0
        events = det.detect(df, cfg)
        atr = atr_series(df, int(cfg["atr_period"]))
        closes = df["close"].to_numpy(dtype=float)
        _log(f"  candidate pool: {len(events)} events")

        probs, prob_note = compute_model_prob(df, pattern, events)
        _log(f"  model_prob: {prob_note}")
        if probs is None:
            _log("  !! model_prob NOT AVAILABLE -> the §2.3 live gate is "
                 "reported as not measurable, never assumed")

        pat: dict[str, Any] = {
            "n_pool": len(events),
            "model_prob_status": prob_note,
            "model_prob_available": probs is not None,
            "pool": stats_for(df, atr, events, costs, years, probs),
            "distributions": {},
            "arms": {},
            "oos": {},
            "premise": {},
            "false_drops": {},
        }

        # ---- distributions (§4 item 3) ---------------------------------
        pat["distributions"]["rule_score"] = distribution(
            [float(e.rule_score or 0.0) for e in events]
        )
        pat["distributions"]["pattern_length_bars"] = distribution(
            [
                float(int(e.attributes["extreme2_bar"]) - int(e.attributes["extreme1_bar"]))
                for e in events
                if "extreme1_bar" in (e.attributes or {})
                and "extreme2_bar" in (e.attributes or {})
            ]
        )
        if probs:
            pat["distributions"]["model_prob"] = distribution(
                [probs[str(e.event_id)] for e in events if str(e.event_id) in probs]
            )
            # joint: how does model_prob relate to a realised outcome?
            pairs = []
            for e in events:
                p = probs.get(str(e.event_id))
                if p is None:
                    continue
                t = simulate_trade(df, e, costs)
                if t.executed:
                    pairs.append((p, t.net_r))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                half = len(pairs) // 2
                low, high = pairs[:half], pairs[half:]
                pat["distributions"]["model_prob_vs_outcome"] = {
                    "n_scored": len(pairs),
                    "bottom_half": {
                        "n": len(low),
                        "mean_prob": round(float(np.mean([p for p, _ in low])), 4),
                        "winrate": round(
                            sum(1 for _, r in low if r > 0) / len(low), 4
                        ),
                        "expectancy_R": round(
                            float(np.mean([r for _, r in low])), 4
                        ),
                    },
                    "top_half": {
                        "n": len(high),
                        "mean_prob": round(float(np.mean([p for p, _ in high])), 4),
                        "winrate": round(
                            sum(1 for _, r in high if r > 0) / len(high), 4
                        ),
                        "expectancy_R": round(
                            float(np.mean([r for _, r in high])), 4
                        ),
                    },
                }

        # ---- cumulative arms (§4 item 1) -------------------------------
        arms: list[tuple[str, dict[str, Any]]] = []
        arms.append(("baseline", {}))
        arms.extend(
            (f"2.2_max_len_{lim}", {"max_len": lim}) for lim in MAX_LEN_SWEEP
        )
        arms.extend(
            (f"2.3_rule_floor_{f}", {"rule": f}) for f in RULE_FLOOR_SWEEP
        )
        arms.extend(
            (f"2.1_trend_lb{lb}_r2{r2}", {"trend": (lb, r2)})
            for lb, r2 in TREND_SWEEP
        )
        # cumulative, at the request's proposed values
        arms.append(("cum_2.2+2.3", {"max_len": SHIPPED_MAX_LEN,
                                     "rule": SHIPPED_RULE_FLOOR}))
        arms.append(("cum_2.1+2.2", {"max_len": SHIPPED_MAX_LEN,
                                     "trend": (TREND_LOOKBACK, TREND_MIN_R2)}))
        arms.append(
            ("cum_all_2.1+2.2+2.3",
             {"max_len": SHIPPED_MAX_LEN, "rule": SHIPPED_RULE_FLOOR,
              "trend": (TREND_LOOKBACK, TREND_MIN_R2)})
        )

        trend_evaluable: bool | None = None
        for name, g in arms:
            kept = []
            for ev in events:
                if not _gate_max_len(ev, g.get("max_len")):
                    continue
                if not _gate_rule(ev, g.get("rule")):
                    continue
                if "trend" in g:
                    ok = _trend_ok(closes, atr, ev, g["trend"][0], g["trend"][1],
                                   TREND_MIN_SLOPE_ATR)
                    if ok is None:
                        trend_evaluable = False
                        continue
                    trend_evaluable = True
                    if not ok:
                        continue
                kept.append(ev)
            s = stats_for(df, atr, kept, costs, years, probs)
            s["kept_pct"] = round(100.0 * len(kept) / max(len(events), 1), 1)
            pat["arms"][name] = s

        pat["trend_gate_evaluable"] = trend_evaluable is not False

        # ---- the SHIPPED gate, measured directly (§2.3) ------------------
        # Once live/engine/hard_gate.probability_gate exists it is the
        # authoritative implementation; measure IT rather than this script's
        # replica, and record both so a divergence is visible.
        shipped: dict[str, Any] = {"status": "not available"}
        try:
            from live.engine.hard_gate import (
                DEFAULT_PROB_GATE_ENABLED,
                DEFAULT_RULE_SCORE_FLOOR,
                probability_gate,
            )

            shipped["defaults"] = {
                "DEFAULT_PROB_GATE_ENABLED": bool(DEFAULT_PROB_GATE_ENABLED),
                "DEFAULT_RULE_SCORE_FLOOR": DEFAULT_RULE_SCORE_FLOOR,
            }
            if probs:
                for e in events:
                    p = probs.get(str(e.event_id))
                    if p is not None:
                        e.model_prob = p
            # OOS validity of the prob gate: the SHIPPED model was fit on the
            # last-40% ... no: on the train split, so the honest test is
            # in-sample vs post-2023-10-12 separately.
            t_fixed = pd.Timestamp(FIXED_OOS_START, tz="UTC")
            is_ev = [e for e in events if pd.Timestamp(e.known_at_ts) < t_fixed]
            oos_ev = [e for e in events if pd.Timestamp(e.known_at_ts) >= t_fixed]

            def _ax(
                thresholds: dict[str, Any],
                subset: list[Any],
                _atr: np.ndarray = atr,
                _probs: dict[str, float] | None = probs,
            ) -> dict[str, Any]:
                keep = [e for e in subset
                        if probability_gate(e, thresholds)[0]]
                s = stats_for(df, _atr, keep, costs, years, _probs)
                s["kept"] = len(keep)
                s["kept_pct_of_subset"] = round(
                    100.0 * len(keep) / max(len(subset), 1), 1)
                return s

            gate_rows: dict[str, Any] = {}
            for thr in (0.3, 0.4, 0.5, 0.55, 0.6):
                key = f"min_model_prob_{thr}"
                gate_rows[key] = {
                    "all": _ax({"enabled": True, "min_model_prob": thr}, events),
                    "in_sample": _ax({"enabled": True, "min_model_prob": thr}, is_ev),
                    "oos": _ax({"enabled": True, "min_model_prob": thr}, oos_ev),
                }
            for fl in (0.5, 0.6, 0.65, 0.7):
                key = f"min_rule_score_{fl}"
                gate_rows[key] = {
                    "all": _ax({"enabled": True, "min_rule_score": fl}, events),
                    "in_sample": _ax({"enabled": True, "min_rule_score": fl}, is_ev),
                    "oos": _ax({"enabled": True, "min_rule_score": fl}, oos_ev),
                }
            gate_rows["AND_prob0.4_rule0.6"] = {
                "all": _ax({"enabled": True, "min_model_prob": 0.4,
                            "min_rule_score": 0.6}, events),
                "in_sample": _ax({"enabled": True, "min_model_prob": 0.4,
                                  "min_rule_score": 0.6}, is_ev),
                "oos": _ax({"enabled": True, "min_model_prob": 0.4,
                            "min_rule_score": 0.6}, oos_ev),
            }
            # the request's proposed B1 values, for the record
            proposed = B1_PROPOSED_PROB.get(pattern, 0.55)
            gate_rows["PROPOSED_B1_0.55"] = {
                "all": _ax({"enabled": True, "min_model_prob": proposed}, events),
                "in_sample": _ax({"enabled": True, "min_model_prob": proposed}, is_ev),
                "oos": _ax({"enabled": True, "min_model_prob": proposed}, oos_ev),
            }
            shipped["thresholds"] = gate_rows
            shipped["oos_split"] = FIXED_OOS_START
            shipped["n_in_sample"] = len(is_ev)
            shipped["n_oos"] = len(oos_ev)
            shipped["status"] = "measured (live/engine/hard_gate.probability_gate)"
        except Exception as exc:
            shipped["status"] = f"not available: {type(exc).__name__}: {exc}"
        pat["shipped_prob_gate"] = shipped

        # ---- OOS sample size (§4 item 2) -------------------------------
        t_fixed = pd.Timestamp(FIXED_OOS_START, tz="UTC")
        for arm in ("baseline", "cum_2.1+2.2", "cum_all_2.1+2.2+2.3"):
            g = dict(
                {"baseline": {},
                 "cum_2.1+2.2": {"max_len": SHIPPED_MAX_LEN,
                                 "trend": (TREND_LOOKBACK, TREND_MIN_R2)},
                 "cum_all_2.1+2.2+2.3": {
                     "max_len": SHIPPED_MAX_LEN, "rule": SHIPPED_RULE_FLOOR,
                     "trend": (TREND_LOOKBACK, TREND_MIN_R2)}}[arm]
            )
            kept = []
            for ev in events:
                if not _gate_max_len(ev, g.get("max_len")):
                    continue
                if not _gate_rule(ev, g.get("rule")):
                    continue
                if "trend" in g:
                    if _trend_ok(closes, atr, ev, g["trend"][0], g["trend"][1],
                                 TREND_MIN_SLOPE_ATR) is not True:
                        continue
                kept.append(ev)
            known = sorted(pd.Timestamp(e.known_at_ts) for e in kept)
            n = len(known)
            train_boundary = (
                str(known[max(0, n - round(n * TRAIN_OOS_FRAC))]) if n else None
            )
            fixed_n = sum(1 for k in known if k >= t_fixed)
            pat["oos"][arm] = {
                "n_total": n,
                "fixed_split_start": FIXED_OOS_START,
                "fixed_split_n": fixed_n,
                "fixed_split_pass": fixed_n >= OOS_MIN_EVENTS,
                "train_split_frac": TRAIN_OOS_FRAC,
                "train_split_boundary": train_boundary,
                "train_split_n": round(n * TRAIN_OOS_FRAC) if n else 0,
                "train_split_pass": (round(n * TRAIN_OOS_FRAC) >= OOS_MIN_EVENTS)
                if n else False,
            }
            _log(
                f"  OOS[{arm}] total={n} fixed(>= {FIXED_OOS_START})={fixed_n} "
                f"{'PASS' if fixed_n >= OOS_MIN_EVENTS else 'FAIL'} | "
                f"last-40%={round(n * TRAIN_OOS_FRAC) if n else 0}"
            )

        # ---- premise check (§1.3 cited events) -------------------------
        oos_lo, oos_hi = FIXED_OOS_START, str(df.index[-1].date())
        cited: dict[str, Any] = {}
        cases = {
            "double_top_prob_0.033": (
                pattern == "double_top", oos_lo, oos_hi, 0.033,
            ),
            "double_bottom_prob_0.069_aug27": (
                pattern == "double_bottom", "2026-08-25", "2026-08-30", 0.069,
            ),
            "double_bottom_prob_0.069_jul08": (
                pattern == "double_bottom", "2026-07-01", "2026-07-12", 0.069,
            ),
        }
        for label, (applies, lo, hi, prob) in cases.items():
            if not applies:
                continue
            hits = _near(events, probs, lo, hi, prob)
            if not hits and prob is not None:
                # widen: list every event in the window so a mismatch in the
                # cited prob is visible rather than silently reported as zero
                hits_all = _near(events, probs, lo, hi, None)
                cited[label] = {
                    "matched_by_prob": None,
                    "window": [lo, hi],
                    "cited_prob": prob,
                    "events_in_window": [
                        _describe(df, atr, e, probs) for e in hits_all
                    ],
                }
                continue
            cited[label] = {
                "window": [lo, hi],
                "cited_prob": prob,
                "matches": [_describe(df, atr, e, probs) for e in hits],
            }
            for e in hits:
                d = cited[label]["matches"][0]
                g22 = _gate_max_len(e, 40)
                g21 = _trend_ok(closes, atr, e, TREND_LOOKBACK, TREND_MIN_R2,
                                TREND_MIN_SLOPE_ATR)
                g23r = _gate_rule(e, 0.6)
                p = probs.get(str(e.event_id)) if probs else None
                g23p = None if p is None else (p >= 0.55)
                cited[label]["matches"][0].update({
                    "gate_2.2_max_len_40_pass": bool(g22),
                    "gate_2.1_trend_pass": g21,
                    "gate_2.3_rule_floor_0.6_pass": bool(g23r),
                    "gate_2.3_prob_0.55_pass": g23p,
                    "caught_by_at_least_one": not (g22 and g21 and g23r and
                                                   (True if g23p is None else g23p)),
                })
        pat["premise"]["cited_events"] = cited

        # 2026-07-06 DT/DB shared-swing case (both patterns)
        j = _near(events, probs, "2026-07-04", "2026-07-09", None)
        if j:
            pat["premise"]["shared_swing_2026_07_06"] = [
                _describe(df, atr, e, probs) for e in j
            ]

        # ---- false drops -------------------------------------------------
        cum = {}
        kept_all, dropped = [], []
        for ev in events:
            ok = (
                _gate_max_len(ev, SHIPPED_MAX_LEN)
                and _gate_rule(ev, SHIPPED_RULE_FLOOR)
                and _trend_ok(closes, atr, ev, TREND_LOOKBACK, TREND_MIN_R2,
                              TREND_MIN_SLOPE_ATR) is True
            )
            (kept_all if ok else dropped).append(ev)
        cum["n_kept"] = len(kept_all)
        cum["n_dropped"] = len(dropped)
        if dropped and kept_all:
            def _wr(evs: list[Any]) -> dict[str, Any]:
                tr = [simulate_trade(df, e, costs) for e in evs]
                tr = [t for t in tr if t.executed]
                if not tr:
                    return {"n": 0}
                w = [t for t in tr if t.net_r > 0]
                return {
                    "n": len(tr),
                    "winrate": round(len(w) / len(tr), 4),
                    "expectancy_R": round(float(np.mean([t.net_r for t in tr])), 4),
                    "total_R": round(float(sum(t.net_r for t in tr)), 2),
                }
            cum["dropped_outcomes"] = _wr(dropped)
            cum["kept_outcomes"] = _wr(kept_all)
            # false-drop candidates: dropped events that LOOK valid -- good
            # rule_score (the geometry the detector trusts) AND a decent
            # realized outcome (the trade would have worked).
            good = []
            for e in dropped:
                t = simulate_trade(df, e, costs)
                if (
                    float(e.rule_score or 0.0) >= 0.75
                    and t.executed
                    and t.net_r > 0
                ):
                    d = _describe(df, atr, e, probs)
                    d["net_r"] = round(float(t.net_r), 3)
                    # The gate responsible is recomputed at the SHIPPED default
                    # (30), not 40: the cost side must be attributed against
                    # what actually ships, or the breakdown describes a gate
                    # nobody runs.
                    d["reason_dropped_by"] = ", ".join(
                        r for r, ok in (
                            ("2.2_max_len_30", _gate_max_len(e, 30)),
                            ("2.3_rule_floor_0.6", _gate_rule(e, 0.6)),
                            ("2.1_trend", _trend_ok(closes, atr, e,
                                                    TREND_LOOKBACK, TREND_MIN_R2,
                                                    TREND_MIN_SLOPE_ATR)),
                        ) if ok is not True
                    )
                    good.append(d)
            good.sort(key=lambda x: -x["net_r"])
            # Aggregate BEFORE truncating: the "which gate caused it" breakdown
            # is the decision-relevant quantity, and a capped 12-row sample
            # cannot yield it.
            by_gate: dict[str, int] = {}
            for d in good:
                by_gate[d["reason_dropped_by"]] = (
                    by_gate.get(d["reason_dropped_by"], 0) + 1
                )
            cum["false_drops_by_gate"] = dict(
                sorted(by_gate.items(), key=lambda kv: -kv[1])
            )
            cum["false_drops_total_R"] = round(
                float(sum(d["net_r"] for d in good)), 2
            )
            cum["false_drops_mean_R"] = (
                round(float(np.mean([d["net_r"] for d in good])), 3) if good else None
            )
            cum["false_drop_rate_of_dropped"] = round(
                len(good) / max(len(dropped), 1), 4
            )
            cum["false_drop_candidates"] = good[:12]
            cum["n_false_drop_candidates"] = len(good)
        pat["false_drops"]["cum_all"] = cum

        report["patterns"][pattern] = pat
        _log(f"  cum_all: kept {len(kept_all)} / {len(events)}")

    # ---- §2.1 option comparison: B (slope) vs C (HMM trend) -------------
    # B is measured by the 2.1_* arms above.  C lives in
    # research/regime/trend_hmm.py and is measured by
    # docs/trend_hmm_option_c.py (its own script because fitting the HMM must
    # honour the causal fit-split convention); pull its JSON in here so this
    # report is a single source of truth rather than two half-results.
    c_json = Path(__file__).resolve().parent / "trend_hmm_option_c.json"
    c_data: Any = None
    if c_json.exists():
        try:
            c_data = json.loads(c_json.read_text())
        except Exception as exc:  # pragma: no cover - diagnostic
            c_data = {"status": "pending", "reason": f"unreadable: {exc}"}
    report["trend_options"] = {
        "phuong_an_B_slope": {
            "status": "measured",
            "module": "research/core/trend_context.py",
            "note": "regression slope + R^2 gate; see the 2.1_* arms above.",
        },
        "phuong_an_C_hmm_trend": (
            c_data if c_data is not None else {
                "status": "pending",
                "reason": (
                    "docs/trend_hmm_option_c.json absent — Phuong an C not "
                    "measured.  Reported as pending rather than invented."
                ),
            }
        ),
    }

    out = Path(__file__).resolve().parent / "rework_trend_hmm_measure.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    _log(f"\nwrote {out}")

    # ---- console summary -----------------------------------------------
    for pattern, pat in report["patterns"].items():
        _log(f"\n--- {pattern} (pool {pat['n_pool']}, "
             f"model_prob {'ok' if pat['model_prob_available'] else 'NOT AVAILABLE'}) ---")
        for name in ("baseline", "cum_2.1+2.2", "cum_all_2.1+2.2+2.3",
                     "cum_2.2+2.3"):
            if name in pat["arms"]:
                s = pat["arms"][name]
                _log(f"  {name:26s} n={s['n_events']:4d} ({s['kept_pct']:5}%) "
                     f"win={s['winrate']} exp={s['expectancy_R']} "
                     f"tgtATR_med={s['target_atr_median']}")
        for arm in ("cum_2.1+2.2", "cum_all_2.1+2.2+2.3"):
            o = pat["oos"][arm]
            _log(f"  OOS {arm:20s} total={o['n_total']:4d} "
                 f"fixed={o['fixed_split_n']:4d} "
                 f"{'PASS' if o['fixed_split_pass'] else 'FAIL'} | "
                 f"last40%={o['train_split_n']:4d} "
                 f"{'PASS' if o['train_split_pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
