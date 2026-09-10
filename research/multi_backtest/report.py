"""
report.py — Multi-Pattern Backtest Report Generator (SPEC v1.1 §12)

Produces the mandatory 5-item report for one symbol:

    1. per-pattern: trades, PF, expectancy, maxDD, PR-AUC, rolling PF;
    2. portfolio equity (dedup/confluence + §4.4 caps applied);
    3. correlation matrix (pairwise % time in the same correlation group);
    4. confluence buckets: expectancy per number of agreeing patterns;
    5. multi-pattern portfolio vs best single pattern on the OOS window.

Rendered as human-readable Markdown + machine-readable JSON, committed under
``research/multi_backtest/reports/`` for ≥2 symbols.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from research.multi_backtest.runner import (
    BacktestTrade,
    PortfolioResult,
)


def _pf(net_r: list[float]) -> float:
    """Profit factor = sum(wins) / |sum(losses)| (guards divide-by-zero)."""
    wins = sum(r for r in net_r if r > 0)
    losses = abs(sum(r for r in net_r if r < 0))
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _max_drawdown(cum: list[float]) -> float:
    """Max drawdown in R units from the running equity peak."""
    if not cum:
        return 0.0
    peak = cum[0]
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return abs(mdd)


def _rolling_pf(net_r: list[float], window: int = 20) -> list[dict[str, float]]:
    """PF over a sliding window of the last ``window`` trades (≥ 10 kept)."""
    out: list[dict[str, float]] = []
    for i in range(len(net_r)):
        lo = max(0, i - window + 1)
        chunk = net_r[lo : i + 1]
        if len(chunk) >= min(10, window):
            out.append({"trades": float(i + 1), "pf": _pf(chunk)})
    # keep ≤ 12 points for a compact report
    if len(out) > 12:
        idx = np.linspace(0, len(out) - 1, 12).round().astype(int)
        out = [out[int(i)] for i in idx]
    return out


def _series_from(rows: list[dict[str, float]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if key in r]


def _ema_filter(values: list[float], alpha: float = 0.5) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


# ---------------------------------------------------------------------------
# Item computations
# ---------------------------------------------------------------------------

def _pr_auc(trades: list[BacktestTrade]) -> float | None:
    """PR-AUC of the pattern's score (model_prob else rule_score) vs win.

    Win = gross outcome positive (target hit or profitable horizon close).
    ``None`` when a single-class outcome makes the metric undefined.
    """
    scores: list[float] = []
    labels: list[int] = []
    for t in trades:
        score = t.model_prob if t.model_prob is not None else t.rule_score / 100.0
        if score is None or not math.isfinite(float(score)):
            continue
        label = 1 if t.gross_r > 0 else 0
        scores.append(float(score))
        labels.append(label)
    if len(scores) < 5:
        return None
    n_pos = sum(labels)
    if n_pos == 0 or n_pos == len(labels):
        return None
    return float(average_precision_score(np.asarray(labels), np.asarray(scores)))


def _equity_curve(trades: list[BacktestTrade]) -> list[dict[str, Any]]:
    """Cumulative net R over executed trades, in entry-time order."""
    curve: list[dict[str, Any]] = []
    cum = 0.0
    for t in sorted(trades, key=lambda x: (x.entry_time, x.event_id)):
        cum += t.net_r * t.risk_fraction
        curve.append({"time": str(t.entry_time), "net_r": round(cum, 4)})
    return curve


def _correlation_matrix(
    result: PortfolioResult,
    events: list[Any] | None = None,
    groups: list[Any] | None = None,
) -> dict[str, Any]:
    """§12 item 3 — pairwise % of pattern i's groups that also contain j.

    ``corr[i][j] = |groups containing both i and j| / |groups containing i|``.
    Symmetrised (averaged with the transpose) and diagonal = 1.0.  ``events``
    / ``groups`` default to the (window-filtered) report lists.
    """
    events = events if events is not None else result.events
    groups = groups if groups is not None else result.groups
    patterns = sorted({e.pattern_name for e in events})
    if not patterns:
        return {"patterns": [], "matrix": []}
    # group -> set of patterns present
    group_patterns: list[set[str]] = []
    for g in groups:
        gp = {e.pattern_name for e in g.events}
        if gp:
            group_patterns.append(gp)
    n = len(patterns)
    mat = np.zeros((n, n))
    for i, pi in enumerate(patterns):
        gi = [gp for gp in group_patterns if pi in gp]
        denom = max(1, len(gi))
        for j, pj in enumerate(patterns):
            if i == j:
                mat[i, j] = 1.0
                continue
            both = sum(1 for gp in gi if pj in gp)
            mat[i, j] = both / denom
    # symmetrise: average with transpose; keep diagonal 1
    sym = (mat + mat.T) / 2.0
    for i in range(n):
        sym[i, i] = 1.0
    return {
        "patterns": patterns,
        "matrix": [[round(float(v), 4) for v in row] for row in sym],
    }


def _confluence_buckets(
    result: PortfolioResult,
    groups: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """§12 item 4 — expectancy (mean net R) per number of agreeing patterns.

    Uses the representative trade's simulated outcome of every group
    (cap-rejected trades excluded from counts but their pattern-agreement is
    still the same idea).
    """
    groups = groups if groups is not None else result.groups
    buckets: dict[int, list[float]] = {}
    counts: dict[int, int] = {}
    for g in groups:
        if g.representative is None:
            continue
        # find the representative's trade outcome by event_id
        trade = _trade_for(result, g.representative.event_id)
        n = max(1, g.n_patterns)
        counts[n] = counts.get(n, 0) + 1
        if trade is not None and trade.executed:
            buckets.setdefault(n, []).append(trade.net_r)
    out: list[dict[str, Any]] = []
    for n in sorted(counts):
        net = buckets.get(n, [])
        out.append(
            {
                "n_patterns": int(n),
                "groups": counts[n],
                "trades": len(net),
                "expectancy_r": round(float(np.mean(net)), 4) if net else None,
                "pf": round(_pf(net), 4) if net else None,
            }
        )
    return out


def _trade_for(result: PortfolioResult, event_id: str) -> BacktestTrade | None:
    for t in result.trades:
        if t.event_id == event_id:
            return t
    for t in result.rejected:
        if t.event_id == event_id:
            return t
    return None


def _oos_window(
    result: PortfolioResult, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Last 30 % of the report window = the OOS block (chronological)."""
    span = (end - start).total_seconds()
    oos_start = end - pd.Timedelta(seconds=span * 0.30)
    return oos_start, end


def _in_window(t: BacktestTrade, lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
    return lo <= t.entry_time <= hi


def _as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Normalise a report-window boundary to UTC (event timestamps are
    tz-aware; naive boundaries would raise on comparison)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def generate_symbol_report(
    result: PortfolioResult,
    df: pd.DataFrame,
    out_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[Path, Path]:
    """Build the §12 5-item report for one symbol; write MD + JSON."""
    start = _as_utc(start)
    end = _as_utc(end)

    # ---- window filter: only events/groups with known_at in [start, end] ----
    in_events = [
        e for e in result.events
        if start <= pd.Timestamp(e.known_at_ts) <= end
    ]
    in_groups = [
        g for g in result.groups
        if g.representative is not None
        and start <= pd.Timestamp(g.representative.known_at_ts) <= end
    ]
    trades = [t for t in result.trades if _in_window(t, start, end)]
    rejected = [t for t in result.rejected if _in_window(t, start, end)]
    executed = [t for t in trades if t.executed]

    patterns = sorted({e.pattern_name for e in in_events})

    # ---- item 1: per-pattern stats -----------------------------------------
    per_pattern: list[dict[str, Any]] = []
    for p in patterns:
        pt = [t for t in executed if t.pattern_name == p]
        net = [t.net_r for t in pt]
        cum = list(np.cumsum(net)) if net else []

        # OOS block for PR-AUC — last 30 % of the *pattern's* trades
        oos_trades: list[BacktestTrade] = []
        if len(pt) >= 5:
            split = max(1, int(len(pt) * 0.7))
            oos_trades = pt[split:]

        per_pattern.append(
            {
                "pattern": p,
                "events": sum(1 for e in in_events if e.pattern_name == p),
                "groups": sum(
                    1 for g in in_groups
                    if g.representative is not None
                    and g.representative.pattern_name == p
                ),
                "trades": len(pt),
                "pf": round(_pf(net), 4) if net else None,
                "expectancy_r": round(float(np.mean(net)), 4) if net else None,
                "max_dd_r": round(_max_drawdown(cum), 4) if cum else None,
                "pr_auc": _pr_auc(oos_trades),
                "rolling_pf": _rolling_pf(net),
                "total_net_r": round(float(np.sum(net)), 4) if net else 0.0,
            }
        )

    # ---- item 2: portfolio equity (dedup/confluence + caps) -----------------
    equity = _equity_curve(executed)
    equity_net = [float(pt["net_r"]) for pt in equity]
    portfolio = {
        "trades": len(executed),
        "total_net_r": round(equity_net[-1], 4) if equity_net else 0.0,
        "pf": round(_pf([t.net_r for t in executed]), 4) if executed else None,
        "expectancy_r": (
            round(float(np.mean([t.net_r for t in executed])), 4) if executed else None
        ),
        "max_dd_r": round(_max_drawdown(equity_net), 4) if equity_net else None,
        "rejected_by_caps": len(rejected),
        "equity": equity,
    }

    # ---- item 3: correlation matrix ----------------------------------------
    corr = _correlation_matrix(result, events=in_events, groups=in_groups)

    # ---- item 4: confluence buckets ----------------------------------------
    confluence = _confluence_buckets(result, groups=in_groups)

    # ---- item 5: multi-pattern portfolio vs best single pattern (OOS) ------
    oos_lo, oos_hi = _oos_window(result, start, end)
    port_oos = [t for t in executed if _in_window(t, oos_lo, oos_hi)]
    port_oos_net = float(np.sum([t.net_r for t in port_oos])) if port_oos else 0.0

    single_oos: dict[str, float] = {}
    for p in patterns:
        pt = [t for t in executed if t.pattern_name == p and _in_window(t, oos_lo, oos_hi)]
        single_oos[p] = round(float(np.sum([t.net_r for t in pt])), 4) if pt else 0.0
    best_pattern = max(single_oos, key=lambda k: float(single_oos[k])) if single_oos else ""
    best_single_net = single_oos.get(best_pattern, 0.0)
    multi_wins = port_oos_net > best_single_net

    comparison = {
        "oos_start": str(oos_lo),
        "oos_end": str(oos_hi),
        "portfolio_net_r": round(port_oos_net, 4),
        "portfolio_trades": len(port_oos),
        "single_pattern_net_r": {p: v for p, v in single_oos.items()},
        "best_single_pattern": best_pattern,
        "best_single_net_r": round(best_single_net, 4),
        "multi_beats_best_single": bool(multi_wins),
        "verdict": (
            "multi-pattern portfolio adds value over best single pattern on OOS"
            if multi_wins
            else "REVIEW NEEDED: multi-pattern portfolio does not beat best single pattern on OOS"
        ),
    }

    report: dict[str, Any] = {
        "schema": "multi_backtest_report_v1",
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "mode": result.mode,
        "window": {"start": str(start), "end": str(end)},
        "costs": result.costs.to_dict(),
        "correlation_config": {
            "corr_time_window": result.corr_config.corr_time_window,
            "corr_price_window_atr": result.corr_config.corr_price_window_atr,
            "bar_seconds": result.corr_config.bar_seconds,
            "max_total_risk_per_symbol": result.corr_config.max_total_risk_per_symbol,
            "max_direction_cluster_risk": result.corr_config.max_direction_cluster_risk,
        },
        "patterns": patterns,
        "per_pattern": per_pattern,          # item 1
        "portfolio": portfolio,              # item 2
        "correlation_matrix": corr,          # item 3
        "confluence_buckets": confluence,    # item 4
        "oos_comparison": comparison,        # item 5
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result.symbol}_multi_backtest.json"
    md_path = out_dir / f"{result.symbol}_multi_backtest.md"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return md_path, json_path


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _render_markdown(r: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Multi-Pattern Backtest Report — {r['symbol']} ({r['timeframe']})")
    lines.append("")
    lines.append(f"- Mode: `{r['mode']}` | Window: `{r['window']['start']}` → `{r['window']['end']}`")
    lines.append(f"- Costs (bps, per side): spread {r['costs']['spread_bps']} / "
                 f"commission {r['costs']['commission_bps']} / slippage {r['costs']['slippage_bps']}")
    lines.append("")

    lines.append("## 1. Per-pattern statistics")
    lines.append("")
    lines.append("| pattern | events | groups | trades | PF | expectancy R | maxDD R | PR-AUC | total R |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in r["per_pattern"]:
        lines.extend(
            [
                f"| {p['pattern']} | {p['events']} | {p['groups']} | {p['trades']} "
                f"| {_fmt(p['pf'])} | {_fmt(p['expectancy_r'])} | {_fmt(p['max_dd_r'])} "
                f"| {_fmt(p['pr_auc'])} | {_fmt(p['total_net_r'])} |"
            ]
        )
    lines.append("")
    for p in r["per_pattern"]:
        if p["rolling_pf"]:
            pts = ", ".join(f"{int(x['trades'])}t:{x['pf']:.2f}" for x in p["rolling_pf"])
            lines.append(f"- **{p['pattern']}** rolling PF: {pts}")
    lines.append("")

    lines.append("## 2. Portfolio equity (dedup/confluence + caps)")
    lines.append("")
    port = r["portfolio"]
    lines.append(
        f"- Trades: **{port['trades']}** | total net R: **{_fmt(port['total_net_r'])}** | "
        f"PF: **{_fmt(port['pf'])}** | expectancy: **{_fmt(port['expectancy_r'])}** R | "
        f"maxDD: **{_fmt(port['max_dd_r'])}** R | rejected by caps: **{port['rejected_by_caps']}**"
    )
    lines.append("")
    eq = port["equity"]
    if eq:
        step = max(1, len(eq) // 12)
        sampled = eq[::step][:12]
        lines.append("| trade # | time | cumulative net R |")
        lines.append("|---|---|---|")
        for i, pt in enumerate(sampled):
            lines.append(f"| {i * step + 1} | {pt['time']} | {_fmt(pt['net_r'])} |")
    lines.append("")

    lines.append("## 3. Correlation matrix — % time in the same group")
    lines.append("")
    cm = r["correlation_matrix"]
    pats = cm["patterns"]
    lines.append("| pattern | " + " | ".join(pats) + " |")
    lines.append("|" + "---|" * (len(pats) + 1))
    for i, row in enumerate(cm["matrix"]):
        lines.append(f"| {pats[i]} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    lines.append("")

    lines.append("## 4. Confluence buckets — expectancy by number of agreeing patterns")
    lines.append("")
    lines.append("| n patterns | groups | trades | expectancy R | PF |")
    lines.append("|---|---|---|---|---|")
    for b in r["confluence_buckets"]:
        lines.extend(
            [
                f"| {b['n_patterns']} | {b['groups']} | {b['trades']} "
                f"| {_fmt(b['expectancy_r'])} | {_fmt(b['pf'])} |"
            ]
        )
    lines.append("")

    lines.append("## 5. Multi-pattern portfolio vs best single pattern (OOS)")
    lines.append("")
    o = r["oos_comparison"]
    lines.append(f"- OOS window: `{o['oos_start']}` → `{o['oos_end']}`")
    lines.append(f"- Portfolio net R (OOS): **{_fmt(o['portfolio_net_r'])}** "
                 f"({o['portfolio_trades']} trades)")
    lines.append(f"- Best single pattern: **{o['best_single_pattern']}** "
                 f"= {_fmt(o['best_single_net_r'])} R")
    lines.append(f"- **Verdict:** {o['verdict']}")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by `research/multi_backtest/runner.py` (§12) — "
                 "backtest ≡ live CorrelationManager code path._")
    return "\n".join(lines) + "\n"


__all__ = ["PortfolioResult", "generate_symbol_report"]