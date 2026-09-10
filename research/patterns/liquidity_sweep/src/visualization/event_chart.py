# mypy: ignore-errors
# src, so visualization code is enforced via ruff + pytest instead.  All plot
# data flows through pandas/numpy values already type-checked upstream.

"""Event chart rendering + human-review round 1 tooling (guide section 24).

Scope (Phase 3 / task t7): the rolling-levels baseline events.  For each
reviewed event a chart shows 30-50 candles before and 16-32 candles after the
sweep, with the liquidity level, the sweep candle, entry, stop, target and the
trade outcome — exported into ``winners/`` (target hit), ``losers/`` (stop
hit), ``time/`` (time barrier) and ``ambiguous/`` (TP+SL same candle)
sub-folders.

Review sampling (guide 24): 100-200 events, deterministic (``seed``),
stratified by primary outcome group (2R / h16 by default) and balanced across
long/short so bullish and bearish sweeps are always represented; events must
have enough history before (``min_pre``) and data after (``min_post``) the
sweep for a complete chart.

Automatic first-pass review (this module is Agent 5's *screening*; the human
sign-off round runs on the confirmation-based events in round 2 / task t12
coordinated with Agent 7): each event gets ``verdict`` + ``reviewer_score``
from documented heuristics on sweep quality:

* ``correct``  — textbook: wick_ratio >= 0.50 and 0.15 <= penetration_atr;
* ``incorrect`` — marginal: wick_ratio < 0.40 or penetration_atr < 0.10
  (spread-noise / barely-visible pattern candidates);
* ``ambiguous`` — needs human eyes (TP+SL same candle, or between thresholds).

The full per-event row keeps the canonical review columns
(``docs/SCHEMAS.md`` §10: review_id/event_id/reviewer/verdict/notes/
review_version/reviewed_at) plus the guide §5.6 screening fields
(detector_correct, liquidity_level_clear, sweep_clear, confirmation_clear,
reviewer_score, reviewer_comment).  The CSV is written to
``reports/manual_review.csv`` and a summary to
``reports/manual_review_summary.md``.

Display-only rule: level lines, rolling liq_low/liq_high context and the
event's own exit marker may extend past the decision bar — charts annotate
past data; every *decision* column used downstream (entry/stop/outcome) is
already causal (see src/labeling/outcome_builder, QA finding F1).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

# Workspace-local matplotlib config/font cache: the sandbox blocks ~/.matplotlib
# and a fresh temp dir per process would rebuild the font cache every run.
_MPL_CONFIG_DIR = os.environ.get("MPLCONFIGDIR") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".mplconfigdir")
)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless-safe: render to PNG, never open a window
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# ---------------------------------------------------------------------------
# Grouping / constants
# ---------------------------------------------------------------------------

#: outcome token -> chart sub-folder (guide 5.6 winner/loser/ambiguous groups,
#: plus the explicit time-barrier group).
OUTCOME_GROUP_DIRS: Mapping[str, str] = {
    "tp": "winners",
    "sl": "losers",
    "time": "time",
    "ambiguous": "ambiguous",
}

REVIEW_VERSION = 1
REVIEWER = "agent_5"

#: Screening thresholds (documented heuristic, see module docstring).
SWEEP_CLEAN_WICK = 0.50
SWEEP_CLEAN_PEN = 0.15
SWEEP_MARGINAL_WICK = 0.40
SWEEP_MARGINAL_PEN = 0.10

GUIDE_REVIEW_FIELDS = [
    "detector_correct",
    "liquidity_level_clear",
    "sweep_clear",
    "confirmation_clear",
    "reviewer_score",
    "reviewer_comment",
]

REVIEW_CSV_COLUMNS = [
    "review_id",
    "event_id",
    "reviewer",
    "verdict",
    "notes",
    "review_version",
    "reviewed_at",
    *GUIDE_REVIEW_FIELDS,
]


# ---------------------------------------------------------------------------
# Sampling (guide 24: 100-200 diverse events, deterministic)
# ---------------------------------------------------------------------------

def sample_review_events(
    candles: pd.DataFrame,
    labeled: pd.DataFrame,
    n: int = 160,
    seed: int = 42,
    min_pre: int = 40,
    min_post: int = 24,
    outcome_col: str = "outcome_2r_h16",
    group_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Deterministic stratified sample of events for round-1 review.

    Stratification: primary outcome group (token from ``outcome_col``) with
    ``group_weights`` for tp/sl/time; every available ``ambiguous`` event is
    included up to 25% of ``n``.  Within each group the draw alternates
    direction quotas so both bullish and bearish sweeps appear.  Only events
    with ``>= min_pre`` candles before and ``> min_post`` candles after the
    sweep (complete chart windows) are eligible.  Returns the sampled rows,
    chronologically sorted, with ``event_id`` unique.
    """
    if not 100 <= n <= 500:
        raise ValueError(f"sample_review_events: n must be in [100, 500], got {n}")
    if outcome_col not in labeled.columns:
        raise ValueError(
            f"sample_review_events: labeled missing outcome column {outcome_col!r}"
        )
    weights: Mapping[str, float] = group_weights or {
        "tp": 0.40,
        "sl": 0.45,
        "time": 0.15,
    }

    rng = np.random.default_rng(seed)
    positions = _event_positions(candles, labeled["event_time"])
    eligible = labeled[
        (positions >= min_pre) & (positions + min_post < len(candles))
    ].copy()
    if eligible.empty:
        raise ValueError(
            "sample_review_events: no events satisfy the chart-window "
            f"requirements (min_pre={min_pre}, min_post={min_post})"
        )

    unknown = set(eligible[outcome_col]) - set(OUTCOME_GROUP_DIRS)
    if unknown:
        raise ValueError(
            f"sample_review_events: unexpected outcome tokens {sorted(unknown)} "
            f"for outcome_col {outcome_col!r}"
        )

    groups = {
        g: eligible[eligible[outcome_col] == g].copy() for g in OUTCOME_GROUP_DIRS
    }
    amb_quota = min(len(groups["ambiguous"]), max(1, round(0.25 * n)))
    remainder = n - amb_quota
    quotas: dict[str, int] = {"ambiguous": amb_quota}
    for group, weight in weights.items():
        pool = groups[group]
        quotas[group] = (
            0 if pool.empty else min(len(pool), max(1, round(remainder * weight)))
        )
    # Hand out any leftover capacity to the largest remaining pools.
    capacity = {g: len(groups[g]) - quotas[g] for g in OUTCOME_GROUP_DIRS}
    ordered = sorted(weights, key=lambda g: -weights[g])
    for _ in range(10):  # bounded fill loop
        if sum(quotas.values()) >= n:
            break
        g = next((g for g in ordered if capacity[g] > 0), None)
        if g is None:
            break
        quotas[g] += 1
        capacity[g] -= 1

    # Direction-balanced draw inside every group.
    picked_idx: list[int] = []
    for group in OUTCOME_GROUP_DIRS:
        quota = int(quotas[group])
        if quota <= 0:
            continue
        pool = groups[group]
        order = np.argsort(rng.random(len(pool)))
        shuffled = pool.iloc[order]
        dirs = sorted(set(shuffled["direction"]))
        per_dir = {d: shuffled[shuffled["direction"] == d] for d in dirs}
        taken = 0
        cursor = 0
        while taken < quota and any(len(sub) > 0 for sub in per_dir.values()):
            d = dirs[cursor % len(dirs)]
            cursor += 1
            sub = per_dir[d]
            if sub.empty:
                continue
            picked_idx.append(int(sub.index[0]))
            per_dir[d] = sub.iloc[1:]
            taken += 1

    out = eligible.loc[picked_idx] if picked_idx else eligible.iloc[0:0]
    out = out.sort_values("event_time", kind="stable").reset_index(drop=True)
    if len(out) < 100:
        raise ValueError(
            f"sample_review_events: only {len(out)} eligible events (< 100); "
            "relax min_pre/min_post or reduce n"
        )
    return out


def _event_positions(candles: pd.DataFrame, event_times: pd.Series) -> np.ndarray:
    positions = candles.index.get_indexer(pd.DatetimeIndex(event_times))
    if int((positions < 0).sum()):
        raise ValueError(
            "some event_time values do not exist in the candle index; candles "
            "and events must come from the same frame"
        )
    return positions


# ---------------------------------------------------------------------------
# First-pass screening (documented heuristics)
# ---------------------------------------------------------------------------

def assess_event(row: Mapping[str, Any]) -> dict[str, Any]:
    """Provisional round-1 review of one merged event row.

    ``row`` must carry the labeled columns (event_id, direction, level_id,
    entry/stop/target/outcome...) plus sweep metrics from the event table
    (penetration_atr, wick_ratio, reclaim_atr).  Thresholds are module
    constants; see the module docstring.
    """
    outcome = str(row.get("outcome_2r_h16", ""))
    exit_reason = str(row.get("exit_reason", ""))
    wick = _safe_float(row.get("wick_ratio")) or 0.0
    pen = _safe_float(row.get("penetration_atr")) or 0.0
    has_metrics = _safe_float(row.get("wick_ratio")) is not None

    if outcome == "ambiguous" or exit_reason == "ambiguous":
        verdict = "ambiguous"
        sweep_clear = "maybe"
        comment = "TP and SL inside the same candle: intrabar order unknown; human review required"
    elif not has_metrics:
        verdict = "ambiguous"
        sweep_clear = "maybe"
        comment = "sweep metrics missing from event row; human review required"
    elif wick >= SWEEP_CLEAN_WICK and pen >= SWEEP_CLEAN_PEN:
        verdict = "correct"
        sweep_clear = "yes"
        comment = "textbook sweep: clean directional wick with clear penetration"
    elif wick < SWEEP_MARGINAL_WICK or pen < SWEEP_MARGINAL_PEN:
        verdict = "incorrect"
        sweep_clear = "no"
        comment = "marginal pattern: tiny wick/penetration candidate for spread noise"
    else:
        verdict = "ambiguous"
        sweep_clear = "maybe"
        comment = "between clean and marginal thresholds; human review required"

    score = round(
        50 * min(max(wick, 0.0), 1.0) + 40 * min(max(pen, 0.0), 0.5) / 0.5
    )
    notes = (
        f"primary_outcome={outcome}; exit={exit_reason}; "
        f"wick_ratio={row.get('wick_ratio')}; penetration_atr={row.get('penetration_atr')}; "
        f"reclaim_atr={row.get('reclaim_atr')}; level_id={row.get('level_id')}; "
        f"bars_held={row.get('bars_held')}; net_result_r={row.get('net_result_r')}; "
        f"review={comment}"
    )
    return {
        "review_id": f"R1-{row.get('event_id')}",
        "event_id": row.get("event_id"),
        "reviewer": REVIEWER,
        "verdict": verdict,
        "notes": notes,
        "review_version": REVIEW_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # guide 5.6 screening fields
        "detector_correct": {
            "correct": "yes",
            "incorrect": "no",
            "ambiguous": "maybe",
        }[verdict],
        "liquidity_level_clear": "yes",  # trailing rolling level (round 1)
        "sweep_clear": sweep_clear,
        "confirmation_clear": "n/a",  # confirmation detector lands in Phase 5
        "reviewer_score": int(max(0, min(100, score))),
        "reviewer_comment": comment,
    }


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def _draw_candles(ax: Any, o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> None:
    """Lightweight candlesticks (no mplfinance dependency)."""
    up = c >= o
    colors = np.where(up, "#26a69a", "#ef5350")
    for i in range(len(o)):
        ax.plot([i, i], [lo[i], h[i]], color=colors[i], lw=0.8, zorder=2)
    for i in range(len(o)):
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        if body_hi - body_lo <= 0:  # doji
            ax.plot([i - 0.25, i + 0.25], [o[i], o[i]], color=colors[i], lw=1.2, zorder=3)
        else:
            ax.add_patch(
                mpatches.Rectangle(
                    (i - 0.32, body_lo),
                    0.64,
                    body_hi - body_lo,
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    zorder=3,
                )
            )
    ax.set_xlim(-0.8, len(o) - 0.2)


def _sync_ylim(ax: Any) -> float:
    """Autoscale after patches/lines and return the top of the y-range."""
    ax.relim()
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    margin = 0.05 * (hi - lo)
    ax.set_ylim(lo - margin, hi + margin)
    return float(hi + margin)


def plot_event_chart(
    candles: pd.DataFrame,
    row: pd.Series,
    *,
    pre_bars: int = 40,
    post_bars: int = 24,
    target_col: str = "target_2r",
    outcome_col: str = "outcome_2r_h16",
) -> Figure:
    """Render one event chart; returns the Figure (caller saves it).

    Window: ``pre_bars`` candles before the sweep .. ``post_bars`` after (both
    clamped to data bounds; extended to include the exit bar).  Displays
    candles, rolling liq_low/liq_high context (thin gray), the swept level
    (dashed), sweep bar highlight, entry/stop/target lines, and the exit
    marker/outcome annotation.
    """
    if not 30 <= pre_bars <= 50:
        raise ValueError(f"plot_event_chart: pre_bars must be in [30, 50], got {pre_bars}")
    if not 16 <= post_bars <= 32:
        raise ValueError(f"plot_event_chart: post_bars must be in [16, 32], got {post_bars}")

    pos = int(candles.index.get_loc(pd.Timestamp(row["event_time"])))
    start = max(0, pos - pre_bars)
    end = min(len(candles), pos + 1 + post_bars)
    exit_time = row.get("exit_time")
    if pd.notna(exit_time):
        try:
            end = max(end, int(candles.index.get_loc(pd.Timestamp(exit_time))) + 1)
        except KeyError:
            pass
    win = candles.iloc[start:end]

    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    lo = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    _draw_candles(ax, o, h, lo, c)

    # rolling liquidity context (display only)
    for col in ("liq_low", "liq_high"):
        if col in win.columns:
            ax.plot(
                np.arange(len(win)), win[col].to_numpy(dtype=float),
                color="#90a4ae", lw=0.8, alpha=0.55, ls=":", zorder=1,
                label="rolling level" if col == "liq_low" else None,
            )

    sweep_i = pos - start
    ax.axvspan(sweep_i - 0.5, sweep_i + 0.5, color="#fff59d", alpha=0.45, zorder=0)

    level_price = _safe_float(row.get("level_price"))
    entry = _safe_float(row.get("entry_price"))
    stop = _safe_float(row.get("stop_price"))
    target = _safe_float(row.get(target_col))
    exit_price = _safe_float(row.get("exit_price"))

    if level_price is not None:
        ax.axhline(level_price, color="#8e24aa", ls="--", lw=1.1, zorder=1,
                   label=f"level {level_price:.2f}")
    if entry is not None:
        ax.axhline(entry, color="#1565c0", ls="--", lw=1.1, zorder=1,
                   label=f"entry {entry:.2f}")
    if stop is not None:
        ax.axhline(stop, color="#c62828", ls="-.", lw=1.1, zorder=1,
                   label=f"stop {stop:.2f}")
    if target is not None:
        ax.axhline(target, color="#2e7d32", ls="-.", lw=1.1, zorder=1,
                   label=f"target {target:.2f}")

    top = _sync_ylim(ax)
    ax.text(sweep_i, top, "sweep", ha="center", va="bottom", fontsize=8,
            color="#5d4037")

    outcome = str(row.get(outcome_col, ""))
    marker_color = {
        "tp": "#2e7d32", "sl": "#c62828", "time": "#1565c0", "ambiguous": "#6d4c41",
    }.get(outcome, "#37474f")
    exit_reason = str(row.get("exit_reason", ""))
    if exit_price is not None and pd.notna(row.get("exit_time")):
        try:
            xexit = int(candles.index.get_loc(pd.Timestamp(row["exit_time"]))) - start
            ax.plot(xexit, exit_price, marker="D", ms=9, color=marker_color, zorder=5,
                    label=f"exit {exit_reason}")
        except KeyError:
            pass
    elif outcome == "ambiguous":
        ax.text(sweep_i, top, "ambiguous (TP&SL same bar)", fontsize=9,
                color="#6d4c41", ha="center", va="bottom")

    # confirmation candle marker (round-2 rows; absent in round 1)
    conf_time = row.get("confirmation_time")
    if conf_time is not None and pd.notna(conf_time):
        try:
            xconf = int(candles.index.get_loc(pd.Timestamp(conf_time))) - start
            ax.axvline(xconf, color="#00695c", lw=1.4, alpha=0.9, ls="--")
            ax.text(xconf, ax.get_ylim()[0], "conf", fontsize=7,
                    color="#00695c", ha="center", va="bottom", rotation=90)
        except KeyError:
            pass

    # time axis ticks
    n = len(win)
    ticks = list(range(0, n, max(1, n // 6)))[:7]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(win.index[t]).strftime("%m-%d %H:%M") for t in ticks],
        rotation=0, fontsize=8,
    )
    ax.axvline(sweep_i, color="#5d4037", lw=0.9, alpha=0.6, ls=":")
    ax.grid(alpha=0.25, ls=":")
    ax.set_ylabel("price")

    net = row.get("net_result_r")
    net_s = "n/a" if net is None or (isinstance(net, float) and np.isnan(net)) else f"{float(net):.2f}R"
    lvl = row.get("level_type")
    lvl_s = f"{lvl}/" if lvl is not None else ""
    title = (
        f"{row.get('event_id')} | {row.get('direction')} | {lvl_s}{row.get('level_id')} | "
        f"{outcome_col}={outcome} | exit={exit_reason} | bars={row.get('bars_held')} | "
        f"net={net_s}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Round-1 orchestration
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def render_event_charts(
    candles: pd.DataFrame,
    labeled: pd.DataFrame,
    out_dir: str,
    *,
    n: int = 160,
    seed: int = 42,
    pre_bars: int = 40,
    post_bars: int = 24,
    dpi: int = 120,
    outcome_col: str = "outcome_2r_h16",
    target_col: str = "target_2r",
) -> tuple[pd.DataFrame, list[str]]:
    """Sample, render, and save round-1 event charts grouped by outcome.

    Charts land in ``<out_dir>/event_charts/<group>/<event_id>.png`` where
    ``<group>`` ∈ winners/losers/time/ambiguous.  Returns the sampled frame
    and the list of written chart paths.
    """
    sample = sample_review_events(
        candles, labeled, n=n, seed=seed,
        min_pre=pre_bars, min_post=post_bars, outcome_col=outcome_col,
    )
    base = _ensure_dir(f"{out_dir.rstrip('/')}/event_charts")
    paths = [
        _save_event_chart(
            candles, row, base, outcome_col=outcome_col, target_col=target_col,
            pre_bars=pre_bars, post_bars=post_bars, dpi=dpi,
        )
        for _, row in sample.iterrows()
    ]
    return sample, paths


def _save_event_chart(
    candles: pd.DataFrame,
    row: pd.Series,
    base: str,
    *,
    outcome_col: str,
    target_col: str,
    pre_bars: int,
    post_bars: int,
    dpi: int,
) -> str:
    """Render one event chart to ``<base>/<group>/<event_id>.png``."""
    group = OUTCOME_GROUP_DIRS[str(row[outcome_col])]
    folder = _ensure_dir(f"{base}/{group}")
    fig = plot_event_chart(
        candles, row, pre_bars=pre_bars, post_bars=post_bars,
        target_col=target_col, outcome_col=outcome_col,
    )
    path = f"{folder}/{row['event_id']}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_manual_review_csv(review_rows: Iterable[Mapping[str, Any]], path: str) -> str:
    """Write the round-1 review CSV (schema §10 + guide §5.6 fields)."""
    df = pd.DataFrame(review_rows, columns=REVIEW_CSV_COLUMNS)
    df = df.sort_values(["reviewed_at", "event_id"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def write_review_summary(review_df: pd.DataFrame, path: str) -> dict[str, Any]:
    """Persist a human-readable summary and return the stats dict."""
    total = len(review_df)
    verdicts = review_df["verdict"].value_counts().to_dict()
    by_group = (
        review_df.groupby("outcome_2r_h16")["event_id"].count().to_dict()
        if "outcome_2r_h16" in review_df.columns
        else {}
    )
    by_dir = (
        review_df["direction"].value_counts().to_dict()
        if "direction" in review_df.columns
        else {}
    )
    incorrect = int(verdicts.get("incorrect", 0))
    false_positive_rate = incorrect / total if total else float("nan")
    summary = {
        "round": 1,
        "n_reviewed": total,
        "verdicts": verdicts,
        "detector_incorrect_rate_proxy": round(false_positive_rate, 4),
        "by_direction": by_dir,
        "by_primary_outcome": by_group,
        "reviewer": REVIEWER,
        "note": (
            "Agent-5 automated first-pass screening of the rolling-levels "
            "baseline (Phase 3). Verdicts follow documented wick/penetration "
            "heuristics; final human sign-off happens in round 2 (task t12) "
            "coordinated with QA. Detector changes: none (baseline frozen, "
            "Phase 3 gate)."
        ),
    }
    lines = [
        "# Liquidity-sweep detector — Human review round 1 (Phase 3)",
        "",
        f"- Events reviewed: **{total}** (deterministic sample, rolling-levels baseline)",
        f"- Reviewer: `{REVIEWER}` (automated first-pass screening; human sign-off in round 2/t12)",
        f"- Verdicts: {verdicts}",
        "- Detector-incorrect proxy rate (marginal wick<0.40 or penetration<0.10 ATR): "
        f"**{false_positive_rate:.1%}**",
        f"- By direction: {by_dir}",
        f"- By primary outcome (2R/h16): {by_group}",
        "",
        "Method: charts show 40 candles before / 24 after the sweep with level, "
        "sweep bar, entry, stop, target and exit; verdict thresholds documented "
        "in `src/visualization/event_chart.py`. `manual_review.csv` columns: "
        "schema §10 (review_id, event_id, reviewer, verdict, notes, "
        "review_version, reviewed_at) + guide §5.6 screening fields.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return summary


def run_round1_review(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    labeled: pd.DataFrame,
    out_dir: str = "reports",
    *,
    n: int = 160,
    seed: int = 42,
    pre_bars: int = 40,
    post_bars: int = 24,
    dpi: int = 120,
) -> dict[str, Any]:
    """Full round-1 review: charts + manual_review.csv + summary.

    Returns a dict with the csv path, chart paths, and stats.
    """
    sample, chart_paths = render_event_charts(
        candles, labeled, out_dir, n=n, seed=seed,
        pre_bars=pre_bars, post_bars=post_bars, dpi=dpi,
    )
    merge_cols = ["event_id"]
    merge_cols.extend(
        col
        for col in ("penetration_atr", "wick_ratio", "reclaim_atr", "level_price")
        if col in events.columns and col not in sample.columns
    )
    merged = sample.merge(events[merge_cols], on="event_id", how="left")
    rows = [assess_event(row) for _, row in merged.iterrows()]
    review_df = pd.DataFrame(rows)
    csv_path = write_manual_review_csv(
        review_df, f"{out_dir.rstrip('/')}/manual_review.csv"
    )

    joined = review_df.set_index("event_id").join(
        sample.set_index("event_id")[["direction", "outcome_2r_h16"]]
    ).reset_index()
    summary_path = f"{out_dir.rstrip('/')}/manual_review_summary.md"
    stats = write_review_summary(joined, summary_path)
    json_path = f"{out_dir.rstrip('/')}/manual_review_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    return {
        "n_reviewed": len(review_df),
        "csv_path": csv_path,
        "summary_path": summary_path,
        "json_path": json_path,
        "n_charts": len(chart_paths),
        "charts": chart_paths,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Repro driver (deterministic; used to generate reports/event_charts/*)
# ---------------------------------------------------------------------------

def generate_round1_artifacts(
    parquet_path: str,
    out_dir: str = "reports",
    *,
    n: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    """End-to-end generator: parquet → causal events → labels → charts+CSV."""
    candles = pd.read_parquet(parquet_path).set_index("timestamp").sort_index()
    from src.events.sweep_detector import build_sweep_events
    from src.labeling.outcome_builder import build_event_labels
    from src.liquidity.rolling_levels import add_rolling_liquidity_levels

    candles = add_rolling_liquidity_levels(candles, lookback=20)  # chart context
    events = build_sweep_events(candles, group_rule="first")  # QA finding F1
    labeled = build_event_labels(candles, events)
    result = run_round1_review(candles, events, labeled, out_dir, n=n, seed=seed)
    result["n_events"] = len(events)
    result["n_labeled"] = len(labeled)
    return result


# ---------------------------------------------------------------------------
# Level-context overlay (round-2 charts)
# ---------------------------------------------------------------------------

LEVEL_CONTEXT_COLORS: Mapping[str, str] = {
    "swing": "#00897b",
    "equal": "#f4511e",
    "prev_day": "#5e35b1",
    "rolling": "#90a4ae",
}


def overlay_level_context(
    fig: Figure,
    context_rows: pd.DataFrame,
    *,
    label: bool = True,
) -> None:
    """Draw active swing/equal/prev-day level prices on an existing chart.

    ``context_rows`` carries ``price`` and ``level_type`` (display-only:
    levels were known at/before the event's decision bar when passed from the
    causal registry).  Lines are drawn without autoscale changes so the trade
    geometry stays readable.
    """
    ax = fig.axes[0]
    y_lo, y_hi = ax.get_ylim()
    drawn = 0
    for _, lv in context_rows.iterrows():
        price = _safe_float(lv.get("price"))
        if price is None or not (y_lo - 0.02 * (y_hi - y_lo) <= price <= y_hi):
            continue
        color = LEVEL_CONTEXT_COLORS.get(str(lv.get("level_type")), "#78909c")
        ax.axhline(price, color=color, lw=0.7, alpha=0.55, ls="--", zorder=1)
        if label:
            ax.text(
                ax.get_xlim()[1] - 0.5, price, f"{lv.get('level_type')} {price:.2f}",
                fontsize=6, color=color, ha="right", va="bottom", alpha=0.85,
            )
        drawn += 1
        if drawn >= 15:
            break

