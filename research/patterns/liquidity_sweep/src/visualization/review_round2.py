# mypy: ignore-errors
# same rationale as src/visualization/event_chart.py:
# matplotlib/plot code is enforced via ruff + pytest (no type stubs).

"""Human review round 2 (Phase 5b / task t12) on the upgraded event flow.

Scope: after Phase 4 (swing/equal/previous-day levels) and Phase 5
(confirmation) are working, re-run the manual review on the *new* event set —
the round-1 conclusions are NOT reused.  Two event families are reviewed:

* **confirmed events** — the official Pipeline-2 rolling-level sweep events
  that carry a confirmation candle (``data/processed/events.parquet``,
  ``is_confirmed == True``); entry follows confirmation (strategy B), so the
  charts also show the confirmation candle and the review checks whether the
  confirmation delay is reasonable;
* **new-level sweep events** — swing / equal / previous-day levels from the
  causal level registry (Agent 2, task t8) whose price was strictly
  penetrated at some bar after ``known_at`` (registry ``first_swept_at``).
  They are the probe for *"do the new levels produce false positives"*: each
  row is checked against the baseline sweep-quality rules (penetration band,
  wick ratio, close reclaim) and flagged ``detector_pass`` when it would have
  been accepted by the round-1 detector.

Everything is causally anchored (QA finding F1): every event_time is a
decision bar that existed in real time (registry ``known_at <= first_swept_at``
and the dedup/event feed uses ``group_rule="first"``).  The bias-audit sign-off
for Phases 4-5 belongs to Agent 7 (t11 pre-ML / t15 final).

Outputs (never overwrite round 1)::

    reports/event_charts_v2/<group>/<event_id>.png   (group: winners/losers/…)
    reports/manual_review_v2.csv                     (schema §10 + §5.6 fields)
    reports/manual_review_summary_v2.md / .json
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

# Workspace-local matplotlib config/font cache (same pin as event_chart.py):
# the sandbox blocks ~/.matplotlib and a fresh temp dir would rebuild the font
# cache on every process.
_MPL_CONFIG_DIR = os.environ.get("MPLCONFIGDIR") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".mplconfigdir")
)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.labeling.outcome_builder import (  # noqa: E402
    build_event_labels,
    outcome_column,
)
from src.liquidity.level_registry import (  # noqa: E402
    build_liquidity_levels,
)
from src.visualization.event_chart import (  # noqa: E402
    REVIEW_CSV_COLUMNS,
    overlay_level_context,
    plot_event_chart,
)

REVIEW_VERSION = 2
REVIEWER = "agent_5"

#: Stratum names used for round-2 sampling and reporting.
STRATA = ("confirmed", "level_swing", "level_equal", "level_prev_day")
STRATUM_WEIGHTS: Mapping[str, float] = {
    "confirmed": 0.45,
    "level_swing": 0.20,
    "level_equal": 0.18,
    "level_prev_day": 0.17,
}

_REVIEW2_EXTRA_FIELDS = [
    "source",
    "level_type",
    "detector_pass",
    "is_confirmed",
    "confirmation_delay_bars",
    "confirmation_type",
    "confirmation_strength",
]
REVIEW2_CSV_COLUMNS = [*REVIEW_CSV_COLUMNS, *_REVIEW2_EXTRA_FIELDS]

# Baseline sweep-quality rules (round-1 detector) used as the FP probe.
MIN_PEN_ATR = 0.05
MAX_PEN_ATR = 0.50
MIN_WICK = 0.35


# ---------------------------------------------------------------------------
# New-level sweep events from the causal registry (FP probe)
# ---------------------------------------------------------------------------

def build_level_sweep_rows(
    candles: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Registry swept rows -> review-event rows with baseline-rule probe.

    Every registry row of type swing/equal/prev_day (non-H1) whose price was
    strictly penetrated after ``known_at`` (``first_swept_at``) becomes one
    review row.  Sweep-bar metrics (penetration/wick/reclaim in ATR units) are
    recomputed from the candle frame; ``detector_pass`` says whether the row
    would also satisfy the round-1 baseline sweep-quality rules.  The
    ``~detector_pass`` rows are the new-level false-positive candidates.
    """
    required = [
        "level_id", "level_type", "direction", "price", "known_at",
        "first_swept_at", "is_h1",
    ]
    missing = [c for c in required if c not in registry.columns]
    if missing:
        raise ValueError(f"build_level_sweep_rows: registry missing {missing}")

    from src.indicators.atr import add_atr

    work = candles
    if "atr" not in work.columns:
        work = add_atr(candles, period=14)
    n = len(work)
    pos = work.index.get_indexer(pd.DatetimeIndex(registry["first_swept_at"]))
    keep = (
        registry["first_swept_at"].notna()
        & (registry["is_h1"] == False)  # noqa: E712 - canonical bool column
        & (pos > 0)
        & (pos < n - 1)
        & registry["known_at"].le(registry["first_swept_at"])
    )
    reg = registry.loc[keep].copy()
    if reg.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "event_time", "direction", "level_id", "level_type",
                "level_price", "penetration_atr", "wick_ratio", "reclaim_atr",
                "detector_pass",
            ]
        )
    p = pos[keep]
    lo = work["low"].to_numpy()
    hi = work["high"].to_numpy()
    opn = work["open"].to_numpy()
    cpn = work["close"].to_numpy()
    atr = work["atr"].to_numpy()
    price = reg["price"].to_numpy(dtype=float)
    low_dir = (reg["direction"].to_numpy() == "low")

    lp, hp, ap, ol, cl = lo[p], hi[p], atr[p], opn[p], cpn[p]
    pen = np.where(low_dir, (price - lp) / ap, (hp - price) / ap)
    body_lo = np.minimum(ol, cl)
    body_hi = np.maximum(ol, cl)
    lw = np.where(low_dir, body_lo - lp, 0.0)
    uw = np.where(low_dir, 0.0, hp - body_hi)
    rng = hp - lp
    wick = np.where(low_dir, lw, uw) / np.where(rng > 0, rng, np.nan)
    reclaim = np.where(low_dir, (cl - price) / ap, (price - cl) / ap)
    dpass = (
        (pen >= MIN_PEN_ATR) & (pen <= MAX_PEN_ATR)
        & (wick >= MIN_WICK) & (reclaim > 0) & np.isfinite(wick)
    )

    out = pd.DataFrame(
        {
            "event_time": reg["first_swept_at"].reset_index(drop=True),
            "direction": np.where(low_dir, "long", "short"),
            "level_id": reg["level_id"].reset_index(drop=True),
            "level_type": reg["level_type"].reset_index(drop=True),
            "level_price": price,
            "penetration_atr": pen,
            "wick_ratio": wick,
            "reclaim_atr": reclaim,
            "detector_pass": dpass,
        }
    )
    out = out.sort_values(["event_time", "level_type", "level_id"], kind="stable")
    out["event_id"] = [f"LVL-{i:06d}" for i in range(len(out))]
    out = out[
        [
            "event_id", "event_time", "direction", "level_id", "level_type",
            "level_price", "penetration_atr", "wick_ratio", "reclaim_atr",
            "detector_pass",
        ]
    ].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Label the two pools (cheap: confirmed pool is ~1k, level pool is sub-sampled)
# ---------------------------------------------------------------------------

def _label_rows(
    candles: pd.DataFrame,
    events: pd.DataFrame,
    cfg: Mapping[str, Any],
    entry_mode: str,
) -> pd.DataFrame:
    """Label one event family and attach its review metadata columns."""
    lbl_cfg = {**dict(cfg), "entry": {"mode": entry_mode}}
    wide = build_event_labels(candles, events, lbl_cfg)
    extras = events.set_index("event_id")
    cols = [
        c
        for c in ("level_type", "is_confirmed", "confirmation_time",
                  "confirmation_delay_bars", "confirmation_type",
                  "confirmation_strength", "detector_pass", "source",
                  "penetration_atr", "wick_ratio", "reclaim_atr")
        if c in extras.columns and c not in wide.columns
    ]
    if cols:
        wide = wide.join(extras[cols], on="event_id", how="left")
    return wide


def assemble_confirmed_pool(
    candles: pd.DataFrame,
    official_events: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Confirmed rolling events (strategy B: entry after confirmation)."""
    confirmed = official_events[official_events["is_confirmed"] == True].copy()  # noqa: E712
    if confirmed.empty:
        raise ValueError("assemble_confirmed_pool: no confirmed events to review")
    confirmed["level_type"] = "rolling"
    confirmed["detector_pass"] = True
    confirmed["source"] = "confirmed"
    return _label_rows(candles, confirmed, cfg, "next_open_after_confirmation")


def assemble_level_pool(
    candles: pd.DataFrame,
    level_rows: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Sampled new-level sweep events (strategy A: entry after the sweep bar)."""
    rows = level_rows.copy()
    rows["source"] = "level_sweep"
    rows["is_confirmed"] = False
    rows["confirmation_time"] = pd.NaT
    rows["confirmation_delay_bars"] = np.nan
    rows["confirmation_type"] = None
    rows["confirmation_strength"] = np.nan
    return _label_rows(candles, rows, cfg, "next_open_after_sweep")


# ---------------------------------------------------------------------------
# Round-2 sampling (guide 24 / Phase 5b priorities)
# ---------------------------------------------------------------------------

def _eligibility(candles: pd.DataFrame, frame: pd.DataFrame,
                 min_pre: int, min_post: int) -> np.ndarray:
    pos = candles.index.get_indexer(pd.DatetimeIndex(frame["event_time"]))
    return (pos >= min_pre) & (pos + min_post < len(candles))


def _take_balanced(pool: pd.DataFrame, quota: int, rng: np.random.Generator) -> pd.DataFrame:
    """Deterministic direction-alternating draw of ``quota`` rows."""
    if pool.empty or quota <= 0:
        return pool.iloc[0:0]
    pool = pool.copy()
    pool["_r"] = rng.random(len(pool))
    pool = pool.sort_values("_r")
    dirs = sorted(set(pool["direction"]))
    per_dir = {d: pool[pool["direction"] == d] for d in dirs}
    idx: list[int] = []
    cursor = 0
    while len(idx) < quota and any(len(sub) > 0 for sub in per_dir.values()):
        d = dirs[cursor % len(dirs)]
        cursor += 1
        sub = per_dir[d]
        if sub.empty:
            continue
        idx.append(sub.index[0])
        per_dir[d] = sub.iloc[1:]
    return pool.loc[idx]


def sample_round2(
    candles: pd.DataFrame,
    confirmed_labeled: pd.DataFrame,
    level_rows: pd.DataFrame,
    n: int = 160,
    seed: int = 42,
    min_pre: int = 40,
    min_post: int = 24,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Deterministic stratified sample for round 2.

    Stratum priorities follow Phase 5b: confirmed events first, then
    swing/equal/previous-day level sweeps (weights default
    :data:`STRATUM_WEIGHTS`).  Confirmed events are additionally stratified by
    primary outcome (tp/sl/time/ambiguous) so winners/losers/ambiguous charts
    stay represented; level strata alternate direction.  Returns the sampled
    rows of the *labeled* confirmed pool plus the *unlabeled* level rows —
    callers label the level part with :func:`assemble_level_pool`.
    """
    if not 100 <= n <= 500:
        raise ValueError(f"sample_round2: n must be in [100, 500], got {n}")
    w = dict(weights or STRATUM_WEIGHTS)
    rng = np.random.default_rng(seed)

    # --- confirmed stratum: per-outcome quotas, direction-balanced ---------
    elig_c = confirmed_labeled[
        _eligibility(candles, confirmed_labeled, min_pre, min_post)
    ].copy()
    quota_c = min(len(elig_c), max(1, round(n * w["confirmed"])))
    outcome_col = outcome_column(2.0, 16)
    picked: list[pd.DataFrame] = []
    picked_ids: set[int] = set()
    outcome_weights = {"tp": 0.30, "sl": 0.35, "time": 0.20, "ambiguous": 0.15}
    for g, share in outcome_weights.items():
        grp = elig_c[elig_c[outcome_col] == g]
        if grp.empty:
            continue
        q = min(len(grp), max(1, round(quota_c * share)))
        taken = _take_balanced(grp, q, rng)
        if len(taken):
            picked_ids.update(int(i) for i in taken.index)
            picked.append(taken)
    # leftover capacity fills from the whole eligible confirmed pool
    if len(picked_ids) < quota_c:
        rest = elig_c.drop(index=sorted(picked_ids))
        fill = _take_balanced(rest, quota_c - len(picked_ids), rng)
        if len(fill):
            picked_ids.update(int(i) for i in fill.index)
            picked.append(fill)

    # --- level strata: per level_type, direction balanced ------------------
    n_confirmed_picked = len(picked_ids)
    level_quota = max(0, n - n_confirmed_picked)
    level_map = {
        "swing": "level_swing",
        "equal": "level_equal",
        "prev_day": "level_prev_day",
    }
    level_rows = level_rows[level_rows["level_type"].isin(level_map)].copy()
    level_rows["stratum"] = level_rows["level_type"].map(level_map)
    elig_l = level_rows[
        _eligibility(candles, level_rows, min_pre, min_post)
    ].copy()
    quotas: dict[str, int] = {}
    for strat in level_map.values():
        share = w.get(strat, 0.0)
        avail = int((elig_l["stratum"] == strat).sum())
        quotas[strat] = 0 if avail == 0 else min(avail, max(1, round(level_quota * share)))
    cap = {s: int((elig_l["stratum"] == s).sum()) - q for s, q in quotas.items()}
    filled = sum(quotas.values())
    ordered = sorted(level_map.values(), key=lambda s: -w.get(s, 0.0))
    guard = 0
    cursor = 0
    while filled < level_quota and guard < 500:
        s = ordered[cursor % len(ordered)]  # round-robin across types
        cursor += 1
        if cap[s] <= 0:
            continue
        quotas[s] += 1
        cap[s] -= 1
        filled += 1
        guard += 1
    for strat, q in quotas.items():
        pool = elig_l[elig_l["stratum"] == strat].drop(columns=["stratum"])
        picked.append(_take_balanced(pool, q, rng))

    if not picked:
        raise ValueError("sample_round2: nothing to sample")
    out = pd.concat(picked, ignore_index=True)
    out = out.sort_values("event_time", kind="stable").reset_index(drop=True)
    if len(out) < 100:
        raise ValueError(
            f"sample_round2: only {len(out)} eligible events (< 100); relax "
            "min_pre/min_post or reduce n"
        )
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


# ---------------------------------------------------------------------------
# Second-pass screening
# ---------------------------------------------------------------------------

def assess_v2(row: Mapping[str, Any], max_wait_bars: int = 3) -> dict[str, Any]:
    """Round-2 verdict for one merged review row.

    Adds the round-1 sweep-quality verdict plus round-2 checks: for confirmed
    events the confirmation delay is inspected (a delay at ``max_wait_bars``
    is noted as borderline); for new-level sweeps ``detector_pass`` drives the
    false-positive verdict (registry penetrations that would fail the baseline
    rules are marked ``incorrect``).  Verdict ∈ correct/incorrect/ambiguous.
    """
    outcome = str(row.get("outcome_2r_h16", ""))
    exit_reason = str(row.get("exit_reason", ""))
    source = str(row.get("source", ""))
    level_type = str(row.get("level_type", ""))
    wick = _f(row.get("wick_ratio"))
    pen = _f(row.get("penetration_atr"))
    has_metrics = _f(row.get("wick_ratio")) is not None

    delay = _f(row.get("confirmation_delay_bars"))
    is_conf = bool(row.get("is_confirmed")) if "is_confirmed" in row else None

    if outcome == "ambiguous" or exit_reason == "ambiguous":
        verdict, sweep_clear, comment = "ambiguous", "maybe", (
            "TP and SL inside the same candle: intrabar order unknown; human review required"
        )
    elif not has_metrics:
        verdict, sweep_clear, comment = "ambiguous", "maybe", (
            "sweep metrics missing from event row; human review required"
        )
    elif source == "level_sweep" and not bool(row.get("detector_pass")):
        verdict, sweep_clear, comment = "incorrect", "no", (
            "new-level penetration fails the baseline sweep-quality rules "
            "(penetration band / wick / close reclaim) - false-positive candidate"
        )
    elif wick >= 0.50 and pen >= 0.15:
        verdict, sweep_clear, comment = "correct", "yes", (
            "textbook sweep: clean directional wick with clear penetration"
        )
    elif wick < 0.40 or pen < 0.10:
        verdict, sweep_clear, comment = "incorrect", "no", (
            "marginal pattern: tiny wick/penetration candidate for spread noise"
        )
    else:
        verdict, sweep_clear, comment = "ambiguous", "maybe", (
            "between clean and marginal thresholds; human review required"
        )

    conf_note = ""
    if is_conf is True and delay is not None:
        if delay >= max_wait_bars:
            conf_note = f"; confirmation delay AT max_wait={max_wait_bars} bars (borderline)"
        else:
            conf_note = f"; confirmation delay {delay:g} bar(s) ok"
    elif is_conf is False and source == "confirmed":
        conf_note = "; UNCONFIRMED but in confirmed pool (inconsistent)"
    elif source == "level_sweep":
        conf_note = "; level-sweep row (confirmation check n/a by design)"

    score = round(50 * min(max(wick or 0.0, 0.0), 1.0) + 40 * min(max(pen or 0.0, 0.0), 0.5) / 0.5)
    notes = (
        f"round2 source={source}; level_type={level_type}; "
        f"primary_outcome={outcome}; exit={exit_reason}; "
        f"wick_ratio={row.get('wick_ratio')}; penetration_atr={row.get('penetration_atr')}; "
        f"reclaim_atr={row.get('reclaim_atr')}; level_id={row.get('level_id')}; "
        f"bars_held={row.get('bars_held')}; net_result_r={row.get('net_result_r')}"
        f"{conf_note}; review={comment}"
    )
    detector_correct = {
        "correct": "yes",
        "incorrect": "no",
        "ambiguous": "maybe",
    }[verdict]
    return {
        "review_id": f"R2-{row.get('event_id')}",
        "event_id": row.get("event_id"),
        "reviewer": REVIEWER,
        "verdict": verdict,
        "notes": notes,
        "review_version": REVIEW_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector_correct": detector_correct,
        "liquidity_level_clear": (
            "yes" if verdict == "correct" else ("no" if verdict == "incorrect" else "maybe")
        ),
        "sweep_clear": sweep_clear,
        "confirmation_clear": (
            "yes"
            if is_conf is True and delay is not None and delay < max_wait_bars
            else ("n/a" if source == "level_sweep" else "review")
        ),
        "reviewer_score": int(max(0, min(100, score))),
        "reviewer_comment": comment,
        "source": source,
        "level_type": level_type,
        "detector_pass": bool(row.get("detector_pass")) if "detector_pass" in row else None,
        "is_confirmed": is_conf,
        "confirmation_delay_bars": delay,
        "confirmation_type": row.get("confirmation_type"),
        "confirmation_strength": _f(row.get("confirmation_strength")),
    }


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Round-2 orchestration + files
# ---------------------------------------------------------------------------

def write_review2_csv(review_rows: Iterable[Mapping[str, Any]], path: str) -> str:
    df = pd.DataFrame(review_rows, columns=REVIEW2_CSV_COLUMNS)
    df = df.sort_values(["reviewed_at", "event_id"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def write_review2_summary(review_df: pd.DataFrame, path: str) -> dict[str, Any]:
    """Persist round-2 summary; returns stats dict (also written as JSON)."""
    total = len(review_df)
    verdicts = review_df["verdict"].value_counts().to_dict()
    by_source = review_df["source"].value_counts().to_dict()
    level_rows = review_df[review_df["source"] == "level_sweep"]
    conf_rows = review_df[review_df["is_confirmed"] == True]  # noqa: E712
    delay = conf_rows["confirmation_delay_bars"].dropna()
    conf_stats: dict[str, Any] = {
        "n_confirmed": len(conf_rows),
        "mean_delay_bars": float(delay.mean()) if len(delay) else None,
        "median_delay_bars": float(delay.median()) if len(delay) else None,
        "p90_delay_bars": float(delay.quantile(0.90)) if len(delay) else None,
        "max_delay_bars": float(delay.max()) if len(delay) else None,
        "share_delay_at_max": (
            float((delay >= 3).mean()) if len(delay) else None
        ),
    }
    per_type = level_rows.groupby("level_type").agg(
        n=("event_id", "count"),
        n_pass=("detector_pass", "sum"),
        n_incorrect=("verdict", lambda s: int((s == "incorrect").sum())),
    )
    stats = {
        "round": 2,
        "n_reviewed": total,
        "verdicts": verdicts,
        "by_source": by_source,
        "confirmed_delay": conf_stats,
        "level_sweep_by_type": per_type.reset_index().to_dict("records"),
        "directions": review_df["direction"].value_counts().to_dict(),
        "reviewer": REVIEWER,
        "note": (
            "Agent-5 automated first-pass screening (Phase 5b). Level-sweep "
            "rows come from the causal level registry (first_swept_at); "
            "detector_pass mirrors the round-1 baseline sweep-quality rules "
            "and flags new-level false-positive candidates. Confirmation "
            "delay checks target confirmed events. Final bias sign-off: "
            "Agent 7 (t11/t15). Detector changes: none."
        ),
    }
    lines = [
        "# Liquidity-sweep detector — Human review round 2 (Phase 5b)",
        "",
        f"- Events reviewed: **{total}** (deterministic sample, seed-configurable)",
        f"- Reviewer: `{REVIEWER}` (automated first-pass screening; human sign-off via t12 + QA t11/t15)",
        f"- Verdicts: {verdicts}",
        f"- By source: {by_source}",
        f"- By direction: {stats['directions']}",
        f"- Confirmation delay (confirmed events): {conf_stats}",
        f"- New-level sweep quality (level rows): {per_type.to_string()}",
        "",
        "Method: charts show 40 candles before / 24 after the decision bar with "
        "level, sweep bar, confirmation candle (when present), entry, stop, "
        "target and exit. New-level rows are strict registry penetrations "
        "checked against baseline sweep-quality rules (detector_pass). "
        "`manual_review_v2.csv` columns: schema §10 (review_id, event_id, "
        "reviewer, verdict, notes, review_version, reviewed_at) + guide §5.6 "
        "screening fields + round-2 audit extras. Round-1 file is untouched.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return stats


def run_round2_review(
    candles: pd.DataFrame,
    official_events: pd.DataFrame,
    registry: pd.DataFrame,
    out_dir: str = "reports",
    *,
    n: int = 160,
    seed: int = 42,
    pre_bars: int = 40,
    post_bars: int = 24,
    dpi: int = 120,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full round-2 review: charts (event_charts_v2) + manual_review_v2.csv.

    ``official_events`` is the Pipeline-2 event table (rolling sweeps +
    confirmation columns); ``registry`` is the causal level registry (Agent 2).
    """
    conf_cfg = dict(cfg or {})
    confirmed_pool = assemble_confirmed_pool(candles, official_events, conf_cfg)
    level_rows = build_level_sweep_rows(candles, registry)

    sample = sample_round2(
        candles, confirmed_pool, level_rows, n=n, seed=seed,
        min_pre=pre_bars, min_post=post_bars,
    )
    sampled_level = sample[sample["event_id"].str.startswith("LVL-")]
    level_labeled = (
        assemble_level_pool(candles, sampled_level, conf_cfg)
        if len(sampled_level)
        else pd.DataFrame(columns=confirmed_pool.columns)
    )
    sampled_conf = sample[~sample["event_id"].str.startswith("LVL-")]
    merged = pd.concat([sampled_conf, level_labeled], ignore_index=True)
    merged = merged.sort_values("event_time", kind="stable").reset_index(drop=True)
    if len(merged) < 100:
        raise ValueError(
            f"run_round2_review: sampled+labeled only {len(merged)} rows (< 100)"
        )

    # ---- charts grouped by outcome ----------------------------------------
    base = _ensure_dir(f"{out_dir.rstrip('/')}/event_charts_v2")
    chart_paths: list[str] = []
    reg_price = registry["price"].to_numpy(dtype=float)
    reg_type = registry["level_type"].to_numpy()
    reg_known = registry["known_at"].to_numpy()

    for _, row in merged.iterrows():
        group = OUTCOME_GROUP_DIRS_LOOKUP(str(row.get("outcome_2r_h16", "time")))
        folder = _ensure_dir(f"{base}/{group}")
        fig = plot_event_chart(
            candles, row, pre_bars=pre_bars, post_bars=post_bars,
            target_col="target_2r", outcome_col="outcome_2r_h16",
        )
        # overlay nearby causal new-level lines (display-only context)
        context = _nearby_levels(
            reg_price, reg_type, reg_known, row, candles,
            pre_bars=pre_bars, post_bars=post_bars, limit=8,
        )
        if len(context):
            overlay_level_context(fig, context)
        path = f"{folder}/{row['event_id']}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        chart_paths.append(path)

    # ---- review CSV + summary --------------------------------------------
    rows = [assess_v2(r) for _, r in merged.iterrows()]
    review_df = pd.DataFrame(rows)
    csv_path = write_review2_csv(
        review_df, f"{out_dir.rstrip('/')}/manual_review_v2.csv"
    )
    joined = review_df.set_index("event_id").join(
        merged.set_index("event_id")[["direction", "outcome_2r_h16"]]
    ).reset_index()
    summary_path = f"{out_dir.rstrip('/')}/manual_review_summary_v2.md"
    stats = write_review2_summary(joined, summary_path)
    json_path = f"{out_dir.rstrip('/')}/manual_review_summary_v2.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    return {
        "n_reviewed": len(merged),
        "n_confirmed_reviewed": int((merged["event_id"].str.startswith("SWP-")).sum()),
        "n_level_reviewed": int((merged["event_id"].str.startswith("LVL-")).sum()),
        "csv_path": csv_path,
        "summary_path": summary_path,
        "json_path": json_path,
        "n_charts": len(chart_paths),
        "charts": chart_paths,
        "stats": stats,
    }


def _nearby_levels(
    reg_price: np.ndarray,
    reg_type: np.ndarray,
    reg_known: np.ndarray,
    row: pd.Series,
    candles: pd.DataFrame,
    *,
    pre_bars: int,
    post_bars: int,
    limit: int,
) -> pd.DataFrame:
    """Up to ``limit`` registry levels known at the event bar, price-near it."""
    ts = pd.Timestamp(row["event_time"])
    known_mask = reg_known <= ts  # registry tz-aware Timestamps
    pos = int(candles.index.get_loc(ts))
    win = candles.iloc[max(0, pos - pre_bars): min(len(candles), pos + 1 + post_bars)]
    price = _f(row.get("level_price"))
    if price is None or int(known_mask.sum()) == 0:
        return pd.DataFrame(columns=["price", "level_type"])
    lo, hi = float(win["low"].min()), float(win["high"].max())
    pad = max(hi - lo, 0.0)
    span = (reg_price >= lo - 0.05 * pad) & (reg_price <= hi + 0.05 * pad) & known_mask
    idx = np.flatnonzero(span)
    if len(idx) > limit:
        idx = idx[np.argsort(np.abs(reg_price[idx] - price))[:limit]]
    return pd.DataFrame(
        {"price": reg_price[idx], "level_type": reg_type[idx]}
    )


def generate_round2_artifacts(
    parquet_path: str,
    events_parquet_path: str,
    out_dir: str = "reports",
    *,
    n: int = 160,
    seed: int = 42,
    registry: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """End-to-end round-2 generator: candles + events + registry -> artifacts."""
    candles = pd.read_parquet(parquet_path).set_index("timestamp").sort_index()
    from src.liquidity.rolling_levels import add_rolling_liquidity_levels

    official = pd.read_parquet(events_parquet_path)
    candles = add_rolling_liquidity_levels(candles, lookback=20)  # chart context
    if registry is None:
        import yaml

        cfg = yaml.safe_load(open("configs/baseline.yaml"))
        registry = build_liquidity_levels(candles, cfg)
    result = run_round2_review(
        candles, official, registry, out_dir, n=n, seed=seed, cfg={}
    )
    result["n_events_official"] = len(official)
    result["n_registry"] = len(registry)
    return result


def OUTCOME_GROUP_DIRS_LOOKUP(token: str) -> str:
    """Group folder for an outcome token (unknown -> time)."""
    from src.visualization.event_chart import OUTCOME_GROUP_DIRS

    return OUTCOME_GROUP_DIRS.get(token, "time")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
