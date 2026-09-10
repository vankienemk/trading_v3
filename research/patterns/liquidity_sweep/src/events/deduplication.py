"""Duplicate-event removal and cooldown for sweep events (guide section 12).

A single level can produce several consecutive sweep candles.  Two mechanisms
remove duplicates:

1. **Run grouping** — consecutive candidate bars (same direction and level,
   adjacent bar positions) are one *run*; each run contributes a single
   representative event chosen by ``group_rule``:

   * ``first`` (default) — the first bar of the run.  The run's first bar is
     known as soon as it closes, so this rule keeps the event table strictly
     causal (truncation-invariant); it is the system-wide default for the
     event feed (QA finding F1, verified by ``tests/test_no_lookahead.py``);
   * ``deepest_penetration`` — the bar with the largest ``penetration_atr``;
     ties resolve to the earliest bar;
   * ``strongest_reclaim`` — the bar with the largest ``reclaim_atr``.

   ``deepest_penetration`` / ``strongest_reclaim`` are the guide section 12
   "improved" representatives and remain available as *explicit* opt-ins for
   retrospective analyses — they may pick a later bar of the run, so a table
   built with them is not truncation-invariant at run boundaries.  Downstream
   decision stages (confirmation t9, labeling t6) must therefore anchor on the
   run's first bar, or use the default ``first`` rule.

2. **Cooldown** — after an event is accepted at bar ``p``, no further event on
   the same side is accepted until a bar strictly more than ``cooldown_bars``
   away (``position - last_event > cooldown_bars``), exactly the guide's
   ``apply_cooldown`` (default ``cooldown_bars = 4``; see
   ``configs/baseline.yaml`` ``sweep.cooldown_bars``).

All decisions look only at the past: grouping reads earlier bars of the run
and the cooldown reads previously accepted events, so deduplication preserves
the no-look-ahead contract of the detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_GROUP_RULES = ("first", "deepest_penetration", "strongest_reclaim")


def apply_cooldown(signal, bars: int = 4) -> np.ndarray:
    """Guide section 12's exact cooldown over a boolean signal.

    A flag at ``position`` is accepted iff it is set and
    ``position - last_event > bars``, where ``last_event`` is the most recent
    accepted position.  Iterates in increasing position; ``bars`` is a cool-off
    measured in candles after the last accepted event.
    """
    if bars < 0:
        raise ValueError(f"apply_cooldown: bars must be >= 0, got {bars}")
    sig = np.asarray(signal, dtype=bool)
    accepted = np.zeros(len(sig), dtype=bool)
    last_event = -10**9
    for position, flag in enumerate(sig):
        if flag and position - last_event > bars:
            accepted[position] = True
            last_event = position
    return accepted


def _validate_candidates(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the candidate event frame by bar position."""
    required = ["position", "direction", "level_id", "penetration_atr", "reclaim_atr"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"candidate events are missing columns {missing}")
    if len(events) == 0:
        return events.copy()
    if not pd.api.types.is_integer_dtype(events["position"]):
        raise TypeError("candidate events 'position' must be an integer dtype")
    if events["position"].duplicated().any():
        # Same-bar candidate of both directions is legal (separate groups); an
        # exact duplicate (same position AND direction) is not.
        dup = events.duplicated(subset=["position", "direction"], keep=False)
        if dup.any():
            raise ValueError(
                "candidate events contain duplicate (position, direction) rows"
            )
    return events.sort_values("position", kind="stable").reset_index(drop=True)


def group_candidate_events(
    events: pd.DataFrame,
    rule: str = "first",
) -> pd.DataFrame:
    """Collapse consecutive same-level runs of candidates into one event each.

    A new run starts whenever the direction or level changes, or when two
    candidate bars are not adjacent (gap > 1 bar).  The representative of each
    run follows ``rule`` (documented at module level).  Returns the reduced
    frame sorted by ``position``; every non-grouping column of the chosen bar
    is carried through unchanged.
    """
    if rule not in _GROUP_RULES:
        raise ValueError(
            f"group_candidate_events: rule must be one of {_GROUP_RULES}, got {rule!r}"
        )
    df = _validate_candidates(events)
    if df.empty:
        return df

    new_run = (
        (df["direction"] != df["direction"].shift())
        | (df["level_id"] != df["level_id"].shift())
        | df["position"].diff().gt(1)
    ).fillna(True)
    df["_run"] = new_run.cumsum()

    picks: list[pd.DataFrame] = []
    for _, run in df.groupby("_run", sort=False):
        if rule == "first":
            pick = run.iloc[[0]]
        elif rule == "deepest_penetration":
            winner = run["penetration_atr"].idxmax()  # ties -> earliest bar
            pick = run.loc[[winner]]
        else:  # strongest_reclaim
            winner = run["reclaim_atr"].idxmax()
            pick = run.loc[[winner]]
        picks.append(pick)
    out = pd.concat(picks, ignore_index=True).drop(columns=["_run"])
    return out.sort_values("position", kind="stable").reset_index(drop=True)


def select_deduplicated_events(
    events: pd.DataFrame,
    cooldown_bars: int = 4,
    group_rule: str = "first",
) -> pd.DataFrame:
    """Full deduplication: run grouping followed by the per-side cooldown.

    ``events`` is the candidate event frame (one row per sweeping bar, carrying
    at least ``position``, ``direction``, ``level_id``, ``penetration_atr``
    and ``reclaim_atr``).  Returns the accepted events, one representative per
    run and never two accepted events of the same side closer than
    ``cooldown_bars + 1`` bars apart.
    """
    grouped = group_candidate_events(events, rule=group_rule)
    if grouped.empty:
        return grouped

    kept: list[pd.DataFrame] = []
    for direction in grouped["direction"].unique():
        sub = grouped[grouped["direction"] == direction].sort_values("position")
        span = int(sub["position"].max()) + 1
        signal = np.zeros(span, dtype=bool)
        signal[sub["position"].to_numpy()] = True
        accepted = apply_cooldown(signal, bars=cooldown_bars)
        kept.append(sub[accepted[sub["position"].to_numpy()]])
    out = pd.concat(kept, ignore_index=True) if kept else grouped.iloc[0:0]
    return out.sort_values("position", kind="stable").reset_index(drop=True)


def deduplicate_events(
    events: pd.DataFrame,
    cooldown_bars: int = 4,
    group_rule: str = "first",
) -> pd.DataFrame:
    """Alias of :func:`select_deduplicated_events` (guide section 12 naming)."""
    return select_deduplicated_events(
        events, cooldown_bars=cooldown_bars, group_rule=group_rule
    )