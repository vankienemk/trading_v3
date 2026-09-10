"""Canonical schema constants — the machine-readable twin of docs/SCHEMAS.md.

Owned by Agent 0 (integrator).  Every module that emits or consumes one of
these artifacts must import the constants here instead of re-declaring column
lists, so a schema change has a single implementation point (rule 30.1).

The authoritative human-readable contract is ``docs/SCHEMAS.md``; when they
diverge, the doc wins and this module must be updated.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Processed OHLCV frame
# ---------------------------------------------------------------------------

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
OHLCV_OPTIONAL_COLUMNS: list[str] = ["tick_volume", "real_volume", "spread"]

# ---------------------------------------------------------------------------
# Liquidity levels table (long format)
# ---------------------------------------------------------------------------

LEVEL_COLUMNS: list[str] = [
    "level_id",
    "level_type",
    "direction",
    "price",
    "price_min",
    "price_max",
    "origin_pos",
    "origin_time",
    "known_at",
    "status",
    "touch_count",
    "first_touch_time",
    "last_touch_time",
]

LEVEL_TYPES: list[str] = ["rolling", "swing", "equal", "prev_day"]
#: H1 swing levels keep ``level_type="swing"`` and carry ``is_h1=True``
#: (extension column) — the type enum itself stays schema-locked.
LEVEL_DIRECTIONS: list[str] = ["low", "high"]
LEVEL_STATUSES: list[str] = ["active", "expired"]

#: Level lifecycle states (extension column ``sweep_state``, Phase 4).
SWEEP_STATES: list[str] = ["active", "swept", "invalidated"]

# ---------------------------------------------------------------------------
# Phase 4 (t8) level-registry extensions — schema lock v1.1.
#
# ``build_liquidity_levels`` emits exactly ``LEVEL_COLUMNS + LEVEL_REGISTRY
# _EXTENSION_COLUMNS``; ``level_state_at_bar`` emits ``LEVEL_STATE_COLUMNS``.
# ---------------------------------------------------------------------------

LEVEL_EXTENSION_COLUMNS: list[str] = [
    "known_pos",                # bar position of known_at (causal usability gate)
    "age_bars",                 # bars since origin_pos, as of the final bar
    "bars_since_last_touch",    # NaN when the level has never been touched
    "sweep_state",              # active | swept | invalidated
    "first_swept_at",           # timestamp of first penetration (NaT if never)
    "invalidated_at",           # timestamp of age-expiry (NaT if never)
    "is_h1",                    # True for H1 swing levels
    "max_age_bars",             # expiry policy applied to this level
    "touch_tolerance_atr",      # per-bar tolerance multiplier for touches
    "equal_dispersion_atr",     # (price_max - price_min)/ATR for equal levels
    "formation_atr_tolerance",  # ATR tolerance at cluster formation (guide §9.3)
    "touch_positions",          # internal: exact touch bar positions (equal levels)
]

#: Full unified-registry column set (schema §4 + §4.1 extensions).
LEVEL_REGISTRY_COLUMNS: list[str] = LEVEL_COLUMNS + LEVEL_EXTENSION_COLUMNS

#: Per-bar causal level-state interface columns (schema §4.2,
#: :func:`src.liquidity.level_registry.level_state_at_bar`).
LEVEL_STATE_COLUMNS: list[str] = [
    "level_id",
    "level_type",
    "direction",
    "price",
    "known_at",
    "touch_count",
    "last_touch_time",
    "age_bars",
    "bars_since_last_touch",
    "sweep_state",
    "status",
    "first_swept_at",
    "invalidated_at",
]

# Rolling levels stored as columns on the candle frame.
ROLLING_LEVEL_COLUMNS: list[str] = ["liq_low", "liq_high"]

# ---------------------------------------------------------------------------
# Event table (detector output — reconciled with src/events/sweep_detector.py)
#
# The merged sweep detector (t4) emits exactly this 12-column set, with
# ``direction`` in sweep-side vocabulary ``bullish``/``bearish`` (guide
# sections 10.1/10.2) and ``event_id`` of the form ``SWP-XXXXXX``.  Downstream
# stages (features/labels/scoring) attach their columns on top and derive the
# *trade* direction ``long``/``short`` from the sweep side — see
# ``SWEEP_TO_TRADE_DIRECTION`` and ``normalize_event_direction`` below.
# ---------------------------------------------------------------------------

EVENT_COLUMNS: list[str] = [
    "event_id",
    "event_time",
    "direction",
    "level_id",
    "level_price",
    "event_open",
    "event_high",
    "event_low",
    "event_close",
    "penetration_atr",
    "wick_ratio",
    "reclaim_atr",
]

#: Sweep-side vocabulary used by the detector (guide 10.1 / 10.2).
EVENT_DIRECTIONS: list[str] = ["bullish", "bearish"]

#: Trade-direction vocabulary used by labeling/features/scoring.
TRADE_DIRECTIONS: list[str] = ["long", "short"]

#: Canonical sweep-side -> trade-direction mapping.
SWEEP_TO_TRADE_DIRECTION: dict[str, str] = {
    "bullish": "long",
    "bearish": "short",
}

#: Deterministic event-id prefix emitted by build_sweep_events.
EVENT_ID_PREFIX = "SWP-"


def normalize_event_direction(direction: str) -> str:
    """Map ``direction`` (any spelling) to the trade-direction vocabulary.

    Accepts both the event-table spelling (``bullish``/``bearish``) and the
    trade spelling (``long``/``short``); raises ``ValueError`` otherwise.
    """
    d = str(direction).strip().lower()
    if d in ("long", "bullish"):
        return "long"
    if d in ("short", "bearish"):
        return "short"
    raise ValueError(
        f"unknown sweep direction {direction!r} (expected long/short or "
        "bullish/bearish)"
    )

# ---------------------------------------------------------------------------
# Confirmation columns
# ---------------------------------------------------------------------------

CONFIRMATION_COLUMNS: list[str] = [
    "is_confirmed",
    "confirmation_time",
    "confirmation_delay_bars",
    "confirmation_close",
    "confirmation_range_atr",
    "confirmation_body_ratio",
    "confirmation_volume",
]

# ---------------------------------------------------------------------------
# Label / outcome columns (dynamic suffixes, checked by the labeling module)
# ---------------------------------------------------------------------------

OUTCOME_VALUES: list[str] = ["tp", "sl", "time", "ambiguous"]

# ---------------------------------------------------------------------------
# Manual review CSV
# ---------------------------------------------------------------------------

REVIEW_COLUMNS: list[str] = [
    "review_id",
    "event_id",
    "reviewer",
    "verdict",
    "notes",
    "review_version",
    "reviewed_at",
]

REVIEW_VERDICTS: list[str] = ["correct", "incorrect", "ambiguous"]