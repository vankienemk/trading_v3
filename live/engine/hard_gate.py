"""
hard_gate.py — HMM regime filter / hard gate (guide v1.0 §8, requirements §5)

The hard gate is an *optional* independent filter: it blocks or allows a
pattern to run depending on the market regime at the event's ``known_at``.
It sits outside the feature vector entirely — so plugging a gate can never
change the behaviour of a model whose ``feature_list`` has no ``hmm_*``
(guide §8: "Hard Gate không ảnh hưởng đến model đã train").

The SAME :func:`is_allowed` function is used by the live ``MultiPatternEngine``
and by the multi-backtest runner (requirements §5 — live ≡ backtest), and a
blocked event is annotated ``attributes["discard_reason"] = "regime_blocked"``
so it is still persisted to the Event Lake (spec §7.1) with an auditable
reason.

This module lives in ``live/engine`` (the t3 verify gate mypy-strict's
``live/engine/feature_emitter.py`` + ``live/state/shared_app_state_v2.py``;
hard_gate is covered by the ruff scope) and imports only the strict-clean
``research.regime`` contracts.

Rework §2.3 — independent AND-gate on ``model_prob`` / ``rule_score``
-------------------------------------------------------------------
``trading_v3_rework_request_trend_hmm.md`` §1.3 / §2.3: the multi-pattern
signal path computed ``combined_score = (rule_score + model_prob) / 2``, so a
strong geometry score averaged away a weak model score (the reported event had
``rule 0.93`` + ``prob 0.180`` → combined ``0.55``) and no threshold existed on
either axis.  :func:`probability_gate` adds the requested AND-gate: an event is
dropped when **either** axis is below its own threshold.

Design constraints honoured here:
  * the two axes are tested INDEPENDENTLY — ``combined_score`` is never the
    deciding value (it stays a ranking/feature input for RiskGuard and the
    exposure caps);
  * ``rule_score`` is normalized to [0, 1] per pattern before the comparison,
    because plugins do not share one scale (``liquidity_sweep`` emits 0..100
    while double/wedge/H&S clamp to [0, 1]) — a naive floor on the raw value
    would drop every LSW event;
  * the gate is OPT-IN and config-driven (request §6 warns the proposed
    ``PROB_THRESHOLDS`` numbers were copied from an older proposal and must be
    re-validated, so they are NOT hardcoded as behaviour);
  * the SAME function is used by the live engine and the multi-backtest runner
    (§12 live ≡ backtest).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from research.core.contracts import PatternEvent
from research.regime.base import RegimeState

#: Reason written to ``PatternEvent.attributes["discard_reason"]`` when the
#: hard gate blocks an event (requirements §5).
DISCARD_REASON_REGIME_BLOCKED = "regime_blocked"

#: Reasons returned by :func:`is_allowed`.
REASON_NO_RULE = "no_rule"
REASON_ALLOWED = "allowed"
REASON_STATE_NOT_ALLOWED = "state_not_allowed"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NO_REGIME = "no_regime"

# ---------------------------------------------------------------------------
# Rework §2.3 — independent AND-gate reasons
# ---------------------------------------------------------------------------
#: ``discard_reason`` stamped when ``model_prob`` is below its threshold.
#:
#: The VALUE is deliberately the pre-existing ``"low_probability"`` string, not
#: a new ``"low_model_prob"``.  The B1 gate in ``signal_engine_v2`` already
#: writes ``"low_probability"`` for exactly this concept; introducing a second
#: name would split one idea across two strings in the Event Lake and break
#: downstream filters that match on it (code_verifier finding F-05, t9 AC5).
#: The CONSTANT name is kept so callers importing ``REASON_LOW_MODEL_PROB``
#: keep working.
REASON_LOW_MODEL_PROB = "low_probability"
#: ``discard_reason`` stamped when ``rule_score`` is below the floor.
REASON_LOW_RULE_SCORE = "low_rule_score"
#: Gate allowed the event.
REASON_PROB_OK = "prob_ok"
#: Gate inactive (no threshold configured) — historical behaviour preserved.
REASON_PROB_GATE_OFF = "prob_gate_off"

#: Rework §2.3 switch.  ``False`` (default) keeps the shipped behaviour: the
#: gate computes nothing and no event is dropped.  Request §4 requires a full
#: 204k-bar measurement before this is enabled in live, so the default MUST
#: stay OFF until the measurement task reports.
DEFAULT_PROB_GATE_ENABLED = False

#: Rework §2.3 rule_score floor, on the NORMALIZED [0, 1] scale.
#: ``None`` (default) = no rule-score threshold.  The request proposed 0.6;
#: that number came from an older proposal (§6) and must be re-validated, so it
#: is documented here as the proposal value only, never applied implicitly.
DEFAULT_RULE_SCORE_FLOOR: float | None = None

#: Rework §2.3 per-pattern ``model_prob`` thresholds, on the calibrated
#: probability scale.  Empty (default) = no probability threshold.
#:
#: The request §2.3 proposed ``{double_bottom: 0.55, double_top: 0.55,
#: liquidity_sweep: 0.50, ...}`` but §6 explicitly flags those as copied from
#: an older B1 proposal and "cần xác nhận vẫn còn đúng ngữ cảnh hiện tại".
#: They are therefore recorded as :data:`PROPOSED_PROB_THRESHOLDS` (reference /
#: documentation only) and are NOT wired into the default behaviour.
PROPOSED_PROB_THRESHOLDS: dict[str, float] = {
    "double_bottom": 0.55,
    "double_top": 0.55,
    "liquidity_sweep": 0.50,
    "falling_wedge": 0.50,
    "rising_wedge": 0.50,
    "head_shoulders": 0.55,
    "inverse_head_shoulders": 0.55,
}

#: Fallback threshold for a pattern absent from the configured map.
PROPOSED_DEFAULT_PROB_THRESHOLD = 0.55

#: Per-pattern ``rule_score`` scale (mirrors ``signal_engine_v2``'s map).
#: ``liquidity_sweep`` declares a 0..100 weighted score; every other plugin
#: clamps to [0, 1].  Anything unlisted is auto-detected by magnitude.
RULE_SCORE_SCALE: dict[str, float] = {"liquidity_sweep": 100.0}


def is_allowed(
    event: PatternEvent,
    regime: RegimeState | None,
    regrules: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Decide whether *event* may run under the given regime filter rules.

    Contract (requirements §5 / guide §8):
      * ``regrules`` = per-pattern rules map ``{pattern_name: {...}}``;
      * rules empty / event pattern has no rule -> ``(True, "no_rule")``;
      * ``regime is None`` (plugin inactive / no closed-bar state) ->
        ``(False, "no_regime")`` — fail-closed so "gate on but plugin off"
        never silently lets everything through (requirements §5);
      * otherwise blocked when ``state_name`` not in ``allowed_states``
        (``state_not_allowed``) or ``confidence < min_confidence``
        (``low_confidence``); else ``(True, "allowed")``.
    """
    rules = regrules or {}
    rule = _rule_for_pattern(rules, event.pattern_name)
    if rule is None:
        return True, REASON_NO_RULE
    if regime is None:
        return False, REASON_NO_REGIME
    allowed_states = _allowed_states(rule)
    if allowed_states and regime.state_name not in allowed_states:
        return False, REASON_STATE_NOT_ALLOWED
    min_conf = float(rule.get("min_confidence", 0.0))
    if min_conf > 0.0 and float(regime.confidence) < min_conf:
        return False, REASON_LOW_CONFIDENCE
    return True, REASON_ALLOWED


def _rule_for_pattern(
    rules: Mapping[str, Any] | Sequence[Any] | None,
    pattern_name: str,
) -> dict[str, Any] | None:
    """One rule dict for a pattern, tolerating both config conventions:
    ``rules[pattern_name] = {...}`` (requirements §5 / tests) and the guide
    §8 list-of-dicts form ``[{"pattern": ..., "allowed_states": ...}]``.

    The raw-list form is checked FIRST (a list has no ``.get`` — calling it
    would AttributeError), so a caller that passes the guide §8 list directly
    is handled safely instead of crashing (t4-b3).
    """
    if not rules:
        return None
    if isinstance(rules, (list, tuple)):
        for item in rules:
            if isinstance(item, Mapping) and str(item.get("pattern", "")).strip().lower() == str(
                pattern_name
            ).strip().lower():
                return dict(item)
        return None
    rules_map = rules if isinstance(rules, Mapping) else {}
    direct = rules_map.get(pattern_name)
    if isinstance(direct, dict):
        return direct
    for item in rules_map.values():
        if isinstance(item, Mapping) and str(item.get("pattern", "")).strip().lower() == str(
            pattern_name
        ).strip().lower():
            return dict(item)
    return None


def _allowed_states(rule: dict[str, Any]) -> list[str]:
    raw = rule.get("allowed_states")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x) for x in raw]


def regime_filter_rules(
    regime_config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve the full per-pattern rules map from a regime_config.

    Returns ``None`` (or ``{}``) when no rules are configured — the gate
    then allows every pattern (no_rule).
    """
    rules = dict((regime_config or {}).get("rules") or {})
    return rules or None


def apply_gate(
    event: PatternEvent,
    regime: RegimeState | None,
    regrules: Mapping[str, Any] | None,
) -> bool:
    """Run :func:`is_allowed` and stamp ``discard_reason`` when blocked.

    Returns ``True`` when the event may proceed, ``False`` when blocked.
    A blocked event keeps ``attributes["discard_reason"] = "regime_blocked"``
    (Event Lake persists it — spec §7.1).
    """
    allowed, _reason = is_allowed(event, regime, regrules)
    if not allowed:
        event.attributes["discard_reason"] = DISCARD_REASON_REGIME_BLOCKED
    return allowed


def normalize_rule_score(pattern_name: str, rule_score: Any) -> float:
    """Normalize a pattern's ``rule_score`` onto [0, 1].

    Single source of truth for BOTH the live AND-gate and the backtest mirror
    (§12 live ≡ backtest).  The behaviour matches
    ``signal_engine_v2._normalize_rule_score`` (B2, bug summary): use the
    declared per-pattern scale when known, otherwise infer it from magnitude
    (> 1 ⇒ 0..100 scale).  The result is always clamped to [0, 1].

    This is what makes an LSW event on the 0..100 scale comparable with a DB/DT
    event already on 0..1: a naive floor on the raw value would drop every LSW
    event (e.g. a good 51.0 LSW score would fail a 0.6 floor as ``51 < 0.6``).
    """
    try:
        v = float(rule_score)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    scale = RULE_SCORE_SCALE.get(str(pattern_name).lower())
    if scale is None:
        scale = 100.0 if abs(v) > 1.0 else 1.0
    return max(0.0, min(1.0, v / scale))


def threshold_config(
    prob_gate: Mapping[str, Any] | None,
    pattern_name: str,
) -> tuple[float | None, float | None]:
    """Resolve ``(min_model_prob, min_rule_score)`` for *pattern_name*.

    ``prob_gate`` shape (all keys optional)::

        {
          "enabled": true,                       # master switch
          "min_model_prob": 0.55,                # scalar fallback
          "min_rule_score": 0.60,                # normalized [0,1] floor
          "per_pattern": {"liquidity_sweep": {"min_model_prob": 0.50}},
        }

    Returns ``(None, None)`` when the gate is disabled or nothing is
    configured — the caller then keeps the historical behaviour.  A per-pattern
    entry wins over the scalar fallback; a per-pattern entry may also disable
    just one axis by setting its own key to ``null``.
    """
    cfg = prob_gate if isinstance(prob_gate, Mapping) else {}
    if not bool(cfg.get("enabled", DEFAULT_PROB_GATE_ENABLED)):
        return None, None
    per = cfg.get("per_pattern")
    per_entry = None
    if isinstance(per, Mapping):
        direct = per.get(pattern_name)
        if isinstance(direct, Mapping):
            per_entry = direct
        candidates: Any = per.values()
    elif isinstance(per, (list, tuple)):
        # guide §8 / test convenience: [{"pattern": ..., ...}, ...]
        candidates = per
    else:
        candidates = ()
    if per_entry is None:
        for val in candidates:
            if isinstance(val, Mapping) and str(
                    val.get("pattern", "")).strip().lower() == str(pattern_name).strip().lower():
                per_entry = val
                break
    src: Mapping[str, Any] = per_entry if per_entry is not None else cfg
    return (
        _as_threshold(src.get("min_model_prob"), cfg.get("min_model_prob")),
        _as_threshold(src.get("min_rule_score"), cfg.get("min_rule_score")),
    )


def _as_threshold(value: Any, fallback: Any) -> float | None:
    """Coerce a configured threshold to a usable float or ``None``."""
    for candidate in (value, fallback):
        if candidate is None:
            continue
        try:
            v = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or not (0.0 <= v <= 1.0):
            continue
        return v
    return None


def rule_score_floor(prob_gate: Mapping[str, Any] | None, pattern_name: str) -> float | None:
    """Normalized ``rule_score`` floor for *pattern_name* (``None`` if off)."""
    return threshold_config(prob_gate, pattern_name)[1]


def probability_gate(
    event: PatternEvent,
    prob_gate: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Rework §2.3 — independent AND-gate on ``model_prob`` and ``rule_score``.

    A candidate is dropped when EITHER axis is below its own threshold:

      * ``model_prob < min_model_prob``      -> ``REASON_LOW_MODEL_PROB``;
      * ``rule_score < min_rule_score``      -> ``REASON_LOW_RULE_SCORE``.

    The two axes are tested SEPARATELY.  ``combined_score`` is deliberately
    never consulted here: the entire point of §1.3 is that the average hides a
    weak axis, so gating on it cannot fix the reported failure.

    Fail-open on missing data, matching the module's other optional switches:
    if the gate is disabled/not configured nothing is dropped; if
    ``model_prob`` is absent (tier-2 model not attached — ``model_prob`` stays
    ``None``) the probability axis cannot be judged, so it does not block.  An
    UNKNOWN ``rule_score`` (0.0) DOES block once a floor is configured: a zero
    geometry score is a real value, not missing data.
    """
    min_prob, min_rule = threshold_config(prob_gate, event.pattern_name)
    if min_prob is None and min_rule is None:
        return True, REASON_PROB_GATE_OFF

    rule_norm = normalize_rule_score(event.pattern_name, getattr(event, "rule_score", 0.0))
    if min_rule is not None and rule_norm < min_rule:
        return False, REASON_LOW_RULE_SCORE

    if min_prob is not None:
        raw_prob = getattr(event, "model_prob", None)
        if raw_prob is not None:
            try:
                prob = float(raw_prob)
            except (TypeError, ValueError):
                prob = float("nan")
            if not math.isnan(prob) and prob < min_prob:
                return False, REASON_LOW_MODEL_PROB

    return True, REASON_PROB_OK


def apply_probability_gate(
    event: PatternEvent,
    prob_gate: Mapping[str, Any] | None,
) -> bool:
    """Run :func:`probability_gate` and stamp ``discard_reason`` when blocked.

    Returns ``True`` when the event may proceed.  A blocked event keeps
    ``attributes["discard_reason"]`` set to ``"low_probability"`` or
    ``"low_rule_score"`` (Event Lake §7.1), so the drop is auditable.
    """
    allowed, reason = probability_gate(event, prob_gate)
    if not allowed:
        event.attributes["discard_reason"] = reason
    return allowed


__all__ = [
    "DEFAULT_PROB_GATE_ENABLED",
    "DEFAULT_RULE_SCORE_FLOOR",
    "DISCARD_REASON_REGIME_BLOCKED",
    "PROPOSED_DEFAULT_PROB_THRESHOLD",
    "PROPOSED_PROB_THRESHOLDS",
    "REASON_ALLOWED",
    "REASON_LOW_CONFIDENCE",
    "REASON_LOW_MODEL_PROB",
    "REASON_LOW_RULE_SCORE",
    "REASON_NO_REGIME",
    "REASON_NO_RULE",
    "REASON_PROB_GATE_OFF",
    "REASON_PROB_OK",
    "REASON_STATE_NOT_ALLOWED",
    "RULE_SCORE_SCALE",
    "apply_gate",
    "apply_probability_gate",
    "is_allowed",
    "normalize_rule_score",
    "probability_gate",
    "regime_filter_rules",
    "rule_score_floor",
    "threshold_config",
]