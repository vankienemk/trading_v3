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
"""

from __future__ import annotations

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


__all__ = [
    "DISCARD_REASON_REGIME_BLOCKED",
    "REASON_ALLOWED",
    "REASON_LOW_CONFIDENCE",
    "REASON_NO_REGIME",
    "REASON_NO_RULE",
    "REASON_STATE_NOT_ALLOWED",
    "apply_gate",
    "is_allowed",
    "regime_filter_rules",
]