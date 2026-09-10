"""
regime_wiring.py — regime plugin wiring helper (guide v1.0 §4/§5/§8/§9, reqs §4/§5)

Turns the disaggregated configuration (plugin YAML master switches + per
pattern ``features.optional_plugins`` + the registry ``optional_plugins``
metadata) into ONE resolved bundle (a :class:`RegimeSlot`) per assignment
that the runtime consumes uniformly — in the live ``MultiPatternEngine`` and
in the backtest runner — so live ≡ backtest keeps sharing the exact same
gate/emitter decisions (requirements §5).

Activation contract (default **OFF**):
  * the emitter only runs when the plugin YAML master is true AND the pattern
    config lists ``hmm_regime`` in ``features.optional_plugins`` (guide §5
    step 4, §4.2A);
  * the hard gate only runs when the *resolved* regime config enables it
    (``regime_filter.enabled`` true AND the plugin active) AND the pattern
    has a rule;
  * a legacy assignment (no plugin, no rules) resolves to ``regime=None`` and
    runs byte-identical to before (golden: no event/feature change).

Metadata (registry §4.3 → hints):
  * ``optional_plugins[].name=="hmm_regime"`` with ``required:true`` on a
    model whose ``feature_list`` *contains* ``hmm_*`` => plugin MUST be active
    (fail-closed §6.3) or the slot emits no signal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from live.engine.feature_emitter import (
    HMMFeatureEmitter,
    lists_hmm_regime,
    requires_hmm_regime,
)
from research.regime.base import BaseRegimePlugin, RegimeState

#: Optional-plugin name matching the registry §4.3 ``optional_plugins[].name``.
HMM_PLUGIN_NAME = "hmm_regime"


class RegimeSlot:
    """One per-assignment regime bundle (plugin + switches + rules + state)."""

    def __init__(
        self,
        plugin: BaseRegimePlugin,
        emitter_enabled: bool,
        gate_enabled: bool,
        rules: dict[str, Any],
        hmm_config: dict[str, Any],
        assignment_id: str,
        pattern_name: str,
        model_feature_list: Sequence[str] | None = None,
        model_optional_plugins: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.plugin = plugin
        self.emitter_enabled = emitter_enabled
        self.gate_enabled = gate_enabled
        self.rules = rules or {}
        self.hmm_config = dict(hmm_config or {})
        self.assignment_id = assignment_id
        self.pattern_name = pattern_name
        self.model_feature_list = list(model_feature_list or [])
        self.model_optional_plugins = list(model_optional_plugins or [])
        self._emitter = HMMFeatureEmitter(plugin, self.hmm_config or None)
        self._state_cache: list[RegimeState] | None = None

    # ------------------------------------------------------------------
    # Feature emitter
    # ------------------------------------------------------------------
    @property
    def emitter(self) -> HMMFeatureEmitter:
        return self._emitter

    def emitter_active(self) -> bool:
        return self.emitter_enabled and self._list_plugin()

    # ------------------------------------------------------------------
    # Hard gate
    # ------------------------------------------------------------------
    def gate_active(self) -> bool:
        return self.gate_enabled and self._list_plugin() and bool(self.rules)

    def gate_rules(self) -> dict[str, Any]:
        return dict(self.rules)

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------
    def set_states(self, states: Sequence[RegimeState]) -> None:
        self._state_cache = list(states)

    def states(self) -> list[RegimeState] | None:
        return self._state_cache

    # ------------------------------------------------------------------
    # Model-guard helpers
    # ------------------------------------------------------------------
    def model_requires_plugin(self) -> bool:
        """§6.3: model trained WITH ``hmm_*`` and required hmm_regime -> the
        plugin must be active and the emitter must supply the features."""
        has_hmm = any(str(f).startswith("hmm_") for f in self.model_feature_list)
        required = requires_hmm_regime(self.model_optional_plugins)
        return has_hmm and required

    def _list_plugin(self) -> bool:
        """True when this assignment's pattern config opts into hmm_regime.

        When the caller supplies registry §4.3 ``model_optional_plugins``
        metadata, the authoritative signal is *listing*: any entry with
        ``name == "hmm_regime"`` (regardless of its ``required`` flag — a
        listed-but-not-required plugin still counts as wired).  Without
        metadata the caller's explicit emitter/gate switches stand in for the
        listing decision (documented: listing is resolved by the caller before
        setting ``emitter_enabled`` / ``gate_enabled``); the default-path
        model (no metadata AND no switches) stays False so nothing activates.
        """
        if self.model_optional_plugins:
            return lists_hmm_regime(self.model_optional_plugins)
        return bool(self.emitter_enabled or self.gate_enabled)


def load_plugin_config(path: str | Path) -> dict[str, Any]:
    """Read a hmm_regime YAML and return the WHOLE mapping (plugin/hmm/
    regime_filter) — distinguished from ``load_hmm_regime_config`` which
    returns only the ``hmm`` section for the plugin itself."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def resolve_regime_config(
    plugin_yaml: Mapping[str, Any] | None,
    hmm: Mapping[str, Any] | None,
    regime_filter: Mapping[str, Any] | None,
    pattern_name: str,
) -> dict[str, Any]:
    """Combine plugin YAML sections into one resolved per-assignment config.

    Returns a dict with keys ``enabled`` (global), ``hmm``, ``rules``,
    ``gate_enabled``, ``emitter_enabled`` — the canonical shape consumed by
    :func:`resolve_regime_wiring`.  A legacy/no-config call yields an all-false
    dict (default OFF).
    """
    py = dict(plugin_yaml or {})
    plugin_enabled = bool(py.get("enabled", False))
    hmm = dict(hmm or {})
    rf = dict(regime_filter or {})
    gate_enabled = plugin_enabled and bool(rf.get("enabled", False))
    rules = rf.get("rules") or {}
    if isinstance(rules, list):
        rules = _rules_list_to_map(list(rules))
    elif not isinstance(rules, dict):
        rules = {}
    return {
        "enabled": plugin_enabled,
        "hmm": dict(hmm),
        "rules": rules or {},
        "gate_enabled": gate_enabled,
        "emitter_enabled": plugin_enabled and bool(hmm.get("enabled", False)),
    }


def _rules_list_to_map(items: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if isinstance(item, Mapping):
            pat = str(item.get("pattern", "")).strip()
            if pat:
                out[pat] = dict(item)
    return out


def resolve_regime_wiring(
    plugin: BaseRegimePlugin,
    resolved_config: Mapping[str, Any] | None,
    assignment_id: str,
    pattern_name: str,
    model_feature_list: Sequence[str] | None = None,
    model_optional_plugins: Sequence[Mapping[str, Any]] | None = None,
) -> RegimeSlot | None:
    """Build a :class:`RegimeSlot` from the resolved config, or ``None`` when
    the assignment wants no regime wiring (default OFF / legacy).

    ``resolved_config`` is the output of :func:`resolve_regime_config` (or
    ``None`` — treated as fully disabled).  The slot's emitter/gate only
    activate when the global switch is on and the pattern lists the plugin.
    """
    if not resolved_config or not resolved_config.get("enabled"):
        return None
    hmm = dict(resolved_config.get("hmm") or {})
    emitter_enabled = bool(resolved_config.get("emitter_enabled", False))
    gate_enabled = bool(resolved_config.get("gate_enabled", False))
    rules = dict(resolved_config.get("rules") or {})
    return RegimeSlot(
        plugin=plugin,
        emitter_enabled=emitter_enabled,
        gate_enabled=gate_enabled,
        rules=rules,
        hmm_config=dict(hmm),
        assignment_id=assignment_id,
        pattern_name=pattern_name,
        model_feature_list=model_feature_list,
        model_optional_plugins=model_optional_plugins,
    )


__all__ = [
    "HMM_PLUGIN_NAME",
    "RegimeSlot",
    "load_plugin_config",
    "resolve_regime_config",
    "resolve_regime_wiring",
]