"""
base.py - HMM regime plugin core contracts (guide v1.0 §3.1, requirements §3.1-§3.2)

The regime plugin is an independent *environment / regime engine* — it never
competes with ``BasePatternDetector`` plugins and never depends on a specific
pattern.  It observes general market data (returns, volatility, volume) and
emits one :class:`RegimeState` per closed bar, purely causally.

Hard constraints (HMM_REGIME_PLUGIN_INTEGRATION_GUIDE.md §2):
  * at bar ``t`` close the plugin only ever reads data ``<= t``;
  * no fit-on-full-history-then-map-backwards retrodiction;
  * ``RegimeState.timestamp`` == the bar close time (``known_at``);
  * any applied lag must be declared in ``RegimeState.lag_bars``;
  * everything reproducible (config_hash, seed, data/model version).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from research.core.contracts import PatternFeature

#: Feature-schema version of the HMM feature emitter (guide §2 #6).  Bump only
#: on a semantic change of the emitted ``hmm_*`` features.
HMM_FEATURE_SCHEMA_VERSION = "hmm-v1.0"


@dataclass
class RegimeState:
    """One causal regime state, aligned to one closed bar (requirements §3.1).

    ``timestamp`` is the bar close time — the only moment the state "exists".
    Contract: ``states[i]`` corresponds to ``df.index[i]`` of the frame passed
    to :meth:`BaseRegimePlugin.predict`, and the returned list has exactly as
    many entries as closed bars.  ``state_prob`` maps each configured state
    name to its filtered posterior ``P(state | x_1..x_t)``.
    """

    timestamp: pd.Timestamp          # bar close time == known_at of the state
    state: int                       # argmax filtered posterior (0, 1, 2, ...)
    state_name: str                  # "trending" | "sideways" | "high_vol" ...
    state_prob: dict[str, float]     # {"trending": 0.82, "sideways": 0.15, ...}
    confidence: float                # max(state_prob.values())
    lag_bars: int                    # applied lag (bar close shifted back by this)
    model_version: str               # plugin version
    config_hash: str                 # sha1(canonical_json(hmm_config))[:12] (§6.2)


class BaseRegimePlugin(ABC):
    """Abstract contract every regime plugin must implement (requirements §3.2).

    Unlike ``BasePatternDetector`` this ABC intentionally carries no
    ``detect`` — the plugin asks "which market regime is this bar in?", not
    "did a pattern form?".  ``get_feature_schema`` returns the features the
    plugin can inject into ``PatternEvent.attributes``; every one of them
    declares ``uses_future_data=False`` (guide §2 #6, requirements §3.4).
    """

    #: Canonical plugin name (matches ``optional_plugins[].name`` in the model
    #: registry §4.3).
    name: str = "hmm_regime"
    #: Plugin semantic version — bump on any behavior change (§6.2); the
    #: default config MUST carry this same ``"version"``.
    version: str = "1.0.0"
    #: Short registry code used in report/GUI annotations.
    short_key: str = "HMM"
    #: Feature-schema version of the emitted features (guide §2 #6).
    feature_schema_version: ClassVar[str] = HMM_FEATURE_SCHEMA_VERSION

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> BaseRegimePlugin:
        """Fit the regime model on HISTORICAL data (research / warm-up only).

        MUST never be called on a window that overlaps the evaluation
        (OOS) region.  Returns ``self`` for chaining.
        """

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> list[RegimeState]:
        """Causal inference — one state per closed bar, in index order.

        ``df`` must be sorted strictly ascending by timestamp.  The state at
        bar ``t`` is derived exclusively from observations ``<= t`` (forward
        filtering).  No backward smoothing, no full-series Viterbi.
        """

    @abstractmethod
    def get_feature_schema(
        self,
        config: dict[str, Any] | None = None,
    ) -> list[PatternFeature]:
        """Schema of the injectable features — every feature declares
        ``uses_future_data=False`` and ``available_at='confirm'`` (bar close).
        """

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Default plugin config — MUST contain ``"version"`` (== ``version``)
        plus at least: n_states, state_names, input_features (log-return /
        atr-norm / volume-zscore), lag_bars, min_confidence, covariance_type
        (requirements §3.2, guide §6).
        """

    def get_feature_names(
        self,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        """``hmm_state`` + ``hmm_prob_<state_name>`` per state + ``hmm_confidence``.

        Feature names are derived dynamically from the configured
        ``state_names`` (requirements §3.4), so a 2-state config yields a
        different column set than the default 3-state one.
        """
        cfg = config or self.get_default_config()
        names: list[str] = list(cfg.get("state_names") or self.get_default_config()["state_names"])
        return ["hmm_state"] + [f"hmm_prob_{n}" for n in names] + ["hmm_confidence"]