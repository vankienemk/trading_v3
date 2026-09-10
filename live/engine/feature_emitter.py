"""
feature_emitter.py — HMM regime feature emitter (guide v1.0 §4/§5, reqs §4)

Runtime facade + implementation for the live signal engine and the backtest
runner, so live ≡ backtest share ONE emitter / gate path (requirements §5).
Lives in ``live/engine`` (the verify gate runs mypy --strict here) and builds
on the strict-clean ``research.regime`` plugin package.

What this module provides
-------------------------
* :class:`HMMFeatureEmitter` — merges ``hmm_state``, ``hmm_prob_<state>``,
  ``hmm_confidence`` into ``PatternEvent.attributes`` at a causal stamp (the
  last closed bar at/just-before the event's ``known_at`` — never a future
  bar, guide §2) and builds the ``hmm_*`` feature frame per event for
  training (guide §4.2A).
* :func:`state_at_known_at` / :func:`state_at_confirm_bar` — causal state
  lookup helper used by the emitter and the hard gate.
* :func:`build_feature_vector` — guide §4.2B: builds the model-input vector
  from ``model_meta["feature_list"]`` ONLY, so a legacy model's inference can
  never be changed by plugging the plugin (old-model immutability, §4.2C).
* :func:`publish_regime_states` — persists a RegimeState series to
  ``<lake_root>/regime/<SYMBOL>_<TF>.parquet`` (guide §5 step 2) via the
  plugin's ``research.regime.persist`` round-trip.

Activation is decided by the caller (engine wiring, :mod:`regime_wiring`) —
this module never injects unless asked (default OFF).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.core.contracts import PatternEvent
from research.regime.base import BaseRegimePlugin, RegimeState

#: Prefix shared by every HMM-emitted feature (guide §3.2).
HMM_FEATURE_PREFIX = "hmm_"


# ---------------------------------------------------------------------------
# Causal state lookup (requirements §3.4 — state at/just-before known_at)
# ---------------------------------------------------------------------------


def _naive_ns(index: pd.Index[Any]) -> np.ndarray[Any, Any]:
    """UTC-naive datetime64[ns] array of a DatetimeIndex (tz-safe compare)."""
    dt = pd.DatetimeIndex(index)
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_numpy(dtype="datetime64[ns]")


def _naive_ts(ts: pd.Timestamp) -> np.datetime64:
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return np.datetime64(t)


def state_at_known_at(
    states: Sequence[RegimeState],
    df: pd.DataFrame,
    known_at: pd.Timestamp,
) -> RegimeState | None:
    """Causal regime lookup: the state of the last closed bar ≤ ``known_at``.

    ``states`` is aligned to ``df.index`` positionally (``states[i]`` ↔
    ``df.index[i]``).  The search is `≤ known_at` only (guide §2 #1).  Returns
    ``None`` when there is no closed bar at/just-before ``known_at`` — the
    caller then leaves ``hmm_*`` attributes absent (scorer NaN -> fillna
    contract, requirements §6.2).
    """
    if not states or df is None or len(df) == 0 or pd.isna(known_at):
        return None
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    times = _naive_ns(index)
    stamp = _naive_ts(pd.Timestamp(known_at))
    pos = int(np.searchsorted(times, stamp, side="right")) - 1
    if pos < 0 or pos >= len(states):
        return None
    return states[pos]


def state_at_confirm_bar(
    states: Sequence[RegimeState],
    df: pd.DataFrame,
    ev: PatternEvent,
) -> RegimeState | None:
    """Causal state lookup honouring the event's stored confirm-bar stamp.

    Priority (requirements §3.4): ``attributes["confirm_bar"]`` when the
    detector stores it (DB/DT do), validated closed at/just-before the
    event's causal stamp; otherwise fall back to :func:`state_at_known_at`
    (confirmation-timestamp search on LSW-style events).

    No-lookahead guard (guide §2 #1): the authoritative causal bound is
    ``ev.known_at_ts`` (the DERIVED ``max(detect, confirm)`` from
    ``PatternEvent.known_at_ts``) — NOT the raw ``ev.known_at`` field, which
    no detector sets and ``validate_causality`` never writes back.
    Comparing against the raw ``known_at`` (None) would honor an attacker-
    planted ``confirm_bar`` pointing at a FUTURE bar and attach a future
    regime state — a live-reachable look-ahead at the emitter/gate seam.
    The stamp comparison is tz-normalized to UTC-naive via :func:`_naive_ts`
    so a state timestamp in any tz is compared consistently against the
    event's causal stamp.
    """
    if not states or ev is None:
        return None
    cfg_bar = (getattr(ev, "attributes", {}) or {}).get("confirm_bar")
    if cfg_bar is not None:
        try:
            bar = int(cfg_bar)
        except (TypeError, ValueError):
            bar = -1
        if 0 <= bar < len(states):
            stamp = states[bar].timestamp
            known = ev.known_at_ts  # derived causal stamp (never the raw field)
            if not pd.isna(known) and np.datetime64(_naive_ts(stamp)) <= np.datetime64(_naive_ts(known)):
                return states[bar]
    return state_at_known_at(states, df, ev.known_at_ts)


# ---------------------------------------------------------------------------
# Feature emitter
# ---------------------------------------------------------------------------


class HMMFeatureEmitter:
    """Merges the plugin's ``hmm_*`` features into events / feature frames.

    The emitter itself never decides *whether* the plugin is enabled — the
    caller only creates it when the plugin is wired for the assignment
    (guide §5 step 4).  Every lookup is causal and ``known_at``-anchored.
    """

    def __init__(
        self,
        plugin: BaseRegimePlugin,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.plugin = plugin
        self.config = dict(config) if config is not None else {}
        self._feature_names: list[str] = plugin.get_feature_names(self.config)

    def attach(
        self,
        events: Sequence[PatternEvent],
        states_by_bar: Sequence[RegimeState],
        df: pd.DataFrame,
    ) -> int:
        """Write ``hmm_*`` into ``event.attributes`` (returns count attached).

        Causal guarantee: the state used is the last closed bar at/just-
        before the event's ``known_at``.  Events without a matching closed
        bar are skipped (no partial/future feature).
        """
        if not events or not states_by_bar:
            return 0
        attached = 0
        for ev in events:
            reg = state_at_confirm_bar(states_by_bar, df, ev)
            if reg is None:
                continue
            attrs = ev.attributes
            attrs["hmm_state"] = int(reg.state)
            for state_name, prob in reg.state_prob.items():
                attrs[f"hmm_prob_{state_name}"] = float(prob)
            attrs["hmm_confidence"] = float(reg.confidence)
            # lineage (guide §2 #5): exact state stamp + config hash so
            # reports can prove causality and reproducibility.
            attrs["hmm_known_at"] = str(reg.timestamp)
            attrs["hmm_config_hash"] = str(reg.config_hash)
            attached += 1
        return attached

    def feature_frame(
        self,
        df: pd.DataFrame,
        events: Sequence[PatternEvent],
        states_by_bar: Sequence[RegimeState] | None = None,
    ) -> pd.DataFrame:
        """Per-``event_id`` ``hmm_*`` matrix (columns = feature names).

        Rows are indexed by ``event_id`` and the columns are *exactly*
        ``plugin.get_feature_names(config)`` in order (train-time
        ``feature_list`` contract, guide §4.2A).  Missing states -> NaN.
        """
        names = self._feature_names
        states_by_bar = (
            states_by_bar
            if states_by_bar is not None
            else self.plugin.predict(df, self.config or None)
        )
        rows: dict[str, dict[str, float]] = {}
        for ev in events:
            reg = state_at_confirm_bar(states_by_bar, df, ev)
            if reg is None:
                continue
            row: dict[str, float] = {"hmm_state": float(int(reg.state))}
            for state_name, prob in reg.state_prob.items():
                row[f"hmm_prob_{state_name}"] = float(prob)
            row["hmm_confidence"] = float(reg.confidence)
            rows[str(ev.event_id)] = row
        if not rows:
            return pd.DataFrame(columns=names)
        X = pd.DataFrame.from_dict(rows, orient="index")
        X = X.reindex(columns=names)
        return X

    def feature_names(self) -> list[str]:
        return list(self._feature_names)


def regime_flags(config: Mapping[str, Any] | None) -> tuple[bool, bool]:
    """Resolved (emitter, hard-gate) activation flags for a regime_config.

    The wiring layer pre-resolves the YAML master switches into these two
    booleans so the runtime stays dumb and default-OFF.
    """
    cfg = dict(config or {})
    return bool(cfg.get("emitter_enabled", False)), bool(cfg.get("gate_enabled", False))


def requires_hmm_regime(optional_plugins: Sequence[Mapping[str, Any]]) -> bool:
    """§6.3 fail-closed check: any ``hmm_regime`` plugin entry with
    ``required: true`` (registry §4.3) forces the plugin to be active."""
    for entry in optional_plugins:
        if str(entry.get("name", "")) == "hmm_regime" and bool(entry.get("required", False)):
            return True
    return False


def lists_hmm_regime(optional_plugins: Sequence[Mapping[str, Any]]) -> bool:
    """§4.3 listing check: the registry ``optional_plugins`` metadata lists
    ``hmm_regime`` (by name — regardless of the ``required`` flag).

    ``optional_plugins[].required`` marks a model *dependency* (fail-closed);
    merely being *listed* is the pattern-level opt-in signal (guide §4.2A).
    The wiring layer uses this so an entry ``{name: hmm_regime,
    required: false}`` still counts as "listed" for the emitter/gate.
    """
    return any(str(entry.get("name", "")) == "hmm_regime" for entry in optional_plugins)


# ---------------------------------------------------------------------------
# Old-model immutability: build_feature_vector (guide §4.2B)
# ---------------------------------------------------------------------------


def build_feature_vector(
    event: Any,
    model_meta: dict[str, Any],
) -> np.ndarray[Any, Any]:
    """Build the exact model-input vector from ``model_meta["feature_list"]``.

    * ``feature_list`` is authoritative (guide §4.2B) — the vector contains
      ONLY the columns the model was trained on, in that exact order;
    * values come from ``event.attributes`` (falling back to NaN, which the
      scorer/``build_feature_frame`` ``fillna(0.0)`` contract absorbs);
    * a legacy model without ``hmm_*`` never reads the injected attributes, so
      plugging the emitter cannot change its inference (byte-identical). Each
      *extra* column the model did not train on is dropped by construction.
    """
    required = list(model_meta.get("feature_list") or model_meta.get("feature_names") or [])
    if not required:
        return np.array([], dtype=np.float64)
    attrs = getattr(event, "attributes", {}) or {}
    vec = []
    for feat in required:
        raw = attrs.get(feat)
        try:
            vec.append(float(raw) if raw is not None else float("nan"))
        except (TypeError, ValueError):
            vec.append(float("nan"))
    return np.asarray(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Publish to Event Lake (guide §5 step 2)
# ---------------------------------------------------------------------------


def publish_regime_states(
    lake_root: str | Path,
    symbol: str,
    timeframe: str,
    states: Sequence[RegimeState],
) -> Path:
    """Persist a RegimeState series to ``<lake_root>/regime/<SYMBOL>_<TF>.parquet``.

    Recreatable materialisation (guide §5 step 2, requirements v1.0 §5):
    the series is a deterministic recompute from (df, config), so a re-run
    replaces the whole symbol/TF file (not append-only event rows).
    """
    from research.regime.persist import regime_states_to_parquet

    ds = "".join(ch for ch in str(timeframe or "M15").upper() if ch.isalnum()) or "M15"
    target = Path(lake_root) / "regime" / f"{str(symbol).upper()}_{ds}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    return regime_states_to_parquet(list(states), target)


def read_regime_states(
    lake_root: str | Path,
    symbol: str,
    timeframe: str,
) -> list[RegimeState]:
    """Load the persisted RegimeState series (``[]`` when absent)."""
    from research.regime.persist import regime_states_from_parquet

    ds = "".join(ch for ch in str(timeframe or "M15").upper() if ch.isalnum()) or "M15"
    path = Path(lake_root) / "regime" / f"{str(symbol).upper()}_{ds}.parquet"
    if not path.is_file():
        return []
    return list(regime_states_from_parquet(path))


__all__ = [
    "HMM_FEATURE_PREFIX",
    "HMMFeatureEmitter",
    "build_feature_vector",
    "lists_hmm_regime",
    "publish_regime_states",
    "read_regime_states",
    "regime_flags",
    "requires_hmm_regime",
    "state_at_confirm_bar",
    "state_at_known_at",
]