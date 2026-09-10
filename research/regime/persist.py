"""
persist.py — RegimeState parquet persistence (guide §5, requirements §3.1)

The causal pipeline stores one :class:`RegimeState` per closed bar; this
module writes that sequence to parquet (state list + lineage metadata) and
reads it back losslessly — the same store the feature emitter / OOS script
consume.  ``state_prob`` is a dict, so it is serialised as canonical JSON in
a dedicated column; everything else round-trips natively.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from research.regime.base import RegimeState

#: Parquet column layout (state_prob serialised as ``state_prob_json``).
STATE_COLUMNS: list[str] = [
    "timestamp",
    "state",
    "state_name",
    "state_prob_json",
    "confidence",
    "lag_bars",
    "model_version",
    "config_hash",
]


def regime_states_frame(states: Sequence[RegimeState]) -> pd.DataFrame:
    """One row per state; ``state_prob`` canonical-JSON in its own column."""
    rows = [
        {
            "timestamp": s.timestamp,
            "state": s.state,
            "state_name": s.state_name,
            "state_prob_json": json.dumps(s.state_prob, sort_keys=True),
            "confidence": s.confidence,
            "lag_bars": s.lag_bars,
            "model_version": s.model_version,
            "config_hash": s.config_hash,
        }
        for s in states
    ]
    frame = pd.DataFrame(rows, columns=STATE_COLUMNS)
    if len(frame) > 0:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def regime_states_to_parquet(states: Sequence[RegimeState], path: str | Path) -> Path:
    """Write the state sequence to ``path`` (*.parquet), index=False."""
    target = Path(path)
    regime_states_frame(states).to_parquet(target, index=False)
    return target


def regime_states_from_parquet(path: str | Path) -> list[RegimeState]:
    """Load a state sequence previously written by :func:`regime_states_to_parquet`."""
    frame = pd.read_parquet(path)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    return [
        RegimeState(
            timestamp=pd.Timestamp(timestamps.iloc[i]),
            state=int(frame["state"].iloc[i]),
            state_name=str(frame["state_name"].iloc[i]),
            state_prob={
                str(k): float(v)
                for k, v in json.loads(str(frame["state_prob_json"].iloc[i])).items()
            },
            confidence=float(frame["confidence"].iloc[i]),
            lag_bars=int(frame["lag_bars"].iloc[i]),
            model_version=str(frame["model_version"].iloc[i]),
            config_hash=str(frame["config_hash"].iloc[i]),
        )
        for i in range(len(frame))
    ]