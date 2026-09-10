"""
research.regime — HMM regime plugin (environment / regime engine, guide v1.0)

Independent plugin: causal Gaussian HMM (3-state default) that emits one
:class:`RegimeState` per closed bar via forward filtering — no hmmlearn, no
new dependencies, no lookahead (guide §2).  Feature schema ``hmm_*`` declares
``uses_future_data=False`` (requirements §3.4); fitted plugins and state
sequences persist to JSON / parquet via :mod:`research.regime.persist`.
"""

from research.regime.base import (
    HMM_FEATURE_SCHEMA_VERSION,
    BaseRegimePlugin,
    RegimeState,
)
from research.regime.config import load_hmm_regime_config
from research.regime.gaussian_hmm import (
    CANONICAL_INPUT_FEATURES,
    DEFAULT_STATE_NAMES,
    CausalGaussianHMM,
    compute_causal_input_features,
)
from research.regime.persist import (
    regime_states_frame,
    regime_states_from_parquet,
    regime_states_to_parquet,
)

__all__ = [
    "CANONICAL_INPUT_FEATURES",
    "DEFAULT_STATE_NAMES",
    "HMM_FEATURE_SCHEMA_VERSION",
    "BaseRegimePlugin",
    "CausalGaussianHMM",
    "RegimeState",
    "compute_causal_input_features",
    "load_hmm_regime_config",
    "regime_states_frame",
    "regime_states_from_parquet",
    "regime_states_to_parquet",
]