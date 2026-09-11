"""
trend_hmm.py — CausalTrendHMM: trend-DIRECTION regime plugin (rework §2.4)

The pre-existing regime plugin (:mod:`research.regime.gaussian_hmm`) classifies
*volatility* regimes (``trending`` / ``sideways`` / ``high_vol``).  It answers
"how wild is the tape?", NOT "which way is it going?".  The rework request
§2.4 / §1.4 needs the missing axis: an ``uptrend`` / ``downtrend`` / ``range``
classification, produced by the *same* proven causal HMM pipeline instead of a
parallel slope heuristic that would have to be calibrated from scratch.

This module reuses that pipeline wholesale — ``BaseRegimePlugin`` /
``RegimeState``, :func:`compute_causal_input_features`,
:func:`_log_gaussian_emissions`, :func:`_init_transition`,
:func:`_index_version` and the artifact codec are all imported from
``research.regime.gaussian_hmm``; only the *semantics* change:

  * the feature vector carries trend information — a normalized mean return
    over a window, a normalized regression slope, plus ATR-normalized
    volatility and a volume z-score;
  * the 3 states are named ``uptrend`` / ``downtrend`` / ``range``;
  * state IDENTITY is pinned to the sign of the learned drift
    (``canonical_state_names``): a raw EM/Baum-Welch solution is only defined
    up to a state permutation, so without this a refit could silently invert
    the direction of the gate.  The ordering rule (downtrend < range <
    uptrend by fitted mean drift) is index-stable by construction and is
    asserted at fit time.

MEASURED NEGATIVE RESULT — READ BEFORE USING THIS AS A GATE
-----------------------------------------------------------
This plugin is mechanically correct and the states are stable, but **the trend
state does not carry a forward-looking directional edge, so a gate built on it
does not earn its keep.**  Two independent measurements on the full XAUUSD M15
frame (this author's, and ``measurement_analyst``'s via
``docs/trend_hmm_option_c.py`` / ``docs/rework_trend_hmm_measure.md``) agree:

  1. **No forward predictive edge.**  Mean forward log-return per bar by state
     is essentially flat across states — at 24/96/288 bars the downtrend vs
     uptrend spread is ~1e-6 to 4e-6 per bar, one to two orders of magnitude
     below the symbol's unconditional drift, while the per-state CIs at
     n≈65,000 bars are ±7e-7.  A Gaussian HMM on returns labels the PAST
     accurately (it is a filter by construction) but does not predict the next
     leg.  This is expected behaviour, not a bug in this module — but it means
     "downtrend before a double bottom" has almost no information to exploit.
  2. **As the §2.4 gate the result is mixed and partly INVERTED.**  At
     ``extreme1_bar`` lookup, ``double_bottom`` with ``downtrend`` is no
     better than ``range`` (the only clearly bad state is ``uptrend``, so the
     defensible rule would be "not uptrend", not "== downtrend").  For
     ``double_top`` the request §2.4 step 4 rule is **inverted by the data**:
     keeping only ``uptrend`` selects the WORST-performing subset, while
     ``downtrend`` is the best.  Shipping §2.4 step 4 as written would make
     ``double_top`` measurably worse.
  3. **§4.2 sample size FAILS.**  Requiring ``downtrend`` before DB /
     ``uptrend`` before DT leaves only ~50 DB and ~40-41 DT events in the OOS
     window, well under the required 100.

Recommendation: keep this plugin as a research/measurement component, leave it
**OFF by default** (there is no activation wiring in this module — nothing runs
unless a caller explicitly constructs it), and do NOT enable a §2.4 gate on the
strength of the evidence above.  No threshold in this module was tuned to
manufacture an OOS≥100 sample; the negative result is reported as measured.

The gate's own per-state numbers are reproducible with::

    PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages \
        /tmp/ptv2_venv/bin/python docs/trend_hmm_option_c.py

RELATED (no-lookahead caveat for the extreme1 lookup)
-----------------------------------------------------
For DB/DT, ``extreme1_bar`` lies BEFORE the confirm bar, so reading the state
at ``extreme1_bar`` is causal at confirm time (measured: confirm - extreme1 has
median 33 bars, min 12, and 0 events have extreme1 after confirm).  The
existing gate seam (``state_at_confirm_bar`` / ``hard_gate.is_allowed``),
however, exposes the state at the CONFIRM bar — and gating on that keeps only
1-2 DB and 0 DT events.  The §2.4 step 3 anchor and the step 4 gate wiring are
therefore not compatible as written.

No-lookahead (guide §2 / requirements §3.4 — the contract this plugin must
honour):
  * every input feature at bar ``t`` is computed from bars ``<= t`` only
    (rolling / ewm with no centering, no ``shift(-1)``);
  * ``fit`` is research/warm-up only and must never overlap the evaluation
    window (the caller owns the fit/eval split, exactly as in the volatility
    plugin — see ``research/multi_backtest/scripts/oos_hmm_regime_filter.py``);
  * ``predict`` is strict forward filtering: state at ``t`` = argmax of
    ``P(s_t | x_1..x_t)``.  No backward pass, no Viterbi, no refit, no
    retro-fitting of a full series back onto itself;
  * ``predict`` is prefix-invariant: states ``<= t`` are bit-identical whether
    the caller passes the full frame or the frame truncated at ``t`` (tested);
  * the state attached to an event is looked up at the event's ``known_at`` /
    ``confirm_bar`` by the existing emitter helper
    (``live.engine.feature_emitter.state_at_confirm_bar``), never at a future
    bar.  For DB/DT the extreme1 bar IS inside the pattern window, so a
    *point lookup at extreme1* would be future information at confirm time —
    see ``trend_state_at_extreme1`` in the module notes below.

Scope note (rework §2.4 step 5): this plugin is a *gate input*, NOT a model
feature.  It adds no ``hmm_*`` column to any trained artifact, therefore
``feature_schema_version`` (``double-v1.0``) is deliberately NOT bumped and no
model needs re-training.  Consuming the trend state as a model feature is a
separate, artifact-invalidating change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import logsumexp  # type: ignore[import-untyped]
from sklearn.mixture import GaussianMixture

from research.core.config_hash import compute_config_hash
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    PatternFeature,
)
from research.regime.base import (
    HMM_FEATURE_SCHEMA_VERSION,
    BaseRegimePlugin,
    RegimeState,
)
from research.regime.gaussian_hmm import (
    _index_version,
    _init_transition,
    _log_gaussian_emissions,
)

#: Trend regime vocabulary (rework §2.4 step 2: "3-state: uptrend / downtrend /
#: range").  ``range`` is index 1, i.e. between the two directional states in
#: canonical order — see :data:`CANONICAL_STATE_NAMES`.
DEFAULT_TREND_STATE_NAMES: tuple[str, ...] = ("downtrend", "range", "uptrend")

#: The canonical (already-ordered) vocabulary.  ``state_names`` supplied by a
#: caller is reordered into this sequence at fit time so the state index always
#: carries the same meaning.
CANONICAL_STATE_NAMES: tuple[str, ...] = ("downtrend", "range", "uptrend")

#: Index of the non-directional state in canonical order.
RANGE_STATE_INDEX = 1

#: Trend input-feature vocabulary.  All four are derived causally from OHLCV.
#:
#: ``volume_zscore_20`` — the fourth feature of the *volatility* plugin — is
#: deliberately NOT in this default set: on the shipped XAUUSD M15 parquet the
#: tick-volume column is zero for 201,518 of 204,133 bars (nonzero only in
#: 2018), so the z-score is NaN for 98.7% of history and a ``dropna()`` fit
#: would keep only 2,619 bars.  That single fact is also why the existing
#: volatility HMM collapses to 99.1% ``trending`` (verified: fit 2018-01-02 →
#: 2023-09-30, predict over the full frame).  ``range_pos_20_norm`` replaces it
#: with a trend-relevant, always-available dimension: where price sits inside
#: its own 20-bar range.
TREND_INPUT_FEATURES: tuple[str, str, str, str] = (
    "mean_return_20_norm",
    "slope_20_norm",
    "range_pos_20_norm",
    "atr_14_norm",
)

#: Names this plugin can compute itself from raw OHLCV.
_CANONICAL_FEATURE_NAMES: frozenset[str] = frozenset(TREND_INPUT_FEATURES)

#: Prefix of the trend feature schema.  Deliberately NOT ``hmm_`` — a ``hmm_*``
#: prefix is what the runtime treats as the volatility-regime feature family
#: (``live.engine.feature_emitter.HMM_FEATURE_PREFIX``) and what triggers the
#: §6.3 fail-closed model dependency.  Keeping the trend family namespaced
#: separately is what makes "gate input, not model feature" mechanically true.
TREND_FEATURE_PREFIX = "trend_hmm_"


# ---------------------------------------------------------------------------
# Causal feature computation (trend axis).  Every function here is causal:
# rolling/ewm never centered, no shift(-1), no full-series statistic.
# ---------------------------------------------------------------------------


def _window_scale(atr: pd.Series[Any], window: int) -> pd.Series[Any]:
    """Causal scale for an N-bar move: ``atr * sqrt(N)`` (zero-safe).

    A mean return of ``r`` over the last N bars moves price by ``r * N``; the
    volatility of that move over the same horizon is ``atr_14 * sqrt(N)``.
    Dividing by that scale makes the trend features scale-free ("how far did
    we travel relative to noise"), so a 2020 level and a 2026 level are
    directly comparable and EM does not end up fitting one state per price era.
    """
    scale = atr * float(np.sqrt(float(window)))
    return scale.where(scale > 0.0, np.nan)


def _mean_return(df: pd.DataFrame, window: int) -> pd.Series[Any]:
    """Rolling SUM of the 1-bar log return, in ATR*sqrt(window) units.

    The cumulative log return over the trailing ``window`` bars is
    ``close_t - close_{t-window}`` in log space, which equals the rolling SUM
    of 1-bar log returns — NOT the rolling mean.  (A rolling mean is
    ``move / window``; using it here would leave the feature dimension and the
    ATR normalization off by a factor of ``window``, which is how an earlier
    revision of this file produced non-unit-scale features.)  Uses only bars
    ``<= t`` (``rolling`` right-aligned, ``min_periods=window``).
    """
    close = df["close"].astype(np.float64)
    ratio = (close / close.shift(1)).to_numpy(dtype=np.float64)
    ret = pd.Series(np.asarray(np.log(ratio)), index=df.index)
    atr = _atr_raw(
        df["high"].astype(np.float64),
        df["low"].astype(np.float64),
        close,
        14,
    )
    move = ret.rolling(window, min_periods=window).sum()
    return move / _window_scale(atr, window)


def _slope(df: pd.DataFrame, window: int) -> pd.Series[Any]:
    """Rolling OLS slope of close over the last ``window`` bars, ATR-normalized.

    Computed in closed form from rolling sums of ``y`` and ``y * t`` with the
    GLOBAL bar axis ``t``; the window-local centered numerator is then
    ``Sum(y*t) - mean(t_window) * Sum(y)`` with ``mean(t_window)`` on that same
    global axis.  (Mixing a global ``Sum(y*t)`` with a window-local ``t`` —
    which starts at 0 for every window — is how an earlier revision of this
    file produced a slope off by ``mean(t) * window``, a value that grows with
    the bar index and swamps the real signal.)

    The fitted move across the window is ``slope * (window - 1)``; dividing by
    ``atr * sqrt(window)`` puts it on the same scale as :func:`_mean_return`,
    so EM sees two comparable trend dimensions instead of one outlier.
    """
    close = df["close"].astype(np.float64)
    atr = _atr_raw(
        df["high"].astype(np.float64),
        df["low"].astype(np.float64),
        close,
        14,
    )
    n = len(close)
    t_axis = pd.Series(np.arange(n, dtype=np.float64), index=close.index)
    # window-local centered t has variance (window^2 - 1) / 12
    sxx = float(window * (window * window - 1)) / 12.0
    if sxx <= 0.0:  # pragma: no cover - window >= 2 is enforced by config
        raise ValueError(f"slope window {window} is too small for an OLS fit")

    y = close
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_ty = (y * t_axis).rolling(window, min_periods=window).sum()
    sum_t = t_axis.rolling(window, min_periods=window).sum()
    count = pd.Series(1.0, index=close.index).rolling(
        window, min_periods=window
    ).sum()
    # Sum(y * (t - mean(t_window))) == Sum(y*t) - mean(t_window) * Sum(y)
    sxy = sum_ty - (sum_t / count) * sum_y
    slope = sxy / sxx
    move = slope * float(window - 1)
    return move / _window_scale(atr, window)


def _atr_raw(
    high: pd.Series[Any],
    low: pd.Series[Any],
    close: pd.Series[Any],
    period: int,
) -> pd.Series[Any]:
    """Wilder ATR(period) in PRICE units — causal (ewm, no centering)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _range_pos(df: pd.DataFrame, window: int) -> pd.Series[Any]:
    """Where close sits inside its own trailing ``window``-bar range, in
    ``[-1, +1]``: ``(close - mid) / half_range``, causal and always defined.

    Complements the two slope features (which agree 0.85 on XAUUSD M15) with
    a mean-reversion / breakout axis: +1 = closing at the top of the range,
    -1 = at the bottom.  Flat windows (zero range) yield 0.0 rather than NaN
    so a quiet window cannot silently drop a bar from the fit.
    """
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    roll_high = high.rolling(window, min_periods=window).max()
    roll_low = low.rolling(window, min_periods=window).min()
    mid = (roll_high + roll_low) / 2.0
    half = (roll_high - roll_low) / 2.0
    pos = (close - mid) / half.where(half > 0.0, np.nan)
    return pos.fillna(0.0).where(close.notna() & roll_high.notna(), np.nan)


def _atr_norm(df: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
    """Wilder ATR(period) normalized by close — causal."""
    close = df["close"].astype(np.float64)
    return _atr_raw(
        df["high"].astype(np.float64), df["low"].astype(np.float64), close, period
    ) / close


def _volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series[Any]:
    """Rolling (volume - mean)/std over the last ``period`` bars — causal."""
    vol = df["volume"].astype(np.float64)
    mean = vol.rolling(period, min_periods=period).mean()
    std = vol.rolling(period, min_periods=period).std()
    return (vol - mean) / std


def compute_causal_trend_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the trend input-feature matrix, computing canonical columns itself.

    Semantics mirror ``compute_causal_input_features`` (a caller-supplied
    column of the same name wins; unknown names raise loudly so a
    misconfigured pipeline fails instead of leaking).  Every produced series
    at bar ``t`` depends only on bars ``<= t``.
    """
    if not feature_names:
        raise ValueError("input_features must not be empty")
    cfg = dict(config or {})
    mean_window = int(cfg.get("mean_return_window", 20))
    slope_window = int(cfg.get("slope_window", 20))
    range_window = int(cfg.get("range_window", 20))
    atr_period = int(cfg.get("atr_period", 14))
    vol_window = int(cfg.get("volume_zscore_window", 20))
    if mean_window < 2 or slope_window < 2 or range_window < 2 or vol_window < 2:
        raise ValueError(
            "feature windows (mean/slope/range/volume) must be >= 2 "
            "and atr_period >= 1"
        )
    if atr_period < 1:
        raise ValueError("atr_period must be >= 1")
    series: dict[str, pd.Series[Any]] = {}
    for name in feature_names:
        if name in df.columns:
            series[name] = df[name].astype(np.float64)
        elif name == "mean_return_20_norm":
            series[name] = _mean_return(df, mean_window)
        elif name == "slope_20_norm":
            series[name] = _slope(df, slope_window)
        elif name == "range_pos_20_norm":
            series[name] = _range_pos(df, range_window)
        elif name == "atr_14_norm":
            series[name] = _atr_norm(df, atr_period)
        elif name == "volume_zscore_20":
            if "volume" not in df.columns:
                raise ValueError(
                    "input feature 'volume_zscore_20' requires a 'volume' column "
                    "in df — cannot compute it from OHLCV alone"
                )
            series[name] = _volume_zscore(df, vol_window)
        else:
            raise ValueError(
                f"unknown input feature {name!r} — provide it as a df column or "
                f"use one of the canonical trend features {TREND_INPUT_FEATURES}"
            )
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class CausalTrendHMM(BaseRegimePlugin):
    """Causal 3-state trend-direction regime (downtrend / range / uptrend).

    Same modelling approach as :class:`research.regime.gaussian_hmm.CausalGaussianHMM`
    (seeded GaussianMixture emission init + numpy Baum-Welch + strict forward
    filtering) so behaviour is consistent across the two regime plugins; the
    difference is the feature vector (trend-carrying) and the state semantics
    (direction, pinned by the canonical ordering rule).
    """

    name = "trend_hmm_regime"
    version = "1.0.0"
    short_key = "TREND"
    feature_schema_version: ClassVar[str] = HMM_FEATURE_SCHEMA_VERSION

    def __init__(self) -> None:
        self._fitted = False
        self._config: dict[str, Any] = {}
        self._config_hash = ""
        self._data_version_ = ""
        self._pi_: np.ndarray[Any, Any] | None = None
        self._transmat_: np.ndarray[Any, Any] | None = None
        self._means_: np.ndarray[Any, Any] | None = None
        self._covars_: np.ndarray[Any, Any] | None = None
        self._feature_mean_: np.ndarray[Any, Any] | None = None
        self._feature_std_: np.ndarray[Any, Any] | None = None
        self._state_names_: list[str] = list(CANONICAL_STATE_NAMES)
        self._direction_dim_: int = 0

    # ------------------------------------------------------------------
    # Public state access (read-only)
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def data_version(self) -> str:
        return self._data_version_

    @property
    def state_names(self) -> list[str]:
        """Canonical state vocabulary actually in force (fit-resolved)."""
        return list(self._state_names_)

    @property
    def transmat(self) -> np.ndarray[Any, Any]:
        self._require_fitted("transmat")
        assert self._transmat_ is not None
        return self._transmat_.copy()

    @property
    def means(self) -> np.ndarray[Any, Any]:
        self._require_fitted("means")
        assert self._means_ is not None
        return self._means_.copy()

    @property
    def covars(self) -> np.ndarray[Any, Any]:
        self._require_fitted("covars")
        assert self._covars_ is not None
        return self._covars_.copy()

    # ------------------------------------------------------------------
    # Config (§6.2 — versioned, canonical, hashable)
    # ------------------------------------------------------------------

    def get_default_config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_states": 3,
            "state_names": list(CANONICAL_STATE_NAMES),
            "input_features": list(TREND_INPUT_FEATURES),
            "mean_return_window": 20,
            "slope_window": 20,
            "range_window": 20,
            "atr_period": 14,
            "volume_zscore_window": 20,
            "direction_feature": "mean_return_20_norm",
            "lag_bars": 0,
            "min_confidence": 0.50,
            "covariance_type": "full",
            "n_iter": 100,
            "tol": 1e-4,
            "reg_covar": 1e-6,
            "random_state": 42,
            "min_fit_bars": 2000,
            "min_predict_bars": 200,
        }

    def _resolve_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        merged = {**self.get_default_config(), **(config or {})}
        merged["version"] = self.version  # plugin version is authoritative (§6.2)
        n_states = int(merged["n_states"])
        state_names = [str(n) for n in merged["state_names"]]
        if n_states != 3:
            raise ValueError(
                "CausalTrendHMM models exactly 3 trend states "
                "(downtrend/range/uptrend) — a different n_states would make the "
                "direction ordering rule ill-defined"
            )
        if len(state_names) != n_states or len(set(state_names)) != n_states:
            raise ValueError(
                f"state_names {state_names} must have exactly n_states={n_states} "
                f"unique entries"
            )
        if set(state_names) != set(CANONICAL_STATE_NAMES):
            raise ValueError(
                f"state_names {state_names} must be a permutation of "
                f"{list(CANONICAL_STATE_NAMES)} — the trend vocabulary is fixed so "
                f"the gate's `allowed_states` config can never drift"
            )
        input_features = list(merged["input_features"])
        if not input_features:
            raise ValueError("input_features must not be empty")
        direction_feature = str(merged["direction_feature"])
        if direction_feature not in input_features:
            raise ValueError(
                f"direction_feature {direction_feature!r} must be one of "
                f"input_features {input_features}"
            )
        for key in (
            "mean_return_window",
            "slope_window",
            "range_window",
            "volume_zscore_window",
        ):
            if int(merged[key]) < 2:
                raise ValueError(f"{key} must be >= 2")
        if int(merged["atr_period"]) < 1:
            raise ValueError("atr_period must be >= 1")
        lag = int(merged["lag_bars"])
        if lag < 0:
            raise ValueError("lag_bars must be >= 0")
        conf = float(merged["min_confidence"])
        if not 0.0 <= conf <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        cov_type = str(merged["covariance_type"])
        if cov_type not in ("full", "diag"):
            raise ValueError(
                f"covariance_type {cov_type!r} unsupported — implement 'full' or 'diag'"
            )
        if int(merged["n_iter"]) < 1:
            raise ValueError("n_iter must be >= 1")
        if float(merged["tol"]) <= 0.0:
            raise ValueError("tol must be > 0")
        if int(merged["min_fit_bars"]) < 1 or int(merged["min_predict_bars"]) < 0:
            raise ValueError("min_fit_bars >= 1 and min_predict_bars >= 0 required")

        merged["n_states"] = n_states
        merged["state_names"] = state_names
        merged["input_features"] = input_features
        merged["direction_feature"] = direction_feature
        merged["mean_return_window"] = int(merged["mean_return_window"])
        merged["slope_window"] = int(merged["slope_window"])
        merged["range_window"] = int(merged["range_window"])
        merged["atr_period"] = int(merged["atr_period"])
        merged["volume_zscore_window"] = int(merged["volume_zscore_window"])
        merged["lag_bars"] = lag
        merged["min_confidence"] = conf
        merged["covariance_type"] = cov_type
        merged["n_iter"] = int(merged["n_iter"])
        merged["tol"] = float(merged["tol"])
        merged["reg_covar"] = float(merged["reg_covar"])
        merged["random_state"] = int(merged["random_state"])
        merged["min_fit_bars"] = int(merged["min_fit_bars"])
        merged["min_predict_bars"] = int(merged["min_predict_bars"])
        return merged

    # ------------------------------------------------------------------
    # Frame validation (strictly-ascending timestamps = anti-lookahead guard)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError(
                "df must be sorted strictly ascending by timestamp — a frame with "
                "future/duplicate stamps is rejected (timeline attack guard)"
            )
        missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required OHLC columns: {missing}")

    def _validate_input_columns(
        self, df: pd.DataFrame, input_features: Sequence[str]
    ) -> None:
        for name in input_features:
            if name in df.columns:
                continue
            if name not in _CANONICAL_FEATURE_NAMES:
                raise ValueError(
                    f"input feature {name!r} is neither a df column nor canonical "
                    f"{TREND_INPUT_FEATURES}"
                )
            if name == "volume_zscore_20" and "volume" not in df.columns:
                raise ValueError(
                    "input feature 'volume_zscore_20' requires a 'volume' column"
                )

    # ------------------------------------------------------------------
    # fit — Baum-Welch EM on the causal train window, then canonical reorder
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> CausalTrendHMM:
        cfg = self._resolve_config(config)
        self._validate_frame(df)
        input_features = list(cfg["input_features"])
        self._validate_input_columns(df, input_features)
        min_bars = int(cfg["min_fit_bars"])
        if len(df) < min_bars:
            raise ValueError(
                f"CausalTrendHMM.fit: frame has {len(df)} bars but "
                f"min_fit_bars={min_bars} — train window too short for stable EM"
            )
        X = compute_causal_trend_features(df, input_features, cfg)
        Xc = X.dropna()
        if len(Xc) < min_bars:
            raise ValueError(
                f"CausalTrendHMM.fit: only {len(Xc)} causal rows after NaN warmup "
                f"but min_fit_bars={min_bars} is required"
            )
        Xm = Xc.to_numpy(dtype=np.float64)
        feature_mean = Xm.mean(axis=0)
        feature_std = Xm.std(axis=0)
        safe_std = np.where(feature_std < 1e-12, 1.0, feature_std)
        Xs = (Xm - feature_mean) / safe_std

        n_states = int(cfg["n_states"])
        cov_type = str(cfg["covariance_type"])
        seed = int(cfg["random_state"])
        reg_covar = float(cfg["reg_covar"])
        n_iter = int(cfg["n_iter"])
        tol = float(cfg["tol"])
        _, d = Xs.shape

        # --- emission init via seeded GaussianMixture (sklearn, no hmmlearn) ---
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type=cov_type,
            random_state=seed,
            n_init=1,
            max_iter=200,
            tol=1e-6,
            reg_covar=reg_covar,
        )
        gmm.fit(Xs)
        means = np.asarray(gmm.means_, dtype=np.float64)
        gmm_covars = np.asarray(gmm.covariances_, dtype=np.float64)
        weights = np.asarray(gmm.weights_, dtype=np.float64)
        covars = np.zeros((n_states, d, d), dtype=np.float64)
        for m in range(n_states):
            if cov_type == "full":
                covars[m] = gmm_covars[m]
            else:  # "diag" — gmm.covariances_ rows hold the diagonal
                covars[m] = np.diag(gmm_covars[m])

        pi = weights / (weights.sum() + 1e-300)
        trans = _init_transition(n_states)
        log_em = _log_gaussian_emissions(Xs, means, covars)

        prev_llik = -np.inf
        for _ in range(n_iter):
            # --- E-step: forward-backward in log space (scipy logsumexp) ---
            log_alpha = np.full((Xs.shape[0], n_states), -np.inf, dtype=np.float64)
            log_alpha[0] = np.log(pi + 1e-300) + log_em[0]
            log_trans = np.log(trans + 1e-300)
            for t in range(1, Xs.shape[0]):
                log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
                log_alpha[t] += log_em[t]
            log_beta = np.full((Xs.shape[0], n_states), -np.inf, dtype=np.float64)
            log_beta[-1] = 0.0
            for t in range(Xs.shape[0] - 2, -1, -1):
                log_beta[t] = logsumexp(
                    log_trans + log_em[t + 1][None, :] + log_beta[t + 1][None, :],
                    axis=1,
                )
            llik = float(logsumexp(log_alpha[-1]))
            gamma = np.exp(log_alpha + log_beta - llik)
            log_xi = (
                log_alpha[:-1][:, :, None]
                + log_trans[None, :, :]
                + log_em[1:][:, None, :]
                + log_beta[1:][:, None, :]
                - llik
            )
            xi = np.exp(log_xi)

            # --- M-step (with tiny smoothing so pi / trans stay > 0) ---
            gamma_sum = gamma.sum(axis=0) + 1e-300
            pi_new = gamma[0] / (gamma.sum() + 1e-300)
            pi_new = pi_new / (pi_new.sum() + 1e-300)
            trans_num = xi.sum(axis=0) + 1e-9
            trans = trans_num / trans_num.sum(axis=1, keepdims=True)
            means_new = (gamma.T @ Xs) / gamma_sum[:, None]
            covars_new = np.zeros((n_states, d, d), dtype=np.float64)
            for m in range(n_states):
                diff = Xs - means_new[m]
                wsqrt = np.sqrt(gamma[:, m : m + 1]) * diff
                cov = (wsqrt.T @ wsqrt) / (gamma_sum[m] + 1e-300)
                cov = cov + np.eye(d) * reg_covar
                covars_new[m] = cov if cov_type == "full" else np.diag(np.diag(cov))
            means, covars, pi = means_new, covars_new, pi_new

            if np.isfinite(llik) and np.isfinite(prev_llik) and abs(llik - prev_llik) < tol:
                break
            prev_llik = llik
            log_em = _log_gaussian_emissions(Xs, means, covars)

        # --- canonical state identity (direction-pinned, permutation-proof) ---
        direction_dim = input_features.index(str(cfg["direction_feature"]))
        state_names, order = canonical_state_order(means, direction_dim)
        means = means[order]
        covars = covars[order]
        pi = pi[order]
        trans = trans[np.ix_(order, order)]

        # --- persist fitted state + lineage metadata (guide §2 #5, §6.2) ---
        self._pi_ = pi
        self._transmat_ = trans
        self._means_ = means
        self._covars_ = covars
        self._feature_mean_ = feature_mean
        self._feature_std_ = safe_std
        self._state_names_ = list(state_names)
        self._direction_dim_ = int(direction_dim)
        cfg["state_names"] = list(state_names)
        self._config = dict(cfg)
        self._config_hash = compute_config_hash(cfg)
        dt_index = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if dt_index is None:  # pragma: no cover - _validate_frame already raised
            raise ValueError("df must have a DatetimeIndex")
        self._data_version_ = _index_version(dt_index)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict — strict causal forward filtering (no backward, no refit)
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> list[RegimeState]:
        self._require_fitted("predict")
        self._validate_frame(df)
        if not isinstance(df.index, pd.DatetimeIndex):  # pragma: no cover
            raise ValueError("df must have a DatetimeIndex")
        dt_index = df.index
        cfg = self._resolve_config(config) if config is not None else self._config
        state_names = list(cfg.get("state_names") or self._state_names_)
        lag = int(cfg["lag_bars"])
        n = len(df)
        if n < int(cfg["min_predict_bars"]):
            return [self._default_state(dt_index[i], lag, state_names) for i in range(n)]
        input_features = list(cfg["input_features"])
        self._validate_input_columns(df, input_features)
        X = compute_causal_trend_features(df, input_features, cfg)
        Xm = X.to_numpy(dtype=np.float64)
        assert self._feature_mean_ is not None and self._feature_std_ is not None
        fill_std = np.where(self._feature_std_ < 1e-12, 1.0, self._feature_std_)
        Xs = (Xm - self._feature_mean_) / fill_std
        valid = np.isfinite(Xs).all(axis=1)
        return self._filter_forward(Xs, valid, dt_index, lag, state_names)

    def _filter_forward(
        self,
        Xs: np.ndarray[Any, Any],
        valid: np.ndarray[Any, Any],
        index: pd.DatetimeIndex,
        lag: int,
        state_names: list[str],
    ) -> list[RegimeState]:
        """Forward-filter over the (T, d) matrix; row t only reads rows <= t.

        Rows without data (NaN features — warmup) carry the previous posterior
        forward: no observation, no regime update.
        """
        n_states = len(state_names)
        t_rows = Xs.shape[0]
        assert self._pi_ is not None and self._transmat_ is not None
        assert self._means_ is not None and self._covars_ is not None
        log_trans = np.log(self._transmat_ + 1e-300)
        log_pi = np.log(self._pi_ + 1e-300)
        log_em = _log_gaussian_emissions(Xs, self._means_, self._covars_)

        log_alpha = np.full((t_rows, n_states), -np.inf, dtype=np.float64)
        for t in range(t_rows):
            if valid[t]:
                if t == 0 or not np.isfinite(log_alpha[t - 1]).all():
                    log_alpha[t] = log_pi + log_em[t]
                else:
                    log_alpha[t] = logsumexp(
                        log_alpha[t - 1][:, None] + log_trans, axis=0
                    )
                    log_alpha[t] += log_em[t]
            elif t > 0:
                log_alpha[t] = log_alpha[t - 1]  # no evidence — carry posterior

        states: list[RegimeState] = []
        for t in range(t_rows):
            if not np.isfinite(log_alpha[t]).all():
                prob_vec = self._pi_
            else:
                prob_vec = np.exp(log_alpha[t] - logsumexp(log_alpha[t]))
            prob_map = {name: float(prob_vec[m]) for m, name in enumerate(state_names)}
            s = int(np.argmax(prob_vec))
            states.append(
                RegimeState(
                    timestamp=index[max(0, t - lag)] if lag > 0 else index[t],
                    state=s,
                    state_name=state_names[s],
                    state_prob=prob_map,
                    confidence=float(prob_vec[s]),
                    lag_bars=lag,
                    model_version=self.version,
                    config_hash=self._config_hash,
                )
            )
        return states

    def _default_state(
        self,
        timestamp: pd.Timestamp,
        lag: int,
        state_names: list[str],
    ) -> RegimeState:
        """Stability guard state: state 0 (``downtrend`` in canonical order).

        Returned when the predict window is shorter than ``min_predict_bars``
        or as the no-evidence anchor row.  A consumer gating on an allowed
        state must NOT treat this as an observation — ``confidence`` is a
        flat 0.5 and every ``state_prob`` is uniform-ish, exactly as the
        volatility plugin does.
        """
        k = len(state_names)
        probs: dict[str, float] = {}
        for m, name in enumerate(state_names):
            probs[name] = 0.5 if m == 0 else (0.5 / (k - 1) if k > 1 else 0.0)
        return RegimeState(
            timestamp=timestamp,
            state=0,
            state_name=state_names[0],
            state_prob=probs,
            confidence=0.5,
            lag_bars=lag,
            model_version=self.version,
            config_hash=self._config_hash,
        )

    def _require_fitted(self, what: str) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"CausalTrendHMM.{what}: model not fitted — call fit(df, config) first"
            )

    # ------------------------------------------------------------------
    # Feature schema — trend_hmm_* (gate side), NOT the hmm_* model family
    # ------------------------------------------------------------------

    def get_feature_schema(
        self,
        config: dict[str, Any] | None = None,
    ) -> list[PatternFeature]:
        cfg = self._resolve_config(config)
        state_names = list(cfg["state_names"])
        schema: list[PatternFeature] = [
            PatternFeature(
                name=f"{TREND_FEATURE_PREFIX}state",
                dtype="str",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description=(
                    "trend-direction regime name at bar close (known_at): "
                    "downtrend | range | uptrend"
                ),
            )
        ]
        schema.extend(
            PatternFeature(
                name=f"{TREND_FEATURE_PREFIX}prob_{name}",
                dtype="float",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description=f"filtered posterior probability of trend regime '{name}'",
            )
            for name in state_names
        )
        schema.append(
            PatternFeature(
                name=f"{TREND_FEATURE_PREFIX}confidence",
                dtype="float",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description="max trend state probability (filtered posterior)",
            )
        )
        return schema

    def get_feature_names(
        self,
        config: dict[str, Any] | None = None,
    ) -> list[str]:
        return [f.name for f in self.get_feature_schema(config)]

    # ------------------------------------------------------------------
    # Artifact persistence — JSON + base64 (no new deps)
    # ------------------------------------------------------------------

    def to_artifact_dict(self) -> dict[str, Any]:
        """Complete reproducible artifact payload (guide §2 #5, §3.3)."""
        self._require_fitted("to_artifact_dict")
        assert self._pi_ is not None
        assert self._transmat_ is not None
        assert self._means_ is not None
        assert self._covars_ is not None
        assert self._feature_mean_ is not None
        assert self._feature_std_ is not None
        return {
            "model_version": self.version,
            "config": self._config,
            "config_hash": self._config_hash,
            "data_version": self._data_version_,
            "state_names": list(self._state_names_),
            "params": {
                "pi": _encode(self._pi_),
                "transmat": _encode(self._transmat_),
                "means": _encode(self._means_),
                "covars": _encode(self._covars_),
                "feature_mean": _encode(self._feature_mean_),
                "feature_std": _encode(self._feature_std_),
            },
        }

    @classmethod
    def from_artifact_dict(cls, payload: dict[str, Any]) -> CausalTrendHMM:
        plugin = cls()
        params = payload["params"]
        plugin._pi_ = _decode(params["pi"])
        plugin._transmat_ = _decode(params["transmat"])
        plugin._means_ = _decode(params["means"])
        plugin._covars_ = _decode(params["covars"])
        plugin._feature_mean_ = _decode(params["feature_mean"])
        plugin._feature_std_ = _decode(params["feature_std"])
        plugin._config = dict(payload["config"])
        plugin._state_names_ = list(
            payload.get("state_names") or plugin._config.get("state_names")
            or CANONICAL_STATE_NAMES
        )
        plugin._direction_dim_ = list(plugin._config["input_features"]).index(
            str(plugin._config["direction_feature"])
        )
        plugin._config_hash = str(payload["config_hash"])
        plugin._data_version_ = str(payload["data_version"])
        plugin._fitted = True
        return plugin

    def save(self, path: str | Any) -> Any:
        """Persist the fitted plugin as a JSON artifact (round-trip safe)."""
        import json
        from pathlib import Path

        target = Path(path)
        target.write_text(
            json.dumps(self.to_artifact_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Any) -> CausalTrendHMM:
        """Load a previously saved artifact back into a fitted plugin."""
        import json
        from pathlib import Path

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_artifact_dict(payload)


# ---------------------------------------------------------------------------
# Canonical state ordering — the label-stability guarantee
# ---------------------------------------------------------------------------


def canonical_state_order(
    means: np.ndarray[Any, Any],
    direction_dim: int,
) -> tuple[list[str], np.ndarray[Any, Any]]:
    """Map raw EM state columns onto the fixed trend vocabulary.

    Baum-Welch is permutation-symmetric: nothing in the likelihood function
    ties column 0 to "downtrend".  A refit on a different window can therefore
    return the same three clusters under a different column order — and a gate
    reading ``state == 0`` would silently flip from "blocks bullish reversals"
    to "blocks bearish reversals" without any config change.  That is exactly
    the failure mode rework §2.4 must not ship with.

    Fix: rank the fitted states by their mean value on the direction feature
    (the normalized mean return) and pin the vocabulary by that rank —
    most-negative drift -> ``downtrend``, middle -> ``range``, most-positive
    -> ``uptrend``.  This ordering depends ONLY on the sign ordering of a
    fitted parameter (not on a threshold, not on the training window's exact
    statistics), so it is stable across refits by construction.

    Returns ``(state_names, order)`` where ``order`` is the permutation such
    that ``means[order]`` is in canonical order.
    """
    if means.ndim != 2 or means.shape[0] != 3:
        raise ValueError(
            f"canonical_state_order expects a (3, d) mean matrix, got {means.shape}"
        )
    if not 0 <= direction_dim < means.shape[1]:
        raise ValueError(
            f"direction_dim {direction_dim} out of range for {means.shape[1]} features"
        )
    drift = means[:, direction_dim]
    order = np.argsort(drift, kind="stable")
    # Degenerate solution: two states share the same drift sign/magnitude to
    # within numerical noise -> the states are not direction-distinguishable
    # and pinning an order would be arbitrary.  Fail loudly rather than ship a
    # coin-flip gate.
    ordered = drift[order]
    if abs(float(ordered[2]) - float(ordered[1])) < 1e-9 or abs(
        float(ordered[1]) - float(ordered[0])
    ) < 1e-9:
        raise ValueError(
            "degenerate trend fit: fitted state means are not distinguishable "
            f"on the direction feature (drift={drift.tolist()}) — cannot pin a "
            "stable downtrend/range/uptrend identity"
        )
    if not (float(ordered[0]) < 0.0 < float(ordered[2])):
        raise ValueError(
            "degenerate trend fit: the extreme-drift states do not straddle zero "
            f"(drift in canonical order={ordered.tolist()}) — the model did not "
            "learn a directional split, so a direction gate built on it would be "
            "meaningless"
        )
    return list(CANONICAL_STATE_NAMES), order


def _encode(arr: np.ndarray[Any, Any]) -> dict[str, Any]:
    import base64

    return {
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.astype(np.float64).tobytes()).decode("ascii"),
    }


def _decode(payload: dict[str, Any]) -> np.ndarray[Any, Any]:
    import base64

    shape = tuple(int(x) for x in payload["shape"])
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()
