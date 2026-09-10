"""
gaussian_hmm.py — CausalGaussianHMM regime plugin (guide v1.0 §2/§3, reqs §3.3)

Self-contained Gaussian HMM built on numpy + scipy.special.logsumexp +
sklearn.mixture.GaussianMixture (initialization only).  **No hmmlearn** —
hard constraint of the requirements (the venv has no hmmlearn and no new
dependency may be added).

No-lookahead guarantees (guide §2, checklist §7):
  * features are computed causally — ``log_return_1``, ``atr_14_norm`` and
    ``volume_zscore_20`` at bar ``t`` only use bars ``<= t``;
  * inference is strict forward filtering: state at ``t`` = argmax of
    ``P(s_t | x_1..x_t)``; there is no backward pass and no full-series
    retrodiction;
  * ``predict`` never refits; the scaling statistics are fixed at fit time so
    a prediction window can never leak fit-window-independent future info;
  * incremental == full: running ``predict`` on the whole frame equals running
    ``predict`` on every prefix and taking the per-bar result (the forward
    recursion only depends on past observations) — enforced by tests.

Learning: Baum-Welch (forward-backward EM) written in numpy with
``scipy.special.logsumexp``; emissions initialized via a seeded
``GaussianMixture`` on the causal-scaled train window.  Everything is
bit-reproducible given ``random_state`` + config (config_hash §6.2).
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
from scipy.special import logsumexp  # type: ignore[import-untyped]
from sklearn.mixture import GaussianMixture

from research.core.config_hash import canonical_json, compute_config_hash
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    PatternFeature,
)
from research.regime.base import (
    HMM_FEATURE_SCHEMA_VERSION,
    BaseRegimePlugin,
    RegimeState,
)

#: Canonical input-feature vocabulary (guide §6 / requirements §3.3).  These
#: are the features computed causally from the OHLCV frame when the frame does
#: not already carry the column.
CANONICAL_INPUT_FEATURES: tuple[str, str, str] = (
    "log_return_1",
    "atr_14_norm",
    "volume_zscore_20",
)

#: Default 3-state vocabulary (guide §3.2 / requirements §3.4).
DEFAULT_STATE_NAMES: list[str] = ["trending", "sideways", "high_vol"]

#: Columns required to compute the canonical features from raw OHLCV.
_OHLC_COLUMNS = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Causal feature computation (shared with the feature emitter, requirements
# §3.4 / guide §5) — every series at bar t depends only on bars <= t.
# ---------------------------------------------------------------------------


def _atr_norm(df: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
    """Wilder ATR(period) normalised by close — causal (ewm, no centering)."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / close


def _volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series[Any]:
    """Rolling (volume - mean)/std over the last ``period`` bars — causal."""
    vol = df["volume"].astype(np.float64)
    mean = vol.rolling(period, min_periods=period).mean()
    std = vol.rolling(period, min_periods=period).std()
    return (vol - mean) / std


def _log_return_1(df: pd.DataFrame) -> pd.Series[Any]:
    close = df["close"].astype(np.float64)
    ratio = (close / close.shift(1)).to_numpy(dtype=np.float64)
    return pd.Series(np.asarray(np.log(ratio)), index=df.index)


def compute_causal_input_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Build the input-feature matrix for the given names, keeping columns
    the caller already provides and computing the canonical ones from OHLCV.

    Every produced series is causal (rolling/ewm never centered, no
    ``shift(-1)``).  Unknown names that are not canonical raise ``ValueError``
    so a misconfigured pipeline fails loudly instead of silently leaking.
    """
    if not feature_names:
        raise ValueError("input_features must not be empty")
    series: dict[str, pd.Series[Any]] = {}
    for name in feature_names:
        if name in df.columns:
            series[name] = df[name].astype(np.float64)
        elif name == "log_return_1":
            series[name] = _log_return_1(df)
        elif name == "atr_14_norm":
            series[name] = _atr_norm(df)
        elif name == "volume_zscore_20":
            if "volume" not in df.columns:
                raise ValueError(
                    "input feature 'volume_zscore_20' requires a 'volume' "
                    "column in df — cannot compute it from OHLCV alone"
                )
            series[name] = _volume_zscore(df)
        else:
            raise ValueError(
                f"unknown input feature {name!r} — provide it as a df column "
                f"or use one of the canonical features {CANONICAL_INPUT_FEATURES}"
            )
    return pd.DataFrame(series)


def _index_version(index: pd.DatetimeIndex) -> str:
    """Deterministic 12-hex version of the frame index (data_version)."""
    payload = canonical_json([ts.isoformat() for ts in index])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _init_transition(n_states: int) -> np.ndarray[Any, Any]:
    """Diagonal-dominant start transition matrix (deterministic)."""
    trans = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
    np.fill_diagonal(trans, 0.9)
    if n_states == 1:
        trans[0, 0] = 1.0
    return trans.astype(np.float64)


# ---------------------------------------------------------------------------
# Gaussian log-emission helpers (vectorised over T; returns (T, K) log probs)
# ---------------------------------------------------------------------------

_LOG_2PI = np.log(2.0 * np.pi)


def _log_gaussian_emissions(
    Xs: np.ndarray[Any, Any],
    means: np.ndarray[Any, Any],
    covars: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Log-pdf of each row under each state's Gaussian, shape (T, K).

    ``covars`` is always stored as full (K, d, d) matrices even for
    ``covariance_type='diag'`` (off-diagonal zeros).
    """
    t_rows, d = Xs.shape
    k = means.shape[0]
    log_em = np.zeros((t_rows, k), dtype=np.float64)
    for m in range(k):
        cov = covars[m]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:  # degenerate guard — tiny jitter on the diagonal
            cov = cov + np.eye(d) * 1e-9
            _, logdet = np.linalg.slogdet(cov)
        diff = Xs - means[m]  # (T, d)
        sol = np.linalg.solve(cov, diff.T).T  # (T, d)
        mahal = np.einsum("td,td->t", diff, sol)
        log_em[:, m] = -0.5 * (d * _LOG_2PI + logdet) - 0.5 * mahal
    return log_em


def _encode_array(arr: np.ndarray[Any, Any]) -> dict[str, Any]:
    """JSON-safe encoding of a float64 numpy array (artifact persistence)."""
    return {
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.astype(np.float64).tobytes()).decode("ascii"),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray[Any, Any]:
    shape = tuple(int(x) for x in payload["shape"])
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()


class CausalGaussianHMM(BaseRegimePlugin):
    """Concrete causal Gaussian-HMM regime plugin (guide §3, requirements §3.3).

    * ``fit``   — Baum-Welch EM on the causal train window (seeded, config
      hashed §6.2; stores transition matrix, means, covars, scaling stats,
      data_version + model_version for the artifact).
    * ``predict`` — forward-filtered causal states for every closed bar; the
      recursion uses only observations ``<= t``; never refits; never
      backward-smooths; never runs a full-series decoder.
    * ``get_feature_schema`` — the 5 ``hmm_*`` features of requirements §3.4:
      ``hmm_state`` (int), ``hmm_prob_*`` (float), ``hmm_confidence`` (float),
      all ``available_at='confirm'`` (bar close) with ``uses_future_data=False``.
    """

    name = "hmm_regime"
    version = "1.0.0"
    short_key = "HMM"
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
    # Config handling (§6.2 — versioned, canonical, hashable)
    # ------------------------------------------------------------------

    def get_default_config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_states": 3,
            "state_names": list(DEFAULT_STATE_NAMES),
            "input_features": list(CANONICAL_INPUT_FEATURES),
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
        if n_states < 2:
            raise ValueError("n_states must be >= 2 (single-state regime is degenerate)")
        if len(state_names) != n_states or len(set(state_names)) != n_states:
            raise ValueError(
                f"state_names {state_names} must have exactly n_states={n_states} unique entries"
            )
        if not merged["input_features"]:
            raise ValueError("input_features must not be empty")
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
    # Frame validation (strictly-ascending timestamps = anti lookahead)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(df: pd.DataFrame) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")
        if not df.index.is_monotonic_increasing or not df.index.is_unique:
            raise ValueError(
                "df must be sorted strictly ascending by timestamp — a frame "
                "with future/duplicate stamps is rejected (timeline attack guard)"
            )
        missing = [c for c in _OHLC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required OHLC columns: {missing}")

    def _validate_input_columns(self, df: pd.DataFrame, input_features: Sequence[str]) -> None:
        for name in input_features:
            if name in df.columns:
                continue
            if name not in CANONICAL_INPUT_FEATURES:
                raise ValueError(
                    f"input feature {name!r} is neither a df column nor canonical "
                    f"{CANONICAL_INPUT_FEATURES}"
                )
            if name == "volume_zscore_20" and "volume" not in df.columns:
                raise ValueError(
                    "input feature 'volume_zscore_20' requires a 'volume' column"
                )

    # ------------------------------------------------------------------
    # fit — Baum-Welch EM on the causal train window (research/warm-up only)
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        config: dict[str, Any] | None = None,
    ) -> CausalGaussianHMM:
        cfg = self._resolve_config(config)
        self._validate_frame(df)
        input_features = list(cfg["input_features"])
        self._validate_input_columns(df, input_features)
        min_bars = int(cfg["min_fit_bars"])
        if len(df) < min_bars:
            raise ValueError(
                f"CausalGaussianHMM.fit: frame has {len(df)} bars but "
                f"min_fit_bars={min_bars} — train window too short for stable EM"
            )
        X = compute_causal_input_features(df, input_features)
        Xc = X.dropna()
        if len(Xc) < min_bars:
            raise ValueError(
                f"CausalGaussianHMM.fit: only {len(Xc)} causal rows after NaN "
                f"warmup but min_fit_bars={min_bars} is required"
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

        # --- persist fitted state + lineage metadata (guide §2 #5, §6.2) ---
        self._pi_ = pi
        self._transmat_ = trans
        self._means_ = means
        self._covars_ = covars
        self._feature_mean_ = feature_mean
        self._feature_std_ = safe_std
        self._config = dict(cfg)
        self._config_hash = compute_config_hash(cfg)
        dt_index = cast(pd.DatetimeIndex, df.index)
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
        dt_index = cast(pd.DatetimeIndex, df.index)
        cfg = self._resolve_config(config) if config is not None else self._config
        state_names = list(cfg["state_names"])
        lag = int(cfg["lag_bars"])
        n = len(df)
        if n < int(cfg["min_predict_bars"]):
            return [
                self._default_state(dt_index[i], lag, state_names) for i in range(n)
            ]
        input_features = list(cfg["input_features"])
        self._validate_input_columns(df, input_features)
        X = compute_causal_input_features(df, input_features)
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

        Rows without data (NaN features — e.g. warmup) carry the previous
        posterior forward: no observation, no regime update.
        """
        n_states = len(state_names)
        t_rows = Xs.shape[0]
        assert self._pi_ is not None and self._transmat_ is not None
        assert self._means_ is not None and self._covars_ is not None
        log_trans = np.log(self._transmat_ + 1e-300)
        log_pi = np.log(self._pi_ + 1e-300)
        log_em = self._log_emissions(Xs)

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

    def _log_emissions(self, Xs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._means_ is not None and self._covars_ is not None
        return _log_gaussian_emissions(Xs, self._means_, self._covars_)

    def _default_state(
        self,
        timestamp: pd.Timestamp,
        lag: int,
        state_names: list[str],
    ) -> RegimeState:
        """Stability guard state (requirements §3.3): state 0, confidence 0.5.

        Returned when the predict window is shorter than ``min_predict_bars``
        (a full default-state frame) or as the no-evidence anchor row; a
        required-feature consumer must fail closed (§6.3 — handled by the
        hard gate in t3).
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
                f"CausalGaussianHMM.{what}: model not fitted — call fit(df, config) first"
            )

    # ------------------------------------------------------------------
    # Feature schema (requirements §3.4)
    # ------------------------------------------------------------------

    def get_feature_schema(
        self,
        config: dict[str, Any] | None = None,
    ) -> list[PatternFeature]:
        cfg = self._resolve_config(config)
        state_names = list(cfg["state_names"])
        schema: list[PatternFeature] = [
            PatternFeature(
                name="hmm_state",
                dtype="int",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description="argmax filtered regime state tại bar đóng (known_at)",
            )
        ]
        schema.extend(
            PatternFeature(
                name=f"hmm_prob_{name}",
                dtype="float",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description=f"filtered posterior xác suất regime '{name}'",
            )
            for name in state_names
        )
        schema.append(
            PatternFeature(
                name="hmm_confidence",
                dtype="float",
                available_at=AVAILABLE_AT_CONFIRM,
                uses_future_data=False,
                description="max state probability (filtered posterior)",
            )
        )
        return schema

    # ------------------------------------------------------------------
    # Artifact persistence — JSON + base64 (no new deps; joblib untyped)
    # ------------------------------------------------------------------

    def to_artifact_dict(self) -> dict[str, Any]:
        """Complete reproducible artifact payload (guide §2 #5, requirements
        §3.3): config (config_hash §6.2), transition matrix, means, covars,
        scaling stats, data_version + model_version."""
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
            "params": {
                "pi": _encode_array(self._pi_),
                "transmat": _encode_array(self._transmat_),
                "means": _encode_array(self._means_),
                "covars": _encode_array(self._covars_),
                "feature_mean": _encode_array(self._feature_mean_),
                "feature_std": _encode_array(self._feature_std_),
            },
        }

    @classmethod
    def from_artifact_dict(cls, payload: dict[str, Any]) -> CausalGaussianHMM:
        plugin = cls()
        params = payload["params"]
        plugin._pi_ = _decode_array(params["pi"])
        plugin._transmat_ = _decode_array(params["transmat"])
        plugin._means_ = _decode_array(params["means"])
        plugin._covars_ = _decode_array(params["covars"])
        plugin._feature_mean_ = _decode_array(params["feature_mean"])
        plugin._feature_std_ = _decode_array(params["feature_std"])
        plugin._config = dict(payload["config"])
        plugin._config_hash = str(payload["config_hash"])
        plugin._data_version_ = str(payload["data_version"])
        plugin._fitted = True
        return plugin

    def save(self, path: str | Path) -> Path:
        """Persist the fitted plugin as a JSON artifact (round-trip safe)."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_artifact_dict(), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> CausalGaussianHMM:
        """Load a previously saved artifact back into a fitted plugin."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_artifact_dict(payload)