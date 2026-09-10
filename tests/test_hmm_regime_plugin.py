"""Tests for t2 — HMM regime plugin (BaseRegimePlugin + CausalGaussianHMM).

Coverage per requirements v1.0 §7 (basic unit + no-lookahead; the full
adversarial suite lives in t4):

  * causal forward-filtering: state at bar t uses only data <= t
    (delete-after equivalence, per-row feature causality);
  * incremental == full-run (guide §7 — per-bar predict == full predict);
  * strict timeline guard (future / duplicate stamps rejected);
  * reproducibility: same seed + config -> identical state sequence and
    config_hash (12 hex); hash changes when the config changes;
  * feature schema: 5 hmm_* features, uses_future_data=False,
    available_at='confirm' (bar close), feature_schema_version 'hmm-v1.0';
  * RegimeState contract: timestamp == bar close, lag declared + shifted;
  * fit guards (min_fit_bars) + predict guards (predict-before-fit raises,
    short window -> default stable states);
  * persistence: RegimeState parquet round-trip + JSON artifact round-trip;
  * NoLookaheadTestBase inheritance for the schema/event causality gate.

The synthetic generator emits genuinely regime-structured OHLCV (3 states,
different drift/vol) so EM has a real signal to learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from research.core.causal_checks import validate_causality
from research.core.contracts import AVAILABLE_AT_CONFIRM, PatternEvent
from research.regime import (
    CausalGaussianHMM,
    compute_causal_input_features,
    load_hmm_regime_config,
    regime_states_from_parquet,
    regime_states_to_parquet,
)
from research.regime.base import (
    HMM_FEATURE_SCHEMA_VERSION,
    BaseRegimePlugin,
)
from tests.no_lookahead_base import NoLookaheadTestBase

# ---------------------------------------------------------------------------
# Synthetic regime-structured OHLCV generator
# ---------------------------------------------------------------------------

_STATE_NAMES = ["trending", "sideways", "high_vol"]
_TEST_CFG = {
    "n_iter": 40,
    "random_state": 42,
    "reg_covar": 1e-4,
    "min_fit_bars": 500,
}


def make_regime_ohlcv(n: int = 2500, seed: int = 11) -> tuple[pd.DataFrame, np.ndarray]:
    """Three-regime synthetic M15 frame: trending (0), sideways (1),
    high_vol (2) with clearly separated drift/volatility."""
    rng = np.random.default_rng(seed)
    trans = np.array(
        [[0.98, 0.015, 0.005], [0.03, 0.95, 0.02], [0.03, 0.02, 0.95]],
        dtype=np.float64,
    )
    drift = np.array([0.0006, 0.0, 0.001])
    vol = np.array([0.0008, 0.0004, 0.0022])
    states = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        states[t] = int(rng.choice(3, p=trans[int(states[t - 1])]))
    rets = drift[states] + vol[states] * rng.standard_normal(n)
    close = 1800.0 * np.exp(np.cumsum(rets))
    open_p = np.empty(n, dtype=np.float64)
    open_p[0] = close[0]
    open_p[1:] = close[:-1]
    span = (np.abs(rets) + 0.0004) * close
    high = np.maximum(open_p, close) + span
    low = np.minimum(open_p, close) - span
    volume = 100.0 * (1.0 + 0.5 * np.abs(rng.standard_normal(n)))
    index = pd.date_range("2020-01-01 00:00", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return df, states


@pytest.fixture()
def df() -> pd.DataFrame:
    return make_regime_ohlcv(n=2500, seed=11)[0]


def fitted_plugin(df: pd.DataFrame, cfg: dict | None = None) -> CausalGaussianHMM:
    return CausalGaussianHMM().fit(df, cfg or dict(_TEST_CFG))


# ---------------------------------------------------------------------------
# Plugin contract (requirements §3.2, acceptance 1)
# ---------------------------------------------------------------------------


class TestPluginContract:
    def test_interface_shape(self) -> None:
        plugin = CausalGaussianHMM()
        assert isinstance(plugin, BaseRegimePlugin)
        assert plugin.name == "hmm_regime"
        assert plugin.version == "1.0.0"
        assert plugin.short_key == "HMM"
        assert plugin.feature_schema_version == HMM_FEATURE_SCHEMA_VERSION

    def test_default_config_contract(self) -> None:
        cfg = CausalGaussianHMM().get_default_config()
        assert cfg["version"] == CausalGaussianHMM.version
        assert cfg["n_states"] == 3
        assert list(cfg["state_names"]) == ["trending", "sideways", "high_vol"]
        assert cfg["input_features"] == ["log_return_1", "atr_14_norm", "volume_zscore_20"]
        assert cfg["lag_bars"] == 0
        assert cfg["min_confidence"] == 0.50
        assert cfg["covariance_type"] == "full"
        assert isinstance(cfg["n_iter"], int) and isinstance(cfg["random_state"], int)
        assert cfg["min_fit_bars"] >= 1 and cfg["min_predict_bars"] >= 0

    def test_default_config_hashes(self) -> None:
        from research.core.config_hash import compute_config_hash

        cfg = CausalGaussianHMM().get_default_config()
        h = compute_config_hash(cfg)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Feature schema (requirements §3.4, acceptance 4)
# ---------------------------------------------------------------------------


class TestFeatureSchema:
    def test_schema_contract(self) -> None:
        schema = CausalGaussianHMM().get_feature_schema()
        names = [f.name for f in schema]
        assert names == [
            "hmm_state",
            "hmm_prob_trending",
            "hmm_prob_sideways",
            "hmm_prob_high_vol",
            "hmm_confidence",
        ]
        by_name = {f.name: f for f in schema}
        assert by_name["hmm_state"].dtype == "int"
        for n in ("hmm_prob_trending", "hmm_prob_sideways", "hmm_prob_high_vol", "hmm_confidence"):
            assert by_name[n].dtype == "float"
        for f in schema:
            assert f.available_at == AVAILABLE_AT_CONFIRM
            assert f.uses_future_data is False

    def test_schema_dynamic_state_names(self) -> None:
        schema = CausalGaussianHMM().get_feature_schema({"n_states": 2, "state_names": ["a", "b"]})
        assert [f.name for f in schema] == [
            "hmm_state",
            "hmm_prob_a",
            "hmm_prob_b",
            "hmm_confidence",
        ]

    def test_get_feature_names(self) -> None:
        plugin = CausalGaussianHMM()
        assert plugin.get_feature_names() == [
            "hmm_state",
            "hmm_prob_trending",
            "hmm_prob_sideways",
            "hmm_prob_high_vol",
            "hmm_confidence",
        ]

    def test_schema_events_validate_causality(self, df: pd.DataFrame) -> None:
        """The full runtime validator accepts hmm_* features stamped at bar
        close (requirements §3.4 — available_at=confirm mapped to known_at)."""
        schema = CausalGaussianHMM().get_feature_schema()
        events: list[PatternEvent] = []
        for i in (200, 500, 800):
            ts = df.index[i]
            events.append(
                PatternEvent(
                    event_id=f"ev{i}",
                    pattern_name="synthetic_hmm",
                    pattern_version="1.0",
                    symbol="TEST",
                    timeframe="M15",
                    direction="bullish",
                    detect_time=ts,
                    confirm_time=ts,
                    entry_time=ts,
                )
            )
        validate_causality(events, feature_schema=schema)  # must not raise


# ---------------------------------------------------------------------------
# fit + causal predict (requirements §3.3, acceptance 2/3/5)
# ---------------------------------------------------------------------------


class TestFitPredict:
    def test_fit_predict_alignment(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        assert len(states) == len(df)
        assert all(s.timestamp == df.index[i] for i, s in enumerate(states))
        for s in states:
            assert 0 <= s.state < 3
            assert s.state_name == _STATE_NAMES[s.state]
            assert set(s.state_prob) == set(_STATE_NAMES)
            assert abs(sum(s.state_prob.values()) - 1.0) < 1e-6
            assert abs(s.confidence - max(s.state_prob.values())) < 1e-9
            assert s.lag_bars == 0
            assert s.model_version == "1.0.0"
            assert len(s.config_hash) == 12

    def test_state_timestamp_is_bar_close(self, df: pd.DataFrame) -> None:
        states = fitted_plugin(df).predict(df)
        assert all(s.timestamp == df.index[i] for i, s in enumerate(states))

    def test_states_not_degenerate(self, df: pd.DataFrame) -> None:
        """EM must actually separate regimes on structured data."""
        states = fitted_plugin(df).predict(df)
        distinct = {s.state for s in states}
        assert len(distinct) >= 2

    def test_fit_min_bars_guard(self, df: pd.DataFrame) -> None:
        short = df.iloc[:100]
        with pytest.raises(ValueError, match="min_fit_bars"):
            CausalGaussianHMM().fit(short, dict(_TEST_CFG))

    def test_predict_before_fit_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            CausalGaussianHMM().predict(df)

    def test_predict_short_window_default_states(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        states = plugin.predict(df.iloc[:50])  # 50 < min_predict_bars=200
        assert len(states) == 50
        for s in states:
            assert s.state == 0
            assert s.state_name == "trending"
            assert s.confidence == 0.5
            assert s.lag_bars == 0
            assert abs(sum(s.state_prob.values()) - 1.0) < 1e-9

    def test_predict_empty_frame_returns_empty(self, df: pd.DataFrame) -> None:
        assert fitted_plugin(df).predict(df.iloc[:0]) == []

    def test_lag_declared_and_shifted(self, df: pd.DataFrame) -> None:
        cfg = {**_TEST_CFG, "lag_bars": 2}
        states = fitted_plugin(df, cfg).predict(df)
        assert all(s.lag_bars == 2 for s in states)
        for i, s in enumerate(states):
            assert s.timestamp == df.index[max(0, i - 2)]


# ---------------------------------------------------------------------------
# No-lookahead (guide §7 checklist — basic matrix, §7 #1/#2/#4/#5)
# ---------------------------------------------------------------------------


@pytest.mark.no_lookahead
class TestNoLookaheadCausality:
    def test_incremental_equals_full(self, df: pd.DataFrame) -> None:
        """guide §7 #4 — per-bar predict == full predict (all bars)."""
        plugin = fitted_plugin(df)
        full = plugin.predict(df)
        for t in (300, 700, 1250, 2200):
            prefix = plugin.predict(df.iloc[: t + 1])
            assert len(prefix) == t + 1
            for i in range(t + 1):
                assert prefix[i].timestamp == full[i].timestamp
                assert prefix[i].state == full[i].state
                assert prefix[i].state_name == full[i].state_name
                assert abs(prefix[i].confidence - full[i].confidence) < 1e-9
            assert prefix[-1].state == full[t].state

    def test_state_at_t_not_affected_by_future_bars(self, df: pd.DataFrame) -> None:
        """§7 #1 — deleting every bar AFTER t must leave states <= t identical."""
        plugin = fitted_plugin(df)
        full = plugin.predict(df)
        for t in (400, 1200, 2000):
            suffix_deleted = plugin.predict(df.iloc[: t + 1])
            for i in range(t + 1):
                assert suffix_deleted[i].state == full[i].state
                assert suffix_deleted[i].timestamp == full[i].timestamp
                assert abs(suffix_deleted[i].confidence - full[i].confidence) < 1e-9

    def test_no_refit_on_predict(self, df: pd.DataFrame) -> None:
        """§7 #2 — predict() never changes the fitted parameters."""
        plugin = fitted_plugin(df)
        trans_0 = plugin.transmat.copy()
        means_0 = plugin.means.copy()
        covars_0 = plugin.covars.copy()
        _ = plugin.predict(df)
        assert np.array_equal(plugin.transmat, trans_0)
        assert np.array_equal(plugin.means, means_0)
        assert np.array_equal(plugin.covars, covars_0)

    def test_timeline_attack_unsorted_rejected(self, df: pd.DataFrame) -> None:
        """§7 #5 — a frame with a future stamp inserted mid-sequence (unsorted)
        must be refused: forward filtering would be meaningless."""
        plugin = fitted_plugin(df)
        shuffled = df.sample(frac=1.0, random_state=5)
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(shuffled)

    def test_timeline_attack_duplicate_stamp_rejected(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        dup = pd.concat([df, df.iloc[[100]]])
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(dup)

    def test_hmm_feature_causal_per_row(self, df: pd.DataFrame) -> None:
        """Every input feature row t depends only on bars <= t: recomputing on
        the prefix must reproduce the identical row."""
        names = ["log_return_1", "atr_14_norm", "volume_zscore_20"]
        full = compute_causal_input_features(df, names)
        for t in (150, 900, 2200):
            prefix = compute_causal_input_features(df.iloc[: t + 1], names)
            pdt.assert_series_equal(
                full.iloc[t],
                prefix.iloc[t],
                check_exact=False,
                rtol=1e-10,
                check_names=False,
            )

    def test_missing_volume_rejected_for_zscore(self, df: pd.DataFrame) -> None:
        nodf = df.drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            CausalGaussianHMM().fit(nodf, dict(_TEST_CFG))


# ---------------------------------------------------------------------------
# Reproducibility (acceptance 3)
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_state_sequence(self, df: pd.DataFrame) -> None:
        p1 = fitted_plugin(df)
        p2 = fitted_plugin(df)
        s1 = np.array([s.state for s in p1.predict(df)])
        s2 = np.array([s.state for s in p2.predict(df)])
        assert np.array_equal(s1, s2)
        assert p1.config_hash == p2.config_hash
        assert np.array_equal(p1.transmat, p2.transmat)
        assert np.array_equal(p1.means, p2.means)

    def test_config_hash_12hex_and_sensitive(self, df: pd.DataFrame) -> None:
        p1 = fitted_plugin(df)
        assert len(p1.config_hash) == 12
        assert all(c in "0123456789abcdef" for c in p1.config_hash)
        p2 = CausalGaussianHMM().fit(df, {**_TEST_CFG, "random_state": 99})
        assert p2.config_hash != p1.config_hash
        assert p2.data_version == p1.data_version  # same index, same data version

    def test_data_version_tracks_index(self, df: pd.DataFrame) -> None:
        p1 = fitted_plugin(df)
        p2 = fitted_plugin(df.iloc[:2000])
        assert p2.data_version != p1.data_version


# ---------------------------------------------------------------------------
# Persistence (guide §5 — RegimeState parquet; artifact round-trip)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_parquet_roundtrip_states(self, df: pd.DataFrame, tmp_path: Path) -> None:
        states = fitted_plugin(df).predict(df)
        target = tmp_path / "regime_states.parquet"
        regime_states_to_parquet(states, target)
        loaded = regime_states_from_parquet(target)
        assert len(loaded) == len(states)
        for a, b in zip(states, loaded):
            assert b.timestamp == a.timestamp
            assert b.state == a.state
            assert b.state_name == a.state_name
            assert b.state_prob == a.state_prob
            assert abs(b.confidence - a.confidence) < 1e-12
            assert b.lag_bars == a.lag_bars
            assert b.model_version == a.model_version
            assert b.config_hash == a.config_hash

    def test_artifact_json_roundtrip(self, df: pd.DataFrame, tmp_path: Path) -> None:
        p1 = fitted_plugin(df)
        target = tmp_path / "regime_artifact.json"
        p1.save(target)
        assert target.is_file()
        p2 = CausalGaussianHMM.load(target)
        s1 = [s.state for s in p1.predict(df)]
        s2 = [s.state for s in p2.predict(df)]
        assert s1 == s2
        assert p2.config_hash == p1.config_hash
        assert p2.data_version == p1.data_version
        assert np.allclose(p2.transmat, p1.transmat)
        assert np.allclose(p2.means, p1.means)


# ---------------------------------------------------------------------------
# Sample config YAML (requirements §9)
# ---------------------------------------------------------------------------

CONFIG_YAML = Path(__file__).resolve().parent.parent / "configs" / "plugins" / "hmm_regime.yaml"


class TestSampleConfig:
    def test_sample_config_matches_defaults(self) -> None:
        assert CONFIG_YAML.is_file()
        cfg = load_hmm_regime_config(CONFIG_YAML)
        defaults = CausalGaussianHMM().get_default_config()
        for key in (
            "n_states",
            "state_names",
            "input_features",
            "lag_bars",
            "min_confidence",
            "covariance_type",
            "n_iter",
            "random_state",
            "min_fit_bars",
            "min_predict_bars",
        ):
            assert key in cfg, f"sample config missing {key}"
            assert cfg[key] == defaults[key], f"sample config {key} diverges from defaults"

    def test_sample_config_merges_into_fit(self, df: pd.DataFrame) -> None:
        cfg = load_hmm_regime_config(CONFIG_YAML)
        cfg["min_fit_bars"] = 500  # fit guard adapted for the small test frame
        plugin = CausalGaussianHMM().fit(df, cfg)
        assert len(plugin.config_hash) == 12
        assert plugin.predict(df)[-1].state_name in _STATE_NAMES


# ---------------------------------------------------------------------------
# NoLookaheadTestBase inheritance (requirements §7 #7 / spec §3.4)
# ---------------------------------------------------------------------------


class TestHmmRegimeSchemaCausality(NoLookaheadTestBase):
    """Inherited tests verify: events timeline coherent, full validate_causality
    accepts the batch, every hmm_* feature has uses_future_data=False and a
    stamp <= known_at (bar close)."""

    feature_schema = CausalGaussianHMM().get_feature_schema()

    def build_events(self) -> list[PatternEvent]:
        frame, _ = make_regime_ohlcv(n=1500, seed=3)
        events: list[PatternEvent] = []
        for i in (300, 600, 900, 1200):
            ts = frame.index[i]
            events.append(
                PatternEvent(
                    event_id=f"hmm_ev{i}",
                    pattern_name="synthetic_hmm",
                    pattern_version="1.0",
                    symbol="TEST",
                    timeframe="M15",
                    direction="bullish",
                    detect_time=ts,
                    confirm_time=ts,
                    entry_time=ts,
                )
            )
        return events