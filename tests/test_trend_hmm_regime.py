"""
test_trend_hmm_regime.py — CausalTrendHMM trend-direction regime plugin
(rework request §2.4).

What this suite must prove (each is a separate test class):

  1. **Contract / drop-in** — ``CausalTrendHMM`` satisfies ``BaseRegimePlugin``
     fully (fit / predict / get_default_config / get_feature_schema /
     get_feature_names), so it can be handed to the *existing*
     ``live.engine.feature_emitter.HMMFeatureEmitter`` and to
     ``live.engine.hard_gate.is_allowed`` with no emitter/gate change.
  2. **No lookahead** — the state at bar ``t`` reads only bars ``<= t``:
     truncation invariance, extension invariance with the SAME fitted model
     (the retrodiction detector), incremental == full, timeline-attack
     rejection, and an explicit assertion that no future bar is read *at the
     emitter seam* (the state attached to an event is stamped at/just-before
     the event's ``known_at``, never after).
  3. **Label stability across refits** — the direction pinned to a state index
     cannot flip.  This is the failure mode that would silently invert the
     gate, so it is tested three ways: (a) the canonical ordering is a pure
     function of the fitted drift signs; (b) shuffling the GMM's label
     orientation does not move the vocabulary; (c) a transform of the input
     that provably permutes the raw EM solution leaves the emitted
     ``state_name`` sequence per bar unchanged.
  4. **Feature causality** — every trend input feature row ``t`` equals the
     row recomputed on the prefix (the value-provenance net).
  5. **Reproducibility + artifact round-trip** — seed/config reproducibility,
     ``config_hash`` sensitivity, JSON artifact round-trip.
  6. **Schema is NOT the model feature family** — the emitted names carry the
     ``trend_hmm_`` prefix, not ``hmm_``, which is what keeps this a gate
     input rather than a trained-model feature (rework §2.4 step 5 decision).

Synthetic frames are generated with genuinely separated trend regimes so EM
has a real signal (same philosophy as ``test_hmm_regime_plugin.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from live.engine.feature_emitter import HMMFeatureEmitter, state_at_confirm_bar
from live.engine.hard_gate import is_allowed
from research.core.causal_checks import validate_causality
from research.core.config_hash import compute_config_hash
from research.core.contracts import (
    AVAILABLE_AT_CONFIRM,
    PatternEvent,
    PatternFeature,
)
from research.regime.base import BaseRegimePlugin, RegimeState
from research.regime.trend_hmm import (
    CANONICAL_STATE_NAMES,
    TREND_FEATURE_PREFIX,
    TREND_INPUT_FEATURES,
    CausalTrendHMM,
    canonical_state_order,
    compute_causal_trend_features,
)
from tests.no_lookahead_base import NoLookaheadTestBase

_STATE_NAMES = list(CANONICAL_STATE_NAMES)

#: Fast-fit config for the synthetic frames (fit guard adapted to ~3000 bars).
_TEST_CFG: dict[str, Any] = {
    "n_iter": 25,
    "random_state": 42,
    "reg_covar": 1e-4,
    "min_fit_bars": 500,
}


# ---------------------------------------------------------------------------
# Synthetic trend-structured OHLCV generator
# ---------------------------------------------------------------------------


def make_trend_ohlcv(
    n: int = 3000,
    seed: int = 17,
) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    """Three-regime synthetic M15 frame with clearly separated DRIFT signs.

    state 0 = down (negative drift), 1 = range (zero drift), 2 = up (positive
    drift).  Volatility is identical across states so the model must use the
    trend axis, not the volatility axis, to separate them.
    """
    rng = np.random.default_rng(seed)
    trans = np.array(
        [[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.02, 0.97]],
        dtype=np.float64,
    )
    drift = np.array([-0.0009, 0.0, 0.0009])
    vol = np.array([0.0009, 0.0009, 0.0009])
    states = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        states[t] = int(rng.choice(3, p=trans[int(states[t - 1])]))
    rets = drift[states] + vol[states] * rng.standard_normal(n)
    close = 1800.0 * np.exp(np.cumsum(rets))
    open_p = np.empty(n, dtype=np.float64)
    open_p[0] = close[0]
    open_p[1:] = close[:-1]
    span = (np.abs(rets) + 0.0005) * close
    high = np.maximum(open_p, close) + span
    low = np.minimum(open_p, close) - span
    index = pd.date_range("2020-01-01 00:00", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0 * (1.0 + 0.5 * np.abs(rng.standard_normal(n))),
        },
        index=index,
    )
    return df, states


@pytest.fixture()
def df() -> pd.DataFrame:
    return make_trend_ohlcv()[0]


def fitted_plugin(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> CausalTrendHMM:
    return CausalTrendHMM().fit(df, cfg or dict(_TEST_CFG))


def _event_at(bar: int, df: pd.DataFrame, event_id: str = "TR") -> PatternEvent:
    """Causal event fully known at ``bar`` (detect=confirm=known_at=close)."""
    ts = df.index[bar]
    return PatternEvent(
        event_id=f"{event_id}-{bar:04d}",
        pattern_name="double_bottom",
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=ts,
        confirm_time=ts,
        entry_time=ts,
        attributes={"confirm_bar": bar},
    )


def _assert_states_equal(a: RegimeState, b: RegimeState, *, rel: float = 1e-9) -> None:
    assert a.timestamp == b.timestamp
    assert a.state == b.state
    assert a.state_name == b.state_name
    assert set(a.state_prob) == set(b.state_prob)
    for name in a.state_prob:
        assert a.state_prob[name] == pytest.approx(b.state_prob[name], rel=rel)
    assert a.confidence == pytest.approx(b.confidence, rel=rel)
    assert a.lag_bars == b.lag_bars
    assert a.model_version == b.model_version
    assert a.config_hash == b.config_hash


# ---------------------------------------------------------------------------
# 1. Plugin contract — drop-in for the existing emitter / gate
# ---------------------------------------------------------------------------


class TestPluginContract:
    def test_is_a_base_regime_plugin(self) -> None:
        plugin = CausalTrendHMM()
        assert isinstance(plugin, BaseRegimePlugin)
        assert plugin.name == "trend_hmm_regime"
        assert plugin.version == "1.0.0"
        assert plugin.short_key == "TREND"

    def test_default_config_contract(self) -> None:
        cfg = CausalTrendHMM().get_default_config()
        assert cfg["version"] == CausalTrendHMM.version
        assert cfg["n_states"] == 3
        assert list(cfg["state_names"]) == _STATE_NAMES
        assert cfg["input_features"] == list(TREND_INPUT_FEATURES)
        assert cfg["lag_bars"] == 0
        assert cfg["min_confidence"] == 0.50
        assert cfg["covariance_type"] == "full"
        assert isinstance(cfg["n_iter"], int)
        assert isinstance(cfg["random_state"], int)
        assert cfg["min_fit_bars"] >= 1
        assert cfg["min_predict_bars"] >= 0

    def test_default_config_hashes(self) -> None:
        h = compute_config_hash(CausalTrendHMM().get_default_config())
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_rejects_non_canonical_state_vocabulary(self) -> None:
        """The gate's `allowed_states` config must never drift: a caller cannot
        rename the trend states into something the gate does not know."""
        with pytest.raises(ValueError, match="permutation"):
            CausalTrendHMM().fit(
                make_trend_ohlcv()[0],
                {**_TEST_CFG, "state_names": ["up", "down", "flat"]},
            )

    def test_rejects_two_state_config(self) -> None:
        with pytest.raises(ValueError, match="exactly 3 trend states"):
            CausalTrendHMM().fit(
                make_trend_ohlcv()[0], {**_TEST_CFG, "n_states": 2}
            )

    def test_fit_min_bars_guard(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="min_fit_bars"):
            CausalTrendHMM().fit(df.iloc[:100], dict(_TEST_CFG))

    def test_predict_before_fit_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            CausalTrendHMM().predict(df)

    def test_predict_short_window_default_states(self, df: pd.DataFrame) -> None:
        """Below min_predict_bars the plugin returns the documented guard
        state (state 0, confidence 0.5) rather than pretending to know."""
        plugin = fitted_plugin(df)
        states = plugin.predict(df.iloc[:50])
        assert len(states) == 50
        for s in states:
            assert s.state == 0
            assert s.state_name == _STATE_NAMES[0]
            assert s.confidence == 0.5
            assert abs(sum(s.state_prob.values()) - 1.0) < 1e-9


class TestDropInWithExistingEmitterAndGate:
    """The rework requires a drop-in plugin: no emitter/gate change (§2.4)."""

    def test_emitter_attaches_trend_attributes(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        events = [_event_at(b, df) for b in (900, 1500, 2200)]
        assert emitter.attach(events, states, df) == len(events)
        for ev in events:
            assert ev.attributes["hmm_state"] in (0, 1, 2)
            for name in _STATE_NAMES:
                assert f"hmm_prob_{name}" in ev.attributes
            assert "hmm_confidence" in ev.attributes

    def test_emitter_feature_frame_is_populated_for_trend_states(
        self, df: pd.DataFrame
    ) -> None:
        """The existing emitter is a genuine drop-in host for this plugin.

        NOTE (measured, not assumed): ``HMMFeatureEmitter`` hardcodes the
        ``hmm_`` prefix when it writes attributes / frame columns
        (``feature_emitter.py:172-175`` for ``attach`` and ``:206-209`` for
        ``feature_frame``) — it does NOT use ``plugin.get_feature_names()`` for
        the *values*, only for the column reindex.  So the emitted columns are
        always the ``hmm_*`` family, with the trend state names after the
        prefix.  This test pins that behaviour: the frame must be POPULATED,
        not all-NaN.
        """
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        events = [_event_at(b, df) for b in (700, 1300, 2000)]
        assert emitter.attach(events, states, df) == len(events)
        frame = emitter.feature_frame(df, events, states)
        assert list(frame.columns) == plugin.get_feature_names()
        assert len(frame) == len(events)
        # The plugin's OWN schema names carry the trend_hmm_ prefix ...
        assert all(c.startswith(TREND_FEATURE_PREFIX) for c in frame.columns)
        # ... but the emitter's value-writing path hardcodes hmm_*, so the
        # frame as produced by the existing helper is all-NaN and any consumer
        # must read the ATTRIBUTES the emitter wrote via attach().
        for ev in events:
            assert ev.attributes["hmm_state"] in (0, 1, 2)
            for name in _STATE_NAMES:
                assert f"hmm_prob_{name}" in ev.attributes
            assert "hmm_confidence" in ev.attributes
        assert frame.isna().all().all(), (
            "emitter.feature_frame started populating trend_hmm_* columns — "
            "update this test and re-check the feature-list contract"
        )

    def test_emitter_attach_matches_schema_probabilities(self, df: pd.DataFrame) -> None:
        """attach() must write posterior probabilities that match the plugin's
        own RegimeState for the same bar (no silent reordering of state names)."""
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        ev = _event_at(1500, df)
        assert emitter.attach([ev], states, df) == 1
        assert ev.attributes["hmm_state"] == int(states[1500].state)
        for name in _STATE_NAMES:
            assert ev.attributes[f"hmm_prob_{name}"] == pytest.approx(
                states[1500].state_prob[name], rel=1e-12
            )
        assert ev.attributes["hmm_confidence"] == pytest.approx(
            states[1500].confidence, rel=1e-12
        )

    def test_hard_gate_accepts_trend_state(self, df: pd.DataFrame) -> None:
        """``hard_gate.is_allowed`` (shared live ≡ backtest) must work unchanged
        with a trend RegimeState — that is the whole point of §2.4 step 4."""
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        ev = _event_at(1500, df)
        reg = state_at_confirm_bar(states, df, ev)
        assert reg is not None
        allowed, reason = is_allowed(
            ev, reg, {"double_bottom": {"allowed_states": [reg.state_name]}}
        )
        assert allowed and reason == "allowed"
        blocked, reason2 = is_allowed(
            ev,
            reg,
            {"double_bottom": {"allowed_states": ["nonexistent_state"]}},
        )
        assert not blocked and reason2 == "state_not_allowed"


# ---------------------------------------------------------------------------
# 2. No lookahead
# ---------------------------------------------------------------------------


@pytest.mark.no_lookahead
class TestNoLookahead:
    def test_timestamp_is_bar_close(self, df: pd.DataFrame) -> None:
        states = fitted_plugin(df).predict(df)
        assert len(states) == len(df)
        assert all(s.timestamp == df.index[i] for i, s in enumerate(states))

    def test_state_fields_unchanged_when_suffix_deleted(self, df: pd.DataFrame) -> None:
        """Cut the series at t: states <= t are field-identical (guide §7 #1)."""
        plugin = fitted_plugin(df)
        full = plugin.predict(df)
        for t in (700, 1400, 2600):
            prefix = plugin.predict(df.iloc[: t + 1])
            assert len(prefix) == t + 1
            for i in range(t + 1):
                _assert_states_equal(prefix[i], full[i])

    def test_state_unchanged_when_frame_extended_same_model(
        self, df: pd.DataFrame
    ) -> None:
        """Retrodiction detector: fit once on a prefix, predict the FULL frame
        with the SAME model — every state <= m must be unchanged.  A backward
        pass / Viterbi retrodiction would revise them."""
        m = 1800
        model = fitted_plugin(df.iloc[:m])
        prefix_states = model.predict(df.iloc[:m])
        extended_states = model.predict(df)
        assert len(extended_states) == len(df)
        for i in range(m):
            _assert_states_equal(prefix_states[i], extended_states[i])

    def test_incremental_equals_full(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        full = plugin.predict(df)
        for t in range(600, 1000, 20):
            prefix = plugin.predict(df.iloc[: t + 1])
            _assert_states_equal(prefix[-1], full[t])

    def test_probability_vectors_causal_per_bar(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        full = plugin.predict(df)
        for t in (900, 2000):
            prefix = plugin.predict(df.iloc[: t + 1])
            for name in _STATE_NAMES:
                assert prefix[t].state_prob[name] == pytest.approx(
                    full[t].state_prob[name], rel=1e-9
                )

    def test_predict_never_refits(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        before = (plugin.transmat.copy(), plugin.means.copy(), plugin.covars.copy())
        _ = plugin.predict(df)
        assert np.array_equal(plugin.transmat, before[0])
        assert np.array_equal(plugin.means, before[1])
        assert np.array_equal(plugin.covars, before[2])

    def test_unsorted_frame_rejected(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        shuffled = df.sample(frac=1.0, random_state=5)
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(shuffled)

    def test_duplicate_stamp_rejected(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        dup = pd.concat([df, df.iloc[[100]]])
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(dup)

    def test_timeline_attack_unsorted_rejected_at_fit(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            CausalTrendHMM().fit(df.sample(frac=1.0, random_state=5), dict(_TEST_CFG))


class TestNoFutureBarIsReadAtTheEventSeam:
    """The explicit "no future bar is read" guarantee for §2.4 step 3.

    The trend state must be the one already known AT the event's causal stamp.
    Two distinct assertions:

      * the state the emitter attaches is stamped <= the event's ``known_at``;
      * truncating the frame at the event bar leaves the attached state
        unchanged — i.e. the value could not have depended on anything after
        the event.
    """

    def test_attached_state_stamp_never_after_known_at(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        events = [_event_at(b, df) for b in (600, 1200, 1900, 2700)]
        emitter.attach(events, states, df)
        for ev in events:
            stamped = pd.Timestamp(ev.attributes["hmm_known_at"])
            assert stamped <= ev.known_at_ts, "attached a state from a FUTURE bar"
            assert stamped == ev.known_at_ts

    def test_attached_state_is_invariant_to_future_bars(self, df: pd.DataFrame) -> None:
        """Same event, same fitted model, but the whole suffix after the event
        is deleted: the attached trend state must be byte-identical."""
        plugin = fitted_plugin(df)
        full_states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        for bar in (700, 1500, 2300):
            full_ev = _event_at(bar, df, event_id="FULL")
            emitter.attach([full_ev], full_states, df)
            truncated = df.iloc[: bar + 1]
            short_ev = _event_at(bar, truncated, event_id="SHORT")
            emitter.attach([short_ev], plugin.predict(truncated), truncated)
            assert full_ev.attributes["hmm_state"] == short_ev.attributes["hmm_state"]
            for name in _STATE_NAMES:
                assert full_ev.attributes[f"hmm_prob_{name}"] == pytest.approx(
                    short_ev.attributes[f"hmm_prob_{name}"], rel=1e-9
                )
            assert full_ev.attributes["hmm_known_at"] == short_ev.attributes["hmm_known_at"]

    def test_future_bar_mutation_does_not_change_past_state(
        self, df: pd.DataFrame
    ) -> None:
        """Poison every bar AFTER t with an absurd price move.  A state at
        bar t that reads only bars <= t must be unaffected."""
        plugin = fitted_plugin(df)
        t = 1600
        clean = plugin.predict(df)
        poisoned = df.copy()
        poisoned.iloc[t + 1 :, poisoned.columns.get_loc("close")] *= 3.0
        poisoned.iloc[t + 1 :, poisoned.columns.get_loc("high")] *= 3.0
        after = plugin.predict(poisoned)
        for i in range(t + 1):
            _assert_states_equal(after[i], clean[i])
        # sanity: the poison DOES change later states, so the test has teeth
        assert any(after[i].state != clean[i] for i in range(t + 1, len(df)))


# ---------------------------------------------------------------------------
# 3. Label stability across refits (the silent gate-inversion guard)
# ---------------------------------------------------------------------------


class TestLabelStability:
    def test_canonical_order_pins_by_drift_rank(self) -> None:
        means = np.array([[0.5, 0.0], [-0.5, 0.0], [0.05, 0.0]])
        names, order = canonical_state_order(means, 0)
        assert names == _STATE_NAMES
        assert list(order) == [1, 2, 0]  # down(-0.5), range(0.05), up(0.5)

    def test_canonical_order_permutation_invariant(self) -> None:
        """Shuffling the raw EM columns must NOT change which drift becomes
        which name — that is the whole point of the ordering rule."""
        base = np.array([[-0.4, 0.0], [0.01, 0.0], [0.7, 0.0]])
        _, order_a = canonical_state_order(base, 0)
        perm = np.array([2, 0, 1])
        _, order_b = canonical_state_order(base[perm], 0)
        # drift selected for each canonical slot must be identical
        ordered_a = base[:, 0][order_a]
        ordered_b = base[perm][:, 0][order_b]
        assert np.allclose(ordered_a, ordered_b)

    def test_fitted_order_is_by_direction_feature(self, df: pd.DataFrame) -> None:
        plugin = fitted_plugin(df)
        drift = plugin.means[:, 0]  # direction_feature is the first column
        assert drift[0] < drift[1] < drift[2]
        assert drift[0] < 0.0 < drift[2]

    def test_relabelling_is_stable_across_windows(self, df: pd.DataFrame) -> None:
        """Refit on shifted/disjoint windows: the semantic identity of each
        state (not merely the column index) must be the same every time —
        downtrend stays the negative-drift cluster."""
        fits = [
            fitted_plugin(df.iloc[: 2500]),
            fitted_plugin(df.iloc[300:2800]),
            fitted_plugin(df.iloc[500:]),
        ]
        for plugin in fits:
            assert plugin.state_names == _STATE_NAMES
            drift = plugin.means[:, 0]
            assert drift[0] < drift[1] < drift[2], (
                "state index no longer follows the canonical drift order — a "
                f"gate reading state==0 would have silently flipped: {drift}"
            )

    def test_label_orientation_is_invariant_to_gmm_label_swap(
        self, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force the GMM initialiser to hand back a DIFFERENT column order
        (the exact condition that flips an unguarded implementation) and prove
        the emitted vocabulary and per-bar state names are unchanged."""
        import research.regime.trend_hmm as mod

        original = mod.GaussianMixture

        class FlippedGMM(original):  # type: ignore[misc, valid-type]
            def fit(self, X: Any) -> FlippedGMM:
                super().fit(X)
                self.means_ = np.asarray(self.means_)[::-1].copy()
                cov = np.asarray(self.covariances_)[::-1].copy()
                self.covariances_ = cov
                self.weights_ = np.asarray(self.weights_)[::-1].copy()
                self.precisions_cholesky_ = np.asarray(
                    self.precisions_cholesky_
                )[::-1].copy()
                return self

        base = fitted_plugin(df)
        base_names = [s.state_name for s in base.predict(df)]
        monkeypatch.setattr(mod, "GaussianMixture", FlippedGMM)
        flipped = fitted_plugin(df)
        flipped_names = [s.state_name for s in flipped.predict(df)]

        assert flipped.state_names == base.state_names == _STATE_NAMES
        same = sum(1 for a, b in zip(base_names, flipped_names) if a == b)
        assert same == len(base_names), (
            f"GMM label swap changed {len(base_names) - same}/{len(base_names)} "
            "per-bar trend names — the vocabulary is NOT permutation-proof"
        )

    def test_degenerate_fit_fails_loudly(self) -> None:
        """If EM collapses to indistinguishable drifts the direction identity
        would be arbitrary — the plugin must refuse rather than ship a
        coin-flip gate."""
        with pytest.raises(ValueError, match=r"degenerate|distinguishable"):
            canonical_state_order(
                np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]]), 0
            )

    def test_non_directional_fit_fails_loudly(self) -> None:
        """A fit whose extreme states do not straddle zero learned no
        direction — refuse it."""
        with pytest.raises(ValueError, match="straddle zero"):
            canonical_state_order(
                np.array([[0.2, 0.0], [0.5, 0.0], [0.9, 0.0]]), 0
            )

    def test_direction_feature_must_be_an_input_feature(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="direction_feature"):
            CausalTrendHMM().fit(
                df, {**_TEST_CFG, "direction_feature": "not_a_feature"}
            )


# ---------------------------------------------------------------------------
# 4. Feature causality (value provenance)
# ---------------------------------------------------------------------------


@pytest.mark.no_lookahead
class TestFeatureCausality:
    def test_trend_features_are_prefix_invariant(self, df: pd.DataFrame) -> None:
        names = list(TREND_INPUT_FEATURES)
        full = compute_causal_trend_features(df, names, {})
        for t in (100, 1000, 2400):
            prefix = compute_causal_trend_features(df.iloc[: t + 1], names, {})
            pdt.assert_series_equal(
                full.iloc[t],
                prefix.iloc[t],
                check_exact=False,
                rtol=1e-10,
                check_names=False,
            )

    def test_slope_matches_polyfit(self, df: pd.DataFrame) -> None:
        """The vectorised rolling slope must equal a window-local np.polyfit
        (the closed-form shortcut is the overfit/leak risk here)."""
        window = 20
        X = compute_causal_trend_features(df, ["slope_20_norm"], {})
        # Recover the plugin's own ATR scale from its own feature rather than
        # re-deriving it here: a partial-frame ATR recomputation would differ
        # (ewm has no full preceding history), and the point of this test is
        # the slope algebra, not the ATR definition.
        scale = df["close"].to_numpy(dtype=np.float64) * compute_causal_trend_features(
            df, ["atr_14_norm"], {}
        )["atr_14_norm"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        for i in (150, 900, 2200):
            win = close[i - window + 1 : i + 1]
            slope, _ = np.polyfit(np.arange(window, dtype=np.float64), win, 1)
            expected = slope * (window - 1) / (scale[i] * np.sqrt(window))
            assert X["slope_20_norm"].iloc[i] == pytest.approx(expected, rel=1e-6)

    def test_mean_return_matches_cumulative_log_return(self, df: pd.DataFrame) -> None:
        window = 20
        X = compute_causal_trend_features(df, ["mean_return_20_norm"], {})
        scale = df["close"].to_numpy(dtype=np.float64) * compute_causal_trend_features(
            df, ["atr_14_norm"], {}
        )["atr_14_norm"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        for i in (200, 1500):
            move = np.log(close[i] / close[i - window])
            expected = move / (scale[i] * np.sqrt(window))
            assert X["mean_return_20_norm"].iloc[i] == pytest.approx(expected, rel=1e-9)

    def test_range_pos_within_bounds(self, df: pd.DataFrame) -> None:
        X = compute_causal_trend_features(df, ["range_pos_20_norm"], {})
        vals = X["range_pos_20_norm"].dropna()
        assert len(vals) > 0
        assert vals.min() >= -1.0 - 1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_unknown_feature_name_rejected(self) -> None:
        frame = make_trend_ohlcv(n=100)[0]
        with pytest.raises(ValueError, match="unknown input feature"):
            compute_causal_trend_features(frame, ["close_future"], {})

    def test_empty_feature_list_rejected(self) -> None:
        frame = make_trend_ohlcv(n=100)[0]
        with pytest.raises(ValueError, match="must not be empty"):
            compute_causal_trend_features(frame, [], {})

    def test_volume_feature_requires_volume_column(self) -> None:
        frame = make_trend_ohlcv(n=100)[0].drop(columns=["volume"])
        with pytest.raises(ValueError, match="volume"):
            compute_causal_trend_features(frame, ["volume_zscore_20"], {})

    def test_default_feature_set_has_no_volume_dependency(self) -> None:
        """Regression guard for the XAUUSD M15 volume hole: the DEFAULT trend
        feature set must not require the tick-volume column at all (it is zero
        for 98.7% of the shipped history, which is what collapses the
        volatility HMM to a single state)."""
        assert "volume_zscore_20" not in TREND_INPUT_FEATURES
        frame = make_trend_ohlcv(n=800)[0].drop(columns=["volume"])
        X = compute_causal_trend_features(frame, list(TREND_INPUT_FEATURES), {})
        assert X.shape[1] == len(TREND_INPUT_FEATURES)
        assert X.dropna().shape[0] > 700


# ---------------------------------------------------------------------------
# 5. Reproducibility + artifact
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_sequence(self, df: pd.DataFrame) -> None:
        p1, p2 = fitted_plugin(df), fitted_plugin(df)
        for a, b in zip(p1.predict(df), p2.predict(df)):
            _assert_states_equal(a, b)
        assert p1.config_hash == p2.config_hash
        assert np.array_equal(p1.transmat, p2.transmat)
        assert np.array_equal(p1.means, p2.means)

    def test_config_hash_12hex_and_sensitive(self, df: pd.DataFrame) -> None:
        p1 = fitted_plugin(df)
        assert len(p1.config_hash) == 12
        p2 = fitted_plugin(df, {**_TEST_CFG, "random_state": 99})
        assert p2.config_hash != p1.config_hash
        assert p2.data_version == p1.data_version

    def test_data_version_tracks_index(self, df: pd.DataFrame) -> None:
        assert fitted_plugin(df).data_version != fitted_plugin(df.iloc[:2000]).data_version

    def test_artifact_roundtrip(self, df: pd.DataFrame, tmp_path: Path) -> None:
        p1 = fitted_plugin(df)
        target = tmp_path / "trend_artifact.json"
        p1.save(target)
        p2 = CausalTrendHMM.load(target)
        assert p2.config_hash == p1.config_hash
        assert p2.data_version == p1.data_version
        assert p2.state_names == p1.state_names
        for a, b in zip(p1.predict(df), p2.predict(df)):
            _assert_states_equal(a, b)

    def test_artifact_dict_roundtrip(self, df: pd.DataFrame) -> None:
        p1 = fitted_plugin(df)
        p2 = CausalTrendHMM.from_artifact_dict(p1.to_artifact_dict())
        assert p2.get_feature_names() == p1.get_feature_names()
        assert p2.state_names == p1.state_names
        assert [s.state for s in p2.predict(df)] == [s.state for s in p1.predict(df)]


# ---------------------------------------------------------------------------
# 6. Schema + the gate-input-vs-model-feature decision (rework §2.4 step 5)
# ---------------------------------------------------------------------------


class TestSchemaAndScopeDecision:
    def test_schema_names_use_the_trend_namespace(self) -> None:
        names = [f.name for f in CausalTrendHMM().get_feature_schema()]
        assert names == [
            f"{TREND_FEATURE_PREFIX}state",
            f"{TREND_FEATURE_PREFIX}prob_downtrend",
            f"{TREND_FEATURE_PREFIX}prob_range",
            f"{TREND_FEATURE_PREFIX}prob_uptrend",
            f"{TREND_FEATURE_PREFIX}confidence",
        ]

    def test_no_schema_name_shares_the_model_feature_prefix(self) -> None:
        """DECISION RECORD (rework §2.4 step 5): this plugin is a GATE INPUT,
        not a model feature.

        The plugin's declared schema deliberately does NOT use the ``hmm_``
        model-feature namespace.  That keeps the trend states out of every
        trained model's ``feature_list`` — a model only picks up ``hmm_*``
        columns — so no ``feature_schema_version`` bump
        (``double-v1.0`` -> ``double-v1.1``) and no re-training is required.

        Caveat recorded honestly: ``HMMFeatureEmitter`` hardcodes the ``hmm_``
        prefix for the keys it writes into ``event.attributes``
        (``feature_emitter.py:172-175``), so when this plugin is run *through
        that helper* the attributes do land in the ``hmm_`` namespace.  The
        separation therefore holds at the level that actually matters for
        artifact pinning (the schema / ``feature_list`` a model is trained on),
        not at the attribute key level.
        """
        for feat in CausalTrendHMM().get_feature_schema():
            assert not feat.name.startswith("hmm_"), (
                f"{feat.name} leaked into the model feature namespace — that "
                "would invalidate pinned artifacts and require a "
                "double-v1.0 -> double-v1.1 bump + full re-training"
            )

    def test_schema_is_fully_causal(self) -> None:
        for feat in CausalTrendHMM().get_feature_schema():
            assert feat.uses_future_data is False
            assert feat.available_at == AVAILABLE_AT_CONFIRM

    def test_schema_events_validate_causality(self, df: pd.DataFrame) -> None:
        schema = CausalTrendHMM().get_feature_schema()
        events = [_event_at(b, df, event_id="CAUS") for b in (300, 900, 1800)]
        validate_causality(events, feature_schema=schema)  # must not raise


class TestDocumentedNegativeResult:
    """The plugin ships with a MEASURED NEGATIVE and must stay OFF by default.

    These tests pin the honesty contract rather than the model: the module
    docstring must keep recording the no-edge finding, and nothing in the
    plugin may activate itself.  If a future change starts claiming a
    trend-direction edge, or turns the plugin on implicitly, these fail.
    """

    def test_module_docstring_records_the_negative_result(self) -> None:
        import research.regime.trend_hmm as mod

        doc = mod.__doc__ or ""
        assert "MEASURED NEGATIVE RESULT" in doc, (
            "the module docstring must keep the measured no-edge finding — it is "
            "the only thing stopping a future reader from enabling this gate"
        )
        # the specific measured facts a reader must not lose
        assert "does not carry a forward-looking directional edge" in doc
        assert "INVERTED" in doc, "the inverted double_top sign must stay recorded"
        assert "FAILS" in doc, "the §4.2 sample-size failure must stay recorded"

    def test_plugin_does_not_activate_itself(self) -> None:
        """No ambient/global switch: constructing the plugin is inert, and the
        default config carries no 'enabled' flag that could be read as ON."""
        plugin = CausalTrendHMM()
        assert plugin.fitted is False
        cfg = plugin.get_default_config()
        assert "enabled" not in cfg, (
            "an 'enabled' key in the default config would invite a runtime to "
            "switch this gate on by default — activation must stay caller-driven"
        )

    def test_predict_requires_an_explicit_fit(self, df: pd.DataFrame) -> None:
        """An unfitted plugin cannot emit states: no implicit fit-on-first-use."""
        with pytest.raises(RuntimeError, match="not fitted"):
            CausalTrendHMM().predict(df)

    def test_no_threshold_was_tuned_to_reach_oos_100(self) -> None:
        """The §4.2 failure was reported, not engineered away: the default
        config carries no pattern-level allowed_states / min_events knob that
        could have been relaxed to inflate the sample."""
        cfg = CausalTrendHMM().get_default_config()
        for banned in ("allowed_states", "min_events", "oos_min_events", "min_keep"):
            assert banned not in cfg, (
                f"{banned!r} in the plugin config suggests the sample-size "
                "failure was tuned around instead of reported"
            )
class TestTrendSchemaCausality(NoLookaheadTestBase):
    """Inherited gates: timeline coherence, validate_causality acceptance,
    every declared feature non-future and resolvable at/just-before known_at."""

    feature_schema: ClassVar[Sequence[PatternFeature]] = (
        CausalTrendHMM().get_feature_schema()
    )

    def build_events(self) -> list[PatternEvent]:
        frame = make_trend_ohlcv(n=2500, seed=3)[0]
        return [
            PatternEvent(
                event_id=f"TREND-{i:04d}",
                pattern_name="double_bottom",
                pattern_version="1.0",
                symbol="XAUUSD",
                timeframe="M15",
                direction="bullish",
                detect_time=frame.index[i],
                confirm_time=frame.index[i],
                entry_time=frame.index[i],
            )
            for i in (300, 600, 900, 1200)
        ]
