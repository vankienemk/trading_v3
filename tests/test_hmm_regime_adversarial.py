"""
test_hmm_regime_adversarial.py — t4: HMM regime + integration adversarial suite
(HMM_REGIME_PLUGIN_INTEGRATION_GUIDE.md §7 #5, §8; requirements v1.0 §5/§6/§7
#5; extends the philosophy of tests/test_adversarial_lookahead.py).

Every look-ahead vector in the HMM path is probed with an injected attack and
MUST be caught by the suite or neutralized by the implementation:

  1. shift(-1) leak on HMM input features — the classic ``close[t+1]`` read:
     the value-provenance oracle (recompute from bars <= known_at) matches the
     honest feature and flags the poisoned column; poisoned states diverge
     from the causal states so the CI net (truncation invariance) fails;
  2. timeline / stamp attacks — future or duplicate stamps raise in
     ``predict``; a ``confirm_bar`` attribute pointing at a FUTURE bar is
     neutralized **only when the event carries an explicit ``known_at``**; with
     the field unset (the normal detector output — no detector sets it and
     ``validate_causality`` never writes it back) the guard in
     ``state_at_confirm_bar`` compares against ``ev.known_at`` (None) and the
     future bar's regime state is attached — a live-reachable look-ahead at
     the emitter/gate seam, encoded as a failing test here so the repair is
     forced; the predict-output alignment contract that the emitter trusts is
     pinned;
  3. staleness window — a stale event (old known_at) must be filtered by the
     state AT its known_at, never by a newer regime;
  4. confidence threshold exploit — a fabricated high-confidence state cannot
     bypass the state filter; real plugin states honour min_confidence;
  5. plug/unplug (guide §4.2C 'quy tắc vàng') — plugin ON → hmm_* attributes
     correct; plugin OFF → NO hmm_* key anywhere; a legacy model
     (feature_list without hmm_*) produces a BYTE-IDENTICAL feature vector
     (NaNs included) and byte-identical engine candidates even when the
     emitter is fully active; extra hmm_* columns are dropped by construction;
  6. hard gate §8 — rule-absent -> allow; state not in allowed_states ->
     reject; confidence < min -> reject; missing regime with a matching rule
     -> fail-closed reject; every block carries the exact
     ``discard_reason="regime_blocked"``; the gate consumes the state at the
     event's known_at (stale-safe).

The whole module carries the registered ``no_lookahead`` marker so it runs in
``pytest -m no_lookahead`` and in every ``pytest tests/`` CI run.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd
import pytest

from live.engine.feature_emitter import (
    HMMFeatureEmitter,
    build_feature_vector,
    state_at_confirm_bar,
)
from live.engine.hard_gate import (
    DISCARD_REASON_REGIME_BLOCKED,
    REASON_ALLOWED,
    REASON_LOW_CONFIDENCE,
    REASON_NO_REGIME,
    REASON_NO_RULE,
    REASON_STATE_NOT_ALLOWED,
    apply_gate,
    is_allowed,
)
from live.engine.regime_wiring import resolve_regime_config, resolve_regime_wiring
from research.core.contracts import LIFECYCLE_LIVE, PatternEvent
from research.core.walkforward_trainer import build_feature_frame
from research.regime import (
    CausalGaussianHMM,
    RegimeState,
    compute_causal_input_features,
)
from tests.test_hmm_regime_no_lookahead import (
    _STATE_NAMES,
    _TEST_CFG,
    _event_at,
    fitted_plugin,
    make_regime_ohlcv,
)

pytestmark = pytest.mark.no_lookahead


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_regime_ohlcv()[0]


@pytest.fixture(scope="module")
def plugin(df: pd.DataFrame) -> CausalGaussianHMM:
    return fitted_plugin(df)


def _regime_state(
    name: str = "trending",
    confidence: float = 0.7,
    timestamp: pd.Timestamp | None = None,
) -> RegimeState:
    probs = {n: (1.0 - confidence) / 2 for n in _STATE_NAMES}
    probs[name] = confidence
    return RegimeState(
        timestamp=timestamp or pd.Timestamp("2026-01-01", tz="UTC"),
        state=_STATE_NAMES.index(name),
        state_name=name,
        state_prob=probs,
        confidence=float(confidence),
        lag_bars=0,
        model_version="1.0.0",
        config_hash="ab" * 6,
    )


# ---------------------------------------------------------------------------
# 1. shift(-1) leak on HMM input features (guide §7 #5; requirements §7 #5)
# ---------------------------------------------------------------------------


def _honest_log_return_at(df: pd.DataFrame, t: int) -> float:
    """Value oracle: log_return_1 recomputed from the causally available
    bars (a prefix ending at t) — the only value bar t may legitimately hold."""
    close = df["close"].astype(np.float64).to_numpy(dtype=float)
    if t <= 0:
        return float("nan")
    return float(np.log(close[t] / close[t - 1]))


class TestShiftMinusOneFeatureLeak:
    def test_shift_minus_one_leak_flags_at_every_bar(self, df: pd.DataFrame) -> None:
        """Poisoned ``log_return_1 = log(close[t+1]/close[t])`` — exactly
        ``close.shift(-1)`` — mismatches the causal oracle at EVERY bar; the
        honest recompute matches it (positive control)."""
        close = df["close"].astype(np.float64)
        poisoned = np.log((close.shift(-1) / close).to_numpy(dtype=float))
        honest_full = compute_causal_input_features(df, ["log_return_1"])[
            "log_return_1"
        ].to_numpy(dtype=float)
        for t in range(1, 400, 7):
            honest = _honest_log_return_at(df, t)
            assert np.isclose(honest, honest_full[t], rtol=1e-10), (
                f"bar {t}: canonical feature is not prefix-reproducible"
            )
            assert not np.isclose(honest, poisoned[t], equal_nan=True), (
                f"bar {t}: the shift(-1) value accidentally equals the causal "
                "value — oracle would not catch it"
            )

    def test_poisoned_column_diverges_from_causal_states(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """The injected column changes the regime states vs the causal
        recompute -> any truncation-invariance / value-provenance net fails
        loudly on the poisoned frame (the injection cannot slip through)."""
        base = np.asarray([s.state for s in plugin.predict(df)])
        poisoned = df.copy()
        close = poisoned["close"].astype(np.float64)
        poisoned["log_return_1"] = np.log((close.shift(-1) / close).to_numpy())
        with_poison = np.asarray(
            [s.state for s in plugin.predict(poisoned, dict(_TEST_CFG))]
        )
        assert not np.array_equal(with_poison, base), (
            "shift(-1)-poisoned HMM feature produced identical states to the "
            "causal recompute — the CI net would NOT catch the injection"
        )

    def test_raw_ohlcv_and_honest_precomputed_columns_agree(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """Positive control: supplying the canonical feature columns computed
        honestly (prefix-stable) reproduces the raw-OHLCV states exactly —
        the causal path is deterministic and the oracle is consistent."""
        raw = np.asarray([s.state for s in plugin.predict(df)])
        honest = df.copy()
        honest["log_return_1"] = compute_causal_input_features(df, ["log_return_1"])[
            "log_return_1"
        ].to_numpy(dtype=float)
        precomputed = np.asarray(
            [s.state for s in plugin.predict(honest, dict(_TEST_CFG))]
        )
        assert np.array_equal(precomputed, raw)


# ---------------------------------------------------------------------------
# 2. Timeline / stamp attacks (predict + emitter)
# ---------------------------------------------------------------------------


class TestTimelineStampAttacks:
    def test_future_stamp_inserted_mid_sequence_rejected(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        shuffled = df.sample(frac=1.0, random_state=7)
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(shuffled)

    def test_duplicate_stamp_rejected(self, df: pd.DataFrame, plugin: CausalGaussianHMM) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            plugin.predict(pd.concat([df, df.iloc[[50]]]))

    def test_future_confirm_bar_guarded_when_known_at_explicit(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """Stamp attack on the emitter with the causal stamp EXPLICIT (the
        contract the events are validated under): attributes['confirm_bar']
        = t+5 claims a state that does not exist yet at the event's known_at.
        The guard must fall back to the causal lookup (last closed bar <=
        known_at) and must NEVER attach the future bar's state."""
        t = 800
        ev = _event_at(t, df, event_id="FB")
        ev.known_at = df.index[t]  # explicit causal stamp (engine contract)
        ev.attributes["confirm_bar"] = t + 5  # attack: future confirm bar
        states = plugin.predict(df)
        reg = state_at_confirm_bar(states, df, ev)
        assert reg is not None
        assert reg.timestamp <= ev.known_at_ts
        assert reg is not states[t + 5]
        assert reg.timestamp == df.index[t]
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        assert emitter.attach([ev], states, df) == 1
        stamped = pd.Timestamp(ev.attributes["hmm_known_at"])
        assert stamped <= ev.known_at_ts

    def test_future_confirm_bar_leak_when_known_at_unset(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """FINDING (t4-b1): the causal guard in
        ``live.engine.feature_emitter.state_at_confirm_bar`` validates the
        confirm-bar stamp against the raw ``PatternEvent.known_at`` FIELD
        instead of the derived ``known_at_ts`` property.  Detector events
        carry ``known_at=None`` (verified: no pattern detector sets it and
        ``validate_causality`` never writes it back), so the guard is
        VACUOUS: an ``attributes['confirm_bar']`` pointing at a FUTURE bar is
        honored and the emitter attaches a future-stamped regime state —
        a look-ahead in the HMM path (shared by the hard-gate lookup: engine
        path signal_engine_v2.py:1046/1177 and runner.attach_regime_features).
        This test MUST pass once the guard uses ``ev.known_at_ts``."""
        t = 800
        ev = _event_at(t, df, event_id="FB-LEAK")
        ev.attributes["confirm_bar"] = t + 5  # attack: future confirm bar
        assert ev.known_at is None  # normal detector output (field unset)
        states = plugin.predict(df)
        reg = state_at_confirm_bar(states, df, ev)
        assert reg is not None
        assert reg.timestamp <= ev.known_at_ts, (
            f"state_at_confirm_bar attached a FUTURE state (stamp "
            f"{reg.timestamp} > known_at {ev.known_at_ts}) when known_at is "
            "unset — the HMM integration must fail closed (guard must "
            "compare against ev.known_at_ts)"
        )
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        emitter.attach([ev], states, df)
        stamped = pd.Timestamp(ev.attributes["hmm_known_at"])
        assert stamped <= ev.known_at_ts, (
            "emitter attached hmm_known_at in the FUTURE of the event's "
            "known_at — live-reachable look-ahead (fix: state_at_confirm_bar "
            "guard on ev.known_at_ts)"
        )

    def test_known_at_beyond_last_bar_uses_last_causal_state(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """An event stamped AFTER the frame still gets a causal state: the
        last closed bar (<= known_at) — never a synthetic/future one."""
        states = plugin.predict(df)
        ev = PatternEvent(
            event_id="AFT-0001",
            pattern_name="double_bottom",
            pattern_version="1.0",
            symbol="XAUUSD",
            timeframe="M15",
            direction="bullish",
            detect_time=df.index[-1],
            confirm_time=df.index[-1],
            known_at=df.index[-1] + pd.Timedelta(hours=1),
            attributes={"confirm_bar": len(df) - 1},
        )
        reg = state_at_confirm_bar(states, df, ev)
        assert reg is states[-1]
        assert reg.timestamp <= ev.known_at_ts

    def test_known_at_before_first_bar_skips_attach(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """No closed bar <= known_at -> the emitter attaches NOTHING and no
        hmm_* key may appear (scorer NaN contract, requirements §6.2)."""
        ev = PatternEvent(
            event_id="BEF-0001",
            pattern_name="double_bottom",
            pattern_version="1.0",
            symbol="XAUUSD",
            timeframe="M15",
            direction="bullish",
            detect_time=df.index[0] - pd.Timedelta(days=2),
            confirm_time=df.index[0] - pd.Timedelta(days=2),
        )
        states = plugin.predict(df)
        assert state_at_confirm_bar(states, df, ev) is None
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        assert emitter.attach([ev], states, df) == 0
        assert not any(k.startswith("hmm_") for k in ev.attributes)

    def test_predict_output_alignment_contract_pinned(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """The invariant the whole emitter trusts: predict output is aligned
        to the frame (states[i] <-> df.index[i]), length matches, timestamps
        are non-decreasing bar closes.  A reversed / misaligned sequence is
        therefore a detectable contract violation."""
        states = plugin.predict(df)
        assert len(states) == len(df)
        assert all(s.timestamp == df.index[i] for i, s in enumerate(states))
        stamps = [s.timestamp for s in states]
        assert all(b >= a for a, b in zip(stamps, stamps[1:]))


# ---------------------------------------------------------------------------
# 3. Staleness window — a stale event is filtered by ITS known_at regime
# ---------------------------------------------------------------------------


class TestStalenessWindow:
    def test_stale_event_uses_state_at_its_known_at(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        stale_bar = 500
        states = plugin.predict(df)
        ev = _event_at(stale_bar, df, event_id="STALE")
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        assert emitter.attach([ev], states, df) == 1
        assert pd.Timestamp(ev.attributes["hmm_known_at"]) == df.index[stale_bar]
        assert int(ev.attributes["hmm_state"]) == int(states[stale_bar].state)
        # far older than the latest regime — never a newer state
        assert pd.Timestamp(ev.attributes["hmm_known_at"]) < df.index[-1]

    def test_gate_uses_stale_state_not_latest(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """The hard gate consumes the regime AT the event's known_at: a rule
        that allows the stale state (but not the latest state) must ALLOW the
        stale event — proving the gate never looks at a newer regime."""
        states = plugin.predict(df)
        latest = states[-1]
        stale_bars = [
            i
            for i in range(400, len(states), 25)
            if states[i].state_name != latest.state_name
        ]
        assert stale_bars, (
            "no stale bar with a different state than the latest — generator "
            "did not separate regimes"
        )
        stale_bar = int(stale_bars[0])
        stale_reg = states[stale_bar]
        ev = _event_at(stale_bar, df, event_id="STALEG")
        allowed = [stale_reg.state_name]
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": allowed, "min_confidence": 0.0}
        }
        # the wiring hands the stale state (state_at_confirm_bar):
        assert state_at_confirm_bar(states, df, ev) is stale_reg
        assert is_allowed(ev, stale_reg, rules)[0] is True
        # the same rule evaluated against the LATEST regime rejects:
        assert is_allowed(ev, latest, rules) == (False, REASON_STATE_NOT_ALLOWED)
        # and the full stale path (attach + gate) mirrors the engine:
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        emitter.attach([ev], states, df)
        attached = state_at_confirm_bar(states, df, ev)
        assert attached is not None
        assert apply_gate(ev, attached, rules) is True
        assert "discard_reason" not in ev.attributes


# ---------------------------------------------------------------------------
# 4. Confidence threshold exploit — fabricated / boundary confidence
# ---------------------------------------------------------------------------


class TestConfidenceThresholdExploit:
    def test_confidence_below_min_rejected_with_reason(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.55)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.6}
        }
        assert is_allowed(ev, reg, rules) == (False, REASON_LOW_CONFIDENCE)

    def test_confidence_boundary_exactly_min_allowed(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.6)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.6}
        }
        assert is_allowed(ev, reg, rules) == (True, REASON_ALLOWED)

    def test_min_confidence_zero_skips_threshold(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.05)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.0}
        }
        assert is_allowed(ev, reg, rules) == (True, REASON_ALLOWED)

    def test_plugin_real_states_honour_min_confidence(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        ev = _event_at(10, df)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": list(_STATE_NAMES), "min_confidence": 1.5}
        }
        states = plugin.predict(df)
        assert all(not is_allowed(ev, s, rules)[0] for s in states)
        rules_loose: dict[str, Any] = {
            "double_bottom": {"allowed_states": list(_STATE_NAMES), "min_confidence": 0.0}
        }
        assert all(is_allowed(ev, s, rules_loose)[0] for s in states)

    def test_fabricated_high_confidence_state_cannot_bypass_state_filter(self) -> None:
        """Exploit attempt: a hand-crafted RegimeState with confidence 1.0
        still cannot pass a rule whose allowed_states exclude its state."""
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="high_vol", confidence=1.0)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.9}
        }
        assert is_allowed(ev, reg, rules) == (False, REASON_STATE_NOT_ALLOWED)


# ---------------------------------------------------------------------------
# 5. Plug / unplug — guide §4.2C 'quy tắc vàng'
# ---------------------------------------------------------------------------


class TestPlugUnplug:
    def test_emitter_on_attributes_correct(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        bars = (400, 900, 1500)
        events = [_event_at(b, df, event_id="ON") for b in bars]
        assert emitter.attach(events, states, df) == len(bars)
        for ev, bar in zip(events, bars):
            reg = states[bar]
            assert int(ev.attributes["hmm_state"]) == int(reg.state)
            for name in _STATE_NAMES:
                assert ev.attributes[f"hmm_prob_{name}"] == pytest.approx(
                    reg.state_prob[name]
                )
            assert ev.attributes["hmm_confidence"] == pytest.approx(reg.confidence)
            assert pd.Timestamp(ev.attributes["hmm_known_at"]) == df.index[bar]
            assert str(ev.attributes["hmm_config_hash"]) == plugin.config_hash

    def test_emitter_off_no_hmm_keys(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """Unplug: when the emitter is not wired, events carry NO hmm_* key —
        neither the value features nor the lineage keys."""
        events = [_event_at(b, df, event_id="OFF") for b in (500, 1200)]
        for ev in events:
            assert not any(k.startswith("hmm_") for k in ev.attributes)
        # even calling attach with an empty/absent state series injects nothing
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        assert emitter.attach(events, [], df) == 0
        for ev in events:
            assert not any(k.startswith("hmm_") for k in ev.attributes)

    def test_wiring_off_resolves_to_no_slot(self, plugin: CausalGaussianHMM) -> None:
        """Default-OFF wiring: an all-false resolved config yields no regime
        slot, so the emitter cannot run by construction."""
        off = {
            "enabled": False,
            "hmm": {},
            "rules": {},
            "emitter_enabled": False,
            "gate_enabled": False,
        }
        assert resolve_regime_wiring(plugin, off, "a1", "double_bottom") is None
        assert resolve_regime_wiring(plugin, None, "a1", "double_bottom") is None

    def test_byte_identical_feature_vector_legacy_model(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """GOLDEN RULE (§4.2C): the same legacy event (feature_list without
        hmm_*) yields a BIT-identical feature vector before and after the
        plugin is attached — NaNs included."""
        bar = 700
        ev = _event_at(bar, df, event_id="BYTE")
        ev.attributes.update({"depth_atr": 1.25, "rule_score": 0.55})
        legacy_meta: dict[str, Any] = {"feature_list": ["depth_atr", "rule_score"]}
        vec_before = build_feature_vector(ev, legacy_meta)
        # plugin on: emitter injects hmm_* into the SAME event
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        assert emitter.attach([ev], states, df) == 1
        vec_after = build_feature_vector(ev, legacy_meta)
        assert np.array_equal(vec_before, vec_after, equal_nan=True)
        assert list(vec_before) == [1.25, 0.55]
        # the identical event read by a NEW model (hmm_* in feature_list)
        # does consume the attached values — plugging only ADDS opt-in reads
        new_meta: dict[str, Any] = {
            "feature_list": ["depth_atr", "hmm_state", "hmm_confidence"]
        }
        vec_new = build_feature_vector(ev, new_meta)
        assert vec_new[0] == pytest.approx(1.25)
        assert vec_new[1] == pytest.approx(float(ev.attributes["hmm_state"]))
        assert vec_new[2] == pytest.approx(float(ev.attributes["hmm_confidence"]))

    def test_legacy_feature_frame_identical_with_hmm_columns_present(
        self, df: pd.DataFrame
    ) -> None:
        """req §6.4 #3: a feature frame containing extra hmm_* columns is
        byte-identical to the frame without them once reindexed to the
        model's own (legacy) feature_names — extra columns are dropped."""
        events = [_bf_event(df, i) for i in (700, 900, 1200)]
        X1, names1 = build_feature_frame(df, events)
        for e in events:
            e.attributes.update(
                {
                    "hmm_state": 1,
                    "hmm_confidence": 0.9,
                    "hmm_prob_trending": 0.9,
                    "hmm_prob_sideways": 0.05,
                    "hmm_prob_high_vol": 0.05,
                }
            )
        X2, names2 = build_feature_frame(df, events)
        assert names1 == names2
        assert all(not c.startswith("hmm_") for c in names1)
        assert X1.equals(X2), "hmm_* columns changed the legacy feature frame"

    def test_engine_plugin_off_no_hmm_keys_on_events(self) -> None:
        """Wiring OFF end-to-end (real engine): no hmm_* key may appear."""
        MultiPatternEngine, PatternAssignment = _engine_modules()
        _ev, _candles, _CandleSource, _make_detector_class = _live_helpers()
        cdf = _candles(close=2650.0, n=40)
        ev = _ev("ADV-OFF-0001", "double_bottom", entry_price=2650.0, bar=15)
        det_cls = _make_detector_class("double_bottom", "DB")
        a = PatternAssignment(
            assignment_id="adv-off@XAUUSD@M15",
            pattern_name="double_bottom",
            timeframe="M15",
            state=LIFECYCLE_LIVE,
            detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
        )
        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=[a],
            candle_fn=_CandleSource(cdf).get_chart_history,
        )
        cands = engine.check_new_bar()
        assert len(cands) == 1
        for e in engine.get_last_events():
            assert not any(k.startswith("hmm_") for k in e.attributes)

    def test_engine_byte_identical_legacy_model_with_emitter_active(
        self, df: pd.DataFrame, plugin: CausalGaussianHMM
    ) -> None:
        """GOLDEN RULE through the live path: with the emitter FULLY active,
        a legacy model (feature_list without hmm_*) produces byte-identical
        candidates to the no-plugin run — even though the plugin's hmm_*
        attributes ARE injected (proving the plugin was truly on)."""
        MultiPatternEngine, PatternAssignment = _engine_modules()
        _ev, _candles, _CandleSource, _make_detector_class = _live_helpers()
        cdf = _candles(close=2650.0, n=40)

        def run(with_plugin: bool) -> tuple[list[Any], list[Any]]:
            ev = _ev("ADV-BYT-0001", "double_bottom", entry_price=2650.0, bar=15)
            det_cls = _make_detector_class("double_bottom", "DB")
            kwargs: dict[str, Any] = {
                "assignment_id": "adv-byt@XAUUSD@M15",
                "pattern_name": "double_bottom",
                "timeframe": "M15",
                "state": LIFECYCLE_LIVE,
                "detector": det_cls([ev]),
                "config": {"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
                "model_feature_list": ["depth_atr", "rule_score"],  # legacy, no hmm_*
            }
            if with_plugin:
                kwargs["regime_plugin"] = plugin
                kwargs["regime_config"] = {
                    "enabled": True,
                    "emitter_enabled": True,
                    "gate_enabled": False,
                }
            a = PatternAssignment(**kwargs)
            engine = MultiPatternEngine(
                symbol="XAUUSD",
                assignments=[a],
                candle_fn=_CandleSource(cdf).get_chart_history,
                regime_plugin=plugin if with_plugin else None,
            )
            cands = engine.check_new_bar()
            return cands, engine.get_last_events()

        off_cands, off_last = run(False)
        on_cands, on_last = run(True)
        assert len(off_cands) == len(on_cands) == 1
        a, b = off_cands[0], on_cands[0]
        assert a.event_id == b.event_id
        assert a.entry_price == b.entry_price
        assert a.model_prob == b.model_prob
        assert a.combined_score == b.combined_score
        assert a.metadata == b.metadata
        # the emitter really was active: hmm_* attributes present on-plugin
        assert len(on_last) == 1
        attrs = on_last[0].attributes
        assert "hmm_state" in attrs and "hmm_confidence" in attrs
        assert not any(k.startswith("hmm_") for k in off_last[0].attributes)


def _bf_event(df: pd.DataFrame, bar: int) -> PatternEvent:
    """Double-bottom-ish event with the attribute keys build_feature_frame
    reads (causal within the frame)."""
    return PatternEvent(
        event_id=f"BFF-{bar:04d}",
        pattern_name="double_bottom",
        pattern_version="1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=df.index[bar - 3],
        confirm_time=df.index[bar],
        entry_time=df.index[bar + 1],
        known_at=df.index[bar],
        entry_price=float(df["close"].iloc[bar]),
        stop_price=float(df["close"].iloc[bar]) * 0.995,
        target_price=float(df["close"].iloc[bar]) * 1.02,
        rule_score=0.55,
        model_prob=0.6,
        structure_levels={"neckline": float(df["close"].iloc[bar])},
        attributes={
            "pivot_known_at_bar": bar - 3,
            "confirm_bar": bar,
            "depth_atr": 1.3,
            "low_offset_atr": 0.2,
            "pattern_length": 8,
            "rule_score": 0.55,
        },
    )


def _engine_modules() -> tuple[Any, Any]:
    """Live engine classes via importlib (keeps mypy --strict clean: the
    engine module carries pre-existing untyped defs outside this suite's
    verify scope)."""
    sv2: Any = import_module("live.engine.signal_engine_v2")
    return sv2.MultiPatternEngine, sv2.PatternAssignment


def _live_helpers() -> tuple[Any, Any, Any, Any]:
    """Stub detector/candle helpers from the live-engine integration suite."""
    tmod: Any = import_module("tests.test_live_engine_integration")
    return tmod._ev, tmod._candles, tmod._CandleSource, tmod._make_detector_class


# ---------------------------------------------------------------------------
# 6. Hard gate §8 — full decision matrix + discard_reason audit
# ---------------------------------------------------------------------------


class TestHardGateSection8:
    def test_no_rule_allows_even_without_regime(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        assert is_allowed(ev, None, {}) == (True, REASON_NO_RULE)
        reg = _regime_state()
        assert is_allowed(ev, reg, {}) == (True, REASON_NO_RULE)

    def test_rule_for_other_pattern_allows(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="high_vol", confidence=0.9)
        rules: dict[str, Any] = {"double_top": {"allowed_states": ["trending"]}}
        assert is_allowed(ev, reg, rules) == (True, REASON_NO_RULE)

    def test_state_not_allowed_rejects_with_reason(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="high_vol", confidence=0.9)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending", "sideways"]}
        }
        assert is_allowed(ev, reg, rules) == (False, REASON_STATE_NOT_ALLOWED)

    def test_low_confidence_rejects_with_reason(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.4)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.5}
        }
        assert is_allowed(ev, reg, rules) == (False, REASON_LOW_CONFIDENCE)

    def test_no_regime_fail_closed(self) -> None:
        """Gate on + plugin giving no state for the event -> BLOCKED (silent
        pass-through is forbidden: requirements §5 fail-closed)."""
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        rules: dict[str, Any] = {"double_bottom": {"allowed_states": ["trending"]}}
        assert is_allowed(ev, None, rules) == (False, REASON_NO_REGIME)

    def test_allowed_path_reports_reason(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="sideways", confidence=0.8)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["sideways"], "min_confidence": 0.7}
        }
        assert is_allowed(ev, reg, rules) == (True, REASON_ALLOWED)

    def test_apply_gate_stamps_discard_reason_when_blocked(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.5)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["sideways"], "min_confidence": 0.4}
        }
        assert apply_gate(ev, reg, rules) is False
        assert ev.attributes["discard_reason"] == DISCARD_REASON_REGIME_BLOCKED
        assert ev.attributes["discard_reason"] == "regime_blocked"

    def test_apply_gate_allowed_leaves_no_stamp(self) -> None:
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.5)
        rules: dict[str, Any] = {
            "double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.0}
        }
        assert apply_gate(ev, reg, rules) is True
        assert "discard_reason" not in ev.attributes

    def test_guide_list_form_rules_resolved_by_wiring(self) -> None:
        """Guide §8 configures rules as a LIST of {pattern, allowed_states,
        min_confidence}; the wiring layer resolves it into the per-pattern
        map BEFORE the gate consumes it (the gate contract is the resolved
        map — requirements §5 / hard_gate.is_allowed signature)."""
        cfg = resolve_regime_config(
            {"name": "hmm_regime", "version": "1.0.0", "enabled": True},
            {"enabled": True},
            {
                "enabled": True,
                "rules": [
                    {
                        "pattern": "double_bottom",
                        "allowed_states": ["trending"],
                        "min_confidence": 0.7,
                    }
                ],
            },
            "double_bottom",
        )
        resolved_rules = cfg["rules"]
        assert isinstance(resolved_rules, dict)
        assert resolved_rules["double_bottom"]["allowed_states"] == ["trending"]
        ev = _event_at(10, make_regime_ohlcv(40)[0])
        reg = _regime_state(name="trending", confidence=0.8)
        assert is_allowed(ev, reg, resolved_rules) == (True, REASON_ALLOWED)
        reg_blocked = _regime_state(name="high_vol", confidence=0.9)
        assert is_allowed(ev, reg_blocked, resolved_rules) == (
            False,
            REASON_STATE_NOT_ALLOWED,
        )

    def test_engine_gate_blocks_in_wrong_regime_with_discard_reason(
        self, plugin: CausalGaussianHMM
    ) -> None:
        """End-to-end: the engine's gate decision + Event-Lake-facing reason.
        (The plugin fitted on the regime frame predicts the CANDLE frame —
        40 flat bars < min_predict_bars -> deterministic default state
        'trending' confidence 0.5 — so rules are provably decided.)"""
        MultiPatternEngine, PatternAssignment = _engine_modules()
        _ev, _candles, _CandleSource, _make_detector_class = _live_helpers()
        cdf = _candles(close=2650.0, n=40)
        ev = _ev("ADV-GATE-0001", "double_bottom", entry_price=2650.0, bar=15)
        det_cls = _make_detector_class("double_bottom", "DB")
        a = PatternAssignment(
            assignment_id="adv-gate@XAUUSD@M15",
            pattern_name="double_bottom",
            timeframe="M15",
            state=LIFECYCLE_LIVE,
            detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            regime_plugin=plugin,
            regime_config={"enabled": True, "emitter_enabled": False, "gate_enabled": True},
            regime_filter={"double_bottom": {"allowed_states": ["sideways"]}},
        )
        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=[a],
            candle_fn=_CandleSource(cdf).get_chart_history,
            regime_plugin=plugin,
        )
        assert engine.check_new_bar() == []  # blocked: default state is trending
        last = engine.get_last_events()
        assert len(last) == 1
        assert last[0].attributes.get("discard_reason") == "regime_blocked"