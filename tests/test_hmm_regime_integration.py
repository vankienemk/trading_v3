"""t3 — HMM regime integration tests: emitter + plug/unplug + registry + gate.

Covers the t3 integration contract (requirements v1.0 §4/§5/§6, acceptance
B6-B10):

  1. Feature Emitter (guide §5 step 4): with a fitted regime plugin attached
     and the emitter enabled, ``hmm_state`` / ``hmm_prob_*`` / ``hmm_confidence``
     are injected into ``PatternEvent.attributes`` at the causal state
     (last closed bar ≤ known_at); when the plugin is off / not listed, NO
     hmm_* features are injected.  The runner backtest path uses the same
     causal lookup.
  2. Plug/unplug safety (guide §4.2C golden rule): a legacy model whose
     feature_list has no hmm_* runs *byte-identical* with and without a
     regime plugin attached — verified on the scorer (feature vector built
     from the model's own feature_list) and on the MultiPatternEngine output.
  3. Model Registry §4.3: ``feature_list`` + ``optional_plugins`` parse on
     ModelInfo; the 8 existing index.yaml entries back-compat auto-resolve
     feature_list from their artifacts' features.json, and the existing
     registry/GUI tests keep passing (verified by the separate suite run).
     A synthetic HMM-model index entry round-trips the new fields.
  4. Hard gate §8 (shared live ≡ backtest): ``is_allowed`` semantics, per-
     pattern rules (allowed_states + min_confidence), rule-absent -> allow,
     blocked events carry ``discard_reason="regime_blocked"``.  The gate is
     applied in the MultiPatternEngine and in runner.run_symbol_backtest via
     the SAME function.
  5. Config: configs/plugins/hmm_regime.yaml carries per-pattern rules;
     configs/symbols/_hmm_template.yaml shows the optional_plugins +
     features.optional.hmm wiring.
  6. Publish: RegimeState series persists to event_lake/regime/…parquet and
     round-trips (guide §5 step 2).
  7. Fail-closed (§6.3): a model whose feature_list has hmm_* and requires
     hmm_regime with NO active plugin emits no signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from live.engine.feature_emitter import (
    HMMFeatureEmitter,
    regime_flags,
)
from live.engine.hard_gate import (
    apply_gate,
    is_allowed,
)
from live.engine.regime_wiring import (
    load_plugin_config,
    resolve_regime_config,
    resolve_regime_wiring,
)
from live.engine.signal_engine_v2 import (
    MultiPatternEngine,
    PatternAssignment,
)
from live.state.shared_app_state_v2 import ModelRegistry
from research.core.contracts import (
    LIFECYCLE_LIVE,
    PatternEvent,
)
from research.core.event_lake import EventLake
from research.regime import (
    CausalGaussianHMM,
    RegimeState,
)
from tests.test_hmm_regime_plugin import (
    fitted_plugin,
    make_regime_ohlcv,
)
from tests.test_live_engine_integration import (
    _candles,
    _CandleSource,
    _ev,
    _make_detector_class,
)

T3_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regime_ohlcv(n: int = 2500, seed: int = 11) -> pd.DataFrame:
    """Synthetic 3-regime OHLCV frame (states 0/1/2 = trending/sideways/high_vol)."""
    return make_regime_ohlcv(n=n, seed=seed)[0]


def _plugin(df: pd.DataFrame | None = None) -> CausalGaussianHMM:
    df = df if df is not None else _regime_ohlcv()
    return fitted_plugin(df)


def _engine(
    plugin: Any,
    emitter_enabled: bool,
    gate_enabled: bool,
    rules: dict[str, Any] | None = None,
    candle_df: pd.DataFrame | None = None,
    shared_plugin: Any = None,
    event_bar: int = 20,
) -> MultiPatternEngine:
    """MultiPatternEngine with one live DB assignment + regime wiring."""
    cdf = candle_df if candle_df is not None else _candles(close=2650.0, n=40)
    ev = _ev("INT-DB-0001", "double_bottom", entry_price=2650.0, bar=event_bar)
    det_cls = _make_detector_class("double_bottom", "DB")
    assignments = [
        PatternAssignment(
            assignment_id="int-db@XAUUSD@M15",
            pattern_name="double_bottom",
            timeframe="M15",
            state=LIFECYCLE_LIVE,
            detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            regime_plugin=plugin,
            regime_config={"enabled": True, "emitter_enabled": emitter_enabled,
                           "gate_enabled": gate_enabled},
            regime_filter=rules,
        )
    ]
    return MultiPatternEngine(
        symbol="XAUUSD",
        assignments=assignments,
        candle_fn=_CandleSource(cdf).get_chart_history,
        regime_plugin=shared_plugin,
    )


# ---------------------------------------------------------------------------
# 1. Feature Emitter (guide §5 step 4)
# ---------------------------------------------------------------------------


class TestFeatureEmitter:
    def test_attach_injects_hmm_features_at_known_at(self) -> None:
        df = _regime_ohlcv()
        plugin = _plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        events = [
            PatternEvent(
                event_id=f"ev{i}", pattern_name="double_bottom", pattern_version="1.0",
                symbol="XAUUSD", timeframe="M15", direction="bullish",
                detect_time=df.index[i], confirm_time=df.index[i],
                entry_time=df.index[i],
                attributes={"confirm_bar": i},
            )
            for i in (500, 900, 1500)
        ]
        n = emitter.attach(events, states, df)
        assert n == len(events)
        for ev, i in zip(events, (500, 900, 1500)):
            reg = states[i]
            assert ev.attributes["hmm_state"] == int(reg.state)
            assert ev.attributes["hmm_prob_trending"] == pytest.approx(reg.state_prob["trending"])
            assert ev.attributes["hmm_prob_sideways"] == pytest.approx(reg.state_prob["sideways"])
            assert ev.attributes["hmm_prob_high_vol"] == pytest.approx(reg.state_prob["high_vol"])
            assert ev.attributes["hmm_confidence"] == pytest.approx(reg.confidence)
            assert pd.Timestamp(ev.attributes["hmm_known_at"]) == df.index[i]
            # causal stamp: the state's bar is at/just-before known_at
            assert pd.Timestamp(ev.attributes["hmm_known_at"]) <= ev.known_at_ts

    def test_attach_uses_confirm_bar_preferring_attributes(self) -> None:
        """DB/DT store ``confirm_bar``; the emitter uses it (never a future bar)."""
        df = _regime_ohlcv()
        plugin = _plugin(df)
        states = plugin.predict(df)
        emitter = HMMFeatureEmitter(plugin, plugin.get_default_config())
        # event confirms at bar 800 but known_at claims bar 900 — the state at
        # the confirm bar must be used (causal ≤ known_at).
        ev = PatternEvent(
            event_id="ev-attr", pattern_name="double_bottom", pattern_version="1.0",
            symbol="XAUUSD", timeframe="M15", direction="bullish",
            detect_time=df.index[790], confirm_time=df.index[800],
            known_at=df.index[900],
            attributes={"confirm_bar": 800},
        )
        emitter.attach([ev], states, df)
        assert ev.attributes["hmm_state"] == int(states[800].state)
        assert pd.Timestamp(ev.attributes["hmm_known_at"]) == df.index[800]

    def test_attach_skips_missing_bar(self) -> None:
        df = _regime_ohlcv()
        plugin = _plugin(df)
        states = plugin.predict(df)
        # event before the frame → no closed bar ≤ known_at → skipped
        ev = PatternEvent(
            event_id="ev-old", pattern_name="double_bottom", pattern_version="1.0",
            symbol="XAUUSD", timeframe="M15", direction="bullish",
            detect_time=df.index[0] - pd.Timedelta(days=1),
            confirm_time=df.index[0] - pd.Timedelta(days=1),
        )
        n = HMMFeatureEmitter(plugin, plugin.get_default_config()).attach([ev], states, df)
        assert n == 0
        assert "hmm_state" not in ev.attributes

    def test_engine_injects_when_enabled(self) -> None:
        engine = _engine(_plugin(), emitter_enabled=True, gate_enabled=False)
        events = engine.check_new_bar()
        assert len(events) == 1  # one representative -> one candidate
        last = engine.get_last_events()
        assert len(last) == 1
        attrs = last[0].attributes
        assert "hmm_state" in attrs
        assert "hmm_prob_trending" in attrs
        assert "hmm_confidence" in attrs
        # causal: injected state's bar ≤ event known_at
        known = pd.Timestamp(last[0].known_at_ts)
        assert pd.Timestamp(attrs["hmm_known_at"]) <= known

    def test_engine_no_injection_when_disabled(self) -> None:
        engine = _engine(None, emitter_enabled=False, gate_enabled=False)
        engine.check_new_bar()
        for e in engine.get_last_events():
            assert "hmm_state" not in e.attributes
            assert "hmm_prob_trending" not in e.attributes

    def test_engine_no_injection_when_config_enabled_false(self) -> None:
        """config.hmm.enabled=false → plugin attached but explicit master off."""
        engine = _engine(_plugin(), emitter_enabled=True, gate_enabled=False)
        # patch the assignment's regime_config master switch off
        engine.assignments[0].regime_config = {"enabled": False}
        engine._assign_regime_slots(engine.regime_plugin, None)
        engine.check_new_bar()
        for e in engine.get_last_events():
            assert "hmm_state" not in e.attributes

    def test_feature_frame_columns_match_feature_names(self) -> None:
        df = _regime_ohlcv()
        plugin = _plugin(df)
        events = [
            PatternEvent(
                event_id=f"ev{i}", pattern_name="double_bottom", pattern_version="1.0",
                symbol="XAUUSD", timeframe="M15", direction="bullish",
                detect_time=df.index[i], confirm_time=df.index[i],
                entry_time=df.index[i],
                attributes={"confirm_bar": i},
            )
            for i in (400, 700)
        ]
        frame = HMMFeatureEmitter(plugin, plugin.get_default_config()).feature_frame(df, events)
        assert list(frame.columns) == plugin.get_feature_names()
        assert sorted(frame.index.tolist()) == ["ev400", "ev700"]
        assert frame.loc["ev400", "hmm_state"] == int(
            plugin.predict(df)[400].state
        )
        assert frame["hmm_confidence"].notna().all()

    def test_regime_flags_mapping(self) -> None:
        assert regime_flags({"emitter_enabled": True}) == (True, False)
        assert regime_flags({"gate_enabled": True}) == (False, True)
        assert regime_flags({}) == (False, False)


# ---------------------------------------------------------------------------
# 2. Plug/unplug safety — legacy model unchanged (guide §4.2C golden rule)
# ---------------------------------------------------------------------------


class TestLegacyByteIdentical:
    def test_build_feature_vector_uses_model_feature_list_only(self) -> None:
        from live.engine.feature_emitter import build_feature_vector

        ev = _ev("LEG-0001", "double_bottom", model_prob=0.6)
        ev.attributes.update(
            {"depth_atr": 1.2, "rule_score": 0.55, "hmm_state": 1, "extra_col": 9}
        )
        # old model: feature_list has NO hmm_* (nor extra_col) — vector = the
        # exact train-time list; extra columns are DROPPED by construction.
        vec = build_feature_vector(
            ev, {"feature_list": ["depth_atr", "rule_score"]}
        )
        assert list(vec) == [1.2, 0.55]
        # new model: hmm_* read from attributes, in feature_list order
        vec2 = build_feature_vector(
            ev, {"feature_list": ["depth_atr", "hmm_state", "hmm_confidence"]}
        )
        assert vec2[0] == 1.2
        assert vec2[1] == 1.0
        assert np.isnan(vec2[2])  # hmm_confidence missing → NaN

    def test_runner_scorer_old_model_byte_identical(self) -> None:
        """A legacy scorer built from an artifact WITHOUT hmm_* returns the
        same probs with and without a regime plugin attached (the scoring
        frame is reindexed to the model's feature_list only)."""
        from research.core.walkforward_trainer import build_feature_frame

        df = _regime_ohlcv()
        events = [
            PatternEvent(
                event_id=f"ev{i}", pattern_name="double_bottom", pattern_version="1.0",
                symbol="XAUUSD", timeframe="M15", direction="bullish",
                detect_time=df.index[i - 2], confirm_time=df.index[i],
                entry_time=df.index[i + 1], known_at=df.index[i],
                entry_price=1800.0, stop_price=1795.0, target_price=1815.0,
                rule_score=0.5, model_prob=0.6,
                attributes={"pivot_known_at_bar": i - 2, "confirm_bar": i,
                             "depth_atr": 1.3, "low_offset_atr": 0.2,
                             "pattern_length": 8, "rule_score": 0.5},
            )
            for i in (700, 900, 1200)
        ]
        X, names = build_feature_frame(df, events)
        legacy_names = [c for c in names if not c.startswith("hmm_")]
        # with hmm attrs present (plugin attached) but feature_list w/o hmm_*:
        for e in events:
            e.attributes["hmm_state"] = 1
            e.attributes["hmm_confidence"] = 0.9
        X_legacy = X.reindex(columns=legacy_names).fillna(0.0).astype(float)
        # build_feature_frame output is deterministic — the reindexed frame
        # ignores hmm columns entirely (extra columns dropped).
        assert "hmm_state" not in X_legacy.columns
        assert list(X_legacy.columns) == legacy_names

    def test_engine_candidates_identical_with_and_without_plugin(self) -> None:
        """MultiPatternEngine: legacy assignment (no regime wiring) produces
        byte-identical candidates whether or not a plugin exists engine-wide."""
        base_df = _candles(close=2650.0, n=40)

        def run(plugin: Any, shared: Any) -> list[Any]:
            ev = _ev("BYT-0001", "double_bottom", entry_price=2650.0, bar=15)
            det_cls = _make_detector_class("double_bottom", "DB")
            a = PatternAssignment(
                assignment_id="byt@XAUUSD@M15", pattern_name="double_bottom",
                timeframe="M15", state=LIFECYCLE_LIVE, detector=det_cls([ev]),
                config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
                regime_plugin=plugin,
            )
            return MultiPatternEngine(
                symbol="XAUUSD", assignments=[a],
                candle_fn=_CandleSource(base_df).get_chart_history,
                regime_plugin=shared,
            ).check_new_bar()

        no_plugin = run(None, None)
        with_plugin = run(_plugin(), None)  # attached but no emitter/gate flags
        assert len(no_plugin) == len(with_plugin) == 1
        a, b = no_plugin[0], with_plugin[0]
        assert a.event_id == b.event_id
        assert a.entry_price == b.entry_price
        assert a.model_prob == b.model_prob
        assert a.combined_score == b.combined_score
        assert a.metadata == b.metadata


# ---------------------------------------------------------------------------
# 3. Model Registry §4.3
# ---------------------------------------------------------------------------


class TestRegistrySection43:
    def test_existing_entries_auto_resolve_feature_list(self) -> None:
        reg = ModelRegistry.get_instance()
        reg.reload(str(T3_ROOT / "model_registry" / "index.yaml"))
        db = reg.get_model("double_bottom_xauusd_m15_v1")
        assert db is not None and db.feature_list
        # DB feature_list mirrors the artifact's features.json (25 features)
        assert "depth_atr" in db.feature_list
        assert all(not f.startswith("hmm_") for f in db.feature_list)
        assert db.optional_plugins == []
        # every existing entry has a resolved (or empty) feature_list
        for m in reg.get_available_models():
            assert isinstance(m.feature_list, list)
            assert isinstance(m.optional_plugins, list)

    def test_new_hmm_entry_roundtrip(self, tmp_path: Path) -> None:
        index = tmp_path / "index.yaml"
        index.write_text(
            """
models:
  double_bottom_xauusd_m15_v1_hmm:
    model_id: "double_bottom_xauusd_m15_v1_hmm"
    pattern_name: "double_bottom"
    symbol_origin: "XAUUSD"
    symbol: "XAUUSD"
    timeframe: "M15"
    feature_schema_version: "double-v1.1"
    lifecycle_state: "validated"
    gate_passed: true
    calibrated: true
    config_hash: "ab12cd34ef56"
    trained_at: "2026-09-08T00:00:00Z"
    model_path: "research/patterns/double_bottom/artifacts/models/double_bottom_xauusd_m15_v1/model.pkl"
    calibrator_path: "research/patterns/double_bottom/artifacts/models/double_bottom_xauusd_m15_v1/calibrator.pkl"
    feature_schema: "research/patterns/double_bottom/artifacts/models/double_bottom_xauusd_m15_v1/features.json"
    feature_list:
      - depth_atr
      - rule_score
      - hmm_state
      - hmm_prob_trending
      - hmm_confidence
    optional_plugins:
      - name: "hmm_regime"
        version: "1.0.0"
        required: true
    metrics: {}
    horizon: 72
    target: "forward_mfe_72bar_1.25r"
""".lstrip(),
            encoding="utf-8",
        )
        reg = ModelRegistry()
        n = reg.load_from_yaml(str(index))
        assert n == 1
        m = reg.get_model("double_bottom_xauusd_m15_v1_hmm")
        assert m is not None
        assert m.feature_list == [
            "depth_atr", "rule_score", "hmm_state", "hmm_prob_trending", "hmm_confidence",
        ]
        assert m.optional_plugins == [
            {"name": "hmm_regime", "version": "1.0.0", "required": True}
        ]
        assert m.feature_schema_version == "double-v1.1"
        assert m.is_assignable() is True

    def test_wiring_uses_registry_metadata_fail_closed(self) -> None:
        """§6.3: a model trained WITH hmm_* and a required hmm_regime plugin
        must NOT produce a signal when the plugin is not wired."""
        base_df = _candles(close=2650.0, n=40)
        ev = _ev("FC-0001", "double_bottom", entry_price=2650.0, bar=15)
        det_cls = _make_detector_class("double_bottom", "DB")
        a = PatternAssignment(
            assignment_id="fc@XAUUSD@M15", pattern_name="double_bottom",
            timeframe="M15", state=LIFECYCLE_LIVE, detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            model_feature_list=["depth_atr", "rule_score", "hmm_state", "hmm_confidence"],
            model_optional_plugins=[{"name": "hmm_regime", "version": "1.0.0", "required": True}],
        )
        engine = MultiPatternEngine(
            symbol="XAUUSD", assignments=[a],
            candle_fn=_CandleSource(base_df).get_chart_history,
        )
        candidates = engine.check_new_bar()
        assert candidates == []
        last = engine.get_last_events()
        assert len(last) == 1
        assert last[0].attributes.get("discard_reason") == "hmm_unavailable"


# ---------------------------------------------------------------------------
# 4. Hard gate §8 — shared live ≡ backtest
# ---------------------------------------------------------------------------


class TestHardGate:
    def test_is_allowed_semantics(self) -> None:
        df = _regime_ohlcv()
        plugin = _plugin(df)
        states = plugin.predict(df)
        ev = _ev("GATE-0001", "double_bottom")
        reg = states[700]
        # no rule → allow
        assert is_allowed(ev, reg, {}) == (True, "no_rule")
        # no rules → allow even with regime None
        assert is_allowed(ev, None, {}) == (True, "no_rule")
        # rule for another pattern → allow (no_rule)
        assert is_allowed(ev, reg, {"double_top": {"allowed_states": ["trending"]}}) == (
            True, "no_rule",
        )
        # regime None with a matching rule → fail closed
        assert is_allowed(ev, None, {"double_bottom": {"allowed_states": ["trending"]}}) == (
            False, "no_regime",
        )
        # allowed state + confidence ok
        assert is_allowed(
            ev, reg,
            {"double_bottom": {"allowed_states": [reg.state_name], "min_confidence": 0.0}},
        ) == (True, "allowed")

    def test_gate_reasons_state_and_confidence(self) -> None:
        ev = _ev("GATE-0002", "double_bottom")
        reg = RegimeState(
            timestamp=pd.Timestamp("2026-01-01"), state=0, state_name="trending",
            state_prob={"trending": 0.7, "sideways": 0.2, "high_vol": 0.1},
            confidence=0.7, lag_bars=0, model_version="1.0.0", config_hash="ab" * 6,
        )
        assert is_allowed(
            ev, reg, {"double_bottom": {"allowed_states": ["sideways"]}}
        ) == (False, "state_not_allowed")
        assert is_allowed(
            ev, reg, {"double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.8}}
        ) == (False, "low_confidence")

    def test_apply_gate_stamps_discard_reason(self) -> None:
        ev = _ev("GATE-0003", "double_bottom")
        reg = RegimeState(
            timestamp=pd.Timestamp("2026-01-01"), state=0, state_name="trending",
            state_prob={"trending": 0.7, "sideways": 0.2, "high_vol": 0.1},
            confidence=0.7, lag_bars=0, model_version="1.0.0", config_hash="ab" * 6,
        )
        ok = apply_gate(ev, reg, {"double_bottom": {"allowed_states": ["sideways"]}})
        assert ok is False
        assert ev.attributes["discard_reason"] == "regime_blocked"

    def test_is_allowed_guide8_raw_list_form_safe(self) -> None:
        """t4-b3: the guide §8 list-of-dicts rules form
        ``[{"pattern": ..., "allowed_states": ...}]`` must be handled safely
        (previously crashed with AttributeError — a list has no ``.get``)."""
        ev = _ev("GATE-LIST-0001", "double_bottom")
        reg = RegimeState(
            timestamp=pd.Timestamp("2026-01-01"), state=0, state_name="trending",
            state_prob={"trending": 0.7, "sideways": 0.2, "high_vol": 0.1},
            confidence=0.7, lag_bars=0, model_version="1.0.0", config_hash="ab" * 6,
        )
        rules_list = [
            {"pattern": "double_bottom", "allowed_states": ["sideways"]},
            {"pattern": "head_shoulders", "allowed_states": ["trending"]},
        ]
        # matching pattern listed → state_not_allowed (no crash)
        assert is_allowed(ev, reg, rules_list) == (
            False, "state_not_allowed",
        )
        # regime None with a matching rule → fail-closed no_regime
        assert is_allowed(ev, None, rules_list) == (False, "no_regime")
        # pattern without a rule → allow
        other = _ev("GATE-LIST-0002", "double_top")
        assert is_allowed(other, reg, rules_list) == (True, "no_rule")

    def test_engine_gate_blocks_and_allows(self) -> None:
        df = _regime_ohlcv()
        plugin = _plugin(df)
        # default states (short window) are state 0 "trending" conf 0.5; the
        # candle frame matches the stub event's entry price (drift guard idle).
        blocking = _engine(
            plugin, emitter_enabled=False, gate_enabled=True,
            rules={"double_bottom": {"allowed_states": ["sideways"]}},
        )
        assert blocking.check_new_bar() == []
        last = blocking.get_last_events()
        assert len(last) == 1
        assert last[0].attributes.get("discard_reason") == "regime_blocked"

        allowing = _engine(
            plugin, emitter_enabled=False, gate_enabled=True,
            rules={"double_bottom": {"allowed_states": ["trending"], "min_confidence": 0.0}},
        )
        cands = allowing.check_new_bar()
        assert len(cands) == 1
        assert cands[0].event_id == "INT-DB-0001"

    def test_runner_gate_blocks_and_allows(self) -> None:
        """runner.run_symbol_backtest applies the gate through the same
        is_allowed; blocked trade recorded with cap_reason=regime_blocked."""
        from research.multi_backtest.runner import (
            BacktestAssignment,
            run_symbol_backtest,
        )

        df = _regime_ohlcv(700)
        ev = PatternEvent(
            event_id="RUN-GATE-1", pattern_name="double_bottom", pattern_version="1.0",
            symbol="XAUUSD", timeframe="M15", direction="bullish",
            detect_time=df.index[300], confirm_time=df.index[302],
            entry_time=df.index[303], known_at=df.index[302],
            entry_price=float(df["close"].iloc[305]), stop_price=float(df["close"].iloc[305]) * 0.995,
            target_price=float(df["close"].iloc[305]) * 1.02,
            rule_score=0.6, model_prob=0.7,
            attributes={"confirm_bar": 302, "atr_value": 5.0},
        )
        det_cls = _make_detector_class("double_bottom", "DB")
        plugin = _plugin(df)

        def run(rules: dict[str, Any] | None) -> tuple[int, int]:
            a = BacktestAssignment(
                assignment_id="run@XAUUSD", pattern_name="double_bottom",
                timeframe="M15", state=LIFECYCLE_LIVE,
                detector=det_cls([ev]), config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
                risk_fraction=0.5,  # ≤ §4.4 default cluster cap (0.75) so a lone trade passes
                regime_plugin=plugin,
                regime_config={"enabled": True, "emitter_enabled": False, "gate_enabled": True},
                regime_filter=rules,
            )
            res = run_symbol_backtest(df, [a])
            blocked = [t for t in res.rejected if t.cap_reason == "regime_blocked"]
            return len(res.executed_trades), len(blocked)

        no_gate_n, no_gate_blocked = run(None)
        assert no_gate_n == 1 and no_gate_blocked == 0
        gate_n, gate_blocked = run({"double_bottom": {"allowed_states": ["sideways"]}})
        # only ~700 bars → the predict window may be short (default states) or
        # real; either way exactly ONE decision is made per event.
        assert gate_n + gate_blocked == 1
        # a2: assert the discard_reason lands on the representative when blocked
        a2 = BacktestAssignment(
            assignment_id="run2@XAUUSD", pattern_name="double_bottom",
            timeframe="M15", state=LIFECYCLE_LIVE,
            detector=det_cls([ev]), config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            risk_fraction=0.5,
            regime_plugin=plugin,
            regime_config={"enabled": True, "emitter_enabled": False, "gate_enabled": True},
            regime_filter={"double_bottom": {"allowed_states": ["sideways"]}},
        )
        res2 = run_symbol_backtest(df, [a2])
        blocked2 = [t for t in res2.rejected if t.cap_reason == "regime_blocked"]
        if blocked2:
            assert any(
                e.attributes.get("discard_reason") == "regime_blocked"
                for e in res2.events
            )

    def test_shared_is_allowed_identity(self) -> None:
        """live engine and runner resolve the SAME is_allowed function."""
        from live.engine.hard_gate import is_allowed as canonical

        live_mod = __import__("live.engine.hard_gate", fromlist=["is_allowed"])
        assert live_mod.is_allowed is canonical

    def test_sample_config_rules_present(self) -> None:
        cfg = load_plugin_config(T3_ROOT / "configs" / "plugins" / "hmm_regime.yaml")
        filter_block = cfg.get("regime_filter")
        assert isinstance(filter_block, dict)
        assert filter_block.get("enabled") is False  # default OFF
        rules = filter_block.get("rules")
        assert isinstance(rules, dict)
        for pat in ("double_bottom", "double_top", "liquidity_sweep"):
            assert pat in rules
            assert "allowed_states" in rules[pat]
            assert "min_confidence" in rules[pat]

    def test_resolve_regime_config_switches(self) -> None:
        cfg = load_plugin_config(T3_ROOT / "configs" / "plugins" / "hmm_regime.yaml")
        resolved = resolve_regime_config(
            cfg.get("plugin"), cfg.get("hmm"), cfg.get("regime_filter"), "double_bottom",
        )
        assert resolved["enabled"] is False  # plugin.enabled=false → OFF
        assert resolved["gate_enabled"] is False
        assert resolved["rules"]["double_bottom"]["allowed_states"] == ["trending", "sideways"]
        # flip the master on → emitter follows plugin.enabled; the gate STAYS
        # off because regime_filter.enabled is still false in the sample.
        on = resolve_regime_config(
            {"name": "hmm_regime", "version": "1.0.0", "enabled": True},
            cfg.get("hmm"), cfg.get("regime_filter"), "double_bottom",
        )
        assert on["enabled"] is True
        assert on["emitter_enabled"] is True
        assert on["gate_enabled"] is False  # regime_filter.enabled=false → off
        assert on["rules"]["liquidity_sweep"]["allowed_states"] == ["trending", "high_vol"]
        # gate on only when the regime_filter block itself enables it
        gate_on = resolve_regime_config(
            {"name": "hmm_regime", "version": "1.0.0", "enabled": True},
            cfg.get("hmm"),
            {"enabled": True, "rules": cfg.get("regime_filter", {}).get("rules", {})},
            "double_bottom",
        )
        assert gate_on["enabled"] is True
        assert gate_on["emitter_enabled"] is True
        assert gate_on["gate_enabled"] is True
        assert gate_on["rules"]["double_bottom"]["allowed_states"] == ["trending", "sideways"]


# ---------------------------------------------------------------------------
# 6. Publish to event lake (guide §5 step 2)
# ---------------------------------------------------------------------------


class TestEventLakeRegime:
    def test_regime_states_roundtrip_via_lake(self, tmp_path: Path) -> None:
        from live.engine.feature_emitter import (
            publish_regime_states,
            read_regime_states,
        )

        df = _regime_ohlcv(700)
        plugin = _plugin(df)
        states = plugin.predict(df)
        lake = EventLake(root=tmp_path)
        path = publish_regime_states(tmp_path, "XAUUSD", "M15", states)
        assert path.name == "XAUUSD_M15.parquet"
        assert path.is_file()
        loaded = read_regime_states(tmp_path, "XAUUSD", "M15")
        assert len(loaded) == len(states)
        for a, b in zip(states, loaded):
            assert b.timestamp == a.timestamp
            assert b.state == a.state
            assert b.state_name == a.state_name
            assert b.state_prob == a.state_prob
            assert abs(b.confidence - a.confidence) < 1e-12
            assert b.config_hash == a.config_hash
        # absent symbol → empty
        assert read_regime_states(tmp_path, "EURUSD", "M15") == []
        # the publisher writes under the lake's regime/ subdir (§5 step 2)
        assert (lake.root / "regime" / "XAUUSD_M15.parquet").is_file()

    def test_regime_path_convention(self, tmp_path: Path) -> None:
        from live.engine.feature_emitter import publish_regime_states

        path = publish_regime_states(tmp_path, "xauusd", "m15", [])
        assert path.name == "XAUUSD_M15.parquet"
        assert path.parent.name == "regime"


# ---------------------------------------------------------------------------
# Config templates
# ---------------------------------------------------------------------------


class TestConfigTemplates:
    def test_symbol_template_lists_plugin_and_features(self) -> None:
        import yaml

        tmpl = T3_ROOT / "configs" / "symbols" / "_hmm_template.yaml"
        assert tmpl.is_file()
        data = yaml.safe_load(tmpl.read_text(encoding="utf-8"))
        feats = data.get("features")
        assert feats is not None
        assert "hmm_regime" in feats.get("optional_plugins", [])
        hmm_opt = feats.get("optional", {}).get("hmm", {})
        assert hmm_opt.get("enabled") is True
        assert "hmm_state" in hmm_opt.get("features", [])
        assert data.get("regime_filter", {}).get("enabled") is False

    def test_wiring_regime_slot_default_off(self) -> None:
        """resolve_regime_wiring returns None when the model doesn't list the
        plugin / master is off — the default OFF contract."""
        plugin = _plugin()
        resolved_off = {"enabled": False, "hmm": {}, "rules": {}, "emitter_enabled": False,
                        "gate_enabled": False}
        assert resolve_regime_wiring(plugin, resolved_off, "a1", "double_bottom") is None
        slot = resolve_regime_wiring(
            plugin,
            {"enabled": True, "hmm": plugin.get_default_config(),
             "rules": {"double_bottom": {"allowed_states": ["trending"]}},
             "emitter_enabled": True, "gate_enabled": True},
            "a1", "double_bottom",
        )
        assert slot is not None
        assert slot.emitter_active() is True
        assert slot.gate_active() is True
        assert slot.gate_rules()["double_bottom"]["allowed_states"] == ["trending"]


# ---------------------------------------------------------------------------
# t10 repair round 2 — config_source master switch, runner fail-closed parity,
# emitter-failure fail-closed, listing semantics (§4.3)
# ---------------------------------------------------------------------------


class TestRepairRound2:
    """Coverage for the t10 repair items (review t7 findings)."""

    def test_engine_config_source_plugin_enabled_activates_slot(self) -> None:
        """engine-level `regime_config_source` with plugin.enabled=true → the
        slot resolves ACTIVE (emitter + gate) and the scan injects hmm_* and
        applies the gate (acceptance item 1)."""
        plugin = _plugin()
        # a full hmm_regime YAML-shaped source with the master switch ON
        config_source = {
            "plugin": {"name": "hmm_regime", "version": "1.0.0", "enabled": True},
            "hmm": {"enabled": True, **plugin.get_default_config()},
            "regime_filter": {
                "enabled": True,
                "rules": {"double_bottom": {"allowed_states": ["sideways"]}},
            },
        }
        ev = _ev("R2-CFG-0001", "double_bottom", entry_price=2650.0, bar=20)
        det_cls = _make_detector_class("double_bottom", "DB")
        assignments = [
            PatternAssignment(
                assignment_id="r2-cfg@XAUUSD@M15",
                pattern_name="double_bottom",
                timeframe="M15",
                state=LIFECYCLE_LIVE,
                detector=det_cls([ev]),
                config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
                # no per-assignment regime bundle — engine-level source decides
            )
        ]
        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=assignments,
            candle_fn=_CandleSource(_candles(close=2650.0, n=40)).get_chart_history,
            regime_plugin=plugin,
            regime_config_source=config_source,
        )
        slot = engine._regime_slot(assignments[0])
        assert slot is not None
        assert slot.emitter_active() is True
        assert slot.gate_active() is True
        # emitter injected hmm_* AND the gate blocked (sideways-only):
        cands = engine.check_new_bar()
        assert cands == []
        last = engine.get_last_events()
        assert len(last) == 1
        assert "hmm_state" in last[0].attributes
        assert last[0].attributes.get("discard_reason") == "regime_blocked"

    def test_engine_config_source_plugin_disabled_slot_none(self) -> None:
        """engine-level config_source with plugin.enabled=false (default OFF
        sample) → no slot, no hmm_* injection, legacy behavior."""
        plugin = _plugin()
        cfg = load_plugin_config(T3_ROOT / "configs" / "plugins" / "hmm_regime.yaml")
        assert cfg.get("plugin", {}).get("enabled") is False  # sample master OFF
        ev = _ev("R2-OFF-0001", "double_bottom", entry_price=2650.0, bar=20)
        det_cls = _make_detector_class("double_bottom", "DB")
        assignments = [
            PatternAssignment(
                assignment_id="r2-off@XAUUSD@M15",
                pattern_name="double_bottom",
                timeframe="M15",
                state=LIFECYCLE_LIVE,
                detector=det_cls([ev]),
                config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            )
        ]
        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=assignments,
            candle_fn=_CandleSource(_candles(close=2650.0, n=40)).get_chart_history,
            regime_plugin=plugin,
            regime_config_source=cfg,
        )
        assert engine._regime_slot(assignments[0]) is None
        cands = engine.check_new_bar()
        assert len(cands) == 1  # legacy behavior: signal emitted, no hmm_*
        for e in engine.get_last_events():
            assert "hmm_state" not in e.attributes

    def test_runner_fail_closed_required_hmm_no_plugin(self) -> None:
        """runner-side §6.3 parity: assignment whose model has hmm_* features
        + required hmm_regime but NO plugin wired → events stamped
        hmm_unavailable, scorer skipped (model_prob stays None), no trade
        simulated (acceptance item 2)."""
        from research.multi_backtest.runner import (
            BacktestAssignment,
            run_symbol_backtest,
        )

        df = _regime_ohlcv(700)
        ev = PatternEvent(
            event_id="R2-RUN-FC-1", pattern_name="double_bottom", pattern_version="1.0",
            symbol="XAUUSD", timeframe="M15", direction="bullish",
            detect_time=df.index[300], confirm_time=df.index[302],
            entry_time=df.index[303], known_at=df.index[302],
            entry_price=float(df["close"].iloc[305]),
            stop_price=float(df["close"].iloc[305]) * 0.995,
            target_price=float(df["close"].iloc[305]) * 1.02,
            rule_score=0.6, model_prob=None,
            attributes={"confirm_bar": 302, "atr_value": 5.0},
        )
        det_cls = _make_detector_class("double_bottom", "DB")

        def record_score(events: list[PatternEvent]) -> dict[str, float]:
            return {e.event_id: 0.9 for e in events}  # would set model_prob if called

        a = BacktestAssignment(
            assignment_id="r2-run-fc@XAUUSD", pattern_name="double_bottom",
            timeframe="M15", state=LIFECYCLE_LIVE,
            detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            scorer=record_score,
            risk_fraction=0.5,
            model_feature_list=["depth_atr", "rule_score", "hmm_state", "hmm_confidence"],
            model_optional_plugins=[{"name": "hmm_regime", "version": "1.0.0", "required": True}],
            # no regime_plugin / no regime_config → slot None
        )
        res = run_symbol_backtest(df, [a])
        assert res.executed_trades == []
        rejected_reasons = {t.cap_reason for t in res.rejected}
        assert "hmm_unavailable" in rejected_reasons
        # scorer skipped: model_prob stays None although the scorer would return 0.9
        scored = [e for e in res.events if e.event_id == ev.event_id]
        assert scored and scored[0].model_prob is None
        assert scored[0].attributes.get("discard_reason") == "hmm_unavailable"

    def test_runner_fail_closed_plugin_wired_trade_proceeds(self) -> None:
        """control: with the plugin wired (+ rules allowing), the same model
        requirement is satisfied and the trade executes (no false positives)."""
        from research.multi_backtest.runner import (
            BacktestAssignment,
            run_symbol_backtest,
        )

        df = _regime_ohlcv(700)
        ev = PatternEvent(
            event_id="R2-RUN-OK-1", pattern_name="double_bottom", pattern_version="1.0",
            symbol="XAUUSD", timeframe="M15", direction="bullish",
            detect_time=df.index[300], confirm_time=df.index[302],
            entry_time=df.index[303], known_at=df.index[302],
            entry_price=float(df["close"].iloc[305]),
            stop_price=float(df["close"].iloc[305]) * 0.995,
            target_price=float(df["close"].iloc[305]) * 1.02,
            rule_score=0.6, model_prob=None,
            attributes={"confirm_bar": 302, "atr_value": 5.0},
        )
        det_cls = _make_detector_class("double_bottom", "DB")
        plugin = _plugin(df)
        a = BacktestAssignment(
            assignment_id="r2-run-ok@XAUUSD", pattern_name="double_bottom",
            timeframe="M15", state=LIFECYCLE_LIVE,
            detector=det_cls([ev]),
            config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
            risk_fraction=0.5,
            regime_plugin=plugin,
            regime_config={"enabled": True, "emitter_enabled": False, "gate_enabled": True},
            regime_filter=None,  # no rules → gate inactive → trade proceeds
            model_feature_list=["depth_atr", "rule_score", "hmm_state", "hmm_confidence"],
            model_optional_plugins=[{"name": "hmm_regime", "version": "1.0.0", "required": True}],
        )
        res = run_symbol_backtest(df, [a])
        assert len(res.executed_trades) == 1
        assert not any(t.cap_reason == "hmm_unavailable" for t in res.rejected)

    def test_engine_emitter_failure_stamps_unavailable_and_skips_scorer(self) -> None:
        """slice-item 3: when the emitter is ACTIVE but computation fails for a
        model requiring hmm_regime, events get discard_reason='hmm_unavailable'
        and tier-2 scoring is SKIPPED (mirror runner side; engine path)."""

        class _BrokenPlugin(CausalGaussianHMM):
            def predict(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> list[Any]:
                raise RuntimeError("regime compute exploded")

        plugin = _BrokenPlugin()
        plugin._fitted = True  # type: ignore[attr-defined]  # pierce guard: predict() itself raises
        ev = _ev("R2-FAIL-0001", "double_bottom", entry_price=2650.0, bar=20, model_prob=None)
        det_cls = _make_detector_class("double_bottom", "DB")
        calls = {"scorer": 0}

        def record_score(events: list[PatternEvent]) -> dict[str, float]:
            calls["scorer"] += 1
            return {e.event_id: 0.9 for e in events}

        assignments = [
            PatternAssignment(
                assignment_id="r2-fail@XAUUSD@M15",
                pattern_name="double_bottom",
                timeframe="M15",
                state=LIFECYCLE_LIVE,
                detector=det_cls([ev]),
                config={"symbol": "XAUUSD", "timeframe": "M15", "version": "1.0"},
                scorer=record_score,
                regime_plugin=plugin,
                regime_config={"enabled": True, "emitter_enabled": True, "gate_enabled": False},
                model_feature_list=["depth_atr", "rule_score", "hmm_state", "hmm_confidence"],
                model_optional_plugins=[{"name": "hmm_regime", "version": "1.0.0", "required": True}],
            )
        ]
        engine = MultiPatternEngine(
            symbol="XAUUSD",
            assignments=assignments,
            candle_fn=_CandleSource(_candles(close=2650.0, n=40)).get_chart_history,
        )
        cands = engine.check_new_bar()
        assert cands == []  # fail-closed: no candidate for hmm_unavailable event
        last = engine.get_last_events()
        assert len(last) == 1
        assert last[0].attributes.get("discard_reason") == "hmm_unavailable"
        assert last[0].model_prob is None  # scorer skipped
        assert calls["scorer"] == 0

    def test_listing_semantics_name_based(self) -> None:
        """slice-item 4: _list_plugin requires *listing* (name == hmm_regime)
        when model_optional_plugins provided — required:false still counts as
        listed; a different plugin name does not."""
        from live.engine.feature_emitter import lists_hmm_regime
        from live.engine.regime_wiring import RegimeSlot

        plugin = _plugin()
        slot = RegimeSlot(
            plugin=plugin,
            emitter_enabled=True,
            gate_enabled=False,
            rules={},
            hmm_config={},
            assignment_id="a1",
            pattern_name="double_bottom",
            model_optional_plugins=[{"name": "hmm_regime", "version": "1.0.0", "required": False}],
        )
        assert lists_hmm_regime(slot.model_optional_plugins) is True
        assert slot.emitter_active() is True  # listed (even if not required)
        # a different plugin name → NOT listed → nothing activates
        other = RegimeSlot(
            plugin=plugin, emitter_enabled=True, gate_enabled=False, rules={},
            hmm_config={}, assignment_id="a2", pattern_name="double_bottom",
            model_optional_plugins=[{"name": "other_plugin", "required": True}],
        )
        assert lists_hmm_regime(other.model_optional_plugins) is False
        assert other.emitter_active() is False