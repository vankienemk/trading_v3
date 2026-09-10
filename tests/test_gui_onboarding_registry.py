"""GUI onboarding + registry bridge tests — model_registry §5.5 GUI flow (t3).

Offscreen (QT_QPA_PLATFORM=offscreen) tests that the SystemBridge exposes the
§10.1 filtered model dropdowns, that pattern assignments persist across bridge
restarts, that the onboarding tab lists only assignable models, and that a
stored assignment (model_id + feature_schema_version) is directly consumable
by the MultiPatternEngine contract (PatternAssignment).

No MCP/network is touched: ModelRegistry reads the on-disk YAML index, the
bridge loads the persisted symbol registry (as a normal GUI boot would), and
all engine work uses the same stubbed detectors as tests/test_live_engine_integration.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from live.engine.signal_engine_v2 import (
    MultiPatternEngine,
    PatternAssignment,
)
from live.gui.gui_bridge import SystemBridge
from live.gui.gui_tab_onboarding import SymbolOnboardingTab
from live.state.shared_app_state_v2 import SymbolConfig
from tests.test_live_engine_integration import (
    _candles,
    _CandleSource,
    _ev,
    _make_detector_class,
)

_ASSIGNMENTS_FILE = (
    Path(__file__).resolve().parent.parent / "live" / "db" / "pattern_assignments.json"
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def assignments_file_clean():
    """Remove the persisted assignment store before/after each test."""
    for path in (_ASSIGNMENTS_FILE, _ASSIGNMENTS_FILE.with_suffix(".json.tmp")):
        if path.is_file():
            path.unlink()
    yield
    for path in (_ASSIGNMENTS_FILE, _ASSIGNMENTS_FILE.with_suffix(".json.tmp")):
        if path.is_file():
            path.unlink()


def _fresh_bridge() -> SystemBridge:
    """Construct a fresh SystemBridge (registry + symbol state loaded)."""
    return SystemBridge()


# ---------------------------------------------------------------------------
# Bridge-level §10.1 filter
# ---------------------------------------------------------------------------

def test_bridge_get_models_for_assignment_filters_by_pattern_and_lifecycle(
    qapp, assignments_file_clean,
) -> None:
    bridge = _fresh_bridge()

    db_models = bridge.get_models_for_assignment(pattern_name="double_bottom")
    assert [m.model_id for m in db_models] == ["double_bottom_xauusd_m15_v1"]
    for m in db_models:
        assert m.lifecycle_state in {"validated", "shadow", "live"}

    # trained explore models are not offered to any dropdown
    fw = bridge.get_models_for_assignment(pattern_name="falling_wedge")
    assert fw == []

    # legacy live models still offered (liquidity_sweep pattern)
    lsw = bridge.get_models_for_assignment(pattern_name="liquidity_sweep")
    assert {m.model_id for m in lsw} == {
        "xauusd_v2_h16_20260905", "eurusd_v2_h16_20260906",
    }


def test_bridge_get_model_back_compat(qapp, assignments_file_clean) -> None:
    bridge = _fresh_bridge()
    info = bridge.get_model("xauusd_v2")  # retired alias must still resolve
    assert info is not None
    assert info.lifecycle_state == "retired"
    assert bridge.get_model("nope_nope") is None


# ---------------------------------------------------------------------------
# Onboarding tab renders filtered models (§10.1)
# ---------------------------------------------------------------------------

def test_onboarding_tab_assignable_models_filter(qapp, assignments_file_clean) -> None:
    bridge = _fresh_bridge()
    tab = SymbolOnboardingTab(bridge)

    db_models = tab._assignable_models_for("double_bottom")
    assert [m.model_id for m in db_models] == ["double_bottom_xauusd_m15_v1"]
    assert tab._assignable_models_for("falling_wedge") == []
    assert tab._assignable_models_for("inverse_head_shoulders") == []

    # §10.1 lifecycle badge palette still maps every registry state
    for st in ("validated", "shadow", "live", "degraded", "retired", "trained"):
        assert tab._lifecycle_color(st) != "?"


def test_onboarding_tab_renders_persisted_assignments(qapp, assignments_file_clean) -> None:
    bridge = _fresh_bridge()
    bridge.set_pattern_assignment(
        "XAUUSD", "double_bottom@M15",
        pattern_name="double_bottom", timeframe="M15", state="live",
        model_id="double_bottom_xauusd_m15_v1",
        feature_schema_version="double-v1.0",
    )
    tab = SymbolOnboardingTab(bridge)
    tab._refresh_pattern_assignments()
    assert tab._pattern_table.rowCount() >= 1
    # symbol column shows the assigned symbol
    symbols = [tab._pattern_table.item(r, 0).text() for r in range(tab._pattern_table.rowCount())]
    assert "XAUUSD" in symbols


# ---------------------------------------------------------------------------
# Assignment persistence across bridge restarts
# ---------------------------------------------------------------------------

def test_assignment_persists_and_restores_across_bridge_restart(
    qapp, assignments_file_clean,
) -> None:
    bridge = _fresh_bridge()
    bridge.set_pattern_assignment(
        "XAUUSD", "double_bottom@M15",
        pattern_name="double_bottom", timeframe="M15", state="live",
        model_id="double_bottom_xauusd_m15_v1",
        feature_schema_version="double-v1.0",
    )
    assert _ASSIGNMENTS_FILE.is_file()

    # simulate a fresh GUI boot: a NEW bridge reads the persisted store
    bridge2 = _fresh_bridge()
    restored = bridge2.get_pattern_assignments("XAUUSD")
    assert len(restored) == 1
    a = restored[0]
    assert a["model_id"] == "double_bottom_xauusd_m15_v1"
    assert a["feature_schema_version"] == "double-v1.0"
    assert a["pattern_name"] == "double_bottom"
    assert a["state"] == "live"

    # removal persists too
    bridge2.remove_pattern_assignment("XAUUSD", "double_bottom@M15")
    assert bridge2.get_pattern_assignments("XAUUSD") == []
    bridge3 = _fresh_bridge()
    assert bridge3.get_pattern_assignments("XAUUSD") == []


# ---------------------------------------------------------------------------
# Assignment consumed by MultiPatternEngine (§9.1) — model_id rides through
# ---------------------------------------------------------------------------

def test_assignment_is_consumed_by_multi_pattern_engine(
    qapp, assignments_file_clean,
) -> None:
    bridge = _fresh_bridge()
    bridge.set_pattern_assignment(
        "XAUUSD", "double_bottom@M15",
        pattern_name="double_bottom", timeframe="M15", state="live",
        model_id="double_bottom_xauusd_m15_v1",
        feature_schema_version="double-v1.0",
    )
    stored = bridge.get_pattern_assignments("XAUUSD")[0]

    db_event = _ev("DB-REG-0001", "double_bottom", entry_price=2650.0,
                   rule_score=60.0, model_prob=0.8)
    det_cls = _make_detector_class("double_bottom", "DB")
    assignment = PatternAssignment(
        assignment_id=stored["assignment_id"],
        pattern_name=stored["pattern_name"],
        timeframe=stored["timeframe"],
        state=stored["state"],
        model_id=stored["model_id"],
        feature_schema_version=stored.get("feature_schema_version", ""),
        detector=det_cls([db_event]),
    )

    engine = MultiPatternEngine(
        symbol="XAUUSD",
        assignments=[assignment],
        candle_fn=_CandleSource(_candles(close=2650.0)).get_chart_history,
    )
    candidates = engine.check_new_bar()
    assert len(candidates) == 1
    # the assignment's model id + schema version are consumed by the engine
    assert engine.assignments[0].model_id == "double_bottom_xauusd_m15_v1"
    assert engine.assignments[0].feature_schema_version == "double-v1.0"
    # every candidate traces back to the assignment_id → model lineage
    assert candidates[0].assignment_id == "double_bottom@M15"
    assert candidates[0].pattern_name == "double_bottom"


# ---------------------------------------------------------------------------
# Legacy symbol-level back-compat (assign_model_to_symbol)
# ---------------------------------------------------------------------------

def test_back_compat_assign_model_to_symbol(qapp, assignments_file_clean) -> None:
    bridge = _fresh_bridge()
    bridge.state.register_symbol(
        "ZZSYM_BT",
        SymbolConfig(name="ZZSYM_BT", status="validated", active=False, model_id=None),
    )
    try:
        ok = bridge.assign_model_to_symbol("ZZSYM_BT", "xauusd_v2_h16_20260905")
        assert ok is True
        cfg = bridge.state.get_symbol_config("ZZSYM_BT")
        assert cfg is not None and cfg.model_id == "xauusd_v2_h16_20260905"

        # retired alias is still assignable through the legacy path
        ok2 = bridge.assign_model_to_symbol("ZZSYM_BT", "xauusd_v2")
        assert ok2 is True
        cfg2 = bridge.state.get_symbol_config("ZZSYM_BT")
        assert cfg2 is not None and cfg2.model_id == "xauusd_v2"

        # unknown model id is rejected (no crash)
        ok3 = bridge.assign_model_to_symbol("ZZSYM_BT", "missing_model")
        assert ok3 is False
    finally:
        bridge.state.unregister_symbol("ZZSYM_BT")
        bridge.state.save_registry()  # persist cleanup