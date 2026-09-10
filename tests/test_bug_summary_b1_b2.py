"""Regression tests for the B1/B2 defects in ``trading_v3_bug_summary.md``.

B1 — no minimum-probability guard in the multi-pattern signal path.
B2 — ``combined_score`` divided ``rule_score`` by 100 for EVERY pattern, but
     only ``liquidity_sweep`` uses the 0..100 scale (DB/DT clamp to [0, 1]),
     so every DB/DT event was scored as if its geometry were worthless
     (0.91 → 0.0091).

These tests pin the fixed behaviour and are hermetic (no MCP, no network).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # trading_v3/ on path (live/, research/)
    sys.path.insert(0, str(ROOT))

from live.engine.signal_engine_v2 import (  # noqa: E402
    _combine_score,
    _normalize_rule_score,
    _resolve_min_model_prob,
)


# ---------------------------------------------------------------------------
# B2 — per-pattern rule_score normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pattern", "raw", "expected"),
    [
        ("liquidity_sweep", 40.0, 0.40),     # LSW declares 0..100
        ("liquidity_sweep", 51.0, 0.51),
        ("liquidity_sweep", 0.0, 0.0),
        ("double_bottom", 0.91, 0.91),       # DB/DT already 0..1
        ("double_top", 0.51, 0.51),
        ("double_bottom", 1.0, 1.0),
        ("rising_wedge", 0.42, 0.42),
    ],
)
def test_rule_score_normalized_per_pattern(
    pattern: str, raw: float, expected: float,
) -> None:
    assert _normalize_rule_score(pattern, raw) == pytest.approx(expected)


def test_unknown_pattern_scale_is_autodetected() -> None:
    """A future plugin emitting 0..100 must still normalize correctly."""
    assert _normalize_rule_score("brand_new_pattern", 55.0) == pytest.approx(0.55)
    assert _normalize_rule_score("brand_new_pattern", 0.55) == pytest.approx(0.55)


@pytest.mark.parametrize("bad", [None, float("nan"), "abc", {}])
def test_rule_score_garbage_is_zero(bad: object) -> None:
    assert _normalize_rule_score("double_bottom", bad) == 0.0


def test_rule_score_is_clamped_to_unit_interval() -> None:
    # raw > 1 on a [0,1]-scale pattern is auto-detected as a 0..100 value
    assert _normalize_rule_score("double_bottom", 5.0) == pytest.approx(0.05)
    assert _normalize_rule_score("liquidity_sweep", 999.0) == pytest.approx(1.0)
    assert _normalize_rule_score("liquidity_sweep", -3.0) == 0.0


def test_combined_score_no_longer_shrinks_db_events() -> None:
    """The exact event from the bug report: DB, rule 0.91, prob 0.16."""
    db = _combine_score(_normalize_rule_score("double_bottom", 0.91), 0.16)
    assert db == pytest.approx(0.535)
    # the buggy formula produced 0.0846 — pin that it is gone for DB/DT
    buggy = round((0.91 / 100.0 + 0.16) / 2.0, 4)
    assert db > buggy * 6


def test_combined_score_lsw_unchanged() -> None:
    """LSW keeps its historical value: (40/100 + 0.5)/2 = 0.45."""
    lsw = _combine_score(_normalize_rule_score("liquidity_sweep", 40.0), 0.5)
    assert lsw == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# B1 — optional minimum model-probability gate
# ---------------------------------------------------------------------------
class _FakeAssignment:
    def __init__(self, config: dict, pattern: str = "double_bottom") -> None:
        self.config = config
        self.pattern_name = pattern


def test_min_model_prob_absent_means_no_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import live.engine.signal_engine_v2 as se
    monkeypatch.setattr(se, "_DEFAULT_MIN_MODEL_PROB", {}, raising=False)
    assert _resolve_min_model_prob(_FakeAssignment({})) is None


def test_min_model_prob_from_assignment_config() -> None:
    assert _resolve_min_model_prob(
        _FakeAssignment({"min_model_prob": 0.55})) == pytest.approx(0.55)


def test_min_model_prob_from_module_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import live.engine.signal_engine_v2 as se
    monkeypatch.setattr(
        se, "_DEFAULT_MIN_MODEL_PROB", {"double_bottom": 0.4}, raising=False)
    assert _resolve_min_model_prob(
        _FakeAssignment({"pattern_name": "double_bottom"})) == pytest.approx(0.4)
    # a different pattern is unaffected
    assert _resolve_min_model_prob(
        _FakeAssignment({}, pattern="double_top")) is None


@pytest.mark.parametrize("bad", ["abc", -1.0, 1.5, float("nan")])
def test_invalid_min_model_prob_is_ignored(bad: object) -> None:
    assert _resolve_min_model_prob(_FakeAssignment({"min_model_prob": bad})) is None


def test_low_probability_event_is_discarded_by_gate() -> None:
    """Simulate the gate decision exactly as ``_group_to_candidate`` does."""
    threshold = _resolve_min_model_prob(_FakeAssignment({"min_model_prob": 0.5}))
    assert threshold is not None
    for prob, blocked in [(0.033, True), (0.16, True), (0.499, True),
                          (0.5, False), (0.9, False)]:
        assert (prob < threshold) is blocked
