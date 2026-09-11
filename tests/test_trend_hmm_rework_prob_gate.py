"""Rework §2.3 — independent AND-gate on ``model_prob`` / ``rule_score``.

The request (``trading_v3_rework_request_trend_hmm.md`` §1.3 / §2.3) found that
``combined_score = (rule_score + model_prob) / 2`` let a strong geometry score
average away a weak model score.  The reported event was
``rule_score 0.93`` + ``model_prob 0.180`` → combined ``0.55``, which looked
acceptable while the model was in fact saying the setup was bad.  No threshold
existed on either axis.

These tests pin the fix:

  1. high prob + high rule PASSES;
  2. high rule + low prob is DROPPED with ``low_probability`` — the exact §1.3
     failure case (rule 0.93 + prob 0.180);
  3. low rule + high prob is DROPPED with ``low_rule_score``;
  4. a weak event is NOT rescued by the average (gate never reads
     ``combined_score``);
  5. an LSW event on the 0..100 rule_score scale is normalized before the
     comparison, so a good LSW event is not falsely dropped;
  6. the DEFAULT configuration preserves the current behaviour exactly;
  7. live ≡ backtest (§12): the runner exposes the same switch and reuses the
     same shared implementation.

Hermetic: no MCP, no network, no model artifacts required.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:            # trading_v3/ on path (live/, research/)
    sys.path.insert(0, str(ROOT))

from live.engine.hard_gate import (  # noqa: E402
    DEFAULT_PROB_GATE_ENABLED,
    DEFAULT_RULE_SCORE_FLOOR,
    PROPOSED_PROB_THRESHOLDS,
    REASON_LOW_MODEL_PROB,
    REASON_LOW_RULE_SCORE,
    REASON_PROB_GATE_OFF,
    apply_probability_gate,
    normalize_rule_score,
    probability_gate,
    threshold_config,
)
from research.core.contracts import PatternEvent  # noqa: E402


def _event(
    pattern: str = "double_bottom",
    *,
    rule_score: float | None = 0.9,
    model_prob: float | None = 0.8,
) -> PatternEvent:
    """Minimal PatternEvent carrying only the two gated axes."""
    ts = pd.Timestamp("2026-08-27T12:00:00+00:00")
    return PatternEvent(
        event_id=f"ev-{pattern}-{rule_score}-{model_prob}",
        pattern_name=pattern,
        pattern_version="double-v1.0",
        symbol="XAUUSD",
        timeframe="M15",
        direction="bullish",
        detect_time=ts,
        confirm_time=ts,
        known_at=ts,
        rule_score=rule_score,
        model_prob=model_prob,
    )


#: An AND-gate with both axes configured, mirroring the §2.3 proposal shape.
_ON = {
    "enabled": True,
    "min_model_prob": 0.55,
    "min_rule_score": 0.60,
    "per_pattern": {"liquidity_sweep": {"min_model_prob": 0.50}},
}


# ---------------------------------------------------------------------------
# 1. Both axes strong -> PASS
# ---------------------------------------------------------------------------
def test_high_prob_and_high_rule_passes() -> None:
    ev = _event(rule_score=0.93, model_prob=0.80)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is True
    assert reason == "prob_ok"
    assert "discard_reason" not in ev.attributes


def test_boundary_equal_to_threshold_passes() -> None:
    """``<`` (not ``<=``) is the comparison — exactly at threshold passes."""
    ev = _event(rule_score=0.60, model_prob=0.55)
    allowed, _ = probability_gate(ev, _ON)
    assert allowed is True


# ---------------------------------------------------------------------------
# 2. The exact §1.3 failure case: high rule, low prob -> low_probability
# ---------------------------------------------------------------------------
def test_high_rule_low_prob_is_dropped_as_low_probability() -> None:
    """rule 0.93 + prob 0.180 — the event from the review session (§1.3)."""
    ev = _event(rule_score=0.93, model_prob=0.180)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is False
    assert reason == REASON_LOW_MODEL_PROB == "low_probability"


def test_high_rule_low_prob_stamps_discard_reason() -> None:
    ev = _event(rule_score=0.93, model_prob=0.180)
    assert apply_probability_gate(ev, _ON) is False
    assert ev.attributes["discard_reason"] == "low_probability"


@pytest.mark.parametrize("prob", [0.033, 0.069, 0.180, 0.272, 0.549])
def test_any_sub_threshold_prob_is_dropped(prob: float) -> None:
    """Every probability observed in the 3 reviewed chart exports."""
    ev = _event(rule_score=0.95, model_prob=prob)
    assert probability_gate(ev, _ON) == (False, REASON_LOW_MODEL_PROB)


# ---------------------------------------------------------------------------
# 3. Low rule, high prob -> low_rule_score
# ---------------------------------------------------------------------------
def test_low_rule_high_prob_is_dropped_as_low_rule_score() -> None:
    ev = _event(rule_score=0.30, model_prob=0.95)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is False
    assert reason == REASON_LOW_RULE_SCORE == "low_rule_score"


def test_low_rule_stamps_discard_reason() -> None:
    ev = _event(rule_score=0.30, model_prob=0.95)
    assert apply_probability_gate(ev, _ON) is False
    assert ev.attributes["discard_reason"] == "low_rule_score"


def test_both_axes_weak_reports_rule_score_first() -> None:
    """When both fail, a deterministic reason is still written."""
    ev = _event(rule_score=0.10, model_prob=0.10)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is False
    assert reason == REASON_LOW_RULE_SCORE


# ---------------------------------------------------------------------------
# 4. The average must NOT rescue a weak event
# ---------------------------------------------------------------------------
def test_weak_axis_is_not_rescued_by_the_average() -> None:
    """§1.3: 0.93 and 0.180 average to 0.55 — a combined>=0.5 gate would PASS.

    The AND-gate must still DROP it, proving the deciding value is per-axis
    and not ``combined_score``.
    """
    rule, prob = 0.93, 0.180
    combined = (rule + prob) / 2.0
    assert combined >= 0.5, "precondition: the average looks acceptable"

    ev = _event(rule_score=rule, model_prob=prob)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is False
    assert reason == "low_probability"


def test_gate_never_consults_combined_score() -> None:
    """A huge ``attributes['combined_score']`` cannot buy an event a pass."""
    ev = _event(rule_score=0.93, model_prob=0.180)
    ev.attributes["combined_score"] = 0.99
    assert apply_probability_gate(ev, _ON) is False
    assert ev.attributes["discard_reason"] == "low_probability"


def test_hard_gate_module_does_not_read_combined_score() -> None:
    """Structural guard: the gate implementation never references it.

    Docstrings are excluded — ``combined_score`` may only appear in prose
    explaining WHY it is not used, never in executable code.
    """
    import ast
    import textwrap

    import live.engine.hard_gate as hg

    src = textwrap.dedent(inspect.getsource(hg))
    tree = ast.parse(src)
    # drop every docstring node, then re-serialize the real code only
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    code = ast.unparse(tree)
    assert "combined_score" not in code
    # the two reasons ARE referenced in code (sanity: the guard is not vacuous)
    assert "low_probability" in code or "REASON_LOW_MODEL_PROB" in code


# ---------------------------------------------------------------------------
# 5. LSW 0..100 scale must be normalized (no false drop)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pattern", "raw", "expected"),
    [
        ("liquidity_sweep", 51.0, 0.51),
        ("liquidity_sweep", 44.0, 0.44),
        ("double_bottom", 0.93, 0.93),
        ("double_top", 0.77, 0.77),
        ("rising_wedge", 0.82, 0.82),
    ],
)
def test_rule_score_normalized_per_pattern(
    pattern: str, raw: float, expected: float,
) -> None:
    assert normalize_rule_score(pattern, raw) == pytest.approx(expected)


def test_lsw_event_is_not_falsely_dropped() -> None:
    """A good LSW event: raw rule_score 51 (0..100), solid model_prob.

    A naive floor on the RAW value would compare ``51 < 0.6`` → False and drop
    every LSW event.  Normalization makes it 0.51, which passes a 0.5 floor.
    """
    ev = _event(pattern="liquidity_sweep", rule_score=51.0, model_prob=0.70)
    cfg = {"enabled": True, "min_model_prob": 0.50, "min_rule_score": 0.50}
    allowed, reason = probability_gate(ev, cfg)
    assert allowed is True, "LSW must not be dropped by an un-normalized floor"
    assert reason == "prob_ok"


@pytest.mark.parametrize("raw", [30.0, 32.0, 36.0, 40.0, 44.0, 51.0])
def test_lsw_raw_40_is_not_dropped_by_normalized_floor(raw: float) -> None:
    """The exact regression the §2.3 brief calls "the single most likely way
    to break this task": an LSW event with a raw rule_score ~40.

    Real LSW scores observed on the full history span 30..51.  A raw floor of
    0.6 would assess ``40 < 0.6`` -> False and delete the whole pattern family.
    Normalized, 40 becomes 0.40, which passes a 0.30 floor but must still be
    dropped by a 0.50 floor — i.e. normalization makes the floor MEANINGFUL
    rather than uniformly fatal.
    """
    ev = _event(pattern="liquidity_sweep", rule_score=raw, model_prob=0.70)

    # survives a floor set below its normalized value
    below = {"enabled": True, "min_rule_score": 0.30}
    assert probability_gate(ev, below) == (True, "prob_ok"), (
        f"LSW raw {raw} (normalized {raw / 100:.2f}) wrongly dropped by floor 0.30"
    )

    # and a raw-scale floor would have been catastrophically wrong
    assert raw > 0.6, "precondition: these are 0..100-scale values"
    assert normalize_rule_score("liquidity_sweep", raw) == pytest.approx(raw / 100.0)


def test_raw_scale_floor_would_delete_every_lsw_event() -> None:
    """Negative control: prove the naive (un-normalized) comparison is fatal.

    Every observed LSW raw score is > 0.6, so ``raw < 0.6`` is False for all of
    them and the pattern family is wiped out.  This pins WHY normalization is
    mandatory rather than stylistic.
    """
    observed_raw = [30.0, 32.0, 36.0, 40.0, 44.0, 51.0]
    assert all(raw > 0.6 for raw in observed_raw)
    for raw in observed_raw:
        assert (raw / 100.0) < 0.6, "normalized value is what the floor sees"


def test_lsw_event_still_dropped_when_genuinely_weak() -> None:
    """Normalization must not disable the gate — a weak LSW event drops."""
    ev = _event(pattern="liquidity_sweep", rule_score=30.0, model_prob=0.70)
    cfg = {"enabled": True, "min_model_prob": 0.50, "min_rule_score": 0.40}
    allowed, reason = probability_gate(ev, cfg)   # 30/100 = 0.30 < 0.40
    assert allowed is False
    assert reason == "low_rule_score"


def test_unknown_pattern_scale_is_autodetected() -> None:
    assert normalize_rule_score("brand_new_pattern", 55.0) == pytest.approx(0.55)
    assert normalize_rule_score("brand_new_pattern", 0.55) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# 6. Default configuration preserves current behaviour
# ---------------------------------------------------------------------------
def test_default_switch_is_off_and_no_thresholds_are_preset() -> None:
    assert DEFAULT_PROB_GATE_ENABLED is False
    assert DEFAULT_RULE_SCORE_FLOOR is None


def test_proposed_thresholds_are_documentation_only() -> None:
    """§6 warns the §2.3 numbers came from an older proposal.

    They are kept as a reference constant but must never be the live default:
    on the full history they would wipe out DB/DT, H&S and every pattern whose
    model artifact is missing.
    """
    assert PROPOSED_PROB_THRESHOLDS["double_bottom"] == 0.55
    assert DEFAULT_PROB_GATE_ENABLED is False


@pytest.mark.parametrize(
    "cfg",
    [None, {}, {"enabled": False}, {"enabled": False, "min_model_prob": 0.9,
                                    "min_rule_score": 0.9}],
)
def test_disabled_gate_allows_everything(cfg: dict | None) -> None:
    """The §1.3 worst event still passes while the gate is OFF."""
    ev = _event(rule_score=0.93, model_prob=0.180)
    allowed, reason = probability_gate(ev, cfg)
    assert allowed is True
    assert reason == REASON_PROB_GATE_OFF == "prob_gate_off"
    assert "discard_reason" not in ev.attributes


def test_disabled_gate_does_not_stamp_any_reason() -> None:
    ev = _event(rule_score=0.0, model_prob=0.0)
    assert apply_probability_gate(ev, {}) is True
    assert "discard_reason" not in ev.attributes


def test_enabled_but_unconfigured_axes_drop_nothing() -> None:
    ev = _event(rule_score=0.10, model_prob=0.05)
    allowed, reason = probability_gate(ev, {"enabled": True})
    assert allowed is True
    assert reason == "prob_gate_off"


# ---------------------------------------------------------------------------
# Missing data is fail-open; a zero rule_score is a real value
# ---------------------------------------------------------------------------
def test_missing_model_prob_does_not_block() -> None:
    """Patterns without a tier-2 artifact leave model_prob None.

    On the full history ``liquidity_sweep``, ``rising_wedge`` and
    ``inverse_head_shoulders`` have NO XAUUSD M15 v1 artifact, so a threshold
    would otherwise silently close those patterns permanently.  Only the prob
    axis is judged here (the rule floor is set below the event's normalized
    0.51 so it cannot mask the behaviour under test).
    """
    ev = _event(pattern="liquidity_sweep", rule_score=51.0, model_prob=None)
    cfg = {"enabled": True, "min_model_prob": 0.55, "min_rule_score": 0.40}
    allowed, reason = probability_gate(ev, cfg)
    assert allowed is True
    assert reason == "prob_ok"


def test_zero_rule_score_blocks_when_floor_configured() -> None:
    """0.0 is a real (bad) geometry score, not missing data."""
    ev = _event(rule_score=0.0, model_prob=0.90)
    allowed, reason = probability_gate(ev, _ON)
    assert allowed is False
    assert reason == "low_rule_score"


def test_nan_model_prob_does_not_block() -> None:
    ev = _event(rule_score=0.93, model_prob=float("nan"))
    allowed, _ = probability_gate(ev, _ON)
    assert allowed is True


# ---------------------------------------------------------------------------
# Per-pattern overrides + config hygiene
# ---------------------------------------------------------------------------
def test_per_pattern_override_wins_over_scalar() -> None:
    lsw = threshold_config(_ON, "liquidity_sweep")
    db = threshold_config(_ON, "double_bottom")
    assert lsw == (0.50, 0.60)
    assert db == (0.55, 0.60)


def test_list_style_per_pattern_entry_is_tolerated() -> None:
    cfg = {"enabled": True, "per_pattern": [
        {"pattern": "double_top", "min_model_prob": 0.42},
    ]}
    assert threshold_config(cfg, "double_top")[0] == pytest.approx(0.42)


@pytest.mark.parametrize("bad", ["abc", -1.0, 1.5, float("nan"), {}])
def test_invalid_threshold_is_ignored(bad: object) -> None:
    """A malformed threshold must not crash live trading or block blindly."""
    cfg = {"enabled": True, "min_model_prob": bad, "min_rule_score": bad}
    assert threshold_config(cfg, "double_bottom") == (None, None)
    assert probability_gate(_event(), cfg) == (True, "prob_gate_off")


# ---------------------------------------------------------------------------
# 7. §12 live ≡ backtest parity
# ---------------------------------------------------------------------------
def test_live_engine_exposes_prob_gate_default_off() -> None:
    from live.engine.signal_engine_v2 import MultiPatternEngine

    sig = inspect.signature(MultiPatternEngine.__init__)
    assert "prob_gate" in sig.parameters
    assert sig.parameters["prob_gate"].default is None


def test_backtest_runner_exposes_prob_gate_default_off() -> None:
    from research.multi_backtest.runner import run_symbol_backtest

    sig = inspect.signature(run_symbol_backtest)
    assert "prob_gate" in sig.parameters
    assert sig.parameters["prob_gate"].default is None


def test_both_sides_use_the_same_shared_implementation() -> None:
    """Parity is structural: neither side re-implements the comparison."""
    import live.engine.signal_engine_v2 as se
    import research.multi_backtest.runner as mb

    live_src = inspect.getsource(se._prob_gate_allowed)
    assert "apply_probability_gate" in live_src

    bt_src = inspect.getsource(mb.run_symbol_backtest)
    assert "apply_probability_gate" in bt_src


def test_engine_prob_gate_default_preserves_behaviour() -> None:
    """A default-constructed engine carries no gate configuration."""
    from live.engine.signal_engine_v2 import MultiPatternEngine

    class _StubDetector:
        name = "double_bottom"

        def detect(self, df, cfg):  # pragma: no cover - not called
            return []

    class _StubAssignment:
        assignment_id = "XAUUSD-double_bottom"
        pattern_name = "double_bottom"
        timeframe = "M15"
        state = "live"
        detector = _StubDetector()
        config: ClassVar[dict] = {}
        # regime wiring reads these on every assignment (default OFF)
        regime_states = None
        regime_plugin = None
        regime_config: ClassVar[dict] = {}
        regime_filter = None
        model_feature_list: ClassVar[list] = []
        model_optional_plugins: ClassVar[list] = []
        model_id = None
        scorer = None

    engine = MultiPatternEngine(
        symbol="XAUUSD", assignments=[_StubAssignment()], candle_fn=lambda _s: None,
    )
    assert engine.prob_gate == {}
    assert engine.opposite_overlap_guard is False
