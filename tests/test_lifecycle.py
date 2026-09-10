"""LifecycleManager tests — Pattern Lifecycle & Shadow Mode (§5, Agent 6 t9).

Covers REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §5:

  1. §5.1  state machine transitions (forward order + retrain edge; illegal
     jumps raise);
  2. §5.2  shadow gate — all four conditions (sample size OR max-days, live PF
     inside backtest OOS CI +/-1sigma, drift healthy, zero CausalityViolation)
     must hold to promote shadow → live; severe drift sinks shadow → degraded;
  3. §5.3  auto-demotion triggers —
       * live PF CI lower < breakeven after >= live_min_trades → live -> degraded
       * PSI drift >= 0.25 on any feature → live -> degraded (+ needs_retrain
         when >= 0.2 on >= 2 core features)
       * win-rate drop p < 0.05 (binomial vs backtest WR) → live -> degraded
       * consecutive losses >= max → pause assignment (no state change)
  4. §5.4  per-pattern kill-switch retires one pattern without touching the
     whole symbol.

Marker: no_lookahead (decisions use only past events/outcomes).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from live.engine.lifecycle_manager import (
    LifecycleConfig,
    LifecycleManager,
    Transition,
    valid_transition,
)
from research.core.contracts import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_LIVE,
    LIFECYCLE_RETIRED,
    LIFECYCLE_SHADOW,
    LIFECYCLE_TRAINED,
    LIFECYCLE_VALIDATED,
)
from research.core.event_lake import DriftReport, FeatureDrift

pytestmark = pytest.mark.no_lookahead

BASE = dict(
    symbol="XAUUSD",
    pattern_name="double_bottom",
    timeframe="M15",
    model_id="db_xauusd_m15_v1",
)


def _manager(**cfg: object) -> LifecycleManager:
    return LifecycleManager(LifecycleConfig(**cfg))


def _ok_drift() -> DriftReport:
    return DriftReport(per_feature=[FeatureDrift("atr", 0.02, "none")])


def _bad_drift() -> DriftReport:
    return DriftReport(per_feature=[FeatureDrift("atr", 0.30, "drift")])


def _retrain_drift() -> DriftReport:
    return DriftReport(
        per_feature=[
            FeatureDrift("atr", 0.22, "drift"),
            FeatureDrift("reclaim_atr", 0.23, "drift"),
        ]
    )


def _shadow_pass_kwargs(drift: DriftReport | None = None) -> dict:
    return dict(
        **BASE,
        current_state=LIFECYCLE_SHADOW,
        shadow_started_at=datetime.now(timezone.utc) - timedelta(days=5),
        n_shadow_events=40,          # >= shadow_min_events (30)
        live_pf=1.35,
        oos_pf=1.30,
        oos_pf_std=0.15,             # 1.30 - 0.15 = 1.15 <= 1.35 → within CI
        drift=drift if drift is not None else _ok_drift(),
        causality_violations=0,
    )


# ---------------------------------------------------------------------------
# §5.1 state machine
# ---------------------------------------------------------------------------

def test_valid_forward_chain() -> None:
    chain = [
        (LIFECYCLE_TRAINED, LIFECYCLE_VALIDATED),
        (LIFECYCLE_VALIDATED, LIFECYCLE_SHADOW),
        (LIFECYCLE_SHADOW, LIFECYCLE_LIVE),
        (LIFECYCLE_LIVE, LIFECYCLE_DEGRADED),
        (LIFECYCLE_DEGRADED, LIFECYCLE_RETIRED),
    ]
    for f, t in chain:
        assert valid_transition(f, t), f"{f} -> {t}"


def test_retrain_edge() -> None:
    assert valid_transition(LIFECYCLE_LIVE, LIFECYCLE_VALIDATED, retrain=True)
    assert valid_transition(LIFECYCLE_DEGRADED, LIFECYCLE_VALIDATED, retrain=True)
    assert valid_transition(LIFECYCLE_SHADOW, LIFECYCLE_VALIDATED, retrain=True)


def test_illegal_jumps_rejected() -> None:
    assert not valid_transition(LIFECYCLE_TRAINED, LIFECYCLE_LIVE)
    assert not valid_transition(LIFECYCLE_VALIDATED, LIFECYCLE_DEGRADED)
    assert not valid_transition(LIFECYCLE_LIVE, LIFECYCLE_SHADOW)
    assert not valid_transition(LIFECYCLE_LIVE, LIFECYCLE_VALIDATED)  # needs retrain=True


def test_transition_raises_on_illegal() -> None:
    from live.engine.lifecycle_manager import LifecycleError

    mgr = _manager()
    with pytest.raises(LifecycleError):
        mgr.transition(LIFECYCLE_TRAINED, LIFECYCLE_LIVE, "jump")


# ---------------------------------------------------------------------------
# §5.2 shadow gate
# ---------------------------------------------------------------------------

def test_shadow_gate_promotes_when_all_conditions_met() -> None:
    mgr = _manager()
    v = mgr.evaluate_shadow(**_shadow_pass_kwargs())
    assert v.final_state == LIFECYCLE_LIVE
    assert v.transitions[-1].reason == "shadow gate passed (§5.2)"
    assert not v.flags.get("gate_blocker")


def test_shadow_gate_blocks_when_sample_too_small() -> None:
    mgr = _manager()
    kw = _shadow_pass_kwargs()
    kw["n_shadow_events"] = 5
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_SHADOW
    assert "shadow sample too small" in v.flags["gate_blocker"]


def test_shadow_gate_blocks_when_pf_outside_ci() -> None:
    mgr = _manager()
    kw = _shadow_pass_kwargs()
    kw["live_pf"] = 0.9       # < oos 1.30 - std 0.15 = 1.15
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_SHADOW
    assert "live PF outside backtest OOS CI" in v.flags["gate_blocker"]


def test_shadow_gate_blocks_on_causality_violation() -> None:
    mgr = _manager()
    kw = _shadow_pass_kwargs()
    kw["causality_violations"] = 2
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_SHADOW
    assert "CausalityViolation" in v.flags["gate_blocker"]


def test_shadow_gate_blocks_on_drift() -> None:
    mgr = _manager()
    kw = _shadow_pass_kwargs(drift=_retrain_drift())
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_SHADOW
    assert "feature drift" in v.flags["gate_blocker"]


def test_shadow_severe_drift_sinks_to_degraded() -> None:
    mgr = _manager()
    kw = _shadow_pass_kwargs(drift=_bad_drift())
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_DEGRADED


def test_shadow_max_days_gate() -> None:
    """shadow_max_days reached -> sample condition satisfied even with few
    events (whichever comes first, §5.2)."""
    mgr = _manager(shadow_max_days=7)
    kw = _shadow_pass_kwargs()
    kw["n_shadow_events"] = 10  # below 30 but max allows
    kw["shadow_started_at"] = datetime.now(timezone.utc) - timedelta(days=10)
    v = mgr.evaluate_shadow(**kw)
    assert v.final_state == LIFECYCLE_LIVE


# ---------------------------------------------------------------------------
# §5.3 auto-demotion triggers
# ---------------------------------------------------------------------------

def test_demotion_pf_ci_lower_breakeven() -> None:
    mgr = _manager(live_min_trades=30)
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=35, live_pf_ci_lower=0.85,
        live_win_rate=0.4, backtest_win_rate=0.55,
        consec_losses=2, drift=_ok_drift(),
    )
    assert v.final_state == LIFECYCLE_DEGRADED
    assert any("live PF CI lower" in t.reason for t in v.transitions)


def test_demotion_pf_not_before_min_trades() -> None:
    mgr = _manager(live_min_trades=30)
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=10, live_pf_ci_lower=0.5,
        live_win_rate=0.5, backtest_win_rate=0.5,
        consec_losses=2, drift=_ok_drift(),
    )
    assert v.final_state == LIFECYCLE_LIVE  # under min trades → no demotion


def test_demotion_drift_flags_retrain_only() -> None:
    """§5.3: PSI >= 0.2 on >= 2 core features → retrain flag + GUI warning,
    but NOT degraded (threshold is 0.25)."""
    mgr = _manager()
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=40, live_pf_ci_lower=1.4,
        live_win_rate=0.5, backtest_win_rate=0.5,
        consec_losses=0, drift=_retrain_drift(),  # 0.22 / 0.23
    )
    assert v.needs_retrain is True
    assert v.final_state == LIFECYCLE_LIVE


def test_demotion_drift_demotes_on_025() -> None:
    """§5.3: any feature PSI >= 0.25 → live → degraded."""
    mgr = _manager()
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=40, live_pf_ci_lower=1.4,
        live_win_rate=0.5, backtest_win_rate=0.5,
        consec_losses=0, drift=_bad_drift(),  # 0.30
    )
    assert v.final_state == LIFECYCLE_DEGRADED
    assert any("feature drift" in t.reason for t in v.transitions)


def test_demotion_winrate_drop() -> None:
    mgr = _manager(wr_p_value=0.05)
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=50, live_pf_ci_lower=1.3,
        live_win_rate=0.10, backtest_win_rate=0.55,  # big drop → p tiny
        consec_losses=0, drift=_ok_drift(),
    )
    assert v.final_state == LIFECYCLE_DEGRADED
    assert any("win-rate drop" in t.reason for t in v.transitions)


def test_demotion_winrate_no_false_positive() -> None:
    mgr = _manager(wr_p_value=0.05)
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=50, live_pf_ci_lower=1.4,
        live_win_rate=0.5, backtest_win_rate=0.5,
        consec_losses=0, drift=_ok_drift(),
    )
    assert v.final_state == LIFECYCLE_LIVE
    assert not v.transitions


def test_demotion_consec_losses_pauses() -> None:
    mgr = _manager(max_consec_losses=5)
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=30, live_pf_ci_lower=1.3,
        live_win_rate=0.5, backtest_win_rate=0.5,
        consec_losses=6, drift=_ok_drift(),
    )
    assert v.paused is True
    assert v.final_state == LIFECYCLE_LIVE  # state unchanged
    assert "consecutive losses" in v.flags["paused_reason"]


def test_demotion_no_trigger_stays_live() -> None:
    mgr = _manager()
    v = mgr.evaluate_demotion(
        **BASE, current_state=LIFECYCLE_LIVE,
        n_live_trades=40, live_pf_ci_lower=1.4,
        live_win_rate=0.55, backtest_win_rate=0.55,
        consec_losses=1, drift=_ok_drift(),
    )
    assert v.final_state == LIFECYCLE_LIVE
    assert not v.transitions
    assert not v.paused


# ---------------------------------------------------------------------------
# §5.4 per-pattern kill switch
# ---------------------------------------------------------------------------

def test_kill_pattern_retires_one_pattern() -> None:
    t = LifecycleManager.kill_pattern(LIFECYCLE_LIVE, "manual kill")
    assert isinstance(t, Transition)
    assert t.to_state == LIFECYCLE_RETIRED
    assert t.from_state == LIFECYCLE_LIVE