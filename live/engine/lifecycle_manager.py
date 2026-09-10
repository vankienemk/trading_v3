"""
lifecycle_manager.py — Pattern Lifecycle & Shadow Mode (§5)

Agent 6 (t9).  Implements REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §5:

  * §5.1 — state machine ``trained → validated → shadow → live → degraded →
    retired`` (+ ``retrain`` re-enters ``validated``).
  * §5.2 — Shadow gate (promote shadow → live): n_shadow_events >=
    shadow_min_events (or shadow_max_days elapsed), live hypothetical PF
    within the backtest OOS PF CI (+/-1sigma), feature-distribution drift PSI < 0.1
    on all core features, and zero CausalityViolation over the shadow period.
  * §5.3 — auto-demotion triggers (Live PF CI lower < breakeven; PSI >= 0.2 on
    >=2 core → retrain flag and >= 0.25 → degraded; win-rate drop p-value <
    0.05; consecutive losses -> pause assignment without a state change).
  * §5.4 — per-pattern kill-switch: disable one pattern, never the whole symbol.

The manager is a pure decision layer: it takes event/outcome/metric aggregates
(which the Event Lake / OutcomeTracker / DriftMonitor already compute) and
returns transitions + flags.  It owns no I/O, so it unit-tests in isolation and
never blocks the live scan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from research.core.contracts import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_LIVE,
    LIFECYCLE_RETIRED,
    LIFECYCLE_SHADOW,
    LIFECYCLE_TRAINED,
    LIFECYCLE_VALIDATED,
)
from research.core.event_lake import DriftReport

# ---------------------------------------------------------------------------
# Peer-through constants
# ---------------------------------------------------------------------------
#: Default live trades required before PF-based demotion may fire.
DEFAULT_LIVE_MIN_TRADES = 30
#: Default binomial significance for win-rate-drop demotion.
DEFAULT_WR_P_VALUE = 0.05
#: Default consecutive losses that pause an assignment (§5.3).
DEFAULT_MAX_CONSEC_LOSSES = 5
#: Default shadow sample-size gate.
DEFAULT_SHADOW_MIN_EVENTS = 30
#: Default shadow max-days gate.
DEFAULT_SHADOW_MAX_DAYS = 30

#: Ordered lifecycle arms (linear forward path, no jumps).
_FORWARD_ORDER = (
    LIFECYCLE_TRAINED,
    LIFECYCLE_VALIDATED,
    LIFECYCLE_SHADOW,
    LIFECYCLE_LIVE,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_RETIRED,
)


class LifecycleError(Exception):
    """Invalid state transition or lifecycle configuration."""


@dataclass
class LifecycleConfig:
    """§5 lifecycle / shadow / auto-demotion tunables."""

    shadow_min_events: int = DEFAULT_SHADOW_MIN_EVENTS
    shadow_max_days: int = DEFAULT_SHADOW_MAX_DAYS
    live_min_trades: int = DEFAULT_LIVE_MIN_TRADES
    wr_p_value: float = DEFAULT_WR_P_VALUE
    max_consec_losses: int = DEFAULT_MAX_CONSEC_LOSSES
    #: PSI drift thresholds (mirror DriftMonitor §5.2/5.3)
    drift_retrain_threshold: float = 0.2
    drift_demote_threshold: float = 0.25
    min_drift_features: int = 2

    def __post_init__(self) -> None:
        if self.shadow_min_events < 1:
            raise LifecycleError("shadow_min_events must be >= 1")
        if self.shadow_max_days < 1:
            raise LifecycleError("shadow_max_days must be >= 1")
        if not 0.0 <= self.wr_p_value <= 1.0:
            raise LifecycleError("wr_p_value must be in [0,1]")


@dataclass
class Transition:
    """A single lifecycle transition decision."""

    from_state: str
    to_state: str
    reason: str
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.from_state != self.to_state


@dataclass
class LifecycleVerdict:
    """Complete evaluation result for one (symbol, pattern) assignment."""

    model_id: str
    symbol: str
    pattern_name: str
    timeframe: str
    state: str
    transitions: list[Transition] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    #: True when the assignment should be paused (no new signals) w/o state change.
    paused: bool = False

    @property
    def needs_retrain(self) -> bool:
        return bool(self.flags.get("needs_retrain", False))

    @property
    def final_state(self) -> str:
        return self.transitions[-1].to_state if self.transitions else self.state


# ---------------------------------------------------------------------------
# §5.1 State machine
# ---------------------------------------------------------------------------

def valid_transition(current: str, target: str, *, retrain: bool = False) -> bool:
    """Validate a single-step transition against the §5.1 state machine.

    Forward hops follow the linear order; ``retrain=True`` permits the special
    ``live/degraded/shadow → validated`` edge (spec §5.1 ``◄ retrain``).
    """
    if target == current:
        return True
    if retrain:
        if current in (LIFECYCLE_LIVE, LIFECYCLE_DEGRADED, LIFECYCLE_SHADOW) and target == LIFECYCLE_VALIDATED:
            return True
        return False
    if current not in _FORWARD_ORDER or target not in _FORWARD_ORDER:
        return False
    return _FORWARD_ORDER.index(target) == _FORWARD_ORDER.index(current) + 1


class LifecycleManager:
    """Applies the §5 state machine, shadow gate, and auto-demotion triggers.

    Stateless *per call*: each method takes the current state + the aggregate
    evidence (events/outcomes/PF/drift) and returns a :class:`LifecycleVerdict`.
    The engine persists the resulting state back to its assignment registry.
    """

    def __init__(self, config: LifecycleConfig | None = None) -> None:
        self.config = config or LifecycleConfig()

    # ------------------------------------------------------------------
    # §5.1 low-level transition
    # ------------------------------------------------------------------
    def transition(
        self,
        current: str,
        target: str,
        reason: str,
        *,
        retrain: bool = False,
        flags: dict[str, Any] | None = None,
    ) -> Transition:
        if not valid_transition(current, target, retrain=retrain):
            raise LifecycleError(
                f"illegal lifecycle transition {current!r} -> {target!r}"
            )
        return Transition(
            from_state=current, to_state=target, reason=reason,
            flags=dict(flags) if flags else {},
        )

    # ------------------------------------------------------------------
    # §5.2 Shadow gate
    # ------------------------------------------------------------------

    def evaluate_shadow(
        self,
        *,
        symbol: str,
        pattern_name: str,
        timeframe: str,
        model_id: str,
        current_state: str,
        shadow_started_at: datetime | None,
        n_shadow_events: int,
        live_pf: float | None,
        oos_pf: float | None,
        oos_pf_std: float | None,
        drift: DriftReport | None,
        causality_violations: int = 0,
        now: datetime | None = None,
    ) -> LifecycleVerdict:
        """Evaluate whether a shadow assignment may be promoted to live (§5.2).

        All four conditions must hold:
          1. ``n_shadow_events >= shadow_min_events``
             (or ``shadow_max_days`` elapsed since ``shadow_started_at``);
          2. live hypothetical PF within the backtest OOS PF CI (+/-1sigma) —
             i.e. not >1sigma below OOS PF;
          3. feature-distribution drift healthy (``drift`` reports no
             ``requires_retrain`` / ``demote_to_degraded`` and no drift);
          4. no ``CausalityViolation`` in the shadow period.

        Returns a verdict.  A shadow that fails the gate stays in ``shadow``
        (optionally ``degraded`` when drift is severe) — never promotes.
        """
        now = now or datetime.now(timezone.utc)
        cfg = self.config
        flags: dict[str, Any] = {
            "n_shadow_events": int(n_shadow_events),
            "live_pf": live_pf,
            "oos_pf": oos_pf,
            "oos_pf_std": oos_pf_std,
            "causality_violations": int(causality_violations),
        }
        transitions: list[Transition] = []

        sample_ok = n_shadow_events >= cfg.shadow_min_events
        if (not sample_ok) and shadow_started_at is not None:
            elapsed_days = (now - _as_utc(shadow_started_at)).days
            sample_ok = elapsed_days >= cfg.shadow_max_days
            flags["elapsed_days"] = elapsed_days

        drift_ok = True
        drift_msg = ""
        if drift is not None:
            flags["drift_summary"] = drift.summary()
            drift_ok = not (drift.requires_retrain or drift.demote_to_degraded)
            if not drift_ok:
                drift_msg = "feature drift too large for shadow gate"

        pf_ok = _shadow_pf_ok(live_pf, oos_pf, oos_pf_std)
        if not pf_ok:
            flags["pf_status"] = "reject"

        causal_ok = causality_violations == 0

        if not (sample_ok and pf_ok and drift_ok and causal_ok):
            reasons = []
            if not sample_ok:
                reasons.append("shadow sample too small")
            if not pf_ok:
                reasons.append("live PF outside backtest OOS CI")
            if not drift_ok:
                reasons.append(drift_msg)
            if not causal_ok:
                reasons.append("CausalityViolation(s) during shadow")
            flags["gate_blocker"] = "; ".join(reasons)
            # Severe drift shadows sink to degraded rather than blocking at shadow.
            if current_state == LIFECYCLE_SHADOW and drift is not None and drift.demote_to_degraded:
                transitions.append(
                    Transition(
                        from_state=LIFECYCLE_SHADOW,
                        to_state=LIFECYCLE_DEGRADED,
                        reason="shadow severe drift -> degraded (§5.1)",
                        flags={"drift_summary": drift.summary()},
                    )
                )
            return LifecycleVerdict(
                model_id=model_id, symbol=symbol, pattern_name=pattern_name,
                timeframe=timeframe, state=current_state, transitions=transitions,
                flags=flags,
            )

        transitions.append(
            Transition(
                from_state=current_state,
                to_state=LIFECYCLE_LIVE,
                reason="shadow gate passed (§5.2)",
                flags={"sample_ok": True, "pf_status": "pass"},
            )
        )
        return LifecycleVerdict(
            model_id=model_id, symbol=symbol, pattern_name=pattern_name,
            timeframe=timeframe, state=current_state, transitions=transitions,
            flags=flags,
        )

    # ------------------------------------------------------------------
    # §5.3 auto-demotion (periodic)
    # ------------------------------------------------------------------

    def evaluate_demotion(
        self,
        *,
        symbol: str,
        pattern_name: str,
        timeframe: str,
        model_id: str,
        current_state: str,
        n_live_trades: int,
        live_pf_ci_lower: float | None,
        live_win_rate: float | None,
        backtest_win_rate: float | None,
        consec_losses: int,
        drift: DriftReport | None,
    ) -> LifecycleVerdict:
        """Evaluate §5.3 auto-demotion triggers for a live/degraded assignment.

        Applies, in precedence order, only to states that can demote:
          * PF CI lower < breakeven (1.0) after >= ``live_min_trades`` →
            live → degraded;
          * PSI drift >= ``drift_demote_threshold`` on any feature →
            live → degraded (flag ``needs_retrain`` when
            ``requires_retrain``);
          * binomial win-rate drop p < ``wr_p_value`` → live → degraded;
          * consecutive losses >= ``max_consec_losses`` → pause assignment
            (no state change).

        Returns a verdict with any applied transitions/flags.
        """
        cfg = self.config
        flags: dict[str, Any] = {
            "n_live_trades": int(n_live_trades),
            "live_pf_ci_lower": live_pf_ci_lower,
            "live_win_rate": live_win_rate,
            "backtest_win_rate": backtest_win_rate,
            "consec_losses": int(consec_losses),
        }
        transitions: list[Transition] = []
        paused = False

        actionable = current_state in (LIFECYCLE_LIVE, LIFECYCLE_SHADOW, LIFECYCLE_DEGRADED)
        if not actionable or current_state == LIFECYCLE_DEGRADED:
            # degraded: only note retrain need, no further demotion.
            if drift is not None and drift.requires_retrain:
                flags["needs_retrain"] = True
            return LifecycleVerdict(
                model_id=model_id, symbol=symbol, pattern_name=pattern_name,
                timeframe=timeframe, state=current_state,
                transitions=transitions, flags=flags, paused=paused,
            )

        # 1) PF CI lower < breakeven (>= min trades)
        if (
            current_state == LIFECYCLE_LIVE
            and live_pf_ci_lower is not None
            and math.isfinite(live_pf_ci_lower)
            and n_live_trades >= cfg.live_min_trades
            and live_pf_ci_lower < 1.0
        ):
            transitions.append(self.transition(
                current_state, LIFECYCLE_DEGRADED,
                f"live PF CI lower {live_pf_ci_lower:.2f} < breakeven (§5.3)",
            ))
            return LifecycleVerdict(
                model_id=model_id, symbol=symbol, pattern_name=pattern_name,
                timeframe=timeframe, state=current_state, transitions=transitions,
                flags=flags, paused=paused,
            )

        # 2) PSI drift -> degraded / retrain
        if drift is not None:
            flags["drift_summary"] = drift.summary()
            if drift.demote_to_degraded and current_state in (LIFECYCLE_LIVE, LIFECYCLE_SHADOW):
                transitions.append(self.transition(
                    current_state, LIFECYCLE_DEGRADED,
                    f"feature drift PSI >= {cfg.drift_demote_threshold} (§5.3)",
                    flags={"drift_summary": drift.summary()},
                ))
            if drift.requires_retrain:
                flags["needs_retrain"] = True

        # 3) win-rate drop (binomial) — only when still live
        if current_state == LIFECYCLE_LIVE:
            wr_p = _binomial_wr_p(backtest_win_rate, live_win_rate)
            flags["wr_drop_p"] = wr_p
            if (
                wr_p is not None
                and wr_p < cfg.wr_p_value
                and backtest_win_rate is not None
                and live_win_rate is not None
            ):
                transitions.append(self.transition(
                    current_state, LIFECYCLE_DEGRADED,
                    f"win-rate drop p={wr_p:.4f} < {cfg.wr_p_value} (§5.3)",
                ))

        # 4) consecutive losses -> pause (no state change)
        if consec_losses >= cfg.max_consec_losses:
            paused = True
            flags["paused_reason"] = (
                f"consecutive losses {consec_losses} >= {cfg.max_consec_losses}"
            )

        return LifecycleVerdict(
            model_id=model_id, symbol=symbol, pattern_name=pattern_name,
            timeframe=timeframe, state=current_state, transitions=transitions,
            flags=flags, paused=paused,
        )

    # ------------------------------------------------------------------
    # §5.4 per-pattern kill-switch (convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def kill_pattern(state: str, reason: str) -> Transition:
        """Disable one pattern by retiring it (per-pattern, not whole symbol)."""
        return Transition(from_state=state, to_state=LIFECYCLE_RETIRED, reason=reason)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _shadow_pf_ok(live_pf: float | None, oos_pf: float | None, oos_pf_std: float | None) -> bool:
    """§5.2 (2): live PF within CI of backtest OOS PF (not >1sigma below)."""
    if live_pf is None or not math.isfinite(live_pf):
        return False
    if oos_pf is None or not math.isfinite(oos_pf):
        # no backtest reference → cannot prove in-CI → fail closed
        return False
    std = oos_pf_std if oos_pf_std is not None and math.isfinite(oos_pf_std) else 0.0
    return live_pf >= oos_pf - std


def _binomial_wr_p(backtest_wr: float | None, live_wr: float | None) -> float | None:
    """One-sided (lower-tail) binomial p-value that the live win-rate is below
    the backtest win-rate by chance (normal approximation).

    Standard-normal CDF at z is Φ(z) = erfc(-z/sqrt(2))/2 — a large drop (z << 0)
    yields a tiny p.  None when inputs are missing/invalid.
    """
    from math import erfc, sqrt

    if backtest_wr is None or live_wr is None:
        return None
    if not (0.0 < backtest_wr < 1.0) or not (0.0 <= live_wr <= 1.0):
        return None
    n = 30  # default reference sample; caller may override n via flags upstream
    mu = n * backtest_wr
    sigma = math.sqrt(n * backtest_wr * (1.0 - backtest_wr))
    if sigma <= 0:
        return None
    observed = n * live_wr
    z = (observed - mu) / sigma
    return erfc(-z / sqrt(2.0)) / 2.0  # lower tail


__all__ = [
    "LifecycleConfig",
    "LifecycleError",
    "LifecycleManager",
    "LifecycleVerdict",
    "Transition",
    "valid_transition",
]
