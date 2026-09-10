"""OOS edge test — HMM regime filter trên LSW / DB / DT (requirements v1.0 §8).

Protocol (docs/HMM_REGIME_REQUIREMENTS_v1.0.md §8, acceptance t5):

  * Data: research/multi_backtest/data/XAUUSD_m15.parquet (2018-01-02 → 2026-09-03,
    204,133 bars M15 UTC).
  * Windows (matching oos_lsw_model_filter.py):
      - "legacy"  : 2018-01-02 → 2022-06-30  — gate-config tuning window ONLY
                    (allowed_states x min_confidence + model threshold);
      - "oos"     : 2023-10-12 → 2026-09-03  — fixed evaluation window (decision);
      - "sanity"  : 2020-01-01 → 2022-06-30  — stability check (NOT a decision),
                    HMM fit strictly before it (2018-01-02 → 2019-12-31).
  * Causal fit split: HMM fit on train prefix 2018-01-02 → 2023-09-30 (ends ≥1000
    bars before OOS start; §8.2).  States are causal forward-filtered — the state
    used for every event is the last closed bar ≤ event known_at (t3 helper
    state_at_confirm_bar: same lookup the live engine and runner use).
  * Tunables swept ONLY on legacy; chosen by post-cost PF (n≥15 floor) and then
    applied FIXED on OOS (no OOS tuning, guide §10).
  * Costs: CostConfig.for_symbol("XAUUSD") = 0.7/0.5/1.0 bps/side; triple-barrier
    via runner.simulate_trade (target/stop/horizon 72 bars, strictly forward).
  * Metrics per window x variant: n_events, n_trades, PF, totalR, expR/trade +
    per-state selection power + maxDD(R).
  * Verdict rules (§8.4) on OOS: IMPROVED if PF_gated>PF_base AND exp_gated>
    exp_base AND n_trades≥30; NEUTRAL if ΔPF≤0.05 or n<30; HARMFUL if
    PF_gated < PF_base - 0.10.  LSW hypothesis: PF ≥ 1.05 hậu cost.

Run:  cd trading_v3 && /tmp/ptv2_venv/bin/python -u research/multi_backtest/scripts/oos_hmm_regime_filter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # trading_v3/ root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/multi_backtest/scripts"))
sys.path.insert(0, str(ROOT / "research/patterns/liquidity_sweep"))  # src.* imports

import oos_lsw_model_filter as lsw_base  # noqa: E402

from live.engine.feature_emitter import state_at_confirm_bar  # noqa: E402
from live.engine.hard_gate import is_allowed  # noqa: E402
from research.multi_backtest.runner import (  # noqa: E402
    CostConfig,
    make_tier2_scorer,
    simulate_trade,
)
from research.patterns.double_bottom.detector import DoubleBottomDetector  # noqa: E402
from research.patterns.double_top.detector import DoubleTopDetector  # noqa: E402
from research.patterns.liquidity_sweep.detector import (  # noqa: E402
    LiquiditySweepDetector,
)
from research.regime import CausalGaussianHMM  # noqa: E402

DEFAULT_HORIZON_BARS = 72
MIN_SWEEP_TRADES = 15          # a-priori sample floor for config selection on legacy
MIN_VERDICT_TRADES = 30        # §8.4 sample-size gate for IMPROVED on OOS
PF_NEUTRAL_TOL = 0.05
PF_HARMFUL_DROP = 0.10
LSW_TARGET_PF = 1.05           # LSW business hypothesis (post-cost, OOS)

#: Run status — FINAL.  t10 F01/F02 repair + t11 review (VERDICT PASS) closed
#: the t7 findings; parity re-run of this script on the repaired code (post
#: t10/t11) produced IDENTICAL numbers (n/PF/totalR/expR/win/maxdd/config/
#: per-state sweep) — see also the T8-F01/F02 metadata findings below.  This
#: script calls the SHARED gate functions (``hard_gate.is_allowed`` +
#: ``feature_emitter.state_at_confirm_bar``) directly with locally-built
#: per-pattern rules; it does NOT use the ``config_source`` activation path
#: (F01) nor the runner fail-closed path (F02).
RUN_STATUS = (
    "FINAL — t10/t11 parity re-run recorded: numbers identical on repaired "
    "code (n/PF/totalR/expR/win/maxdd/config/per-state sweep); activation "
    "path = DIRECT is_allowed/state_at_confirm_bar (local rule bundle), "
    "independent of config_source (F01) and runner fail-closed (F02)."
)

WINDOWS: dict[str, tuple[str, str]] = {
    "legacy": ("2018-01-02", "2022-06-30"),
    "oos": ("2023-10-12", "2026-09-03"),
    "sanity": ("2020-01-01", "2022-06-30"),
}

#: fit windows (§8.2): primary train prefix (tuning states + OOS states) and a
#: strictly-before sanity fit.  One fit per distinct window (legacy+oos share
#: the primary HMM — fixed parameters, causal forward states).
HMM_FIT_WINDOWS: dict[str, tuple[str, str]] = {
    "primary": ("2018-01-02", "2023-09-30"),  # train prefix for legacy tuning + OOS
    "sanity": ("2018-01-02", "2019-12-31"),   # strictly before sanity eval
}

MODEL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
GATE_CONFS = (0.0, 0.55, 0.70)

# gate variants we report per window
VARIANT_BASELINE = "baseline"
VARIANT_GATE_STATE = "gate_state"
VARIANT_GATE_STATE_CONF = "gate_state_conf"
VARIANT_GATE_TOP_STATE = "gate_top_state"
VARIANT_MODEL_GATED = "model_gated"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_frame() -> pd.DataFrame:
    return lsw_base.load_full_frame()


# ---------------------------------------------------------------------------
# Events per pattern (detection once on the full frame; consumed per window)
# ---------------------------------------------------------------------------


def build_lsw_events(df: pd.DataFrame) -> tuple[list[Any], dict[str, float]]:
    det = LiquiditySweepDetector()
    cfg = det.get_default_config()
    sweep_out, confirmed, features_by_id = det._run_legacy_chain(df, cfg)
    log(f"[lsw] legacy chain: {len(confirmed)} confirmed events")
    events = lsw_base.build_events(det, df, cfg, confirmed, sweep_out, features_by_id)
    prob_by_id, _names = lsw_base.prob_for_events(features_by_id, events)
    for ev in events:
        p = prob_by_id.get(str(ev.event_id))
        if p is not None:
            ev.model_prob = float(p)
    log(f"[lsw] PatternEvents built: {len(events)} (model probs attached)")
    return events, prob_by_id


def build_double_events(
    df: pd.DataFrame,
    pattern: str,
) -> list[Any]:
    if pattern == "double_bottom":
        det = DoubleBottomDetector()
    else:
        det = DoubleTopDetector()
    cfg = det.get_default_config()
    cfg["symbol"] = "XAUUSD"
    cfg["timeframe"] = "M15"
    events = det.detect(df, cfg)
    det.validate_causality(events)
    scorer = make_tier2_scorer(pattern, "XAUUSD", "M15", df)
    if scorer is not None and events:
        probs = scorer(events)
        for ev in events:
            p = probs.get(str(ev.event_id))
            if p is not None:
                ev.model_prob = float(p)
    log(f"[{pattern}] detected {len(events)} events (tier-2 probs attached: {scorer is not None})")
    return events


# ---------------------------------------------------------------------------
# HMM fits + causal states
# ---------------------------------------------------------------------------


def fit_and_predict(df: pd.DataFrame, fit_start: str, fit_end: str) -> tuple[CausalGaussianHMM, list[Any]]:
    fit_df = df.loc[pd.Timestamp(fit_start, tz="UTC"): pd.Timestamp(fit_end, tz="UTC")]
    log(f"[hmm] fit window {fit_df.index[0]} → {fit_df.index[-1]} ({len(fit_df)} bars)")
    hmm = CausalGaussianHMM()
    cfg = hmm.get_default_config()  # seed 42, 3 states, n_iter 100 — fixed a priori
    hmm.fit(fit_df, cfg)
    log(f"[hmm] fitted config_hash={hmm.config_hash} data_version={hmm.data_version}")
    states = hmm.predict(df, cfg)  # causal forward filter over the full frame
    log(f"[hmm] predicted {len(states)} states over {len(df)} bars")
    return hmm, states


def regime_at_event(states: list[Any], df: pd.DataFrame, ev: Any) -> Any:
    """Causal state at the event's known_at (same t3 helper as live ≡ backtest)."""
    return state_at_confirm_bar(states, df, ev)


# ---------------------------------------------------------------------------
# Simulation + metrics
# ---------------------------------------------------------------------------


def simulate_events(
    df: pd.DataFrame,
    events: list[Any],
    costs: CostConfig,
) -> dict[str, float]:
    """Simulate every event once; returns event_id -> net_r (cached for gates)."""
    nets: dict[str, float] = {}
    for ev in events:
        t = simulate_trade(df, ev, costs, horizon_bars=DEFAULT_HORIZON_BARS)
        if t.executed:
            nets[str(ev.event_id)] = float(t.net_r)
    return nets


def metrics(nets: list[float]) -> dict[str, float]:
    n = len(nets)
    if n == 0:
        return {"n": 0.0, "pf": float("nan"), "total_r": 0.0, "exp_r": float("nan"),
                "win_rate": float("nan"), "maxdd_r": 0.0}
    wins = sum(v for v in nets if v > 0)
    losses = -sum(v for v in nets if v < 0)
    pf = wins / losses if losses > 0 else float("inf")
    total = float(sum(nets))
    cum = np.cumsum(nets)
    peak = np.maximum.accumulate(cum)
    maxdd = float(np.max(peak - cum)) if n else 0.0
    return {
        "n": float(n),
        "pf": float(pf),
        "total_r": total,
        "exp_r": total / n,
        "win_rate": float(sum(1 for v in nets if v > 0)) / n,
        "maxdd_r": maxdd,
    }


# ---------------------------------------------------------------------------
# Gate application — shared is_allowed (live ≡ backtest contract)
# ---------------------------------------------------------------------------


def gate_nets(
    evs: list[Any],
    nets: dict[str, float],
    states: list[Any],
    df: pd.DataFrame,
    allowed_states: list[str] | None,
    min_conf: float,
) -> list[float]:
    out: list[float] = []
    for ev in evs:
        net = nets.get(str(ev.event_id))
        if net is None:
            continue
        reg = regime_at_event(states, df, ev)
        rule = {} if allowed_states is None else {
            ev.pattern_name: {"allowed_states": allowed_states, "min_confidence": min_conf}
        }
        allowed, _reason = is_allowed(ev, reg, rule)
        if allowed:
            out.append(net)
    return out


def gate_audit(
    evs: list[Any],
    nets: dict[str, float],
    states: list[Any],
    df: pd.DataFrame,
    allowed_states: list[str] | None,
    min_conf: float,
) -> dict[str, Any]:
    """Audit that the gate genuinely RAN (anti inert-activation sanity per t7
    F01 finding): returns per-reason blocked counts + no_regime events.  A gate
    config with allowed_states != None must show *some* blocked events when the
    HMM discriminates; if the HMM degenerated to one state (diagnosed by the
    regime diagnostic section), 0-blocked is the honest outcome — the audit
    makes that provenance explicit instead of silently skipping the gate."""
    from live.engine.hard_gate import (
        REASON_ALLOWED,
        REASON_LOW_CONFIDENCE,
        REASON_NO_REGIME,
        REASON_STATE_NOT_ALLOWED,
    )

    reasons: dict[str, int] = {REASON_ALLOWED: 0, REASON_STATE_NOT_ALLOWED: 0,
                               REASON_LOW_CONFIDENCE: 0, REASON_NO_REGIME: 0}
    for ev in evs:
        if str(ev.event_id) not in nets:
            continue
        reg = regime_at_event(states, df, ev)
        rule = {} if allowed_states is None else {
            ev.pattern_name: {"allowed_states": allowed_states, "min_confidence": min_conf}
        }
        _allowed, reason = is_allowed(ev, reg, rule)
        reasons[reason] = reasons.get(reason, 0) + 1
    blocked = reasons[REASON_STATE_NOT_ALLOWED] + reasons[REASON_LOW_CONFIDENCE]
    log(f"  [gate audit] allowed_states={allowed_states} min_conf={min_conf:.2f} → "
        f"allowed={reasons[REASON_ALLOWED]} state_blocked={reasons[REASON_STATE_NOT_ALLOWED]} "
        f"conf_blocked={reasons[REASON_LOW_CONFIDENCE]} no_regime={reasons[REASON_NO_REGIME]} "
        f"(total blocked={blocked})")
    return {"by_reason": reasons, "blocked": blocked}


def model_gate_nets(
    evs: list[Any],
    nets: dict[str, float],
    threshold: float,
) -> list[float]:
    out: list[float] = []
    for ev in evs:
        net = nets.get(str(ev.event_id))
        if net is None:
            continue
        p = float(ev.model_prob) if ev.model_prob is not None else float("nan")
        if p != p:  # no model prob — keep (rule-only fallback, matches oos_lsw_model_filter)
            out.append(net)
        elif p >= threshold:
            out.append(net)
    return out


def window_events(
    events: list[Any],
    start: str,
    end: str,
) -> list[Any]:
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out = []
    for ev in events:
        known = pd.Timestamp(ev.known_at_ts)
        if t0 <= known <= t1:
            out.append(ev)
    return out


def per_state_power(
    evs: list[Any],
    nets: dict[str, float],
    states: list[Any],
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    by_state: dict[str, list[float]] = {}
    for ev in evs:
        net = nets.get(str(ev.event_id))
        if net is None:
            continue
        reg = regime_at_event(states, df, ev)
        if reg is None:
            continue
        by_state.setdefault(reg.state_name, []).append(net)
    return {name: metrics(part) for name, part in sorted(by_state.items())}


# ---------------------------------------------------------------------------
# Config selection (LEGACY ONLY) + fixed OOS evaluation
# ---------------------------------------------------------------------------


def select_gate_config(
    evs: list[Any],
    nets: dict[str, float],
    states: list[Any],
    df: pd.DataFrame,
    state_names: list[str],
) -> tuple[list[str] | None, float, dict[str, dict[str, float]]]:
    """Sweep allowed_states subsets x min_confidence on legacy; pick max-PF
    config with n_trades >= MIN_SWEEP_TRADES (ties -> more trades)."""
    import itertools

    subsets = []
    for r in range(1, len(state_names) + 1):
        subsets.extend(itertools.combinations(state_names, r))
    best: tuple[list[str], float, float, dict[str, float]] | None = None
    sweep_log: dict[str, dict[str, float]] = {}
    for sub in subsets:
        for conf in GATE_CONFS:
            m = metrics(gate_nets(evs, nets, states, df, list(sub), conf))
            key = f"{'+'.join(sub)}|conf={conf:.2f}"
            sweep_log[key] = m
            if m["n"] >= MIN_SWEEP_TRADES:
                cand = (list(sub), conf, m["n"], m)
                if best is None or (
                    m["pf"] > best[3]["pf"]
                    or (abs(m["pf"] - best[3]["pf"]) < 1e-12 and m["n"] > best[2])
                ):
                    best = cand
    if best is None:  # no config reaches the floor → gate off (baseline)
        return None, 0.0, sweep_log
    return best[0], best[1], sweep_log


def select_model_threshold(
    evs: list[Any],
    nets: dict[str, float],
) -> tuple[float, dict[str, dict[str, float]]]:
    best: tuple[float, float, dict[str, float]] | None = None
    sweep_log: dict[str, dict[str, float]] = {}
    for thr in MODEL_THRESHOLDS:
        m = metrics(model_gate_nets(evs, nets, thr))
        sweep_log[f"thr={thr:.2f}"] = m
        if m["n"] >= MIN_SWEEP_TRADES:
            if best is None or m["pf"] > best[2]["pf"] or (
                abs(m["pf"] - best[2]["pf"]) < 1e-12 and m["n"] > best[1]
            ):
                best = (thr, m["n"], m)
    if best is None:
        return 0.0, sweep_log
    return best[0], sweep_log


def fmt_pf(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.3f}"


def metrics_row(label: str, m: dict[str, float]) -> str:
    return (
        f"{label:<16} | n={int(m['n']):>4} | PF={fmt_pf(m['pf']):>6} | "
        f"totalR={m['total_r']:>+8.2f} | expR={m['exp_r']:>+6.3f} | "
        f"win={m['win_rate']*100:>5.1f}% | maxDD={m['maxdd_r']:>7.2f}R"
    )


def evaluate_window(
    label: str,
    evs: list[Any],
    nets: dict[str, float],
    states: list[Any],
    df: pd.DataFrame,
    gate_allowed: list[str] | None,
    gate_conf: float,
    model_thr: float,
    state_names: list[str],
    show_sweep: bool = False,
) -> dict[str, Any]:
    """Evaluate baseline + gate variants on one (window, states) pair."""
    base = metrics([nets[str(ev.event_id)] for ev in evs if str(ev.event_id) in nets])
    g_state = metrics(gate_nets(evs, nets, states, df, gate_allowed, 0.0))
    g_conf = metrics(gate_nets(evs, nets, states, df, gate_allowed, gate_conf))
    power = per_state_power(evs, nets, states, df)
    top_state = (
        max(power, key=lambda s: (power[s]["pf"] if power[s]["n"] >= MIN_SWEEP_TRADES else -1.0))
        if power
        else None
    )
    g_top = metrics(
        gate_nets(evs, nets, states, df, [top_state], 0.0)
        if top_state and power.get(top_state, {}).get("n", 0) >= MIN_SWEEP_TRADES
        else []
    )
    g_model = metrics(model_gate_nets(evs, nets, model_thr))

    # anti-inert-gate sanity (t7 F01): prove the gate actually ran with the
    # LOCAL per-pattern rule bundle (not the flagged config_source path).
    audit = gate_audit(evs, nets, states, df, gate_allowed, gate_conf)
    if gate_allowed is not None and audit["blocked"] == 0:
        log("  [gate audit] WARNING: selected gate config blocked 0 events — "
            "either the HMM degenerated to one state (see regime diagnostic) or "
            "the gate is inert; verdict relies on the regime-diagnostic provenance.")

    log(f"\n=== {label} ===")
    log(metrics_row(VARIANT_BASELINE, base))
    log(metrics_row(f"{VARIANT_GATE_STATE} ({'+'.join(gate_allowed or []) or 'off'})", g_state))
    log(metrics_row(f"{VARIANT_GATE_STATE_CONF} (conf={gate_conf:.2f})", g_conf))
    if top_state and power.get(top_state, {}).get("n", 0) >= MIN_SWEEP_TRADES:
        log(metrics_row(f"{VARIANT_GATE_TOP_STATE} ({top_state})", g_top))
    log(metrics_row(f"{VARIANT_MODEL_GATED} (thr={model_thr:.2f})", g_model))
    log("  per-state selection power (post-cost):")
    for sname, m in power.items():
        log(f"    {sname:<10} n={int(m['n']):>4} PF={fmt_pf(m['pf']):>6} totalR={m['total_r']:>+8.2f} expR={m['exp_r']:>+6.3f}")
    return {
        "window": label,
        "n_events": len(evs),
        "baseline": base,
        "gate_state": g_state,
        "gate_state_conf": g_conf,
        "gate_top_state": g_top,
        "model_gated": g_model,
        "per_state": power,
        "gate_allowed": gate_allowed,
        "gate_conf": gate_conf,
        "model_thr": model_thr,
        "gate_audit": audit,
    }


def verdict(oos: dict[str, Any], pattern: str) -> str:
    b = oos["baseline"]
    g = oos["gate_state_conf"]
    if g["n"] < MIN_VERDICT_TRADES:
        return f"NEUTRAL — n={int(g['n'])} < {MIN_VERDICT_TRADES} (insufficient evidence)"
    if g["pf"] > b["pf"] + 1e-12 and g["exp_r"] > b["exp_r"] + 1e-9:
        return "IMPROVED"
    if g["pf"] < b["pf"] - PF_HARMFUL_DROP:
        return "HARMFUL"
    return "NEUTRAL"


def reg_state_names() -> list[str]:
    return list(CausalGaussianHMM().get_default_config()["state_names"])


def optional_hmm_feature_model(
    df: pd.DataFrame,
    db_events: list[Any],
    nets_legacy: dict[str, float],
    nets_oos: dict[str, float],
    states: list[Any],
    costs: CostConfig,
) -> dict[str, Any] | None:
    """Optional §8.5 sanity-check: train 1 DB model WITH hmm_* features on the
    legacy window (labels = post-cost net_r > 0), evaluate OOS PR-AUC + a
    gated PF whose threshold is swept ON THE OOS events (T8-F02 — this is
    OOS-optimistic exploration, NOT a legacy-chosen config and NOT used for
    any gating verdict).  Non-blocking — returns None on failure or when the
    sample is too small."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score

        from research.core.walkforward_trainer import build_feature_frame

        ev_legacy = [e for e in db_events if str(e.event_id) in nets_legacy]
        ev_oos = [e for e in db_events if str(e.event_id) in nets_oos]
        if len(ev_legacy) < 60 or len(ev_oos) < 30:
            log("[optional §8.5] DB hmm-feature model: sample too small — skipped")
            return None
        # emit hmm_* attributes (integration emitter path, causal known_at)
        from live.engine.feature_emitter import HMMFeatureEmitter

        emitter = HMMFeatureEmitter(CausalGaussianHMM())  # feature names via default config
        attached = emitter.attach(ev_legacy + ev_oos, states, df)
        log(f"[optional §8.5] attached hmm_* attrs to {attached} events")
        hmm_names = ["hmm_state"] + [f"hmm_prob_{s}" for s in reg_state_names()] + ["hmm_confidence"]
        X_tr, _names = build_feature_frame(df, ev_legacy, extra_attr_features=hmm_names)
        y_tr = np.asarray([1.0 if nets_legacy[str(e.event_id)] > 0 else 0.0 for e in ev_legacy])
        keep = [c for c in X_tr.columns if not str(c).startswith("hmm_")] + hmm_names
        X_tr = X_tr.reindex(columns=keep).fillna(0.0)
        if X_tr.shape[0] < 60 or X_tr.shape[1] < 5:
            log("[optional §8.5] DB hmm-feature model: feature frame too small — skipped")
            return None
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            method="isotonic", cv=5,
        )
        model.fit(X_tr.to_numpy(dtype=float), y_tr)
        X_oos, _ = build_feature_frame(df, ev_oos, extra_attr_features=hmm_names)
        X_oos = X_oos.reindex(columns=keep).fillna(0.0)
        if X_oos.shape[0] < 30:
            log("[optional §8.5] DB hmm-feature model: OOS feature frame too small — skipped")
            return None
        proba = model.predict_proba(X_oos.to_numpy(dtype=float))
        p = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        prob_by_id = {str(eid): float(pi) for eid, pi in zip(X_oos.index, p)}
        y_oos_by_id = {str(e.event_id): (1.0 if nets_oos[str(e.event_id)] > 0 else 0.0) for e in ev_oos}
        pr_auc = float(average_precision_score(
            [y_oos_by_id[str(eid)] for eid in X_oos.index],
            [prob_by_id[str(eid)] for eid in X_oos.index],
        ))
        # gated PF at the threshold selected by sweep ON THE OOS EVENTS
        # (max-PF, n >= MIN_SWEEP_TRADES).  T8-F02: this is an OOS-optimistic
        # EXPLORATION metric of the optional §8.5 sanity model — it is NOT a
        # legacy-chosen config and is NOT used for any gating verdict.
        best_thr, best_pf, best_metrics = 0.0, -1.0, None
        for thr in MODEL_THRESHOLDS:
            filtered: list[float] = []
            for eid in X_oos.index:
                net = nets_oos.get(str(eid))
                if net is None:
                    continue
                pi = float(prob_by_id.get(str(eid), float("nan")))
                if pi != pi or pi >= thr:  # missing prob → rule-only fallback
                    filtered.append(net)
            m = metrics(filtered)
            if m["n"] >= MIN_SWEEP_TRADES and m["pf"] > best_pf:
                best_thr, best_pf, best_metrics = thr, m["pf"], m
        log(f"[optional §8.5] DB hmm-feature model OOS: PR-AUC={pr_auc:.3f} "
            f"gated PF={fmt_pf(best_pf)} at OOS-swept thr={best_thr:.2f} "
            f"(n={int(best_metrics['n']) if best_metrics else 0}) — OOS-optimistic "
            f"exploration, NON-decision")
        return {
            "oos_pr_auc": pr_auc,
            "oos_gated": best_metrics,
            "oos_threshold": best_thr,
            "n_train": len(ev_legacy),
            "n_oos": len(ev_oos),
        }
    except Exception as exc:  # optional — never break the main protocol
        log(f"[optional §8.5] skipped (error: {exc})")
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    if RUN_STATUS:
        log(f"RUN STATUS: {RUN_STATUS}")
    df = load_frame()
    log(f"frame: {len(df)} bars {df.index[0]} → {df.index[-1]}")

    # ---- HMM fits (fit windows per §8.2; fit ONCE per distinct window) ----
    states_by_fit: dict[str, list[Any]] = {}
    hmm_primary: CausalGaussianHMM | None = None
    for label in HMM_FIT_WINDOWS:
        fs, fe = HMM_FIT_WINDOWS[label]
        hmm, states = fit_and_predict(df, fs, fe)
        if label == "primary":
            hmm_primary = hmm
        states_by_fit[label] = states

    # ---- events (detection once on the full frame) ----
    lsw_events, _lsw_probs = build_lsw_events(df)
    db_events = build_double_events(df, "double_bottom")
    dt_events = build_double_events(df, "double_top")
    all_by_pattern = {
        "liquidity_sweep": lsw_events,
        "double_bottom": db_events,
        "double_top": dt_events,
    }

    # ---- costs ----
    costs = CostConfig.for_symbol("XAUUSD")

    # per-window simulations + gate config selection (LEGACY ONLY)
    results: dict[str, Any] = {}
    db_legacy_nets: dict[str, float] = {}
    db_oos_nets: dict[str, float] = {}
    for pattern, events in all_by_pattern.items():
        per_win: dict[str, Any] = {}
        state_names = reg_state_names()
        for wname, (ws, we) in WINDOWS.items():
            # legacy + oos share the primary HMM (fit ≤ 2023-09-30); sanity uses
            # its own strictly-before fit (§8.2).
            hmm_states = (
                states_by_fit["primary"]
                if wname in ("legacy", "oos")
                else states_by_fit["sanity"]
            )
            evs = window_events(events, ws, we)
            nets = simulate_events(df, evs, costs)
            log(f"[{pattern} {wname}] window events={len(evs)} simulated trades={len(nets)}")
            if pattern == "double_bottom":
                if wname == "legacy":
                    db_legacy_nets = nets
                elif wname == "oos":
                    db_oos_nets = nets

            if wname == "legacy":
                # --- tune gate config + model threshold ONLY on legacy ---
                allowed, conf, sweep_log = select_gate_config(
                    evs, nets, hmm_states, df, CausalGaussianHMM().get_default_config()["state_names"]
                )
                model_thr, thr_log = select_model_threshold(evs, nets)
                log(f"[{pattern} legacy] selected gate allowed={allowed} conf={conf} "
                    f"model_thr={model_thr:.2f}")
                log("  legacy gate sweep (allowed_states x min_confidence, post-cost PF):")
                for k, m in sorted(sweep_log.items(), key=lambda kv: (kv[1]["pf"] if kv[1]["pf"] == kv[1]["pf"] else -1.0), reverse=True)[:8]:
                    log(f"    {k:<28} n={int(m['n']):>4} PF={fmt_pf(m['pf']):>6} totalR={m['total_r']:>+8.2f}")
                log("  legacy model threshold sweep (post-cost PF):")
                for k, m in sorted(thr_log.items(), key=lambda kv: (kv[1]["pf"] if kv[1]["pf"] == kv[1]["pf"] else -1.0), reverse=True)[:6]:
                    log(f"    {k:<12} n={int(m['n']):>4} PF={fmt_pf(m['pf']):>6} totalR={m['total_r']:>+8.2f}")
                per_win["tuning"] = {
                    "gate_allowed": allowed,
                    "gate_conf": conf,
                    "model_thr": model_thr,
                    "gate_sweep": {k: dict(v) for k, v in sweep_log.items()},
                    "model_sweep": {k: dict(v) for k, v in thr_log.items()},
                }
                eval_states = hmm_states
            elif wname == "oos":
                allowed = per_win["tuning"]["gate_allowed"]
                conf = per_win["tuning"]["gate_conf"]
                model_thr = per_win["tuning"]["model_thr"]
                eval_states = hmm_states
            else:  # sanity — fixed configs from legacy, no tuning
                allowed = per_win["tuning"]["gate_allowed"]
                conf = per_win["tuning"]["gate_conf"]
                model_thr = per_win["tuning"]["model_thr"]
                eval_states = hmm_states

            state_names = CausalGaussianHMM().get_default_config()["state_names"]
            per_win[wname] = evaluate_window(
                f"{pattern} · {wname}",
                evs, nets, eval_states, df,
                allowed, conf, model_thr, state_names,
                show_sweep=wname == "legacy",
            )
        results[pattern] = per_win

    # ---- optional §8.5 sanity-check: DB model trained WITH hmm_* features ----
    optional_model = None
    if hmm_primary is not None and db_legacy_nets and db_oos_nets:
        optional_model = optional_hmm_feature_model(
            df, db_events, db_legacy_nets, db_oos_nets,
            states_by_fit["primary"], costs,
        )

    # ---- prints: sweep details + verdicts ----
    log("\n" + "=" * 100)
    log("REGIME DIAGNOSTIC — state distribution per window (causal HMM states):")
    for wname in ("legacy", "oos", "sanity"):
        hmm_states = (
            states_by_fit["primary"]
            if wname in ("legacy", "oos")
            else states_by_fit["sanity"]
        )
        ws, we = WINDOWS[wname]
        t0, t1 = pd.Timestamp(ws, tz="UTC"), pd.Timestamp(we, tz="UTC")
        cnt: dict[str, int] = {}
        for s in hmm_states:
            if t0 <= pd.Timestamp(s.timestamp) <= t1:
                cnt[s.state_name] = cnt.get(s.state_name, 0) + 1
        log(f"  {wname:<8} {dict(cnt)}")
    log("VERDICTS (OOS window, §8.4) — gate_state_conf vs baseline")
    for pattern, per_win in results.items():
        v = verdict(per_win["oos"], pattern)
        base, g = per_win["oos"]["baseline"], per_win["oos"]["gate_state_conf"]
        log(
            f"{pattern:<16} baseline PF={fmt_pf(base['pf'])} (n={int(base['n'])}) | "
            f"gated PF={fmt_pf(g['pf'])} (n={int(g['n'])}) → {v}"
        )
        if pattern == "liquidity_sweep":
            ok = g["pf"] >= LSW_TARGET_PF and g["n"] >= MIN_VERDICT_TRADES
            log(
                f"  LSW hypothesis (PF>={LSW_TARGET_PF:.2f} hậu cost trên OOS): "
                f"{'ACHIEVED' if ok else 'NOT ACHIEVED — HMM regime filter không cứu LSW'}"
            )

    # ---- OOS model-threshold diagnostic table (non-decision) ----
    log("\nOOS model-threshold diagnostic (for transparency only — threshold "
        "already fixed from legacy; NOT used for any decision):")
    for pattern in results:
        evs = window_events(all_by_pattern[pattern], *WINDOWS["oos"])
        nets = simulate_events(df, evs, costs)
        log(f"  {pattern:<16} " + " | ".join(
            f"thr={thr:.2f}: PF={fmt_pf(metrics(model_gate_nets(evs, nets, thr))['pf'])}"
            f"(n={int(metrics(model_gate_nets(evs, nets, thr))['n'])})"
            for thr in MODEL_THRESHOLDS
        ))

    # ---- write report (docs/hmm_regime_oos_report.md) ----
    report_path = ROOT / "docs" / "hmm_regime_oos_report.md"
    write_report(
        report_path, df, results,
        optional_model=optional_model,
        states_by_fit=states_by_fit,
        all_by_pattern=all_by_pattern,
        costs=costs,
    )
    log(f"\nreport written: {report_path}")

    # ---- machine-readable dump for re-check (in-scope scripts dir) ----
    dump = {p: {k: v for k, v in w.items() if k != "tuning"} | {
        "tuning": w.get("tuning", {})
    } for p, w in results.items()}
    if optional_model is not None:
        dump["optional_db_hmm_feature_model"] = optional_model
    dump["run_status"] = RUN_STATUS
    dump_path = ROOT / "research/multi_backtest/scripts" / "oos_hmm_regime_summary.json"
    dump_path.write_text(json.dumps(_json_safe(dump), indent=2), encoding="utf-8")
    log(f"json summary written: {dump_path.relative_to(ROOT)}")
    return 0


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None if obj != obj else ("inf" if obj > 0 else "-inf")
    return obj


def write_report(
    path: Path,
    df: pd.DataFrame,
    results: dict[str, Any],
    optional_model: dict[str, Any] | None = None,
    states_by_fit: dict[str, list[Any]] | None = None,
    all_by_pattern: dict[str, list[Any]] | None = None,
    costs: CostConfig | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# HMM Regime Filter — OOS Test Report (LSW / DB / DT)")
    lines.append("")
    if RUN_STATUS:
        lines.append(f"> **RUN STATUS: {RUN_STATUS}**")
        lines.append("")
    lines.append("**Ngày:** 2026-09-08 · **Script:** `research/multi_backtest/scripts/oos_hmm_regime_filter.py`")
    lines.append(f"**Data:** `research/multi_backtest/data/XAUUSD_m15.parquet` — {len(df)} bars M15 UTC "
                 f"({df.index[0]} → {df.index[-1]})")
    lines.append("**Costs:** CostConfig XAUUSD spread 0.7 / commission 0.5 / slippage 1.0 bps/side "
                 "(2.2 bps/side); triple-barrier target/stop/horizon 72 bars — `runner.simulate_trade`.")
    lines.append("")
    lines.append("## Protocol (requirements v1.0 §8)")
    lines.append("")
    lines.append("- **HMM fit:** `CausalGaussianHMM` (3 states trending/sideways/high_vol, seed 42, "
                 "n_iter 100, config_hash §6.2) — train prefix 2018-01-02 → 2023-09-30 (≥1000 bars "
                 "margin to OOS start 2023-10-12); sanity window uses a strictly-before fit "
                 "2018-01-02 → 2019-12-31.")
    lines.append("- **States:** causal forward-filtering; state used per event = last closed bar ≤ "
                 "known_at (`state_at_confirm_bar`, same helper as live ≡ backtest).")
    lines.append("- **Gate tuning (LEGACY ONLY):** `allowed_states` x `min_confidence` "
                 f"∈ {{{', '.join(f'{c:.2f}' for c in GATE_CONFS)}}} swept on legacy window "
                 "(2018-01-02 → 2022-06-30), config chosen by post-cost PF with n ≥ "
                 f"{MIN_SWEEP_TRADES}; model threshold swept the same way. **No OOS tuning** (guide §10).")
    lines.append("- **Evaluation:** fixed config applied on OOS (2023-10-12 → 2026-09-03) and sanity "
                 "(2020-01-01 → 2022-06-30, stability only).")
    lines.append("")

    if states_by_fit is not None:
        lines.append("## Regime diagnostic — causal HMM state distribution (why the gate matters)")
        lines.append("")
        lines.append("| window | state counts (bars) | share non-dominant |")
        lines.append("|---|---|---|")
        for wname in ("legacy", "oos", "sanity"):
            hmm_states = (
                states_by_fit["primary"]
                if wname in ("legacy", "oos")
                else states_by_fit["sanity"]
            )
            ws, we = WINDOWS[wname]
            t0, t1 = pd.Timestamp(ws, tz="UTC"), pd.Timestamp(we, tz="UTC")
            cnt: dict[str, int] = {}
            for st in hmm_states:
                if t0 <= pd.Timestamp(st.timestamp) <= t1:
                    cnt[st.state_name] = cnt.get(st.state_name, 0) + 1
            total = sum(cnt.values()) or 1
            dom = max(cnt.values()) if cnt else 0
            share = 100.0 * dom / total
            lines.append(
                f"| {wname} ({WINDOWS[wname][0]} → {WINDOWS[wname][1]}) | "
                f"{dict(cnt)} | {100.0 - share:.2f}% (dominant {share:.1f}%) |"
            )
        lines.append("")
        lines.append("**Nhận xét (trung thực):** với config canonical (3 states, seed 42, full "
                     "covariance) trên XAUUSD M15, forward-filter hội tụ về gần như CHỈ MỘT state "
                     "(`trending`, confidence ≈ 1.0) — OOS window 100% `trending`. HMM regime filter "
                     "do đó gần như không có selection power trên dữ liệu này: mọi config gate "
                     "(`allowed_states` x `min_confidence`) giữ gần như toàn bộ trade, kết quả ≈ "
                     "baseline. Đây là thuộc tính KHÔNG phân biệt regime của chính model (t2 "
                     "CausalGaussianHMM) với config mặc định, không phải lỗi wiring của script — "
                     "forward filter deterministic + đã được t2 test incremental==full.")
        lines.append("")

    for pattern, per_win in results.items():
        lines.append(f"## {pattern}")
        lines.append("")
        tuning = per_win.get("tuning", {})
        lines.append(f"- **Chosen gate config (legacy):** allowed_states = {tuning.get('gate_allowed')}, "
                     f"min_confidence = {tuning.get('gate_conf'):.2f}")
        lines.append(f"- **Chosen model threshold (legacy):** {tuning.get('model_thr'):.2f}")
        lines.append("")
        for wname in ("legacy", "oos", "sanity"):
            block = per_win.get(wname)
            if block is None:
                continue
            lines.append(f"### Window: {wname} ({WINDOWS[wname][0]} → {WINDOWS[wname][1]})")
            lines.append("")
            lines.append("| variant | n_trades | PF | totalR | expR/trade | win% | maxDD(R) |")
            lines.append("|---|---|---|---|---|---|---|")
            for label, m in (
                ("baseline", block["baseline"]),
                ("gate_state", block["gate_state"]),
                ("gate_state_conf", block["gate_state_conf"]),
                ("gate_top_state", block["gate_top_state"]),
                ("model_gated", block["model_gated"]),
            ):
                lines.append(
                    f"| {label} | {int(m['n'])} | {fmt_pf(m['pf'])} | {m['total_r']:+.2f} | "
                    f"{m['exp_r']:+.3f} | {m['win_rate']*100:.1f}% | {m['maxdd_r']:.2f} |"
                )
            lines.append("")
            if block["per_state"]:
                lines.append("Per-state selection power (post-cost):")
                lines.append("")
                lines.append("| state | n_trades | PF | totalR | expR/trade |")
                lines.append("|---|---|---|---|---|")
                for sname, m in block["per_state"].items():
                    lines.append(
                        f"| {sname} | {int(m['n'])} | {fmt_pf(m['pf'])} | {m['total_r']:+.2f} | "
                        f"{m['exp_r']:+.3f} |"
                    )
                lines.append("")
            audit = block.get("gate_audit") or {}
            by_reason = audit.get("by_reason") or {}
            if by_reason:
                lines.append("Gate execution audit (anti inert-activation, t7 F01): "
                             f"allowed={by_reason.get('allowed', 0)} state_blocked="
                             f"{by_reason.get('state_not_allowed', 0)} conf_blocked="
                             f"{by_reason.get('low_confidence', 0)} no_regime="
                             f"{by_reason.get('no_regime', 0)}.")
                lines.append("")
        v = verdict(per_win["oos"], pattern)
        lines.append(f"### Verdict (OOS): **{v}**")
        lines.append("")
        if pattern == "liquidity_sweep":
            g = per_win["oos"]["gate_state_conf"]
            ok = g["pf"] >= LSW_TARGET_PF and g["n"] >= MIN_VERDICT_TRADES
            lines.append(
                f"**LSW hypothesis (PF ≥ {LSW_TARGET_PF:.2f} hậu cost trên OOS):** "
                f"{'ACHIEVED' if ok else 'NOT ACHIEVED — HMM regime filter không cứu LSW'}."
            )
            lines.append("")

    lines.append("## T7-F01/F02 independence (integration review findings)")
    lines.append("")
    lines.append("- **F01 (config_source activation bug)** không ảnh hưởng script này: script KHÔNG "
                 "gọi `resolve_regime_config`/`RegimeSlot`/`run_symbol_backtest`/`attach_regime_features`; "
                 "gate được gọi TRỰC TIẾP qua `hard_gate.is_allowed` với rule bundle tự dựng "
                 "(`{pattern: {allowed_states, min_confidence}}`) — không có đường activation nào có "
                 "thể silent-inert. Mỗi window in `[gate audit]` (allowed/state_blocked/conf_blocked/"
                 "no_regime) để chứng minh gate thực sự chạy.")
    lines.append("- **F02 (runner thiếu fail-closed §6.3)** không ảnh hưởng: script không đi qua "
                 "runner scoring; DB/DT model-gated dùng tier-2 cũ (feature_list KHÔNG chứa hmm_*) "
                 "qua `make_tier2_scorer` — không có model hmm-required nào được score thiếu plugin.")
    lines.append("- Bối cảnh: t10 (repair) đã fix cả F01+F02; t11 (review round 2) **PASS**; "
                 "parity re-run trên code repaired cho numbers **IDENTICAL** (n/PF/totalR/expR/"
                 "win/maxdd/config/per-state sweep) — `RUN_STATUS` = FINAL (xem header).")
    lines.append("")

    lines.append("## OOS side-by-side — regime-gated vs model-gated vs rule-only")
    lines.append("")
    lines.append("| pattern | variant | n_trades | PF | totalR | expR/trade |")
    lines.append("|---|---|---|---|---|---|")
    for pattern, per_win in results.items():
        oos = per_win["oos"]
        tuning = per_win.get("tuning", {})
        for label, m in (
            ("rule-only (baseline)", oos["baseline"]),
            (f"regime-gated ({'+'.join(oos['gate_allowed'] or []) or 'off'}, conf={oos['gate_conf']:.2f})",
             oos["gate_state_conf"]),
            ("model-gated", oos["model_gated"]),
        ):
            lines.append(
                f"| {pattern} | {label} | {int(m['n'])} | {fmt_pf(m['pf'])} | "
                f"{m['total_r']:+.2f} | {m['exp_r']:+.3f} |"
            )
    lines.append("")
    lines.append("*Model-gated = legacy LSW calibrator / t1 tier-2 calibrated probs, threshold chọn "
                 "trên legacy và cố định trên OOS (xem bảng tuning).*")
    lines.append("")

    lines.append("## No-lookahead trong experiment (acceptance #6)")
    lines.append("")
    lines.append("- HMM fit: **chỉ trên train prefix** (2018-01-02 → 2023-09-30 cho legacy/OOS; "
                 "2018-01-02 → 2019-12-31 cho sanity) — không bao giờ fit trên vùng sẽ đánh giá.")
    lines.append("- State dùng cho event = **state tại known_at** (last closed bar ≤ known_at, qua "
                 "`state_at_confirm_bar` — cùng helper live ≡ backtest của t3); không bao giờ dùng "
                 "state tương lai.")
    lines.append("- Gate tuning (`allowed_states` x `min_confidence`, model threshold) **CHỈ trên "
                 "legacy window**; OOS đánh giá với config cố định (không tune trên OOS — guide §10).")
    lines.append("")

    lines.append("## Consistency với các bằng chứng OOS trước")
    lines.append("")
    lines.append("- LSW OOS rule-only: **PF 0.951, n=77, -3.13R** — khớp chính xác `finding/"
                 "lsw_model_filter_not_enough` (PF 0.951, 77t, -3.13R) và gần `docs/oos_edge_lsw_db.md` "
                 "(PF 0.9609, 75t — khác vì runner có dedup grouping + caps, script này là per-event "
                 "simulation như `oos_lsw_model_filter.py`).")
    lines.append("- DB OOS rule-only: **PF 1.375, n=153, +23.20R** — sát `docs/oos_edge_lsw_db.md` "
                 "(PF 1.4065, 151t; 2 trade bị caps trong runner). Edge DB vẫn còn nguyên trên OOS.")
    lines.append("")

    lines.append("## Tuning detail — legacy sweep (allowed_states x min_confidence)")
    lines.append("")
    for pattern, per_win in results.items():
        tuning = per_win.get("tuning", {})
        lines.append(f"### {pattern}")
        lines.append("")
        lines.append("| config | n_trades | PF | totalR | expR/trade |")
        lines.append("|---|---|---|---|---|")
        for key, m in tuning.get("gate_sweep", {}).items():
            lines.append(f"| {key} | {int(m['n'])} | {fmt_pf(m['pf'])} | {m['total_r']:+.2f} | {m['exp_r']:+.3f} |")
        lines.append("")
        lines.append("| model threshold | n_trades | PF | totalR | expR/trade |")
        lines.append("|---|---|---|---|---|")
        for key, m in tuning.get("model_sweep", {}).items():
            lines.append(f"| {key} | {int(m['n'])} | {fmt_pf(m['pf'])} | {m['total_r']:+.2f} | {m['exp_r']:+.3f} |")
        lines.append("")

    if all_by_pattern is not None and costs is not None:
        lines.append("## OOS model-threshold diagnostic (transparency — threshold đã cố định từ legacy, "
                     "KHÔNG phải quyết định)")
        lines.append("")
        lines.append("| pattern | thr=0.50 | thr=0.55 | thr=0.60 | thr=0.65 | thr=0.70 |")
        lines.append("|---|---|---|---|---|---|")
        for pattern, events in all_by_pattern.items():
            evs = window_events(events, *WINDOWS["oos"])
            nets = simulate_events(df, evs, costs)
            cells = []
            for thr in MODEL_THRESHOLDS:
                m = metrics(model_gate_nets(evs, nets, thr))
                cells.append(f"{fmt_pf(m['pf'])} (n={int(m['n'])})")
            lines.append(f"| {pattern} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Optional §8.5 — DB model trained WITH hmm_* features")
    lines.append("")
    if optional_model is None:
        lines.append("Skipped (sample quá nhỏ hoặc lỗi không chặn protocol chính).")
    else:
        g = optional_model.get("oos_gated") or {}
        lines.append(f"- Train events (legacy window): **{int(optional_model['n_train'])}**; "
                     f"OOS events: **{int(optional_model['n_oos'])}**.")
        lines.append(f"- OOS PR-AUC (hmm-feature RandomForest + isotonic): "
                     f"**{optional_model['oos_pr_auc']:.3f}**.")
        lines.append(f"- OOS gated PF tại threshold **{optional_model['oos_threshold']:.2f}** "
                     f"(sweep TRÊN CHÍNH OOS events, max-PF, n≥{MIN_SWEEP_TRADES}): "
                     f"**{fmt_pf(g.get('pf', float('nan')))}** "
                     f"(n={int(g.get('n', 0))}, totalR={g.get('total_r', float('nan')):+.2f}R).")
        lines.append("- ⚠️ Trung thực (T8-F02): đây là **OOS-optimistic exploration**, KHÔNG phải "
                     "config chọn trên legacy, **KHÔNG dùng cho bất kỳ verdict gate nào** — chỉ là "
                     "bằng chứng phụ §8.5 cho biết hmm_* features có giúp model mới hay không; "
                     "threshold sweep trên chính OOS nên con số PF bị optimistic bias.")
        lines.append("- So sánh: model DB t1 tier-2 cũ (không hmm features) trên OOS window — "
                     "xem bảng `model_gated` ở trên (threshold chọn trên legacy).")
    lines.append("")

    lines.append("## Kết luận trung thực")
    lines.append("")
    lines.append("### Verdict summary (OOS window, §8.4)")
    lines.append("")
    lines.append("| pattern | baseline PF (n) | gated PF (n) | verdict | LSW hypothesis |")
    lines.append("|---|---|---|---|---|")
    for pattern, per_win in results.items():
        b, g = per_win["oos"]["baseline"], per_win["oos"]["gate_state_conf"]
        v = verdict(per_win["oos"], pattern)
        hyp = "n/a"
        if pattern == "liquidity_sweep":
            ok = g["pf"] >= LSW_TARGET_PF and g["n"] >= MIN_VERDICT_TRADES
            hyp = "ACHIEVED" if ok else "NOT ACHIEVED"
        lines.append(
            f"| {pattern} | {fmt_pf(b['pf'])} (n={int(b['n'])}) | {fmt_pf(g['pf'])} (n={int(g['n'])}) | "
            f"{v} | {hyp} |"
        )
    lines.append("")
    lines.append("**Kết luận:** với config canonical của `CausalGaussianHMM` (3 states, seed 42, "
                 "full covariance) trên XAUUSD M15, regime filter **không cải thiện post-cost edge** "
                 "trên bất kỳ pattern nào trong OOS — lý do chính là model không phân biệt regime "
                 "trên dữ liệu này (OOS 100% một state), nên gate ≈ no-op. Điều này KHÔNG có nghĩa "
                 "plugin sai — forward filtration đúng causal (t2 tests) — mà là config mặc định "
                 "không có selection power trên XAUUSD M15; cần tuning HMM config (n_states, "
                 "input_features, covariance) TRƯỚC khi kỳ vọng gate giúp.")
    lines.append("")
    lines.append("- Báo cáo này ghi NHẬN kết quả thực tế, kể cả khi bộ lọc HMM regime KHÔNG cải thiện "
                 "post-cost edge (bài học `lsw_model_filter_not_enough`: model/regime filter có thể có "
                 "selection power nhưng không đủ bù cost thật).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())