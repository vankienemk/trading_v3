"""
risk_guard_v2.py — Risk Guard V2 + Kill-switch Module (Block Bootstrap per Asset)

Dynamic per-symbol asset registry — reads from ``configs/symbols/*.yaml``
instead of hardcoding XAUUSD/EURUSD.  Per-symbol kill-switch using block
bootstrap confidence intervals on Profit Factor.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from live.logging.logger_v2 import get_recent_trades
from live.logging.logger_v2 import log as syslog
from live.state.shared_app_state_v2 import SYMBOL_CONFIG_DIR, SymbolConfig, get_state

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
# Single-source the per-symbol config location with the Signal Engine and the
# Execution Layer so order-sizing params (risk%, base_multiplier, max/min
# volume) are read from one canonical directory rather than diverging per module.
_DEFAULT_SYMBOLS_DIR = SYMBOL_CONFIG_DIR

DEFAULT_MAX_OPEN_PER_ASSET = 1
DEFAULT_MAX_OPEN_TOTAL = 2
DEFAULT_MIN_INTERVAL_SINCE_LAST_S = 300
DEFAULT_N_TRADES_CAP = 60
BOOTSTRAP_RESAMPLES = 3000
BOOTSTRAP_SEED_OFFSET = 1000
PF_CI_LOWER_ALPHA = 2.5
NEW_SYMBOL_MIN_TRADES = 15
NEW_SYMBOL_MAX_MULTIPLIER = 0.5


def _block_size(n: int) -> int:
    return min(25, max(1, int(2.0 * math.sqrt(n))))


def _block_bootstrap_pf_ci(
    net_r_array: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    n = len(net_r_array)
    if n == 0:
        return None, None
    block_sz = _block_size(n)
    n_blocks = max(1, n // block_sz)
    rng = np.random.default_rng(seed)
    pf_vals: list[float] = []
    for _ in range(n_resamples):
        blocks: list[np.ndarray] = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, n - block_sz + 1))
            blocks.append(net_r_array[start:start + block_sz])
        residual = n - n_blocks * block_sz
        if residual > 0:
            start = rng.integers(0, max(1, n - residual + 1))
            blocks.append(net_r_array[start:start + residual])
        boot = np.concatenate(blocks)[:n]
        pos_sum = float(boot[boot > 0].sum())
        neg_sum = float(abs(boot[boot < 0].sum()))
        if neg_sum > 0:
            pf_vals.append(pos_sum / neg_sum)
    pf_arr = np.array(pf_vals)
    pf_arr = pf_arr[np.isfinite(pf_arr)]
    if len(pf_arr) < 100:
        return None, None
    ci_lower = float(np.percentile(pf_arr, PF_CI_LOWER_ALPHA))
    ci_upper = float(np.percentile(pf_arr, 100.0 - PF_CI_LOWER_ALPHA))
    return ci_lower, ci_upper


class RiskGuard:
    """Per-symbol risk guard with block bootstrap kill-switch.
    V2: dynamic asset registry from configs/symbols/*.yaml.
    New symbols: size <= 0.5, min_trades = 15."""

    def __init__(self, db_path: str | None = None,
                 max_open_per_asset: int = DEFAULT_MAX_OPEN_PER_ASSET,
                 max_open_total: int = DEFAULT_MAX_OPEN_TOTAL,
                 min_interval_since_last_s: int = DEFAULT_MIN_INTERVAL_SINCE_LAST_S):
        self._db_path = db_path
        self._state = get_state()
        self.max_open_per_asset = max_open_per_asset
        self.max_open_total = max_open_total
        self.min_interval_since_last_s = min_interval_since_last_s
        self._symbol_configs: dict[str, dict[str, Any]] = {}
        self._kill_switch_activated_at: dict[str, float] = {}
        self._last_order_time: dict[str, float] = {}

    def load_symbol_configs(self, symbols_dir: str | None = None) -> dict[str, dict[str, Any]]:
        sym_dir = Path(symbols_dir) if symbols_dir else _DEFAULT_SYMBOLS_DIR
        if not sym_dir.is_dir():
            return {}
        loaded: dict[str, dict[str, Any]] = {}
        for yaml_path in sorted(sym_dir.glob("*.yaml")):
            if yaml_path.name == "_template.yaml":
                continue
            try:
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            name = raw.get("symbol", {}).get("name", "")
            if not name:
                continue
            is_active = raw.get("symbol", {}).get("is_active", True)
            ps = raw.get("position_sizing", {})
            base_mul = ps.get("base_multiplier", 1.0)
            status = "validated" if is_active else "candidate"
            sc = SymbolConfig(name=name, status=status,
                              position_size_multiplier=base_mul, active=is_active)
            self._state.register_symbol(name, sc)
            be_cost = ps.get("risk_per_trade_pct", 0.0) * 0.01
            self._state.set_breakeven_cost(name, be_cost)
            self._symbol_configs[name] = raw
            self._kill_switch_activated_at.setdefault(name, 0.0)
            self._last_order_time.setdefault(name, 0.0)
            loaded[name] = raw
        if loaded:
            syslog("INFO", "RiskGuard", "",
                   f"Loaded {len(loaded)} symbol configs: {', '.join(sorted(loaded.keys()))}",
                   extra={"symbols": sorted(loaded.keys())})
        return loaded

    def get_symbol_config(self, asset: str) -> dict[str, Any] | None:
        return self._symbol_configs.get(asset)

    def get_registered_symbols(self) -> list[str]:
        return self._state.get_registered_symbols()

    def _get_risk_guard_config(self, asset: str) -> dict[str, Any]:
        raw = self._symbol_configs.get(asset, {})
        rg = raw.get("risk_guard", {})
        ke = rg.get("kill_switch_enabled", True)
        mt = rg.get("kill_switch_min_trades", DEFAULT_N_TRADES_CAP)
        th = rg.get("kill_switch_threshold_pf_ci_lower", 1.0)
        sc = self._state.get_symbol_config(asset)
        if sc is not None and sc.status == "validated" and asset not in ("XAUUSD", "EURUSD"):
            mt = min(mt, NEW_SYMBOL_MIN_TRADES)
            if sc.position_size_multiplier > NEW_SYMBOL_MAX_MULTIPLIER:
                sc.position_size_multiplier = NEW_SYMBOL_MAX_MULTIPLIER
        return {"kill_switch_enabled": ke, "min_trades": mt,
                "pf_ci_lower_threshold": th,
                "max_open_positions": rg.get("max_open_positions", 1),
                "min_interval_since_last_s": rg.get("min_interval_since_last_s", 300)}

    def get_kill_switch_status(self, asset: str) -> dict[str, Any]:
        ks = self._state.kill_switch_status.get(asset)
        if ks is None:
            return {"active": False, "reason": "unknown_asset", "activated_at": 0.0}
        return {"active": ks.active, "reason": ks.reason, "activated_at": ks.activated_at}

    def get_all_kill_switch_status(self) -> dict[str, dict[str, Any]]:
        return {a: self.get_kill_switch_status(a) for a in self._state.get_registered_symbols()}

    def get_metrics(self, asset: str) -> dict[str, Any]:
        return self._calc_metrics(asset)

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        return {a: self._calc_metrics(a) for a in self._state.get_registered_symbols()}

    def get_position_size(self, asset: str) -> float:
        sc = self._state.get_symbol_config(asset)
        if sc is not None:
            return sc.position_size_multiplier
        return self._symbol_configs.get(asset, {}).get("position_sizing", {}).get("base_multiplier", 1.0)

    def refresh(self) -> None:
        for asset in self._state.get_registered_symbols():
            self._refresh_asset(asset)

    def refresh_asset(self, asset: str) -> dict[str, Any]:
        return self._refresh_asset(asset)

    def evaluate_new_signal(self, asset: str, current_open_positions: list[dict[str, Any]],
                            order_sent_time: float | None = None) -> dict[str, Any]:
        if self._state.emergency_stop:
            syslog("ERROR", "RiskGuard", asset, "Signal blocked: global emergency stop active")
            return {"allowed": False, "reason": "Global emergency stop is active"}
        # ``asset`` is the instrument name as stored in the shared state /
        # registry and used by MCP (e.g. "XAUUSDm"); ``base`` is only the
        # risk-config file name ("XAUUSD").  The unknown-asset gate accepts
        # either form, while every shared-state lookup below keys on the
        # stored name (asset first, base fallback) so the guard actually
        # binds to the live kill-switch / position / order-time records
        # instead of silently missing them for 'm'-suffixed symbols.
        base = asset.rstrip("m") if asset.endswith("m") else asset
        registered = self._state.get_registered_symbols()
        if asset not in registered and base not in registered:
            syslog("ERROR", "RiskGuard", asset, f"Signal blocked: unknown asset {asset}")
            return {"allowed": False, "reason": f"Unknown asset: {asset}"}
        rg = self._get_risk_guard_config(base)
        ks = self._state.kill_switch_status.get(asset) or self._state.kill_switch_status.get(base)
        if ks is not None and ks.active:
            syslog("WARNING", "RiskGuard", asset, f"Signal blocked by kill-switch: {ks.reason}")
            return {"allowed": False, "reason": ks.reason or "Kill-switch active"}
        mpa = rg.get("max_open_positions", self.max_open_per_asset)
        if sum(1 for p in current_open_positions
               if p.get("asset") in (asset, base)) >= mpa:
            reason = f"Max open per asset ({mpa}) for {asset}"
            syslog("WARNING", "RiskGuard", asset, f"Signal blocked: {reason}")
            return {"allowed": False, "reason": reason}
        if len(current_open_positions) >= self.max_open_total:
            reason = f"Max total open ({self.max_open_total})"
            syslog("WARNING", "RiskGuard", asset, f"Signal blocked: {reason}")
            return {"allowed": False, "reason": reason}
        now = order_sent_time if order_sent_time is not None else time.time()
        lt = self._last_order_time.get(asset, self._last_order_time.get(base, 0.0))
        if now - lt < self.min_interval_since_last_s:
            rem = self.min_interval_since_last_s - (now - lt)
            reason = f"Min interval {self.min_interval_since_last_s}s for {asset} ({rem:.0f}s remaining)"
            syslog("WARNING", "RiskGuard", asset, f"Signal blocked: {reason}")
            return {"allowed": False, "reason": reason}
        syslog("INFO", "RiskGuard", asset, "Signal passed all risk checks",
               extra={"asset": asset, "base": base,
                      "open_positions": len(current_open_positions)})
        return {"allowed": True, "reason": ""}

    def record_order_sent(self, asset: str, timestamp: float | None = None) -> None:
        self._last_order_time[asset] = timestamp if timestamp is not None else time.time()

    def _calc_metrics(self, asset: str) -> dict[str, Any]:
        trades = get_recent_trades(asset, n=DEFAULT_N_TRADES_CAP, db_path=self._db_path)
        n_valid = len([t for t in trades if t.get("net_result_r") is not None])
        net_r = np.array([t["net_result_r"] for t in trades if t.get("net_result_r") is not None], dtype=float)
        if n_valid == 0:
            return {"asset": asset, "n": 0, "n_valid": 0, "profit_factor": None,
                    "pf_ci_lower": None, "pf_ci_upper": None, "block_size": 0,
                    "kill_switch_active": False, "kill_switch_reason": ""}
        pos_sum = float(net_r[net_r > 0].sum())
        neg_sum = float(abs(net_r[net_r < 0].sum()))
        pf = pos_sum / neg_sum if neg_sum > 0 else None
        seed = BOOTSTRAP_SEED_OFFSET + hash(asset) % 10000
        bs = _block_size(n_valid)
        ci_l, ci_u = _block_bootstrap_pf_ci(net_r, n_resamples=BOOTSTRAP_RESAMPLES, seed=seed)
        rg = self._get_risk_guard_config(asset)
        mt = rg.get("min_trades", DEFAULT_N_TRADES_CAP)
        th = rg.get("pf_ci_lower_threshold", 1.0)
        ka, kr = False, ""
        if rg.get("kill_switch_enabled", True) and n_valid >= mt:
            if ci_l is not None and ci_l < th:
                ka, kr = True, f"Kill-switch: {asset} CI lower {ci_l:.4f} < {th:.4f} ({n_valid} trades)"
        else:
            kr = f"Only {n_valid}/{mt} trades for evaluation"
        m = {"asset": asset, "n": len(trades), "n_valid": n_valid,
             "profit_factor": round(pf, 4) if pf is not None else None,
             "pf_ci_lower": round(ci_l, 4) if ci_l is not None else None,
             "pf_ci_upper": round(ci_u, 4) if ci_u is not None else None,
             "block_size": bs, "kill_switch_active": ka,
             "kill_switch_reason": kr, "min_trades_for_auto": mt}
        if ka:
            already_active = self._state.kill_switch_status.get(asset)
            was_active = already_active.active if already_active else False
            self._state.update_kill_switch(asset, True, kr)
            if not was_active:
                self._kill_switch_activated_at[asset] = time.time()
                syslog("CRITICAL", "RiskGuard", asset,
                       f"Kill-switch ACTIVATED: CI lower {ci_l:.4f} < {th:.4f} ({n_valid} trades)",
                       extra={"asset": asset, "n_valid": n_valid,
                              "pf_ci_lower": round(ci_l, 4) if ci_l else None,
                              "threshold": th})
            else:
                syslog("WARNING", "RiskGuard", asset,
                       f"Kill-switch remains active: {kr}",
                       extra={"asset": asset, "ci_lower": round(ci_l, 4) if ci_l else None})
        else:
            was_inactive = not (self._state.kill_switch_status.get(asset).active if self._state.kill_switch_status.get(asset) else True)
            self._state.update_kill_switch(asset, False, kr)
            if not was_inactive:
                syslog("INFO", "RiskGuard", asset,
                       f"Kill-switch DEACTIVATED — PF CI lower {ci_l:.4f} >= {th:.4f}",
                       extra={"asset": asset, "n_valid": n_valid,
                              "pf_ci_lower": round(ci_l, 4) if ci_l else None,
                              "pf": round(pf, 4) if pf else None})
            elif n_valid >= mt:
                syslog("DEBUG", "RiskGuard", asset,
                       f"Kill-switch evaluation: PF={pf:.4f}, CI_lower={ci_l:.4f}, n={n_valid} (safe)")
        return m

    def _refresh_asset(self, asset: str) -> dict[str, Any]:
        return self._calc_metrics(asset)

    def manual_override_kill_switch(self, asset: str, active: bool, reason: str = "") -> None:
        if asset not in self._state.get_registered_symbols():
            raise ValueError(f"Unknown asset: {asset}")
        self._state.update_kill_switch(asset, active, reason)
        if active:
            self._kill_switch_activated_at[asset] = time.time()


def create_risk_guard(db_path: str | None = None,
                      max_open_per_asset: int = DEFAULT_MAX_OPEN_PER_ASSET,
                      max_open_total: int = DEFAULT_MAX_OPEN_TOTAL,
                      min_interval_since_last_s: int = DEFAULT_MIN_INTERVAL_SINCE_LAST_S) -> RiskGuard:
    return RiskGuard(db_path=db_path, max_open_per_asset=max_open_per_asset,
                     max_open_total=max_open_total, min_interval_since_last_s=min_interval_since_last_s)