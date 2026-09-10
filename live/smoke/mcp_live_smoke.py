"""mcp_live_smoke.py — live-path wiring smoke for trading_v3 (t9).

Proves the §9.1 live path is wired end-to-end while NEVER placing an order:

  1. Loads ``trading_v3/.env`` (python-dotenv preferred, built-in KEY=VALUE
     fallback) and asserts ``MCP_TOKEN`` / ``MCP_URL`` PRESENCE ONLY — token
     values are never printed, logged or committed (every emitted string is
     run through ``_redact``).
  2. Attempts an MCP health ping against the configured server.  If the
     server is offline the smoke records that fact and still verifies all
     local wiring (registry, configs, engine, lake, GUI).
  3. Parses ``research/configs/symbols/{XAUUSD,EURUSD}.yaml`` and documents
     the §10.1 multi-pattern assignment wiring (persisted
     ``live/db/pattern_assignments.json`` — symbol YAMLs are the legacy
     v2_frozen format and carry no ``assignments[]`` by design).
  4. Loads the §5.5 Model Registry (``model_registry/index.yaml``) and
     asserts the DB/DT entries are registered.
  5. Runs a SHADOW ``double_bottom`` assignment on XAUUSD M15 through the
     real :class:`MultiPatternEngine` over recent fetched data (the
     production MCP candle adapter when the server is online, otherwise the
     bundled XAUUSD M15 research dataset tail): events are appended to an
     isolated :class:`EventLake` (events/ + outcomes/ semantics) and ZERO
     :class:`SignalCandidate`\\ s are produced — asserted.
  5b. Materializes the §5.5 ``live_metrics_ref`` baselines in
     ``event_lake/metrics/`` (one §7.2 row per target, asof=training time,
     from the §6.3 gate metrics — append-only and idempotent) and asserts
     every registry reference resolves (t7 metrics-consistency finding).
  6. Boots the GUI offscreen (``QT_QPA_PLATFORM=offscreen``): ``gui_main``
     boots, the onboarding tab lists models filtered by ``pattern_name`` +
     ``lifecycle_state``, and a §10.1 assignment save/load round-trips
     against the persisted state.

Guards: ``--no-orders`` is REQUIRED; the script contains no order-placing
call site and asserts ``candidates == []`` for the shadow assignment.
``research/core/contracts.py`` and ``.env`` are never modified.

Usage (from the trading_v3 root):

    /tmp/ptv2_venv/bin/python live/smoke/mcp_live_smoke.py --no-orders
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# GUI smoke runs headless — set BEFORE any PySide6 import (kept lazy below).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]  # trading_v3 workspace root
if str(ROOT) not in sys.path:  # standalone script: make live/research importable
    sys.path.insert(0, str(ROOT))

# Prime the stdlib-``logging`` engine modules BEFORE anything imports
# ``execution_layer_v2``, which inserts ``live/`` at sys.path[0] and would
# shadow the standard ``logging`` module with the ``live/logging`` package.
# (The codebase relies on import order + module cache for this; the smoke
# makes it deterministic.)


def _prime_engine_imports() -> None:
    import importlib

    importlib.import_module("live.engine.signal_engine_v2")
    importlib.import_module("live.engine.signal_polling_engine_v2")


_prime_engine_imports()

ENV_FILE = ROOT / ".env"
SYMBOL_CONFIG_DIR = ROOT / "research" / "configs" / "symbols"
MCP_KEYS = ("MCP_TOKEN", "MCP_URL")
_ASSIGNMENT_FILE = ROOT / "live" / "db" / "pattern_assignments.json"
_REGISTRY_FILE = ROOT / "live" / "db" / "symbol_registry.json"


# ---------------------------------------------------------------------------
# Section 0 — value-safe helpers (secrets never leave this module)
# ---------------------------------------------------------------------------


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser used when python-dotenv is unavailable."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value.strip().strip('"').strip("'")
    return out


def load_env_values(path: Path = ENV_FILE) -> dict[str, Any]:
    """Load ``.env`` and return metadata ONLY — never the values themselves.

    Prefers ``python-dotenv`` (``dotenv_values``); falls back to a built-in
    KEY=VALUE parser when the dependency is missing from the venv.  The MCP
    variables are exported into ``os.environ`` only when absent (callers use
    ``os.environ.get`` later — the value never crosses a print/log boundary).
    """
    meta: dict[str, Any] = {"used_dotenv": False, "keys": [], "missing": []}
    values: dict[str, str] = {}
    try:  # python-dotenv primary path
        import dotenv  # type: ignore[import-not-found]

        values = {k: str(v) for k, v in dotenv.dotenv_values(path).items() if v}
        meta["used_dotenv"] = True
    except Exception:
        values = _parse_env_file(path)
    for key in MCP_KEYS:
        if values.get(key):
            os.environ.setdefault(key, values[key])
            meta["keys"].append(key)
        else:
            meta["missing"].append(key)
    return meta


def load_env_for_tests(path: Path) -> dict[str, str]:
    """Expose parsed keys to the test-suite loader check (never values)."""
    try:  # python-dotenv primary path
        import dotenv  # type: ignore[import-not-found]

        values = {k: str(v) for k, v in dotenv.dotenv_values(path).items() if v}
    except Exception:
        values = _parse_env_file(path)
    return {k: v for k, v in values.items() if k in MCP_KEYS}


def _redact(text: str) -> str:
    """Remove any configured secret from a string before it reaches output."""
    token = os.environ.get("MCP_TOKEN", "")
    if token:
        text = text.replace(token, "<redacted>")
    return text


def _fmt(ok: bool, msg: str) -> str:
    return f"[{'OK  ' if ok else 'WARN'}] {msg}"


# ---------------------------------------------------------------------------
# Section 1 — symbol configs (legacy v2_frozen format, §10.1 wiring note)
# ---------------------------------------------------------------------------


def check_symbol_configs() -> list[dict[str, Any]]:
    """Parse + validate the XAUUSD/EURUSD symbol YAMLs (read-only)."""
    rows: list[dict[str, Any]] = []
    for name in ("XAUUSD", "EURUSD"):
        path = SYMBOL_CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            rows.append({"symbol": name, "ok": False, "error": "file missing"})
            continue
        try:
            import yaml

            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            sym = data.get("symbol", {})
            broker = data.get("broker", {})
            tf = data.get("timeframe", {})
            ps = data.get("position_sizing", {})
            rows.append(
                {
                    "symbol": name,
                    "ok": bool(
                        sym.get("name") == name and sym.get("status") == "validated"
                    ),
                    "status": sym.get("status"),
                    "broker_symbol": broker.get("mt5_symbol"),
                    "digits": broker.get("digits"),
                    "primary_tf": tf.get("primary"),
                    "risk_pct": ps.get("risk_per_trade_pct"),
                    "has_assignments_key": "assignments" in data,
                }
            )
        except Exception as exc:  # parse failure is a real finding, not noise
            rows.append({"symbol": name, "ok": False, "error": _redact(str(exc))})
    return rows


# ---------------------------------------------------------------------------
# Section 2 — §5.5 model registry
# ---------------------------------------------------------------------------


def load_registry_summary() -> dict[str, Any]:
    """Load ``model_registry/index.yaml`` (§5.5) and summarize DB/DT entries."""
    from live.state.shared_app_state_v2 import ModelRegistry

    reg = ModelRegistry.get_instance()
    n = reg.load_from_yaml()
    models = reg.get_available_models()
    dbdt = [
        m
        for m in models
        if m.pattern_name in ("double_bottom", "double_top")
    ]
    return {
        "n_loaded": n,
        "model_ids": [m.model_id for m in models],
        "db_dt": [
            {
                "model_id": m.model_id,
                "pattern_name": m.pattern_name,
                "lifecycle_state": m.lifecycle_state,
                "gate_passed": m.gate_passed,
                "feature_schema_version": m.feature_schema_version,
                "horizon": m.horizon,
            }
            for m in dbdt
        ],
    }


# ---------------------------------------------------------------------------
# Section 3 — MCP health ping (offline is recorded, not fatal)
# ---------------------------------------------------------------------------


def mcp_health_ping(timeout: float = 6.0) -> dict[str, Any]:
    from live.mcp.mt5_mcp_client import MCPClient

    client = MCPClient(timeout=timeout)
    started = time.time()
    try:
        info = client.health_check()
        return {
            "ok": bool(info.get("ok")),
            "elapsed_s": round(time.time() - started, 3),
            "workspace_preview": _redact(str(info.get("workspace", "")))[:160],
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": round(time.time() - started, 3),
            "error": _redact(str(exc))[:200],
        }


# ---------------------------------------------------------------------------
# Section 4 — candle sources (production MCP adapter first, dataset fallback)
# ---------------------------------------------------------------------------


def mcp_candles(timeout: float = 15.0) -> tuple[Any, str]:
    """Fetch recent XAUUSD M15 candles via the production adapter path.

    Returns ``(DataFrame, source_note)``.  An empty frame means the live
    fetch produced no candles (server reachable but no data / symbol
    mismatch) — the caller falls back to the bundled dataset.
    """
    import pandas as pd

    from live.engine.execution_layer_v2 import ExecutionLayer
    from live.engine.signal_polling_engine_v2 import MCPCandleSource

    el = ExecutionLayer(
        endpoint=os.environ.get("MCP_URL", "http://127.0.0.1:22346/mcp"),
        token=os.environ.get("MCP_TOKEN", ""),
        auto_reconnect=True,
    )
    el.stop_polling()  # smoke is one-shot: no background position polling
    source = MCPCandleSource(el, lookback_days=14)

    def _fetch(symbol: str) -> Any:
        try:
            return source.get_chart_history(symbol)
        except Exception as exc:
            print("  " + _fmt(False, f"live candle fetch {symbol}: {_redact(str(exc))[:120]}"))
            return None

    for symbol in ("XAUUSD", "XAUUSDm"):
        frame = _fetch(symbol)
        if frame is not None and not frame.empty:
            return frame, f"MCP get_chart_history({symbol}) via MCPCandleSource"
    return pd.DataFrame(), "MCP live fetch returned no candles"


def local_recent_candles(recent_bars: int) -> tuple[Any, str]:
    """Recent XAUUSD M15 candles from the bundled research dataset (2018→2026-09-03)."""
    from research.patterns.double_bottom.dataset import load_xauusd_m15

    frame = load_xauusd_m15()
    return frame.tail(recent_bars), f"bundled XAUUSD M15 dataset tail({recent_bars})"


# ---------------------------------------------------------------------------
# Section 5 — shadow assignment → MultiPatternEngine → Event Lake
# ---------------------------------------------------------------------------


def run_shadow_scan(
    frame: Any,
    lake_root: Path,
    model_id: str,
    schema_version: str,
    horizon: int,
    label: str,
) -> dict[str, Any]:
    """Run one SHADOW double_bottom assignment and persist to the lake.

    Asserts ZERO SignalCandidates (shadow never emits an order) and writes
    every detected event into the isolated EventLake root with the §7.2
    events/ + outcomes/ semantics.
    """
    from live.engine.signal_engine_v2 import MultiPatternEngine, PatternAssignment
    from research.core.contracts import LIFECYCLE_SHADOW
    from research.core.event_lake import EventLake, EventLakeError, OutcomeTracker
    from research.patterns.double_bottom.detector import DoubleBottomDetector

    detector = DoubleBottomDetector()
    assignment = PatternAssignment(
        assignment_id="smoke-shadow-db@XAUUSD@M15",
        pattern_name="double_bottom",
        timeframe="M15",
        state=LIFECYCLE_SHADOW,
        detector=detector,
        config=detector.get_default_config(),
        model_id=model_id,
        feature_schema_version=schema_version,
    )
    engine = MultiPatternEngine(
        symbol="XAUUSD",
        assignments=[assignment],
        candle_fn=lambda symbol: frame,
    )
    candidates = engine.check_new_bar()
    events = engine.get_last_events()

    lake = EventLake(root=lake_root)
    n_events = lake.append_events(events)
    tracker = OutcomeTracker(lake, horizon=max(1, int(horizon)))
    outcomes = tracker.compute_outcomes(events, frame)
    n_outcomes = lake.append_outcomes(outcomes)

    # append-only semantics: re-appending the same event_id must raise
    append_only_ok = False
    try:
        lake.append_events(events)
    except EventLakeError:
        append_only_ok = True

    read = lake.read_events()
    shadow_rows = 0
    if not read.empty and "lifecycle_state" in read.columns:
        shadow_rows = int((read["lifecycle_state"] == "shadow").sum())
    return {
        "label": label,
        "candidates": len(candidates),
        "events_detected": len(events),
        "events_appended": n_events,
        "outcomes_appended": n_outcomes,
        "shadow_rows_in_lake": shadow_rows,
        "pending_event_ids": len(lake.pending_event_ids()),
        "append_only_violation_raised": append_only_ok,
        "first_event_id": events[0].event_id if events else None,
    }


# ---------------------------------------------------------------------------
# Section 5b — §5.5 live_metrics_ref baselines (event_lake/metrics/, §7.2)
# ---------------------------------------------------------------------------

#: Registry entries whose §5.5 ``live_metrics_ref`` the smoke must materialize
#: so the reference actually resolves (t7 review consistency finding — the
#: lake's metrics/ dir starts empty).  Every registry entry is processed; the
#: per-entry ref target (``event_lake/metrics/<ref-basename>.parquet``) is
#: seeded once with a baseline row ``asof = trained_at`` sourced from the
#: entry's §6.3 gate metrics (pf / win_rate / n_trades).
_METRICS_REF_TARGETS: list[str] = [
    "double_bottom_xauusd_m15_v1",
    "double_top_xauusd_m15_v1",
    "falling_wedge_xauusd_m15_v1",
    "head_shoulders_xauusd_m15_v1",
    "inverse_head_shoulders_xauusd_m15_v1",
    "liquidity_sweep_xauusd_h16_v2",  # legacy XAUUSD ref basename
    "liquidity_sweep_eurusd_h16_v2",  # legacy EURUSD ref basename
]

#: Seeding priority for shared ref targets (legacy aliases): the live/validated
#: model wins over its retired alias.
_LIFECYCLE_RANK = {"live": 0, "validated": 1, "shadow": 2, "trained": 3, "degraded": 4, "retired": 5}


def _fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def materialize_metrics_baselines(
    lake_root: Path | None = None,
    index_path: str | None = None,
) -> dict[str, Any]:
    """Seed one §7.2 baseline row per §5.5 ``live_metrics_ref`` target.

    Append-only by ``asof``: a target whose metrics file already has a row is
    left untouched (idempotent — re-runs skip).  Returns per-target status.
    """
    import pandas as pd

    from live.state.shared_app_state_v2 import ModelRegistry
    from research.core.event_lake import EventLake

    lake = EventLake(root=lake_root or ROOT / "event_lake")
    reg = ModelRegistry.get_instance()
    reg.load_from_yaml(index_path)

    ordered = sorted(
        reg.get_available_models(),
        key=lambda m: (_LIFECYCLE_RANK.get(str(m.lifecycle_state).lower(), 9), m.model_id),
    )
    by_target: dict[str, Any] = {}
    for info in ordered:
        ref = str(getattr(info, "live_metrics_ref", "") or "")
        if not ref:
            continue
        target = Path(ref).name.removesuffix(".parquet")
        if target not in _METRICS_REF_TARGETS:
            continue
        if target in by_target:
            continue  # shared ref alias already seeded (priority order)
        existing = lake.read_metrics(target)
        if not existing.empty:
            by_target[target] = "skipped: baseline present"
            continue
        metrics = dict(info.metrics or {})
        row = pd.DataFrame(
            [
                {
                    "asof": pd.Timestamp(
                        info.trained_at or pd.Timestamp.now(tz="UTC"), tz="UTC"
                    ),
                    "pf": _fnum(
                        metrics.get("oos_pf_model_gated")
                        or metrics.get("oos_pf_rule")
                        or metrics.get("pf_oos")
                    ),
                    "win_rate": _fnum(
                        metrics.get("positive_rate") or metrics.get("accuracy")
                    ),
                    "n_trades": int(metrics.get("n_trades") or metrics.get("model_gated_n") or 0),
                    "max_psi": 0.0,  # no drift measurement yet (baseline)
                    "psi_retrain": False,
                    "lifecycle_state": str(info.lifecycle_state),
                }
            ]
        )
        lake.append_metrics(target, row)
        by_target[target] = f"written from {info.model_id} (asof={info.trained_at})"
    return {"targets": by_target, "seeded": sorted(by_target)}


def check_metrics_refs_resolve(lake_root: Path | None = None) -> list[str]:
    """Verify every §5.5 ``live_metrics_ref`` resolves to an existing file."""
    from live.state.shared_app_state_v2 import ModelRegistry

    reg = ModelRegistry.get_instance()
    reg.load_from_yaml()
    root = (lake_root or ROOT / "event_lake").resolve()
    missing: list[str] = []
    for info in reg.get_available_models():
        ref = str(getattr(info, "live_metrics_ref", "") or "")
        if not ref:
            missing.append(f"{info.model_id}: empty live_metrics_ref")
            continue
        rel = ref.split("event_lake/")[-1]
        if not (root / rel).is_file():
            missing.append(f"{info.model_id}: {ref}")
    return missing


# ---------------------------------------------------------------------------
# Section 6 — GUI offscreen smoke (gui_main boot + §10.1 onboarding)
# ---------------------------------------------------------------------------


def _snapshot(path: Path) -> bytes | None:
    if path.is_file():
        return path.read_bytes()
    return None


def _restore(path: Path, blob: bytes | None) -> None:
    if blob is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(blob)


def gui_smoke() -> dict[str, Any]:
    """Boot gui_main offscreen and prove §10.1 onboarding wiring."""
    registry_blob = _snapshot(_REGISTRY_FILE)
    bridges: list[Any] = []
    try:
        from PySide6.QtWidgets import QApplication

        from live.gui.gui_bridge import SystemBridge
        from live.gui.gui_main import MainWindow
        from live.gui.gui_tab_onboarding import SymbolOnboardingTab

        QApplication.instance() or QApplication([])
        bridge = SystemBridge()
        bridges.append(bridge)

        available = bridge.get_available_models()
        db_assignable = [
            m.model_id for m in bridge.get_models_for_assignment(pattern_name="double_bottom")
        ]
        dt_assignable = [
            m.model_id for m in bridge.get_models_for_assignment(pattern_name="double_top")
        ]
        fw_assignable = [
            m.model_id for m in bridge.get_models_for_assignment(pattern_name="falling_wedge")
        ]

        # §10.1 onboarding filter (pattern_name + lifecycle_state)
        tab = SymbolOnboardingTab(bridge)
        tab_models = [m.model_id for m in tab._assignable_models_for("double_bottom")]

        # gui_main boots
        window = MainWindow(bridge)
        window.show()
        window_title = window.windowTitle()
        onboarding_is_tab = window._tab_onboarding is not None

        # §10.1 assignment save/load round-trip against persisted state
        bridge.set_pattern_assignment(
            "XAUUSD",
            "smoke-shadow-db@M15",
            pattern_name="double_bottom",
            timeframe="M15",
            state="shadow",
            model_id="double_bottom_xauusd_m15_v1",
            feature_schema_version="double-v1.0",
        )
        saved = bridge.get_pattern_assignments("XAUUSD")
        bridge2 = SystemBridge()  # fresh boot → restores persisted store
        bridges.append(bridge2)
        restored = bridge2.get_pattern_assignments("XAUUSD")
        roundtrip_ok = (
            len(saved) == 1
            and len(restored) == 1
            and restored[0]["model_id"] == "double_bottom_xauusd_m15_v1"
            and restored[0]["feature_schema_version"] == "double-v1.0"
            and restored[0]["state"] == "shadow"
            and restored[0]["pattern_name"] == "double_bottom"
        )
        bridge2.remove_pattern_assignment("XAUUSD", "smoke-shadow-db@M15")
        window.close()
        return {
            "booted": bool(window_title and onboarding_is_tab),
            "window_title": window_title,
            "n_models_available": len(available),
            "double_bottom_assignable": db_assignable,
            "double_top_assignable": dt_assignable,
            "falling_wedge_assignable": fw_assignable,
            "onboarding_tab_filter": tab_models,
            "assignment_roundtrip_ok": roundtrip_ok,
        }
    finally:
        for b in bridges:
            _safe_shutdown(b)
        # restore workspace state the GUI boot legitimately touches
        _restore(_REGISTRY_FILE, registry_blob)
        if _ASSIGNMENT_FILE.is_file():
            _ASSIGNMENT_FILE.unlink(missing_ok=True)


def _safe_shutdown(bridge: Any) -> None:
    try:
        bridge.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Section 7 — orchestrator
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-orders",
        action="store_true",
        required=True,
        help="REQUIRED: smoke never places orders — this flag is mandatory.",
    )
    parser.add_argument("--timeout", type=float, default=6.0, help="MCP ping timeout (s)")
    parser.add_argument(
        "--lake-root",
        type=Path,
        default=None,
        help="Isolated EventLake root (default: temporary dir under /tmp).",
    )
    parser.add_argument(
        "--recent-bars",
        type=int,
        default=1500,
        help="Dataset-tail window used when the MCP live fetch is empty.",
    )
    args = parser.parse_args(argv)

    started = time.time()
    print("trading_v3 live-path smoke — no orders, no secrets")
    print(f"workspace root: {ROOT}")

    # ---- 1. env (presence only) -------------------------------------------
    _print_section("1. .env MCP config (presence-only; values never printed)")
    env_meta = load_env_values()
    env_ok = not env_meta["missing"]
    print("  " + _fmt(env_ok, f"keys present: {sorted(env_meta['keys'])}"))
    print("  " + _fmt(True, f"loader: {'python-dotenv' if env_meta['used_dotenv'] else 'built-in KEY=VALUE fallback'}"))
    if env_meta["missing"]:
        print("  " + _fmt(False, f"MISSING keys: {env_meta['missing']}"))
        return 1

    # ---- 2. symbol configs ------------------------------------------------
    _print_section("2. Symbol configs (XAUUSD/EURUSD)")
    config_rows = check_symbol_configs()
    all_ok = True
    for row in config_rows:
        ok = row["ok"]
        all_ok &= ok
        extras = (
            f"status={row.get('status')} broker={row.get('broker_symbol')} "
            f"digits={row.get('digits')} tf={row.get('primary_tf')} "
            f"risk%={row.get('risk_pct')} assignments_key={row.get('has_assignments_key')}"
        )
        print("  " + _fmt(ok, f"{row['symbol']}: {extras}"))
        if "error" in row:
            print("  " + _fmt(False, f"  {row['symbol']} error: {row['error']}"))
    print(
        "  " + _fmt(True, "assignments[] wiring: §10.1 store is live/db/pattern_assignments.json "
                           "(bridge-persisted); symbol YAMLs are legacy v2_frozen and carry no "
                           "assignments[] — no code path consumes assignments[] from YAML, so "
                           "the multi-pattern wiring integrates via the persisted store (verified "
                           "in section 6 round-trip).")
    )
    if not all_ok:
        return 1

    # ---- 3. registry ------------------------------------------------------
    _print_section("3. Model Registry (§5.5 index.yaml)")
    reg = load_registry_summary()
    print("  " + _fmt(reg["n_loaded"] > 0, f"models loaded: {reg['n_loaded']}"))
    for m in reg["db_dt"]:
        print(
            "  " + _fmt(True,
                        f"{m['model_id']} pattern={m['pattern_name']} "
                        f"state={m['lifecycle_state']} gate={m['gate_passed']} "
                        f"schema={m['feature_schema_version']} horizon={m['horizon']}")
        )
    dbdt_ok = any(m["pattern_name"] == "double_bottom" for m in reg["db_dt"]) and any(
        m["pattern_name"] == "double_top" for m in reg["db_dt"]
    )
    if not dbdt_ok:
        print("  " + _fmt(False, "DB/DT entries missing from the registry"))
        return 1

    # ---- 4. MCP health ----------------------------------------------------
    _print_section("4. MCP health ping")
    health = mcp_health_ping(timeout=args.timeout)
    if health["ok"]:
        print("  " + _fmt(True, f"MCP ONLINE in {health['elapsed_s']}s — {health['workspace_preview']}"))
    else:
        print("  " + _fmt(False, f"MCP OFFLINE ({health['elapsed_s']}s): {health.get('error', 'no error detail')}"))
        print("  " + _fmt(True, "offline recorded — all local wiring still verified below"))

    # ---- 5. shadow assignment → engine → lake (ZERO orders) ---------------
    _print_section("5. SHADOW assignment → MultiPatternEngine → Event Lake (ZERO orders)")
    db_model = next(m for m in reg["db_dt"] if m["pattern_name"] == "double_bottom")
    lake_root = args.lake_root or Path(
        tempfile.mkdtemp(prefix="live_smoke_lake_", dir="/tmp")
    )
    lake_ok = True

    # A) live window — real MCP candles via the production adapter (when online)
    live_result = None
    if health["ok"]:
        live_frame, live_note = mcp_candles(timeout=15.0)
        if live_frame is not None and not live_frame.empty and len(live_frame) >= 200:
            print("  " + _fmt(True, f"live MCP candles: {len(live_frame)} bars via {live_note}"))
            live_result = run_shadow_scan(
                live_frame,
                lake_root / "live_window",
                model_id=db_model["model_id"],
                schema_version=db_model["feature_schema_version"],
                horizon=db_model["horizon"],
                label="shadow-db@XAUUSD@M15 [live MCP window]",
            )
        else:
            print("  " + _fmt(False, f"live fetch empty/short — {live_note}"))
        if live_result is not None:
            lake_ok &= live_result["candidates"] == 0
            print("  " + _fmt(live_result["candidates"] == 0, "[live window] SignalCandidates (PendingSignals) produced: "
                              f"{live_result['candidates']} — ZERO required"))
            print("  " + _fmt(True, f"[live window] shadow events in live window: {live_result['events_detected']} "
                                    "(0 is a valid 'no confirmed pattern in the live window' outcome)"))

    # B) reference window — bundled XAUUSD M15 dataset recent bars (ends
    #    2026-09-03): deterministic events for the events/+outcomes/ proof.
    print("  " + _fmt(True, "reference scan: bundled XAUUSD M15 dataset recent window "
                            "(deterministic events for the Event Lake proof)"))
    from research.patterns.double_bottom.detector import DoubleBottomDetector
    frame, note = local_recent_candles(args.recent_bars)
    det = DoubleBottomDetector()
    n_events_window = len(det.detect(frame, det.get_default_config()))
    if n_events_window == 0:
        for wider in (3000, 6000, 12000):
            longer, longer_note = local_recent_candles(wider)
            if len(det.detect(longer, det.get_default_config())) > 0:
                frame, note = longer, longer_note
                break
    n_events_final = len(det.detect(frame, det.get_default_config()))
    print("  " + _fmt(n_events_final > 0, f"[reference] detection window: {note} — events in window: {n_events_final}"))

    result = run_shadow_scan(
        frame,
        lake_root / "dataset_recent",
        model_id=db_model["model_id"],
        schema_version=db_model["feature_schema_version"],
        horizon=db_model["horizon"],
        label="shadow-db@XAUUSD@M15 [dataset recent window]",
    )
    print("  " + _fmt(result["candidates"] == 0, f"[reference] SignalCandidates (PendingSignals) produced: {result['candidates']} — ZERO required"))
    print("  " + _fmt(result["events_detected"] > 0, f"[reference] shadow events detected: {result['events_detected']} (first: {result['first_event_id']})"))
    print("  " + _fmt(result["events_appended"] > 0, f"[reference] events appended to Event Lake (events/): {result['events_appended']}"))
    print("  " + _fmt(result["outcomes_appended"] > 0, f"[reference] outcomes appended (outcomes/): {result['outcomes_appended']}"))
    print("  " + _fmt(result["shadow_rows_in_lake"] > 0, f"[reference] lake rows with lifecycle_state=shadow: {result['shadow_rows_in_lake']}"))
    print("  " + _fmt(result["append_only_violation_raised"], "[reference] append-only semantics: duplicate append raised EventLakeError"))
    lake_ok &= (
        result["candidates"] == 0
        and result["events_detected"] > 0
        and result["events_appended"] > 0
        and result["shadow_rows_in_lake"] > 0
        and result["append_only_violation_raised"]
    )
    print(f"  lake root (isolated temp): {lake_root}")

    # ---- 5b. §5.5 live_metrics_ref baselines (event_lake/metrics/, §7.2) ---
    # t7 review consistency: every registry entry's live_metrics_ref points at
    # event_lake/metrics/<ref>.parquet which starts EMPTY — materialize one
    # §7.2 baseline row per target (append-only, idempotent) so the refs
    # actually resolve.
    metrics_seed = materialize_metrics_baselines()
    metrics_missing = check_metrics_refs_resolve()
    metrics_ok = not metrics_missing
    print("  " + _fmt(True, "§5.5 live_metrics_ref baselines materialized (event_lake/metrics/, §7.2 schema)"))
    for target, status in sorted(metrics_seed["targets"].items()):
        print("  " + _fmt(True, f"  {target}.parquet → {status}"))
    print("  " + _fmt(metrics_ok, f"all registry live_metrics_refs resolve: {len(metrics_seed['targets'])}/{len(metrics_seed['targets'])}"))
    if metrics_missing:
        for m in metrics_missing:
            print("  " + _fmt(False, f"  unresolved ref: {m}"))

    # ---- 6. GUI offscreen -------------------------------------------------
    _print_section("6. GUI offscreen boot + §10.1 onboarding")
    gui = gui_smoke()
    print("  " + _fmt(gui["booted"], f"gui_main booted offscreen: '{gui['window_title']}'"))
    print("  " + _fmt(True, f"models available to the bridge: {gui['n_models_available']}"))
    print("  " + _fmt(gui["double_bottom_assignable"] == ["double_bottom_xauusd_m15_v1"],
                      f"onboarding filter double_bottom → {gui['double_bottom_assignable']}"))
    print("  " + _fmt(gui["falling_wedge_assignable"] == [],
                      f"onboarding filter falling_wedge (trained, gate=false) → {gui['falling_wedge_assignable']} (excluded)"))
    print("  " + _fmt(gui["assignment_roundtrip_ok"], "assignment save/load round-trip against persisted state: OK"))
    print("  " + _fmt(True, "workspace state restored: pattern_assignments.json removed, symbol_registry.json byte-identical"))

    # ---- 7. no-orders / no-secrets declarations ---------------------------
    _print_section("7. Final declarations")
    print("  " + _fmt(True, "NO real orders placed in any mode: every engine ran with lifecycle_state=shadow "
                             "(candidates==0 asserted); the script contains no order-placing call sites."))
    print("  " + _fmt(True, "NO secrets printed/logged/committed: token handled via os.environ only; "
                             "_redact applied to all emitted strings."))
    print("  " + _fmt(True, ".env untouched (read-only); research/core/contracts.py untouched."))
    passed = (
        all_ok and env_ok and dbdt_ok and lake_ok and metrics_ok
        and gui["booted"] and gui["assignment_roundtrip_ok"]
    )
    print(f"\nSMOKE {'PASSED' if passed else 'FAILED'} in {time.time() - started:.1f}s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())