"""Live-path wiring smoke tests — t9 (live-verifier).

Hermetic tests for the trading_v3 §9.1 live wiring that NEVER touch the
network (no MCP calls, no order paths):

  1. ``.env`` loader exposes MCP key presence only — values never leave it.
  2. §5.5 Model Registry loads with DB/DT entries; GUI assignment filter
     excludes trained/retired models.
  3. XAUUSD/EURUSD symbol configs parse (legacy v2_frozen; the §10.1
     assignment store lives in ``live/db/pattern_assignments.json`` — symbol
     YAMLs carry no ``assignments[]`` by design, asserted).
  4. A SHADOW ``double_bottom`` assignment on real XAUUSD M15 data runs
     through the real MultiPatternEngine: ZERO SignalCandidates and every
     event appended to an isolated EventLake (events/ + outcomes/ semantics,
     append-only violation raised on duplicates).
  5. Offscreen GUI: ``gui_main`` boots, onboarding lists models filtered by
     ``pattern_name`` + ``lifecycle_state``, and a §10.1 assignment
     save/load round-trips against the persisted state (workspace files
     restored afterwards).
  6. Static guard: the smoke script contains no order-placing call sites.

The shadow engine path here mirrors the committed smoke script
(``live/smoke/mcp_live_smoke.py``) — the acceptance runnable is exercised
by the same helpers this suite imports.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from live.smoke.mcp_live_smoke import (
    _ASSIGNMENT_FILE,
    _REGISTRY_FILE,
    ENV_FILE,
    ROOT,
    _parse_env_file,
    check_metrics_refs_resolve,
    check_symbol_configs,
    gui_smoke,
    load_env_for_tests,
    load_registry_summary,
    materialize_metrics_baselines,
    run_shadow_scan,
)

# ---------------------------------------------------------------------------
# 1. Env loader — presence only, values never printed/exposed
# ---------------------------------------------------------------------------


def test_env_loader_exposes_key_presence_only() -> None:
    result = load_env_for_tests(ENV_FILE)
    assert set(result) == {"MCP_TOKEN", "MCP_URL"}
    for value in result.values():
        assert value, "env loader must find non-empty values (presence only)"


def test_env_parser_handles_comments_quotes_and_blank_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nMCP_TOKEN='abc123'\nMCP_URL=\"http://127.0.0.1:22346/mcp\"\n",
        encoding="utf-8",
    )
    parsed = _parse_env_file(env)
    assert parsed == {
        "MCP_TOKEN": "abc123",
        "MCP_URL": "http://127.0.0.1:22346/mcp",
    }


# ---------------------------------------------------------------------------
# 2. Registry (§5.5) + GUI assignment filter
# ---------------------------------------------------------------------------


def test_registry_loads_db_dt_entries() -> None:
    reg = load_registry_summary()
    assert reg["n_loaded"] >= 7
    db = next(m for m in reg["db_dt"] if m["pattern_name"] == "double_bottom")
    dt = next(m for m in reg["db_dt"] if m["pattern_name"] == "double_top")
    for m in (db, dt):
        assert m["lifecycle_state"] == "validated"
        assert m["gate_passed"] is True
        assert m["feature_schema_version"].startswith("double-")


def test_assignment_filter_excludes_trained_and_retired() -> None:
    from live.state.shared_app_state_v2 import ModelRegistry

    reg = ModelRegistry.get_instance()
    reg.load_from_yaml()
    assignable = reg.get_models_for_assignment(pattern_name="double_bottom")
    assert [m.model_id for m in assignable] == ["double_bottom_xauusd_m15_v1"]
    # trained explore models and the retired alias are never offered by the
    # default §10.1 filter ({validated, shadow, live})
    assert reg.get_models_for_assignment(pattern_name="falling_wedge") == []
    liquidity = reg.get_models_for_assignment(pattern_name="liquidity_sweep")
    assert "xauusd_v2" not in {m.model_id for m in liquidity}
    # an explicit retired filter still resolves the alias (audit/back-compat)
    retired = reg.get_models_for_assignment(
        pattern_name="liquidity_sweep", lifecycle_states={"retired"}
    )
    assert [m.model_id for m in retired] == ["xauusd_v2"]


# ---------------------------------------------------------------------------
# 3. Symbol configs
# ---------------------------------------------------------------------------


def test_symbol_configs_parse() -> None:
    rows = {row["symbol"]: row for row in check_symbol_configs()}
    assert set(rows) == {"XAUUSD", "EURUSD"}
    for name in ("XAUUSD", "EURUSD"):
        row = rows[name]
        assert row["ok"] is True
        assert row["status"] == "validated"
        assert row["digits"] == 5
        assert row["primary_tf"] == "M15"
        # §10.1: symbol YAMLs are the legacy v2_frozen format — multi-pattern
        # assignments live in live/db/pattern_assignments.json (bridge store)
        assert row["has_assignments_key"] is False


def test_assignment_store_is_the_multi_pattern_source() -> None:
    """The §10.1 assignments[] wiring is the bridge-persisted JSON store —
    verified by exercising the same round-trip the smoke does."""
    assert _ASSIGNMENT_FILE.parent.is_dir()  # live/db exists (store created on save)
    gui = gui_smoke()
    assert gui["assignment_roundtrip_ok"] is True


# ---------------------------------------------------------------------------
# 4. SHADOW assignment → real engine → Event Lake (zero orders)
# ---------------------------------------------------------------------------


def test_shadow_scan_zero_candidates_and_lake_events(tmp_path: Path) -> None:
    from research.patterns.double_bottom.dataset import load_xauusd_m15
    from research.patterns.double_bottom.detector import DoubleBottomDetector

    frame = load_xauusd_m15().tail(2000)  # recent XAUUSD M15 (dataset ends 2026-09-03)
    det = DoubleBottomDetector()
    assert len(det.detect(frame, det.get_default_config())) >= 1

    result = run_shadow_scan(
        frame,
        lake_root=tmp_path / "lake",
        model_id="double_bottom_xauusd_m15_v1",
        schema_version="double-v1.0",
        horizon=72,
        label="test-shadow-db@XAUUSD@M15",
    )
    # ZERO PendingSignals / orders — shadow never emits
    assert result["candidates"] == 0
    # events ARE written to the lake (events/ + outcomes/ semantics)
    assert result["events_detected"] >= 1
    assert result["events_appended"] == result["events_detected"]
    assert result["outcomes_appended"] == result["events_detected"]
    assert result["shadow_rows_in_lake"] >= 1
    assert result["append_only_violation_raised"] is True

    # spot-check the persisted layout + lifecycle stamp
    from research.core.event_lake import EventLake

    lake = EventLake(root=tmp_path / "lake")
    events = lake.read_events()
    assert not events.empty
    assert set(events["lifecycle_state"].unique()) == {"shadow"}
    assert (tmp_path / "lake" / "events").is_dir()
    assert (tmp_path / "lake" / "outcomes").is_dir()


# ---------------------------------------------------------------------------
# 4b. §5.5 live_metrics_ref baselines (event_lake/metrics/, §7.2)
# ---------------------------------------------------------------------------


def test_metrics_baselines_materialize_idempotently(tmp_path: Path) -> None:
    from research.core.event_lake import EventLake

    # hermetic: seed into a temp lake root
    first = materialize_metrics_baselines(lake_root=tmp_path / "lake")
    seeded = set(first["seeded"])
    # all five new §5.5 entries materialize...
    for model_id in (
        "double_bottom_xauusd_m15_v1",
        "double_top_xauusd_m15_v1",
        "falling_wedge_xauusd_m15_v1",
        "head_shoulders_xauusd_m15_v1",
        "inverse_head_shoulders_xauusd_m15_v1",
    ):
        assert model_id in seeded, f"missing metrics baseline: {model_id}"
        assert (tmp_path / "lake" / "metrics" / f"{model_id}.parquet").is_file()
    # ...and legacy liquidity_sweep ref basenames resolve too
    assert "liquidity_sweep_xauusd_h16_v2" in seeded
    assert "liquidity_sweep_eurusd_h16_v2" in seeded

    # §7.2 schema + baseline row sanity
    lake = EventLake(root=tmp_path / "lake")
    db_metrics = lake.read_metrics("double_bottom_xauusd_m15_v1")
    assert len(db_metrics) == 1
    assert set(db_metrics.columns) >= {
        "asof", "pf", "win_rate", "n_trades", "max_psi", "psi_retrain",
        "lifecycle_state",
    }
    assert db_metrics.iloc[0]["lifecycle_state"] == "validated"
    assert bool(db_metrics.iloc[0]["psi_retrain"]) is False

    # append-only by asof: a second run must skip, not raise
    second = materialize_metrics_baselines(lake_root=tmp_path / "lake")
    assert all(v.startswith("skipped") for v in second["targets"].values())
    assert len(lake.read_metrics("double_bottom_xauusd_m15_v1")) == 1


def test_registry_metrics_refs_resolve_after_materialization() -> None:
    # idempotent against the real lake (verify order: smoke ran first, but the
    # helper is safe to run here as well)
    materialize_metrics_baselines()
    missing = check_metrics_refs_resolve()
    assert missing == [], f"unresolved live_metrics_refs: {missing}"
    from live.state.shared_app_state_v2 import ModelRegistry

    reg = ModelRegistry.get_instance()
    reg.load_from_yaml()
    for info in reg.get_available_models():
        ref = str(getattr(info, "live_metrics_ref", "") or "")
        if ref:
            assert (ROOT / "event_lake" / ref.split("event_lake/")[-1]).is_file()


# ---------------------------------------------------------------------------
# 5. GUI offscreen boot + onboarding filter + assignment round-trip
# ---------------------------------------------------------------------------


def test_gui_offscreen_boot_onboarding_and_roundtrip() -> None:
    gui = gui_smoke()
    assert gui["booted"] is True
    assert gui["window_title"] == "Paper Trading V2 — Liquidity Sweep"
    # §10.1: assignable DB model offered; trained/explore FW excluded
    assert gui["double_bottom_assignable"] == ["double_bottom_xauusd_m15_v1"]
    assert gui["double_top_assignable"] == ["double_top_xauusd_m15_v1"]
    assert gui["falling_wedge_assignable"] == []
    assert gui["onboarding_tab_filter"] == ["double_bottom_xauusd_m15_v1"]
    assert gui["assignment_roundtrip_ok"] is True

    # workspace state is clean at rest (test-clean philosophy)
    assert not _ASSIGNMENT_FILE.exists()


def test_gui_restores_symbol_registry_bytes() -> None:
    blob = _REGISTRY_FILE.read_bytes() if _REGISTRY_FILE.is_file() else None
    gui_smoke()
    after = _REGISTRY_FILE.read_bytes() if _REGISTRY_FILE.is_file() else None
    assert after == blob


# ---------------------------------------------------------------------------
# 6. Static no-orders guard
# ---------------------------------------------------------------------------


def test_smoke_source_has_no_order_call_sites() -> None:
    source = (ROOT / "live" / "smoke" / "mcp_live_smoke.py").read_text(encoding="utf-8")
    for forbidden in (
        ".send_order(",
        ".place_order(",
        ".close_position(",
        ".market_order(",
        "order_send(",
        "trade_order(",
        ".close_order(",
    ):
        assert forbidden not in source, f"order call site leaked into smoke: {forbidden}"


def test_env_file_never_modified_by_smoke_gui() -> None:
    """Guard: the smoke never touches .env (mtime must not change)."""
    before = ENV_FILE.stat().st_mtime_ns
    gui_smoke()
    assert ENV_FILE.stat().st_mtime_ns == before


def test_contracts_py_untouched(tmp_path: Path) -> None:
    """Guard: research/core/contracts.py is frozen — the smoke cycle must not
    rewrite it (mtime before == mtime after running the shadow scan + GUI)."""
    from research.patterns.double_bottom.dataset import load_xauusd_m15

    contracts = ROOT / "research" / "core" / "contracts.py"
    assert contracts.is_file()
    before = contracts.stat().st_mtime_ns

    frame = load_xauusd_m15().tail(1500)
    run_shadow_scan(
        frame,
        lake_root=tmp_path / "lake",
        model_id="double_bottom_xauusd_m15_v1",
        schema_version="double-v1.0",
        horizon=72,
        label="contracts-guard",
    )
    gui_smoke()

    assert contracts.stat().st_mtime_ns == before