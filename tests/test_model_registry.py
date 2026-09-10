"""Model Registry tests — model_registry/index.yaml §5.5 schema migration (t3).

Covers the spec v1.1 §5.5 contract:
  1. Every registry entry carries the §5.5 fields
     (pattern_name, feature_schema_version, lifecycle_state, live_metrics_ref,
     gate_passed, calibrated, config_hash, trained_at).
  2. ALL artifact paths resolve inside trading_v3 (no ``trading_live/``
     references) — the registry is self-contained.
  3. ModelInfo/ModelRegistry parse the new fields; the §10.1 assignment
     filter (pattern_name + feature_schema_version + lifecycle_state
     ∈ {validated, shadow, live}) behaves correctly.
  4. t1 DB/DT models are registered validated with live_metrics_ref pointing
     at ``event_lake/metrics/...``; FW/HS/IHS explore models stay trained and
     are NOT assignable.
  5. Legacy xauusd_v2/eurusd_v2 entries are migrated with in-trading_v3
     artifacts (copied verbatim) — and the legacy back-compat model_id lookup
     (used by create_symbol_engine) still resolves to loadable artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from live.state.shared_app_state_v2 import ModelRegistry

# ---------------------------------------------------------------------------
# Consts
# ---------------------------------------------------------------------------

T3_ROOT = Path(__file__).resolve().parent.parent  # trading_v3/
INDEX = T3_ROOT / "model_registry" / "index.yaml"

REQUIRED_S55_FIELDS = [
    "pattern_name",
    "feature_schema_version",
    "lifecycle_state",
    "live_metrics_ref",
    "gate_passed",
    "calibrated",
    "config_hash",
    "trained_at",
]

ASSIGNABLE = {"validated", "shadow", "live"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _registry() -> ModelRegistry:
    """A freshly-(re)loaded registry singleton (idempotent across tests)."""
    reg = ModelRegistry.get_instance()
    reg.reload(str(INDEX))
    return reg


def _raw_index() -> dict[str, Any]:
    with open(INDEX, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict) and isinstance(data.get("models"), dict)
    return data


# ---------------------------------------------------------------------------
# §5.5 schema conformance
# ---------------------------------------------------------------------------

def test_every_entry_carries_full_s55_schema() -> None:
    """§5.5: every entry has the 8 required fields (values may be empty only
    when the legacy artifact predates the convention and is documented)."""
    for model_id, entry in _raw_index()["models"].items():
        missing = [k for k in REQUIRED_S55_FIELDS if k not in entry]
        assert not missing, f"{model_id} missing §5.5 fields: {missing}"


def test_no_trading_live_references_anywhere() -> None:
    """All paths resolve inside trading_v3 — no path field references
    ``trading_live/`` (documented migration notes may mention the source)."""
    raw = _raw_index()["models"]
    for model_id, entry in raw.items():
        for key in ("model_path", "calibrator_path", "feature_schema", "live_metrics_ref"):
            val = entry.get(key, "")
            assert "trading_live" not in val, (
                f"{model_id}.{key} must not reference trading_live/: {val}"
            )
            assert isinstance(val, str) and not val.startswith("/"), (
                f"{model_id}.{key} must be a relative path, got {val!r}"
            )
    # the canonical path fields exist on every entry
    for model_id, entry in raw.items():
        assert entry.get("model_path"), model_id
        assert entry.get("calibrator_path"), model_id
        assert entry.get("feature_schema"), model_id


def test_resolved_paths_exist_inside_trading_v3() -> None:
    """Every model/calibrator/feature-schema path resolves to a real file
    inside the trading_v3 root (self-contained registry)."""
    reg = _registry()
    root = str(T3_ROOT.resolve())
    for m in reg.get_available_models():
        for key in ("model_path", "calibrator_path", "feature_schema"):
            abs_path = m._resolved_paths[key]
            assert abs_path, f"{m.model_id}.{key} resolved empty"
            assert abs_path.startswith(root), (
                f"{m.model_id}.{key} resolves outside trading_v3: {abs_path}"
            )
            assert Path(abs_path).is_file(), (
                f"{m.model_id}.{key} artifact missing: {abs_path}"
            )


# ---------------------------------------------------------------------------
# ModelInfo parse round-trip (§5.5 fields)
# ---------------------------------------------------------------------------

def test_parse_roundtrip_modelinfo_s55_fields() -> None:
    reg = _registry()
    m = reg.get_model("double_bottom_xauusd_m15_v1")
    assert m is not None
    assert m.pattern_name == "double_bottom"
    assert m.feature_schema_version == "double-v1.0"
    assert m.lifecycle_state == "validated"
    assert m.gate_passed is True
    assert m.calibrated is True
    assert m.config_hash == "bb7129921f24"
    assert m.trained_at.startswith("2026-09-07T")
    assert m.symbol == "XAUUSD"
    assert m.timeframe == "M15"
    assert m.is_assignable() is True
    # back-compat legacy fields still parsed
    assert m.symbol_origin == "XAUUSD"
    assert m.horizon == 72
    assert "pr_auc" in m.metrics


def test_modelinfo_is_assignable_only_for_eligible_lifecycles() -> None:
    cases: list[tuple[str, bool]] = [
        ("double_bottom_xauusd_m15_v1", True),    # validated
        ("double_top_xauusd_m15_v1", True),       # validated
        ("xauusd_v2_h16_20260905", True),         # live (legacy, migrated)
        ("eurusd_v2_h16_20260906", True),         # live (legacy, migrated)
        ("falling_wedge_xauusd_m15_v1", False),   # trained
        ("head_shoulders_xauusd_m15_v1", False),  # trained
        ("inverse_head_shoulders_xauusd_m15_v1", False),  # trained
        ("xauusd_v2", False),                     # retired alias
    ]
    reg = _registry()
    for model_id, expected in cases:
        info = reg.get_model(model_id)
        assert info is not None, f"missing {model_id}"
        assert info.is_assignable() is expected, (
            f"{model_id} assignable={info.is_assignable()} != {expected}"
        )


# ---------------------------------------------------------------------------
# t1 registration (DB/DT validated + live_metrics_ref; FW/HS/IHS trained)
# ---------------------------------------------------------------------------

def test_db_dt_models_registered_validated_with_lake_metrics_ref() -> None:
    reg = _registry()
    for mid in ("double_bottom_xauusd_m15_v1", "double_top_xauusd_m15_v1"):
        m = reg.get_model(mid)
        assert m is not None
        assert m.lifecycle_state == "validated"
        assert m.gate_passed is True
        assert m.live_metrics_ref.startswith("event_lake/metrics/")
        assert m.live_metrics_ref.endswith(".parquet")


def test_fw_hs_ihs_explore_models_trained_not_assignable() -> None:
    reg = _registry()
    for mid in (
        "falling_wedge_xauusd_m15_v1",
        "head_shoulders_xauusd_m15_v1",
        "inverse_head_shoulders_xauusd_m15_v1",
    ):
        m = reg.get_model(mid)
        assert m is not None
        assert m.lifecycle_state == "trained"
        assert m.gate_passed is False
        assert m.calibrated is True
        assert not m.is_assignable()


# ---------------------------------------------------------------------------
# Legacy migration + back-compat
# ---------------------------------------------------------------------------

def test_legacy_entries_migrated_with_in_trading_v3_artifacts() -> None:
    reg = _registry()
    root = str(T3_ROOT.resolve())
    for mid in ("xauusd_v2_h16_20260905", "eurusd_v2_h16_20260906"):
        m = reg.get_model(mid)
        assert m is not None
        assert m.pattern_name == "liquidity_sweep"
        assert m.lifecycle_state == "live"  # legacy running model (spec §13 row 4)
        for key in ("model_path", "calibrator_path", "feature_schema"):
            abs_path = m._resolved_paths[key]
            assert abs_path.startswith(root), key
            assert Path(abs_path).is_file(), key


def test_legacy_bare_alias_xauusd_v2_retired_and_documented() -> None:
    raw = _raw_index()["models"]
    entry = raw.get("xauusd_v2")
    assert entry is not None, "persisted symbol_registry.json references xauusd_v2"
    assert entry.get("lifecycle_state") == "retired"
    assert "RETIRED" in str(entry.get("notes", "")).upper()
    reg = _registry()
    alias = reg.get_model("xauusd_v2")
    assert alias is not None  # back-compat get_model must still resolve
    assert not alias.is_assignable()


def test_legacy_model_ids_still_resolvable_for_engine_load_path() -> None:
    """Back-compat: create_symbol_engine's registry lookup (model_id ->
    resolved path -> joblib load) keeps working for legacy ids."""
    reg = _registry()
    for mid in ("xauusd_v2_h16_20260905", "eurusd_v2_h16_20260906", "xauusd_v2"):
        info = reg.get_model(mid)
        assert info is not None, mid
        model_path = info._resolved_paths["model_path"]
        calibrator_path = info._resolved_paths["calibrator_path"]
        assert Path(model_path).is_file() and Path(calibrator_path).is_file(), mid

        import joblib

        raw = joblib.load(model_path)
        model = raw["model"] if isinstance(raw, dict) else raw
        calibrator = joblib.load(calibrator_path)
        # the engine asserts an .estimator on the wrapper; calibrator must be a
        # fitted sklearn-ish calibrator
        assert getattr(model, "estimator", None) is not None, mid
        assert hasattr(calibrator, "predict_proba"), mid


# ---------------------------------------------------------------------------
# §10.1 assignment filter
# ---------------------------------------------------------------------------

def test_get_models_for_assignment_filters_by_pattern_and_lifecycle() -> None:
    reg = _registry()

    db_models = reg.get_models_for_assignment(pattern_name="double_bottom")
    assert [m.model_id for m in db_models] == ["double_bottom_xauusd_m15_v1"]
    # trained/retired models never appear for ANY pattern filter
    for m in db_models:
        assert m.lifecycle_state in ASSIGNABLE

    dt_models = reg.get_models_for_assignment(pattern_name="double_top")
    assert [m.model_id for m in dt_models] == ["double_top_xauusd_m15_v1"]

    # explore-only + retired models are excluded from EVERY pattern filter
    for mid in (
        "falling_wedge_xauusd_m15_v1",
        "head_shoulders_xauusd_m15_v1",
        "inverse_head_shoulders_xauusd_m15_v1",
        "xauusd_v2",
    ):
        info = reg.get_model(mid)
        assert info is not None
        pat_ids = [m.model_id for m in reg.get_models_for_assignment(
            pattern_name=info.pattern_name)]
        assert mid not in pat_ids, f"{mid} leaked into assignment filter: {pat_ids}"


def test_get_models_for_assignment_schema_and_symbol_filters() -> None:
    reg = _registry()
    # feature_schema_version filter
    only_db_v1 = reg.get_models_for_assignment(
        pattern_name="double_bottom", feature_schema_version="double-v1.0",
    )
    assert [m.model_id for m in only_db_v1] == ["double_bottom_xauusd_m15_v1"]
    none_wrong_schema = reg.get_models_for_assignment(
        pattern_name="double_bottom", feature_schema_version="nope",
    )
    assert none_wrong_schema == []

    # symbol filter
    xau_lsw = reg.get_models_for_assignment(pattern_name="liquidity_sweep", symbol="XAUUSD")
    assert {m.model_id for m in xau_lsw} == {"xauusd_v2_h16_20260905"}
    eur_lsw = reg.get_models_for_assignment(pattern_name="liquidity_sweep", symbol="EURUSD")
    assert {m.model_id for m in eur_lsw} == {"eurusd_v2_h16_20260906"}

    # explicit lifecycle override
    only_live = reg.get_models_for_assignment(
        pattern_name="liquidity_sweep", lifecycle_states={"live"},
    )
    assert {m.model_id for m in only_live} == {
        "xauusd_v2_h16_20260905", "eurusd_v2_h16_20260906",
    }


def test_get_available_models_back_compat_returns_all() -> None:
    reg = _registry()
    ids = {m.model_id for m in reg.get_available_models()}
    assert ids == {
        "double_bottom_xauusd_m15_v1",
        "double_top_xauusd_m15_v1",
        "falling_wedge_xauusd_m15_v1",
        "head_shoulders_xauusd_m15_v1",
        "inverse_head_shoulders_xauusd_m15_v1",
        "xauusd_v2_h16_20260905",
        "eurusd_v2_h16_20260906",
        "xauusd_v2",
    }


def test_get_model_missing_returns_none() -> None:
    reg = _registry()
    assert reg.get_model("does_not_exist") is None