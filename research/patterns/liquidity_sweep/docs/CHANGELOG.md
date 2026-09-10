# CHANGELOG

All schema/interface lock changes (rule 30.1) are logged here.

## v1.1 — Phase 4 level registry extensions (t8, 2026-09-02)

- **`src/schema.py`**: added `LEVEL_EXTENSION_COLUMNS`,
  `LEVEL_REGISTRY_COLUMNS` (= `LEVEL_COLUMNS + LEVEL_EXTENSION_COLUMNS`),
  `LEVEL_STATE_COLUMNS` and `SWEEP_STATES` (schema lock 1.0 → 1.1).
  `LEVEL_TYPES` unchanged — H1 swings use `level_type="swing"` +
  `is_h1=True`; canonical `status` enum stays `active|expired`.
- **`docs/SCHEMAS.md`**: version banner bumped to 1.1; §1 Level ID documents
  the `h1_` prefix; §4 updated (`known_at` per level type, cumulative
  `touch_count` for all types superseding the v1.0 "swing/rolling: 1" note,
  `first/last_touch_time` for all types); new §4.1 (registry extension
  columns) and §4.2 (`level_state_at_bar` per-bar causal state interface).
- **`docs/INTERFACES.md` §3**: added `detect_h1_swing_levels`,
  `detect_previous_day_levels`, `level_state_at_bar`; `build_liquidity_levels`
  registry output documented as schema §4 + §4.1.
- **`src/liquidity/{swing_levels,equal_levels,level_registry}.py`**: modules
  now import `LEVEL_COLUMNS` / `LEVEL_REGISTRY_COLUMNS` / `LEVEL_STATE_COLUMNS`
  from `src.schema` (single implementation point). Behavioral change:
  `level_registry.detect_previous_day_levels` / `detect_h1_swing_levels` /
  `level_state_at_bar` are new public functions (Phase 4).
- **Tests**: `tests/test_levels.py` asserts the registry column set equals
  `LEVEL_REGISTRY_COLUMNS` and uses `LEVEL_STATE_COLUMNS` in the per-bar
  causal invariance test; `tests/test_contracts.py` gained a lock test for the
  new constants.