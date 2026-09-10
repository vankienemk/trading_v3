# PATTERN_SPECS — liquidity_sweep (P0 baseline)

> Per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 — every pattern ships this file.
> Plugin home: `trading_v3/research/patterns/liquidity_sweep/`
> Detector class: `LiquiditySweepDetector` (`detector.py`), implements
> `research.core.contracts.BasePatternDetector` (frozen v1.1, Agent 1).

## 1. Identity & registration

| Field | Value |
|---|---|
| `pattern_name` | `liquidity_sweep` |
| `pattern_version` | `2.0` (semantic version of the detector behavior, §6.2) |
| `short_name` (pattern_short) | `LSW` — registered in `contracts.PATTERN_SHORT_NAMES` and as the plugin class attribute `short_name`; the order-comment code is `LSW-v2-<event_id_tail>` (§9.2) |
| `feature_schema_version` | `lsw-2.0` (stamped on every `PatternEvent.config_hash`) |
| Registry entry | `detector.py` exposes `DETECTOR_CLASS`, `get_detector()`, `PATTERN_ENTRY` for `live/engine/pattern_registry.py` (§2.2 dynamic load) |

The plugin is the *refactor* of the legacy V2 sweep path — it composes the
same frozen research modules, in the same order, as the legacy
`signal_engine_v2._check_new_bar` (golden DoD, §13 Agent 2). The external
interface of `live/engine/signal_engine_v2.py` is unchanged.

## 2. Geometry definition

A **liquidity sweep** (V2, counter-trend) is a single candle that thrusts
through a liquidity level formed by prior rolling highs/lows, then closes
back inside the level while still printing a long lower/upper wick, in the
direction **against** the causal H1 trend:

* **Bullish (long)** — `low[t] < liq_low[t]`, `close[t] > liq_low[t]`,
  penetration `(liq_low - low)/ATR ∈ [0.05, 0.20]`, `lower_wick_ratio ≥ 0.35`,
  and H1 trend `+1` (a down-thrust into an up-trend = counter-trend sweep;
  the position follows the H1 trend, i.e. long).
* **Bearish (short)** — mirror on `liq_high` (`high[t] > liq_high[t]`,
  `close[t] < liq_high[t]`, upward penetration in ATR, `upper_wick_ratio ≥
  0.35`) with H1 trend `-1`.

Levels are **rolling liquidity levels**, not fractal swing pivots:

```
liq_low[t]  = min(low[ t-lookback : t ])     # lookback = 20 prior bars
liq_high[t] = max(high[ t-lookback : t ])
```

implemented as `shift(1).rolling(lookback).min()/max()` — the current candle
never forms the level it is measured against, and no centered window is used
(`src/liquidity/rolling_levels.py`, `src/events/sweep_detector.py`).

### Swing formula (§3.2 — pivot/right-bar rule)

The liquidity-sweep baseline uses **rolling extrema** as its swing reference
(more robust on M15 than a single left/right-bar fractal), so the
documented formula is:

| Element | Formula | Source price |
|---|---|---|
| Level formation | rolling `min(low)` / `max(high)` over the prior `level_lookback=20` bars, `shift(1)` (strictly causal) | `low` for bullish, `high` for bearish |
| Sweep bar | `t` closes beyond the level with the body closing back inside | `open/high/low/close` of bar `t` |
| Wick ratio | `lower_wick/(high-low) ≥ 0.35` (bullish), `upper_wick/(high-low) ≥ 0.35` (bearish) | `high`, `low` of bar `t` |
| Penetration | `(liq_low - low)/ATR(14) ∈ [0.05, 0.20]` / `(high - liq_high)/ATR(14) ∈ [0.05, 0.20]` | `low`/`high` of bar `t`, ATR(14) at `t` |
| Reclaim (v2) | `(close - liq_low)/ATR ≥ 0.0` / `(liq_high - close)/ATR ≥ 0.0` — no positive reclaim required in V2 | `close` of bar `t` |
| Trend filter | H1 trend from closed H1 candles: `ema_slope_sign(EMA50 of closed H1 closes, lag=1)`; bullish sweeps require `+1`, bearish require `-1` (`src/features/htf.py`) | closed H1 close series |

Minimum/maximum distance between pivots (§3.2): the rolling level requires a
full `level_lookback=20` window of *prior* bars to exist (first 20 bars are
NaN and can never sweep); the penetration cap `≤ 0.20 ATR` bounds how far a
bar may exceed the level (deep sweeps are rejected).

## 3. Detection pipeline (bit-identical legacy order)

The detector runs exactly the legacy `signal_engine_v2._check_new_bar`
order — detect → dedup → assign event_id → confirm → features → rule score →
entry/SL/TP:

1. `detect_sweeps_v2(df, atr_period, level_lookback, min/max_penetration_atr,
   min_wick_ratio, min_reclaim_atr, v2_nguoc_trend)` → per-bar sweep flags.
2. `_candidate_rows_v2` → one candidate row per sweeping (bar, direction).
3. `select_deduplicated_events(candidates, cooldown_bars=4, group_rule="first")`
   — run grouping (keep the FIRST bar of each consecutive same-level run) +
   per-direction cooldown (a new event on the same side requires
   `position - last_event > cooldown_bars`).
4. Stable sort by `(event_time, direction)`; assign `event_id =
   f"{symbol}-V2-{i:06d}"`; stamp `v2_target_r`.
5. `attach_confirmations(df, deduped, {}, max_wait_bars=3, min_body_ratio=0.60,
   min_range_atr=0.80, require_break_sweep_extreme=True)` → confirmation
   columns; keep only `is_confirmed == True`.
6. `build_liquidity_levels(df, {})` → level registry; `build_event_features`
   → the registered feature matrix (one row per event).
7. `compute_rule_scores(confirmed, {}, levels)` → `rule_score` (0–100).
8. Entry/SL/TP projection (see §5) with the legacy skip rule: an event whose
   entry bar would be beyond the frame end is dropped.

Every decision reads only bars **strictly before** the relevant decision
point (§3): `known_at = max(detect_time, confirm_time)`; no scoring feature
reads a bar at/after `known_at` (enforced by `validate_causality` + the
inherited no-lookahead CI gate).

## 4. Confirmation rule

A sweep is **confirmed** by the first candle `k` in
`[first_run_bar+1, first_run_bar+max_wait_bars]` (up to 3 M15 bars) that
satisfies the `close_break` type:

```
close beyond the causal sweep extreme:   close[k] > max(high[anchor..k-1])   (long)
                                          close[k] < min(low[anchor..k-1])    (short)
AND body_ratio[k] = |close-open|/(high-low) ≥ 0.60
AND range_atr[k]  = (high-low)/ATR(14)[k] ≥ 0.80
```

`anchor` is the first bar of the sweep run (recovered causally from the
per-bar sweep flags); the extreme is accumulated over bars before `k`.
`confirmation_time` = close time of candle `k`; unconfirmed candidates are
dropped (never emitted as events).

## 5. Entry / Stop / Target

| Field | Rule |
|---|---|
| Entry | `next_open_after_confirmation` — open of the candle after the confirmation candle: `entry_bar = conf_bar + 1`, `entry_price = open[entry_bar]` |
| Stop | `sweep_extreme ± buffer_atr·ATR(14)[anchor]`: long `stop = extreme_low - 0.10·ATR`, short `stop = extreme_high + 0.10·ATR`, where `extreme` = `min(low[anchor..anchor])` / `max(high[anchor..anchor])` (the sweep bar's opposite extreme) and `anchor` = first bar of the sweep run |
| Target | `entry_price ± target_r · (entry_price − stop_price)` with `target_r = 3.0` (frozen v2_target_r) |
| Causal skip | if `entry_bar ≥ len(df)` the event is dropped on that frame (legacy behavior) |

`PatternEvent.entry_time` = the entry bar's timestamp; `structure_levels`
carries `level_price` (swept rolling level) and `sweep_extreme`.

## 6. Staleness window

* Confirmation staleness: a sweep is stale (rejected) if no qualifying
  confirmation candle appears within `max_wait_bars = 3` bars after the
  sweep run's first bar.
* Cooldown: after an accepted event on one side, no new event on the same
  side is accepted for `cooldown_bars = 4` bars (dedup layer).
* The rolling level reference itself never goes stale — it always uses the
  last `level_lookback=20` prior bars; by construction every emitted event
  is "fresh" against the current level.

## 7. Default config + version (§6.2)

Default config is the frozen live pipeline parameter set
(`research/configs/symbols/XAUUSD.yaml` pipeline_overrides →
`v2_frozen.yaml`; legacy `_extract_pipeline_params` defaults). The config
MUST carry `"version"`; the resolved config hash
(`sha1(canonical_json(config)+version+indicators_version)[:12]`,
`research.core.config_hash`) is stamped on every event.

| Key | Default | Meaning |
|---|---|---|
| `version` | `2.0` | detector semantic version (change ⇒ bump) |
| `symbol` / `timeframe` | `XAUUSD` / `M15` | event identity |
| `atr_period` | `14` | ATR window |
| `level_lookback` | `20` | rolling level window (prior bars) |
| `min_penetration_atr` / `max_penetration_atr` | `0.05` / `0.20` | sweep penetration band |
| `min_wick_ratio` | `0.35` | wick share of bar range |
| `min_reclaim_atr` | `0.0` | no positive reclaim required (V2) |
| `v2_nguoc_trend` | `True` | counter-trend filter against H1 EMA50 slope |
| `cooldown_bars` | `4` | same-side cooldown |
| `group_rule` | `first` | run representative = first bar |
| `confirmation.max_wait_bars` | `3` | confirmation window |
| `confirmation.min_body_ratio` | `0.60` | body share of signal bar |
| `confirmation.min_range_atr` | `0.80` | signal bar range relative to ATR |
| `confirmation.require_break_sweep_extreme` | `True` | must trade beyond the sweep extreme |
| `target_r` | `3.0` | R:R target (frozen v2_target_r) |
| `buffer_atr` | `0.10` | stop buffer in ATR |

## 8. Features consumed (with `available_at`)

Declared schema on the plugin (`feature_schema`), runtime-validated by
`validate_causality` (§3.4). All features have `uses_future_data=False`; no
feature reads any bar at/after the event's `known_at`.

| Feature | dtype | `available_at` | Notes |
|---|---|---|---|
| `atr` | float | detect | ATR(14) at the sweep bar |
| `penetration_atr` | float | detect | sweep penetration in ATR |
| `wick_ratio` | float | detect | lower/upper wick ratio |
| `reclaim_atr` | float | detect | close-back distance in ATR (v2: ≥ 0.0) |
| `h1_trend` | int | detect | closed-H1 EMA50 slope sign (−1/0/1) |
| `level_price` | float | detect | swept rolling liquidity level |
| `confirmation_delay_bars` | float | confirm | bars sweep → confirmation candle |
| `confirmation_range_atr` | float | confirm | confirmation range / ATR |
| `confirmation_strength` | float | confirm | 0.5·min(body,1) + 0.5·min(range/2,1) |
| `rule_score` | float | confirm | weighted rule score 0–100 (levels/sweep/reclaim/confirmation/context/volume) |

The full registered feature matrix built by `build_event_features` is also
attached per event under `attributes["features"]` (Event Lake §7).

## 9. Expected event frequency (data audit)

Audited on the XAUUSD M15 dataset
(`data/processed/xauusd_m15.parquet`; 2022-06-02 → 2026-08-27 ≈ 4.2 years,
~99.7k M15 bars) with the frozen default config:

| Stage | Count | Frequency |
|---|---|---|
| Sweep candidate (bar, direction) rows | 495 | ≈ 5.0 / 1,000 bars |
| After dedup (cooldown 4, group first) | 462 | ≈ 4.6 / 1,000 bars |
| **Confirmed + scored events (tradable)** | **121** | **≈ 1.2 / 1,000 bars ≈ 29 / year ≈ one per ~13 calendar days** |

So the pattern yields roughly one tradable setup every ~13 calendar days
per symbol — a low-frequency, high-conviction baseline suitable as the
P0 regression gate for the multi-pattern engine. Counts are pinned exactly
by `tests/test_liquidity_sweep_golden.py::test_legacy_event_count_audit`
(current run: 121 events; plugin `config_hash=02ab91d2bb28`).

## 10. Golden DoD (Agent 2)

`tests/test_liquidity_sweep_golden.py` composes the legacy modules directly
in the exact `_check_new_bar` order on the same XAUUSD frame and asserts the
plugin's ``PatternEvent`` projection is **event-by-event equal**: identical
`event_id`s, timings, penetration/wick/reclaim, H1 trend, rule scores,
entry/stop/target and level metadata. The legacy reference is model-free
(the ML probability layer remains in the live engine); everything else is
bit-identical.