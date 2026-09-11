# Trend/HMM Rework — Measurement & Decision Document

**Task:** t6 (verification) — mandatory measurement per rework request §4 ("đo trước, bật sau")
**Author:** measurement_analyst
**Date:** 2026-09-11
**Request under test:** `rework/trading_v3_rework_request_trend_hmm.md`

---

## 0. Bottom line

| Gate | Recommendation | Confidence |
|---|---|---|
| **§2.1 trend-context gate (Phương án B, slope+R²)** | **OFF** | high — no selection power; the *sole* cause of 100% of the measured false-drop cost; fails §4.2 |
| **§2.1 trend-context gate (Phương án C, HMM trend direction)** | **OFF — status pending/not-live-wired** | high — no forward predictive edge; §2.4's rule *inverted* for `double_top`; regime is inert live, so no live result exists |
| **§2.2 `max_pattern_length_bars`** | **OFF** (keep as inert defence-in-depth at the shipped 30) | very high — measured no-op; removes 0 of 664 events in 8.3 years |
| **§2.3 `model_prob` AND-gate** | **ON for `double_bottom` only, at ≤ 0.40; OFF for `double_top`** | medium — DB gain is OOS-validated but on only 24 OOS events; DT gain is overfit |
| **§2.3 `rule_score` floor** | **OFF** (floor applies to the NORMALISED score — see §2.3) | high — no measurable effect at any tested floor |
| **§3 (`stop_mode=neckline`, 4 ATR target cap)** | **NOT re-applied** | — request explicitly forbids; not touched by this measurement |

The request's two headline premises — "range 55+ nến vẫn được chấp nhận" (§1.2) and "trend context là gate cần thiết" (§1.1) — are **both contradicted by the full-history measurement**. The one change that measurably works is the *third* one the request ranked lowest in effort (§2.3 model_prob), and it works for only one of the two patterns. §2.1 — the request's top recommendation — is the only gate that destroys measurable value (66 DB and 52 DT winning events, +119R combined) while improving nothing.

---

## 1. How to reproduce every number

```bash
cd "/Users/a/Documents/deepseek harness/kien-workspace/trading_v3"
PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages \
    /tmp/ptv2_venv/bin/python docs/rework_trend_hmm_measure.py     # main study
PYTHONPATH=/tmp/pytest_fix:/tmp/ptv2_venv/lib/python3.9/site-packages \
    /tmp/ptv2_venv/bin/python docs/trend_hmm_option_c.py           # Phương án C (HMM)
```

Outputs: `docs/rework_trend_hmm_measure.json`, `docs/trend_hmm_option_c.json`.

Both scripts are `ruff check`-clean (`select = ["E","F","I","UP","B","RUF","PERF"]`, matching the sibling `docs/rework_measure.py`), and re-running them after the lint fixes reproduced the JSON **byte-identically**, so every number below is from the current revision of the scripts.

**Baseline reference — all new gates OFF** (cite these; the arms below are all relative to them):

| pattern | n_events | n_trades | winrate | expectancy R | total R | target ATR med | risk ATR med |
|---|---|---|---|---|---|---|---|
| double_bottom | 354 | 354 | 0.4802 | **−0.0099** | −3.50 | 6.839 | 4.560 |
| double_top | 310 | 310 | 0.4065 | **−0.1662** | −51.53 | 6.672 | 4.448 |

Exit reasons on the current tree — DB: stop 131 / target 82 / horizon 141; DT: stop 146 / horizon 89 / target 75. Note these differ by a few events from the older `docs/rework_measure_baseline.json` (DB 360 / DT 316): NMS + `drop_opposite_overlap` landed since, shrinking the pool to 354 / 310. That difference is expected; for any "all gates off" statement use the table above, not the old JSON.

**Environment quirk (as flagged in the brief, and worse than described):** pytest in this venv is not merely "corrupted" — `tests/test_gui_onboarding_registry.py` **aborts the interpreter with SIGABRT (exit 134) producing zero output** during collection. This is pre-existing and unrelated to the rework (the file is from the initial commit, `19a5295`, unmodified). Every full-suite invocation must add `--ignore=tests/test_gui_onboarding_registry.py`, or the suite silently dies with no test results at all. Verified in §5.

Data: full XAUUSD M15 history, `load_symbol_frame('XAUUSD', start='2018-06-01', end='2026-09-03')` → **195,893 bars**, 2018-05-09 → 2026-09-03 (8.32 years).

**Bar-count reconciliation** (the brief calls this "the full 204k-bar history" — both figures are correct about different things, and the JSON records this too):

| quantity | rows | range |
|---|---|---|
| raw parquet `research/multi_backtest/data/XAUUSD_m15.parquet` | **204,133** | 2018-01-02 09:00 → 2026-09-03 22:45 |
| `load_symbol_frame('XAUUSD')` (no window) | **204,133** | identical to raw — proves no rows are dropped/malformed |
| **windowed frame used here** (`start='2018-06-01'`) | **195,893** | 2018-05-09 → 2026-09-03 18:45 |

The 8,240-row delta is purely the dropped pre-2018-06-01 prefix. The `start='2018-06-01'`/`end='2026-09-03'` window is the same call the existing `docs/rework_measure.py` uses, which is why the brief labels it the 204k history.

**Method — why this is measurable before the detector gates land.** Every arm detects **once** with all new gates disabled and then applies the gate *post-hoc* to the identical candidate pool. All arms therefore differ only in the gate decision, never in the pool, which is strictly better controlled than re-running `detect()` per arm. `model_prob` is produced through the **live inference contract** (canonical causal `build_feature_frame` → per-pattern `calibrator.pkl` → `predict_proba[:, 1]`), and the §2.3 arms call the **shipped** `live/engine/hard_gate.probability_gate` directly, not a replica.

---

## 2. Gate-by-gate findings

### 2.1 §2.2 `max_pattern_length_bars` — measured NO-OP ⇒ **OFF**

`pattern_length_bars = extreme2_bar − extreme1_bar`, full pool:

> **Computation label.** These figures are derived as `extreme2_bar − extreme1_bar`, **not** read from an event attribute. At the time of measurement `pattern_length` was **declared** in `feature_schema` (`research/patterns/double_bottom/detector.py:100-103`, `AVAILABLE_AT_DETECT`) but **never computed and never written to `attributes`** — t1 filed this as `F-01`, a blocker for a naive §2.2 gate. The derivation above is unaffected (it is the same quantity the gate needs), and gate_engineer has since fixed `F-01` by computing `pattern_length = i3 - i1` once in `_detect_double` and emitting it. **The defect was real and is worth recording**: a `feature_schema` entry advertising `AVAILABLE_AT_DETECT` for a key that does not exist is a latent trap for any consumer, and it is the reason the request's §1.2 wording ("đã được tính và emit như một feature") reads as true while being false in the code as it stood.

| pattern | n | min | median | p90 | **max** |
|---|---|---|---|---|---|
| double_bottom | 354 | 8 | 14 | 20 | **29** |
| double_top | 310 | 8 | 14 | 19 | **26** |

Arms at 26 / 27 / 28 / 29 / 30 / 40 / 50 / 60 bars, all on the same pooled detector output with every other new gate off:

| max_len | double_bottom kept | dropped | DB win / exp R | double_top kept | dropped | DT win / exp R |
|---|---|---|---|---|---|---|
| 26 | 352 | 2 | 0.4830 / −0.0067 | 310 | 0 | 0.4065 / −0.1662 |
| 27 | 353 | 1 | 0.4816 / −0.0096 | 310 | 0 | 0.4065 / −0.1662 |
| 28 | 353 | 1 | 0.4816 / −0.0096 | 310 | 0 | 0.4065 / −0.1662 |
| 29 | **354** | **0** | 0.4802 / −0.0099 | 310 | 0 | 0.4065 / −0.1662 |
| 30 | **354** | **0** | 0.4802 / −0.0099 | **310** | **0** | 0.4065 / −0.1662 |
| 40 | 354 | 0 | 0.4802 / −0.0099 | 310 | 0 | 0.4065 / −0.1662 |
| 50 / 60 | 354 | 0 | identical | 310 | 0 | identical |

The tightest 0-drop bound is **29 for double_bottom** (its own observed max) and **26 for double_top**. A single shared default of **30** is 0-drop for both and is the recommended value: it is the smallest bound that is a no-op across the union of the two patterns. Setting 26 or 27 would start deleting DB events (2 and 1 respectively) for no measured benefit.

**The request's premise is false.** §1.2 asserts "range 55+ nến vẫn được chấp nhận như một double-top/double-bottom hợp lệ". No event in 8.3 years exceeds even 30 bars. The existing `min_separation_bars` + `max_bars_between_detect_and_confirm` (60) + `min_separation_bars` (4) chain plus NMS already caps the span. A gate at the proposed 40–60 therefore **cannot** fix the reported symptom, because the events the request wants removed do not exist in the detector's output.

This does not mean the config key is worthless — it is cheap, correct and would catch a future regression — but it must **not** be described as an improvement, and it must not be enabled as if it were one. Enabling it changes nothing, so "OFF" and "ON-at-40" are behaviourally identical; keeping it OFF avoids implying a benefit that was not measured. If the team wants the gate to bite, the threshold would have to be ~20 bars, which is an *unmeasured and much more destructive* change — I did not measure it as a recommendation.

### 2.2 §2.3 `model_prob` AND-gate — the only gate that works, and only for DB

Measured by calling the shipped `probability_gate` on the real pool:

**double_bottom (pool 354; in-sample 203 / OOS 151):**

| threshold | kept | winrate | expectancy R (all) | **OOS kept** | **OOS expectancy R** |
|---|---|---|---|---|---|
| (baseline) | 354 | 0.4802 | −0.0099 | 151 | +0.1346 |
| 0.30 | 138 | 0.5580 | +0.1944 | 46 | +0.1819 |
| **0.40** | **84** | **0.5952** | **+0.3150** | **24** | **+0.3865** |
| 0.50 | 56 | 0.6429 | +0.4342 | 11 | +0.7051 |
| 0.55 (proposed) | 50 | 0.6400 | +0.4213 | 7 | +0.7718 |

**double_top (pool 310; in-sample 212 / OOS 98):**

| threshold | kept | winrate | expectancy R (all) | **OOS kept** | **OOS expectancy R** |
|---|---|---|---|---|---|
| (baseline) | 310 | 0.4065 | −0.1662 | 98 | −0.1636 |
| 0.30 | 124 | 0.4516 | +0.0358 | 33 | −0.0762 |
| 0.40 | 77 | 0.5325 | +0.2472 | 16 | **−0.0296** |
| 0.50 | 48 | 0.5833 | +0.3716 | 9 | −0.1329 |
| 0.55 (proposed) | 39 | 0.5641 | +0.3309 | 9 | −0.1329 |

**Reading this honestly:**
- The gate is **real and directionally correct** — it is not noise. The mechanism the request describes in §1.3 is genuine: the model "knows" an event is weak and the average hides it. Sorting the pool by `model_prob` splits outcomes cleanly (below).
- **`double_bottom`: the gain survives OOS validation.** At 0.40, OOS expectancy goes +0.1346R → +0.3865R with winrate 0.5629 → 0.6250. That is the only gate in this study that improves OOS performance.
- **`double_top`: the gain does NOT survive.** Every threshold turns *negative* out-of-sample (0.40 → −0.0296R on n=16; 0.55 → −0.1329R on n=9). The in-sample +0.2472R is overfit. Enabling the DT prob gate would be an in-sample-artifact decision.
- **Sample size is the binding constraint.** Even the best arm keeps only 24 OOS events. The request's §4.2 requirement of ≥ 100 OOS events is **unreachable** for a `model_prob` gate: at a threshold low enough to keep 100 events, the measured edge is nearly gone, and at a threshold that shows an edge, only ~7–24 events survive.

**Keep-rates (fraction of pool with `model_prob ≥ thr`)** — the request §6 warning was well-founded, and the proposed numbers are badly placed:

| threshold | double_bottom | double_top |
|---|---|---|
| 0.40 | 23.7% | 24.8% |
| 0.45 | 19.8% | 19.7% |
| 0.50 | 15.8% | 15.5% |
| **0.55 (B1 proposal)** | **14.1%** | **12.6%** |
| 0.60 | 12.7% | 9.0% |

The B1 values (0.55 for both) would keep ~50 DB and ~39 DT events **over 8.3 years** (~6 and ~5 events/year) — too few to trade and structurally incompatible with the §4.2 OOS gate. They should not be shipped.

**Distribution (the data the request §2.3 thresholds should have been derived from):**

| pattern | n | min | p10 | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| double_bottom | 354 | 0.0694 | 0.0694 | 0.1604 | **0.2163** | 0.3880 | 0.6651 | 1.0000 | 0.3049 |
| double_top | 310 | 0.0333 | 0.1091 | 0.1667 | **0.2565** | 0.3976 | 0.5661 | 0.8519 | 0.2999 |

Both models are **conservative** (median ≈ 0.22–0.26) — a threshold near the *population median*, not near 0.5, is the scale-appropriate place to start. A threshold of 0.55 is not "0.05 above the midpoint of the output range", it is the ~86th percentile of the model's own output.

**Selection power (the reason the axis is worth keeping at all):**

| pattern | bottom half | top half |
|---|---|---|
| double_bottom | n=177, win 0.4350, exp **−0.1273R** | n=177, win 0.5254, exp **+0.1076R** |
| double_top | n=155, win 0.3806, exp **−0.3134R** | n=155, win 0.4323, exp **−0.0190R** |

### 2.3 §2.3 `rule_score` floor — measured NO EFFECT ⇒ **OFF**

> **Scale warning — the floor applies to the NORMALISED `[0,1]` score, never the raw value.** `rule_score` does **not** have one scale in this repo: `liquidity_sweep` normalizes to **0..100** (`research/patterns/liquidity_sweep/src/scoring/rule_score.py:420-428`: `… * (100 / total_weight)`, then `.clip(0, 100)`), while DB/DT and the wedge/H&S plugins clamp to **[0,1]**. A `0.6` floor applied to a RAW LSW score of `51.0` would evaluate `51.0 < 0.6 → False` and pass trivially; a floor applied to a raw vs normalized DB score means the same thing only by coincidence. Both the shipped live gate (`live/engine/hard_gate.normalize_rule_score`, with `RULE_SCORE_SCALE = {"liquidity_sweep": 100.0}`) and `signal_engine_v2._normalize_rule_score` normalize first — so the table below is on the normalized scale and is **not** transferable to raw values. Every number in this table is from DB/DT, which are already `[0,1]`, so no normalization changes them here.

| floor | DB kept | DB win | DB exp R | DT kept | DT win | DT exp R |
|---|---|---|---|---|---|---|
| 0.50 | 354 (100%) | 0.4802 | −0.0099 | 310 (100%) | 0.4065 | −0.1662 |
| 0.60 (proposed) | 338 (95.5%) | 0.4882 | −0.0045 | 300 (96.8%) | 0.4100 | −0.1640 |
| 0.65 | 313 (88.4%) | 0.4920 | +0.0037 | 278 (89.7%) | 0.4065 | −0.1631 |
| 0.70 | 268 (75.7%) | 0.4888 | −0.0078 | 234 (75.5%) | 0.4316 | −0.0990 |

`rule_score` distribution is narrow and high (DB min 0.5165 / median 0.7665; DT min 0.5078 / median 0.7746), so a floor below ~0.5 removes nothing, and every effect above it is inside noise (DB moves −0.0099 → −0.0045 at 0.6; DT is essentially flat and negative). Note also that the cited problem event has `rule_score = 0.79` (§7) — **a rule_score floor cannot catch the very event the request uses to motivate §2.3**, because that event's geometry score is high by construction. This confirms the request's own §1.3 diagnosis ("công thức score chỉ đo depth/symmetry/offset, không đo trend context") while undermining the proposed remedy: raising a floor on a score that does not measure the failure mode does not remove the failures.

### 2.4 §2.1 trend-context gate, Phương án B (slope + R²) ⇒ **OFF**

`min_slope_atr = 0.05`; request default is `lookback=30, min_r2=0.3`.

**double_bottom (baseline win 0.4802 / exp −0.0099R):**

| lookback / min_r2 | kept | kept% | winrate | expectancy R |
|---|---|---|---|---|
| 20 / 0.2 | 143 | 40.4% | 0.4965 | +0.0069 |
| 20 / 0.3 | 132 | 37.3% | 0.4924 | +0.0012 |
| **30 / 0.3 (default)** | **118** | **33.3%** | **0.4746** | **−0.0101** |
| 30 / 0.5 | 88 | 24.9% | 0.4205 | −0.1203 |
| 40 / 0.3 | 103 | 29.1% | 0.4466 | −0.0111 |
| 60 / 0.3 | 96 | 27.1% | 0.5208 | +0.1187 |

**double_top (baseline win 0.4065 / exp −0.1662R):**

| lookback / min_r2 | kept | kept% | winrate | expectancy R |
|---|---|---|---|---|
| 20 / 0.2 | 113 | 36.5% | 0.3628 | −0.2412 |
| 20 / 0.3 | 101 | 32.6% | 0.3861 | −0.1861 |
| **30 / 0.3 (default)** | **105** | **33.9%** | **0.4095** | **−0.1599** |
| 30 / 0.5 | 74 | 23.9% | 0.3784 | −0.1821 |
| 40 / 0.3 | 96 | 31.0% | 0.3958 | −0.1967 |
| 60 / 0.3 | 88 | 28.4% | 0.4545 | −0.0492 |

**The gate has no selection power.** At the request's default, expectancy is statistically unchanged from baseline (DB −0.0101 vs −0.0099; DT −0.1599 vs −0.1662) while two-thirds of events are deleted. The decisive evidence is what the gate *removes*:

| pattern | kept | kept win / exp | dropped | dropped win / exp |
|---|---|---|---|---|
| double_bottom | 113 | 0.4779 / −0.0131R | 241 | 0.4813 / **−0.0084R** |
| double_top | 102 | 0.4020 / −0.1798R | 208 | 0.4087 / **−0.1596R** |

**The events the gate deletes perform the same as the ones it keeps** — in the DB case the dropped set is (insignificantly) *better*. That is the signature of a filter removing signal and noise at the same rate.

**Corroborated independently (two separate measurements agree).** The captain re-measured kept-vs-dropped from the same gate definition and reached the same conclusion:

| pattern | measurement_analyst kept / dropped | captain kept / dropped |
|---|---|---|
| double_bottom | −0.0131R / **−0.0084R** | −0.0101R / **−0.0097R** |
| double_top | −0.1798R / **−0.1596R** | −0.1599R / **−0.1695R** |

The absolute expectancy levels differ slightly between the two runs (cumulative-arm definition vs §2.1-alone definition), but **both agree on the finding that matters: the dropped set is not worse than the kept set — on the DB side it is marginally better in both runs.** Two independent derivations therefore agree that the §2.1 gate separates nothing. This is the headline result of the study.

The best-looking cells are almost certainly noise: the `(lookback, min_r2)` grid is 6 cells and the "winner" (DB lb60/r2 0.3, +0.1187R on n=96) is a post-hoc pick — precisely the overfit failure mode §2.1 warned about, and there is no reason from the data to prefer lookback 60 over 30.

**False drops (the cost side, §2.1 of the task).** With the gates cumulative, 66 DB and 52 DT dropped events have `rule_score ≥ 0.75` **and** a winning outcome — high-geometry-quality patterns with a profitable trade. Examples, all dropped by the trend gate alone:

| pattern | known_at | rule_score | model_prob | net R | dropped by |
|---|---|---|---|---|---|
| double_bottom | 2026-04-07 | 0.8448 | 0.1614 | +1.476 | 2.1_trend |
| double_bottom | 2026-08-05 | 0.9077 | 0.1604 | +1.468 | 2.1_trend |
| double_bottom | 2025-10-10 | 0.8304 | 0.1751 | +1.461 | 2.1_trend |
| double_bottom | 2025-11-12 | 0.9456 | 0.1930 | +1.453 | 2.1_trend |
| double_top | 2026-01-29 | 0.8639 | 0.1942 | +1.486 | 2.1_trend |
| double_top | 2026-06-29 | 0.8794 | 0.3391 | +1.463 | 2.1_trend |

Cumulative effect of all three gates: DB 354 → 113 events, DT 310 → 102.

### 2.5 §2.1 Phương án C (HMM trend direction) ⇒ **OFF** — offline-only, **not live-wired**

Measured with `research/regime/trend_hmm.py` (971 lines, produced by hmm_trend_engineer). **The module IS available and WAS measured** — the task brief anticipated it might be pending; the offline measurement is not.

> **Scope caveat (input from t1): the HMM regime is INERT in the live system, so option C is not a live gate.** Verified directly on the current tree: `research/configs/symbols/XAUUSD.yaml` (the live symbol dir, `SYMBOL_CONFIG_DIR` in `live/state/shared_app_state_v2.py:39`) contains **no `regime`/`hmm`/`plugin` keys at all**, and `live/engine/signal_polling_engine_v2.py` **never constructs `MultiPatternEngine`** (zero matches). The HMM therefore does not run in the live path today. Consequences for this comparison:
> - **The B-vs-C table below is NOT a like-for-like production comparison.** B is measured offline too (the detector gate is opt-in and off), but B is at least wired where the engine runs; C is not reachable live.
> - **All C numbers here are OFFLINE-ONLY** — a full-history backtest of the HMM state as a filter. They say nothing about live behaviour, and they cannot be promoted to a live gate without the wiring work in `signal_polling_engine_v2` / the symbol config.
> - Treat C's result as "**pending / not live-wired**" for any decision about live defaults, and as "measured offline" for the question of whether the signal has any edge (it does not — see below).

**Mechanics are sound.** It imports, fits, and forward-filters correctly. Per the repo's existing regime convention (from `research/multi_backtest/scripts/oos_hmm_regime_filter.py`) the HMM is fit on the **2018-01-02 → 2023-09-30** prefix only and then strict-causal `predict` runs over the full frame. The state-identity pinning works: fitted mean drift is correctly ordered (downtrend −0.6319, range −0.0778, uptrend +0.7169 on the `mean_return_20_norm` column), so a refit did not silently invert the gate's direction. Bar shares are balanced (downtrend 32.9% / range 33.9% / uptrend 33.2%), mean confidence 0.963.

**But the states have no forward predictive edge.** Mean *forward* return by state — the only property that makes a state useful as a gate:

| horizon | downtrend | range | uptrend |
|---|---|---|---|
| 24 bars | +7.04e−6 | +6.11e−6 | +7.48e−6 |
| 96 bars | +7.16e−6 | +3.76e−6 | +9.59e−6 |
| 288 bars | +6.74e−6 | +5.32e−6 | +8.29e−6 |

With n ≈ 65,000 bars per state, 95% CIs are ≈ ±7e−7 — so the ordering is real, but the spread (~1–4e−6) is 1–2 orders of magnitude below the symbol's unconditional drift. The filter labels the **past** accurately by construction; it does not indicate the **next** leg.

**Applied as the §2.4 gate, results are mixed and partly INVERTED:**

| pattern | gate rule | kept | kept win / exp | dropped win / exp |
|---|---|---|---|---|
| double_bottom | keep iff `downtrend` | 150 (42.4%) | 0.5000 / +0.0124R | 0.4657 / −0.0262R |
| double_top | keep iff `uptrend` | 129 (41.6%) | 0.3721 / **−0.2199R** | 0.4309 / **−0.1280R** |

Per-state breakdown (the actionable part):

| state | double_bottom | double_top |
|---|---|---|
| downtrend | n=150 win 0.5000 exp **+0.0124R** | n=80 win 0.4500 exp **−0.0807R** |
| range | n=129 win 0.4729 exp **+0.0160R** | n=101 win 0.4158 exp −0.1655R |
| uptrend | n=75 win 0.4533 exp **−0.0988R** | n=129 win 0.3721 exp **−0.2199R** |

Two consequences:

1. **For `double_bottom`, "downtrend" is the wrong rule** — `range` performs equally well (+0.0160R vs +0.0124R); only `uptrend` is clearly bad (−0.0988R). The useful rule is "**not uptrend**", not "**== downtrend**". If §2.4 step 4 ships as written it discards good `range`-context events for no reason.
2. **For `double_top`, the §2.4 rule is inverted by the data.** Keeping only `uptrend` selects the *worst* subset (−0.2199R) while `downtrend` is the best (−0.0807R). §2.4 step 4 ("double_bottom chỉ pass nếu `trend_regime_state == "downtrend"` (và ngược lại cho double_top)") would make `double_top` **measurably worse** if implemented literally. This is a correctness risk in the request, not in the plugin.

**B vs C comparison (both measured OFFLINE — C is not live-wired, see the caveat above):**

| | kept% | kept exp R | dropped exp R | separation | OOS fixed / last-40% | live status |
|---|---|---|---|---|---|---|
| **B** DB | 33.3% | −0.0101 | −0.0084 | **negative** | 43 / 47 → FAIL | opt-in, off |
| **C** DB | 42.4% | +0.0124 | −0.0262 | positive | 50 / 60 → FAIL | **not wired** |
| **B** DT | 33.9% | −0.1599 | −0.1596 | ~zero | 38 / 42 → FAIL | opt-in, off |
| **C** DT | 41.6% | −0.2199 | −0.1280 | **inverted** | 41 / 52 → FAIL | **not wired** |

Phương án C separates better than B for `double_bottom` (and keeps more events doing it), but it is *worse* than B for `double_top` because of the sign inversion. **Neither option earns a default-ON**, and both fail §4.2. Because C is not live-wired, its column should be read as "offline edge evidence", not as a shippable alternative to B; the honest status for a live decision is **pending**.

### 2.5b False-drop cost — the finding that decides §2.1 (prioritised)

Because §2.1 has no selection power (§2.4), the **cost side is the whole case against it**. Measured uncapped over the full pool. A "false drop" is defined as an event that the cumulative gates remove **and** that has `rule_score ≥ 0.75` (the geometry score the detector trusts) **and** a realised winning trade (`net_r > 0`) — i.e. a structurally good pattern with a profitable outcome that the gates discard anyway.

| pattern | false drops | as % of all dropped | R foregone | mean per false drop | **responsible gate** |
|---|---|---|---|---|---|
| double_bottom | **66** | 66/241 = **27.4%** | **+66.74R** | +1.011R | **100% §2.1 trend** |
| double_top | **52** | 52/208 = **25.0%** | **+52.42R** | +1.008R | **100% §2.1 trend** |

**Every single false drop in both patterns is caused by the §2.1 trend gate alone. Neither §2.2 nor §2.3 causes one.** That is the sharpest possible statement of where the cost sits: the gates the request ranked as "cheap and safe" (§2.2, §2.3) destroy nothing of value, and the gate the request recommends most strongly (§2.1, Phương án B/C) destroys ~a quarter of all dropped events, worth ~+1R each.

Worked examples (all dropped by §2.1 alone, all winning trades):

| pattern | known_at | rule_score | model_prob | span | net R |
|---|---|---|---|---|---|
| double_bottom | 2026-04-07 | 0.8448 | 0.1614 | 14 | +1.476 |
| double_bottom | 2026-08-05 | 0.9077 | 0.1604 | 17 | +1.468 |
| double_bottom | 2025-11-12 | 0.9456 | 0.1930 | 11 | +1.453 |
| double_bottom | 2024-05-09 | 0.8922 | 0.0694 | 23 | +1.452 |
| double_top | 2026-01-29 | 0.8639 | 0.1942 | 11 | +1.486 |
| double_top | 2020-02-28 | 0.8008 | 0.2720 | 15 | +1.465 |
| double_top | 2020-04-27 | 0.7547 | 0.7855 | 12 | +1.437 |
| double_top | 2021-03-08 | 0.8207 | 0.6356 | 11 | +1.433 |

Note the last two rows: `double_top` 2020-04-27 has `model_prob = 0.7855` and 2021-03-08 has `0.6356` — **the highest-conviction events by the model's own score** — and the trend gate still throws them away. A gate that discards the model's most confident winners while keeping the model's least confident ones is not selecting on anything the model agrees with, which is exactly what the "no selection power" result predicts.

### 2.6 §4.2 OOS sample-size gate — **FAILED**

The requirement: ≥ 100 OOS events measured **after** the semantic gates (2.1 + 2.2). Two split conventions exist in the repo and I report both rather than choosing silently:

- **`fixed`** — absolute date, `≥ 2023-10-12` (the convention in `research/multi_backtest/scripts/oos_lsw_model_filter.py`, `oos_recent_2023H2_2026`). Used as the headline because a fixed date is reproducible independently of how many events a gate removes.
- **`last-40%`** — the convention the *training* uses (`sequential_split(oos_frac=0.40)`, `patterns/double_bottom/scripts/train_walkforward.py`). This is the split the shipped `model_prob` model was fit on, so it is the honest window for judging a model_prob-based decision.

Measured `(max_len=40 + trend lb30/r2 0.3)`:

| pattern | total kept | OOS fixed | fixed verdict | last-40% | last-40% verdict |
|---|---|---|---|---|---|
| double_bottom | 118 | **43** | **FAIL** | **47** | **FAIL** |
| double_top | 105 | **38** | **FAIL** | **42** | **FAIL** |

With the rule floor 0.6 added: DB 40 / 45, DT 37 / 41 — also FAIL. And note **`double_top` fails the fixed split even at baseline (98 < 100) before any new gate**, so the pattern's sample budget was already marginal.

**Verdict: FAILED.** Reported per-pattern, as the captain has ruled (not combined): averaging the two patterns would produce a single number that hides the fact that each pattern individually fails, and the gate is a per-pattern property.

The captain independently confirmed the stronger form of this finding: **`double_top` fails the fixed-split OOS gate at baseline — 98 events before any new gate is applied** (DB 151). Its sample budget was already short of 100, so no new gate could have rescued it.

Per the request's own rule this is reported as **FAILED**, not rounded up. Adding the `model_prob` gate makes it far worse (DB OOS 24 at 0.40, 7 at 0.55). **No combination of the proposed gates reaches 100 OOS events while retaining a measurable edge.** These measurements do not license enabling any of the §2.1/§2.2 gates by default.

### 2.7 Premise check — do the gates catch the cited events?

All cited events were located on the full history. Reproduced exactly:

| request §7 citation | located event | rule_score | model_prob | len | 2.2 | 2.1 trend | 2.3 rule | 2.3 prob | caught? |
|---|---|---|---|---|---|---|---|---|---|
| `double_top SELL p=0.033` | XAUUSD-DT-000234, 2024-03-21 | 0.8293 | **0.0333** | 11 | pass | **FAIL** | pass | **FAIL** | **yes** (2.1 + prob) |
| `double_bottom BUY p=0.069` | XAUUSD-DB-000362, 2026-08-27 | 0.9007 | **0.0694** | 15 | pass | **PASS** | pass | **FAIL** | yes (prob only) |
| `double_top s=0.79` (~7–8 Jul) | XAUUSD-DT-000318, 2026-07-08 | 0.7880 | 0.2267 | 12 | pass | — | — | fail (≥0.55) | yes (prob) |
| `double_bottom p=0.069` (27 Aug) | as row 2 | | | | | | | | |
| 2026-07-06 DT/DB shared swing | DT-000317 (07-06) + DB-000352 (07-06) | 0.7619 / 0.7217 | 0.1942 / 0.0891 | 14 / 12 | pass | — | — | fail | yes (prob) |

Four further DT events share the exact cited `model_prob = 0.0333` (2025-03-26, 2025-06-20, 2026-07-16, 2026-09-02) — the model saturates at that floor, so 0.0333 is a calibrated output value, not a unique fingerprint.

**Findings:**
- The `0.033` triple-barrier range does **not** identify a unique event; the request's citation is reproducible but not specific.
- **The 2026-08-27 double_bottom (p=0.069, rule 0.90) PASSES the trend gate** — the gate that §1.1 says should have caught it does not, because the HMM/slope both see a preceding downtrend there. Only the `model_prob` gate catches it. This directly contradicts §1.1's diagnosis of that event.
- The request §1.3 narrative quotes different numbers than §7 (`0.93 / 0.77` rule and `0.180 / 0.272` prob in §1.3 vs `0.79` / `0.033` / `0.069` in §7). I could not reconcile the two sets to specific events on the full history; they appear to come from a different (later, possibly live/re-run) detector state than the 2018–2026 backtest. **Flagged as unresolved, not assumed.**

---

## 3. What I could NOT verify

1. **The request §1.3 rule/model numbers (`0.93/0.180`, `0.77/0.272`).** Not reconcilable to events on the full history; likely from a different run. Anything in the report that depends on those exact values is unverified.
2. **Peak-TWR / equity-curve impact of each gate.** Only per-trade expectancy, winrate and total R were measured. Correlated/overlapping positions, exposure caps and RiskGuard interaction are **not** modelled — a gate that improves per-trade expectancy can still worsen drawdown or trade count. Not measured, therefore not claimed.
3. **`double_top` model_prob gate behaviour at any threshold** — OOS n=9–16 is too small for any confident statement either way. I report "does not survive OOS validation", which is what the data supports; I do **not** claim a profitable DT gate is impossible, only that this measurement does not demonstrate one.
4. **Whether `rule_score` could be redesigned to measure trend context** (request §1.3's actual root cause). Out of scope for t6 and not measured.
5. **Interaction with the live/wired engine path.** I measured the detector pool + shipped `probability_gate` function directly. The end-to-end live `SignalEngineV2` → `PendingSignal` path (including `min_model_prob` resolution order, confluence grouping, lifecycle states) was **not** exercised; that belongs to t3's and t8's scope.
6. **`config_hash` / `feature_schema_version` regression for the HMM feature.** `trend_hmm.py` documents that it adds no `hmm_*` model feature and therefore does not bump the schema; I did not independently verify that claim.
7. **The `2026-09-03` end-of-data truncation.** `simulate_trade` needs forward bars; the last few events may have shortened horizon windows. Baseline `n_trades` equals `n_events` in all arms, so no arm was silently truncated, but late-2026 events are marginally less reliable.
8. **Phương án C as a LIVE gate.** I measured the HMM offline only. Because the regime is inert live (no `regime`/`hmm` keys in `research/configs/symbols/XAUUSD.yaml`; `MultiPatternEngine` never constructed in `signal_polling_engine_v2.py`), **no live C result exists**. My C numbers must not be read as live behaviour, and I did not attempt the wiring.
9. **Whether `model_prob` values measured offline equal what the live engine attaches.** I reproduced the live inference contract (`build_feature_frame` + per-pattern calibrator) as faithfully as I could and it resolved 25/25 schema features, but the live `a.scorer` path was not exercised end-to-end (see item 5). If live feature assembly differs in any column, live `model_prob` could differ from my offline values.
10. **`pattern_length` semantics for non-double patterns.** My §2.2 finding is specific to DB/DT, whose span is `extreme2_bar − extreme1_bar`. Other plugins define `pattern_length` differently, so my "max 29/26" figures say nothing about them; I did not measure those.

---

## 4. Recommendations

1. **Ship all gates OFF / opt-in.** This matches the existing shipped defaults (`DEFAULT_PROB_GATE_ENABLED = False`, `DEFAULT_RULE_SCORE_FLOOR = None`, detector `max_pattern_length_bars = 30`, `min_rule_score = 0.0`, `trend_context_enabled = False`) — the implementation already honours §4's "measure before enabling". No default needs to change to satisfy this measurement.
2. **Do not enable §2.1 (B or C) by default.** No measured benefit (§2.4); both fail §4.2 (§2.6); §2.4's DT rule is inverted (§2.5); and it is the sole cause of 100% of the measured false-drop cost (§2.5b). Phương án C additionally **cannot** be a live gate today — the regime is inert live, so its status is pending/not-live-wired, not "offline-measured and ready".
3. **Do not enable §2.2 by default.** It removes zero events. The shared default **30** that gate_engineer shipped is the right choice (0-drop for both patterns, tightest no-op bound across both) — keep it as inert defence-in-depth. Correct the "range 55+ nến" premise (§1.2) in the request and in any code comment, since leaving a false premise in the repo invites a future session to "fix" a non-problem.
4. **Do not implement the §2.4 step-4 rule as written.** If a trend gate is ever revisited, the measured rule for `double_bottom` is "exclude `uptrend`" (not "require `downtrend`"), and for `double_top` the sign as proposed is backwards; neither should ship before a re-measurement on a larger OOS sample.
5. **Do not ship `PROB_THRESHOLDS = {0.55, ...}`.** Those B1 values keep ~14%/13% of events and were drawn at the ~86th percentile of the models' own output, not from the distributions. If a prob gate is pursued, re-derive from the measured distribution (median 0.216/0.257) and re-validate OOS.
6. **The one change worth pursuing is the `model_prob` AND-gate for `double_bottom` at ≤ 0.40**, and even that needs a larger OOS sample before it becomes default behaviour. It is the only gate in this study with an OOS-validated expectancy gain (+0.1346R → +0.3865R on 24 events).

---

## 5. Regression evidence (request §4.3)

Run `2026-09-11` on the current working tree:

| Check | Command | Result |
|---|---|---|
| `no_lookahead` | `PYTHONPATH=… /tmp/ptv2_venv/bin/python -m pytest -m no_lookahead --ignore=tests/test_gui_onboarding_registry.py -q` | **265 passed, 2 skipped, 369 deselected** (exit 0) |
| Golden LSW | `… -m pytest tests/test_liquidity_sweep_golden.py -q` | **15 passed** (exit 0, 191s) — bit-identical, matches the 15/15 the request requires |
| Prob-gate suite | `… -m pytest tests/test_trend_hmm_rework_prob_gate.py -q` | **45 passed** (exit 0) |

**Two environment caveats, both pre-existing and both must be disclosed:**

1. **`tests/test_gui_onboarding_registry.py` aborts the interpreter (SIGABRT / exit 134) with zero output during collection.** Unmarked, it kills the entire `-m no_lookahead` run before any test executes — the run looks like "no output, exit 134", not a normal failure. The file is from the initial commit (`19a5295`, 2026-09-08) and is untouched by this rework. **`--ignore=tests/test_gui_onboarding_registry.py` is mandatory** for any full-suite run until it is fixed.
2. **The request's "168 tests" figure is stale.** The `no_lookahead` selection now yields 265 passed + 2 skipped. Reported as measured, not adjusted to match the document.

I did **not** run the `config_hash` gate: no measurement in t6 changes the feature schema or any pinned config, and `trend_hmm.py` documents that it adds no model feature. If a later task does bump `feature_schema_version`, that gate must be re-run by its owner (t8).

---

## 6. Provenance

| Artifact | Role |
|---|---|
| `docs/rework_trend_hmm_measure.py` | main study (extends `docs/rework_measure.py`) |
| `docs/rework_trend_hmm_measure.json` | all numbers in §2.1–2.4, 2.6, 2.7 |
| `docs/trend_hmm_option_c.py` | Phương án C (HMM) measurement |
| `docs/trend_hmm_option_c.json` | all numbers in §2.5 |
| `docs/rework_measure_baseline.json` | prior-session baseline, used only for cross-check (DB 360 / DT 316 vs 354 / 310 today — the pool shrank because NMS + `drop_opposite_overlap` landed since; the difference is expected, not a discrepancy) |

**Measured vs assumed:** every number in §2 is measured by the commands in §1. §3 lists what is not. The two places where I explicitly did *not* substitute a value: `model_prob` (never defaulted to 0.5 — computed via the real contract, and the §2.3 gate is reported as not measurable if an artifact were missing) and Phương án C (measured because the module exists; I did not assume it was pending, and had it been absent I would have reported `pending`).
