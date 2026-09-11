# §2.1 Trend-context gate — measurement report (t4, trend_gate_engineer)

**Ngày:** 2026-09-11
**Task:** t4 — implement rework §2.1 (trend-context gate for reversal patterns), Phương án B.
**Trạng thái:** **COMPLETE** — module + tests + measurement + detector wiring all done.

---

## 0. Trạng thái task

t4 depended on t2 (§2.2/§2.3 detector gates, `gate_engineer`). t2 landed
(`detector.py` 23175 → 29863 bytes); I then re-read the file, preserved their
`# --- §2.2 / §2.3 rework detector gates ---` block intact, and inserted the
§2.1 gate as an adjacent contiguous block.

The dependency chain that delayed this (t1 → t9 → t10 → t11, all four
requirements rounds failing with needs_revision) is recorded in the team history
and is not re-litigated here.

### Wiring changes to `research/patterns/double_bottom/detector.py`

| # | Change |
|---|---|
| 1 | Import the §2.1 API + documented default constants from `research.core.trend_context` |
| 2 | 4 config keys in `get_default_config`: `trend_context_enabled` (False), `trend_lookback_bars`, `min_slope_atr`, `min_r2` |
| 3 | `self.trend_context_provider` seam on `__init__` (§2.4 / Phương án C swap point) |
| 4 | `_trend_ok(closes, atr_k, extreme1_bar, cfg)` helper — disabled ⇒ always True |
| 5 | Gate block in `_detect_double`, immediately AFTER the §2.2/§2.3 block, before NMS/dedupe |
| 6 | Provenance on emitted events: `trend_context_enabled`, `trend_context_atr_bar` |

**`FEATURE_SCHEMA_VERSION` deliberately NOT bumped** (stays `double-v1.0`): the
gate is a *filter*, not a scoring feature, per `verification_findings.md` F-05
and the hardcoded `"double-v1.0"` in `scripts/train_walkforward.py:80,86`.

**config_hash** moved `d7d4c40092ee` → `c82311b35696` (t2's two keys + my four).
Attribution is pinned by `test_each_trend_key_individually_changes_the_hash` and
`test_trend_keys_do_not_collide_with_gate_engineer_keys`, so any future drift can
be assigned to one owner rather than guessed at. t6/t8 must re-baseline.

### Cross-gate interaction found and pinned

gate_engineer's `§2.2` default (`max_pattern_length_bars = 30`) filters the
synthetic test frame, whose structure window is 31 bars, *before* the trend gate
could run — which would have made every §2.1 detector assertion pass vacuously.
The §2.1 tests now disable §2.2 explicitly via `isolated_trend_cfg()` to isolate
the gate under test, and
`test_gate_composes_with_foreach_other_detector_gate` pins the coupling so a
future §2.2 default change cannot silently neutralize §2.1's tests.

A second, subtler issue: the control frames' first V-turn had to be made
**shallow** (6 bars ≈ 9 units against a 140-unit leg). With a deep V the 20-bar
window ending at `extreme1_bar` reaches back across the reversal into the
rebound, producing a genuinely negative slope so the negative control passed —
correct behaviour that nonetheless made the direction test meaningless.
`test_shallow_turn_keeps_the_control_honest` now guards that geometry.

### End-to-end verification on the real full history

FULL window (204,117 bars, no `start`):

```
default (gate OFF)      DB 376 events, 0 discards   DT 328 events, 0 discards
explicit enabled=False  DB 376 events, 0 discards   (default-preserving)
gate ON (lb20/ms.05/r2 .2)  DB 158 kept / 226 low_trend_context
                            DT 126 kept / 217 low_trend_context
§2.3 still works: min_rule_score=0.7 -> DB 286 kept   (gate_engineer intact)
```

---

## 1. Deliverables

| File | Status |
|---|---|
| `research/core/trend_context.py` | NEW — Phương án B, fail-closed, causal, provider seam |
| `tests/test_trend_context.py` | NEW — 42 tests, **all passing, 0 skipped** |
| `research/scripts/trend_context_sweep.py` | NEW — reproducible sweep |
| `research/multi_backtest/reports/trend_context_sweep.json` | NEW — machine-readable output |

Ruff: **clean** on all three Python files.

### Verification output (current)

```
$ pytest tests/test_trend_context.py tests/test_double_bottom.py \
         tests/test_double_top.py tests/test_pattern_rework.py -q
93 passed in 17.26s

$ pytest tests/test_trend_hmm_rework_prob_gate.py tests/test_core_contracts.py \
         tests/test_multi_backtest.py -q
100 passed in 36.46s

$ ruff check research/core/trend_context.py \
              research/patterns/double_bottom/detector.py \
              tests/test_trend_context.py
All checks passed!
```

The `@requires_wiring` marker on the detector tests auto-detected the wiring via
`"trend_context_enabled" in DoubleBottomDetector().get_default_config()` and
stopped skipping by itself — no test edit was needed to activate them, and they
could not silently pass while the gate was unwired.

### The shared suites I must keep green are green

`pytest tests/test_double_bottom.py tests/test_double_top.py tests/test_pattern_rework.py -q`
→ **51 passed**. This matters because t11 (requirements r4) reported that
"AC5 was PARTIALLY fixed and in doing so introduced a NEW test regression" — that
regression is **not** in the DB/DT detector path I own.

---

## 2. Causality (the highest-risk property)

The window is `closes[extreme1_bar - lookback : extreme1_bar]` — an **exclusive**
upper bound, so bar `extreme1_bar` and everything after it is never read. The
ATR is the one known **at** `extreme1_bar`, not at the later confirm bar.

Proven mechanically, not by inspection:

* `assert_causal_window` wraps the array in a proxy that records **every index
  numpy actually reads**, then raises if any read is `>= extreme1_bar`.
* `test_poisoned_future_bars_do_not_change_the_decision` overwrites everything
  at/after `extreme1_bar` with an absurd trend that would flip the answer, and
  asserts the decision is bit-identical — in both directions.
* `test_detector_uses_atr_at_extreme1_bar_not_confirm_bar` asserts the recorded
  `trend_context_atr_bar == extreme1_bar < confirm_bar`.

**The ATR anchor is material, not cosmetic.** Replaying the exact intended
insertion point over the 354 real DB events at `lookback=20/min_r2=0.2`,
substituting the ATR at the detect bar (`pivot_known_at_bar`) for the ATR at
`extreme1_bar` **flips the gate decision for 7 of 354 events**. Using the wrong
anchor is therefore a real behaviour change, not a rounding detail — which is
why the module takes `atr_k` as an explicit argument rather than reading the
series itself.

The flip count is **parameter-dependent**, so the arm must be stated. Measured
independently by `gate_engineer` and reproduced here, all six arms agreeing
exactly:

| lookback | min_slope_atr | min_r2 | flips / 354 |
|---|---|---|---|
| 20 | 0.05 | 0.2 | **7** ← shipped defaults |
| 30 | 0.05 | 0.3 | 4 |
| 20 | 0.05 | 0.3 | 1 |
| 30 | 0.00 | 0.3 | 0 |
| 30 | 0.05 | 0.0 | 13 |
| 30 | 0.10 | 0.3 | 23 |

At the rejected `min_slope_atr=0.10` arm the disagreement reaches 23/354. A
gate whose verdict depends on the choice of anchor is latent non-determinism we
should not ship, which is why the detector indexes `atr[i1]` explicitly rather
than reusing the in-loop `atr_k`.

No real event has a window that would start before bar 0 (`extreme1_bar >= 20`
for all 354), so fail-closed window handling costs nothing in practice while
still protecting short series.

### Ownership boundary with the §2.2/§2.3 block

`gate_engineer` owns the shared local `pattern_length = i3 - i1`. My §2.1
discard record originally read that local (safe, since their definition precedes
my block, but a silent coupling across ownership). It now computes
`int(i3 - i1)` locally, so the two blocks share no variable, key, or line.

---

## 3. Directionality

A double bottom is bullish and must be preceded by a **down**trend; a double top
is bearish and must be preceded by an **up**trend.

`expected_slope_sign` maps `DIRECTION_BULLISH -> -1`, `DIRECTION_BEARISH -> +1`,
unknown -> `0` (which no real slope satisfies, so it fails closed).

Note the request's §2.1 snippet uses `abs(slope)`, which would accept an
**uptrend before a double bottom** — the exact inverse of the intent. The sign
check replaces it. `test_both_directions_pin_the_sign` pins both directions, and
the end-to-end test asserts DB-after-uptrend is dropped while DT-after-uptrend
survives.

---

## 4. Measured defaults — and an honest caveat

**WINDOW LABEL (cite this when referring to these numbers):**
`load_symbol_frame('XAUUSD', end='2026-09-03')` — **no `start`**, so the FULL
file: **204,117 bars, 2018-01-02 .. 2026-09-03**.

An earlier draft of this report used `start='2018-06-01'`, which silently
dropped **8,224 leading bars (4.03%)** and 22 DB + 18 DT events — the truncation
identified by `reviewer_adversarial_review.md` R-1 and confirmed by the captain.
Every conclusion below is **unchanged** on the full window; only the in-sample
pool sizes differ (DB **376** not 354, DT **328** not 310). The OOS counts are
byte-identical, because all 8,224 recovered bars are pre-2023 and therefore
in-sample.

Baseline pool (full window): DB **376** events (OOS 151), DT **328** (OOS 98).

Sweep, `min_slope_atr=0.05` (combined OOS = DB + DT, confirm_time >= 2023-01-01):

| lookback | min_r2 | DB kept | DB OOS | DT kept | DT OOS | combined OOS |
|---|---|---|---|---|---|---|
| 20 | 0.2 | 155 | 66 | 122 | 54 | **120** |
| 20 | 0.3 | 142 | 61 | 110 | 49 | **110** |
| 30 | 0.2 | 141 | 58 | 121 | 47 | **105** |
| 30 | 0.3 | 130 | 53 | 116 | 46 | 99 |
| 40 | 0.2 | 121 | 52 | 111 | 41 | 93 |
| 60 | 0.2 | 108 | 44 | 99 | 41 | 85 |

**The request's own suggested defaults (lookback=30, min_r2=0.3) give 99
combined OOS — one short of §4.2.** lookback 40/60 fail at every `min_r2`;
`min_slope_atr=0.10` fails everywhere.

`min_r2` is the dominant driver, not the slope: median `|slope|/ATR` ≈ 0.09,
median R² ≈ 0.44 → `min_r2=0.3` sits at ~the 35th percentile and necessarily
cuts ~65% of events. `min_r2=0.5` can never reach 100 OOS.

Cross-check on the full window (§2.2 is a proven no-op here, matching the
captain's independent measurement exactly): DB n=376 max pattern_length 29,
DT n=328 max 26, **zero events > 40 bars** in either.

Chosen constants (documented in the module, **not** tuned to the 2 sample
charts the request warns about): `lookback=20`, `min_slope_atr=0.05`,
`min_r2=0.2` — the loosest combination still enforcing a real trend and the
only ones clearing §4.2 on the combined reading.

---

## 5. §4.2 verdict: **FAIL** — reported, not papered over

**The captain ruled the OOS floor is PER-PATTERN, not combined.** Rationale:
the request §4.2 states it per pattern; the whole point of the gate is that DB
and DT behave differently; and a combined count would let a strong DB mask a
weak DT. My earlier "combined reaches 120" reading is therefore **rejected** and
is recorded only to show what that reading would have given.

Per-pattern OOS under the captain's fixed split (confirm_time >= 2023-10-12):

| pattern | baseline OOS (gate off) | lb20/r2=0.2 (approved default) | lb30/r2=0.3 (request default) |
|---|---|---|---|
| double_bottom | **151** | 55 | 43 |
| double_top | **98 ← already FAIL** | 46 | 38 |

* **Every setting fails both patterns.** Nothing reaches 100.
* `double_top` is **already below 100 at baseline (98)** before any gate runs —
  the threshold disagrees with the pre-existing data, and this is not caused by
  the gate.
* `measurement_analyst` reproduced the same FAIL under the repo's alternative
  last-40% training split (DB 118→47, DT 105→42).

**No parameter was loosened to reach 100.** The captain will carry the FAIL to
the user; a failed gate reported honestly is the correct outcome.

---

## 6. The gate has NO demonstrated selection power

Confirmed independently by **three** members: this engineer, `measurement_analyst`
(via `docs/rework_trend_hmm_measure.py`), and the captain. All three used a
post-hoc filter over an identical candidate pool, so no arm has a pool advantage.

Expectancy of kept vs dropped at lookback=30/min_r2=0.3 (`measurement_analyst`;
their counts are on their own frame — see the window note below):

| pattern | kept n | exp(kept) | dropped n | exp(dropped) | gap |
|---|---|---|---|---|---|
| double_bottom | 118 | -0.0101R | 236 | -0.0084R | 0.0017R |
| double_top | 105 | -0.1599R | 205 | -0.1596R | 0.0003R |

The gap is inside the noise — the deleted events do **as well as** the kept ones.
Baseline with every gate off: DB 354, winrate 0.4802, exp -0.0099R; DT 310,
winrate 0.4065, exp -0.1662R.

**False drops verified against the detector output** (not just quoted): 66 DB and
52 DT dropped events have `rule_score >= 0.75` AND a winning outcome. I
re-checked the two named examples directly at lb30/r2=0.3 and both are in the
real pool and both dropped:

```
DB rule 0.9456  detect_time 2025-11-12  kept=False  -> +1.453R
DT rule 0.8639  detect_time 2026-01-29  kept=False  -> +1.486R
```

Independent corroboration on an axis measured **before any trade outcome** — the
candidates' own `rule_score`, at lb30/r2=0.3. **Re-measured on the FULL window**
(pool DB 376 / DT 328), which is why these counts are larger than
`measurement_analyst`'s; the money conclusion is identical:

| pattern | kept n | mean rule_score | dropped n | mean rule_score | diff |
|---|---|---|---|---|---|
| DB | 130 | 0.7611 | 246 | 0.7663 | **-0.0052** |
| DT | 116 | 0.7758 | 212 | 0.7713 | **+0.0045** |

Both partitions are exact (130+246=376, 116+212=328). A gate with selection
power would keep the higher-quality candidates. This one keeps marginally
*lower*-scoring ones — on both windows and both patterns, the sign of the
difference is noise, never a gain.

The best-looking single cell is `lb60/r2=0.3` (DB exp +0.1187R) — almost
certainly noise: n=96 and a 6-cell grid, so picking the winner after seeing the
grid is exactly the overfitting the request §2.1 warns against. No cell shows a
statistically defensible gain.

**Consequence:** the gate ships **opt-in, default OFF**, and the module
docstring carries this evidence with an explicit instruction not to enable it on
the strength of it. `test_module_does_not_document_an_expectancy_improvement`
and `test_no_improvement_claim_in_default_constants` now pin that contract so a
future default flip or a docstring rewritten into a performance claim trips a
test instead of passing silently.

---

## 7. Provider seam for §2.4

`TrendContextProvider` is a `runtime_checkable` Protocol; `SlopeTrendContextProvider`
implements it. The HMM trend-direction work can replace the heuristic by
assigning `detector.trend_context_provider = HmmTrendProvider(...)` with **no
detector change**. `test_alternative_provider_can_be_injected` proves a
state-based provider is drop-in.

---

## 8. Known impact on other tasks

* **config_hash (reviewer F-05):** adding the 4 config keys to
  `get_default_config` changes `compute_config_hash` for every DB/DT event.
  Current hash of existing defaults: `d7d4c40092ee`. t6/t8 must re-baseline.
* **Rule for the wiring step:** insert the gate immediately **after** the depth
  gate (`detector.py:257-258`), per `rework/verification_findings.md` §2, and
  preserve gate_engineer's §2.2/§2.3 edits intact.
* `discard_reason = "low_trend_context"` — verified to match the existing Event
  Lake convention (`verification_findings.md` §5).
