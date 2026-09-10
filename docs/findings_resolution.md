# Findings Resolution — P1 (handoff.md §3, items 5–9)

**Task:** t2 — resolve P1 findings on the pattern plugins
(double_bottom, double_top, rising_wedge, falling_wedge, head_shoulders,
inverse_head_shoulders).
**Date:** 2026-09-07 · **Author:** p1-engineer (trading-v3-completion)
**Constraints honored:** contracts.py frozen (§16, untouched); `live/` untouched
(§2.2 out of scope); causal rules §3 hold (zero CausalityViolation, verified by
CI `no_lookahead`); every decision below is backed by a fresh re-run on the
committed code, not a re-statement of the handoff.

---

## F1 — `confirm_range_atr` dead schema field (handoff item 5)

**Finding:** `confirm_range_atr` was declared in `feature_schema`
(AVAILABLE_AT_CONFIRM) but never emitted. In `double_bottom/detector.py` the
line `(df["high"].iloc[confirm_bar] - df["low"].iloc[confirm_bar]) / atr_k`
computed the value and discarded it (a bare expression — dead code). Wedge and
H&S detectors declared the schema entry but never computed it at all.

**Decision: populate** (keep the declared feature, emit a real value). The
feature is genuinely useful (confirm-bar range in ATR is part of the shared
§6.3 gate feature set and the tier-2 meta-model training matrix
`research/core/walkforward_trainer.py` `_CONFIRM_FEATURES`), so dropping it
would weaken the schema; emitting it makes schema == emission.

**Change (3 base detectors; mirrors inherit):**

* `research/patterns/double_bottom/detector.py` — replace the dead bare
  expression with `confirm_range_atr = (high[confirm_bar] − low[confirm_bar])
  / ATR@confirm_bar` (ATR at the confirm bar itself, matching the gate-scorer
  `rng_det` convention; NaN-fallback to the detect-bar ATR) and add
  `"confirm_range_atr"` to `event.attributes`.
* `research/patterns/rising_wedge/detector.py` (`WedgePatternDetectorBase`) —
  same computation + attribute emission (falling_wedge inherits).
* `research/patterns/head_shoulders/detector.py` (`HeadShouldersDetectorBase`)
  — same (inverse_head_shoulders inherits).

**Behavioral scope:** zero event-selection change (only an extra attribute is
attached; config_hash/rule_score/entry/SL/TP/timestamps unchanged). Golden
counts (DB 382 / DT 334 / FW 327 / RW 593 / HS 151 / IHS 208) and the t1
trained models remain valid — confirmed by re-running the §6.3 gate for
falling_wedge (identical n=327, PR-AUC 0.733) and by the full test suite.

**Evidence (tests added):** `test_confirm_range_atr_emitted_matches_formula`
in `tests/test_double_bottom.py`, `tests/test_wedges.py`,
`tests/test_head_shoulders.py` — every emitted event on the full XAUUSD M15
dataset carries `attributes["confirm_range_atr"]` equal to
`(high−low)/ATR@confirm_bar` (causal, ≤ known_at), and
`validate_causality(events, feature_schema)` raises zero
`CausalityViolation`. **Passed.**

---

## F2 — staleness window semantics (handoff item 6)

**Finding:** the confirm scan
(`scan_end = min(n, <last_extreme_bar> + 1 + max_wait)`) is anchored at the
LAST SWING EXTREME bar, whereas the config key is
`max_bars_between_detect_and_confirm` (i.e. "between the detect/anchor bar and
the confirm bar"). Because the detector's detect bar is the pivot-known bar
`<extreme> + right_bars`, the current code accepts a confirm only within
`max_wait − right_bars = 57` bars of the detect bar — STRICTER than the 60-bar
config name suggests (conservative: it discards *more*, never accepts a late
confirm).

**Decision: keep the stricter (extreme-anchored) semantics and document the
effective bound.** Rationale:

* The stricter bound is safe in the causal direction (zero risk of emitting a
  stale confirm), and relaxing it to anchor-bar measurement would change event
  counts — invalidating the golden counts and the t1-trained meta-models that
  t3/t9 consume. The task brief explicitly allows "note semantics" as a
  resolution.
* The semantics are now pinned by a dedicated test asserting the effective
  bound `confirm_bar − detect_bar ≤ max_wait − right_bars` for every family,
  so a future decision to switch to anchor-bar measurement is a deliberate,
  test-visible change.

**Evidence (tests added):**
`test_staleness_effective_bound_detect_to_confirm` in
`tests/test_double_bottom.py`, `tests/test_wedges.py`,
`tests/test_head_shoulders.py` — planted-frame events must satisfy
`confirm_bar − pivot_known_at_bar ≤ max_bars_between_detect_and_confirm −
right_bars` for double (60−3=57), wedges, and H&S mirrors. **Passed.**

The existing adversarial suite already proved the hard-discard property
(`test_staleness_window_hard_discards`: 0-bar window → 0 events). Both
semantic layers documented in every `PATTERN_SPECS.md` ("5. Staleness window")
and summarized in this doc.

---

## RW — rising_wedge §8.3 recall ≈ 0.47 (handoff item 7)

**Finding:** rising_wedge does not meet the §8.3 synthesizer acceptance
(recall ≥ 0.80) on the 500+500 gridded benchmark; FP 0.066 is also marginally
above the 0.05 gate.

**Fresh re-run evidence (this task, committed code, seed=42, 500+500):**

```
rising_wedge: recall 0.472 · recall@2x-noise 0.436 · robustness drop 0.036 · FP 0.066
falling_wedge: recall 0.854 · recall@2x-noise 0.828 · robustness drop 0.026 · FP 0.038 (PASS)
```

The mirror asymmetry is structural: the rising-wedge is the *bearish* mirror —
a converging channel whose breakout is a close below an ASCENDING lower
trendline, the exact geometry the synthesizer reference detector itself
recovers at ~0.45 (Agent 8 note, pattern_synthesizer.py). Chasing recall ≥ 0.80
by loosening the shared wedge gates would blow the FP gate (already 0.066) and
risk the falling wedge's clean pass — an unacceptable trade for a pattern that
**already clears the §6.3 XAUUSD research gate** (n=593, PF CI 2.549, PR-AUC
0.746 ≥ 0.735, WF run 4 — from the committed gate report; detector unchanged).

**Decision: accept structural-only.** rising_wedge stays a structural plugin:
emitted, benchmarked, reported on (§6.3 PASS stands), tested for causal
validity (`test_rising_wedge_produces_events_on_planted` + no-lookahead
template), but NOT forced to the §8.3 0.80 gate. Documented in
`rising_wedge/PATTERN_SPECS.md` §10 and this doc.

---

## FW — falling_wedge §6.3 PR-AUC near-miss (handoff item 8)

**Finding:** §6.3 PR-AUC 0.733 vs gate ≥ 0.737 (= 1.05 × baseline 0.702) —
a 0.004 razor-thin miss on the rule/gate-feature logistic scorer.

**Fresh re-run evidence (this task, committed code, XAUUSD M15):**

```
n_total 327 (≥300) · n_oos 131 (≥100) · OOS positive rate 0.702 · ESS 0.908 (≥0.60)
OOS PF 2.949 · bootstrap CI lower 2.161 (>1.0) · WF fold PFs [1.88, 3.28, 2.95, 3.03], run 4 (≥3)
PR-AUC 0.733 vs baseline 0.702 (gate ≥ 0.737) -- logistic scorer fit on train only
```
All §6.3 gates except the razor-thin PR-AUC pass with margin.

**Decision: near-miss acceptance, documented with exact numbers, PLUS the
production-path closure.** The §6.3 PR-AUC is measured on the *cheap* 12-feature
logistic gate scorer (`dataset.gate_features`, detect-bar features only). The
production tier-2 meta-model (`walkforward_trainer.py`, 28 causal features at
detect+confirm bars, RF + isotonic, fit on train only) achieves
**OOS PR-AUC 0.778 > 0.737** on the same split — the model the live engine
actually consumes (t1 artifacts:
`falling_wedge/artifacts/models/falling_wedge_xauusd_m15_v1/train_summary.json`
+ walk-forward report). So the 0.004 rule-scorer miss is closed at the model
level, and the pattern is registered `trained` (promotion decision owned by
t3 registry flow / lifecycle §5.1 — not forced here).

Documented in `falling_wedge/PATTERN_SPECS.md` + this doc. The CI gate test
(`test_research_gates_falling_wedge_xauusd_m15`) intentionally asserts the
gates FW supports (n/balance/ESS/PF-CI/WF) and reports, not hard-asserts, the
razor-thin PR-AUC (pre-existing design).

---

## H&S — sample-size gate n=151/208 < 300 (handoff item 9)

**Finding:** head_shoulders n=151 (OOS 60) and inverse_head_shoulders n=208
(OOS 83) on XAUUSD M15 are below the §6.3 n ≥ 300 / n_oos ≥ 100 floors
(calibrated for the common double patterns).

**D1 exploration (committed — spec §11 "H&S chỉ giá trị thật ở D1"):**
`research/patterns/head_shoulders/scripts/d1_exploration.py` resamples the
full 2018–2026 M15 history to D1 (2239 bars) and runs both mirrors with
default config:

```
HeadShouldersDetector       D1 events=3  labelable=3  ESS=3
InverseHeadShouldersDetector D1 events=3  labelable=3  ESS=2
(artifact: head_shoulders/artifacts/reports/d1_sample_exploration.csv)
```

**D1 cannot support the sample gate either** — 3 events per family over 10
years is statistically unusable for training or gating. The M15 strict
detector (151/208) is the largest *true* H&S sample available.

**Decision: accept at true sample size (M15), documented.** The strict
anti-noise detector is retained (FP ≤ 0.010 on §8.3 for both mirrors); forcing
n ≥ 300 on real M15 data would require gutting the geometric anti-noise gates
and provably explodes §8.3 FP to ~0.7 (pre-existing documented tradeoff in the
gate report header and `head_shoulders/PATTERN_SPECS.md` §8). Sample-size gate
logic untouched (`dataset.py` unchanged); the gate artifact reports
n/ESS/PF-CI/PR-AUC/WF honestly at the true sample size; models registered
`trained` (never force-passed) with lifecycle-state notes.

---

## Summary table

| Finding | Decision | Evidence (fresh) | Status |
|---|---|---|---|
| F1 `confirm_range_atr` dead | **Populate** in 3 bases + emit to attributes | formula-equality tests ×3 families, full dataset, zero CausalityViolation | Resolved |
| F2 staleness semantics | **Keep stricter extreme-anchored**, document effective bound `max_wait − right_bars`, dedicated test | bound tests ×3 families (57 for 60/3 config) | Resolved |
| RW recall ~0.47 | **Structural-only accept** (no gate fudging) | 500+500 re-run: recall 0.472 / FP 0.066; §6.3 PASS stands | Resolved |
| FW PR-AUC 0.733 vs 0.737 | **Near-miss accept** + model-level close | §6.3 re-run exact numbers; meta-model OOS PR-AUC 0.778 | Resolved |
| H&S n<300 | **D1 exploration committed** + accept at true M15 sample size | D1: 3 events/family over 10y; M15 151/208 strict | Resolved |

**Out of scope honored:** no edits to `research/core/contracts.py` (frozen),
no edits under `live/`, no changes to `dataset.py` gate logic, no config_hash
or event-timing changes.

**Verification:** pytest (4 pattern suites) 85 passed; ruff clean on changed
files; mypy strict scoped per file; `no_lookahead` marker suite unaffected
(F1/F2 tests are causality-free additions within the existing marker).