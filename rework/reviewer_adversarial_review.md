# Adversarial review — trend/HMM rework (t7, reviewer)

**Reviewer:** `reviewer` (independent — did not write any implementation)
**Date:** 2026-09-11
**Request:** `rework/trading_v3_rework_request_trend_hmm.md`
**Verdict: `needs_revision`**

Every numeric claim below is raw command output I re-executed myself, not a
paraphrase of anyone's summary. Review state at time of writing:
`research/patterns/double_bottom/detector.py` sha256 `a23bea0a5e5977d1`
(Sep 11 17:52:48), `research/core/contracts.py` sha256 `b15c70e966778a0f`
(unchanged from baseline).

---

## 0. What is actually delivered vs what the request asks for

The request's four items, against the code as it now stands:

| Request item | Delivered? | Evidence |
|---|---|---|
| **§2.2** `max_pattern_length_bars` | ✅ yes, default 30 | `detector.py:192`, gate at `:456` |
| **§2.3** AND-gate in **detector** | ✅ yes, default OFF | `detector.py:212` (`min_rule_score: 0.0`), gate at `:470` |
| **§2.3** AND-gate in **hard_gate.py** / live | ✅ yes, default OFF | `hard_gate.py:85,89`, `:306-346`; live `signal_engine_v2.py:1060`; runner `:1025` |
| **§2.1** trend-context gate | ❌ **NOT WIRED** | `grep -c "trend_context_enabled\|_trend_ok\|trend_context_atr_bar" research/patterns/double_bottom/detector.py` → **0**; no `import trend_context` |
| **§2.4** HMM trend direction | ⚠️ module built, **not wired, not measured** | `research/regime/trend_hmm.py` exists (931 lines); `docs/rework_trend_hmm_measure.json` → `"phuong_an_C_hmm_trend": {"status": "pending", "module": null}` |

So §2.1's *helper* is complete and correct, but the detector-side gate that the
request actually asks for does not exist yet.

---

## 1. CAUSALITY — verified sound (no lookahead found)

This was the highest-risk item. I attacked it directly rather than trusting the
comments.

**1a. Mechanical proof the window never touches a future bar.** My own
instrumented attack (`assert_causal_window` + my own mutation test):

```
$ python /tmp/rev_poison.py
bullish, clean downtrend before e1 -> True (expect True)
after poisoning bars >= e1     -> True (must equal True)
after poisoning bar == e1 only -> True (must equal True)
after poisoning whole window   -> False (may differ: window is legitimately read)
```

Poisoning every bar at/after `extreme1_bar` — including exactly `extreme1_bar` —
**cannot change the decision**. The window `closes[start:bar]` at
`research/core/trend_context.py:104` has an exclusive upper bound, as claimed.

**1b. Direction sign — correct for BOTH directions.** I constructed synthetic
frames where the answer is known, for both signs:

```
DB (bullish) with UPtrend before e1   -> False (expect False)
DT (bearish) with UPtrend before e1   -> True  (expect True)
DB (bullish) with DOWNtrend before e1 -> True  (expect True)
DT (bearish) with DOWNtrend before e1 -> False (expect False)
```

A double bottom requires a preceding DOWNTREND and a double top an UPTREND —
not inverted. `expected_slope_sign` (`trend_context.py:62-75`) returns `-1` for
bullish / `+1` for bearish and fails closed with `0` on unknown directions, so
`abs(slope)` (the request's snippet, which would have accepted the *inverse*
context) is correctly tightened.

**1c. ATR at `extreme1_bar`, not the confirm bar.** The helper only ever uses the
ATR value the caller passes; I proved it is not silently re-deriving one:

```
atr=1.0 vs atr=1000 (same window) -> True False (2nd must be False: tiny slope/ATR)
```

The *detector-side* half of this claim **cannot be verified yet**, because §2.1
is not wired. The test that would pin it
(`test_trend_context.py::test_detector_uses_atr_at_extreme1_bar...`, asserting a
`trend_context_atr_bar` stamp equal to `extreme1_bar`) currently **skips** —
`test_trend_context.py` reports `33 passed, 8 skipped`, and the skips are the
wiring-dependent tests. **This is a real gap in the evidence: the ATR-at-
extreme1 guarantee is unproven for the detector path, not disproven.**

**1d. No-lookahead regression suites — re-run per file** (single-process full run
SIGABRTs on this machine from memory accumulation + PySide6 GUI smoke tests;
that is an **environment limitation, not a product defect**):

```
test_adversarial_lookahead: 19 passed in 3.79s
test_correlation: 15 passed in 0.44s
test_double_bottom: 17 passed in 10.34s
test_double_top: 14 passed in 4.35s
test_head_shoulders: 28 passed in 6.73s
test_wedges: 26 passed in 48.20s
test_trend_hmm_regime: 17 passed, 39 deselected in 15.88s
test_hmm_regime_no_lookahead: 39 passed, 2 skipped in 94.18s
test_hmm_regime_adversarial: 34 passed in 5.28s
test_hmm_regime_plugin: 7 passed, 29 deselected in 14.54s
test_lifecycle: 20 passed in 0.34s
test_multi_backtest: 12 passed in 34.69s
test_walkforward_trainer: 7 passed, 3 deselected in 4.90s
test_live_engine_integration: 10 passed in 1.15s
```

**~304 no_lookahead-marked tests pass, 0 failures.**

**§2.4 HMM causality** — independently verified two properties on a synthetic
3-regime series (`/tmp/rev_hmm_probe.py`):

```
fitted; state_names: ['downtrend', 'range', 'uptrend']
means on direction dim (drift): [-1.112   0.0095  1.1212]
=== LABEL STABILITY (canonical ordering) ===
canonical_state_order -> ['downtrend', 'range', 'uptrend']
MONOTONIC downtrend<range<uptrend: True
=== PREFIX-INVARIANCE (no full-series retrodiction) ===
states<=t identical on truncated frame: True
```

The forward filter is genuinely causal, and `canonical_state_order`
(`trend_hmm.py:871`) solves the real EM label-permutation hazard — without it a
refit could silently invert the gate. This is genuinely good engineering.

---

## 2. GATE LOGIC — §2.3 correct; the request's failure case reproduces

**2a. Independent AND-gate, never `combined_score`.** `hard_gate.probability_gate`
(`hard_gate.py:306-346`) tests `rule_norm < min_rule` then `prob < min_prob` as
two separate branches; `combined_score` is never read.

**2b. The exact §1.3 failure case is REJECTED:**

```
=== REQUEST FAILURE CASE (needs ON) ===
rule=0.93 model=0.180 combined= 0.555 -> (False, 'low_model_prob')
rule=0.93 model=0.600 -> (True, 'prob_ok')
```

`combined_score` = 0.555 would have passed an average-based gate; the AND-gate
drops it. That is precisely what §1.3 asked for.

**2c. The rule_score floor IS applied on a normalised scale (LSW 0..100 case).**

```
normalize_rule_score('liquidity_sweep', 51.0) = 0.51
normalize_rule_score('liquidity_sweep', 32.0) = 0.32
normalize_rule_score('double_bottom', 0.93)   = 0.93
```

LSW 51.0 → 0.51, **not** dropped as `51 < 0.6`. `RULE_SCORE_SCALE`
(`hard_gate.py:113`, mirrored from `signal_engine_v2.py:70`) declares the LSW
scale, so the naive-floor bug is avoided. The detector-side floor also sits on a
native 0..1 scale (verified: DB `rule_score` min 0.5165 / max 0.9890).

**2d. FINDING (medium) — a uniform 0.6 floor would delete 100% of LSW events.**
The normalisation is right, but the *proposed number* is not portable. Measured
on the LSW event dataset:

```
LSW rule_score: n=974 min=15.00 max=50.00 mean=33.45 median=34.00
  normalized floor 0.50 -> raw cutoff 50.0 ; pass 3/974 (0.3%) ; DROP 99.7%
  normalized floor 0.60 -> raw cutoff 60.0 ; pass 0/974 (0.0%) ; DROP 100.0%
```

The LSW rule_score vocabulary tops out at 50.0 on the 0..100 scale (= 0.50
normalised). Request §2.3's snippet shows a single `RULE_SCORE_FLOOR = 0.6`
applied to all patterns, which would drop every LSW event. The implementation
does **not** do this — `DEFAULT_RULE_SCORE_FLOOR = None`
(`hard_gate.py:94`) and it is per-pattern config — so **the code is safe**; the
finding is that the request's suggested constant must not be adopted globally.
The `threshold_config` per-pattern path already supports a distinct LSW value.

---

## 3. DEFAULTS — all preserve shipped behaviour

```
=== DEFAULTS of every new/rework key ===
  stop_mode                = 'legacy'
  target_cap_atr           = 0.0
  structure_target_atr     = 0.0
  min_rr                   = 1.0
  nms_overlap              = True
  max_pattern_length_bars  = 30      <- ON
  min_rule_score           = 0.0     <- OFF
  trend_context_enabled    = <ABSENT>  (not wired)

=== DEFAULT (no config) preserves history ===
  double_bottom    rule=  0.93 prob=  0.18 -> (True, 'prob_gate_off')
  double_top       rule=   0.4 prob=  0.03 -> (True, 'prob_gate_off')
  liquidity_sweep  rule=  15.0 prob=  0.05 -> (True, 'prob_gate_off')
```

`DEFAULT_PROB_GATE_ENABLED = False` (`hard_gate.py:85`),
`DEFAULT_RULE_SCORE_FLOOR = None` (`:94`), and the live engine's `prob_gate`
defaults to `None` (`signal_engine_v2.py:820`). The request's proposed
`PROB_THRESHOLDS` (0.55/0.50) are recorded as
`PROPOSED_PROB_THRESHOLDS` (`:97`) — **documentation only, deliberately not
wired**, exactly as §4 "measure before enabling" requires.

**The one gate that IS on by default is §2.2 at 30 — and I verified it is a
provable no-op, so it cannot change behaviour.** On the FULL 204k history:

```
double_bottom  n=376 pattern_length min=8 max=29 p50=14 | >30:0 >40:0 >60:0
   default max_pattern_length_bars=30 -> drops 0 events
double_top     n=328 pattern_length min=8 max=26 p50=14 | >30:0 >40:0 >60:0
   default max_pattern_length_bars=30 -> drops 0 events
```

Acceptable under "measured before default". Note it therefore **cannot** fix the
request's §1.2 symptom, and the code says so honestly
(`detector.py:192-195`: *"Do NOT advertise this gate as fixing the sideways-range
symptom: it cannot, because it never fires on real data"*).

---

## 4. PARITY — live ≡ backtest holds

```
  signal_engine_v2._prob_gate_allowed -> calls apply_probability_gate: True
  runner.run_symbol_backtest -> calls apply_probability_gate: True
```

Both sides delegate to the **same** `hard_gate.apply_probability_gate`, at the
same pipeline point (before a `PendingSignal`/trade is built). The runner mirror
is at `runner.py:1025-1032` with an explicit comment that turning on only one
side breaks §12 parity. `tests/test_multi_backtest.py`:
**`12 passed in 42.63s`**, including
`test_parity_group_decisions_same_as_live_engine`.

---

## 5. REQUEST §3 — no harmful change re-applied ✅

```
$ grep -n '"stop_mode"\|"target_cap_atr"' detector.py
160:            "target_cap_atr": 0.0,
168:            "stop_mode": "legacy",
```

and measured geometry on real data confirms the wide legacy stop is in force:

```
default stop_mode = legacy
default target_cap_atr = 0.0
MEASURED risk_atr: median=4.700  (legacy expected ~4.5, neckline ~0.87)
```

`stop_mode="neckline"` (winrate 48%→23%) and the 4 ATR cap (96% of events
deleted) are **not** re-applied. Both remain available only as explicit A/B
options with the measurement recorded in the config comments.

> Note: `docs/rework_measure_after.json` (mtime Sep 11 **07:14**, i.e. from the
> *earlier* session, before this team started at 16:18) reports
> `risk_atr_median = 0.868` — a neckline signature. **That file is a stale
> artifact and does not reflect current code**; the current code measures 4.700
> as shown above. Anyone reading that JSON must not treat it as this team's
> output.

---

## 6. §4.2 OOS SAMPLE SIZE — **BLOCKER: fails, and is measured on 96% of history**

**6a. Post-gate OOS is far below the required 100, on both patterns:**

```
double_bottom  cum_2.1+2.2                OOS=43   required=100 -> FAIL
double_bottom  cum_all_2.1+2.2+2.3        OOS=40   required=100 -> FAIL
double_top     cum_2.1+2.2                OOS=38   required=100 -> FAIL
double_top     cum_all_2.1+2.2+2.3        OOS=37   required=100 -> FAIL
```

Request §4.2 requires ≥100 OOS events **measured after** the semantic gates.
The measurement *did* measure after the gates (correct methodology) and reports
**40–43**. Even the ungated `double_top` baseline is **98** — below 100 before any
gate exists, which means the threshold is inconsistent with the shipped data
rather than caused by the rework. The correct response is to report FAIL rather
than tune parameters until the number passes; the captain's memo already commits
to that, and I agree.

**6b. NEW BLOCKER (missed by both the captain memo and t1) — the measurement
covers only 96.0% of the mandated history.**

`docs/rework_trend_hmm_measure.py:65` sets `START = "2018-06-01"` and `:411`
passes it to `load_symbol_frame`. The source parquet holds the full history, so
the shortfall is the script's own default:

```
FULL frame bars: 204133 2018-01-02 09:00 -> 2026-09-03 22:45
TRUNC frame bars: 195909 2018-05-09 13:00 -> 2026-09-03 22:45
dropped bars: 8224
BARS OMITTED FROM MEASUREMENT: 8224 (4.03% of history)
```

and the report records `n_bars: 195893` for a file of 204,133 rows. I
reproduced **both** windows to prove this is not a harness difference of mine —
my truncated numbers match the published report exactly:

```
double_bottom  FULL204k   n= 376 winrate=0.4787 expectancy=-0.0122
double_bottom  TRUNCATED  n= 354 winrate=0.4802 expectancy=-0.0099   <- matches report
double_top     FULL204k   n= 328 winrate=0.4116 expectancy=-0.1637
double_top     TRUNCATED  n= 310 winrate=0.4065 expectancy=-0.1662   <- matches report
```

The omitted period contains **22 DB + 18 DT events**. t1's own
`verification_findings.md:556` repeats "204,133 bars" but cites it from
`docs/pattern_rework_resolution.md` rather than from the loader, so the
discrepancy was never noticed. Required fix: re-run with the full window before
any gate default is decided, and state `n_bars` explicitly.

---

## 7. §2.1 has no selection power (independent corroboration)

From the measurement's own `false_drops` block — the events the gate **keeps**
are *worse* than the ones it drops, on both patterns:

```
double_bottom  dropped n=241 exp=-0.0084 | kept n=113 exp=-0.0131 | delta -0.0047R
double_top     dropped n=208 exp=-0.1596 | kept n=102 exp=-0.1798 | delta -0.0202R
```

A gate with negative selection power must not be enabled on this evidence. That
independently confirms `rework/captain_decision_memo.md` §2. Keeping it opt-in
with the measurement recorded in the docstring is the correct call; advertising
it as an improvement would not be.

---

## 8. §2.4 HMM trend — honest but incomplete

`docs/rework_trend_hmm_measure.json`:
```
"phuong_an_C_hmm_trend": {"status": "pending", "module": null,
  "note": "Phuong an C (HMM trend DIRECTION) module not present at
   research/core/hmm_trend_regime.py -> NOT MEASURED. Comparable numbers are
   reported as pending rather than invented."}
```

Reporting `pending` instead of fabricating a comparison is the right behaviour,
but it means §2.4 is **unmeasured and unwired**. (The module does exist, at
`research/regime/trend_hmm.py` — the script simply checks a different path. So
the measurement could be completed; it just has not been.)

---

## 9. FROZEN SURFACES ✅

```
$ shasum -a 256 research/core/contracts.py
b15c70e966778a0f   (= baseline, unmodified)
$ git diff --stat research/core/contracts.py
(no output)
$ grep -n "FEATURE_SCHEMA_VERSION = " research/patterns/double_bottom/detector.py
47:FEATURE_SCHEMA_VERSION = "double-v1.0"
```

`contracts.py` unchanged; `feature_schema_version` **not** bumped. The rationale
is sound and I verified it: the trend plugin emits `trend_hmm_*`
(`trend_hmm.py:119`), a distinct prefix from the model's `hmm_*` family, and it
is a *gate input*, not a model feature — so no re-training follow-up is owed.

---

## 10. Regression evidence (raw)

```
tests/test_liquidity_sweep_golden.py ............... [100%]
15 passed in 197.08s (0:03:17)          <- GOLDEN LSW 15/15, matches my pre-change baseline run

test_double_bottom: 17 passed in 9.22s
test_double_top: 14 passed in 3.77s
test_pattern_rework: 20 passed in 8.82s
test_trend_context: 33 passed, 8 skipped in 0.40s
test_trend_hmm_rework_prob_gate: 52 passed in 1.40s
test_trend_hmm_regime: 54 passed, 2 skipped in 67.11s
test_bug_summary_b1_b2: 23 passed in 0.74s
test_multi_backtest: 12 passed in 42.63s
test_adversarial_lookahead: 19 passed in 4.45s
```

Green. Two things a reader must not mistake for defects:

* a single-process `pytest -m no_lookahead` run **aborts with SIGABRT (exit
  134)** even at `--collect-only`, from memory accumulation plus PySide6 import
  in conftest. **Environment limitation, not a product failure** — all suites
  pass when run per file, which is how the numbers above were produced.
* while the detector was being written I transiently observed a `NameError:
  max_pattern_length_bars is not defined` mid-save. That was a **race with an
  in-flight edit**, not a defect; the settled file defines both locals at
  `detector.py:250-251` and detects correctly. I am explicitly *not* reporting it
  as a finding. It is, however, a process risk: reviewing a moving tree means any
  pre-settle observation must be re-confirmed before it is reported.

---

## 11. Findings

| id | severity | file | problem | required fix |
|---|---|---|---|---|
| **R-1** | **blocker** | `docs/rework_trend_hmm_measure.py:65` | Measurement uses `START="2018-06-01"` and covers 195,893 of 204,133 bars — **8,224 bars (4.03%) omitted**, contradicting request §4.1 "toàn bộ … (204k nến)". Omits 22 DB + 18 DT events. | Re-run with the full history (`start=2018-01-02` or no `--start`); report `n_bars` explicitly; re-check the OOS table against the corrected pool. |
| **R-2** | **blocker** | `research/patterns/double_bottom/detector.py` | **§2.1 trend-context gate is not implemented in the detector** (0 references, no import). The request's item 2.1 is undelivered; `tests/test_trend_context.py`'s wiring tests are skipped (8 skipped). | Wire `trend_context.has_trend_context` into `_detect_double` with `trend_context_enabled` (default **False**), passing the ATR known at `extreme1_bar` and stamping `trend_context_atr_bar`, then un-skip the tests. |
| **R-3** | **blocker** | `docs/rework_trend_hmm_measure.json` | §4.2 fails: post-gate OOS is **43/40 (DB) and 38/37 (DT)** against the required ≥100. | Report FAIL per the request's own rule — do **not** widen parameters to reach 100. Note that ungated `double_top` is 98, i.e. the threshold is inconsistent with the shipped data. |
| **R-4** | high | `rework/trading_v3_rework_request_trend_hmm.md` §2.3 | The snippet's single `RULE_SCORE_FLOOR = 0.6` applied to all patterns would drop **100% of LSW events** (LSW tops out at 50.0 on its 0..100 scale = 0.50 normalised). | Never adopt a global 0.6 floor. Keep it per-pattern (the code already supports this) and set an LSW-appropriate value only after measuring. |
| **R-5** | high | `docs/rework_trend_hmm_measure.json` | §2.4 (HMM trend) is `status: pending` / `NOT MEASURED`; it is also unwired. The chosen option C cannot be compared against option B. | Point the measurement at the real module path (`research/regime/trend_hmm.py`) and produce the B-vs-C comparison, or state plainly that §2.4 ships unmeasured and stays off. |
| **R-6** | medium | `rework/captain_decision_memo.md` §1 / `verification_findings.md:556` | The "204,133 bars" figure is cited from a secondary document instead of from the loader, which is why R-1 went unnoticed. | Cite the loader's actual `len(df)` in the memo, and state the window explicitly. |
| **R-7** | low | `tests/test_trend_context.py:414` | `test_disabled_gate_does_not_change_detection` omitted the documented `isolated_trend_cfg` helper that every sibling §2.1 test uses, so the §2.2 default (30) filtered the fixture (`pattern_length=35`) and the assertion failed vacuously. Now passing, so this is recorded for the record only. | Keep using `isolated_trend_cfg` for §2.1 tests; it is the documented guard against exactly this cross-gate masking. |

---

## 12. Process finding — the requirements loop is in a circular dependency

Not a code finding, but it is why the delivery is incomplete. Read from
`team.json`:

```
t2  pending  deps=t11   gate_engineer
t4  pending  deps=t2    trend_gate_engineer
t5  pending  deps=t11   hmm_trend_engineer
t6  pending  deps=t2,t3,t4
t11 in_progress deps=-  code_verifier
(t1, t9, t10 FAILED needs_revision)
```

`t2` depends on `t11`, while `t11`'s acceptance criterion #1 **is to write
`t2`'s production code** — adding `pattern_length` to the detector. So `t2`
cannot start until `t11` passes, and `t11` cannot pass until `t2`'s code exists.
Four consecutive rounds (t1, t9, t10, t11) returned `needs_revision` without
state change, and the team is now **escalated** at the review-loop ceiling — an
outcome this cycle fully predicts. The fix is to move the detector edit out of
the requirements contract and into `t2`, where it belongs.

I have reported this to the captain separately.

---

## 13. What I could not falsify

Stated so the verdict is not read as broader than the evidence:

* **§2.3 is genuinely correct and I could not break it.** The AND-gate is
  independent, never consults `combined_score`, handles the LSW 0..100 scale, is
  off by default, and is shared verbatim by live and backtest.
* **§2.1's helper is causally sound and correctly signed in both directions.**
  My own poison attacks and both-direction sign probes all passed. Its defect is
  unwiredness (R-2) and lack of measured edge (§7), not correctness.
* **§2.2 is a proven no-op** on the full history, so its default of 30 is
  behaviour-preserving — though it cannot fix the symptom the request cites.
* **Golden LSW 15/15, contracts.py frozen, config_hash and
  feature_schema_version unchanged, and the no-lookahead suites green.**
* **No §3 violation** — neither harmful change was re-applied.

The implementation quality on the parts that exist is high, and the measurement
work is honest (it reports FAIL and `pending` rather than inventing numbers).
The verdict is `needs_revision` because §2.1 is undelivered, §2.4 is unmeasured,
the OOS gate fails, and the measurement does not cover the mandated history.
