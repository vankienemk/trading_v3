# PATTERN_SPECS — rising_wedge (P3, mirror of falling_wedge)

Pattern plugin spec per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 / §13 Agent 4.
Detector: `research/patterns/rising_wedge/detector.py` (class
`RisingWedgeDetector`, version `1.0`).

Rising Wedge is the exact mirror of Falling Wedge and reuses **100 % of P3
infra** — `WedgePatternDetectorBase` + `research/core/trendline.py` + the
shared `SwingDetector`.

## 1. Geometry definition

A **rising wedge** is a bearish reversal on an uptrend approach — a converging
channel where BOTH boundary trendlines ascend and the lower line rises faster
(the channel narrows toward an apex), bounded by ascending swing HIGHS and
lows:

* three **ascending** swing lows (`low1 < low2 < low3`) anchoring the *lower*
  trendline;
* ≥ 2 **ascending** swing highs strictly between the first and last low,
  anchoring the *upper* trendline;
* the channel **converges** (`min_convergence`) and keeps ≥ `min_interior`
  closes inside it (ATR-tolerant interior gate);
* absolute anti-noise floor `|low3 - low1| ≥ min_vol_span × median(high-low)`;
* confirmation: after `low3`'s pivot is causally known (`low3.bar +
  right_bars`, §3.2), a close crosses **below** the lower trendline within the
  staleness window.

## 2. Swing formula / 3. Confirmation / 4. Entry-SL-TP

Mirrored from falling_wedge: left/right bars 3, causal existence at
`low3.bar + right_bars`, opposing line = the **lower** trendline (close must
cross **below** it for a bullish→bearish breakout), stop above the higher
extreme (`wedge_high + ATR buffer`), target below entry.  Staleness default 60
bars (§3.3).

## 5. Staleness window (§3.3)

* `max_bars_between_detect_and_confirm = 60` default; events discarded when
  the confirm close-cross misses the window — Event Lake logs
  `discard_reason = "stale"` (engine responsibility, §3.3).
* **Semantics (handoff P1#6 / docs/findings_resolution.md):** the confirm
  close-cross scan is anchored at the LAST EXTREME bar
  (`low3.bar + 1 + max_wait`), making the effective detect→confirm bound
  `max_wait − right_bars` (57 at the 60/3 defaults) — STRICTER than the config
  key name reads.  Kept deliberately (conservative: never accepts a late
  confirm; relaxing would change event counts and invalidate trained models);
  pinned by `test_staleness_effective_bound_detect_to_confirm`.

## 6. Default config + version

Same as falling_wedge:
```python
{
  "version": "1.0", "left_bars": 3, "right_bars": 3, "atr_period": 14,
  "min_separation_bars": 2, "min_depth_atr": 1.2, "min_vol_span": 2.0,
  "min_pattern_bars": 8, "max_line_error_atr": 1.4, "min_convergence": 0.05,
  "min_interior": 0.70, "interior_tol_atr": 0.8, "stop_buffer_atr": 0.5,
  "target_r": 1.5, "max_bars_between_detect_and_confirm": 60, "cooldown_bars": 5,
  "symbol": "XAUUSD", "timeframe": "M15",
}
```
`direction = "bearish"`, `short_name = "RW"`, `feature_schema_version =
"wedge-v1.0"`, same feature schema as falling_wedge (available_at detect /
confirm).

## 7. Features / 8. Dataset / 9. Expected frequency

Identical machinery + schema to falling_wedge.  On XAUUSD M15 (2018-2026) the
rising wedge is the higher-frequency mirror (~590 events vs ~327 falling), and
it **clears the full §6.3 gate** on XAUUSD M15 (n 593, PF CI 2.55, PR-AUC
0.746 ≥ 0.735, WF 4).

## 10. Synthesizer benchmark (§8.3)

**Fresh re-run (2026-09-07, committed code, seed 42, 500+500):**
`rising_wedge: recall 0.472 · recall@2x-noise 0.436 · robustness drop 0.036 ·
FP 0.066` — the recall gate (≥ 0.80) and the FP gate (≤ 0.05) both miss
(`falling_wedge` mirror: recall 0.854 · drop 0.026 · FP 0.038 — PASS).

**P1#7 resolution (docs/findings_resolution.md): accepted structural-only.**
The rising mirror is the hardest geometry for a causal right-bar detector —
the synthesizer's own reference detector recovers it at ~0.45, and loosening
the shared wedge gates to chase recall would blow the FP gate (already 0.066)
and risk the falling wedge's clean pass.  It is still produced, benchmarked,
reported and **clears the full §6.3 XAUUSD research gate** (n 593, PF CI 2.55,
PR-AUC 0.746 ≥ 0.735, WF 4); structurally asserted in `tests/test_wedges.py`,
but not forced to the 0.80 synthesizer gate.
