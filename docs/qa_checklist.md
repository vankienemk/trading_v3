# QA Checklist & Bias Audit — REVERSAL_PATTERN_ENGINE_SPEC_v1.1 (§3, §13 A7)

| | |
|---|---|
| Auditor | bias-auditor (Agent 7) |
| Reviewed PRs | t5 (Liquidity Sweep refactor), t7 (Double Bottom/Top), t8 (Wedge + Head & Shoulders) |
| Spec sections | §3.1–3.5 (Causal Rules), §6.2 (config hash), §6.3/§6.5 (gates/cost), §8.3 (synthesizer), §11 (PATTERN_SPECS), §12 (backtest ≡ live), §15.6 (CI) |
| Automated counterpart | `trading_v3/tests/test_adversarial_lookahead.py` (19 tests, CI-permanent) |
| Date | 2026-09-07 |

---

## 1. Tóm tắt kết quả (Executive Summary)

**Verdict (t5 — Liquidity Sweep refactor): ✅ PASS** — không phát hiện `CausalityViolation` tiềm ẩn; timeline detect→confirm→entry nhất quán; mọi feature khai báo `available_at ≤ known_at`; golden test chứng minh plugin bit-identical với legacy live chain.

**Verdict (t7 — Double Bottom/Top): ✅ PASS (2 findings low, không blocking)** — Right-Bar Rule đúng, staleness config-driven, labeling forward-only. 2 non-blocking items được ghi nhận (§5.2 F1, F2) và chuyển cho Agent 3.

**Verdict (t8 — Wedge + H&S): ✅ PASS (1 finding low)** — cùng cơ chế nhân quả t7; finding F1 áp dụng chung; hạn chế mẫu H&S đã được ghi nhận trung thực trong PR.

Mọi detector đều: kế thừa `NoLookaheadTestBase`, khai báo `feature_schema` không `uses_future_data`, gán `config_hash` + `feature_schema_version`, và được bảo vệ bởi adversarial suite mới (CI cố định).

---

## 2. Checklist §3.1–3.5 — từng detector PR

### 2.1 PR t5 — Liquidity Sweep → plugin (golden)

| # | Checklist item (§) | Kết quả | Bằng chứng |
|---|---|---|---|
| C1 | §3.1 Tách setup/confirmation/entry/label | ✅ | detect (`event_time`) → confirm (`confirmation_time`) → entry (open của confirm+1); labeling chỉ đọc forward window (`outcome_builder.py`) |
| C2 | §3.2 Pivot/Swing Right-Bar Rule | ✅ (n/a cho bar-based) | sweep không dùng pivot; run-anchor là bar đầu của run (`run_anchor_position`), `group_rule="first"` mặc định — thỏa F1 contract của labeling (không intra-run look-ahead) |
| C3 | §3.2 Feature availability | ✅ | schema: ATR/penetration/wick/reclaim/h1_trend/level_price `available_at=detect`; confirmation_* + rule_score `available_at=confirm`; validator pass trên XAUUSD thật |
| C4 | §3.3 Staleness | ✅ | window confirm = `max_wait_bars=3` scan sau anchor; §3.3 entry-drift guard (`max_entry_drift_atr`) **chưa có ở detector** → finding F3 (engine-level, Agent 6) |
| C5 | §3.4 Runtime causal check | ✅ | `validate_causality` pass; entry_bar = confirm_bar + 1; `known_at_ts == confirm_time` (asserted trong adversarial suite) |
| C6 | §3.5 Timeframe-per-assignment | ✅ | `timeframe` là field của PatternEvent/assignment; H1 context dùng closed-H1 + `ffill + shift(1)` (causal) |
| C7 | §6.2 config hash | ✅ | `config_hash_for_detector` versioned; golden test pin `config_hash=02ab91d2bb28` |
| C8 | §12 backtest ≡ live | ✅ | plugin compose đúng thứ tự legacy `_check_new_bar` (detect→dedup→sort→event_id→confirm→levels→features→scores→entry/SL/TP); golden bit-identical 121/121 events |
| C9 | §6.5 Cost model | ✅ | `cost_r = 2*(half_spread+slippage)/risk + commission_r`, `net = gross − cost`; guard reject nonzero `entry.spread_price`/`slippage_price` (test xác nhận) |

**Sign-off t5: ✅ bias-auditor**

### 2.2 PR t7 — Double Bottom / Double Top plugin

| # | Checklist item (§) | Kết quả | Bằng chứng |
|---|---|---|---|
| C1 | §3.1 Tách setup/confirm/entry/label | ✅ | detect = `s3.known_at_bar`, confirm = max(close-cross, detect), entry = confirm+1 open (fallback close-confirm chỉ ở mép frame, không vào labeling) |
| C2 | §3.2 Pivot Right-Bar Rule | ✅ | `SwingPoint.known_at_bar = bar + right_bars`; detector chỉ tham chiếu pivot sau known; `detect_time = s3.known_at_bar` (test t7: detect == extreme2 + right_bars) |
| C3 | §3.2 Feature availability | ✅ | 5 features detect + 2 features confirm (`confirm_reclaim_atr`, `confirm_range_atr`); tất cả ≤ known_at |
| C4 | §3.3 Staleness | ✅ | `max_bars_between_detect_and_confirm=60` config-driven; 0-bar window → 0 events; mọi event thỏa `confirm − detect ≤ max_wait` (adversarial test) |
| C5 | §3.4 Runtime causal check | ✅ | zero CausalityViolation full XAUUSD 2018–2026 (t7 gate report); template `NoLookaheadTestBase` inherited |
| C7 | §6.2 config hash | ✅ | versioned default config; hash 12-hex; test pin |
| C8 | §8.3 synthesizer benchmark | ✅ | bottom recall 1.000 / top recall 1.000 / FP ≤ 0.018 |
| C9 | Labeling no-leakage | ✅ | window `[entry+1, entry+H]`, `s == entry_bar+1` chứng minh lại độc lập; feature zone kết thúc tại detect < entry |
| C10 | §6.3 research gates | ✅ | n≥300, n_oos≥100, ESS≥60%, PF CI>1, PR-AUC>baseline, WF≥3 |

**Sign-off t7: ✅ bias-auditor (2 findings low — xem §5.2 F1, F2)**

### 2.3 PR t8 — Wedge + Head & Shoulders plugins

| # | Checklist item (§) | Kết quả | Bằng chứng |
|---|---|---|---|
| C1 | §3.1 Tách setup/confirm/entry/label | ✅ | detect = `s2.known_at_bar` (wedge) / `s5.known_at_bar` (H&S); công thức swing `shift(1).rolling(20)` (sweep) / left-right bars (classical) ghi trong PATTERN_SPECS |
| C2 | §3.2 Pivot Right-Bar Rule | ✅ | cùng `SwingDetector`; trendline chỉ fit trên pivot đã known (`known_at_bar ≤ detect_bar`); `structure_levels` lưu slopes/intercepts |
| C3 | §3.2 Feature availability | ✅ | detect: depth/width/convergence/interior/line_error; confirm: pierce/range — đều ≤ known_at |
| C4 | §3.3 Staleness | ✅ | 0-bar window → 0 events (t8 §3.3 proof); mọi event confirm−detect ≤ 60 |
| C5 | §3.4 Runtime causal check | ✅ | t8: `detect_bar == right_shoulder.bar + right_bars` proof; zero CausalityViolation |
| C7 | §6.2 config hash | ✅ | versioned defaults `wedge-v1.0` / `hs-v1.0` |
| C8 | §8.3 benchmark | ✅ | falling_wedge recall 0.854 / H&S 0.954 / iHS 0.954 / FP ≤ 0.038; rising_wedge ~0.47 documented (mirror difficulty, follow Agent 8 exclusion precedent) |
| C9 | §6.3 research gates | ✅ | rising_wedge FULL-PASS; falling_wedge PR-AUC near-miss (0.733 vs 0.737) documented; H&S n=151 sample-size limitation honest |
| D1 | H&S staleness (pattern dài) | ✅ | §3.3 discard đúng: cross qua neckline ngoài 60 bars → không emit |

**Sign-off t8: ✅ bias-auditor (1 finding low chung F1)**

---

## 3. Audit Labeling — không leakage giữa forward window và features

| Mẫu | Kiểm tra | Kết quả |
|---|---|---|
| Double / Wedge / H&S (`label_events`) | window = `[entry_bar+1, entry_bar+H]` — KHÔNG đọc bar ≤ entry | ✅ recompute độc lập từ raw highs/lows/closes khớp chính xác; `detect_bar < entry_bar` ⇒ feature zone rời label zone |
| Sweep (`outcome_builder`) | entry = open sau decision bar; mọi simulation anchor tại entry | ✅ `group_rule="first"` mặc định (causal); F1 note về `deepest_penetration` đã được loại khỏi backtest |
| Train/OOS split | purge (96) + embargo (24) | ✅ mọi train event: `entry + horizon + purge + embargo ≤ boundary`; candidate sát boundary bị loại (asserted) |
| ESS | greedy non-overlap window = horizon | ✅ các event giữ cách nhau ≥ window bars |
| Gate PR-AUC | logistic fit **trên train only**, features tại pivot-known bar | ✅ 9–12 causal features đọc tối đa bar detect; không đọc sau `known_at` |

**Verdict labeling: ✅ không leakage.**

---

## 4. Audit Backtest — costs đầy đủ, code path backtest ≡ live

| Kiểm tra | Kết quả | Bằng chứng |
|---|---|---|
| Cost model §6.5 exact | ✅ | `cost_in_r` formula đúng; `net = gross − cost`; columns `gross_result_r`/`cost_r`/`net_result_r` có mặt; zero-cost default là placeholder tường minh |
| Guard understate cost | ✅ | `resolve_labeling_config` raise khi `entry.spread_price`/`slippage_price` nonzero (points vs price) — test xác nhận |
| Backtest ≡ live (sweep) | ✅ | golden test bit-identical plugin vs legacy `_check_new_bar`; registry export `PATTERN_ENTRY["liquidity_sweep"]` = class plugin | 
| Backtest ≡ live (classical) | ✅ | benchmark adapter wrap **đúng class** `DETECTOR_CLASS` mà `pattern_registry` load (asserted cho cả 6 plugin) |
| §8.3 benchmark | ✅ | detector-quality gate (không cần cost — không phải backtest PnL) |

**Open item (không thuộc PR này):** `multi_backtest/runner.py` (§12, cost + CorrelationManager + caps) chưa tồn tại — nằm trong scope Agent 8 (**t4 pattern_synthesizer đã xong; runner là deliverable riêng**) và phải pass trước §15.4/§15.6. Đã track trong tìm của team; bias-auditor sẽ review khi runner land.

---

## 5. Findings

### 5.1 Blocking findings
Không có.

### 5.2 Non-blocking findings (low)

| ID | PR | Mức | Vấn đề | Fix yêu cầu | Trạng thái |
|---|---|---|---|---|---|
| F1 | t7 | low | `DoublePatternDetectorBase.feature_schema` khai báo `confirm_range_atr` (available_at=confirm) nhưng detector **không bao giờ emit** feature này; dòng `(df["high"].iloc[confirm_bar] - df["low"].iloc[confirm_bar]) / atr_k` (detector.py:264) là dead expression — contract drift (schema tuyên bố feature không tồn tại trong event). Không phải look-ahead (không feature nào đọc vượt known_at). | Populate `attributes["confirm_range_atr"]` từ dòng 264 (rõ ý định ban đầu) HOẶC drop khỏi schema + PATTERN_SPECS | routed → Agent 3 |
| F2 | t7 | low | Staleness window đo từ **extreme bar thứ 2** (`i3+1 + max_wait`) thay vì từ detect bar; hiệu quả là `confirm − detect ≤ max_wait − right_bars` (chặt hơn config). Không vi phạm nhân quả — chỉ là semantics config; đã verify mọi event thỏa. | Ghi rõ semantics vào PATTERN_SPECS (`max_wait` tính từ extreme cuối); hoặc đổi scan start = detect_bar nếu muốn đúng nghĩa "từ detect" | note, không cần code |
| F3 | t5 | low | §3.3 entry-drift guard (`max_entry_drift_atr`) chưa implemented ở sweep plugin (chỉ có trong spec) — nguy cơ đuổi giá ở live, không phải look-ahead. | Live engine (Agent 6 t9) thêm guard khi tạo signal; detector giữ nguyên để golden không vỡ | routed → Agent 6 |
| F4 | t8 | low | Entry fallback `entry_bar = confirm_bar` khi confirm là bar cuối frame (close-entry) — causal nhưng lệch convention "next open"; các event này không bao giờ vào labeling (thiếu forward window) nên không gây leakage. | Giữ nguyên; thêm comment trong detector đã có | note |

---

## 6. Adversarial suite — trong CI cố định

- File: `trading_v3/tests/test_adversarial_lookahead.py` (19 tests, ~2.5s).
- Marker: `no_lookahead` (registered trong pyproject) + `testpaths=["tests"]` ⇒ chạy trong **mọi** lần `pytest tests/`.
- Attack catalog đã chứng minh bị bắt:
  1. `shift(-1)` leak (feature đọc `close[t+1]`) → value-provenance check "reported == recompute từ bars ≤ known_at" fail tại mọi bar (lab detector poisoned fail, honest pass).
  2. Confirm-bar close dùng trong detect feature kèm `known_at` lùi 1 bar → `CausalityViolation` + template stamp gate fail.
  3. Timeline attacks: confirm < detect; known_at < confirm; feature confirm/entry trên event thiếu stamp → raise.
  4. `uses_future_data=True` → rejected tại declaration.
  5. Truncation invariance trên 5 detector classical (events trong prefix không đổi khi frame dài hơn).
  6. Staleness: 0-bar window → 0 events; `confirm − detect ≤ max_wait`.
  7. Labeling: window `entry+1..entry+H`; purge/embargo; ESS non-overlap.
  8. Backtest: cost model exact; guard cost-understate; backtest ≡ live (registry classes).
  9. Sweep real-data zero-CausalityViolation + `entry == confirm + 1` (XAUUSD slice).

---

## 7. Kết luận & routing

| Item | Hành động | Owner |
|---|---|---|
| F1 | Populate hoặc drop `confirm_range_atr` | Agent 3 (trong repair t7 nếu captain mở) |
| F3 | `max_entry_drift_atr` guard ở live path | Agent 6 (t9) |
| Open | `multi_backtest/runner.py` (§12 costs + caps) — review khi land | Agent 8 → bias-auditor re-review |
| CI | Suite đã nằm trong `testpaths`; khuyến nghị alias marker `adversarial` trong pyproject khi merge (tùy chọn) | captain |

**Tổng sign-off: t5 ✅ / t7 ✅ / t8 ✅ — bias-auditor, 2026-09-07**
---

## 8. Phase-2 completion — FULL CI gate evidence (t10, registry-engineer — 2026-09-08)

### 8.1 Full CI numbers (from trading_v3/, /tmp/ptv2_venv)

| Gate | Command | Result |
|---|---|---|
| Full suite | `pytest -q` (timeout > 5 min) | **315 passed** in 291 s |
| Golden bit-identical (§15 item 1) | `pytest tests/test_liquidity_sweep_golden.py -q` | **15 passed** in 174 s |
| Causality gate | `pytest -q -m no_lookahead` | **168 passed** (147 deselected) |
| Adversarial | `pytest tests/test_adversarial_lookahead.py -q` | **19 passed** |
| Synthesizer benchmarks + gates (§15 item 2) | synth + DB/DT/wedges suites | **91 passed** (DB/DT §6.3 PASS; ≥3 patterns exceed §8.3) |
| Dedup/confluence/shadow/kill-switch (§15 item 3) | integration + lifecycle + correlation | **45 passed** |
| Multi-backtest (§15 item 4) | `pytest tests/test_multi_backtest.py -q` | **12 passed**; reports XAUUSD+EURUSD committed (5 items) |
| Event Lake (§15 item 5) | `pytest tests/test_event_lake.py -q` | **24 passed** (100% outcome join, append-only) |
| GUI multi-pattern (§15 item 7) | offscreen onboarding registry suite + PerformanceTab probe | **7 passed**; §10.2 pattern_breakdown rows verified offscreen |
| Ruff | `ruff check .` | **All checks passed!** |
| Mypy strict (scoped per file) | contracts.py + shared_app_state_v2 + correlation_manager + lifecycle_manager + pattern_registry | **Success × 5** |
| Legacy imports (§15 legacy) | `create_symbol_engine` import; contracts.py checksum | importable OK; **b15c70e9…b5a07** (frozen) |

### 8.2 t10 mandatory-fix records

1. **correlation_manager.py:166 (mypy 1.19.1 no-any-return)** — getattr Optional field; fixed by typed local `seconds: float | None`. P Re-existing, NOT attributable to t4. `mypy --strict` on the module now Success.
2. **execution_layer_v2.py sys.path[0]** — `insert(0, _PARENT)` → `append` (would shadow stdlib `logging` for later imports); same fix in `live/db/run.py`; bootstrap imports carry `# noqa: E402` (documented rationale).
3. **Legacy ruff gate** — `research/patterns/liquidity_sweep/{scripts,pipeline_v2}` carry ~490 pre-existing violations (incl. F821 `aggregate` in train.py — dead unreachable line, removed). DECISION: **exclude from CI ruff** (captain-preferred) — enforced via `.gitignore` (ruff 0.16.6 traversal honors gitignore; config `exclude` is NOT honored for `check .` auto-discovery in this workspace — verified with clean-tree repros), with `[tool.ruff] exclude/force-exclude` retained as defense-in-depth for clean-tree CI layouts. `.gitignore` caveat documented in-file (`git add -f` required to archive those dirs). The RUNTIME `src/` package + v1.1 `detector.py` are NOT excluded and are lint-clean; legacy `src/` was cleaned (style-only + dead-code removal), golden verifies no behavior change.
4. **Stale `# trading_live/` comments** — signal_engine_v2.py:64-65 + live/db/run.py header → trading_v3 (paths verified clean).
5. **d1_exploration.py:82** — name-sniff now `"Inverse" in name → (M15: 208)` (was mislabeling IHS as 151).
6. **test_walkforward_trainer.py marker rationale** — docstring added: `no_lookahead` marker (not `NoLookaheadTestBase` inheritance) is deliberate — adversarial-corruption assertions vs shared-base structural checks.
7. **FW PR-AUC report-not-assert asymmetry** — pre-existing Agent-4-era design (gate tests assert 6/7 gates + REPORT PR-AUC; `assert_gates` in double_bottom/dataset.py still hard-asserts all 7 incl. PR-AUC ≥ baseline for DB/DT). NOT weakened — kept on record.
8. **shadow→EventLake hook** — lives at caller layer (t9 assignment loop), isolated; left functional (no refactor) per risk-minimization; documented for a future MultiPatternEngine internalization.
9. **multi_backtest §12 item-5 — HONEST NEGATIVE (golden signal)**: XAUUSD 419 trades PF 0.8931 netR −26.18; EURUSD 401 trades PF 0.5855 netR −123.39 — **multi does NOT beat best single on this window → verdict REVIEW NEEDED (committed as-is, not spun as a pass)**. This is the §12 monitoring contract working: reports + shadow data will drive the meta-model/confluence decision.
10. **RW training follow-up** — RW detector deliberately unchanged (t2 structural-only acceptance); tier-2 RW training is a documented next-step after any mirror-geometry revisit (see docs/findings_resolution.md §RW).

### 8.3 Registry coherence (t3 + t7 artifacts)

- `model_registry/index.yaml` — 8 entries, §5.5 fields on every entry, all paths resolve inside trading_v3; DB/DT validated with `live_metrics_ref → event_lake/metrics/<id>.parquet` (paths materialized by t9 wiring as shadow metrics accrue); FW/HS/IHS trained (not assignable); legacy xauusd_v2_h16_20260905/eurusd_v2_h16_20260906 migrated (lifecycle live, artifacts in trading_v3/artifacts/); `xauusd_v2` retired alias documented.
- Reports/smoke/findings docs committed: `research/multi_backtest/reports/{XAUUSD,EURUSD}_multi_backtest.{md,json}`, `docs/smoke_report_live.md`, `docs/findings_resolution.md`.

### 8.4 Frozen-contract integrity

- `research/core/contracts.py` — untouched throughout phase 2 (mtime 2026-09-07 22:02, pre-t1); `mypy --strict` Success; **SHA-256 b15c70e966778a0fce7c7e01b65e84abb7d939ffa8a988d1f969949ae95b5a07**.

**Sign-off phase 2 completion: registry-engineer (t10), 2026-09-08 — FULL CI GREEN, §15 7/7 re-verified.**
