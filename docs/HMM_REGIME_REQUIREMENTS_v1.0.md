# HMM Regime Plugin — Requirements Contract v1.0

**Phiên bản:** 1.0 — **Ngày:** 2026-09-08 — **Trạng thái:** Requirements (t1) — **Nguồn:** `HMM_REGIME_PLUGIN_INTEGRATION_GUIDE.md` v1.0 (§1–§10) → `REVERSAL_PATTERN_ENGINE_SPEC_v1.1.md` (§3, §5.5, §15)

---

## 1. Mục đích

Tích hợp HMM Regime Plugin vào Trading V3 như một **Environment / Regime Engine độc lập**:

1. **Feature Emitter** — inject `hmm_state`, `hmm_prob_*`, `hmm_confidence` vào `PatternEvent.attributes` (và feature frame khi train model mới), plug/unplug an toàn, model cũ **không đổi hành vi**.
2. **Hard Gate / Filter** — chặn/cho phép pattern chạy theo regime tại `known_at` của event (nằm ngoài feature vector, không ảnh hưởng model).
3. **Registry §4.3** — metadata bắt buộc `feature_list` (exact train-time list) + `optional_plugins` trên mỗi entry `model_registry/index.yaml`.
4. **OOS test** — đo lường hiệu quả của regime filter trên **LSW, DB, DT** bằng protocol có kiểm soát no-lookahead.

**Ràng buộc tối thượng (hard constraints, không thương lượng):**
- **KHÔNG thêm dependency mới.** Venv `/tmp/ptv2_venv` (numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, joblib) **KHÔNG có hmmlearn**. Mọi HMM phải tự implement trên numpy + scipy.special + sklearn (GaussianMixture để khởi tạo emission — đã verify import OK).
- Mọi predict phải **causal** (§2 guide): tại bar `t` đóng, chỉ dùng dữ liệu ≤ `t`.
- **Không fit trên full series rồi map state ngược** (§2.2 guide).
- Model cũ (train trước HMM) — `feature_list` không chứa `hmm_*` — phải chạy y nguyên sau tích hợp (§4.2C guide).

---

## 2. Vị trí module & file layout (đích)

```
trading_v3/
├── research/
│   └── regime/                          # NEW package (plugin regime)
│       ├── __init__.py
│       ├── base.py                      # RegimeState + BaseRegimePlugin (ABC)
│       ├── causal_hmm.py                # CausalGaussianHMM (numpy/sklearn, forward-filter)
│       ├── emitter.py                   # HMMFeatureEmitter (event.attributes + feature frame)
│       └── hard_gate.py                 # is_allowed() + regime_filter rule parsing
├── research/configs/plugins/
│   └── hmm_regime.yaml                  # NEW — config mẫu guide §6
├── tests/
│   ├── test_hmm_regime.py               # NEW — unit + no-lookahead (§7 guide)
│   ├── test_hmm_adversarial.py          # NEW — synthetic future-data injection suite
│   └── (mở rộng) test_multi_backtest.py # parity test: backtest ≡ live khi có hard gate
├── research/multi_backtest/scripts/
│   └── oos_hmm_regime.py               # NEW — OOS protocol LSW/DB/DT (t5)
└── model_registry/index.yaml           # MODIFIED — §4.3/§5.5 fields
```

**File KHÔNG được sửa** (frozen): `research/core/contracts.py` (Agent-1 freeze), `research/core/causal_checks.py`, `research/core/config_hash.py` — mọi mở rộng qua module mới, không edit contract cũ.

---

## 3. Interface contract

### 3.1. `RegimeState` (dataclass — align guide §3.1, causal semantics §2)

```python
@dataclass
class RegimeState:
    timestamp: pd.Timestamp          # bar close time = known_at của state
    state: int                       # 0, 1, 2, ...
    state_name: str                  # "trending" | "sideways" | "high_vol" (từ config)
    state_prob: dict[str, float]     # {"trending": 0.82, "sideways": 0.15, ...}
    confidence: float                # max(state_prob.values())
    lag_bars: int                    # độ trễ áp dụng (default 0)
    model_version: str               # plugin version
    config_hash: str                 # §6.2 sha1(canonical_json(hmm_config))[:12]
```

**Ràng buộc bắt buộc:**
- `timestamp` = `df.index[bar]` (close time của bar) — state chỉ "tồn tại" tại close.
- Index alignment contract: `states[i]` tương ứng `df.index[i]`, list trả về có độ dài bằng số bar closed.
- `lag_bars` phải khai báo rõ; nếu > 0 thì `timestamp` được dịch chuyển lùi `lag_bars` bar và **test ảnh hưởng performance** bắt buộc (guide §2.4, §7).

### 3.2. `BaseRegimePlugin` (ABC — align guide §3.1)

```python
class BaseRegimePlugin(ABC):
    name: str = "hmm_regime"
    version: str = "1.0.0"
    short_key: str = "HMM"            # registry §4.3 optional_plugins name khớp name

    @abstractmethod
    def fit(self, df: pd.DataFrame, config: dict) -> "BaseRegimePlugin":
        """Fit HMM trên dữ liệu LỊCH SỬ (chỉ dùng cho research / warm-up).
        Không bao giờ fit trên dữ liệu chứa vùng OOS sẽ đánh giá."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, config: dict) -> list[RegimeState]:
        """Causal inference. df đã sort tăng dần theo timestamp.
        Trả về state cho MỌI bar đã đóng, chỉ dùng dữ liệu ≤ bar hiện tại.
        KHÔNG backward smoothing, KHÔNG Viterbi full-path."""

    @abstractmethod
    def get_feature_schema(self) -> list[PatternFeature]:
        """Schema các feature inject — mọi feature uses_future_data=False."""

    @abstractmethod
    def get_default_config(self) -> dict:
        """MUST chứa 'version' (§6.2) + n_states, state_names, features,
        lag_bars, min_confidence, random_state, covariance_type."""

    def get_feature_names(self) -> list[str]:
        """Tên cột feature emitter: hmm_state + hmm_prob_<state_name> + hmm_confidence."""
```

### 3.3. `CausalGaussianHMM` (concrete — numpy/sklearn, KHÔNG hmmlearn)

Requirements-level specification (chi tiết implement thuộc t2):

- **Emissions:** Gaussian đa biến per state. Khởi tạo qua `sklearn.mixture.GaussianMixture(n_components=n_states)` trên input features (causal-scaled); tham số EM học trên cửa sổ fit.
- **Transition:** ma trận chuyển trạng thái (numpy), học từ chuỗi state huấn luyện (có thể dùng forward-backward EM đơn giản viết tay trong numpy — scipy.special.logsumexp; flexible cho t2, miễn thỏa các contract dưới).
- **Inference CAUSAL bắt buộc:** forward-filtering chỉ. State tại bar `t` = argmax của `P(s_t | x_1..x_t)` (filtered posterior) — tuyệt đối **không** backward pass, **không** full-series Viterbi (cả hai đều nhìn tương lai).
- **Reproducible:** `random_state` trong config → hash; `config_hash` = `compute_config_hash(hmm_config)` (§6.2 convention) — kèm `model_version`, `data_version` (hash index của df) ghi trong artifact/report.
- **Incremental contract (test bắt buộc):** chạy `predict` trên full df == chạy `predict(df.iloc[:t+1])` lấy state cuối tại TỪNG t (forward filter là đệ quy → kết quả phải giống hệt; sai lệch = fail).
- **Input features khuyến nghị (configurable):** `log_return_1`, `atr_14_norm`, `volume_zscore_20` (guide §6) — mọi feature phải causal (giá trị tại bar `t` chỉ từ ≤ `t`).
- **Stability guard:** nếu số bar fit quá ít (< `min_fit_bars`, default 2000) → `fit()` raise; live/predict trên ít bar hơn `min_predict_bars` → trả state mặc định (state 0, confidence 0.5, `lag_bars=0`) và **fail-closed khi dùng làm required feature** (xem §6.3).

### 3.4. Feature schema (guide §3.2 → contract vocab)

Mapping `available_at` của guide ("bar_close") vào vocab `PatternFeature` (§3.4 spec): **confirm**.

```python
# NOTE: dtype theo vocab PatternFeature frozen (§2.1 contracts.py): "float"/"int"/"bool"/"str"
HMM_FEATURE_SCHEMA = [
    PatternFeature("hmm_state", "int",    AVAILABLE_AT_CONFIRM, False, "argmax filtered state tại bar đóng (known_at)"),
    PatternFeature("hmm_prob_trending",  "float", AVAILABLE_AT_CONFIRM, False, "filtered posterior trending"),
    PatternFeature("hmm_prob_sideways",  "float", AVAILABLE_AT_CONFIRM, False, "filtered posterior sideways"),
    PatternFeature("hmm_prob_high_vol",  "float", AVAILABLE_AT_CONFIRM, False, "filtered posterior high_vol"),
    PatternFeature("hmm_confidence",     "float", AVAILABLE_AT_CONFIRM, False, "max state probability"),
]
```

- State lookup của một event: dùng bar đóng **ngay tại/trước `known_at`** (ưu tiên `attributes["confirm_bar"]` khi có — DB/DT có; LSW dùng `confirmation_time` → bar index). `stamp(state) == df.index[bar] <= known_at` → luôn causal.
- `hmm_prob_<name>` được sinh động theo `state_names` trong config (n_state 2 → 2 cột prob + state + confidence).
- Mọi feature `uses_future_data=False`; schema này phải được `validate_causality` chấp nhận nguyên batch event (§3.4 spec).

---

## 4. Feature Emitter — plug/unplug an toàn (guide §4 — quan trọng nhất)

### 4.1. Nguyên tắc (guide §4.1)

- Mọi model đã train lưu **exact feature_list** + `feature_schema_version` trong artifact — **đã có sẵn** trong hệ thống:
  - `model.pkl` envelope `["feature_names"]` (walkforward_trainer §write_model_artifacts),
  - `features.json`/`feature_schema.json` cạnh artifact.
- Feature HMM = **optional**: model cũ không có `hmm_*` trong feature_list → chạy y nguyên, plugin active hay không không đổi kết quả.
- **Nguyên tắc vàng:** không bao giờ tự ý thêm/bớt feature so với lúc train — enforce bằng `X.reindex(columns=feature_names)` (scorer runner đã làm), thêm test: cột thừa phải bị loại.

### 4.2. Emitter contract

```python
class HMMFeatureEmitter:
    def __init__(self, plugin: BaseRegimePlugin, config: dict): ...
    def attach(self, events: list[PatternEvent], states_by_bar: list[RegimeState],
               df: pd.DataFrame) -> None:
        """Ghi vào event.attributes: hmm_state, hmm_prob_*, hmm_confidence
        (state tại confirm bar / bar ≤ known_at). KHÔNG đọc bar > known_at."""
    def feature_frame(self, df: pd.DataFrame, events: list[PatternEvent]) -> pd.DataFrame:
        """Ma trận hmm_* per event_id (index=event_id) cho training —
        cột = get_feature_names(), cùng semantics như attach()."""
```

- **Training (guide §4.2A):** `walkforward_trainer.build_feature_frame` mở rộng **không phá vỡ signature cũ**: thêm tham số optional `hmm_frame: pd.DataFrame | None = None` + `hmm_feature_names: list[str] = ()`. Khi `hmm.enabled=true` trong config training → gọi `emitter.feature_frame` và **append cột hmm_* vào cuối** X + feature_names (sau `extra_attr_features`). Khi false → zero thay đổi, model artifact cũ y nguyên.
- **Live / backtest scorer:** KHÔNG sửa `make_tier2_scorer` logic lõi — plugin chạy **trước** scorer trong cùng assignment; `${regime_plugin}.predict()` chạy 1 lần per (symbol, timeframe), cache trong scan; emitter `attach()` ghi attributes; `build_feature_frame` đọc `hmm_*` từ attributes khi cần. Scorer reindex theo `feature_names` của model → model cũ (không hmm) tự loại cột thừa, model mới (có hmm) nhận đủ.

### 4.3. MultiPatternEngine integration (guide §5/§9.1)

```python
@dataclass
class PatternAssignment:  # EXTEND (giữ mọi field cũ + default → zero break)
    ...
    regime_plugin: BaseRegimePlugin | None = None   # None = pattern không dùng HMM
    regime_config: dict = field(default_factory=dict)
    regime_filter: dict | None = None               # hard gate rules (§8), None = tắt
```

- `MultiPatternEngine.__init__`: nhận thêm `regime_plugin` dùng chung (hoặc per-assignment). `check_new_bar()`:
  1. Với mỗi (symbol, TF) có assignment cần regime → `states = plugin.predict(df, cfg)` (1 lần, bỏ cache).
  2. `_run_assignment`: sau `detect` + `validate_causality`, gọi `emitter.attach(events, states, df)` — chỉ khi assignment có regime_plugin.
  3. Scorer chạy sau attach (nhìn feature_names).
- **Old-model immutability test bắt buộc:** chạy MultiPatternEngine với cùng dữ liệu, assignment có `regime_plugin=None` → danh sách `SignalCandidate` **bit-identical** trước/sau khi thêm plugin vào engine (chỉ khác khi assignment chủ động bật HMM).

### 4.4. Registry §4.3 → §5.5 schema (model_registry/index.yaml)

Mở rộng metadata bắt buộc (guide §4.3):

```yaml
- model_id: "double_bottom_xauusd_m15_v1_hmm"
  feature_schema_version: "double-v1.1"        # bump khi thêm hmm_* vào feature_list
  feature_list:                                # EXACT train-time list (mirror features.json)
    - depth_atr
    - ...
    - rule_score
    - hmm_state
    - hmm_prob_trending
    - hmm_prob_sideways
    - hmm_prob_high_vol
    - hmm_confidence
  optional_plugins:
    - name: "hmm_regime"
      version: "1.0.0"
      required: true                            # true = model PHỤ THUỘC HMM
```

- `ModelInfo` (live/state/shared_app_state_v2.py) **mở rộng** (chỉ thêm field default, không phá):
  - `feature_list: list[str] = field(default_factory=list)` — khi YAML không khai báo → tự resolve từ `features.json` của artifact tại load (back-compat với 8 entry hiện có).
  - `optional_plugins: list[dict] = field(default_factory=list)`.
- **GUI onboarding (§10.1):** filter dropdown không đổi cho model cũ; model `required=true` chỉ assignable khi symbol config đã bật hmm plugin (GUI hiển thị badge "HMM" nếu optional_plugins không rỗng — nice-to-have, không chặn).
- Entry hiện có: `feature_list` được điền tự động từ features.json; **không sửa lifecycle_state / gate_passed / metrics** của 8 model đang có (7 model + 1 alias retired `xauusd_v2`).

---

## 5. Hard Gate / Regime Filter (guide §8 — tùy chọn, độc lập model)

```yaml
# configs/plugins/hmm_regime.yaml
regime_filter:
  enabled: true
  rules:                      # per pattern_name
    double_bottom:
      allowed_states: ["trending", "sideways"]
      min_confidence: 0.60
    liquidity_sweep:
      allowed_states: ["trending", "high_vol"]
      min_confidence: 0.55
```

Contract:

```python
def is_allowed(event: PatternEvent, regime: RegimeState | None,
               rules: dict) -> tuple[bool, str]:
    """→ (allowed, reason). reason ∈ {"no_rule", "allowed",
    "state_not_allowed", "low_confidence", "no_regime"}.
    rules = {} → (True, "no_rule"). regime=None → (False, "no_regime")."""
```

- Vị trí tích hợp **giống hệt live ≡ backtest** (spec §12):
  - Live: trong `MultiPatternEngine.check_new_bar`, sau CorrelationManager.group, **trước** `_group_to_candidate` — chỉ áp lên representative của group live/shadow; reject → set `rep.attributes["discard_reason"] = "regime_blocked"` + Event Lake vẫn ghi (spec §7.1), KHÔNG tạo candidate.
  - Backtest: trong `runner.run_symbol_backtest`, tại vòng replay group, trước `simulate_trade` — cùng hàm `is_allowed` import từ `research/regime/hard_gate.py`. Parity test assert hai đường dùng chung hàm.
- Hard gate **không** đụng feature vector / model_prob → model cũ chắc chắn không đổi (guide §8).
- Fail-closed mặc định khi `regime_filter.enabled=true` nhưng assign chưa có regime state (plugin inactive) → pattern đó bị gate chặn (`no_regime`), log WARN — tránh "bật gate mà tắt plugin" âm thầm.

---

## 6. Các quy tắc plug/unplug chi tiết khi inference (guide §4.2B)

### 6.1. Model cũ (feature_list không hmm_*) + plugin active
- Scorer reindex → không thấy cột hmm → bỏ qua. Emitter attach attributes nhưng không ai đọc. **Kết quả y nguyên** — phải có test.

### 6.2. Model mới (feature_list có hmm_*) + plugin active
- Emitter phải tính được đủ `hmm_*` cho mọi event (nếu thiếu bar → NaN → `fillna(0.0)` theo NA_FILL contract train/live — giống `_compute_model_prob`). Nếu plugin có mà predict fail → scorer trả `{}` → `model_prob=None` → candidate fallback `0.5` như hiện tại + `discard_reason="hmm_unavailable"` ghi Event Lake, KHÔNG crash symbol scan.

### 6.3. Model mới (feature_list có hmm_*) + plugin INACTIVE / chưa assign
- **Fail-closed:** assignment đó KHÔNG được emit signal (scorer không có cột hmm → reindex toàn NaN → model_prob nghĩa đen vô nghĩa). Engine log WARN + GUI Log tab. Đây là enforcement của guide §4.2C ("bắt buộc phải có HMM plugin active").

### 6.4. Enforce bằng test
1. `test_old_model_unchanged_with_plugin`: dataset cũ + model cũ → probs bit-identical có/không plugin.
2. `test_new_model_requires_plugin`: feature_list hmm + plugin None → assignment fail-closed.
3. `test_extra_columns_dropped`: X có hmm cột thừa nhưng feature_names không có → kết quả == không có cột thừa.

---

## 7. No-Lookahead checklist HMM (guide §7 → test matrix bắt buộc)

| # | Yêu cầu | Test |
|---|---|---|
| 1 | `predict()` chỉ dùng dữ liệu ≤ bar hiện tại | `test_predict_causal_known_at`: state tại t không đổi khi xóa khối dữ liệu SAU t (so sánh từng state) |
| 2 | Không fit full-history rồi map ngược | `test_no_refit_on_predict`: `predict()` không làm thay đổi tham số (fit chỉ qua `fit()`); mutate params sau fit → đoán được qua state change |
| 3 | `RegimeState.timestamp` = bar close | `test_state_timestamp_is_bar_close` |
| 4 | Incremental == full | `test_incremental_equals_full`: per-bar predict == full predict (toàn bộ bars) |
| 5 | Synthetic future-data injection bị chặn | `test_adversarial_future_injection` (suite riêng `test_hmm_adversarial.py`): x_shift(-1), close của confirm bar+1, timeline attack → predict phải raise hoặc state giống causal |
| 6 | Lag khai báo + test impact | `test_lag_declared_and_impact`: `lag_bars>0` → timestamp dịch đúng, report ghi ảnh hưởng PF |
| 7 | Feature schema `uses_future_data=False` | kế thừa `NoLookaheadTestBase` (§3.4) cho test_hmm_regime |
| 8 | Artifact lưu đúng feature_list | round-trip test features.json == feature_names envelope == registry feature_list |
| 9 | Reproducible | cùng seed + config → cùng state sequence (toàn frame); config_hash 12 hex |
| 10 | Causal fit/OOS split | fit window kết thúc TRƯỚC OOS start + purge/embargo margin (§6.3 spec) — test assert không overlap |

**Suite phải mang marker `no_lookahead`** (pyproject) và kế thừa `tests/no_lookahead_base.py::NoLookaheadTestBase` ở phần schema/event causality.

---

## 8. OOS Test Protocol — LSW / DB / DT (t5)

### 8.1. Dữ liệu & window (nhất quán với bằng chứng OOS hiện có)

- Dữ liệu: `research/multi_backtest/data/XAUUSD_m15.parquet` (2018-01-02 09:00 → 2026-09-03 22:45 UTC, 204,133 bars).
- **OOS window chính (decision window):** `2023-10-12 04:45 → 2026-09-03` (giống `docs/oos_edge_lsw_db.md`, `oos_lsw_model_filter.py`).
- **Sanity window phụ:** `2020-01-01 → 2022-06-30` (legacy-ish; tránh chạm 2018–2019 vì cần fit prefix) — chỉ dùng để kiểm tra tính ổn định, KHÔNG phải decision.

### 8.2. Causal fit split (bắt buộc, chống lookahead)

- **Train prefix (fit HMM):** `2018-01-02 → 2023-09-30` (cho OOS chính) — kết thúc ≥ 11 ngày trước window start + **purge/embargo margin** (spec §3.1/§6.3: không overlap forward window của bất kỳ event nào trong OOS). **Effective margin đo được trên XAUUSD M15 thật:** `2023-09-30 23:45 → 2023-10-12 04:45` = **751 bars** (weekend-gapped calendar; ≥ 1000 bars KHÔNG đạt được cho split này — t5 phải dùng effective margin 751 bars và báo rõ trong report; khoảng ≥ 11 ngày dương lịch vẫn là điều kiện bắt buộc, và cấm tuning margin trên OOS).
- **Gate tuning:** `allowed_states` / `min_confidence` phải được chọn trên **train prefix** (đo per-state PF của prefix), **cấm tuning trên OOS** — report ghi rõ threshold fixed a priori + số liệu chọn. Đây là rule chống look-ahead quan trọng nhất của protocol.
- Sanity window: fit prefix `2018-01-02 → 2019-12-31`, eval `2020-01-01 → 2022-06-30` (strictly after fit).

### 8.3. Ma trận so sánh (mỗi pattern × window)

| Variant | Mô tả |
|---|---|
| `baseline` | Không gate — chạy nguyên `run_symbol_backtest` (LSW rule-only / DB & DT attach tier-2 models) |
| `gate_state` | Hard gate: chỉ `allowed_states` (min_confidence=0.0) |
| `gate_state_conf` | Hard gate: `allowed_states` + `min_confidence` từ config (sweep 0.0/0.55/0.70) |
| `gate_top_state` | (chẩn đoán) chỉ giữ state có edge cao nhất đo trên train prefix |

Metrics per variant (đủ §12 report): `n_events`, `n_trades`, `PF` (post-cost, CostConfig XAUUSD 0.7/0.5/1.0 bps/side), `totalR`, `expectancy R/trade`, `maxDD`, `win_rate`, **selection power** (PF theo từng state bucket — như `oos_lsw_model_filter.py` halves pattern), rolling PF.

### 8.4. Quyết định (verdict rules — t5 phải báo cáo trung thực, không ép pass)

Cho mỗi pattern trên **OOS window chính**:
- **IMPROVED:** gate (state_conf) có `PF_gated > PF_baseline` VÀ `expectancy_gated > expectancy_baseline` VÀ `n_trades_gated ≥ 30` (sample-size gate tinh thần §6.3) — kèm nêu rõ state nào mang edge.
- **NEUTRAL:** PF chênh lệch ≤ 0.05 hoặc n quá nhỏ (< 30) — báo cáo "không đủ bằng chứng".
- **HARMFUL:** `PF_gated < PF_baseline − 0.10` — báo "gate làm hỏng", không deploy.
- **Bắt buộc so sánh với mục tiêu kinh doanh LSW:** baseline LSW post-cost PF ~0.96 (rule-only) — test giả thuyết "regime gate đưa LSW lên PF ≥ 1.05 hậu cost"; nếu không đạt → kết luận trung thực "HMM regime filter không cứu LSW" (giống kết luận model filter hướng (a) trước đây — CI + n phải đủ).
- Decision report file: `research/multi_backtest/reports/oos_hmm_regime/{pattern}_{window}_{variant}.{md,json}` + bảng tổng hợp.

### 8.5. Feature-emitter OOS (secondary, nếu t5 đủ thời gian — KHÔNG chặn t9)

Train 1 model DB (và DT nếu được) có hmm features trên train prefix → so sánh OOS PR-AUC / model-gated PF vs model cũ không hmm. Đây là bằng chứng trực tiếp của guide §9.6 nhưng **không mandatory** cho integration gate (t9) — đánh dấu optional trong báo cáo.

---

## 9. File config mẫu (guide §6 → trading_v3)

`research/configs/plugins/hmm_regime.yaml`:

```yaml
plugin:
  name: hmm_regime
  version: "1.0.0"
  enabled: false                       # cờ tổng; bật cùng feature emitter

hmm:
  n_states: 3
  state_names: ["trending", "sideways", "high_vol"]
  input_features: ["log_return_1", "atr_14_norm", "volume_zscore_20"]
  lag_bars: 0
  min_confidence: 0.50
  covariance_type: "full"
  n_iter: 100
  random_state: 42
  min_fit_bars: 2000                   # guard fit
  min_predict_bars: 200                # guard predict

regime_filter:
  enabled: false
  rules: {}                            # §5 — từng pattern, KHÔNG tự suy từ train
```

- `get_default_config()` của plugin ≤ config YAML này; mọi semantic change → bump `plugin.version` (không âm thầm đổi nghĩa, §6.2).

---

## 10. Acceptance criteria (DoD — guide §9, spec §15, spec §13 CI)

### A. Plugin + causal HMM (t2)
1. Import thành công trên venv hiện tại — **không import hmmlearn**, không dependency mới (verify bằng `pip freeze` diff rỗng / grep imports).
2. `CausalGaussianHMM` full-featured: fit/predict/get_feature_schema/get_default_config; `predict` causal forward-filter; reproducible theo seed; config_hash 12 hex.
3. Feature schema `uses_future_data=False` và `validate_causality` pass.
4. Test matrix §7 (10 items) xanh, kế thừa NoLookaheadTestBase, marker `no_lookahead`; adversarial suite (`test_hmm_adversarial.py`) bắt được injection.
5. ruff + mypy strict sạch trên `research/regime/` (theo pyproject — mypy strict scope giống runner: `research/regime` strict-clean).

### B. Integration + emitter + registry + hard gate (t3)
6. `build_feature_frame` mở rộng backward-compatible (cũ → kết quả y nguyên; tests hiện có không đổi).
7. Old-model immutability: `test_old_model_unchanged_with_plugin` + regression suite hiện hữu (DB/DT/LSW golden, runner parity) vẫn xanh.
8. Registry §4.3: ModelInfo mở rộng, index.yaml load back-compat (8 entry cũ resolve feature_list từ features.json), entry HMM mẫu được parse + GUI filter không đổi.
9. Hard gate live ≡ backtest qua shared `is_allowed`; parity test; `discard_reason="regime_blocked"` vào Event Lake.
10. Fail-closed §6.3 đúng (model hmm required + plugin inactive → không signal).

### C. OOS test (t5)
11. Script `oos_hmm_regime.py` chạy được ≤ 30 phút trên XAUUSD M15 full frame; report đủ §8.3 metrics + selection power.
12. Causal fit split assert (fit ⊂ trước OOS, purge/embargo margin); threshold tuning chỉ trên train prefix (ghi trong report).
13. Verdict LSW/DB/DT per §8.4 — trung thực, có n ≥ 30 cho kết luận IMPROVED.

### D. Integration gate (t9)
14. CI xanh: ruff + mypy (strict cho research/regime + research/multi_backtest) + pytest full + `no_lookahead` marker + adversarial suite (spec §15.6).
15. Golden LSW bit-identical giữ nguyên (spec §15.1) — không có bất kỳ thay đổi nào làm đổi event LSW.
16. Báo cáo HMM regime verdict LSW/DB/DT (IMPROVED/NEUTRAL/HARMFUL) + docs cập nhật; Registry §4.3 fields có mặt trong index.yaml schema header.

---

## 11. Out of scope (rõ ràng để tránh scope creep)

- Không retrain/sửa artifact của 7 model hiện có (chỉ registry metadata back-compat).
- Không sửa `research/core/contracts.py`, `causal_checks.py`, `config_hash.py`.
- Không thêm dependency mới (không hmmlearn, không statsmodels, không pytorch).
- Không chuyển HMM sang dùng làm pattern detector (Environment/Regime Engine only, guide §1).
- Không bắt buộc GUI badge HMM (nice-to-have) cho gate t9.
- Confluence meta-model (§6.4 spec) ngoài phạm vi — HMM feature chỉ là optional feature của tier-2 hiện hữu.

---

## 12. Task mapping (từ contract này)

| Task | Nhận từ contract |
|---|---|
| t2 (regime-engineer) | §3 (interfaces), §3.4 (schema), §7 (test matrix), A1–A5 |
| t3 (integrator) | §4 (emitter/plug-unplug), §4.3/§4.4 (engine + registry), §5 (hard gate), §6 (enforce), B6–B10 |
| t4 (verifier) | §7 full matrix + adversarial + old-model immutability; CI doD phần A/B |
| t5 (experimenter) | §8 OOS protocol LSW/DB/DT + verdict rules; C11–C13 |
| t6/t7/t8 (reviewer) | Đối chiếu acceptance A/B/C theo từng task |
| t9 (gate-keeper) | §10 D14–D16 integration gate |

*Contract này là single source of truth cho t2–t9; mọi thay đổi phải qua captain + review.*