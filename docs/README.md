# 📊 Paper Trading V2 — Hệ thống Trading Tự động

Hệ thống Paper Trading V2 là GUI desktop kết nối với MetaTrader 5 qua MCP (Model Context Protocol), cho phép quản lý symbol, gán model ML, theo dõi tín hiệu real-time và đặt lệnh tự động.

---

## 🏗️ Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                     GUI (5 tabs)                             │
│  [Onboarding] [Live Control] [Performance] [Account] [Log]   │
├─────────────────────────────────────────────────────────────┤
│                       SystemBridge                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Signal   │ │ RiskGuard│ │Execution │ │  Model        │  │
│  │ Engine   │ │          │ │ Layer    │ │  Registry     │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     MCP Server (MT5)                         │
│              http://127.0.0.1:22346/mcp                      │
└─────────────────────────────────────────────────────────────┘
```

Hệ thống gồm **5 tab** trên GUI, kết nối với **MCP/MT5** để nhận dữ liệu thị trường real-time. Chỉ symbol có `status=validated` + `model_id` hợp lệ mới được Signal Engine xử lý.

---

## 📋 Tab 1: Symbol Onboarding

**Mục đích:** Quản lý symbol và gán model từ Model Registry

**Luồng thêm symbol mới:**
1. Click **+ Add New Symbol**
2. Dialog hiện ra với:
   - **Dropdown Symbol**: chọn từ MCP market watch (BTCUSDm, XAUUSDm, EURUSDm...)
   - **Dropdown Model**: chọn model từ Model Registry (hoặc Browse local)
   - **Initial status**: validated / candidate / rejected
3. Symbol xuất hiện trong bảng 6 cột:
   - **Symbol** — Tên (ví dụ: XAUUSDm)
   - **Status** — validated (xanh) / candidate (vàng) / rejected (đỏ)
   - **Giá (Bid)** — Giá thị trường real-time từ MCP
   - **Model đang dùng** — model_id đã gán
   - **Active** — ✅ Yes / ❌ No
   - **Actions** — Các nút thao tác

**Actions trên mỗi dòng:**

| Nút | Chức năng | Ghi chú |
|-----|-----------|---------|
| 🔧 Model | Đổi model | Chỉ hiện khi status=validated |
| Activate | Bật runtime | Chỉ khi validated + có model_id |
| Deactivate | Tắt runtime | Giữ nguyên status |
| Status | Đổi validated/candidate/rejected | Nếu chuyển khỏi validated → auto-deactivate |
| 🗑 Remove | Xoá khỏi registry | Cần gõ REMOVE để xác nhận |

**Quy tắc an toàn:**
- Chỉ `validated` + có `model_id` → mới được Activate → Signal Engine mới chạy
- `rejected` không thể Activate
- Tất cả thay đổi đều persist (sống qua restart) và ghi vào System Log

---

## 🎮 Tab 2: Live Control

**Mục đích:** Theo dõi và điều khiển live trading

**Dropdown Symbol:** Chỉ hiện symbol có `status=validated` (từ YAML config hoặc đã onboard qua GUI)

**Automation Level (3 mức):**
- **Level 1 — Manual Confirm**: Xác nhận từng lệnh qua GUI
- **Level 2 — Semi-Auto**: Agent tự động gửi lệnh sau delay
- **Level 3 — Full Auto**: Gửi lệnh ngay, không cần review

**Bảng Pending Signals:**
- Hiển thị tín hiệu từ Signal Engine cho symbol đang chọn
- Mỗi dòng: Asset | Hướng (BUY/SELL) | Entry | Stop Loss | Take Profit | Rule Score | Model Prob | [Send Order]
- Click **Send Order** → confirmation → gửi lệnh market qua MCP

**Bảng Open Positions:**
- Vị thế đang mở: Asset | Hướng | Entry | Giá hiện tại | Size | P/L (R) | Open Time | [Close]
- Click **Close** → confirmation (gõ CLOSE) → đóng vị thế

⚠️ Mọi hành động đều cần confirmation dialog

---

## 📈 Tab 3: Performance Monitor

**Mục đích:** Theo dõi hiệu suất và rủi ro của từng symbol

**Các chỉ số:**
- Profit Factor, Sharpe Ratio, Drawdown
- Số lượng giao dịch, win rate
- Kill-switch status (block bootstrap)

**Kill-Switch:**
- Tự động kích hoạt khi PF CI lower < ngưỡng
- Khi active → chặn mọi lệnh mới cho symbol đó
- Có thể reset thủ công

---

## 🔌 Tab 4: Account & Connection

**Mục đích:** Quản lý kết nối MCP/MT5 và thông tin tài khoản

**Kết nối MCP:**
- Token từ env `MCP_TOKEN` hoặc nhập tay (ô **Enter new MCP token** → **Update Token**)
- Click **Connect / Reconnect MCP** để kết nối
- Trạng thái: ✅ Connected / ❌ Disconnected / 🔄 Reconnecting

**Thông tin tài khoản MT5 (sau khi connect):**
- Account Number | Type (demo/real) | Server
- Balance | Equity | Margin | Free Margin
- Leverage | Currency

**An toàn:**
- Nếu phát hiện tài khoản REAL → cảnh báo đỏ toàn màn hình, chặn mọi thao tác
- Phải xác nhận mới có thể tiếp tục

---

## 📋 Tab 5: System Log

**Mục đích:** Xem real-time mọi hoạt động của hệ thống

**Định dạng mỗi dòng:**
```
[TIMESTAMP] [LEVEL] [SOURCE] [SYMBOL] Message
```

Ví dụ: `2026-09-05 21:45:00.123 [INFO] [SignalEngine] [XAUUSD] New M15 candle closed @ 2650.45 → running detection`

**Level và màu sắc:**

| Level | Màu | Ví dụ |
|-------|-----|-------|
| DEBUG | Xám | Bắt đầu poll, đọc OHLC, feature raw |
| INFO | Trắng | Nến mới, detection, signal, order sent/filled |
| WARNING | Vàng | Reconnect, data delay, score thấp |
| ERROR | Đỏ | Lỗi kết nối, model load fail, exception |
| CRITICAL | Đỏ đậm | Emergency Stop, kill-switch |

**Bộ lọc:**
- **Level**: multi-select checkbox (mặc định ẩn DEBUG)
- **Symbol**: dropdown (All hoặc chọn 1 symbol)
- **Source**: SignalEngine / RiskGuard / Execution / MCP / System / UserAction
- **Time range**: Last 5 min / 15 min / 1 hour / Today / Custom
- **Search text**: tìm kiếm trong message
- **Nút Clear filters**: reset tất cả
- **Nút Clear log**: xoá log (cần gõ CLEAR)

**Tính năng:**
- **Auto-scroll**: tự động xuống dòng mới nhất (có nút Pause)
- **Click vào dòng**: xem chi tiết JSON (extra context)
- **Export**: CSV hoặc TXT (chọn qua QFileDialog)
- **Giới hạn**: 10.000 dòng gần nhất (thread-safe ring buffer)

---

## 🔁 Luồng dữ liệu tổng thể

```
MCP Server (MT5)
    │
    ▼
Execution Layer ─── tự động reconnect nếu mất kết nối
    │
    ▼
SystemBridge ─── poll mỗi 1 giây: health → account → positions → symbols
    │
    ├──▶ SharedAppState (trạng thái toàn cục, thread-safe)
    ├──▶ Signal Engine (khi nến M15 mới)
    │       ├── Load model từ ModelRegistry (model_id → model.pkl + calibrator)
    │       ├── Detect sweep events
    │       ├── Tính score, sinh signal
    │       └── Log vào System Log
    ├──▶ RiskGuard (kiểm tra kill-switch trước mỗi lệnh)
    └──▶ GUI (5 tabs, QTimer cập nhật mỗi giây)
            └── System Log (ring buffer, poll 500ms)
```

**Chi tiết từng bước:**
1. **MCP Server** cung cấp: giá bid/ask, nến M15, tài khoản, vị thế, market watch symbols
2. **ExecutionLayer** gọi MCP tools, tự động reconnect nếu session expired, position poll 30s
3. **SystemBridge** là trung gian: fetch data → update state → dispatch signal engine
4. **Signal Engine** chạy khi nến M15 mới đóng: load model từ ModelRegistry theo `model_id`, chạy detection, tính score, tạo pending signal
5. **RiskGuard** đánh giá rủi ro: kill-switch (block bootstrap PF CI), nếu pass mới cho gửi lệnh
6. **Execution Layer** đặt lệnh market với magic=20791, comment LSW-V2-{event_id}, theo dõi fill
7. **GUI** hiển thị real-time: 5 tabs, System Log từ ring buffer, Emergency Stop luôn sẵn sàng

---

## 🛠️ Thông tin kỹ thuật

| Hạng mục | Giá trị |
|----------|---------|
| 🌐 **MCP Server** | http://127.0.0.1:22346/mcp (Streamable HTTP) |
| 🔑 **Token** | Env var `MCP_TOKEN` hoặc nhập trong GUI Tab Account |
| 📁 **Model Registry** | `trading_live/model_registry/index.yaml` |
| 💾 **Symbol Registry** | `trading_live/live/db/symbol_registry.json` (persist) |
| 🗄️ **Database** | `trading_live/live/db/paper_trading_v2.db` |
| 🔬 **Pipeline nghiên cứu** | `trading_live/research/` (xauusd-liquidity-sweep) |
| 🚀 **Chạy GUI** | `cd trading_live && export MCP_TOKEN=... && export PYTHONPATH=$PWD && /tmp/ptv2_venv/bin/python3 -m live.gui.gui_main` |

---

## 🗂️ Cấu trúc thư mục

```
trading_live/
├── live/                          # Hệ thống vận hành
│   ├── gui/                       # GUI (5 tabs + bridge)
│   ├── engine/                    # Signal Engine, RiskGuard
│   ├── mcp/                       # MCP client
│   ├── logging/                   # Logger
│   ├── state/                     # SharedAppState, ModelRegistry
│   └── db/                        # SQLite database, persist files
│
├── research/                      # Pipeline nghiên cứu
│   └── xauusd-liquidity-sweep/    # Source code pipeline
│
├── artifacts/                     # Model artifacts
│   └── models/
│       ├── EURUSD/                # EURUSD model (train riêng)
│       └── XAUUSD/                # XAUUSD model (từ research)
│
├── model_registry/                # Model metadata
│   └── index.yaml                 # Danh sách tất cả models
│
├── docs/                          # Tài liệu
│   └── decisions/                 # Quyết định đóng băng theo mốc
│
└── archive/                       # Code cũ đã archive
    └── README.md                  # Giải thích từng phần archive
```