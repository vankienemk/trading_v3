"""
gui_tab_live_control.py — Tab 2: Live Control

Spec section 7.2 Tab 2:
  - Dropdown per symbol (only validated symbols)
  - Automation level switch (0/1/2) per symbol, displayed prominently
  - Per-symbol signal statistics panel (Trigger / Found / Pass), live-updated
    from SharedAppState counters with per-symbol and global reset
  - Pending signals table with Send Order / Skip buttons per row (Level 0 = manual)
  - Open positions table with P/L in R
  - Emergency Stop button always visible (rendered in gui_main.py, not here)

All controls are immediate-response (no web reload).
Every destructive action needs a second confirmation dialog.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, ClassVar

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from live.gui.chart_widget import CandlestickChart
from live.gui.gui_bridge import SystemBridge
from live.gui.gui_components import (
    COLOR_BG,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_SEC,
    COLOR_WARNING,
    ConfirmationDialog,
    fmt_time,
)

# Manual confirmation signals older than this many seconds are treated as
# stale: the "Send Order" button is disabled (and dispatching is refused) so
# we never act on a manual confirmation that is over an hour old.
MAX_MANUAL_SIGNAL_AGE_S = 3600


def _signal_epoch_s(sig: dict[str, Any]) -> float:
    """Return the epoch (unix, seconds) that represents when a signal became
    stale-able -- i.e. the OLDEST real clock the signal carries.

    GUI surfaces show the order's age by its **Entry Time** (bar open carried
    end-to-end from the engine).  On a long-running live session an old signal
    row survives and merely gets re-listed/rescanned with a *recent* ``find``
    timestamp (``timestamp`` = when the engine last emitted it, could be
    minutes old) while its ``entry_time`` is hours/days in the past.  A stale
    guard keyed only off that recent find-time therefore left an hours-old
    order with Send Order still enabled -- the reported bug.  Age is measured
    from the EARLIEST of the recorded signal clocks (entry bar time, then find
    time).  Tolerates int / float / numeric-string values and a naive
    (non-epoch) or millisecond epoch defensively.

    Returns ``0.0`` when no usable clock exists (caller then treats the signal
    as not-ageable, not stale, preserving old/schema-less behaviour).
    """
    raw = 0.0
    for source_key in ("entry_time", "timestamp"):
        v = sig.get(source_key, 0)
        if v in (None, ""):
            continue
        try:
            ts = float(v)
        except (TypeError, ValueError):
            continue
        if not ts > 0:
            continue
        now = time.time()
        # Milliseconds epoch (e.g. ~1.7e12 far above the current ~1.7e9 wall
        # clock) -- normalise down before comparing.
        if ts > now * 100:
            ts /= 1000.0
        if not ts > 0:
            continue
        raw = ts if raw == 0.0 else min(raw, ts)
    return raw


def _signal_age_s(sig: dict[str, Any]) -> float:
    """Age in seconds of the signal from its oldest recorded signal clock.

    ``0.0`` signals a signal with no usable timestamp (not considered stale).
    """
    epoch = _signal_epoch_s(sig)
    if epoch <= 0:
        return 0.0
    return max(0.0, time.time() - epoch)


def _is_stale_signal(sig: dict[str, Any]) -> bool:
    """True when the signal should be treated as too old to act on."""
    age = _signal_age_s(sig)
    return age > 0 and age > MAX_MANUAL_SIGNAL_AGE_S


def _stale_reason(sig: dict[str, Any]) -> str:
    """Human-readable reason when a signal is stale, else ``""``."""
    if not _is_stale_signal(sig):
        return ""
    # Prefer reporting the age from the entry bar (the age the user reads on
    # the Entry Time column); fall back to whichever clock was used.
    et = _signal_epoch_s(sig)  # oldest clock
    age_min = int(_signal_age_s(sig) / 60)
    verdict = "older than 1 hour"
    if et > 0:
        try:
            from datetime import datetime as _dt
            verdict = f"entered over {age_min} min ago ({_dt.fromtimestamp(et).strftime('%Y-%m-%d %H:%M')})"
        except Exception:
            pass
    return (f"Signal {sig.get('signal_id', '')} is {verdict} "
            f"(max {int(MAX_MANUAL_SIGNAL_AGE_S / 60)} min) — Send Order refused.")


def _direction_label(direction: Any) -> str:
    """Map any direction spelling to the canonical display label.

    The signal engine emits ``"long"``/``"short"`` (see
    ``SignalCandidate.direction`` in signal_engine_v2.py), while order
    contracts use ``"buy"``/``"sell"``.  Rendering the raw value made the
    Pending Signals Dir column ALWAYS show SELL (because ``"long" != "buy"``)
    even though the row was a buy, and the inspect dialog showed
    ``LONG``/``SHORT``.  Normalise to a single canonical BUY/SELL label.
    """
    d = (direction if isinstance(direction, str) else str(direction or "")).strip().lower()
    if d in ("buy", "long", "bullish", "0"):
        return "🟢 BUY"
    if d in ("sell", "short", "bearish", "1"):
        return "🔴 SELL"
    return str(direction or "—").upper()



class LiveControlTab(QWidget):
    """Tab 2: Live Control — signals, positions, automation."""

    # Compact neutral button style for the statistics panel (dark theme).
    _RESET_BTN_QSS = f"""
    QPushButton {{
        background-color: #333333;
        color: {COLOR_TEXT};
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 3px 12px;
        min-height: 20px;
        font-weight: normal;
    }}
    QPushButton:hover {{
        background-color: #444444;
        border-color: {COLOR_TEXT_SEC};
    }}
    QPushButton:pressed {{
        background-color: #555555;
    }}
    QPushButton:disabled {{
        background-color: #2a2a2a;
        color: #666666;
        border-color: #3d3d3d;
    }}
    """

    def __init__(self, bridge: SystemBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._current_symbol: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        # --- Header ---
        header = QLabel("🎮 Live Control")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(header)

        desc = QLabel(
            "Monitor and control live trading. "
            "Only <b>validated</b> symbols appear in the dropdown."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_NEUTRAL}; padding-bottom: 4px;")
        layout.addWidget(desc)

        # --- Symbol selector + Automation level row ---
        selector_row = QHBoxLayout()
        selector_row.setSpacing(16)

        # Symbol dropdown
        selector_row.addWidget(QLabel("Symbol:"))
        self._symbol_combo = QComboBox()
        self._symbol_combo.setMinimumWidth(140)
        self._symbol_combo.currentTextChanged.connect(self._on_symbol_changed)
        selector_row.addWidget(self._symbol_combo)
        selector_row.addSpacing(20)

        # Automation level
        selector_row.addWidget(QLabel("Automation Level:"))
        self._auto_combo = QComboBox()
        self._auto_combo.addItems(["1 — Manual Confirm", "2 — Semi-Auto", "3 — Full Auto"])
        self._auto_combo.setMinimumWidth(180)
        self._auto_combo.currentIndexChanged.connect(self._on_auto_level_changed)
        selector_row.addWidget(self._auto_combo)
        selector_row.addStretch()

        layout.addLayout(selector_row)

        # --- Status info ---
        self._status_label = QLabel("Select a symbol above.")
        self._status_label.setStyleSheet(f"color: {COLOR_NEUTRAL};")
        layout.addWidget(self._status_label)

        # --- Per-symbol signal statistics (Trigger / Found / Pass) ---
        # Counters are recorded per symbol by the signal engine in
        # SharedAppState (signal_stats) — this panel only displays them.
        self._stats_group = QGroupBox("📊 Signal Statistics")
        stats_row = QHBoxLayout(self._stats_group)
        stats_row.setSpacing(12)

        self._stats_label = QLabel("Trigger: — | Found: — | Pass: —")
        self._stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_row.addWidget(self._stats_label)
        stats_row.addStretch()

        self._reset_stats_btn = QPushButton("Reset Counters")
        self._reset_stats_btn.setToolTip(
            "Clear Trigger / Found / Pass counters for the selected symbol"
        )
        self._reset_stats_btn.setEnabled(False)
        self._reset_stats_btn.clicked.connect(self._on_reset_stats)
        self._reset_stats_btn.setStyleSheet(self._RESET_BTN_QSS)
        stats_row.addWidget(self._reset_stats_btn)

        self._reset_all_stats_btn = QPushButton("Reset All")
        self._reset_all_stats_btn.setToolTip(
            "Clear Trigger / Found / Pass counters for every symbol"
        )
        self._reset_all_stats_btn.setEnabled(False)
        self._reset_all_stats_btn.clicked.connect(self._on_reset_all_stats)
        self._reset_all_stats_btn.setStyleSheet(self._RESET_BTN_QSS)
        stats_row.addWidget(self._reset_all_stats_btn)

        layout.addWidget(self._stats_group)

        # --- Split: Pending Signals (left) | Open Positions (right) ---
        split_row = QHBoxLayout()
        split_row.setSpacing(8)

        # LEFT: Pending Signals
        sig_group = QGroupBox("📡 Pending Signals (Level 1 — Manual)")
        sig_layout = QVBoxLayout(sig_group)
        self._signals_table = QTableWidget(0, 5)
        self._signals_table.setHorizontalHeaderLabels(
            ["Entry Time", "Asset", "Dir", "Entry", "check"]
        )
        # Pending Signals is a compact summary — the detailed columns (Stop
        # Loss / Take Profit / Rule Score / Model Prob) move into the inspect
        # popup opened by the row's "check" button.  The Time column holds a
        # 19-char "YYYY-MM-DD HH:MM:SS" timestamp so it is sized to its content
        # (ResizeToContents) and never wraps; Asset/Dir/Entry are short, so they
        # STRETCH to fill the pane (no dead gap on the right, which the previous
        # all-ResizeToContents layout left); the "check" column is Fixed to the
        # button width so the ✓ button sits inside its row.
        self._signals_table.horizontalHeader().setStretchLastSection(False)
        self._signals_table.horizontalHeader().setMinimumSectionSize(64)
        self._signals_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for _col in (1, 2, 3):
            self._signals_table.horizontalHeader().setSectionResizeMode(_col, QHeaderView.Stretch)
        self._signals_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Fixed)
        self._signals_table.setColumnWidth(4, 112)
        self._signals_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._signals_table.verticalHeader().setDefaultSectionSize(34)
        self._signals_table.setAlternatingRowColors(True)
        self._signals_table.verticalHeader().setVisible(False)
        self._signals_table.itemDoubleClicked.connect(self._on_signal_row_double_clicked)
        sig_layout.addWidget(self._signals_table)
        split_row.addWidget(sig_group, stretch=7)

        # RIGHT: Open Positions
        pos_group = QGroupBox("💼 Open Positions")
        pos_layout = QVBoxLayout(pos_group)
        # Compact 4-column layout (Entry Time / Lot / P/L (R) / Close) so the
        # key info is readable and the Close action is never clipped.
        self._positions_table = QTableWidget(0, 4)
        self._positions_table.setHorizontalHeaderLabels(
            ["Entry Time", "Lot", "Profit ($)", "Close"]
        )
        # Entry Time (col 0) holds a 19-char "YYYY-MM-DD HH:MM:SS" timestamp
        # (gui_components.fmt_time) so it is sized to its content (never elided
        # to "2026-09-05 ..."); Lot / Profit are short, so they STRETCH to fill
        # the pane; the Close column stays Fixed so its button is never pinched.
        self._positions_table.horizontalHeader().setStretchLastSection(False)
        self._positions_table.horizontalHeader().setMinimumSectionSize(64)
        self._positions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for _col in (1, 2):
            self._positions_table.horizontalHeader().setSectionResizeMode(_col, QHeaderView.Stretch)
        self._positions_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self._positions_table.setColumnWidth(3, 110)
        self._positions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._positions_table.setAlternatingRowColors(True)
        self._positions_table.verticalHeader().setVisible(False)
        self._positions_table.verticalHeader().setDefaultSectionSize(34)
        pos_layout.addWidget(self._positions_table)
        split_row.addWidget(pos_group, stretch=5)

        layout.addLayout(split_row)

        # --- Refresh symbols on load ---
        self._refresh_symbol_dropdown()

    # ------------------------------------------------------------------
    # Symbol dropdown management
    # ------------------------------------------------------------------

    def _refresh_symbol_dropdown(self) -> None:
        """Re-populate the symbol dropdown with validated symbols."""
        current = self._symbol_combo.currentText()
        self._symbol_combo.blockSignals(True)
        self._symbol_combo.clear()

        symbols = self._get_validated_symbols()
        if symbols:
            self._symbol_combo.addItems(symbols)
            if current in symbols:
                self._symbol_combo.setCurrentText(current)
            else:
                self._symbol_combo.setCurrentIndex(0)
                self._current_symbol = symbols[0]
        else:
            self._symbol_combo.addItem("(no symbols)")
            self._current_symbol = ""

        self._symbol_combo.blockSignals(False)

    def _get_validated_symbols(self) -> list[str]:
        """Get list of validated symbol names from shared state."""
        # Construction/preview path: the tab may be instantiated without a
        # bridge (e.g. LiveControlTab(None) in headless verification); in that
        # case there is no state to read, so a validatable-but-empty result is
        # returned instead of crashing on None.state.
        if self._bridge is None:
            return []
        snap = self._bridge.state.get_snapshot()
        reg = snap.get("symbol_registry", {})
        return sorted([
            name for name, cfg in reg.items()
            if isinstance(cfg, dict) and cfg.get("status") == "validated"
        ])

    def _on_symbol_changed(self, symbol: str) -> None:
        """React to symbol selection change."""
        if not symbol or symbol == "(no symbols)":
            return
        self._current_symbol = symbol

        # Update automation level combo to current level
        snap = self._bridge.state.get_snapshot()
        auto_levels = snap.get("automation_levels", {})
        level = auto_levels.get(symbol, 1)
        self._auto_combo.blockSignals(True)
        # Combo items are 0-based ("1 — Manual Confirm" = index 0) while the
        # stored automation level is 1-based (1/2/3), so the index is `level - 1`.
        # Using the raw level previously showed the WRONG level (level 1 showed
        # "Semi-Auto") and, for a symbol at level 3 (Full Auto), setCurrentIndex(3)
        # fell outside the 0..2 range and rendered the combo BLANK — the
        # automation control appeared hidden/empty.
        self._auto_combo.setCurrentIndex(max(0, level - 1))
        self._auto_combo.blockSignals(False)

        self._status_label.setText(f"Selected: {symbol} | Automation: Level {level}")
        self._refresh_display(snap)

    def _on_auto_level_changed(self, index: int) -> None:
        """Handle automation level change — requires confirmation for level > 0."""
        if not self._current_symbol or self._current_symbol == "(no symbols)":
            return

        if index == 0:
            # Level 1 = Manual — no confirmation needed (default)
            self._apply_auto_level(index + 1)
        else:
            # Level 2 or 3 — require confirmation
            labels = {1: "Semi-Auto", 2: "Full Auto"}
            level = index + 1
            msg = (
                f"Change automation level for <b>{self._current_symbol}</b> "
                f"to <b>Level {level} — {labels.get(index, '?')}</b>?\n\n"
                f"{'Agent will send orders automatically after a delay.' if index == 0 else '⚠️ Orders placed entirely without manual review.'}"
            )
            dialog = ConfirmationDialog(
                title="Change Automation Level",
                message=msg,
                # The auto-level combo is 0-based (`index`), but the stored /
                # displayed automation level is 1-based (`level = index + 1`).
                # Using the raw `index` here made a "Level 3 — Full Auto" title
                # show a "Set Level 2" confirm button (off-by-one).
                confirm_text=f"✅ Set Level {level}",
                parent=self,
            )
            if dialog.exec():
                self._apply_auto_level(level)

    def _apply_auto_level(self, level: int) -> None:
        """Apply the automation level change."""
        self._bridge.state.set_automation_level(self._current_symbol, level)
        self._bridge.log_action("change_automation_level", {
            "symbol": self._current_symbol,
            "new_level": level,
        })
        self._status_label.setText(
            f"Automation for {self._current_symbol} set to Level {level}"
        )

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Called periodically by the main window timer."""
        snap = self._bridge.state.get_snapshot()
        self._refresh_symbol_dropdown()
        self._refresh_display(snap)

    def _refresh_display(self, snap: dict[str, Any]) -> None:
        """Refresh signals and positions tables from snapshot."""
        # Statistics panel always refreshes (shows "—" while no symbol is
        # selected / recorded); tables need a valid symbol.
        self._refresh_stats(snap)

        if not self._current_symbol or self._current_symbol == "(no symbols)":
            return
        symbol = self._current_symbol

        # Pending signals filtered by symbol
        pending = [s for s in snap.get("pending_signals", [])
                   if s.get("asset", "").upper() == symbol.upper()]
        self._populate_signals_table(pending)

        # Open positions filtered by symbol
        positions = [p for p in snap.get("open_positions", [])
                     if p.get("asset", "").upper() == symbol.upper()]
        self._populate_positions_table(positions)

    def _populate_signals_table(self, signals: list[dict]) -> None:
        """Fill the pending signals table (compact 5-col summary)."""
        self._signals_table.setRowCount(len(signals))

        for row, sig in enumerate(signals):
            ts = sig.get("timestamp", 0) or 0
            # Column 0 shows the signal's TRUE entry time (entry-bar open,
            # carried end-to-end from the engine).  Signals recorded before
            # entry_time existed fall back to the scan/found time with a tooltip
            # that says so — never a misleading blank value.
            et_raw = sig.get("entry_time", 0) or 0
            try:
                et = float(et_raw)
            except (TypeError, ValueError):
                et = 0.0
            shown_time = fmt_time(et) if et > 0 else fmt_time(ts)
            entry_label = fmt_time(et) if et > 0 else "— (not recorded)"
            dir_text = _direction_label(sig.get("direction", ""))
            entry = sig.get("entry_price", 0)

            cells = {
                0: (shown_time,
                    f"Entry time: {entry_label} | Found at: {fmt_time(ts)}"),
                1: (sig.get("asset", ""), sig.get("asset", "")),
                2: (dir_text, f"Direction: {sig.get('direction', '')}"),
                3: (f"{entry:.5f}", f"Entry: {entry:.5f}"),
            }
            stale = _is_stale_signal(sig)
            for col, (text, tip) in cells.items():
                item = QTableWidgetItem(text)
                if stale:
                    item.setToolTip(f"{tip}\n⚠️ Stale ({_stale_reason(sig)})")
                    item.setForeground(Qt.GlobalColor.gray)
                else:
                    item.setToolTip(tip)
                self._signals_table.setItem(row, col, item)

            # "check" column — opens the full-precision inspect popup (chart +
            # risk lot + Send Order).  The detailed columns (Stop Loss / Take
            # Profit / Rule Score / Model Prob) live in the popup.
            signal_id = sig.get("signal_id", str(row))
            check_btn = QPushButton("✓ Inspect")
            check_btn.setToolTip("Inspect signal (full detail + chart + Send Order)")
            # Constrain the button to the row so it never overflows the cell:
            # stretch it to fill the cell horizontally and cap its height to the
            # row height, keeping the ✓ glyph centred.
            check_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            check_btn.setFixedHeight(26)
            check_btn.setStyleSheet(
                "background-color: #3a6ea5; color: white; font-weight: bold; "
                "border: none; border-radius: 4px; padding: 0px 6px;")
            check_btn.clicked.connect(lambda checked, sid=signal_id: self._inspect_signal(sid))
            # A wrapper widget with zero contents margin keeps the button exactly
            # inside the cell (a raw setCellWidget lets a taller button spill
            # past the row boundary).
            cell_wrap = QWidget()
            wrap_layout = QVBoxLayout(cell_wrap)
            wrap_layout.setContentsMargins(4, 3, 4, 3)
            wrap_layout.addWidget(check_btn)
            self._signals_table.setCellWidget(row, 4, cell_wrap)

    def _inspect_signal(self, signal_id: str) -> None:
        """Open the full-precision inspect popup for a pending signal."""
        snap = self._bridge.state.get_snapshot()
        sig = next(
            (s for s in snap.get("pending_signals", [])
             if str(s.get("signal_id")) == str(signal_id)), None)
        if sig is None:
            self._status_label.setText(f"Signal {signal_id} not found.")
            return
        # Pass the tab's own send-order handler so the popup reuses the
        # existing confirm dialog + 1h stale guard + dispatch logic.
        dialog = SignalInspectDialog(self._bridge, sig, self._on_send_order, parent=self)
        dialog.exec()

    def _on_signal_row_double_clicked(self, item: Any) -> None:
        """Double-clicking a signal row opens the inspect popup."""
        row = item.row()
        snap = self._bridge.state.get_snapshot()
        pend = [s for s in snap.get("pending_signals", [])
                if s.get("asset", "").upper() == (self._current_symbol or "").upper()]
        if 0 <= row < len(pend):
            signal_id = pend[row].get("signal_id")
            if signal_id:
                self._inspect_signal(signal_id)

    # ------------------------------------------------------------------
    # Position P/L helper ($ and R)
    # ------------------------------------------------------------------

    _CONTRACT_SPECS: ClassVar[dict[str, dict[str, float]]] = {
        # base symbol (suffix-stripped): {"contract_size": units/lot, "point_value": $ per 1.0 quote move per unit}
        "XAUUSD": {"contract_size": 100.0, "point_value": 1.0},
        "EURUSD": {"contract_size": 100000.0, "point_value": 1.0},
    }

    @classmethod
    def _contract_spec(cls, asset: str) -> dict:
        """Return {point_value, contract_size} for a symbol (suffix-tolerant).

        Accepts ``EURUSDm`` / ``XAUUSDm`` (MT5 suffixes) by matching the base
        symbol prefix; defaults to the EURUSD standard FX contract when unknown.
        """
        base = (asset or "").upper()
        # Strip a trailing broker suffix (letters that aren't a known base).
        for known in ("XAUUSD", "EURUSD"):
            if base.startswith(known):
                return cls._CONTRACT_SPECS[known]
        return {"contract_size": 100000.0, "point_value": 1.0}

    def _populate_positions_table(self, positions: list[dict]) -> None:
        """Fill the open positions table (Entry Time / Lot / Profit ($) / Close)."""
        self._positions_table.setRowCount(len(positions))

        for row, pos in enumerate(positions):
            # 0: Entry Time — full timestamp; tooltip repeats it for safety.
            open_time = pos.get("open_time", 0)
            time_text = fmt_time(open_time)
            time_item = QTableWidgetItem(time_text)
            time_item.setToolTip(f"Entry Time: {time_text}")
            self._positions_table.setItem(row, 0, time_item)
            # 1: Lot (position size)
            self._positions_table.setItem(row, 1, QTableWidgetItem(f"{pos.get('position_size', 0):.2f}"))
            # 2: Profit ($) — current unrealised profit in account currency.
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", 0)
            size = pos.get("position_size", 0)
            direction = str(pos.get("direction", "buy")).lower()
            spec = self._contract_spec(pos.get("asset", ""))
            # profit$ = price_move * point_value * contract_size * lots
            # For a LONG  : (current - entry);  for a SHORT: (entry - current).
            # Only valid when the position has a real size & price; otherwise
            # present a "—" placeholder rather than a misleading $0.00.
            if size and entry and current:
                price_move = (current - entry) if direction in ("buy", "long") else (entry - current)
                profit_usd = price_move * spec["point_value"] * spec["contract_size"] * size
                profit_text = f"${profit_usd:,.2f}"
            else:
                profit_usd = 0.0
                profit_text = "—"
            pl_item = QTableWidgetItem(profit_text)
            pl_item.setToolTip(f"Unrealised P/L: {profit_text}")
            pl_item.setForeground(Qt.GlobalColor.darkGreen if profit_usd >= 0 else Qt.GlobalColor.darkRed)
            self._positions_table.setItem(row, 2, pl_item)

            # 3: Close button — constrained to the row so it never overflows.
            pos_id = pos.get("position_id", "")
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet(
                "background-color: #b03030; color: white; font-weight: bold; "
                "border: none; border-radius: 4px;")
            close_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            close_btn.setFixedHeight(26)
            close_btn.clicked.connect(lambda checked, pid=pos_id: self._on_close_position(pid))
            cell_wrap = QWidget()
            wrap_layout = QVBoxLayout(cell_wrap)
            wrap_layout.setContentsMargins(4, 3, 4, 3)
            wrap_layout.addWidget(close_btn)
            self._positions_table.setCellWidget(row, 3, cell_wrap)

    # ------------------------------------------------------------------
    # Signal statistics panel (Trigger / Found / Pass)
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_stats_entry(stats: dict[str, Any], symbol: str) -> dict[str, int] | None:
        """Find a symbol's stats entry — exact key first, then case-insensitive.

        Returns ``None`` when the symbol is empty or has never recorded stats.
        """
        if not symbol:
            return None
        entry = stats.get(symbol)
        if entry is not None:
            return entry
        upper = symbol.upper()
        for key, candidate in stats.items():
            if key.upper() == upper:
                return candidate
        return None

    @staticmethod
    def _stats_html(trigger: int | None, found: int | None, passed: int | None) -> str:
        """Build the dark-theme rich-text stats line.

        Format: ``Trigger: XXXX | Found: XXX | Pass: XX``; ``None`` values
        render as an em dash ("—").
        """
        dash = "—"

        def number(value: int | None, color: str) -> str:
            return f'<b style="color:{color};">{dash if value is None else value}</b>'

        sep = f'<span style="color:{COLOR_TEXT_SEC};"> &nbsp;|&nbsp; </span>'
        return (
            f'<span style="color:{COLOR_NEUTRAL};">Trigger: </span>'
            f'{number(trigger, COLOR_TEXT)}'
            + sep
            + f'<span style="color:{COLOR_NEUTRAL};">Found: </span>'
            f'{number(found, COLOR_TEXT)}'
            + sep
            + f'<span style="color:{COLOR_NEUTRAL};">Pass: </span>'
            f'{number(passed, COLOR_POSITIVE)}'
        )

    def _refresh_stats(self, snap: dict[str, Any]) -> None:
        """Update the per-symbol statistics panel from a snapshot.

        Reads ``snap["signal_stats"]`` — the thread-safe per-symbol counters
        the signal engine records in SharedAppState — for the currently
        selected symbol.  Pure GUI-thread read; nothing here mutates state.
        Runs on every ``refresh_from_state`` timer tick, so the numbers
        update live.
        """
        stats = snap.get("signal_stats", {}) or {}
        symbol = self._current_symbol
        entry = self._lookup_stats_entry(stats, symbol)

        if entry is not None:
            self._stats_label.setText(
                self._stats_html(
                    entry.get("trigger", 0),
                    entry.get("found", 0),
                    entry.get("pass", 0),
                )
            )
        else:
            # No symbol selected, or nothing recorded for it yet.
            self._stats_label.setText(self._stats_html(None, None, None))

        # Per-symbol reset only makes sense while this symbol has counters.
        self._reset_stats_btn.setEnabled(entry is not None)

        # "Reset All" is useful when any counters exist or any validated
        # symbol is present (it is a harmless no-op otherwise).
        registry = snap.get("symbol_registry", {}) or {}
        has_validated = any(
            isinstance(cfg, dict) and cfg.get("status") == "validated"
            for cfg in registry.values()
        )
        self._reset_all_stats_btn.setEnabled(bool(stats) or has_validated)

        # Keep the group title in sync with the selected symbol.
        title = f"📊 Signal Statistics — {symbol}" if symbol else "📊 Signal Statistics"
        if self._stats_group.title() != title:
            self._stats_group.setTitle(title)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_send_order(self, signal_id: str, lot_size: float | None = None) -> bool:
        """Send order for a pending signal — requires confirmation.

        After the user confirms, locates the selected signal in the snapshot
        and dispatches it through ``gui_bridge.send_order``.  ``lot_size`` is
        ``None`` by default (let the bridge risk-size, as the manual path does);
        when the caller (the inspect popup) supplies a lot it is honoured
        exactly via ``send_order(lot_size=...)``.

        Returns ``True`` when the order was sent, ``False`` otherwise.
        """
        dialog = ConfirmationDialog(
            title="Send Order Confirmation",
            message=f"Confirm sending market order for signal <b>{signal_id}</b>?\n\n"
                    "The order will be placed at the current market price.",
            confirm_text="✅ SEND ORDER",
            parent=self,
        )
        if not dialog.exec():
            return False
        # Locate the confirmed signal so we can dispatch its order fields.
        snap = self._bridge.state.get_snapshot()
        sig = next(
            (s for s in snap.get("pending_signals", [])
             if s.get("signal_id") == signal_id), None
        )
        if sig is None:
            self._bridge.log_action("send_order_click", {
                "signal_id": signal_id, "dispatched": False,
                "error": "signal_not_found"})
            self._status_label.setText(f"Signal {signal_id} not found.")
            return False
        # Defence-in-depth: even if the button was somehow still enabled,
        # refuse to dispatch a manual confirmation that is too old. Age is
        # measured from the signal's oldest recorded clock (entry bar time; a
        # brand-new list with a fresh find-time but a 5-day-old entry is old).
        stale_reason = _stale_reason(sig)
        if stale_reason:
            self._bridge.log_action("send_order_click", {
                "signal_id": signal_id, "dispatched": False,
                "error": "signal_stale"})
            self._status_label.setText(
                f"Signal {signal_id} refused: {stale_reason}")
            return False
        result = self._bridge.send_order(
            asset=sig.get("asset", ""),
            direction=sig.get("direction", "buy"),
            entry_price=float(sig.get("entry_price", 0.0)),
            stop_loss=float(sig.get("stop_loss", 0.0)),
            take_profit=float(sig.get("take_profit", 0.0)),
            lot_size=lot_size,
        )
        ok = bool(result.get("success", result.get("ok", False)))
        self._bridge.log_action("send_order_click", {
            "signal_id": signal_id, "dispatched": True,
            "lot_size": lot_size, "ok": ok, "result": str(result)})
        if ok:
            self._status_label.setText(f"Order sent for signal {signal_id}.")
        else:
            err = result.get(
                "error",
                (result.get("order_result") or {}).get("error", "unknown error"))
            self._status_label.setText(
                f"Order FAILED for signal {signal_id}: {err}")
        return ok

    def _on_close_position(self, position_id: str) -> None:
        """Close an open position — requires confirmation."""
        # Resolve the position's symbol so the MCP close call can pass the
        # required `symbol` safety-check argument (otherwise the server rejects
        # with "symbol not found in market watch", and the close never happens).
        snap = self._bridge.state.get_snapshot()
        pos = next(
            (p for p in snap.get("open_positions", [])
             if str(p.get("position_id")) == str(position_id)), None)
        symbol = (pos or {}).get("asset", "")
        dialog = ConfirmationDialog(
            title="Close Position Confirmation",
            message=f"Confirm closing position <b>{position_id}</b>?\n\n"
                    "The position will be closed at the current market price.",
            confirm_text="✅ CLOSE POSITION",
            require_input=True,
            input_match="CLOSE",
            input_placeholder='Type "CLOSE" to confirm',
            parent=self,
        )
        if dialog.exec():
            self._bridge.close_position(position_id, symbol=symbol)
            self._bridge.log_action("close_position_click", {"position_id": position_id})
            self._status_label.setText(f"Position {position_id} closed.")

    def _on_reset_stats(self) -> None:
        """Reset the selected symbol's counters — confirmation required."""
        symbol = self._current_symbol
        if not symbol or symbol == "(no symbols)":
            return
        dialog = ConfirmationDialog(
            title="Reset Signal Counters",
            message=(
                f"Reset <b>Trigger / Found / Pass</b> counters for "
                f"<b>{symbol}</b>?\n\n"
                "Only the displayed statistics are cleared — pending signals, "
                "open positions and automation settings are not affected."
            ),
            confirm_text="✅ RESET",
            parent=self,
        )
        if dialog.exec():
            self._bridge.state.reset_signal_stats(symbol)
            self._bridge.log_action("reset_signal_stats", {"symbol": symbol, "scope": "symbol"})
            self._status_label.setText(f"Signal statistics reset for {symbol}")
            self._refresh_stats(self._bridge.state.get_snapshot())

    def _on_reset_all_stats(self) -> None:
        """Reset counters for every symbol — confirmation required."""
        dialog = ConfirmationDialog(
            title="Reset All Signal Counters",
            message=(
                "Reset <b>Trigger / Found / Pass</b> counters for "
                "<b>ALL symbols</b>?\n\n"
                "Only the displayed statistics are cleared — pending signals, "
                "open positions and automation settings are not affected."
            ),
            confirm_text="✅ RESET ALL",
            parent=self,
        )
        if dialog.exec():
            self._bridge.state.reset_signal_stats()  # symbol=None → every symbol
            self._bridge.log_action("reset_signal_stats", {"symbol": "*", "scope": "all"})
            self._status_label.setText("Signal statistics reset for all symbols")
            self._refresh_stats(self._bridge.state.get_snapshot())


class _InspectLoadWorker(QObject):
    """Background worker for the inspect dialog's slow data fetches.

    Runs ``SystemBridge.fetch_candles`` and ``compute_risk_lot`` on a worker
    thread (they hit the MCP/execution layer over the network) and emits the
    results back to the GUI thread via Qt signals.  This keeps the inspect
    dialog instant to open — the chart and risk-lot default fill in as soon as
    the data arrives instead of blocking the GUI on the MCP roundtrip.
    """

    lot_ready = Signal(float)
    candles_ready = Signal(object)
    finished = Signal()

    def __init__(self, bridge: SystemBridge, asset: str, limit: int = 50,
                 risk: tuple | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._asset = asset
        self._limit = limit
        self._risk = risk  # (entry, stop) for risk lot sizing

    def run(self) -> None:
        try:
            lot = 0.0
            if self._risk:
                entry, stop = self._risk
                try:
                    lot = float(self._bridge.compute_risk_lot(self._asset, entry, stop) or 0.0)
                except Exception:
                    lot = 0.0
            self.lot_ready.emit(lot)
            try:
                candles = self._bridge.fetch_candles(self._asset, limit=self._limit)
            except Exception:
                candles = None
            self.candles_ready.emit(candles)
        finally:
            self.finished.emit()


class SignalInspectDialog(QDialog):
    """Full-precision, read-only inspector for a pending signal.

    Layout (top → bottom): candlestick chart of the last ~50 M15 candles with
    entry (white dashed) / take-profit (green dashed) / stop-loss (red dashed)
    levels marked; an adjustable auto-lot spinbox defaulting to the risk-based
    size; entry / stop-loss / take-profit (.5f); rule score (.3f) and model
    probability (.4f); the signal id + timestamp/age; then a Send Order button.

    The Send Order button reuses ``LiveControlTab._on_send_order`` (the existing
    confirmation dialog + 1-hour stale guard).  When the lot spinbox is left at
    0 (auto/unset) the risk-sized path is used (``send_order(lot_size=None)``);
    otherwise the chosen lot is passed through.
    """

    def __init__(self, bridge: SystemBridge, signal: dict[str, Any],
                 send_callback: Callable[[str, float | None], bool],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._signal = signal
        self._send_callback = send_callback
        self.setWindowTitle("📊 Inspect Signal")
        # Generous default canvas so the candlestick chart (at the top, stretch
        # 2) gets real vertical room. The previous default made the popup fit
        # its content so tightly that the chart was compressed / hard to read.
        self.setMinimumSize(560, 620)
        self.setModal(True)
        # Dark theme (high contrast) — a plain QDialog does NOT inherit the app
        # stylesheet, so apply the surface/text colours explicitly.
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLOR_SURFACE};
                color: {COLOR_TEXT};
            }}
            QLabel {{
                color: {COLOR_TEXT};
                font-size: 13px;
            }}
            QDoubleSpinBox {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
            }}
            QPushButton {{
                background-color: #3a3a3a;
                color: {COLOR_TEXT};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 14px;
                min-height: 24px;
            }}
            QPushButton:hover {{
                background-color: #4a4a4a;
            }}
        """)
        self._build_ui()
        # Kick off candle + risk-lot loading in the background so the dialog
        # opens instantly (previously fetch_candles() blocked the GUI thread
        # for ~1s on every open).  The worker emits a signal that updates the
        # chart + lot spinbox on the GUI thread when the data is ready.
        self._start_async_load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        sig = self._signal

        # 1. Candlestick chart at the top -- give it a real, expanding canvas.
        #    A plain QWidget defaults to a tiny Preferred/Preferred sizeHint, so
        #    previously the QVBox could collapse it into a thin strip. Explicit
        #    Expanding + a solid minimum height + the largest stretch share make
        #    the candles readable.
        self._chart = CandlestickChart()
        self._chart.setMinimumHeight(260)
        self._chart.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._chart, stretch=6)

        # 2. Adjustable auto-lot spinbox (default = risk-based size; 0 = auto).
        #    The default risk-lot is populated asynchronously by _start_async_load
        #    (never fetched on the GUI thread).
        lot_row = QHBoxLayout()
        lot_row.addWidget(QLabel("Auto Lot:"))
        self._lot_spin = QDoubleSpinBox()
        self._lot_spin.setRange(0.0, 100.0)
        self._lot_spin.setDecimals(2)
        self._lot_spin.setSingleStep(0.01)
        self._lot_spin.setValue(0.0)
        self._lot_spin.setToolTip(
            "0.00 = auto (risk-sized via RiskGuard); set a value to override the lot.")
        lot_row.addWidget(self._lot_spin)
        lot_row.addStretch()
        layout.addLayout(lot_row)

        # 3. Entry / Stop Loss / Take Profit (.5f) — full precision.
        form = QFormLayout()
        form.setSpacing(8)
        form.addRow("Asset:", QLabel(str(sig.get("asset", "—"))))
        form.addRow("Direction:", QLabel(_direction_label(sig.get("direction", ""))))
        form.addRow("Entry:", QLabel(f"{float(sig.get('entry_price', 0) or 0):.5f}"))
        form.addRow("Stop Loss:", QLabel(f"{float(sig.get('stop_loss', 0) or 0):.5f}"))
        form.addRow("Take Profit:", QLabel(f"{float(sig.get('take_profit', 0) or 0):.5f}"))
        # 4. Rule Score (.3f) + Model Probability (.4f).
        form.addRow("Rule Score:", QLabel(f"{float(sig.get('rule_score', 0) or 0):.3f}"))
        form.addRow("Model Prob:", QLabel(f"{float(sig.get('model_probability', 0) or 0):.4f}"))
        form.addRow("Signal ID:", QLabel(str(sig.get("signal_id", "—"))))
        # True entry time (entry-bar open from the engine) — informational only.
        # The Age row drives the 1h stale guard and is measured from the oldest
        # recorded signal clock (entry bar time preferred, else find time), so
        # the label always matches whether Send Order is enabled/disabled.
        et = float(sig.get("entry_time", 0.0) or 0.0)
        form.addRow("Entry Time:",
                    QLabel(fmt_time(et) if et > 0 else "— (not recorded)"))
        form.addRow("Age (since found):", QLabel(self._age_text()))
        layout.addLayout(form)

        # 5. Send Order / Cancel.
        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("✅ SEND ORDER")
        self._send_btn.setStyleSheet(
            f"background-color: {COLOR_POSITIVE}; color: #1e1e1e; font-weight: bold;")
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"background-color: {COLOR_NEGATIVE}; color: white;")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Status line.
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Stale guard (MAX_MANUAL_SIGNAL_AGE_S): grey + disable Send Order.
        # Age is measured from the signal's OLDEST recorded signal clock (the
        # entry-bar time first, else the find time). A signal whose entry is
        # 5 days old but that got freshly re-listed (find age = 45 min) MUST be
        # disabled -- it sat on an old bar and its levels are no longer valid.
        if _is_stale_signal(sig):
            age = _signal_age_s(sig)
            age_min = int(age / 60)
            self._send_btn.setEnabled(False)
            self._send_btn.setStyleSheet("background-color: #444444; color: #999999;")
            self._send_btn.setToolTip(
                _stale_reason(sig))
            self._status_label.setText(
                f"⚠️ Signal is {age_min} minutes old -> max "
                f"{int(MAX_MANUAL_SIGNAL_AGE_S / 60)} min -- Send Order disabled.")
            self._status_label.setStyleSheet(f"color: {COLOR_WARNING};")

    # ------------------------------------------------------------------
    # Background loading (chart candles + risk lot) — keeps MCP network calls
    # OFF the GUI thread so the popup opens instantly (~1s freeze eliminated).
    # ------------------------------------------------------------------

    def _start_async_load(self) -> None:
        """Fetch candles + risk lot on a worker thread; apply via signal."""
        asset = str(self._signal.get("asset", ""))
        entry = float(self._signal.get("entry_price", 0.0) or 0.0)
        stop = float(self._signal.get("stop_loss", 0.0) or 0.0)

        # Show a "loading…" placeholder immediately (chart paints no data).
        self._chart.set_loading(True)

        # Fetch up to 200 M15 candles (the MCP call returns up to 14 days; the
        # extra rows are sliced locally — no extra network) so _on_candles_ready
        # can anchor the chart at the signal's entry bar when it is present.
        self._worker = _InspectLoadWorker(self._bridge, asset, limit=200,
                                          risk=(entry, stop))
        self._worker.lot_ready.connect(self._on_lot_ready)
        self._worker.candles_ready.connect(self._on_candles_ready)
        # Keep a reference + let Qt manage the thread lifecycle.
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_lot_ready(self, lot: float) -> None:
        """Apply the risk-sized default lot to the spinbox (GUI thread)."""
        if lot > 0:
            self._lot_spin.setValue(round(float(lot), 2))

    def _on_candles_ready(self, candles: Any) -> None:
        """Apply the fetched candles + signal levels to the chart (GUI thread)."""
        sig = self._signal
        try:
            entry = float(sig.get("entry_price", 0.0) or 0.0)
            stop = float(sig.get("stop_loss", 0.0) or 0.0)
            tp = float(sig.get("take_profit", 0.0) or 0.0)
        except (TypeError, ValueError):
            entry = stop = tp = 0.0
        candles = self._slice_from_entry(candles)
        self._chart.set_data(
            candles, entry=entry, stop=stop, take_profit=tp,
            direction=sig.get("direction", ""))

    def _slice_from_entry(self, candles: Any) -> Any:
        """Cut the candle frame from the signal's entry bar onward.

        When price has moved far from the entry zone, the entry/SL/TP band
        would otherwise be crushed against one edge of the last-50-candle
        window.  If ``entry_time`` is recorded and its bar lies inside the
        fetched frame, slicing to that bar keeps the marker lines centred on
        visible context.  Pure in-memory pandas slicing (no network) — falls
        back to the frame as-is for old signals / gaps / non-DataFrame input.
        """
        try:
            et = float(self._signal.get("entry_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            return candles
        if et <= 0 or candles is None or getattr(candles, "empty", True):
            return candles
        try:
            import pandas as pd
            idx = candles.index
            if not isinstance(idx, pd.DatetimeIndex):
                return candles
            # entry_time is a naive-local epoch; convert to the same wall-clock
            # Timestamp the M15 index uses so the comparison is apples-to-apples.
            entry_ts = pd.Timestamp(datetime.fromtimestamp(et))
            if not (idx.min() <= entry_ts <= idx.max()):
                return candles
            return candles.loc[idx >= entry_ts]
        except Exception:
            return candles

    def _age_text(self) -> str:
        # Report the effective age (oldest recorded signal clock) that drives
        # the stale guard, so the label matches the button state.
        age = _signal_age_s(self._signal)
        if age <= 0:
            return "— (no timestamp)"
        m, s = int(age // 60), int(age % 60)
        return f"{m}m {s}s"

    def _on_send(self) -> None:
        """Reuse ``LiveControlTab._on_send_order`` (confirm + stale guard)."""
        sig = self._signal
        signal_id = str(sig.get("signal_id", ""))
        lot = self._lot_spin.value()
        lot_size = round(lot, 2) if lot > 0 else None
        ok = self._send_callback(signal_id, lot_size)
        if ok:
            self._status_label.setText(f"✅ Order sent for signal {signal_id}.")
            self._status_label.setStyleSheet(f"color: {COLOR_POSITIVE};")
            self.accept()
        else:
            self._status_label.setText(
                "❌ Order was not sent (cancelled, stale, or failed).")
            self._status_label.setStyleSheet(f"color: {COLOR_NEGATIVE};")