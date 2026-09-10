"""
gui_tab_system_log.py — Tab 5: System Log

Real-time log viewer with filters, auto-scroll, export, and clear.

Spec: GUI_REWORK_REQUIREMENTS.md Section 5.

Features:
  - Log table: timestamp, level (colored), source, symbol, message
  - Filters: level (multi-checkbox), symbol dropdown, source dropdown,
    time range (last 5/15/60 min, today, custom), search text
  - Auto-scroll on/off (Pause button)
  - Export CSV / TXT
  - Clear log (with confirmation dialog)
  - Click row → detail popup (extra JSON)
  - QTimer at 500 ms polls the thread-safe ring buffer
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from live.gui.gui_bridge import SystemBridge
from live.gui.gui_components import (
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_SEC,
    COLOR_WARNING,
    ConfirmationDialog,
)
from live.logging.logger_v2 import (
    clear_ring_buffer,
    clear_system_logs,
    get_ring_buffer,
    query_system_logs,
)

# ---------------------------------------------------------------------------
# Level colours (GUI_REWORK §5.4)
# ---------------------------------------------------------------------------
LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "#888888",   # grey
    "INFO":     "#aaaaaa",   # light grey / white
    "WARNING":  "#ffaa00",   # yellow
    "ERROR":    "#ff4444",   # red
    "CRITICAL": "#ff0000",   # bold red
}

_LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ---------------------------------------------------------------------------
# Time-range presets
# ---------------------------------------------------------------------------
TIME_PRESETS: dict[str, timedelta | None] = {
    "All":       None,
    "Last 5m":   timedelta(minutes=5),
    "Last 15m":  timedelta(minutes=15),
    "Last 1h":   timedelta(hours=1),
    "Today":     timedelta(hours=24),
}


class SystemLogTab(QWidget):
    """Tab 5: Centralized System Log — real-time log viewer."""

    def __init__(self, bridge: SystemBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._auto_scroll = True
        self._all_entries: list[dict[str, Any]] = []
        # Cap the in-memory buffer: the table rebuilds ALL rows on every
        # refresh, so an unbounded buffer made this tab freeze the whole GUI.
        self._MAX_ENTRIES = 2000
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        # --- Header ---
        header = QLabel("📋 System Log")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(header)

        desc = QLabel(
            "Real-time log of all system activity. "
            "Use filters to zoom in on specific levels, sources, or symbols."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_NEUTRAL}; padding-bottom: 4px;")
        layout.addWidget(desc)

        # --- Toolbar row ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Pause / Resume auto-scroll
        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLOR_WARNING};
                color: #1e1e1e;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
            }}
            QPushButton:checked {{
                background-color: {COLOR_POSITIVE};
            }}
            """
        )
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        toolbar.addWidget(self._pause_btn)

        toolbar.addSpacing(12)

        # Export CSV
        self._export_csv_btn = QPushButton("📥 Export CSV")
        self._export_csv_btn.clicked.connect(self._on_export_csv)
        toolbar.addWidget(self._export_csv_btn)

        # Export TXT
        self._export_txt_btn = QPushButton("📥 Export TXT")
        self._export_txt_btn.clicked.connect(self._on_export_txt)
        toolbar.addWidget(self._export_txt_btn)

        toolbar.addSpacing(12)

        # Clear log
        self._clear_btn = QPushButton("🗑 Clear Log")
        self._clear_btn.setStyleSheet(
            f"background-color: {COLOR_NEGATIVE}; color: white; font-weight: bold;"
        )
        self._clear_btn.clicked.connect(self._on_clear_log)
        toolbar.addWidget(self._clear_btn)

        toolbar.addStretch()

        # Log count
        self._count_label = QLabel("0 entries")
        self._count_label.setStyleSheet(f"color: {COLOR_TEXT_SEC}; font-size: 11px;")
        toolbar.addWidget(self._count_label)

        layout.addLayout(toolbar)

        # --- Filters row ---
        filters = QHBoxLayout()
        filters.setSpacing(8)

        # Level checkboxes (multi-select)
        level_group = QGroupBox("Level")
        level_layout = QHBoxLayout(level_group)
        level_layout.setSpacing(2)
        level_layout.setContentsMargins(4, 2, 4, 2)
        self._level_checks: dict[str, QCheckBox] = {}
        for level in _LEVEL_ORDER:
            cb = QCheckBox(level)
            if level == "DEBUG":
                cb.setChecked(False)
            else:
                cb.setChecked(True)
            cb.stateChanged.connect(self._on_filter_changed)
            self._level_checks[level] = cb
            level_layout.addWidget(cb)
        filters.addWidget(level_group)

        # Source dropdown
        self._source_combo = QComboBox()
        self._source_combo.setMinimumWidth(130)
        self._source_combo.addItem("All Sources")
        self._source_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(QLabel("Source:"))
        filters.addWidget(self._source_combo)

        # Symbol dropdown
        self._symbol_combo = QComboBox()
        self._symbol_combo.setMinimumWidth(110)
        self._symbol_combo.addItem("All Symbols")
        self._symbol_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(QLabel("Symbol:"))
        filters.addWidget(self._symbol_combo)

        # Time range
        self._time_combo = QComboBox()
        self._time_combo.addItems(list(TIME_PRESETS.keys()))
        self._time_combo.currentIndexChanged.connect(self._on_filter_changed)
        filters.addWidget(QLabel("Time:"))
        filters.addWidget(self._time_combo)

        # Search text
        filters.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search messages...")
        self._search_input.setMinimumWidth(140)
        self._search_input.textChanged.connect(self._on_filter_changed)
        filters.addWidget(self._search_input)

        # Clear filters
        clear_filters_btn = QPushButton("✕ Clear Filters")
        clear_filters_btn.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px;")
        clear_filters_btn.clicked.connect(self._on_clear_filters)
        filters.addWidget(clear_filters_btn)

        layout.addLayout(filters)

        # --- Log table ---
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Timestamp", "Level", "Source", "Symbol", "Message"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.cellDoubleClicked.connect(self._on_row_clicked)
        layout.addWidget(self._table, stretch=1)

        # --- Start timer for periodic refresh ---
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_log)
        # 1.5s (was 500ms): rebuilding a multi-thousand-row table at 2Hz froze
        # the GUI. With the conditional repaint and buffer cap below, 1.5s is
        # plenty for a responsive live log.
        self._timer.start(1500)

        # --- Initial load from DB ---
        self._load_from_db()

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------

    def _refresh_log(self) -> None:
        """Called by QTimer — poll ring buffer and update table.

        Instead of rebuilding the whole (up to ``_MAX_ENTRIES``) table on every
        tick, it diffs the previously-visible rows against the newly-visible ones
        and only removes the front rows that fell out of the window and appends
        the new rows at the bottom.  The main thread's per-tick cost is thus
        O(#new + #dropped) item creations instead of O(rows), which keeps the
        System Log responsive even with a full 2000-entry buffer.
        """
        buf = get_ring_buffer()
        known_ids = {e.get("id") for e in self._all_entries}
        new_entries = [e for e in buf if e.get("id") not in known_ids]
        if not new_entries:
            return

        # Snapshot the currently-visible rows (filtered) before mutating.
        old_visible = self._filter_entries(self._all_entries)
        old_ids = {e.get("id") for e in old_visible}

        # Trim + append the in-memory buffer (sliding window).
        self._all_entries.extend(new_entries)
        if len(self._all_entries) > self._MAX_ENTRIES:
            self._all_entries = self._all_entries[-self._MAX_ENTRIES:]
        self._rebuild_filter_dropdowns()

        # Diff by id: rows that left the window (front) and rows that entered (back).
        new_visible = self._filter_entries(self._all_entries)
        new_ids = {e.get("id") for e in new_visible}
        to_remove = [e for e in old_visible if e.get("id") not in new_ids]
        to_add = [e for e in new_visible if e.get("id") not in old_ids]
        self._apply_incremental(to_remove, to_add)

    def _load_from_db(self) -> None:
        """Initial load: pull recent entries from SQLite."""
        try:
            rows = query_system_logs(limit=5_000)
            known_ids = {e.get("id") for e in self._all_entries}
            for r in rows:
                if r.get("id") not in known_ids:
                    self._all_entries.append(r)
            if len(self._all_entries) > self._MAX_ENTRIES:
                self._all_entries = self._all_entries[-self._MAX_ENTRIES:]
            self._rebuild_filter_dropdowns()
            self._populate_table()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _set_cell_items(self, entry: dict[str, Any], row: int) -> None:
        """Populate *row* of the table from a log entry dict (single-cell helper)."""
        # Timestamp
        ts = entry.get("timestamp", "")
        ts_item = QTableWidgetItem(ts)
        ts_item.setForeground(Qt.GlobalColor.white)
        self._table.setItem(row, 0, ts_item)

        # Level (colored)
        level = entry.get("level", "INFO")
        level_item = QTableWidgetItem(level)
        colour_hex = LEVEL_COLORS.get(level, COLOR_TEXT)
        level_item.setForeground(QColor(colour_hex) if colour_hex else Qt.GlobalColor.white)
        if level == "CRITICAL":
            font = level_item.font()
            font.setBold(True)
            level_item.setFont(font)
        self._table.setItem(row, 1, level_item)

        # Source
        source = entry.get("source", "")
        src_item = QTableWidgetItem(source)
        src_item.setForeground(Qt.GlobalColor.white)
        self._table.setItem(row, 2, src_item)

        # Symbol
        symbol = entry.get("symbol", "")
        sym_item = QTableWidgetItem(symbol)
        sym_item.setForeground(Qt.GlobalColor.white)
        self._table.setItem(row, 3, sym_item)

        # Message
        msg = entry.get("message", "")
        msg_item = QTableWidgetItem(msg)
        msg_item.setForeground(Qt.GlobalColor.white)
        self._table.setItem(row, 4, msg_item)

        # Store entry data for double-click detail lookup
        ts_item.setData(Qt.UserRole, entry)

    def _populate_table(self) -> None:
        """Apply filters and rebuild the table display (full rebuild).

        Used on initial load, filter changes, clear, and buffer overflow.  The
        steady-state refresh path appends incrementally via ``_append_entries``
        so the main thread does not pay an O(rows) item-creation cost on every
        tick.
        """
        entries = self._filter_entries(self._all_entries)

        # Rebuilding the table creates thousands of cells; suspend repaints so
        # Qt paints ONCE at the end instead of once per item (avoids a frozen UI).
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                self._set_cell_items(entry, row)
        finally:
            self._table.setUpdatesEnabled(True)

        # Update count
        self._count_label.setText(
            f"{len(entries)} entries (buffer: {len(self._all_entries)})"
        )

        # Auto-scroll to bottom
        if self._auto_scroll and len(entries) > 0:
            self._table.scrollToBottom()

    def _apply_incremental(self, to_remove: list[dict[str, Any]],
                           to_add: list[dict[str, Any]]) -> None:
        """Apply the sliding-window diff to the table.

        Removes the rows that fell out of the window from the front and appends
        the new rows at the bottom, rather than rebuilding the whole table.
        Suspends repaints so Qt paints once at the end.
        """
        if not to_remove and not to_add:
            return
        self._table.setUpdatesEnabled(False)
        try:
            for _ in to_remove:
                self._table.removeRow(0)
            start = self._table.rowCount()
            self._table.setRowCount(start + len(to_add))
            for i, entry in enumerate(to_add):
                self._set_cell_items(entry, start + i)
        finally:
            self._table.setUpdatesEnabled(True)

        total_entries = len(self._filter_entries(self._all_entries))
        self._count_label.setText(
            f"{total_entries} entries (buffer: {len(self._all_entries)})"
        )

        if self._auto_scroll and total_entries > 0:
            self._table.scrollToBottom()

    # ------------------------------------------------------------------
    # Filter logic
    # ------------------------------------------------------------------

    def _filter_entries(self, entries: list[dict]) -> list[dict]:
        """Apply all active filters to the entry list."""
        # Level filter
        active_levels = {
            level for level, cb in self._level_checks.items() if cb.isChecked()
        }
        if active_levels:
            entries = [e for e in entries if e.get("level", "INFO") in active_levels]

        # Source filter
        source_filter = self._source_combo.currentText()
        if source_filter and source_filter != "All Sources":
            entries = [e for e in entries if e.get("source", "") == source_filter]

        # Symbol filter
        symbol_filter = self._symbol_combo.currentText()
        if symbol_filter and symbol_filter != "All Symbols":
            entries = [e for e in entries if e.get("symbol", "") == symbol_filter]

        # Time range filter
        time_label = self._time_combo.currentText()
        delta = TIME_PRESETS.get(time_label)
        if delta is not None:
            cutoff = datetime.now(timezone.utc) - delta
            cutoff_str = cutoff.isoformat()
            entries = [e for e in entries if e.get("timestamp", "") >= cutoff_str]

        # Search text
        search_text = self._search_input.text().strip().lower()
        if search_text:
            entries = [
                e for e in entries
                if search_text in e.get("message", "").lower()
                or search_text in e.get("source", "").lower()
                or search_text in e.get("symbol", "").lower()
            ]

        return entries

    def _rebuild_filter_dropdowns(self) -> None:
        """Rebuild the source and symbol dropdowns from all entries."""
        sources = sorted(
            set(e.get("source", "") for e in self._all_entries if e.get("source"))
        )
        symbols = sorted(
            set(e.get("symbol", "") for e in self._all_entries if e.get("symbol"))
        )

        current_source = self._source_combo.currentText()
        current_symbol = self._symbol_combo.currentText()

        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        self._source_combo.addItem("All Sources")
        for s in sources:
            self._source_combo.addItem(s)
        if current_source == "All Sources" or current_source in sources:
            self._source_combo.setCurrentText(current_source)
        self._source_combo.blockSignals(False)

        self._symbol_combo.blockSignals(True)
        self._symbol_combo.clear()
        self._symbol_combo.addItem("All Symbols")
        for s in symbols:
            self._symbol_combo.addItem(s)
        if current_symbol == "All Symbols" or current_symbol in symbols:
            self._symbol_combo.setCurrentText(current_symbol)
        self._symbol_combo.blockSignals(False)

    def _on_filter_changed(self) -> None:
        """Re-apply filters when any filter control changes."""
        self._populate_table()

    def _on_clear_filters(self) -> None:
        """Reset all filters to defaults."""
        for level, cb in self._level_checks.items():
            cb.blockSignals(True)
            cb.setChecked(level != "DEBUG")
            cb.blockSignals(False)
        self._source_combo.setCurrentIndex(0)
        self._symbol_combo.setCurrentIndex(0)
        self._time_combo.setCurrentIndex(0)
        self._search_input.clear()
        self._populate_table()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_pause_toggled(self, checked: bool) -> None:
        """Toggle auto-scroll on/off."""
        self._auto_scroll = not checked
        if checked:
            self._pause_btn.setText("▶ Resume")
        else:
            self._pause_btn.setText("⏸ Pause")
            if not checked:
                self._table.scrollToBottom()

    def _on_row_clicked(self, row: int, column: int) -> None:
        """Double-click a row to see detailed extra JSON."""
        item = self._table.item(row, 0)
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        if not entry:
            return
        self._show_detail_dialog(entry)

    def _show_detail_dialog(self, entry: dict[str, Any]) -> None:
        """Show a dialog with full log entry details including extra_json."""
        extra = entry.get("extra_json")
        detail_text = (
            f"Timestamp: {entry.get('timestamp', '')}\n"
            f"Level:     {entry.get('level', '')}\n"
            f"Source:    {entry.get('source', '')}\n"
            f"Symbol:    {entry.get('symbol', '')}\n"
            f"Message:   {entry.get('message', '')}\n"
        )
        if extra and extra != "null":
            import json as _json
            try:
                parsed = _json.loads(extra)
                extra = _json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                pass
            detail_text += f"\nExtra context:\n{extra}"

        dialog = QDialog(self)
        dialog.setWindowTitle("📄 Log Entry Detail")
        dialog.setMinimumSize(550, 350)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {COLOR_SURFACE};
                color: {COLOR_TEXT};
            }}
            QPushButton {{
                background-color: #3a3a3a;
                color: {COLOR_TEXT};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: #4a4a4a;
            }}
        """)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(detail_text)
        text_edit.setStyleSheet(f"background-color: {COLOR_SURFACE}; color: {COLOR_TEXT};")
        layout.addWidget(text_edit)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_csv(self) -> None:
        """Export visible logs to CSV file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log as CSV", "system_log.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        entries = self._filter_entries(self._all_entries)
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("Timestamp,Level,Source,Symbol,Message\n")
                for e in entries:
                    # Escape commas and quotes in message
                    msg = e.get("message", "").replace('"', '""')
                    f.write(
                        f'"{e.get("timestamp", "")}",'
                        f'"{e.get("level", "")}",'
                        f'"{e.get("source", "")}",'
                        f'"{e.get("symbol", "")}",'
                        f'"{msg}"\n'
                    )
            self._count_label.setText(
                f"{len(entries)} entries | Exported CSV: {path}"
            )
        except Exception as ex:
            self._count_label.setText(f"Export failed: {ex}")

    def _on_export_txt(self) -> None:
        """Export visible logs to TXT file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log as TXT", "system_log.txt", "Text Files (*.txt)"
        )
        if not path:
            return
        entries = self._filter_entries(self._all_entries)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("System Log Export\n")
                f.write("=" * 80 + "\n\n")
                for e in entries:
                    f.write(
                        f"[{e.get('timestamp', '')}] "
                        f"[{e.get('level', '')}] "
                        f"[{e.get('source', '')}] "
                        f"[{e.get('symbol', '')}] "
                        f"{e.get('message', '')}\n"
                    )
            self._count_label.setText(
                f"{len(entries)} entries | Exported TXT: {path}"
            )
        except Exception as ex:
            self._count_label.setText(f"Export failed: {ex}")

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def _on_clear_log(self) -> None:
        """Clear log — requires confirmation."""
        dialog = ConfirmationDialog(
            title="Clear System Log",
            message="⚠️ Clear all log entries?\n\n"
                    "This removes entries from the in-memory buffer "
                    "AND the database. This action cannot be undone.",
            confirm_text="🗑 CLEAR ALL LOGS",
            require_input=True,
            input_match="CLEAR",
            input_placeholder='Type "CLEAR" to confirm',
            parent=self,
        )
        if dialog.exec():
            # Clear ring buffer
            clear_ring_buffer()
            # Clear SQLite table
            try:
                clear_system_logs()
            except Exception:
                pass
            self._all_entries.clear()
            self._populate_table()
            self._count_label.setText("0 entries — log cleared")

    # ------------------------------------------------------------------
    # Public refresh (called by main window polling)
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Called by main window timer — triggers our internal QTimer refresh."""
        pass  # We use our own 500ms QTimer; nothing extra needed from main poll