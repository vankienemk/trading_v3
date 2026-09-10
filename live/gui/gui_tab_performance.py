"""
gui_tab_performance.py — Tab 3: Performance Monitor

Spec section 7.2 Tab 3:
  - Per-symbol metrics table: trade count, win rate, PF, CI 95%, breakeven comparison
  - Equity curve chart (cumulative net R by trade)
  - Kill-switch status per symbol: active/inactive, reason, manual override button
  - Manual override requires second confirmation with text input
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
    COLOR_WARNING,
    ConfirmationDialog,
)


class PerformanceTab(QWidget):
    """Tab 3: Performance Monitor — metrics, equity curve, kill-switch."""

    def __init__(self, bridge: SystemBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        # --- Header ---
        header = QLabel("📈 Performance Monitor")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(header)

        desc = QLabel(
            "Real-time performance metrics per symbol, equity curve, and kill-switch management."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_NEUTRAL}; padding-bottom: 4px;")
        layout.addWidget(desc)

        # --- Metrics Table ---
        perf_group = QGroupBox("📊 Performance Per Symbol")
        perf_layout = QVBoxLayout(perf_group)
        self._perf_table = QTableWidget(0, 8)
        self._perf_table.setHorizontalHeaderLabels(
            ["Symbol", "Trades", "Win Rate", "Profit Factor",
             "Avg Net R", "Breakeven", "Above BE", "95% CI"]
        )
        self._perf_table.horizontalHeader().setStretchLastSection(True)
        self._perf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._perf_table.setAlternatingRowColors(True)
        self._perf_table.verticalHeader().setVisible(False)
        perf_layout.addWidget(self._perf_table)
        layout.addWidget(perf_group)

        # --- Pattern Breakdown (§10.2): per (pattern, TF) table ---
        pat_group = QGroupBox("🧩 Pattern Breakdown (per pattern x TF §10.2)")
        pat_layout = QVBoxLayout(pat_group)
        self._pattern_table = QTableWidget(0, 7)
        self._pattern_table.setHorizontalHeaderLabels(
            ["Pattern", "TF", "Trades", "Win Rate", "PF", "Expectancy (R)",
             "Rolling Equity (R)"]
        )
        self._pattern_table.horizontalHeader().setStretchLastSection(True)
        self._pattern_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._pattern_table.setAlternatingRowColors(True)
        self._pattern_table.verticalHeader().setVisible(False)
        pat_layout.addWidget(self._pattern_table)
        layout.addWidget(pat_group)

        # --- Confluence panel (§10.2): expectancy by number of agreeing
        # patterns (validates the §6.4 hypothesis with real data) ---
        conf_group = QGroupBox("🧬 Confluence Expectancy (by # agreeing patterns)")
        conf_layout = QVBoxLayout(conf_group)
        self._confluence_display = QTextEdit()
        self._confluence_display.setReadOnly(True)
        self._confluence_display.setMaximumHeight(110)
        self._confluence_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLOR_SURFACE};
                color: {COLOR_TEXT};
                border: 1px solid #3d3d3d;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }}
        """)
        conf_layout.addWidget(self._confluence_display)
        layout.addWidget(conf_group)

        # --- Equity Curve ---
        eq_group = QGroupBox("📉 Equity Curve (Cumulative Net R)")
        eq_layout = QVBoxLayout(eq_group)
        self._equity_display = QTextEdit()
        self._equity_display.setReadOnly(True)
        self._equity_display.setMaximumHeight(120)
        self._equity_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLOR_SURFACE};
                color: {COLOR_TEXT};
                border: 1px solid #3d3d3d;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }}
        """)
        eq_layout.addWidget(self._equity_display)
        layout.addWidget(eq_group)

        # --- Kill-Switch ---
        ks_group = QGroupBox("🔒 Kill-Switch Status")
        ks_layout = QVBoxLayout(ks_group)
        self._ks_container = QVBoxLayout()
        ks_layout.addLayout(self._ks_container)
        layout.addWidget(ks_group)

    # ------------------------------------------------------------------
    # Periodic refresh
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Called periodically by the main window timer."""
        snap = self._bridge.state.get_snapshot()
        self._refresh_perf_table(snap)
        self._refresh_equity(snap)
        self._refresh_kill_switch(snap)
        self._refresh_pattern_breakdown()
        self._refresh_confluence()

    # ------------------------------------------------------------------
    # §10.2 Pattern Breakdown + Confluence
    # ------------------------------------------------------------------

    def _refresh_pattern_breakdown(self) -> None:
        """Refresh the per-(pattern, TF) performance table (§10.2)."""
        try:
            rows = self._bridge.pattern_breakdown()
        except Exception:
            rows = []
        self._pattern_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._pattern_table.setItem(r, 0, QTableWidgetItem(row.get("pattern", "—")))
            self._pattern_table.setItem(r, 1, QTableWidgetItem(row.get("timeframe") or "—"))
            n = row.get("trades", 0)
            self._pattern_table.setItem(r, 2, QTableWidgetItem(str(n)))
            wr = row.get("win_rate", 0.0)
            self._pattern_table.setItem(r, 3, QTableWidgetItem(f"{wr*100:.1f}%" if n else "—"))
            pf = row.get("profit_factor", 0.0)
            pf_text = "∞" if pf == float("inf") else (f"{pf:.3f}" if n else "—")
            self._pattern_table.setItem(r, 4, QTableWidgetItem(pf_text))
            exp = row.get("expectancy_r", 0.0)
            exp_item = QTableWidgetItem(f"{exp:+.4f}R" if n else "—")
            exp_item.setForeground(Qt.GlobalColor.darkGreen if exp > 0 else Qt.GlobalColor.darkRed)
            self._pattern_table.setItem(r, 5, exp_item)
            eq = row.get("rolling_equity_r", 0.0)
            self._pattern_table.setItem(r, 6, QTableWidgetItem(f"{eq:+.4f}R" if n else "—"))
        if not rows:
            self._pattern_table.setRowCount(1)
            item = QTableWidgetItem("No pattern-labelled trades yet — assign patterns in Onboarding.")
            self._pattern_table.setItem(0, 0, item)

    def _refresh_confluence(self) -> None:
        """Refresh the confluence expectancy panel (§10.2)."""
        try:
            buckets = self._bridge.confluence_expectancy()
        except Exception:
            buckets = []
        if not buckets:
            self._confluence_display.setText("No confluence data yet.")
            return
        lines = []
        for b in buckets:
            n = b.get("n_patterns", 0)
            cnt = b.get("count", 0)
            avg = b.get("avg_net_r", 0.0)
            if cnt > 0:
                lines.append(
                    f"  {n} pattern(s) agreeing : {cnt} event(s)  "
                    f"avg net R = {avg:+.4f}"
                )
            else:
                lines.append(f"  {n} pattern(s) agreeing : no data yet")
        self._confluence_display.setText("Expectancy by confluence group size:\n" + "\n".join(lines))

    def _refresh_perf_table(self, snap: dict[str, Any]) -> None:
        """Update the performance metrics table from snapshot."""
        reg = snap.get("symbol_registry", {})
        snap.get("trade_log", [])
        symbols = sorted(reg.keys()) if isinstance(reg, dict) else []

        self._perf_table.setRowCount(len(symbols))
        for row, sym in enumerate(symbols):
            perf = self._bridge.state.get_performance(sym)
            tc = perf.get("trade_count", 0)
            wr = perf.get("win_rate", 0.0)
            pf = perf.get("profit_factor", 0.0)
            avg_r = perf.get("avg_net_r", 0.0)
            be = perf.get("breakeven_cost", 0.0)
            above = perf.get("above_breakeven", False)
            ci_low = perf.get("ci_95_low", 0.0)
            ci_high = perf.get("ci_95_high", 0.0)

            self._perf_table.setItem(row, 0, QTableWidgetItem(sym))
            self._perf_table.setItem(row, 1, QTableWidgetItem(str(tc)))

            wr_item = QTableWidgetItem(f"{wr*100:.1f}%" if tc > 0 else "—")
            self._perf_table.setItem(row, 2, wr_item)

            pf_text = f"{pf:.3f}" if pf and pf != float("inf") else "∞" if tc > 0 else "—"
            self._perf_table.setItem(row, 3, QTableWidgetItem(pf_text))

            avg_item = QTableWidgetItem(f"{avg_r:.4f}R" if tc > 0 else "—")
            self._perf_table.setItem(row, 4, avg_item)

            self._perf_table.setItem(row, 5, QTableWidgetItem(f"{be:.4f}R"))

            above_item = QTableWidgetItem("✅ Yes" if above else "❌ No")
            above_item.setForeground(Qt.GlobalColor.darkGreen if above else Qt.GlobalColor.darkRed)
            self._perf_table.setItem(row, 6, above_item)

            ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]" if tc >= 4 else "N/A (<4)"
            self._perf_table.setItem(row, 7, QTableWidgetItem(ci_str))

    def _refresh_equity(self, snap: dict[str, Any]) -> None:
        """Update the equity curve text display."""
        trades = snap.get("trade_log", [])
        if not trades:
            self._equity_display.setText("No trades yet.")
            return

        # Calculate cumulative net R per symbol
        from collections import defaultdict
        cum_lines = []
        cum_by_sym: dict[str, float] = defaultdict(float)

        sorted_trades = sorted(trades, key=lambda t: t.get("exit_time", 0))
        for t in sorted_trades:
            asset = t.get("asset", "?")
            net_r = t.get("net_result_r", 0.0) or 0.0
            cum_by_sym[asset] += net_r

        for sym in sorted(cum_by_sym.keys()):
            final_r = cum_by_sym[sym]
            cum_lines.append(f"{sym}: {final_r:+.2f}R")

        self._equity_display.setText(
            "Cumulative Net R:\n  " + "\n  ".join(cum_lines)
        )

    def _refresh_kill_switch(self, snap: dict[str, Any]) -> None:
        """Update the kill-switch status panel."""
        # Clear existing
        while self._ks_container.count():
            item = self._ks_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ks_status = snap.get("kill_switch_status", {})
        reg = snap.get("symbol_registry", {})
        symbols = sorted(reg.keys()) if isinstance(reg, dict) else []

        if not symbols:
            label = QLabel("No symbols registered.")
            label.setStyleSheet(f"color: {COLOR_NEUTRAL};")
            self._ks_container.addWidget(label)
            return

        for sym in symbols:
            ks = ks_status.get(sym, {"active": False, "reason": "", "activated_at": 0.0})
            active = ks.get("active", False)
            reason = ks.get("reason", "")
            ks.get("activated_at", 0.0)

            row = QHBoxLayout()
            row.setSpacing(8)

            indicator = "⛔" if active else "✅"
            status_text = "ACTIVE" if active else "Inactive"
            status_color = f"color: {COLOR_NEGATIVE};" if active else f"color: {COLOR_POSITIVE};"

            label = QLabel(f"{indicator} <b>{sym}</b>: {status_text}")
            label.setStyleSheet(f"font-size: 14px; {status_color}")
            row.addWidget(label)

            if active:
                reason_label = QLabel(f"Reason: {reason[:80]}" if reason else "No reason recorded.")
                reason_label.setStyleSheet(f"color: {COLOR_WARNING};")
                row.addWidget(reason_label, stretch=1)

                # Manual deactivate button
                deactivate_btn = QPushButton("✅ Override OFF")
                deactivate_btn.setStyleSheet(f"background-color: {COLOR_POSITIVE}; color: #1e1e1e;")
                deactivate_btn.clicked.connect(lambda checked, s=sym: self._confirm_ks_override(s, False))
                row.addWidget(deactivate_btn)
            else:
                # Manual activate button
                activate_btn = QPushButton("⛔ Override ON")
                activate_btn.setStyleSheet(f"background-color: {COLOR_NEGATIVE}; color: white;")
                activate_btn.clicked.connect(lambda checked, s=sym: self._confirm_ks_override(s, True))
                row.addWidget(activate_btn)

            self._ks_container.addLayout(row)

            # Reason detail
            if reason:
                detail = QLabel(f"  {reason}")
                detail.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px; padding-left: 24px;")
                detail.setWordWrap(True)
                self._ks_container.addWidget(detail)

        self._ks_container.addStretch()

    # ------------------------------------------------------------------
    # Kill-switch override actions
    # ------------------------------------------------------------------

    def _confirm_ks_override(self, symbol: str, activate: bool) -> None:
        """Show confirmation dialog for kill-switch manual override."""
        action = "ACTIVATE" if activate else "DEACTIVATE"
        msg = (
            f"Confirm manual override to <b>{action.lower()}</b> "
            f"kill-switch for <b>{symbol}</b>?\n\n"
            f"{'This will block all new signals for this symbol.' if activate else 'This will re-enable signals for this symbol.'}\n"
            "Type the symbol name below to confirm."
        )
        dialog = ConfirmationDialog(
            title=f"Kill-Switch Override — {action} for {symbol}",
            message=msg,
            confirm_text=f"✅ {action} KILL-SWITCH",
            require_input=True,
            input_match=symbol,
            input_placeholder=f'Type "{symbol}" to confirm',
            parent=self,
        )
        if dialog.exec():
            self._bridge.state.update_kill_switch(
                symbol, active=activate,
                reason=f"Manual override by user: {action.lower()}"
            )
            self._bridge.log_action("override_kill_switch", {
                "symbol": symbol, "action": action.lower()
            })