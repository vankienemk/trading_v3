"""
gui_main.py — Paper Trading V2 Main Window Shell

Brings together all 4 tabs with:
  - Tab switching (QTabWidget)
  - Emergency Stop button (RED, always visible, fixed at bottom)
  - Status bar with MT5/MCP connection status
  - Background polling via QTimer
  - Second confirmation dialog for all destructive actions

Spec compliance:
  - All controls immediate-response (no web reload)
  - Emergency Stop button RED, always visible
  - Every destructive action needs a second confirmation dialog
  - Automation level resets to 0 on every startup
  - Log every user action to logger DB
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from live.gui.gui_bridge import SystemBridge, get_bridge
from live.gui.gui_components import (
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_SURFACE,
    COLOR_TEXT_SEC,
    DARK_STYLESHEET,
    EMERGENCY_RED,
    ConfirmationDialog,
)
from live.gui.gui_tab_account import AccountTab
from live.gui.gui_tab_live import LiveControlTab
from live.gui.gui_tab_onboarding import SymbolOnboardingTab
from live.gui.gui_tab_performance import PerformanceTab
from live.gui.gui_tab_system_log import SystemLogTab

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window with 5 tabs and fixed emergency stop."""

    def __init__(self, bridge: SystemBridge) -> None:
        super().__init__()
        self._bridge = bridge

        # Reset all automation levels to 1 on startup (spec 7.3)
        self._reset_automation_levels()

        self.setWindowTitle("Paper Trading V2 — Liquidity Sweep")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 900)

        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._start_polling()

    def _reset_automation_levels(self) -> None:
        """Reset every symbol's automation level to 1 on startup."""
        for name in self._bridge.state.get_registered_symbols():
            self._bridge.state.set_automation_level(name, 1)

    def _build_ui(self) -> None:
        """Build the complete main window layout."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 6, 8, 6)

        # --- Tab widget (takes most of the space) ---
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setDocumentMode(True)

        # Create the 5 tabs
        self._tab_onboarding = SymbolOnboardingTab(self._bridge)
        self._tab_live_control = LiveControlTab(self._bridge)
        self._tab_performance = PerformanceTab(self._bridge)
        self._tab_account = AccountTab(self._bridge)
        self._tab_system_log = SystemLogTab(self._bridge)

        # Add tabs with emoji labels (spec section 7.2)
        self._tabs.addTab(self._tab_onboarding, "🔬 Symbol Onboarding")
        self._tabs.addTab(self._tab_live_control, "🎮 Live Control")
        self._tabs.addTab(self._tab_performance, "📈 Performance Monitor")
        self._tabs.addTab(self._tab_account, "🔌 Account & Connection")
        self._tabs.addTab(self._tab_system_log, "📋 System Log")

        layout.addWidget(self._tabs)

        # --- Emergency Stop Bar (always visible, fixed at bottom) ---
        e_stop_bar = QHBoxLayout()
        e_stop_bar.setSpacing(12)

        # Emergency stop button — ALWAYS RED, prominent
        self._e_stop_btn = QPushButton("🛑 EMERGENCY STOP")
        self._e_stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {EMERGENCY_RED};
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: 3px solid #cc0000;
                border-radius: 8px;
                padding: 12px 28px;
                min-width: 220px;
            }}
            QPushButton:hover {{
                background-color: #ff3333;
                border-color: {EMERGENCY_RED};
            }}
            QPushButton:pressed {{
                background-color: #cc0000;
            }}
        """)
        self._e_stop_btn.clicked.connect(self._on_emergency_stop)
        e_stop_bar.addWidget(self._e_stop_btn)

        # Status text next to emergency stop
        self._e_stop_status = QLabel("Signal engines running normally.")
        self._e_stop_status.setStyleSheet(f"color: {COLOR_POSITIVE}; font-size: 14px; font-weight: bold;")
        e_stop_bar.addWidget(self._e_stop_status)

        e_stop_bar.addStretch()

        # Time of last refresh
        self._refresh_time_label = QLabel("")
        self._refresh_time_label.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px;")
        e_stop_bar.addWidget(self._refresh_time_label)

        layout.addLayout(e_stop_bar)

        # --- Status bar (bottom) ---
        status = QStatusBar()
        status.setStyleSheet(f"background-color: {COLOR_SURFACE}; color: {COLOR_TEXT_SEC};")
        self._status_mt5 = QLabel("MT5: ❌")
        self._status_mcp = QLabel("MCP: ❌")
        self._status_trades = QLabel("Trades: 0")
        status.addWidget(self._status_mt5)
        status.addPermanentWidget(self._status_mcp)
        status.addPermanentWidget(self._status_trades)
        self.setStatusBar(status)

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _start_polling(self) -> None:
        """Start a 1-second timer to refresh state from MCP."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_poll)
        self._timer.start(1000)  # 1 second interval

    def _on_poll(self) -> None:
        """Periodic poll: signal-engine scan, refresh state, update all tabs."""
        # Kick the background signal engine (registered-symbol scan where the
        # Trigger/Found/Pass counters and pending signals are produced).  It
        # runs on its own daemon thread — this never blocks the GUI.
        self._bridge.poll_once()

        # MCP state is refreshed on a background daemon thread
        # (SystemBridge._mcp_refresh_loop) so we NEVER block the GUI thread on
        # a slow/unreachable MCP server.  We only read the latest snapshot.
        snap = self._bridge.state.get_snapshot()

        # Update emergency stop display
        e_stop = snap.get("emergency_stop", False)
        if e_stop:
            self._e_stop_btn.setText("🚨 EMERGENCY STOP ACTIVE")
            self._e_stop_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #cc0000;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    border: 4px solid {EMERGENCY_RED};
                    border-radius: 8px;
                    padding: 12px 28px;
                    min-width: 220px;
                }}
            """)
            self._e_stop_status.setText("🚨 ALL SIGNAL ENGINES HALTED")
            self._e_stop_status.setStyleSheet(f"color: {EMERGENCY_RED}; font-size: 14px; font-weight: bold;")
        else:
            self._e_stop_btn.setText("🛑 EMERGENCY STOP")
            self._e_stop_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {EMERGENCY_RED};
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    border: 3px solid #cc0000;
                    border-radius: 8px;
                    padding: 12px 28px;
                    min-width: 220px;
                }}
                QPushButton:hover {{
                    background-color: #ff3333;
                    border-color: {EMERGENCY_RED};
                }}
            """)
            self._e_stop_status.setText("Signal engines running normally.")
            self._e_stop_status.setStyleSheet(f"color: {COLOR_POSITIVE}; font-size: 14px; font-weight: bold;")

        # Update status bar
        mt5_ok = snap.get("mt5_status", {}).get("connected", False)
        mcp_state = snap.get("mcp_connection_state", "disconnected")
        trade_count = len(snap.get("trade_log", []))
        self._status_mt5.setText(f"MT5: {'✅' if mt5_ok else '❌'}")
        self._status_mcp.setText(f"MCP: {'✅' if mcp_state == 'connected' else '🔄' if mcp_state == 'reconnecting' else '❌'}")
        self._status_trades.setText(f"Trades: {trade_count}")

        # Update the current tab
        self._tabs.currentIndex()
        tab = self._tabs.currentWidget()
        if hasattr(tab, 'refresh_from_state'):
            tab.refresh_from_state()

        # Update refresh time
        from datetime import datetime
        self._refresh_time_label.setText(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # ------------------------------------------------------------------
    # Emergency Stop (spec requirement: RED button, always visible)
    # ------------------------------------------------------------------

    def _on_emergency_stop(self) -> None:
        """Toggle emergency stop — requires confirmation to activate."""
        if self._bridge.state.emergency_stop:
            # Reset — requires confirmation
            dialog = ConfirmationDialog(
                title="Reset Emergency Stop",
                message="Confirm reset: signal engines will resume.\n"
                        "Open positions will NOT be affected.",
                confirm_text="✅ RESET",
                parent=self,
            )
            if dialog.exec():
                self._bridge.state.set_emergency_stop(False)
                self._bridge.log_action("emergency_stop_toggle",
                                        {"active": False})
        else:
            # Activate — requires confirmation
            dialog = ConfirmationDialog(
                title="🚨 EMERGENCY STOP CONFIRMATION",
                message="⚠️ This will halt ALL signal engines immediately!\n\n"
                        "Open positions will NOT be closed.\n"
                        "No new signals will be generated until reset.\n\n"
                        "Are you absolutely sure?",
                confirm_text="🛑 ACTIVATE EMERGENCY STOP",
                require_input=True,
                input_match="EMERGENCY STOP",
                input_placeholder='Type "EMERGENCY STOP" to confirm',
                parent=self,
            )
            if dialog.exec():
                self._bridge.state.set_emergency_stop(True)
                self._bridge.log_action("emergency_stop_toggle",
                                        {"active": True})

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Clean shutdown on close."""
        self._timer.stop()
        self._bridge.shutdown()
        self._bridge.log_action("gui_shutdown", {})
        event.accept()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the Paper Trading V2 GUI application.

    Returns:
        Exit code (0 = success).
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the bridge (singleton)
    bridge = get_bridge()

    # Create and show the main window
    window = MainWindow(bridge)
    window.show()

    # Run the event loop
    return app.exec()


# Alias for verification compatibility
PaperTradingGUI = MainWindow


if __name__ == "__main__":
    sys.exit(main())