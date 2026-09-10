"""
gui_tab_account.py — Tab 4: Account & Connection

Spec section 7.2 Tab 4:
  - MT5 account info display: account number, type, server, balance
  - RED fullscreen warning if account is NOT demo (blocks other operations)
  - MCP connection status (connected/disconnected/reconnecting)
  - Reconnect button (manual MCP reconnect)
  - MCP token input field (update token without restart)

This tab should be the first functional tab on startup (spec step 3):
the user must confirm the demo account before any trading can begin.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from live.gui.gui_bridge import SystemBridge
from live.gui.gui_components import (
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_TEXT,
    COLOR_TEXT_SEC,
    COLOR_WARNING,
    EMERGENCY_RED,
)
from live.logging.logger_v2 import log as syslog

# ---------------------------------------------------------------------------
# Real Account Warning Dialog (RED fullscreen overlay)
# ---------------------------------------------------------------------------

class RealAccountWarning(QDialog):
    """Full-viewport RED warning dialog for non-demo accounts.

    Blocks all other operations until the user confirms or stops.
    """

    def __init__(self, account_type: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚠️ WARNING: REAL ACCOUNT DETECTED")
        self.setModal(True)
        self.setMinimumSize(600, 400)

        # Solid red background
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {EMERGENCY_RED};
                border: 4px solid #cc0000;
            }}
            QLabel {{
                color: white;
                font-size: 16px;
                padding: 10px;
            }}
            QPushButton {{
                background-color: white;
                color: #cc0000;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #cc0000;
                border-radius: 8px;
                padding: 12px 24px;
                min-width: 200px;
            }}
            QPushButton:hover {{
                background-color: #ffeeee;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Big warning text
        warning_title = QLabel("⛔ REAL TRADING ACCOUNT DETECTED")
        warning_title.setStyleSheet("font-size: 28px; font-weight: bold; color: white;")
        warning_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(warning_title)

        message = QLabel(
            f"This system is designed for <b>PAPER TRADING</b> only.\n\n"
            f"Account type detected: <b>{account_type.upper()}</b>\n\n"
            f"If this is a REAL account, STOP IMMEDIATELY.\n"
            f"The system will NOT place real orders, but please confirm below."
        )
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(message)

        layout.addSpacing(30)

        # Confirm button
        confirm_btn = QPushButton("✅ I CONFIRM THIS IS A DEMO ACCOUNT")
        confirm_btn.clicked.connect(self.accept)
        layout.addWidget(confirm_btn, alignment=Qt.AlignCenter)

        # Stop button
        stop_btn = QPushButton("❌ STOP — THIS IS A REAL ACCOUNT")
        stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid white;
                border-radius: 8px;
                padding: 12px 24px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #ff2222;
            }
        """)
        stop_btn.clicked.connect(self.reject)
        layout.addWidget(stop_btn, alignment=Qt.AlignCenter)


class AccountTab(QWidget):
    """Tab 4: Account & Connection — MT5, MCP, and token settings."""

    def __init__(self, bridge: SystemBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 10, 12, 10)

        # --- Header ---
        header = QLabel("🔌 Account & Connection")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(header)

        desc = QLabel(
            "MT5 account information, MCP connection management, and security settings."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_NEUTRAL}; padding-bottom: 4px;")
        layout.addWidget(desc)

        # --- Split: Account info (left) | MCP Status (right) ---
        split_row = QHBoxLayout()
        split_row.setSpacing(12)

        # LEFT: Account info
        acct_group = QGroupBox("💳 MT5 Account Info")
        acct_layout = QVBoxLayout(acct_group)
        self._acct_info = QLabel("Not connected. Click Connect MCP.")
        self._acct_info.setWordWrap(True)
        self._acct_info.setStyleSheet(f"color: {COLOR_TEXT_SEC}; font-family: monospace; font-size: 13px;")
        acct_layout.addWidget(self._acct_info)
        split_row.addWidget(acct_group, stretch=1)

        # RIGHT: MCP Connection
        mcp_group = QGroupBox("🔗 MCP Connection")
        mcp_layout = QVBoxLayout(mcp_group)

        self._mcp_status_label = QLabel("Status: disconnected")
        self._mcp_status_label.setStyleSheet(f"color: {COLOR_NEGATIVE}; font-size: 14px;")
        mcp_layout.addWidget(self._mcp_status_label)

        self._reconnect_btn = QPushButton("🔄 Connect / Reconnect MCP")
        self._reconnect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_WARNING};
                color: #1e1e1e;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #ffbb33;
            }}
        """)
        self._reconnect_btn.clicked.connect(self._on_reconnect)
        mcp_layout.addWidget(self._reconnect_btn)

        mcp_layout.addStretch()
        split_row.addWidget(mcp_group, stretch=1)

        layout.addLayout(split_row)

        # --- Token Input ---
        token_group = QGroupBox("🔑 MCP Token")
        token_layout = QHBoxLayout(token_group)

        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText(
            "Enter new MCP token (currently: " +
            ("set" if self._bridge.state.mcp_token else "not set") + ")")
        self._token_input.setEchoMode(QLineEdit.Password)
        self._token_input.setMinimumWidth(300)
        token_layout.addWidget(self._token_input)

        self._update_token_btn = QPushButton("Update Token")
        self._update_token_btn.clicked.connect(self._on_update_token)
        token_layout.addWidget(self._update_token_btn)

        layout.addWidget(token_group)

        # --- Warning area ---
        self._warning_area = QLabel("")
        self._warning_area.setWordWrap(True)
        layout.addWidget(self._warning_area)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Periodic refresh
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Called periodically by the main window timer."""
        snap = self._bridge.state.get_snapshot()
        self._refresh_account_info(snap)
        self._refresh_mcp_status(snap)

    def _refresh_account_info(self, snap: dict[str, Any]) -> None:
        """Update the account info display."""
        mt5 = snap.get("mt5_status", {})
        connected = mt5.get("connected", False)
        acct_info = mt5.get("account_info", {})
        acct_type = mt5.get("account_type", "unknown")

        if not connected:
            self._acct_info.setText("❌ Not connected to MT5.\nClick 'Connect / Reconnect MCP' above.")
            self._acct_info.setStyleSheet(f"color: {COLOR_NEGATIVE}; font-family: monospace;")
            return

        lines = [
            f"Account Number:  {acct_info.get('login', acct_info.get('number', 'N/A'))}",
            f"Account Type:    {acct_type.upper()}",
            f"Server:          {acct_info.get('server', acct_info.get('company', 'N/A'))}",
            f"Balance:         ${acct_info.get('balance', 0):,.2f}",
            f"Equity:          ${acct_info.get('equity', 0):,.2f}",
            f"Margin:          ${acct_info.get('margin', 0):,.2f}",
            f"Free Margin:     ${acct_info.get('margin_free', 0):,.2f}",
            f"Leverage:        1:{acct_info.get('leverage', 0)}",
            f"Name:            {acct_info.get('name', acct_info.get('company', 'N/A'))}",
            f"Currency:        {acct_info.get('currency', 'USD')}",
        ]
        self._acct_info.setText("\n".join(lines))
        self._acct_info.setStyleSheet(f"color: {COLOR_TEXT}; font-family: monospace; font-size: 13px;")

        # Check for real account
        is_demo = "demo" in str(acct_type).lower()
        if not is_demo:
            self._show_real_account_warning(acct_type)

    def _refresh_mcp_status(self, snap: dict[str, Any]) -> None:
        """Update the MCP connection status display."""
        mcp_state = snap.get("mcp_connection_state", "disconnected")
        snap.get("mt5_status", {}).get("connected", False)

        status_colors = {
            "connected": COLOR_POSITIVE,
            "disconnected": COLOR_NEGATIVE,
            "reconnecting": COLOR_WARNING,
        }
        status_icons = {
            "connected": "✅",
            "disconnected": "❌",
            "reconnecting": "🔄",
        }
        color = status_colors.get(mcp_state, COLOR_NEUTRAL)
        icon = status_icons.get(mcp_state, "?")
        self._mcp_status_label.setText(f"{icon} Status: {mcp_state}")
        self._mcp_status_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_reconnect(self) -> None:
        """Attempt MCP reconnection."""
        self._reconnect_btn.setText("⏳ Reconnecting...")
        self._reconnect_btn.setEnabled(False)

        syslog("INFO", "MCP", "", "User initiated MCP reconnect from Account tab")
        try:
            success = self._bridge.reconnect_mcp()
            self._bridge.log_action("reconnect_mcp", {"success": success})
        except Exception as e:
            success = False
            syslog("ERROR", "MCP", "", f"MCP reconnect failed with exception: {e}")

        if success:
            self._reconnect_btn.setText("✅ Connected")
        else:
            self._reconnect_btn.setText("🔄 Reconnect MCP")
            self._reconnect_btn.setEnabled(True)

    def _on_update_token(self) -> None:
        """Update the MCP token from the input field."""
        new_token = self._token_input.text().strip()
        if not new_token:
            self._update_token_btn.setText("⚠️ Cannot be empty")
            return

        # 1) Save to state
        self._bridge.state.set_mcp_token(new_token)
        self._bridge.log_action("set_mcp_token", {"token_set": True})
        syslog("INFO", "MCP", "", "MCP token updated by user")

        # 2) Push live to ExecutionLayer (creates/reconnects the MCP client)
        if self._bridge._exec_layer is not None:
            self._bridge._exec_layer.set_token(new_token)

        self._token_input.clear()
        self._token_input.setPlaceholderText(
            "Enter new MCP token (currently: set)")
        self._update_token_btn.setText("✅ Updated")
        # Refresh the MCP status display
        self._refresh_mcp_status(self._bridge.state.get_snapshot())

    # ------------------------------------------------------------------
    # Real account warning
    # ------------------------------------------------------------------

    def _show_real_account_warning(self, acct_type: str) -> None:
        """Show RED fullscreen warning if account is not demo."""
        warning = RealAccountWarning(acct_type, self)
        result = warning.exec()
        if result == QDialog.Accepted:
            # User confirmed demo
            self._warning_area.setText("")
            syslog("INFO", "System", "", "User confirmed demo account in RealAccountWarning")
        else:
            # User wants to stop
            self._bridge.state.set_emergency_stop(True)
            self._bridge.log_action("emergency_stop_toggle",
                                    {"reason": "User reported REAL account", "active": True})
            syslog("CRITICAL", "System", "",
                   "EMERGENCY STOP ACTIVATED — user reported REAL account",
                   extra={"account_type": acct_type})
            self._warning_area.setStyleSheet(f"color: {EMERGENCY_RED}; font-size: 16px; font-weight: bold;")
            self._warning_area.setText(
                "⛔ SYSTEM PAUSED — Real account detected.\n"
                "Emergency stop activated. All signal engines halted.\n"
                "Close the application and reconnect with a demo account."
            )