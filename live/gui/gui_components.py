"""
gui_components.py — Shared UI components for the Paper Trading V2 GUI.

Provides:
  1. Global stylesheet (dark theme)
  2. Confirmation dialog factory
  3. Status badge widgets (connected / disconnected / active / inactive)
  4. Color constants matching the v2_frozen.yaml color scheme
  5. Formatted table data helpers
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Color constants (from v2_frozen.yaml gui.colors)
# ---------------------------------------------------------------------------
COLOR_POSITIVE = "#00cc66"
COLOR_NEGATIVE = "#ff4444"
COLOR_WARNING  = "#ffaa00"
COLOR_NEUTRAL  = "#888888"
COLOR_BG       = "#1e1e1e"
COLOR_SURFACE  = "#2d2d2d"
COLOR_TEXT     = "#ffffff"
COLOR_TEXT_SEC = "#aaaaaa"

EMERGENCY_RED = "#ff0000"

# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

DARK_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLOR_BG};
}}
QWidget {{
    color: {COLOR_TEXT};
    font-size: 13px;
}}
QTabWidget::pane {{
    background-color: {COLOR_SURFACE};
    border: 1px solid #3d3d3d;
    border-top: none;
}}
QTabBar::tab {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_SEC};
    padding: 8px 18px;
    border: 1px solid #3d3d3d;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 140px;
}}
QTabBar::tab:selected {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    border-bottom: 2px solid {COLOR_POSITIVE};
}}
QTabBar::tab:hover {{
    background-color: #3a3a3a;
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
    border-color: {COLOR_POSITIVE};
}}
QPushButton:pressed {{
    background-color: #555555;
}}
QPushButton:disabled {{
    background-color: #2a2a2a;
    color: #666666;
    border-color: #3d3d3d;
}}
QTableWidget {{
    background-color: {COLOR_SURFACE};
    alternate-background-color: #333333;
    gridline-color: #3d3d3d;
    border: 1px solid #3d3d3d;
    selection-background-color: #3a6ea5;
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QHeaderView::section {{
    background-color: #3a3a3a;
    color: {COLOR_TEXT};
    padding: 6px;
    border: 1px solid #4a4a4a;
    font-weight: bold;
}}
QLabel {{
    color: {COLOR_TEXT};
}}
QComboBox {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox::drop-down {{
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    selection-background-color: #3a6ea5;
}}
QLineEdit {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}}
QTextEdit {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid #555555;
}}
QGroupBox {{
    color: {COLOR_TEXT};
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 8px;
}}
QScrollBar:vertical {{
    background-color: {COLOR_BG};
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background-color: #555555;
    min-height: 20px;
    border-radius: 4px;
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QProgressBar {{
    background-color: {COLOR_SURFACE};
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {COLOR_POSITIVE};
    border-radius: 3px;
}}
QMessageBox {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
}}
QMessageBox QLabel {{
    color: {COLOR_TEXT};
}}
QMessageBox QPushButton {{
    background-color: #3a3a3a;
    color: {COLOR_TEXT};
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 24px;
    min-width: 80px;
}}
QMessageBox QPushButton:hover {{
    background-color: #4a4a4a;
}}
"""

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def fmt_time(ts: float) -> str:
    """Format a Unix timestamp."""
    if ts == 0:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_r(val: float) -> str:
    """Format an R-multiple value with sign."""
    if val >= 0:
        return f"+{val:.2f}R"
    return f"{val:.2f}R"


# ---------------------------------------------------------------------------
# Confirmation dialog factory
# ---------------------------------------------------------------------------

class ConfirmationDialog(QDialog):
    """A modal confirmation dialog with OK / Cancel.

    Shows a title, message body, and optional text input field (e.g.
    for typing a symbol name or confirmation word).

    Args:
        title: Dialog window title.
        message: Main text body (supports \n).
        confirm_text: Label for the confirm button (default "Confirm").
        require_input: If True, the confirm button is disabled until
                       the user types the exact word in `input_match`.
        input_match: The exact text the user must type to enable the
                     confirm button. If empty, any non-empty input
                     is accepted.
        input_placeholder: Placeholder text for the input field.
    """

    def __init__(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        require_input: bool = False,
        input_match: str = "",
        input_placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)
        self.setModal(True)

        # Dark theme background for dialog
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLOR_SURFACE};
                color: {COLOR_TEXT};
            }}
            QLabel {{
                color: {COLOR_TEXT};
                font-size: 13px;
            }}
            QLineEdit {{
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

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Message label
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 13px;")
        layout.addWidget(msg_label)

        # Optional input
        self._input_field: QLineEdit | None = None
        self._require_input = require_input
        self._input_match = input_match

        if require_input:
            self._input_field = QLineEdit()
            self._input_field.setPlaceholderText(input_placeholder)
            layout.addWidget(self._input_field)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok
        )
        confirm_btn = button_box.button(QDialogButtonBox.Ok)
        if confirm_btn:
            confirm_btn.setText(confirm_text)
            confirm_btn.setStyleSheet(
                f"background-color: {COLOR_NEGATIVE}; color: white; font-weight: bold;"
            )

        layout.addWidget(button_box)

        if require_input and self._input_field:
            confirm_btn.setEnabled(False)
            self._input_field.textChanged.connect(
                lambda text: confirm_btn.setEnabled(
                    text.strip() == (input_match if input_match else text.strip())
                )
            )

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

    @property
    def input_text(self) -> str:
        """Return the text entered in the input field (if any)."""
        if self._input_field:
            return self._input_field.text().strip()
        return ""


# ---------------------------------------------------------------------------
# Status badge helpers
# ---------------------------------------------------------------------------

def status_badge_connected() -> str:
    """HTML for a green 'Connected' badge."""
    return '<span style="color: #00cc66; font-weight: bold;">● Connected</span>'


def status_badge_disconnected() -> str:
    """HTML for a red 'Disconnected' badge."""
    return '<span style="color: #ff4444; font-weight: bold;">● Disconnected</span>'


def status_badge_reconnecting() -> str:
    """HTML for a yellow 'Reconnecting' badge."""
    return '<span style="color: #ffaa00; font-weight: bold;">● Reconnecting</span>'


def status_badge_kill_active() -> str:
    """HTML for a red 'Kill-Switch ACTIVE' badge."""
    return '<span style="background-color: #ff4444; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;">⛔ ACTIVE</span>'


def status_badge_kill_inactive() -> str:
    """HTML for a green 'Kill-Switch Inactive' badge."""
    return '<span style="background-color: #00cc66; color: white; padding: 2px 8px; border-radius: 3px; font-weight: bold;">✅ Inactive</span>'


# ---------------------------------------------------------------------------
# Auto-refresh timer label helper
# ---------------------------------------------------------------------------

def build_refresh_label(elapsed_s: int, interval_s: int) -> str:
    """Build a formatted 'Refresh in N s' string."""
    remaining = max(0, interval_s - elapsed_s)
    return f"Auto-refresh in {remaining}s  |  Interval: {interval_s}s"


# ---------------------------------------------------------------------------
# Emergency Stop Button factory
# ---------------------------------------------------------------------------

def make_emergency_stop_button() -> QPushButton:
    """Create a RED emergency stop button with bold text."""
    btn = QPushButton("🛑 EMERGENCY STOP")
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {EMERGENCY_RED};
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #cc0000;
            border-radius: 6px;
            padding: 12px 24px;
            min-height: 40px;
        }}
        QPushButton:hover {{
            background-color: #cc0000;
            border-color: #ff6666;
        }}
        QPushButton:pressed {{
            background-color: #990000;
        }}
        """
    )
    return btn


def make_emergency_reset_button() -> QPushButton:
    """Create a 'Reset' button for when emergency stop is active."""
    btn = QPushButton("🔄 Reset Emergency Stop")
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {COLOR_WARNING};
            color: #1e1e1e;
            font-size: 14px;
            font-weight: bold;
            border: 2px solid #cc8800;
            border-radius: 6px;
            padding: 12px 24px;
            min-height: 40px;
        }}
        QPushButton:hover {{
            background-color: #eebb00;
        }}
        """
    )
    return btn


# ---------------------------------------------------------------------------
# Automation level helpers
# ---------------------------------------------------------------------------

AUTO_LABELS = {0: "Manual (Level 0)", 1: "Semi-Auto (Level 1)", 2: "Full Auto (Level 2)"}


def automation_color(level: int) -> str:
    """Return the hex color for a given automation level."""
    if level == 0:
        return COLOR_WARNING  # yellow — manual
    elif level == 1:
        return COLOR_POSITIVE  # green — semi-auto
    elif level == 2:
        return COLOR_NEGATIVE  # red — full auto
    return COLOR_NEUTRAL


def automation_stylesheet(level: int) -> str:
    """Return a stylesheet snippet for a label showing automation level."""
    return f"color: {automation_color(level)}; font-weight: bold;"