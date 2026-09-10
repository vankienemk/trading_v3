"""
gui_tab_onboarding.py — Tab 1: Symbol Onboarding

Implements:
  - Symbol registry table: Symbol | Status | Giá (Bid) | Model đang dùng |
    Active | Risk (Size) | Actions
  - "Add New Symbol" dialog: QComboBox for MCP symbols + QComboBox for models
    + Browse local model button (custom path)
  - "Change Model" dialog for validated symbols
  - Activation / deactivation toggle
  - Per-symbol risk setting (position size multiplier) for active symbols
  - Only validated/candidate/rejected statuses (no intermediate wizard steps)
"""

from __future__ import annotations

from typing import Any, ClassVar

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from live.gui.gui_bridge import SystemBridge
from live.gui.gui_components import (
    COLOR_BG,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_WARNING,
    ConfirmationDialog,
)
from live.logging.logger_v2 import log as syslog
from live.state.shared_app_state_v2 import (
    MAX_POSITION_SIZE_MULTIPLIER,
    MIN_POSITION_SIZE_MULTIPLIER,
)

# Shorter aliases used by the risk editor UI.
MIN_RISK_MULTIPLIER = MIN_POSITION_SIZE_MULTIPLIER
MAX_RISK_MULTIPLIER = MAX_POSITION_SIZE_MULTIPLIER


# ---------------------------------------------------------------------------
# Dark-theme helper for dialogs
# ---------------------------------------------------------------------------

def _style_dialog(dialog: QDialog) -> None:
    """Apply dark theme stylesheet to a QDialog."""
    dialog.setStyleSheet(f"""
        QDialog {{
            background-color: {COLOR_SURFACE};
            color: {COLOR_TEXT};
        }}
        QLabel {{
            color: {COLOR_TEXT};
        }}
        QComboBox {{
            background-color: {COLOR_BG};
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
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
            selection-background-color: #3a6ea5;
        }}
        QLineEdit {{
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 24px;
        }}
        QDoubleSpinBox {{
            background-color: {COLOR_BG};
            color: {COLOR_TEXT};
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px 8px;
            min-height: 24px;
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: #3a3a3a;
            border: none;
            width: 18px;
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


# ---------------------------------------------------------------------------
# Bid price lookup with case-insensitive matching and 'm' suffix fallback
# ---------------------------------------------------------------------------

def _lookup_bid_price(symbol: str, price_map: dict[str, float]) -> float | None:
    """Look up bid price for *symbol* in *price_map* using case-insensitive
    matching with 'm' suffix fallback.

    Matching order:
    1. Exact match (case-insensitive).
    2. Registry name + 'm' suffix (e.g. ``XAUUSD`` → ``XAUUSDm``).
    3. If registry name ends with 'm', try without the 'm'.
    4. Partial match — one is a case-insensitive prefix of the other.
    """
    if not symbol or not price_map:
        return None

    # 1. Exact match (case-insensitive)
    sym_lower = symbol.lower()
    for k, v in price_map.items():
        if k.lower() == sym_lower:
            return v

    # 2. Try registry name + 'm' suffix
    if not sym_lower.endswith("m"):
        m_key = symbol + "m"
        for k, v in price_map.items():
            if k.lower() == m_key.lower():
                return v

    # 3. If registry name ends with 'm', try without it
    if sym_lower.endswith("m"):
        no_m = symbol[:-1]
        for k, v in price_map.items():
            if k.lower() == no_m.lower():
                return v

    # 4. Partial prefix match (case-insensitive, no uppercasing)
    for k, v in price_map.items():
        kl = k.lower()
        if kl.startswith(sym_lower) or sym_lower.startswith(kl):
            return v

    return None


class SymbolOnboardingTab(QWidget):
    """Tab 1: Symbol Onboarding — registry table with model selection."""

    def __init__(self, bridge: SystemBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._build_ui()
        self._refresh_symbols()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 12)

        # --- Header ---
        header = QLabel("📋 Symbol Onboarding")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(header)

        desc = QLabel(
            "Manage symbols and assign models. "
            "Only symbols with status <b>validated</b> and a model assigned "
            "receive live signal engines."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_NEUTRAL}; padding-bottom: 8px;")
        layout.addWidget(desc)

        # --- Button row ---
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Add New Symbol")
        self._add_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLOR_POSITIVE};
                color: #1e1e1e;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 4px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #00dd77;
            }}
            """
        )
        self._add_btn.clicked.connect(self._on_add_symbol)
        btn_row.addWidget(self._add_btn)

        self._reload_btn = QPushButton("🔄 Reload Model Registry")
        self._reload_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLOR_SURFACE};
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #3a3a3a;
            }}
            """
        )
        self._reload_btn.clicked.connect(self._on_reload_registry)
        btn_row.addWidget(self._reload_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # --- Pattern Assignments (§10.1 multi-pattern): per-symbol rows of
        # (pattern, TF, lifecycle state, model).  A symbol may carry MULTIPLE
        # pattern assignments — the LiveEngine consumes them via
        # MultiPatternEngine (spec §9.1). ---
        pat_group = QGroupBox("🧩 Pattern Assignments (multi-pattern §10.1)")
        pat_layout = QVBoxLayout(pat_group)
        self._pattern_table = QTableWidget(0, 6)
        self._pattern_table.setAlternatingRowColors(True)
        self._pattern_table.setHorizontalHeaderLabels(
            ["Symbol", "Pattern", "TF", "State", "Model", "Lifecycle"]
        )
        self._pattern_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._pattern_table.verticalHeader().setVisible(False)
        self._pattern_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._pattern_table.setSelectionMode(QTableWidget.SingleSelection)
        self._pattern_table.setEditTriggers(QTableWidget.NoEditTriggers)
        pat_layout.addWidget(self._pattern_table)
        pat_btn_row = QHBoxLayout()
        add_pat_btn = QPushButton("+ Assign Pattern")
        add_pat_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_SURFACE}; color: white; "
            "padding: 6px 14px; border-radius: 4px; }}"
        )
        add_pat_btn.clicked.connect(self._on_add_assignment)
        pat_btn_row.addWidget(add_pat_btn)
        rm_pat_btn = QPushButton("🗑 Remove Selected")
        rm_pat_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLOR_SURFACE}; color: white; "
            "padding: 6px 14px; border-radius: 4px; }}"
        )
        rm_pat_btn.clicked.connect(self._on_remove_assignment)
        pat_btn_row.addWidget(rm_pat_btn)
        pat_btn_row.addStretch()
        pat_layout.addLayout(pat_btn_row)
        layout.addWidget(pat_group)

        # --- Symbol table: Symbol | Status | Giá (Bid) | Model đang dùng |
        #                 Active | Risk (Size) | Actions ---
        self._table = QTableWidget(0, 7)
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalHeaderLabels(
            ["Symbol", "Status", "Giá (Bid)", "Model đang dùng", "Active",
             "Risk (Size)", "Actions"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Interactive)
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Interactive)
        self._table.setColumnWidth(0, 130)
        self._table.setColumnWidth(2, 100)
        self._table.setColumnWidth(5, 130)
        self._table.setColumnWidth(6, 350)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, stretch=1)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_symbols(self) -> None:
        """Refresh the symbol table from shared state."""
        snapshot = self._bridge.state.get_snapshot()
        registry = snapshot.get("symbol_registry", {})

        # Pre-load model display names
        models = self._bridge.get_available_models()
        model_names: dict[str, str] = {}
        for m in models:
            model_names[m.model_id] = f"{m.model_id} ({m.symbol_origin})"

        # Fetch current bid prices from MCP
        price_map: dict[str, float] = {}
        try:
            price_map = self._bridge.fetch_symbol_prices()
        except Exception:
            pass

        self._table.setRowCount(len(registry))

        for row, (name, cfg) in enumerate(sorted(registry.items())):
            # Symbol
            self._table.setItem(row, 0, QTableWidgetItem(name))

            # Status with color
            status_item = QTableWidgetItem(cfg["status"])
            status_color = {
                "validated": COLOR_POSITIVE,
                "candidate": COLOR_WARNING,
                "rejected": COLOR_NEGATIVE,
            }.get(cfg["status"], COLOR_NEUTRAL)
            status_item.setForeground(QColor(status_color))
            self._table.setItem(row, 1, status_item)

            # Giá (Bid) — case-insensitive matching with 'm' suffix fallback
            bid_price = _lookup_bid_price(name, price_map)
            price_text = f"{bid_price:.5f}" if bid_price is not None else "—"
            price_item = QTableWidgetItem(price_text)
            price_item.setForeground(QColor(COLOR_TEXT))
            self._table.setItem(row, 2, price_item)

            # Model đang dùng (now column 3)
            model_id = cfg.get("model_id") or ""
            model_display = model_names.get(model_id, model_id) if model_id else "—"
            model_item = QTableWidgetItem(model_display)
            model_item.setForeground(
                QColor(COLOR_POSITIVE) if model_id else QColor(COLOR_NEUTRAL)
            )
            self._table.setItem(row, 3, model_item)

            # Active (now column 4)
            active_item = QTableWidgetItem("✅ Yes" if cfg.get("active", False) else "❌ No")
            active_item.setForeground(
                QColor(COLOR_POSITIVE) if cfg.get("active", False) else QColor(COLOR_NEUTRAL)
            )
            self._table.setItem(row, 4, active_item)

            # Risk (Size) — column 5.  Active validated symbols get a "🛡 Risk"
            # button (opens the position-size edit dialog); every other symbol
            # shows its current multiplier read-only.
            multiplier = float(cfg.get("position_size_multiplier", 1.0))
            if cfg["status"] == "validated" and cfg.get("active", False):
                risk_widget = QWidget()
                risk_layout = QHBoxLayout(risk_widget)
                risk_layout.setContentsMargins(4, 0, 4, 0)
                risk_layout.setSpacing(4)
                risk_btn = QPushButton(f"🛡 {multiplier:.1f}x")
                risk_btn.setToolTip(
                    f"Set position size risk for {name} "
                    f"(order size = base x multiplier, current {multiplier:.1f}x)"
                )
                risk_btn.setStyleSheet(
                    "background-color: #3a6ea5; color: white; padding: 2px 6px; "
                    "font-weight: bold; font-size: 11px; border-radius: 4px;"
                )
                risk_btn.clicked.connect(
                    lambda checked, s=name: self._on_set_risk(s)
                )
                risk_layout.addWidget(risk_btn)
                risk_layout.addStretch()
                self._table.setCellWidget(row, 5, risk_widget)
            else:
                risk_item = QTableWidgetItem(f"{multiplier:.1f}x")
                if cfg["status"] == "candidate":
                    risk_item.setForeground(QColor(COLOR_WARNING))
                elif cfg["status"] == "rejected":
                    risk_item.setForeground(QColor(COLOR_NEGATIVE))
                else:  # validated but not active
                    risk_item.setForeground(QColor(COLOR_NEUTRAL))
                risk_item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row, 5, risk_item)

            # Actions row (column 6)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(4, 0, 4, 0)

            # Change Model button (for validated symbols only)
            if cfg["status"] == "validated":
                change_model_btn = QPushButton("🔧 Model")
                change_model_btn.setStyleSheet(
                    f"background-color: {COLOR_WARNING}; color: #1e1e1e; padding: 2px 6px; font-weight: bold; font-size: 11px;"
                )
                change_model_btn.clicked.connect(
                    lambda checked, s=name: self._on_change_model(s)
                )
                actions_layout.addWidget(change_model_btn)

            # Deactivate / Activate toggle
            if cfg.get("active", False):
                deactivate_btn = QPushButton("Deact")
                deactivate_btn.setStyleSheet(
                    f"background-color: {COLOR_NEGATIVE}; color: white; padding: 2px 6px; font-weight: bold; font-size: 11px;"
                )
                deactivate_btn.clicked.connect(
                    lambda checked, s=name: self._on_deactivate(s)
                )
                actions_layout.addWidget(deactivate_btn)
            else:
                activate_btn = QPushButton("Activate")
                activate_btn.setStyleSheet(
                    f"background-color: {COLOR_POSITIVE}; color: #1e1e1e; padding: 2px 6px; font-weight: bold; font-size: 11px;"
                )
                activate_btn.clicked.connect(
                    lambda checked, s=name: self._on_activate(s)
                )
                actions_layout.addWidget(activate_btn)

            # Change Status button (always visible)
            change_status_btn = QPushButton("Status")
            change_status_btn.setStyleSheet(
                f"background-color: {COLOR_NEUTRAL}; color: white; padding: 2px 6px; font-size: 11px;"
            )
            change_status_btn.clicked.connect(
                lambda checked, s=name: self._on_change_status(s)
            )
            actions_layout.addWidget(change_status_btn)

            # Remove button (with strong confirmation)
            remove_btn = QPushButton("🗑")
            remove_btn.setStyleSheet(
                f"background-color: {COLOR_NEGATIVE}; color: white; padding: 2px 6px; font-weight: bold; font-size: 11px;"
            )
            remove_btn.clicked.connect(
                lambda checked, s=name: self._on_remove_symbol(s)
            )
            actions_layout.addWidget(remove_btn)

            actions_layout.addStretch()
            self._table.setCellWidget(row, 6, actions_widget)

        self._table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Add New Symbol
    # ------------------------------------------------------------------

    def _on_add_symbol(self) -> None:
        """Open 'Add New Symbol' dialog with symbol + model selection."""
        # Fetch available MCP symbols
        raw_symbols = self._bridge.fetch_symbols()
        if not raw_symbols:
            QMessageBox.warning(
                self,
                "No Symbols Available",
                "No symbols are available from MCP/MT5. "
                "Please ensure MCP is connected and MT5 market watch has symbols.",
            )
            return

        # Normalize to string list
        available_symbols: list[str] = []
        for s in raw_symbols:
            if isinstance(s, str):
                available_symbols.append(s)
            elif isinstance(s, dict):
                available_symbols.append(s.get("symbol", s.get("name", "")))
        available_symbols = sorted([s for s in available_symbols if s])

        # Filter out already-registered symbols
        registered = set(self._bridge.state.get_registered_symbols())
        unregistered = sorted(
            [s for s in available_symbols
             if s.upper() not in {r.upper() for r in registered}]
        )

        if not unregistered:
            QMessageBox.information(
                self,
                "All Symbols Registered",
                "All available symbols from MCP are already in the registry.",
            )
            return

        # Fetch assignable models from Model Registry (§10.1 lifecycle filter)
        models = self._bridge.get_models_for_assignment(
            lifecycle_states={"validated", "shadow", "live"},
        )

        # Build dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Symbol")
        dialog.setMinimumWidth(480)
        _style_dialog(dialog)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        # --- Symbol selection ---
        symbol_label = QLabel("Select symbol from MCP market watch:")
        symbol_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(symbol_label)

        symbol_combo = QComboBox()
        symbol_combo.addItems(unregistered)
        symbol_combo.setEditable(False)
        layout.addWidget(symbol_combo)

        # --- Model selection ---
        model_label = QLabel("Select model from registry:")
        model_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(model_label)

        model_combo = QComboBox()
        model_combo.addItem("— No model (select later) —", None)
        for m in models:
            display = f"{m.model_id} — {m.symbol_origin} (h{m.horizon})"
            model_combo.addItem(display, m.model_id)
        model_combo.setEditable(False)
        layout.addWidget(model_combo)

        # --- Browse local model ---
        browse_label = QLabel("Or browse a local model path (optional):")
        browse_label.setStyleSheet(f"color: {COLOR_NEUTRAL}; margin-top: 8px;")
        layout.addWidget(browse_label)

        browse_row = QHBoxLayout()
        self._browse_path_input = QLineEdit()
        self._browse_path_input.setPlaceholderText("Path to model file (e.g. .pkl)...")
        browse_row.addWidget(self._browse_path_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse_model)
        browse_row.addWidget(browse_btn)
        layout.addLayout(browse_row)

        # --- Status selection ---
        status_label = QLabel("Initial status:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(status_label)

        status_combo = QComboBox()
        status_combo.addItems(["validated", "candidate", "rejected"])
        status_combo.setCurrentText("candidate")
        layout.addWidget(status_combo)

        # --- Activate checkbox ---
        activate_after = QLabel("(Status 'validated' + model assigned = engine-ready)")
        activate_after.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px;")
        layout.addWidget(activate_after)

        # --- Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        confirm_btn = button_box.button(QDialogButtonBox.Ok)
        confirm_btn.setText("Add Symbol")
        confirm_btn.setStyleSheet(
            "background-color: #00cc66; color: #1e1e1e; font-weight: bold;"
        )
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            name = symbol_combo.currentText().strip()
            if not name:
                return

            selected_model_id = model_combo.currentData()
            status = status_combo.currentText().strip()
            active = (status == "validated" and selected_model_id is not None)

            from live.state.shared_app_state_v2 import SymbolConfig
            config = SymbolConfig(
                name=name,
                status=status,
                active=active,
                model_id=selected_model_id,
            )
            self._bridge.state.register_symbol(name, config)
            self._bridge.log_action("add_symbol", {"symbol": name, "model_id": selected_model_id,
                                                    "status": status})
            syslog("INFO", "System", name,
                   f"Symbol added: status={status}, model={selected_model_id or 'none'}, active={active}",
                   extra={"status": status, "model_id": selected_model_id, "active": active})

            # If a local path was browsed, log it (registry does not store path yet)
            local_path = self._browse_path_input.text().strip()
            if local_path:
                self._bridge.log_action("add_symbol_local_path",
                                        {"symbol": name, "path": local_path})

            self._refresh_symbols()

    def _on_browse_model(self) -> None:
        """Open a file dialog to pick a local model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pkl *.pt *.h5 *.onnx *.joblib *.zip);;All Files (*)",
        )
        if path:
            self._browse_path_input.setText(path)

    # ------------------------------------------------------------------
    # Change Model
    # ------------------------------------------------------------------

    def _on_change_model(self, symbol: str) -> None:
        """Open a dialog to change the model assigned to a validated symbol."""
        # Fetch assignable models (§10.1 lifecycle filter)
        models = self._bridge.get_models_for_assignment(
            lifecycle_states={"validated", "shadow", "live"},
        )
        if not models:
            QMessageBox.warning(
                self,
                "No Models Available",
                "No assignable models are registered in the Model Registry "
                "(lifecycle validated/shadow/live). "
                "Please add models to model_registry/index.yaml and reload.",
            )
            return

        current_cfg = self._bridge.state.get_symbol_config(symbol)
        current_model_id = current_cfg.model_id if current_cfg else None

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Change Model — {symbol}")
        dialog.setMinimumWidth(440)
        _style_dialog(dialog)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        msg = QLabel(f"Select a new model for <b>{symbol}</b>:")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        if current_model_id:
            current_label = QLabel(f"Current model: <b>{current_model_id}</b>")
            current_label.setStyleSheet(f"color: {COLOR_NEUTRAL};")
            layout.addWidget(current_label)

        model_combo = QComboBox()
        model_combo.addItem("— No model —", None)
        for m in self._bridge.get_models_for_assignment(
            lifecycle_states={"validated", "shadow", "live"},
        ):
            display = f"{m.model_id} — {m.symbol_origin} (h{m.horizon}, {m.lifecycle_state})"
            model_combo.addItem(display, m.model_id)
            # Pre-select current model
            if m.model_id == current_model_id:
                model_combo.setCurrentIndex(model_combo.count() - 1)
        layout.addWidget(model_combo)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        confirm_btn = button_box.button(QDialogButtonBox.Ok)
        confirm_btn.setText("Change Model")
        confirm_btn.setStyleSheet(
            "background-color: #ffaa00; color: #1e1e1e; font-weight: bold;"
        )
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            new_model_id = model_combo.currentData()
            if new_model_id == current_model_id:
                return  # No change

            success = self._bridge.assign_model_to_symbol(symbol, new_model_id)
            if success:
                self._bridge.log_action("change_model",
                                        {"symbol": symbol,
                                         "old_model": current_model_id,
                                         "new_model": new_model_id})
                syslog("INFO", "System", symbol,
                       f"Model changed: {current_model_id or 'none'} → {new_model_id}",
                       extra={"old_model": current_model_id, "new_model": new_model_id})
            else:
                    QMessageBox.warning(
                    self,
                    "Change Failed",
                    f"Could not assign model '{new_model_id}' to {symbol}. "
                    "The model may not exist in the registry.",
                )
            self._refresh_symbols()

    # ------------------------------------------------------------------
    # External refresh (called by main window timer)
    # ------------------------------------------------------------------

    def refresh_from_state(self) -> None:
        """Called by the main window timer to refresh symbol data."""
        self._refresh_symbols()
        self._refresh_pattern_assignments()

    # ------------------------------------------------------------------
    # Pattern assignments (§10.1 multi-pattern)
    # ------------------------------------------------------------------

    _LIFECYCLE_COLORS: ClassVar[dict[str, tuple[str, str]]] = {
        "shadow": ("#00aaff", "xanh dương"),    # shadow blue
        "live": ("#00cc66", "xanh lá"),          # live green
        "degraded": ("#ffaa00", "vàng"),         # degraded orange/amber
        "retired": ("#888888", "xám"),           # retired gray
        "validated": ("#bbbbff", "tím"),
        "trained": ("#888888", "xám"),
    }

    def _lifecycle_color(self, state: str) -> str:
        return self._LIFECYCLE_COLORS.get(str(state).lower(), ("#888888", "?"))[0]

    def _assignable_models_for(self, pattern_name: str) -> list[Any]:
        """§10.1 GUI filter: models for *pattern_name* whose lifecycle_state
        is assignable ({validated, shadow, live})."""
        return self._bridge.get_models_for_assignment(
            pattern_name=pattern_name,
            lifecycle_states={"validated", "shadow", "live"},
        )

    def _on_add_assignment(self) -> None:
        """Open a small dialog: symbol + registered pattern + TF + state."""
        snap = self._bridge.state.get_snapshot()
        reg = snap.get("symbol_registry", {})
        symbols = sorted(reg.keys()) if isinstance(reg, dict) else []
        if not symbols:
            QMessageBox.information(self, "Assign Pattern", "Add a symbol first.")
            return
        patterns = self._bridge.get_pattern_names()
        if not patterns:
            QMessageBox.information(
                self, "Assign Pattern",
                "No pattern plugins registered (pattern_registry empty).",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Assign Pattern to Symbol")
        dlayout = QVBoxLayout(dialog)
        form = QHBoxLayout()
        sym_combo = QComboBox()
        sym_combo.addItems(symbols)
        pat_combo = QComboBox()
        for p in patterns:
            meta = self._bridge.get_pattern_metadata(p) or {}
            short = meta.get("short_name", "")
            pat_combo.addItem(f"{p} ({short})" if short else p, p)
        tf_combo = QComboBox()
        tf_combo.addItems(["M15", "H1", "H4", "D1"])
        state_combo = QComboBox()
        for st in ("live", "shadow"):
            state_combo.addItem(st, st)
        # --- Model dropdown (§10.1): filtered by the selected pattern +
        # lifecycle_state ∈ {validated, shadow, live}; repopulated whenever
        # the pattern (or the current TF) changes. ---
        model_combo = QComboBox()
        model_combo.setEditable(False)

        def repopulate_models() -> None:
            """Fill assignable models for the selected pattern (spec §10.1)."""
            pat = pat_combo.currentData()
            model_combo.clear()
            model_combo.addItem("— No model —", None)
            if not pat:
                return
            for m in self._assignable_models_for(pat):
                model_combo.addItem(
                    f"{m.model_id} — {m.symbol or m.symbol_origin} "
                    f"(schema {m.feature_schema_version}, {m.lifecycle_state})",
                    (m.model_id, m.feature_schema_version),
                )

        pat_combo.currentIndexChanged.connect(lambda _i: repopulate_models())
        repopulate_models()

        form.addWidget(QLabel("Symbol:"))
        form.addWidget(sym_combo, 1)
        dlayout.addLayout(form)
        form2 = QHBoxLayout()
        form2.addWidget(QLabel("Pattern:"))
        form2.addWidget(pat_combo, 1)
        dlayout.addLayout(form2)
        form3 = QHBoxLayout()
        form3.addWidget(QLabel("TF:"))
        form3.addWidget(tf_combo, 1)
        form3.addWidget(QLabel("State:"))
        form3.addWidget(state_combo, 1)
        dlayout.addLayout(form3)
        form4 = QHBoxLayout()
        form4.addWidget(QLabel("Model:"))
        form4.addWidget(model_combo, 1)
        dlayout.addLayout(form4)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlayout.addWidget(buttons)
        if not dialog.exec():
            return

        symbol = sym_combo.currentText()
        pattern = pat_combo.currentData()
        tf = tf_combo.currentText()
        state = state_combo.currentData()
        aid = f"{pattern}@{tf}"
        model_pick = model_combo.currentData()
        model_id = model_pick[0] if model_pick else ""
        feature_schema_version = model_pick[1] if model_pick else ""
        self._bridge.set_pattern_assignment(
            symbol, aid,
            pattern_name=pattern, timeframe=tf, state=state,
            model_id=model_id, feature_schema_version=feature_schema_version,
        )
        self._bridge.log_action("add_pattern_assignment", {
            "symbol": symbol, "pattern": pattern, "timeframe": tf, "state": state,
            "model_id": model_id,
        })
        syslog("INFO", "Onboarding", symbol,
               f"Pattern assignment added: {pattern} on {tf} state={state} "
               f"model={model_id or 'none'}")
        self._refresh_pattern_assignments()

    def _on_remove_assignment(self) -> None:
        row = self._pattern_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Remove Assignment", "Select a row first.")
            return
        symbol = self._pattern_table.item(row, 0).text()
        assignment_id = self._pattern_table.item(row, 1).data(Qt.UserRole)
        if assignment_id:
            self._bridge.remove_pattern_assignment(symbol, assignment_id)
        self._bridge.log_action("remove_pattern_assignment", {
            "symbol": symbol, "assignment_id": assignment_id,
        })
        self._refresh_pattern_assignments()

    def _refresh_pattern_assignments(self) -> None:
        """Populate the §10.1 pattern table from the bridge store."""
        reg = self._bridge.state.symbol_registry
        symbols = sorted(reg.keys()) if isinstance(reg, dict) else []
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            rows.extend([
                {
                    "symbol": symbol,
                    "assignment_id": a.get("assignment_id", ""),
                    "pattern": a.get("pattern_name", ""),
                    "timeframe": a.get("timeframe", ""),
                    "state": a.get("state", "live"),
                    "model_id": a.get("model_id", ""),
                }
                for a in self._bridge.get_pattern_assignments(symbol)
            ])
        self._pattern_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._pattern_table.setItem(r, 0, QTableWidgetItem(row["symbol"]))
            pat_item = QTableWidgetItem(row["pattern"])
            pat_item.setData(Qt.UserRole, row["assignment_id"])
            self._pattern_table.setItem(r, 1, pat_item)
            self._pattern_table.setItem(r, 2, QTableWidgetItem(row["timeframe"]))
            state_item = QTableWidgetItem(row["state"])
            color = self._lifecycle_color(row["state"])
            state_item.setForeground(QColor(color))
            self._pattern_table.setItem(r, 3, state_item)
            self._pattern_table.setItem(r, 4, QTableWidgetItem(row["model_id"] or "—"))
            badge = QTableWidgetItem(f"● {row['state']}")
            badge.setForeground(QColor(color))
            self._pattern_table.setItem(r, 5, badge)

    def _on_activate(self, symbol: str) -> None:
        """Activate a symbol (requires validated status and model_id)."""
        cfg = self._bridge.state.get_symbol_config(symbol)
        if not cfg:
            return

        if cfg.status != "validated":
            QMessageBox.warning(
                self,
                "Cannot Activate",
                f"Symbol '{symbol}' has status '{cfg.status}'. "
                "Only validated symbols can be activated.",
            )
            return

        if not cfg.model_id:
            QMessageBox.warning(
                self,
                "No Model Assigned",
                f"Symbol '{symbol}' has no model assigned. "
                "Use [Change Model] to assign one before activating.",
            )
            return

        dialog = ConfirmationDialog(
            title="Activate Symbol",
            message=f"Activate {symbol}? It will start receiving signal-engine calls.",
            confirm_text="Activate",
        )
        if dialog.exec() == ConfirmationDialog.Accepted:
            self._bridge.state.register_symbol(
                symbol,
                type(cfg)(
                    name=cfg.name,
                    status=cfg.status,
                    model_id=cfg.model_id,
                    active=True,
                    position_size_multiplier=cfg.position_size_multiplier,
                ),
            )
            self._bridge.log_action("activate_symbol", {"symbol": symbol})
            syslog("INFO", "System", symbol, "Symbol activated — will receive signal-engine calls")
            self._refresh_symbols()

    def _on_deactivate(self, symbol: str) -> None:
        """Deactivate a symbol with confirmation."""
        dialog = ConfirmationDialog(
            title="Deactivate Symbol",
            message=f"Deactivate {symbol}? It will remain registered but inactive.",
            confirm_text="Deactivate",
        )
        if dialog.exec() == ConfirmationDialog.Accepted:
            cfg = self._bridge.state.get_symbol_config(symbol)
            if cfg:
                self._bridge.state.register_symbol(
                    symbol,
                    type(cfg)(
                        name=cfg.name,
                        status=cfg.status,
                        model_id=cfg.model_id,
                        active=False,
                        position_size_multiplier=cfg.position_size_multiplier,
                    ),
                )
            self._bridge.log_action("deactivate_symbol", {"symbol": symbol})
            syslog("INFO", "System", symbol, "Symbol deactivated — no longer receives signal-engine calls")
            self._refresh_symbols()

    def _on_reload_registry(self) -> None:
        """Reload the Model Registry from disk and refresh the table."""
        n = self._bridge.reload_model_registry()
        QMessageBox.information(
            self,
            "Model Registry Reloaded",
            f"Loaded {n} model(s) from configs/models/index.yaml.",
        )
        self._refresh_symbols()

    # ------------------------------------------------------------------
    # Change Status
    # ------------------------------------------------------------------

    def _on_change_status(self, symbol: str) -> None:
        """Open a dialog to change the symbol's status."""
        cfg = self._bridge.state.get_symbol_config(symbol)
        if not cfg:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Change Status — {symbol}")
        dialog.setMinimumWidth(350)
        _style_dialog(dialog)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        msg = QLabel(f"Select new status for <b>{symbol}</b> (current: <b>{cfg.status}</b>):")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        combo = QComboBox()
        combo.addItems(["validated", "candidate", "rejected"])
        combo.setCurrentText(cfg.status)
        layout.addWidget(combo)

        note = QLabel(
            "Note: Only 'validated' symbols with a model can be activated.\n"
            "'rejected' symbols cannot be activated."
        )
        note.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        confirm_btn = button_box.button(QDialogButtonBox.Ok)
        confirm_btn.setText("Change Status")
        confirm_btn.setStyleSheet(
            f"background-color: {COLOR_WARNING}; color: #1e1e1e; font-weight: bold;"
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            new_status = combo.currentText().strip()
            if new_status == cfg.status:
                return
            # If switching away from validated → auto-deactivate
            new_active = cfg.active if new_status == "validated" else False
            from live.state.shared_app_state_v2 import SymbolConfig
            self._bridge.state.register_symbol(
                symbol,
                SymbolConfig(
                    name=symbol,
                    status=new_status,
                    active=new_active,
                    model_id=cfg.model_id,
                    position_size_multiplier=cfg.position_size_multiplier,
                ),
            )
            self._bridge.log_action("change_status",
                                    {"symbol": symbol,
                                     "old_status": cfg.status,
                                     "new_status": new_status})
            syslog("INFO", "System", symbol,
                   f"Status changed: {cfg.status} → {new_status}",
                   extra={"old_status": cfg.status, "new_status": new_status})
            self._refresh_symbols()

    # ------------------------------------------------------------------
    # Set Risk (position size multiplier)
    # ------------------------------------------------------------------

    def _on_set_risk(self, symbol: str) -> None:
        """Open the position-size risk editor for an active validated symbol."""
        cfg = self._bridge.state.get_symbol_config(symbol)
        if cfg is None:
            return
        if cfg.status != "validated" or not cfg.active:
            return

        current = float(cfg.position_size_multiplier)

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Set Risk — {symbol}")
        dialog.setMinimumWidth(380)
        _style_dialog(dialog)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        msg = QLabel(
            f"Position-size risk multiplier for <b>{symbol}</b>\n"
            f"(order size = <b>base x multiplier</b>, current: "
            f"<b>{current:.1f}x</b>):"
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        spin = QDoubleSpinBox()
        spin.setRange(MIN_RISK_MULTIPLIER, MAX_RISK_MULTIPLIER)
        spin.setDecimals(1)
        spin.setSingleStep(0.1)
        spin.setValue(current)
        spin.setSuffix(" x")
        layout.addWidget(spin)

        note = QLabel(
            f"Allowed range: {MIN_RISK_MULTIPLIER:.1f}x - {MAX_RISK_MULTIPLIER:.1f}x. "
            "This is the per-symbol risk the risk layer uses when sizing "
            "orders (RiskGuard.get_position_size)."
        )
        note.setStyleSheet(f"color: {COLOR_NEUTRAL}; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        confirm_btn = button_box.button(QDialogButtonBox.Ok)
        confirm_btn.setText("Set Risk")
        confirm_btn.setStyleSheet(
            f"background-color: {COLOR_POSITIVE}; color: #1e1e1e; font-weight: bold;"
        )
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec() == QDialog.Accepted:
            self._apply_risk(symbol, current, spin.value())

    def _apply_risk(self, symbol: str, old_multiplier: float, new_multiplier: float) -> None:
        """Confirm and persist a risk-multiplier change for *symbol*."""
        if abs(new_multiplier - old_multiplier) < 1e-9:
            self._refresh_symbols()
            return  # no change

        dialog = ConfirmationDialog(
            title="Confirm Risk Change",
            message=(
                f"Set position-size multiplier for <b>{symbol}</b> "
                f"from <b>{old_multiplier:.1f}x</b> to <b>{new_multiplier:.1f}x</b>?\n\n"
                "Future orders for this symbol will be sized at "
                f"base x {new_multiplier:.1f}. The value is persisted and "
                "survives restarts."
            ),
            confirm_text="✅ SAVE RISK",
            parent=self,
        )
        if dialog.exec() != ConfirmationDialog.Accepted:
            self._refresh_symbols()
            return

        ok = self._bridge.state.set_position_size_multiplier(symbol, new_multiplier)
        if ok:
            self._bridge.log_action("set_risk", {
                "symbol": symbol,
                "old_multiplier": old_multiplier,
                "new_multiplier": new_multiplier,
            })
            syslog("INFO", "System", symbol,
                   f"Position-size risk multiplier set: "
                   f"{old_multiplier:.1f}x → {new_multiplier:.1f}x",
                   extra={"old_multiplier": old_multiplier,
                          "new_multiplier": new_multiplier})
        else:
            QMessageBox.warning(
                self,
                "Risk Update Failed",
                f"Could not set multiplier {new_multiplier:.1f} for {symbol}. "
                "Value must be between "
                f"{MIN_RISK_MULTIPLIER:.1f} and {MAX_RISK_MULTIPLIER:.1f}.",
            )
        self._refresh_symbols()

    # ------------------------------------------------------------------
    # Remove Symbol
    # ------------------------------------------------------------------

    def _on_remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the registry with strong confirmation."""
        dialog = ConfirmationDialog(
            title="Remove Symbol",
            message=f"Remove <b>{symbol}</b> from the registry permanently?\n\n"
                    "This action cannot be undone. The symbol will need to be "
                    "re-added and re-validated to use it again.",
            confirm_text="REMOVE",
            require_input=True,
            input_match="REMOVE",
            input_placeholder='Type "REMOVE" to confirm',
        )
        if dialog.exec() == ConfirmationDialog.Accepted:
            self._bridge.state.unregister_symbol(symbol)
            self._bridge.log_action("remove_symbol", {"symbol": symbol})
            syslog("WARNING", "System", symbol, "Symbol removed from registry permanently")
            self._refresh_symbols()