"""
live — Live trading package.

Sub-packages:
    gui/        — PySide6 GUI (main window, bridge, components, tabs)
    engine/     — Signal engine, polling engine, execution layer
    mcp/        — MT5 MCP client
    logging/    — SQLite logger (signals, user actions, system log)
    state/      — Thread-safe singleton state + risk guard
    db/         — Database files, run entry point, configs, knowledge base
"""

from __future__ import annotations