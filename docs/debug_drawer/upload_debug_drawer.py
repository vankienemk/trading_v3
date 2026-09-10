"""Upload DebugDrawer files into the MT5 (Wine) terminal via MCP.

Reads the master files under trading_v3/docs/debug_drawer/ and creates them
in the live MT5 terminal (C:\\Program Files\\MetaTrader 5\\MQL5\\).
Usage:
  MCP_TOKEN=... python3 upload_debug_drawer.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root -> mt5_mcp_client.py

from mt5_mcp_client import MCPClient  # noqa: E402

ROOT = Path(__file__).resolve().parent
MQL5 = r"C:\Program Files\MetaTrader 5\MQL5"

# write_file overwrites existing files (create_new_file fails on existing)
FILES = [
    ("write_file", MQL5 + r"\Scripts\DebugDrawer.mq5", ROOT / "DebugDrawer.mq5"),
    ("write_file", MQL5 + r"\Files\debug_events.txt", ROOT / "debug_events.txt"),
    ("write_file", MQL5 + r"\Files\debug_events_README.txt", ROOT / "debug_events_README.txt"),
]


def main() -> int:
    client = MCPClient()
    health = client.health_check()
    print("MCP health:", health.get("ok"), health.get("url"))
    for tool, path, local in FILES:
        content = local.read_text(encoding="utf-8")
        print(f"-> {tool} {path} ({len(content)} chars from {local.name})")
        result = client.call_tool(tool, {"path": path, "content": content, "overwrite": True})
        print("   ", str(result)[:500])
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())