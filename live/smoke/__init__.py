"""Live-path smoke tooling for trading_v3 (t9).

``mcp_live_smoke.py`` proves the §9.1 live wiring end-to-end without ever
placing an order: .env MCP config load, symbol-config parse, §5.5 registry
load, a SHADOW multi-pattern assignment (events into the Event Lake, zero
signals), and an offscreen GUI boot with §10.1 onboarding filters.
"""