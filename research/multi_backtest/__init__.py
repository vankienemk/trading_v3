"""multi_backtest — Multi-Pattern Backtest & Reporting (SPEC v1.1 §12).

The runner mirrors the *live* multi-pattern path: the same
:class:`~live.engine.correlation_manager.CorrelationManager` and §4.4
exposure caps imported from ``live.engine`` are applied to the historical
event stream (backtest ≡ live — one code path, no duplicated grouping
logic), and the report generator emits the mandatory 5 §12 items on ≥2
symbols with costs applied.
"""

from __future__ import annotations

__version__ = "1.0.0"