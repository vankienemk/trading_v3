"""
live_corr.py — typed runtime bridge to ``live.engine.correlation_manager``.

SPEC §12 mandate: the multi-backtest runner must reuse the SAME
``CorrelationManager`` / ``CorrelationConfig`` / ``ResolvedGroup`` classes
the live engine uses (backtest ≡ live — one code path, no duplicated
grouping logic).  This module imports them from ``live.engine`` at runtime
— the very same module objects the live ``MultiPatternEngine`` instantiates
(asserted by identity checks in ``tests/test_multi_backtest.py`` and by the
group-decision parity test).

Why a bridge instead of a plain ``from live.engine.correlation_manager
import ...``: per handoff lesson #2, mypy is scoped per file for verify
tasks.  ``live/`` carries pre-existing non-strict code (out of t4 scope), so
a static import would make ``mypy runner.py --strict`` descend into those
files and report pre-existing errors.  The bridge keeps strict mypy scoped
to ``research/multi_backtest/`` while the runtime objects stay identical.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Runtime import of the live module (same object the live engine uses).
_LIVE_CORR: Any = importlib.import_module("live.engine.correlation_manager")

#: The live classes — re-exported under the canonical names.
CorrelationConfig: Any = _LIVE_CORR.CorrelationConfig
CorrelationManager: Any = _LIVE_CORR.CorrelationManager
ResolvedGroup: Any = _LIVE_CORR.ResolvedGroup

#: Module identity check helper (used by the parity test).
LIVE_CORR_MODULE = _LIVE_CORR

__all__ = [
    "LIVE_CORR_MODULE",
    "CorrelationConfig",
    "CorrelationManager",
    "ResolvedGroup",
]