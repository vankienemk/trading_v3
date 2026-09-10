"""
detector.py — Falling Wedge plugin (P3, mirror of Rising Wedge).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P3): Falling Wedge is the mirror of
Rising Wedge and reuses **100 % of P3 infra** — the shared
:class:`research.core.swing_detector.SwingDetector`, the trendline module
(``research.core.trendline``) and the machinery in
``research.patterns.rising_wedge.detector.WedgePatternDetectorBase``.

Geometry (bullish): 3 DESCENDING swing lows with ≥ 2 DESCENDING swing highs
between them forming a CONVERGING channel; confirmation = a close crosses
ABOVE the upper trendline after the last low's pivot is causally known;
entry on the next open; stop below the wedge low - ATR buffer; target
``target_r`` R above entry.

See ``patterns/falling_wedge/PATTERN_SPECS.md`` for the full spec (mirrored
geometry, swing formula, confirmation, entry/SL/TP, staleness, features).
"""

from __future__ import annotations

from typing import Any

from research.core.contracts import DIRECTION_BULLISH
from research.patterns.rising_wedge.detector import WedgePatternDetectorBase


class FallingWedgeDetector(WedgePatternDetectorBase):
    """Falling Wedge (P3) — bullish reversal off a converging falling channel."""

    name = "falling_wedge"
    version = "1.0"
    short_name = "FW"
    direction = DIRECTION_BULLISH


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> FallingWedgeDetector:
    return FallingWedgeDetector(config)


DETECTOR_CLASS = FallingWedgeDetector