"""
detector.py — Double Top plugin (P2, mirror of Double Bottom).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P2): Double Top is the exact mirror
of Double Bottom and reuses **100 % of P1 infra** — the shared
:class:`research.core.swing_detector.SwingDetector` and the machinery in
``research.patterns.double_bottom.detector.DoublePatternDetectorBase``.

Geometry (bearish): two swing HIGHS at approximately the same level with a
swing LOW (neckline touch) between them; confirmation = a close crosses
BELOW the neckline after the second high's pivot is causally known; entry on
the next open; stop above the higher extreme + ATR buffer; target
``target_r`` R below entry.

See ``patterns/double_top/PATTERN_SPECS.md`` for the full spec (mirrored
geometry, swing formula, confirmation, entry/SL/TP, staleness, features).
"""

from __future__ import annotations

from typing import Any

from research.core.contracts import DIRECTION_BEARISH
from research.patterns.double_bottom.detector import DoublePatternDetectorBase


class DoubleTopDetector(DoublePatternDetectorBase):
    """Double Top (P2) — bearish reversal off two equal swing highs."""

    name = "double_top"
    version = "1.0"
    short_name = "DT"
    direction = DIRECTION_BEARISH
    kind_seq = ("H", "L", "H")


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> DoubleTopDetector:
    return DoubleTopDetector(config)


DETECTOR_CLASS = DoubleTopDetector