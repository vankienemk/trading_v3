"""
detector.py — Inverse Head & Shoulders plugin (P4, mirror of H&S).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 (P4): Inverse Head & Shoulders is the
mirror of Head & Shoulders and reuses **100 % of P4 infra** — the shared
:class:`research.core.swing_detector.SwingDetector` and the machinery in
``research.patterns.head_shoulders.detector.HeadShouldersDetectorBase``.

Geometry (bullish): 5 swing lows/highs (L-H-L-H-L) — left shoulder TROUGH,
neckline peak, HEAD trough (strictly lower), neckline peak, right shoulder
trough; confirmation = a close crosses ABOVE the neckline (lower peak)
after the right shoulder's pivot is causally known; entry on the next open;
stop below the head - ATR buffer; target ``target_r`` R above entry.

See ``patterns/inverse_head_shoulders/PATTERN_SPECS.md`` for the full spec
(mirrored geometry, swing formula, confirmation, entry/SL/TP, staleness,
features).
"""

from __future__ import annotations

from typing import Any

from research.core.contracts import DIRECTION_BULLISH
from research.patterns.head_shoulders.detector import HeadShouldersDetectorBase


class InverseHeadShouldersDetector(HeadShouldersDetectorBase):
    """Inverse Head & Shoulders (P4) — bullish reversal off a lower-middle low."""

    name = "inverse_head_shoulders"
    version = "1.0"
    short_name = "IHS"
    direction = DIRECTION_BULLISH
    kind_seq = ("L", "H", "L", "H", "L")


# ---------------------------------------------------------------------------
# Convenience factory (registry entry point per §2.2)
# ---------------------------------------------------------------------------
def get_detector(config: dict[str, Any] | None = None) -> InverseHeadShouldersDetector:
    return InverseHeadShouldersDetector(config)


DETECTOR_CLASS = InverseHeadShouldersDetector