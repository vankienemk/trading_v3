"""dedupe.py — cross-pattern de-overlap helpers for the multi-pattern engine.

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §4 is about *correlation grouping*: several
patterns firing at the same time are one trade idea.  This module covers the
complementary case that grouping does not catch — two detectors reading the
**same price structure** in **opposite directions** (a double top and a double
bottom built on one shared middle swing).  Grouping correctly refuses to merge
them (opposite directions are never one idea), so without this filter both are
emitted and the engine can open opposing trades on a single swing.

This is deliberately NOT a scoring change: it only removes events whose
structure is claimed by a stronger, contrary reading.  Ordering is by
``rule_score`` (then ``event_id``) so the outcome is deterministic and does
not depend on detector registration order.
"""

from __future__ import annotations

from typing import Any

from research.core.contracts import PatternEvent

__all__ = ["drop_opposite_overlap", "structure_span"]


def structure_span(ev: PatternEvent) -> tuple[int, int] | None:
    """Bar span ``(first, last)`` of the event's own structure anchors.

    Each pattern family names its pivots differently, so the span is derived
    from whichever keys the plugin actually publishes — never from a single
    hard-coded key, which would silently collapse to bar 0 for every family
    that does not publish it (manufacturing phantom overlaps).
    """
    a: dict[str, Any] = getattr(ev, "attributes", None) or {}
    if "extreme1_bar" in a and "extreme2_bar" in a:
        return int(a["extreme1_bar"]), int(a["extreme2_bar"])
    bars = [a.get("low1_bar"), a.get("low2_bar"), a.get("low3_bar")]
    if all(b is None for b in bars):
        bars = [
            a.get("left_shoulder_bar"),
            a.get("head_bar"),
            a.get("right_shoulder_bar"),
        ]
    vals = [int(b) for b in bars if b is not None]
    if len(vals) < 2:
        return None
    return min(vals), max(vals)


def _neckline_bar(ev: PatternEvent) -> int | None:
    a: dict[str, Any] = getattr(ev, "attributes", None) or {}
    for key in ("neckline_bar", "neckline1_bar"):
        if a.get(key) is not None:
            return int(a[key])
    return None


def drop_opposite_overlap(events: list[PatternEvent]) -> list[PatternEvent]:
    """Drop an event when a contrary-direction event claims its structure.

    Two events conflict when they point in **opposite** directions AND their
    structure spans intersect AND one event's neckline falls inside the
    other's span — i.e. both readings are built on the same pivot sequence.
    The higher ``rule_score`` reading is kept.

    Same-direction events are left untouched: that case is handled by the
    per-detector NMS and by §4 correlation grouping.
    """
    if len(events) < 2:
        return list(events)

    ordered = sorted(
        events, key=lambda e: (-float(e.rule_score or 0.0), str(e.event_id))
    )
    kept: list[PatternEvent] = []
    for ev in ordered:
        ev_dir = str(ev.direction)
        ev_span = structure_span(ev)
        ev_neck = _neckline_bar(ev)
        conflict = False
        for other in kept:
            if str(other.direction) == ev_dir:
                continue
            o_span = structure_span(other)
            if o_span is None or ev_span is None:
                continue
            if ev_span[1] < o_span[0] or ev_span[0] > o_span[1]:
                continue  # spans are disjoint -> different structures
            o_neck = _neckline_bar(other)
            shared = (
                ev_neck is not None and o_span[0] <= ev_neck <= o_span[1]
            ) or (o_neck is not None and ev_span[0] <= o_neck <= ev_span[1])
            if shared:
                conflict = True
                break
        if not conflict:
            kept.append(ev)

    kept.sort(key=lambda e: (str(e.known_at_ts), str(e.event_id)))
    return kept
