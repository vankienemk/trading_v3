"""Trading-session boundaries and hour-based session flags (guide section 13.6).

Session opening/closing hours must live in config (``sessions`` section of
``configs/baseline.yaml``, UTC) so they can be changed for timezone / DST
policy.  The helpers here are pure functions of an hour (0..23) and a session
definition, so they are trivially causal (a session flag depends only on the
event's own timestamp).

Default session hours (UTC, baseline: FX Asia / London / New York overlap):

* Asia     ``[0, 8)``   — 00:00-08:00 UTC
* London   ``[7, 16)``  — 07:00-16:00 UTC
* New York ``[12, 21)`` — 12:00-21:00 UTC

A half-open interval ``[start, end)``; ``end <= start`` means the session wraps
midnight (e.g. ``[20, 4)`` = 20:00-04:00).  An event may belong to more than
one session at once (``session_overlap`` counts them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Default session hours (UTC) when no ``sessions`` config section is present.
DEFAULT_SESSIONS: dict[str, list[int]] = {
    "asia": [0, 8],
    "london": [7, 16],
    "new_york": [12, 21],
}


@dataclass(frozen=True)
class SessionSpec:
    """One named session as a half-open hour interval ``[start, end)``."""

    name: str
    start: int
    end: int

    def contains_hour(self, hour: int) -> bool:
        """True when ``hour`` falls inside ``[start, end)`` (wrap-aware)."""
        if self.start == self.end:
            return False
        if self.start < self.end:
            return self.start <= hour < self.end
        # wraps midnight: [start, 24) OR [0, end)
        return hour >= self.start or hour < self.end


def _parse_hours(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"sessions.{name}: expected a 2-element [start, end] hour list, "
            f"got {value!r}"
        )
    start, end = int(value[0]), int(value[1])
    if not (0 <= start <= 24 and 0 <= end <= 24):
        raise ValueError(
            f"sessions.{name}: hours must be in [0, 24], got [{start}, {end}]"
        )
    return [start, end]


def session_specs(
    config: dict[str, Any] | None = None,
) -> dict[str, SessionSpec]:
    """Build the ordered ``{name: SessionSpec}`` map from config or defaults.

    Reads ``config["sessions"]`` and accepts arbitrary session names; the
    canonical three (asia / london / new_york) are always present in the
    default config.  Config session hours are interpreted in the configured
    timezone (default UTC) and applied to the already-UTC event timestamps.
    """
    raw = (config or {}).get("sessions", DEFAULT_SESSIONS)
    if isinstance(raw, dict):
        # Drop metadata keys that are not session definitions (e.g. timezone).
        meta = {"timezone", "tz", "name"}
        merged = {
            **DEFAULT_SESSIONS,
            **{k: v for k, v in raw.items() if k not in meta},
        }
    else:
        merged = dict(DEFAULT_SESSIONS)
    return {
        name: SessionSpec(name, *_parse_hours(hours, name))
        for name, hours in merged.items()
    }


def session_flags(
    hour_utc: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return session feature values for a UTC hour: flags, overlap, minutes.

    Produces ``session_<name>`` booleans for every configured session, plus
    ``session_overlap`` (count of simultaneous sessions) and
    ``minutes_from_session_open`` (minutes since the earliest session that
    contains ``hour_utc`` opened today; NaN when no session is active).
    """
    specs = session_specs(config)
    active = [
        (name, spec) for name, spec in specs.items() if spec.contains_hour(hour_utc)
    ]
    flags: dict[str, Any] = {}
    for name in specs:
        flags[f"session_{name}"] = any(n == name for n, _ in active)

    flags["session_overlap"] = len(active)
    if active:
        earliest_start = min(spec.start for _, spec in active)
        flags["minutes_from_session_open"] = (hour_utc - earliest_start) * 60
    else:
        flags["minutes_from_session_open"] = float("nan")
    return flags


__all__ = [
    "DEFAULT_SESSIONS",
    "SessionSpec",
    "session_flags",
    "session_specs",
]
