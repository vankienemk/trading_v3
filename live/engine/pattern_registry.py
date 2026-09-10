"""
pattern_registry.py — Dynamic plugin discovery (§2.2) + pattern_short map (§9.2)

Agent 6 (t9).  Scans ``research/patterns/*/detector.py`` once at load time,
imports every class subclassing :class:`BasePatternDetector`, and registers
it under its canonical ``name`` (the same ``pattern_name`` a
:class:`PatternEvent` carries and the ``pattern_name`` field of the model
registry §5.5).

Discovery contract (§2.2):
  * each pattern package exposes ``detector.py`` declaring one of:
      - ``PATTERN_ENTRY: dict[name, detector_class]``   (liquidity_sweep)
      - ``DETECTOR_CLASS: type[BasePatternDetector]``   (double_bottom/top)
  * a plugin that fails to import is logged as a WARNING and skipped — the
    engine keeps running (fail-open technically, fail-closed in signals: a
    broken pattern simply produces no events).

Alongside the detector registry this module also centralises the §9.2
``pattern_short`` table (2-4 unique chars per pattern, used by
:func:`order_comment_for` / the execution layer).  Detector classes already
declare ``short_name``; this table keeps a stable contract-visible map so the
order-comment schema does not depend on import order.

Normalisation:
  * ``name`` is downcased and whitespace-stripped so GUI config strings and
    event ``pattern_name`` always match.
  * plugin ordering is deterministic (sorted by pattern name) so the registry
    is reproducible across runs.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import research.patterns as _patterns_pkg
from research.core.contracts import (
    PATTERN_SHORT_NAMES as CONTRACT_SHORT_NAMES,
)
from research.core.contracts import (
    BasePatternDetector,
)

logger = logging.getLogger("pattern_registry")

#: Pattern package root (``trading_v3/research/patterns``).
_PATTERNS_ROOT = Path(_patterns_pkg.__file__).resolve().parent

#: Fallback short code for an unknown pattern (§9.2 "PAT").
DEFAULT_SHORT = "PAT"


def _detector_classes_from_module(
    module: Any,
) -> list[type[BasePatternDetector]]:
    """Collect detector classes declared by a plugin module.

    Prefers the explicit ``PATTERN_ENTRY`` map (dict[name -> class]) then
    ``DETECTOR_CLASS``; as a last resort scans the module namespace for any
    ``BasePatternDetector`` subclass.
    """
    from research.core.contracts import BasePatternDetector as _BPD

    classes: list[type[BasePatternDetector]] = []

    entry = getattr(module, "PATTERN_ENTRY", None)
    if isinstance(entry, dict):
        classes.extend(
            cls
            for cls in entry.values()
            if isinstance(cls, type)
            and issubclass(cls, _BPD)
            and not inspect_is_abstract(cls)
        )

    dc = getattr(module, "DETECTOR_CLASS", None)
    if isinstance(dc, type) and issubclass(dc, _BPD) and dc not in classes and not inspect_is_abstract(dc):
        classes.append(dc)

    if not classes:
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, _BPD)
                and obj is not _BPD
                and not inspect_is_abstract(obj)
            ):
                classes.append(obj)
    return classes


def inspect_is_abstract(cls: type) -> bool:
    """True when *cls* or a base still declares abstract methods.

    ``DoublePatternDetectorBase`` is the shared abstract base for DB/DT and
    must NOT be registered on its own — only its concrete subclasses.
    """
    from abc import ABCMeta

    if not isinstance(cls, ABCMeta):
        return False
    try:
        return bool(getattr(cls, "__abstractmethods__", None))
    except Exception:
        return False


def _instance_of(cls: type[BasePatternDetector], config: dict[str, Any] | None = None) -> BasePatternDetector:
    """Instantiate a detector, tolerating both (config) and no-arg ctors."""
    try:
        return cls(config)  # type: ignore[call-arg]  # detectors may accept config
    except TypeError:
        return cls()


class PatternRegistry:
    """Thread-safe registry mapping pattern name → detector class + meta.

    ``load()`` performs discovery once over ``research/patterns``.  A consumer
    typically uses the module-level ``registry`` singleton.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, type[BasePatternDetector]] = {}
        self._short_names: dict[str, str] = {}
        self._loaded = False
        self._errors: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def load(self, patterns_root: Path | str | None = None) -> int:
        """Discover plugin detector classes under ``research/patterns``.

        Returns the number of registers; plugins that fail to import are
        captured in :attr:`errors` and logged (never raise).
        """
        root = Path(patterns_root) if patterns_root is not None else _PATTERNS_ROOT
        detectors: dict[str, type[BasePatternDetector]] = {}
        short_names: dict[str, str] = dict(CONTRACT_SHORT_NAMES)
        errors: dict[str, str] = {}

        # Discovery scans on-disk package dirs (``*/detector.py``) rather than
        # ``pkgutil.iter_modules`` so namespace packages without an
        # ``__init__.py`` (e.g. liquidity_sweep) are still found.  A plugin
        # that fails to import is logged + skipped — never crashes the engine.
        pkg_names = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and (d / "detector.py").exists()
        )
        for pkg_name in pkg_names:
            try:
                mod = importlib.import_module(
                    f"research.patterns.{pkg_name}"
                )
            except Exception as exc:  # plugin broken → skip, never crash engine
                msg = f"import failed: {exc}"
                errors[pkg_name] = msg
                logger.warning("pattern_registry: %s %s", pkg_name, msg)
                continue

            # try detector submodule next to package __init__
            try:
                det_mod = importlib.import_module(
                    f"research.patterns.{pkg_name}.detector"
                )
            except Exception as exc:
                # package __init__ may itself expose DETECTOR_CLASS
                det_mod = mod
                if not hasattr(det_mod, "DETECTOR_CLASS") and not hasattr(det_mod, "PATTERN_ENTRY"):
                    errors[f"{pkg_name}.detector"] = f"import failed: {exc}"
                    logger.warning("pattern_registry: %s.detector %s", pkg_name, exc)
                    continue

            try:
                classes = _detector_classes_from_module(det_mod)
            except Exception as exc:
                errors[f"{pkg_name}.detector"] = f"class scan failed: {exc}"
                logger.warning("pattern_registry: %s class scan failed: %s", pkg_name, exc)
                continue

            for cls in classes:
                name = _normalise(getattr(cls, "name", pkg_name))
                detectors[name] = cls
                short = getattr(cls, "short_name", "") or ""
                if short:
                    short_names[name] = short.upper()

        self._detectors = detectors
        self._short_names = short_names
        self._errors = errors
        self._loaded = True
        return len(detectors)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    @property
    def pattern_names(self) -> list[str]:
        return sorted(self._detectors)

    @property
    def error_messages(self) -> dict[str, str]:
        return dict(self._errors)

    def is_loaded(self) -> bool:
        return self._loaded

    def has_pattern(self, name: str) -> bool:
        return _normalise(name) in self._detectors

    def get(self, name: str) -> type[BasePatternDetector] | None:
        return self._detectors.get(_normalise(name))

    def create(
        self, name: str, config: dict[str, Any] | None = None
    ) -> BasePatternDetector | None:
        """Instanciate a detector plugin by pattern name (or None if absent)."""
        cls = self.get(name)
        if cls is None:
            return None
        return _instance_of(cls, config)

    def short_name(self, name: str) -> str:
        """§9.2 pattern_short lookup (2-4 chars, unique)."""
        n = _normalise(name)
        return self._short_names.get(n, DEFAULT_SHORT)

    # ------------------------------------------------------------------
    # Registration metadata (for GUI / Lifecycle / order comments)
    # ------------------------------------------------------------------
    def metadata(self, name: str) -> dict[str, Any] | None:
        """Public descriptor for one pattern: name/version/short_name."""
        cls = self.get(name)
        if cls is None:
            return None
        return {
            "name": cls.name,
            "version": getattr(cls, "version", ""),
            "short_name": self.short_name(name),
            "feature_schema": [
                {
                    "name": f.name,
                    "available_at": f.available_at,
                }
                for f in getattr(cls, "feature_schema", [])
            ],
        }


def _normalise(name: str) -> str:
    return str(name).strip().lower()


#: Module-level singleton (lazy).
_registry: PatternRegistry | None = None


def get_registry() -> PatternRegistry:
    """Return the process-wide PatternRegistry, discovering on first call."""
    global _registry
    if _registry is None:
        _registry = PatternRegistry()
        _registry.load()
    return _registry


def pattern_short_name(pattern_name: str) -> str:
    """Convenience: §9.2 short-code for a pattern name."""
    return get_registry().short_name(pattern_name)


def order_comment_for_event(ev: Any) -> str:
    """§9.2 standard order comment from a PatternEvent.

    ``{pattern_short}-v{major}-{event_id_short}`` where ``pattern_short`` is
    resolved through the registry (detector ``short_name`` first, then the
    contract's PATTERN_SHORT_NAMES, then ``PAT`` fallback).  This is the
    registry-owned twin of ``research.core.contracts.order_comment_for`` —
    it exists so patterns registered after the contract freeze (DB/DT/...)
    still produce the standard comment without editing the frozen file.
    """

    pattern = getattr(ev, "pattern_name", "") or ""
    version = getattr(ev, "pattern_version", "") or ""
    event_id = getattr(ev, "event_id", "") or ""

    short = get_registry().short_name(pattern)
    major = version.split(".")[0] if version else "0"
    id_short = event_id[-4:] if len(event_id) >= 4 else event_id
    return f"{short}-v{major}-{id_short}"


__all__ = [
    "DEFAULT_SHORT",
    "PatternRegistry",
    "get_registry",
    "order_comment_for_event",
    "pattern_short_name",
]
