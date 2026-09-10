"""
correlation_manager.py — Dedup / Correlation Contract (§4)

Agent 6 (t9).  Implements REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §4:

  * §4.1  — two events confirming on the same candle are often "one idea";
    if RiskGuard doesn't know, real risk silently doubles.
  * §4.2  — Correlation grouping is *transitive union-find*: e1~e2 and e2~e3
    ⇒ e1, e2, e3 share a group even when e1 and e3 do not directly match.
  * §4.3  — per-symbol policy: ``dedup`` (default) / ``confluence`` /
    ``independent``.
  * §4.4  — exposure caps (max_total_risk_per_symbol,
    max_direction_cluster_risk) — mandatory, independent of mode.
  * §4.5  — standard confluence score (weighted n-patterns + time + price
    proximity), a *feature* for the §6.4 meta-model, never a pass/fail gate.

The manager is stateless per-call on price data: given a batch of
:class:`PatternEvent`s (produced in a single scan pass across all of a
symbol's assignments) it partitions them into correlation groups and applies
the configured policy, returning :class:`ResolvedGroup` items that the signal
engine turns into *one* PendingSignal per group (dedup/confluence) or passes
through (independent).

§4.2 time window is expressed in *bars* of the larger of two timeframes.  The
manager resolves this to an absolute seconds delta via ``CFG.bar_seconds``
(when the engine supplies a seconds-per-bar estimate); otherwise the window
is interpreted against the raw timestamp delta as a conservative same-bar
match.  Code that passes a per-symbol ``corr_time_window_s`` in seconds is
also supported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from research.core.contracts import PatternEvent

# ---------------------------------------------------------------------------
# Policy constants (§4.3 / §4.4)
# ---------------------------------------------------------------------------
MODE_DEDUP = "dedup"                 # keep 1 best per group (§4.3)
MODE_CONFLUENCE = "confluence"       # merge into 1 signal, score + size boost (§4.3)
MODE_INDEPENDENT = "independent"     # pass-through (§4.3)

VALID_MODES = (MODE_DEDUP, MODE_CONFLUENCE, MODE_INDEPENDENT)

#: Default correlation window in bars of the larger timeframe (§4.2).
DEFAULT_CORR_TIME_WINDOW = 5
#: Default price proximity in ATR units at known_at (§4.2).
DEFAULT_CORR_PRICE_WINDOW_ATR = 0.5
#: §4.5 default weights (n-patterns / time / price).
DEFAULT_W1, DEFAULT_W2, DEFAULT_W3 = 0.5, 0.25, 0.25
#: §4.5 default time decay constant (bars).
DEFAULT_TAU_TIME = 5.0
#: §4.5 default price decay constant (ATR).
DEFAULT_TAU_PRICE = 0.5
#: §4.4 default cap on total risk per symbol (in risk-normalised units).
DEFAULT_MAX_TOTAL_RISK = 1.0
#: §4.4 default cap on same-direction cluster risk within 1 ATR.
DEFAULT_MAX_CLUSTER_RISK = 0.75
#: §9.3 default confluence size boost (< = total-risk cap).
DEFAULT_CONFLUENCE_SIZE_BOOST = 1.5


class CorrelationError(Exception):
    """Invalid mode / configuration / event payload for the manager."""


# ---------------------------------------------------------------------------
# ATR resolution (used for the §4.2 price window)
# ---------------------------------------------------------------------------

def default_atr_resolver(events: Sequence[PatternEvent]) -> dict[str, float | None]:
    """Resolve a per-event ATR from the event's own attributes.

    Order of preference: ``attributes["atr_value"]`` then
    ``attributes["atr"]`` then ``structure_levels["atr"]`` — the keys the
    sweep plugin (``atr_value``) and double plugins record.
    """
    out: dict[str, float | None] = {}
    for ev in events:
        attrs = getattr(ev, "attributes", {}) or {}
        val: float | None = None
        for key in ("atr_value", "atr"):
            v = attrs.get(key)
            if v is not None and not (isinstance(v, float) and not math.isfinite(v)):
                val = float(v)
                break
        if val is None:
            levels = getattr(ev, "structure_levels", {}) or {}
            v = levels.get("atr")
            if v is not None:
                val = float(v)
        out[ev.event_id] = val
    return out


def _event_ts(ev: PatternEvent) -> pd.Timestamp:
    """Causal availability stamp of an event (known_at)."""
    return pd.Timestamp(ev.known_at_ts)


def _dt_seconds(e1: PatternEvent, e2: PatternEvent) -> float:
    return abs((_event_ts(e1) - _event_ts(e2)).total_seconds())


def _same_bar(e1: PatternEvent, e2: PatternEvent) -> bool:
    """True when both events' known_at fall on the same minute."""
    try:
        a = pd.Timestamp(_event_ts(e1)).floor("min")
        b = pd.Timestamp(_event_ts(e2)).floor("min")
        return abs((b - a).total_seconds()) < 1.0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pairwise correlation (§4.2)
# ---------------------------------------------------------------------------

def _pairwise_correlated(
    e1: PatternEvent,
    e2: PatternEvent,
    cfg: CorrelationConfig,
    atr: Callable[[str], float | None],
) -> bool:
    """§4.2 — two events are correlated when ALL three hold:

    1. same ``symbol`` and same ``direction``;
    2. |known_at1 - known_at2| within the correlation time window;
    3. |entry_price1 - entry_price2| <= corr_price_window_atr * ATR.
    """
    if e1.symbol != e2.symbol:
        return False
    if e1.direction != e2.direction:
        return False
    if not _within_time_window(e1, e2, cfg):
        return False

    a1 = atr(e1.event_id)
    a2 = atr(e2.event_id)
    atr_val = _merged_atr(a1, a2)
    if atr_val is None or not math.isfinite(atr_val) or atr_val <= 0:
        # No ATR scale → conservative same-bar fallback (still de-dups
        # same-candle events, which is the §4.1 failure mode we must prevent).
        return _same_bar(e1, e2)
    price_gap = abs(e1.entry_price - e2.entry_price)
    return price_gap <= cfg.corr_price_window_atr * atr_val


def _within_time_window(e1: PatternEvent, e2: PatternEvent, cfg: CorrelationConfig) -> bool:
    """§4.2 time condition, expressed in bars of the larger timeframe.

    The window in bars (``cfg.corr_time_window``) is converted to seconds
    using ``cfg.bar_seconds`` when supplied; an explicit
    ``cfg.corr_time_window_s`` (seconds) overrides everything.  Without a bar
    scale, only exact same-bar events satisfy the window (conservative).
    """
    seconds: float | None = getattr(cfg, "corr_time_window_s", None)
    if seconds is not None and seconds > 0:
        return _dt_seconds(e1, e2) <= seconds
    if cfg.bar_seconds and cfg.bar_seconds > 0:
        return _dt_seconds(e1, e2) <= cfg.corr_time_window * cfg.bar_seconds
    return _same_bar(e1, e2)


def _merged_atr(a1: float | None, a2: float | None) -> float | None:
    vals = [v for v in (a1, a2) if v is not None and math.isfinite(v) and v > 0]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Union-find (§4.2 transitivity)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Union-find over integer node ids (path-halving + union by rank)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Policy config + group output
# ---------------------------------------------------------------------------

@dataclass
class CorrelationConfig:
    """Per-symbol correlation policy (§4.3/§4.4/§4.5)."""

    mode: str = MODE_DEDUP
    corr_time_window: float = DEFAULT_CORR_TIME_WINDOW   # bars of larger TF
    corr_price_window_atr: float = DEFAULT_CORR_PRICE_WINDOW_ATR
    bar_seconds: float | None = None     # seconds per bar of the larger TF
    corr_time_window_s: float | None = None  # optional explicit seconds override
    # §4.5 confluence weights + decay constants
    w1: float = DEFAULT_W1
    w2: float = DEFAULT_W2
    w3: float = DEFAULT_W3
    tau_time: float = DEFAULT_TAU_TIME
    tau_price: float = DEFAULT_TAU_PRICE
    # §4.4 exposure caps
    max_total_risk_per_symbol: float = DEFAULT_MAX_TOTAL_RISK
    max_direction_cluster_risk: float = DEFAULT_MAX_CLUSTER_RISK
    # §9.3 confluence size boost
    confluence_size_boost: float = DEFAULT_CONFLUENCE_SIZE_BOOST
    #: §4.5 denominator ``n_patterns_max`` — the max number of patterns that
    #: could realistically agree on one group (e.g. the symbol's active pattern
    #: count).  ``None`` → the group's own size (n-term saturates at 1).
    n_patterns_max: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise CorrelationError(
                f"mode must be one of {VALID_MODES}, got {self.mode!r}"
            )
        if self.w1 + self.w2 + self.w3 <= 0:
            raise CorrelationError("confluence weights must sum > 0")
        if self.confluence_size_boost < 1.0:
            raise CorrelationError("confluence_size_boost must be >= 1.0")
        if self.n_patterns_max is not None and self.n_patterns_max < 2:
            raise CorrelationError("n_patterns_max must be >= 2 (or None)")

    def with_overrides(self, overrides: dict[str, Any] | None = None) -> CorrelationConfig:
        """Return a copy with any provided keys overridden (None → no change)."""
        cfg = CorrelationConfig(
            mode=self.mode,
            corr_time_window=self.corr_time_window,
            corr_price_window_atr=self.corr_price_window_atr,
            bar_seconds=self.bar_seconds,
            corr_time_window_s=self.corr_time_window_s,
            w1=self.w1, w2=self.w2, w3=self.w3,
            tau_time=self.tau_time, tau_price=self.tau_price,
            max_total_risk_per_symbol=self.max_total_risk_per_symbol,
            max_direction_cluster_risk=self.max_direction_cluster_risk,
            confluence_size_boost=self.confluence_size_boost,
            n_patterns_max=self.n_patterns_max,
        )
        if overrides:
            for key in (
                "mode", "corr_time_window", "corr_price_window_atr",
                "bar_seconds", "corr_time_window_s", "w1", "w2", "w3",
                "tau_time", "tau_price", "max_total_risk_per_symbol",
                "max_direction_cluster_risk", "confluence_size_boost",
                "n_patterns_max",
            ):
                if key in overrides and overrides[key] is not None:
                    setattr(cfg, key, overrides[key])
        return cfg


@dataclass
class ResolvedGroup:
    """One actionable unit after the correlation policy has been applied.

    * ``mode=dedup``       → one group keeps the best event; others discarded.
    * ``mode=confluence``  → one merged signal (representative + score).
    * ``mode=independent`` → every event passes through unchanged.
    """

    mode: str
    symbol: str
    direction: str
    group_id: str | None
    events: list[PatternEvent] = field(default_factory=list)
    representative: PatternEvent | None = None
    confluence_score: float = 0.0
    n_patterns: int = 1
    discarded: list[PatternEvent] = field(default_factory=list)

    @property
    def kept_event(self) -> PatternEvent | None:
        """§4.3 best event per group (dedup) or sole event (independent)."""
        return self.representative


def _priority(ev: PatternEvent) -> float:
    """§4.3 priority = model_prob x rule_score (fallback rule_score)."""
    rs = float(getattr(ev, "rule_score", 0.0) or 0.0)
    mp = getattr(ev, "model_prob", None)
    if mp is None or (isinstance(mp, float) and not math.isfinite(mp)):
        return rs  # no model prob → pure rule priority
    return float(mp) * rs


def _best_event(events: Sequence[PatternEvent]) -> PatternEvent:
    return max(events, key=_priority)


def confluence_score(group: Sequence[PatternEvent], cfg: CorrelationConfig) -> float:
    """§4.5 standard confluence score.

    ``s = w1*term_n + w2*exp(-dt/tau_time) + w3*exp(-dprice/tau_price)`` where
    term_n = (n-1)/max(n-1,1), dt = min time gap (bars) and dprice = min price
    gap (ATR) from the group's best event.  Weighted sum is normalised by
    ``w1+w2+w3`` so the score lives in [0,1].
    """
    if not group:
        return 0.0
    best = _best_event(group)
    n = len(group)
    n_max = (
        cfg.n_patterns_max
        if cfg.n_patterns_max is not None and cfg.n_patterns_max > 1
        else n
    )
    term_n = (n - 1) / max(n_max - 1, 1) if n > 1 else 0.0

    p0 = best.entry_price
    atr_val = _merged_atr(default_atr_resolver(group).get(best.event_id), None) or 1.0
    bar_s = cfg.bar_seconds if cfg.bar_seconds and cfg.bar_seconds > 0 else 1.0

    dt_bars: list[float] = []
    dprice_atr: list[float] = []
    for ev in group:
        if ev.event_id == best.event_id:
            continue
        dt_bars.append(_dt_seconds(ev, best) / bar_s)
        dprice_atr.append(abs(ev.entry_price - p0) / max(atr_val, 1e-9))
    dt = min(dt_bars) if dt_bars else 0.0
    dprice = min(dprice_atr) if dprice_atr else 0.0

    wsum = cfg.w1 + cfg.w2 + cfg.w3
    score = (
        cfg.w1 * term_n
        + cfg.w2 * math.exp(-dt / max(cfg.tau_time, 1e-9))
        + cfg.w3 * math.exp(-dprice / max(cfg.tau_price, 1e-9))
    ) / wsum
    return round(float(score), 4)


class CorrelationManager:
    """Partitions a batch of events into correlation groups and applies the
    per-symbol policy (§4.2-§4.5)."""

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self.config = config or CorrelationConfig()
        self._group_counter = 0

    # ------------------------------------------------------------------

    def _new_group_id(self, symbol: str) -> str:
        self._group_counter += 1
        return f"corr-{symbol}-{self._group_counter:04d}"

    def group(
        self,
        events: Sequence[PatternEvent],
        config: CorrelationConfig | None = None,
        atr: Callable[[str], float | None] | None = None,
    ) -> list[ResolvedGroup]:
        """Partition *events* into correlation groups and apply the policy.

        * ``config`` defaults to the manager's config (a per-symbol override
          may be passed per call).
        * ``atr`` resolves per-event ATR for the §4.2 price window; defaults to
          :func:`default_atr_resolver` over the batch.

        Returns one :class:`ResolvedGroup` per correlation group (or per event
        when mode is ``independent``).  Mutates each event's
        ``confluence_group_id`` (§2.1) and, for events deduped away, stamps
        ``attributes.discard_reason = "correlated"`` (§7.1).
        """
        cfg = config or self.config
        evs = list(events)
        if not evs:
            return []

        atr_map = default_atr_resolver(evs)
        atr_resolver = atr or (lambda eid: atr_map.get(eid))

        # ---------------- union-find over pairwise correlation ----------------
        uf = _UnionFind(len(evs))
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                if _pairwise_correlated(evs[i], evs[j], cfg, atr_resolver):
                    uf.union(i, j)

        groups: dict[int, list[PatternEvent]] = {}
        for idx, ev in enumerate(evs):
            groups.setdefault(uf.find(idx), []).append(ev)

        order = {id(ev): i for i, ev in enumerate(evs)}
        results: list[ResolvedGroup] = []
        for members in groups.values():
            members = sorted(members, key=lambda e: order[id(e)])
            sym = members[0].symbol
            direction = members[0].direction
            gid = self._new_group_id(sym) if len(members) > 1 else None
            for m in members:
                m.confluence_group_id = gid

            if cfg.mode == MODE_INDEPENDENT:
                results.extend(
                    ResolvedGroup(
                        mode=cfg.mode, symbol=m.symbol, direction=m.direction,
                        group_id=None, events=[m], representative=m,
                    )
                    for m in members
                )
                continue

            best = _best_event(members)
            if cfg.mode == MODE_CONFLUENCE:
                results.append(
                    ResolvedGroup(
                        mode=cfg.mode, symbol=sym, direction=direction,
                        group_id=gid, events=members, representative=best,
                        confluence_score=confluence_score(members, cfg),
                        n_patterns=len(members),
                    )
                )
            else:  # dedup
                discarded = [m for m in members if m.event_id != best.event_id]
                for d in discarded:
                    d.attributes["discard_reason"] = "correlated"
                results.append(
                    ResolvedGroup(
                        mode=cfg.mode, symbol=sym, direction=direction,
                        group_id=gid, events=members, representative=best,
                        n_patterns=len(members), discarded=discarded,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # §4.4 Exposure caps
    # ------------------------------------------------------------------

    def check_exposure_caps(
        self,
        symbol: str,
        direction: str,
        proposed_risk: float,
        proposed_entry: float,
        proposed_atr: float | None,
        open_positions: Sequence[Any],
        config: CorrelationConfig | None = None,
    ) -> tuple[bool, str]:
        """Enforce §4.4 mandatory caps before opening a new position.

        * ``max_total_risk_per_symbol``: total risk of all open positions on
          the symbol (incl. the proposed one, in risk-normalised units) must
          stay within the cap.
        * ``max_direction_cluster_risk``: sum of risk of same-direction
          positions whose entry lies within ``corr_price_window_atr`` ATR of
          the proposed entry must stay within the cap.

        Returns ``(ok, reason)``; ``reason`` is empty when ``ok``.  This is a
        policy helper — the engine/RiskGuard supplies per-position risk via
        :func:`position_risk` / :func:`position_direction` / :func:`position_entry`.
        """
        cfg = config or self.config
        total = proposed_risk + sum(self.position_risk(p) for p in open_positions)
        if total > cfg.max_total_risk_per_symbol:
            return (
                False,
                f"symbol {symbol}: total risk {total:.3f} > cap "
                f"{cfg.max_total_risk_per_symbol:.3f} (max_total_risk_per_symbol)",
            )

        # direction-cluster: same direction, entry within 1 ATR of proposed
        cluster = proposed_risk
        atr_ref = proposed_atr if proposed_atr and math.isfinite(proposed_atr) and proposed_atr > 0 else 1.0
        for p in open_positions:
            if not _direction_match(self.position_direction(p), direction):
                continue
            e = self.position_entry(p)
            if math.isfinite(e) and abs(e - proposed_entry) <= 1.0 * atr_ref:
                cluster += self.position_risk(p)
        if cluster > cfg.max_direction_cluster_risk:
            return (
                False,
                f"symbol {symbol}: direction-cluster risk {cluster:.3f} > cap "
                f"{cfg.max_direction_cluster_risk:.3f} "
                f"(max_direction_cluster_risk)",
            )
        return True, ""

    # -- position accessor hooks (overridable for test / adapter) ---------

    @staticmethod
    def position_risk(p: Any) -> float:
        r = getattr(p, "risk", None)
        if r is not None:
            try:
                return float(r)
            except (TypeError, ValueError):
                pass
        if isinstance(p, dict):
            for key in ("risk", "risk_fraction", "position_size", "volume"):
                if p.get(key) is not None:
                    try:
                        return float(p[key])
                    except (TypeError, ValueError):
                        pass
        s = getattr(p, "position_size", None)
        if s is not None:
            try:
                return float(s)
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def position_direction(p: Any) -> str:
        d = getattr(p, "direction", None)
        if d is None and isinstance(p, dict):
            d = p.get("direction", p.get("type", ""))
        return str(d or "").lower()

    @staticmethod
    def position_entry(p: Any) -> float:
        e: Any = getattr(p, "entry_price", None)
        if e is None and isinstance(p, dict):
            e = p.get("entry_price", p.get("open_price", p.get("price")))
        try:
            return float(e)
        except (TypeError, ValueError):
            return float("nan")


def _direction_match(pos_dir: str, proposed_dir: str) -> bool:
    a = pos_dir.lower()
    b = proposed_dir.lower()
    aliases = {"bullish": "long", "buy": "long", "bearish": "short", "sell": "short"}
    return aliases.get(a, a) == aliases.get(b, b)


__all__ = [
    "MODE_CONFLUENCE",
    "MODE_DEDUP",
    "MODE_INDEPENDENT",
    "VALID_MODES",
    "CorrelationConfig",
    "CorrelationError",
    "CorrelationManager",
    "ResolvedGroup",
    "confluence_score",
    "default_atr_resolver",
]
