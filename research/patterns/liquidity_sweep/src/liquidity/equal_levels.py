"""Equal highs / equal lows — liquidity pools (guide §9.3).

Two or more touches within ``tolerance_atr x ATR`` of each other form an equal
level.  The cluster becomes a *level* at the touch that brings it to
``min_touches``; that touch's timestamp is ``known_at`` (the earliest moment
the level is usable).  Until then it is only a pending cluster.

Causality contract
------------------
- Clustering is causal: bar ``t`` may only join clusters formed from bars
  ``< t`` (single streaming pass, no look-ahead).
- A cluster's ``price`` evolves as the mean of its touched prices; a level is
  emitted exactly once, at the touch that first reaches ``min_touches``.
- ``known_at = timestamp[last touch at emission]``.
- Clusters whose last touch is older than ``max_age_bars`` expire and can no
  longer absorb new touches (they may never reach ``min_touches``).

Rows follow ``src.schema.LEVEL_COLUMNS`` plus ``known_pos``/``is_h1``/
``max_age_bars`` and the internal column ``touch_positions`` (exact bar
positions of every touch in the cluster at emission; consumed — and dropped —
by ``level_registry``).  Level-id scheme: ``equal_{direction}_{origin_pos}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import LEVEL_COLUMNS

#: Equal-level output = schema §4 columns + known_pos/is_h1/max_age_bars and
#: the internal ``touch_positions`` column (exact cluster touch positions,
#: consumed and dropped by ``level_registry``).
EQUAL_OUTPUT_COLUMNS: list[str] = [
    *LEVEL_COLUMNS, "known_pos", "is_h1", "max_age_bars", "touch_positions",
]


def _validate_frame(df: pd.DataFrame, func: str) -> None:
    missing = [c for c in ("high", "low") if c not in df.columns]
    if missing:
        raise ValueError(f"{func}: missing required columns {missing}")
    if len(df) == 0:
        raise ValueError(f"{func}: empty DataFrame")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"{func}: index must be chronologically sorted (monotonic "
            "increasing); an unsorted frame breaks causal (no-look-ahead) "
            "semantics"
        )
    non_numeric = [c for c in ("high", "low") if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"{func}: columns {non_numeric} must be numeric")


def detect_equal_levels(
    df: pd.DataFrame,
    tolerance_atr: float = 0.10,
    min_touches: int = 2,
    max_age_bars: int = 200,
    atr: pd.Series | None = None,
) -> pd.DataFrame:
    """Detect equal-high and equal-low clusters causally.

    Parameters
    ----------
    tolerance_atr:
        Cluster tolerance as a multiple of the per-bar ATR (baseline 0.10).
    min_touches:
        Minimum touches (incl. the forming one) to turn a cluster into a level.
    max_age_bars:
        Cluster lifetime; a cluster whose last touch is older than this expires.
    atr:
        Causal ATR series (``add_atr(df)["atr"]``).  Required — the tolerance
        is ATR-scaled.  Bars with non-finite ATR (indicator warm-up) are skipped.

    Returns a long-format levels table sorted by ascending ``known_at``.
    """
    _validate_frame(df, "detect_equal_levels")
    if atr is None:
        raise ValueError("detect_equal_levels: an ATR series is required")
    if len(atr) != len(df):
        raise ValueError(
            f"detect_equal_levels: ATR length {len(atr)} != frame length {len(df)}"
        )
    if tolerance_atr <= 0:
        raise ValueError(
            f"detect_equal_levels: tolerance_atr must be > 0, got {tolerance_atr}"
        )
    if min_touches < 2:
        raise ValueError(
            f"detect_equal_levels: min_touches must be >= 2, got {min_touches}"
        )
    if max_age_bars < 1:
        raise ValueError(
            f"detect_equal_levels: max_age_bars must be >= 1, got {max_age_bars}"
        )

    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    atr_arr = np.asarray(atr, dtype="float64")
    ts = df.index.to_numpy()
    n = len(df)

    levels: list[dict] = []

    def _cluster_passes(direction: str, price_arr: np.ndarray) -> None:
        """Stream bars and accumulate equal-price clusters for one direction."""
        active: list[dict] = []  # pending / emitted clusters, by price proximity

        for i in range(n):
            tol = tolerance_atr * atr_arr[i]
            if not np.isfinite(tol) or tol <= 0:
                # ATR warm-up: bar cannot be judged against a tolerance.
                continue
            px = float(price_arr[i])

            # Nearest active cluster within tolerance (ties -> earliest).
            best = None
            best_dist = np.inf
            for c in active:
                d = abs(px - c["price"])
                if d <= tol and d < best_dist:
                    best = c
                    best_dist = d

            if best is None:
                active.append(
                    {
                        "price": px,
                        "price_min": px,
                        "price_max": px,
                        "touch_count": 1,
                        "first_pos": i,
                        "first_touch": ts[i],
                        "last_pos": i,
                        "last_touch": ts[i],
                        "positions": [i],
                        "emitted": False,
                    }
                )
            else:
                best["touch_count"] += 1
                best["price_min"] = min(best["price_min"], px)
                best["price_max"] = max(best["price_max"], px)
                best["price"] = (best["price_min"] + best["price_max"]) / 2.0
                best["last_pos"] = i
                best["last_touch"] = ts[i]
                best["positions"].append(i)

            # Emit levels that just reached min_touches (snapshot at emission).
            for c in active:
                if c["touch_count"] >= min_touches and not c["emitted"]:
                    c["emitted"] = True
                    levels.append(
                        {
                            "level_id": f"equal_{direction}_{c['first_pos']}",
                            "level_type": "equal",
                            "direction": direction,
                            "price": c["price"],
                            "price_min": c["price_min"],
                            "price_max": c["price_max"],
                            "origin_pos": int(c["first_pos"]),
                            "origin_time": c["first_touch"],
                            "known_at": c["last_touch"],
                            "status": "active",
                            "touch_count": int(c["touch_count"]),
                            "first_touch_time": c["first_touch"],
                            "last_touch_time": c["last_touch"],
                            "known_pos": int(c["last_pos"]),
                            "is_h1": False,
                            "max_age_bars": int(max_age_bars),
                            "touch_positions": list(c["positions"]),
                        }
                    )

            # Expire clusters whose last touch is older than max_age_bars.
            active = [c for c in active if i - c["last_pos"] <= max_age_bars]

    _cluster_passes("low", low)
    _cluster_passes("high", high)

    if not levels:
        return pd.DataFrame(columns=EQUAL_OUTPUT_COLUMNS)
    out = pd.DataFrame(levels, columns=EQUAL_OUTPUT_COLUMNS)
    out = out.sort_values("known_at").reset_index(drop=True)
    return out