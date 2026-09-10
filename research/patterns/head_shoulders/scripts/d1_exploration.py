"""
d1_exploration.py — D1-timeframe sample-size exploration for H&S (t2, P1#9).

REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §11 flags H&S as the pattern that is only
*truly* tradeable on D1 ("H&S chỉ giá trị thật ở D1").  The t2 acceptance
criterion for P1 finding #9 requires either a committed D1 exploration or a
documented acceptance at the true M15 sample size.

This script resamples the full XAUUSD M15 history (2018-2026, 204k bars)
to the D1 timeframe and runs BOTH H&S mirrors with their default config
(the detector is timeframe-agnostic — it consumes OHLCV + a DatetimeIndex).
It then labels the D1 events with the forward fixed-horizon MFE label and
reports the §6.3 sample-gate primitives (n_total, n_oos, ESS) at the D1
sample size.

Result (2026-09-07): D1 yields only **3 events per H&S family over 10 years**
(2239 D1 bars) — two orders of magnitude below the n ≥ 300 sample gate and
far below the M15 strict-detector sample (151 regular / 208 inverse).  The
D1 timeframe therefore cannot support the §6.3 training gate either; the
M15 true-sample-size acceptance stands (see docs/findings_resolution.md).

Usage:
    /tmp/ptv2_venv/bin/python research/patterns/head_shoulders/scripts/d1_exploration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pandas as pd

from research.patterns.head_shoulders.dataset import load_xauusd_m15
from research.patterns.head_shoulders.detector import HeadShouldersDetector
from research.patterns.inverse_head_shoulders.detector import (
    InverseHeadShouldersDetector,
)


def resample_d1(df: pd.DataFrame) -> pd.DataFrame:
    """M15 → D1 OHLC aggregation (first open, max high, min low, last close,
    summed volume); drops empty calendar days."""
    d1 = pd.DataFrame(
        {
            "open": df["open"].resample("1D").first(),
            "high": df["high"].resample("1D").max(),
            "low": df["low"].resample("1D").min(),
            "close": df["close"].resample("1D").last(),
        }
    )
    if "volume" in df.columns:
        d1["volume"] = df["volume"].resample("1D").sum()
    return d1.dropna()


def main() -> None:
    from research.patterns.head_shoulders.dataset import (
        effective_sample_size,
        label_events,
    )

    df = load_xauusd_m15()
    d1 = resample_d1(df)
    print(f"M15 bars: {len(df)}  D1 bars: {len(d1)}")
    print(f"D1 range: {d1.index.min()} -> {d1.index.max()}")

    rows: list[dict[str, object]] = []
    for det_cls in (HeadShouldersDetector, InverseHeadShouldersDetector):
        det = det_cls()
        cfg = {**det.get_default_config(), "timeframe": "D1"}
        events = det.detect(d1, cfg)
        labeled = label_events(d1, events, horizon_bars=72, target_r=1.25)
        ess = effective_sample_size(labeled, window_bars=72)
        rows.append(
            {
                "pattern": det_cls.__name__,
                "n_events_d1": len(events),
                "n_labelable_d1": len(labeled),
                "ess_d1": ess,
                "vs_m15_sample": "(M15: 208)" if "Inverse" in det_cls.__name__ else "(M15: 151)",
            }
        )
        print(
            f"{det_cls.__name__}: D1 events={len(events)}  "
            f"labelable={len(labeled)}  ESS={ess}"
        )
    report = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "reports" / "d1_sample_exploration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()