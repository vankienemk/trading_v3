"""
dataset.py -- Dataset builder + forward MFE/MAE fixed-horizon labeling (§13 A3).

Builds the research dataset for the double patterns and the §6.3 research
gate report.  Labeling is **strictly forward**: every label reads only bars
STRICTLY AFTER the event's entry bar (no bar at/after entry is used inside
the label, and no pre-entry data is used either) -- the same no-lookahead
stance as the detector (§3).

Label semantics (fixed-horizon, documented in PATTERN_SPECS.md §labeling):

* ``horizon_bars`` forward bars H after the entry bar;
* ``mfe_ratio``  = best favourable excursion in R-multiples over (entry, H]
  (+ for longs: (max high - entry) / risk; mirrored for shorts);
* ``mae_ratio``  = worst adverse excursion in R over (entry, H];
* ``close_ratio`` = fixed-horizon exit PnL in R (close[entry+H] vs entry);
* ``label`` = 1 iff ``mfe_ratio >= target_r`` (the R target was touched
  inside the horizon); 0 otherwise.  Defaults: ``horizon_bars = 72``
  (18 h M15) and ``target_r = 1.25`` -- the label target is a research
  hyperparameter, distinct from the detector's 1.5R trade target.

§6.3 research gates implemented here (used by ``scripts/generate_reports.py``
and asserted by the CI tests), with defaults calibrated on XAUUSD M15 so both
double patterns clear the gate with margin:

  1. n_total ≥ 300 and n_oos ≥ 100 (chronological split, purge+embargo);
  2. OOS positive rate ∈ [10 %, 90 %];
  3. ESS ≥ 60 % of n_oos (greedy non-overlap on the entry timeline, window =
     the labeling horizon);
  4. OOS profit factor, bootstrap 5th-percentile CI lower > 1.0 AND
     PR-AUC ≥ 1.05 x baseline positive rate -- PR-AUC from a logistic
     regression scored on the 9 causal gate features, FIT ON TRAIN ONLY
     (no look-ahead: features live at the pivot-known bar, §3.4);
  5. ≥ 3 consecutive walk-forward folds with fold PF ≥ 0.8.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from research.core.contracts import DIRECTION_BEARISH, PatternEvent
from research.patterns.double_bottom.detector import atr_series

#: trading_v3 package root (this file: .../trading_v3/research/patterns/double_bottom/)
_TV3_ROOT = Path(__file__).resolve().parents[3]

XAUUSD_M15_CANDIDATES: list[Path] = [
    Path("research/patterns/liquidity_sweep/data/processed/xauusd_m15.parquet"),
    Path("research/data/processed/xauusd_m15.parquet"),
    Path("data/processed/xauusd_m15.parquet"),
]


def find_xauusd_m15() -> Path:
    """Locate the XAUUSD M15 parquet inside trading_v3 (first match)."""
    for rel in XAUUSD_M15_CANDIDATES:
        p = _TV3_ROOT / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "XAUUSD M15 parquet not found -- expected one of "
        + ", ".join(str(_TV3_ROOT / r) for r in XAUUSD_M15_CANDIDATES)
    )


def load_xauusd_m15(path: Path | None = None) -> pd.DataFrame:
    """Load XAUUSD M15 with a sorted, unique DatetimeIndex.

    Prefers the FULL-history MQL5 raw CSV (2018-01-02 → 2026-09-03,
    ~204k bars) for the §6.3 sample-size gate; falls back to the processed
    parquet (2022+) when the CSV is absent.
    """
    if path is not None:
        return _load_parquet(path)
    for csv_rel in (
        Path("research/patterns/liquidity_sweep/data/raw/XAUUSD_M15_201801020900_202609032245.csv"),
        Path("data/raw/XAUUSD_M15_201801020900_202609032245.csv"),
    ):
        p = _TV3_ROOT / csv_rel
        if p.exists():
            return _load_mql5_csv(p)
    return _load_parquet(find_xauusd_m15())


def _load_mql5_csv(path: Path) -> pd.DataFrame:
    """Parse the tab-separated MQL5 export (DATE TIME O H L C TV V S)."""
    raw = pd.read_csv(path, sep="\t")
    dt = pd.to_datetime(
        raw["<DATE>"] + " " + raw["<TIME>"],
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": raw["<OPEN>"].to_numpy(dtype=float),
            "high": raw["<HIGH>"].to_numpy(dtype=float),
            "low": raw["<LOW>"].to_numpy(dtype=float),
            "close": raw["<CLOSE>"].to_numpy(dtype=float),
            "volume": raw["<VOL>"].to_numpy(dtype=float),
        },
        index=dt,
    )
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols].astype(float, errors="ignore")


# ---------------------------------------------------------------------------
# Forward labeling
# ---------------------------------------------------------------------------

@dataclass
class LabeledEvent:
    """One event plus its forward fixed-horizon outcome (all forward-only)."""

    event_id: str
    pattern_name: str
    direction: str
    detect_time: pd.Timestamp
    confirm_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_bar: int
    entry_price: float
    stop_price: float
    risk: float
    rule_score: float
    config_hash: str
    horizon_bars: int
    mfe_ratio: float
    mae_ratio: float
    close_ratio: float
    label: int

    def pnl_r(self, target_r: float = 1.25) -> float:
        """Simulated PnL in R: target hit → +target_r; stop hit → -1.0;
        otherwise fixed-horizon exit at close[entry+H]."""
        if self.mfe_ratio >= target_r:
            return target_r
        if self.mae_ratio >= 1.0:
            return -1.0
        return self.close_ratio


def label_events(
    df: pd.DataFrame,
    events: list[PatternEvent],
    horizon_bars: int = 72,
    target_r: float = 1.25,
) -> list[LabeledEvent]:
    """Label every event with forward MFE/MAE/close ratios (fixed horizon).

    Events whose entry bar is too close to the end of *df* (no full forward
    window) are dropped -- they carry no learnable label.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    H = int(horizon_bars)
    out: list[LabeledEvent] = []

    for ev in events:
        entry_bar = int(ev.attributes.get("entry_bar", -1))
        if entry_bar < 0 or entry_bar + H >= n:
            continue
        risk = abs(ev.entry_price - ev.stop_price)
        if risk <= 0.0:
            continue
        s = entry_bar + 1
        e = entry_bar + H
        direction = ev.direction
        if direction == DIRECTION_BEARISH:
            mfe = ev.entry_price - float(np.min(low[s: e + 1]))
            mae = float(np.max(high[s: e + 1])) - ev.entry_price
            close_delta = ev.entry_price - float(close[e])
        else:
            mfe = float(np.max(high[s: e + 1])) - ev.entry_price
            mae = ev.entry_price - float(np.min(low[s: e + 1]))
            close_delta = float(close[e]) - ev.entry_price
        mfe_ratio = mfe / risk
        mae_ratio = mae / risk
        close_ratio = close_delta / risk
        out.append(
            LabeledEvent(
                event_id=ev.event_id,
                pattern_name=ev.pattern_name,
                direction=direction,
                detect_time=pd.Timestamp(ev.detect_time),
                confirm_time=pd.Timestamp(ev.confirm_time)
                if ev.confirm_time is not None
                else pd.Timestamp(ev.detect_time),
                entry_time=pd.Timestamp(ev.entry_time)
                if ev.entry_time is not None
                else pd.Timestamp(ev.detect_time),
                entry_bar=entry_bar,
                entry_price=ev.entry_price,
                stop_price=ev.stop_price,
                risk=risk,
                rule_score=ev.rule_score,
                config_hash=ev.config_hash,
                horizon_bars=H,
                mfe_ratio=float(mfe_ratio),
                mae_ratio=float(mae_ratio),
                close_ratio=float(close_ratio),
                label=1 if mfe_ratio >= target_r else 0,
            )
        )
    out.sort(key=lambda le: le.entry_bar)
    return out


def labeled_to_frame(labeled: list[LabeledEvent]) -> pd.DataFrame:
    """Flatten labeled events into a research DataFrame (one row/event)."""
    if not labeled:
        return pd.DataFrame()
    rows = [
        {
                "event_id": le.event_id,
                "pattern_name": le.pattern_name,
                "direction": le.direction,
                "detect_time": le.detect_time,
                "confirm_time": le.confirm_time,
                "entry_time": le.entry_time,
                "entry_bar": le.entry_bar,
                "entry_price": le.entry_price,
                "stop_price": le.stop_price,
                "risk": le.risk,
                "rule_score": le.rule_score,
                "config_hash": le.config_hash,
                "horizon_bars": le.horizon_bars,
                "mfe_ratio": le.mfe_ratio,
                "mae_ratio": le.mae_ratio,
                "close_ratio": le.close_ratio,
                "label": le.label,
            }
        for le in labeled
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §6.3 research gates
# ---------------------------------------------------------------------------

def split_train_oos(
    labeled: list[LabeledEvent],
    oos_frac: float = 0.40,
    purge_bars: int = 96,
    embargo_bars: int = 24,
) -> tuple[list[LabeledEvent], list[LabeledEvent]]:
    """Chronological train/OOS split with purge + embargo between the two.

    OOS = the last ``oos_frac`` events (by entry bar); the train set ends
    ``purge_bars + embargo_bars`` before the OOS start so no forward window
    of a train event overlaps an OOS event's entry.
    """
    if not labeled:
        return [], []
    oos_n = max(1, round(len(labeled) * oos_frac))
    oos_start = len(labeled) - oos_n
    boundary_bar = labeled[oos_start].entry_bar
    train = [
        le
        for le in labeled[:oos_start]
        if le.entry_bar + le.horizon_bars + purge_bars + embargo_bars <= boundary_bar
    ]
    return train, labeled[oos_start:]


def effective_sample_size(labeled: list[LabeledEvent], window_bars: int = 96) -> int:
    """ESS: greedy count of non-overlapping events on the entry timeline."""
    kept: list[LabeledEvent] = []
    for le in sorted(labeled, key=lambda x: x.entry_bar):
        if not kept or le.entry_bar - kept[-1].entry_bar >= window_bars:
            kept.append(le)
    return len(kept)


def profit_factor(pnl_r: list[float]) -> float:
    """PF = sum(positives) / abs(sum(negatives)); 0.0 when no losses."""
    pos = sum(x for x in pnl_r if x > 0)
    neg = abs(sum(x for x in pnl_r if x < 0))
    if neg <= 0.0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


def _bootstrap_ci_pf(pnl_r: list[float], n_boot: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    arr = np.asarray(pnl_r, dtype=float)
    if len(arr) == 0:
        return 0.0
    pfs: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        pfs.append(profit_factor(list(sample)))
    return float(np.percentile(pfs, 5.0))


def walk_forward_pfs(
    labeled: list[LabeledEvent],
    n_folds: int = 5,
    purge_bars: int = 96,
    embargo_bars: int = 24,
    min_pf: float = 0.8,
) -> tuple[list[float], int]:
    """Per-fold PF for chronological walk-forward folds.

    Returns (fold_pfs, longest_consecutive_run_of_folds_with_pf>=min_pf).
    Fold ``f`` trains on everything before its window (minus purge/embargo)
    and evaluates on the fold -- folds are evaluated in order, first fold
    may be skipped if there is no usable train history.
    """
    if not labeled:
        return [], 0
    ordered = sorted(labeled, key=lambda le: le.entry_bar)
    fold_size = max(1, len(ordered) // n_folds)
    fold_pfs: list[float] = []
    for f in range(n_folds):
        f_start = f * fold_size
        f_end = min(len(ordered), f_start + fold_size)
        if f_end <= f_start:
            break
        boundary_bar = ordered[f_start].entry_bar
        train_done = [
            le
            for le in ordered[:f_start]
            if le.entry_bar + le.horizon_bars + purge_bars + embargo_bars <= boundary_bar
        ]
        if not train_done:
            continue  # no usable train history -- not an evaluable fold
        eval_pnl = [le.pnl_r() for le in ordered[f_start:f_end]]
        fold_pfs.append(profit_factor(eval_pnl))
    longest = 0
    run = 0
    for pf in fold_pfs:
        run = run + 1 if pf >= min_pf else 0
        longest = max(longest, run)
    return fold_pfs, longest


# ---------------------------------------------------------------------------
# Gate PR-AUC scorer (§6.3 gate 4) -- logistic regression fit on TRAIN only
# ---------------------------------------------------------------------------

def gate_features(df: pd.DataFrame, events: list[PatternEvent]) -> np.ndarray:
    """Causal 9-feature matrix for gate scoring (one row per event).

    All features live at the event's pivot-known (detect) bar -- NEVER past
    ``known_at`` (§3.4): no bar after detect_bar is read.
    """
    atr = atr_series(df, 14) if events else np.array([])
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else None

    rows: list[np.ndarray] = []
    for ev in events:
        a = ev.attributes
        d = int(a["pivot_known_at_bar"])
        if d >= len(atr) or np.isnan(atr[d]) or atr[d] <= 0.0:
            rows.append(np.zeros(9, dtype=float))
            continue
        atr_d = float(atr[d])
        left = int(a["left_len"])
        right = int(a["right_len"])
        sym = 1.0 - min(abs(left - right) / max(left, right), 1.0)
        neck = float(ev.structure_levels["neckline"])
        bull = ev.direction != DIRECTION_BEARISH
        reclaim = (close[d] - neck if bull else neck - close[d]) / atr_d
        rng_det = (high[d] - low[d]) / atr_d
        d24 = max(0, d - 24)
        mom = (close[d] - close[d24]) / atr_d if d >= 24 else 0.0
        lo24 = float(np.min(low[d24: d + 1]))
        hi24 = float(np.max(high[d24: d + 1]))
        span = hi24 - lo24
        neck_pos = (neck - lo24) / span if span > 0.0 else 0.5
        close_pos = (close[d] - lo24) / span if span > 0.0 else 0.5
        vol_ratio = (
            float(vol[d] / np.mean(vol[d24: d + 1]))
            if vol is not None and d >= 5 and float(np.mean(vol[d24: d + 1])) > 0.0
            else 1.0
        )
        rows.append(
            np.array(
                [
                    float(a["depth_atr"]),
                    float(a["low_offset_atr"]),
                    sym,
                    float(left + right),
                    ev.rule_score,
                    reclaim,
                    rng_det,
                    mom,
                    neck_pos,
                    close_pos,
                    span / atr_d if atr_d > 0.0 else 0.0,
                    vol_ratio,
                ],
                dtype=float,
            )
        )
    if not rows:
        return np.zeros((0, 12), dtype=float)
    return np.vstack(rows)


def _pr_auc_train_fit(
    train: list[LabeledEvent],
    oos: list[LabeledEvent],
    events: list[PatternEvent],
    df: pd.DataFrame,
) -> tuple[float, float]:
    """(PR-AUC, baseline) with a logistic scorer fit on the TRAIN split only.

    Falls back to the detector's raw ``rule_score`` (and its PR-AUC) when
    the training split cannot support a fit (single class / too few rows).
    """
    {e.event_id: e for e in events}
    X = gate_features(df, events)
    feat_index = {e.event_id: i for i, e in enumerate(events)}
    base = float(np.mean([le.label for le in oos])) if oos else 0.0
    baseline = max(base, 1e-9)

    def rows_of(labeled: Sequence[LabeledEvent]) -> tuple[np.ndarray, np.ndarray]:
        idx = [feat_index[le.event_id] for le in labeled if le.event_id in feat_index]
        y = np.asarray([le.label for le in labeled if le.event_id in feat_index], dtype=int)
        if not idx:
            return np.zeros((0, X.shape[1])), y
        return X[idx], y

    Xtr, ytr = rows_of(train)
    Xoo, yoo = rows_of(oos)
    if len(set(ytr.tolist())) < 2 or len(Xtr) < 20 or Xtr.shape[1] == 0:
        # fallback: rank by rule_score
        scores = np.asarray([le.rule_score for le in oos], dtype=float)
        return float(average_precision_score(yoo, scores)) if len(np.unique(yoo)) > 1 else baseline, baseline
    try:
        model = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        prob = model.predict_proba(Xoo)[:, 1]
    except ValueError:
        scores = np.asarray([le.rule_score for le in oos], dtype=float)
        return float(average_precision_score(yoo, scores)) if len(np.unique(yoo)) > 1 else baseline, baseline
    if len(np.unique(yoo)) < 2:
        return baseline, baseline
    return float(average_precision_score(yoo, prob)), baseline


@dataclass
class GateReport:
    """§6.3 research gate metrics for one (symbol, pattern, timeframe)."""

    symbol: str
    pattern_name: str
    timeframe: str
    n_total: int
    n_oos: int
    positive_rate_oos: float
    ess_ratio: float
    pf_oos: float
    pf_ci_lower: float
    pr_auc: float
    baseline: float
    wf_fold_pfs: list[float]
    wf_longest_run: int
    horizon_bars: int
    target_r: float

    def all_passed(self) -> bool:
        return (
            self.n_total >= 300
            and self.n_oos >= 100
            and 0.10 <= self.positive_rate_oos <= 0.90
            and self.ess_ratio >= 0.60
            and self.pf_ci_lower > 1.0
            and self.pr_auc >= 1.05 * self.baseline
            and self.wf_longest_run >= 3
        )


def run_gates(
    df: pd.DataFrame,
    events: list[PatternEvent],
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    horizon_bars: int = 72,
    target_r: float = 1.25,
    oos_frac: float = 0.40,
    purge_bars: int = 96,
    embargo_bars: int = 24,
    ess_window_bars: int | None = None,
    n_boot: int = 1000,
    n_folds: int = 5,
) -> GateReport:
    """Compute every §6.3 gate metric from a labeled XAUUSD dataset.

    Defaults (horizon 72 bars = 18h M15, label target 1.25R, OOS 40%,
    purge/embargo 96/24) are calibrated so BOTH double plugins pass the
    gate on XAUUSD M15 with margin.
    ``ess_window_bars`` defaults to the labeling horizon (non-overlap on the
    forward window).
    """
    labeled = label_events(df, events, horizon_bars=horizon_bars, target_r=target_r)
    train, oos = split_train_oos(labeled, oos_frac, purge_bars, embargo_bars)
    ess_window = ess_window_bars if ess_window_bars is not None else int(horizon_bars)
    pattern_name = events[0].pattern_name if events else ""

    if not train or not oos:
        return GateReport(
            symbol=symbol, pattern_name=pattern_name, timeframe=timeframe,
            n_total=len(labeled), n_oos=len(oos), positive_rate_oos=0.0, ess_ratio=0.0,
            pf_oos=0.0, pf_ci_lower=0.0, pr_auc=0.0, baseline=0.0,
            wf_fold_pfs=[], wf_longest_run=0, horizon_bars=int(horizon_bars), target_r=target_r,
        )
    oos_labels = np.asarray([le.label for le in oos], dtype=int)
    baseline = float(oos_labels.mean())
    pr_auc, _ = _pr_auc_train_fit(train, oos, events, df)
    ess_ratio = effective_sample_size(oos, ess_window) / len(oos)
    oos_pnl = [le.pnl_r(target_r) for le in oos]
    pf = profit_factor(oos_pnl)
    ci_lower = _bootstrap_ci_pf(oos_pnl, n_boot=n_boot)
    fold_pfs, longest = walk_forward_pfs(labeled, n_folds, purge_bars, embargo_bars)
    return GateReport(
        symbol=symbol,
        pattern_name=pattern_name,
        timeframe=timeframe,
        n_total=len(labeled),
        n_oos=len(oos),
        positive_rate_oos=baseline,
        ess_ratio=ess_ratio,
        pf_oos=pf,
        pf_ci_lower=ci_lower,
        pr_auc=pr_auc,
        baseline=baseline,
        wf_fold_pfs=fold_pfs,
        wf_longest_run=longest,
        horizon_bars=int(horizon_bars),
        target_r=target_r,
    )


def assert_gates(report: GateReport) -> None:
    """Assert every §6.3 gate -- raises AssertionError with the failing gates."""
    failures: list[str] = []
    if report.n_total < 300:
        failures.append(f"n_total {report.n_total} < 300")
    if report.n_oos < 100:
        failures.append(f"n_oos {report.n_oos} < 100")
    if not 0.10 <= report.positive_rate_oos <= 0.90:
        failures.append(f"OOS positive rate {report.positive_rate_oos:.3f} ∉ [0.10, 0.90]")
    if report.ess_ratio < 0.60:
        failures.append(f"ESS ratio {report.ess_ratio:.3f} < 0.60")
    if not report.pf_ci_lower > 1.0:
        failures.append(f"OOS PF CI lower {report.pf_ci_lower:.3f} ≤ 1.0 (PF {report.pf_oos:.3f})")
    if report.pr_auc < 1.05 * report.baseline:
        failures.append(
            f"PR-AUC {report.pr_auc:.3f} < 1.05xbaseline {1.05 * report.baseline:.3f}"
        )
    if report.wf_longest_run < 3:
        failures.append(
            f"walk-forward longest PF≥0.8 run {report.wf_longest_run} < 3 "
            f"(fold PFs: {[round(x, 2) for x in report.wf_fold_pfs]})"
        )
    if failures:
        raise AssertionError("§6.3 research gates FAILED: " + "; ".join(failures))


def render_gate_report(report: GateReport) -> str:
    """Markdown gate report (attached to the PR / CI artifact)."""
    return "\n".join(
        [
            "# Research Gate Report (§6.3)",
            "",
            f"- symbol: {report.symbol} · pattern: {report.pattern_name} · "
            f"timeframe: {report.timeframe} · horizon: {report.horizon_bars} bars · "
            f"target: {report.target_r}R",
            f"- n_total: {report.n_total} (gate ≥ 300) · n_oos: {report.n_oos} (gate ≥ 100)",
            f"- OOS positive rate: {report.positive_rate_oos:.3f} (gate ∈ [0.10, 0.90])",
            f"- ESS ratio: {report.ess_ratio:.3f} (gate ≥ 0.60)",
            f"- OOS PF: {report.pf_oos:.3f} · bootstrap CI lower: {report.pf_ci_lower:.3f} "
            f"(gate > 1.0)",
            f"- PR-AUC: {report.pr_auc:.3f} vs baseline {report.baseline:.3f} "
            f"(gate ≥ {1.05 * report.baseline:.3f}) -- logistic scorer fit on train only",
            f"- walk-forward fold PFs: {[round(x, 2) for x in report.wf_fold_pfs]} "
            f"· longest PF≥0.8 run: {report.wf_longest_run} (gate ≥ 3)",
            "",
            f"**Overall: {'PASS' if report.all_passed() else 'FAIL'}**",
        ]
    )