"""
walkforward_trainer.py — self-contained causal walk-forward trainer (§6.2 / §6.4).

Implements the research → train → registry leg of the multi-pattern pipeline
for the *classical* pattern plugins (double_bottom / double_top / wedges /
H&S).  It is deliberately **pattern-agnostic**: it consumes the canonical
causal event stamps the shared detectors already emit

* ``attributes["pivot_known_at_bar"]``  — the detect bar (§3.2 right-bar rule);
* ``attributes["confirm_bar"]``         — the effective confirmation bar;
* ``attributes["depth_atr"]`` /
  ``attributes["low_offset_atr"]`` /
  ``attributes["left_len"]`` /
  ``attributes["right_len"]``           — canonical structural attributes
  (wedge / H&S detectors alias their family-specific geometry onto these);
* ``structure_levels["neckline"]``      — the breakout level;

and derives its whole feature matrix from **raw OHLCV bars ≤ confirm_bar**
plus those attributes — it never reads a bar past the event's ``known_at``
(§3.4).  Labels are the forward fixed-horizon MFE labels produced by
``pattern.dataset.label_events`` (they read only bars > entry_bar).

The trained model is the §6.4 **tier-2 meta-model**: a binary classifier
learning "trade vs skip" from the causal event state; its calibrated output
is exactly ``PatternEvent.model_prob``.  Every fit — including the isotonic
calibrator (``CalibratedClassifierCV``) — happens **on the train split only**;
the chronological OOS block and every walk-forward fold are evaluated once,
never touched by any fit.

Artifacts layout (mirrors the eurusd/v2 reference the live engine consumes):

    <out_dir>/model.pkl        joblib dict {model, scaler, feature_names, impute_medians}
    <out_dir>/calibrator.pkl   the CalibratedClassifierCV itself (engine fallback)
    <out_dir>/features.json    ordered feature contract for live inference
    <out_dir>/train_summary.json  §5.5 registry-ready metadata

The feature matrix is written with ``fillna(0.0)`` preprocessing baked in so
training == live inference input contract (`live.engine._compute_model_prob`
also fills NaN with 0.0).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

#: How NaN feature values are preprocessed (baked into the artifact contract).
NA_FILL = 0.0

#: Detect-bar features present for every pattern family (all read bars
#: strictly ≤ the pivot-known / detect bar).
_DETECT_FEATURES: tuple[str, ...] = (
    "depth_atr",
    "low_offset_atr",
    "symmetry_ratio",
    "pattern_length",
    "atr_level",
    "reclaim_atr",
    "range_atr",
    "mom_24_atr",
    "neck_pos_24",
    "close_pos_24",
    "span_24_atr",
    "vol_ratio_24",
)

#: Confirm-bar features (read bars ≤ the effective confirm bar).
_CONFIRM_FEATURES: tuple[str, ...] = (
    "confirm_reclaim_atr",
    "confirm_range_atr",
    "confirm_body_ratio",
    "confirm_close_location",
    "confirm_vol_zscore",
    "bars_confirm_delay",
)

#: Market-context features (detect-bar timestamp context).
_CONTEXT_FEATURES: tuple[str, ...] = (
    "atr_ratio_200",
    "hour_utc",
    "day_of_week",
    "session_asia",
    "session_london",
    "session_new_york",
    "rule_score",
)

#: Session windows (UTC) used for the session one-hot flags.
_SESSION_ASIA = (0, 8)
_SESSION_LONDON = (8, 16)
_SESSION_NEW_YORK = (13, 21)


def _atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray[Any, Any]:
    """Causal ATR(period): value at bar ``b`` uses only bars ``<= b``."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    rolling = pd.Series(tr).rolling(int(period), min_periods=int(period)).mean()
    return np.asarray(rolling.to_numpy(), dtype=float)


def _session_flags(hour: int) -> tuple[int, int, int]:
    def in_(w: tuple[int, int]) -> int:
        return 1 if w[0] <= hour < w[1] else 0

    return (in_(_SESSION_ASIA), in_(_SESSION_LONDON), in_(_SESSION_NEW_YORK))


def build_feature_frame(
    df: pd.DataFrame,
    events: list[Any],
    extra_attr_features: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[str]]:
    """Build the causal per-event feature matrix (rows keyed by event_id).

    Every feature is computed from data available **at or before** the
    event's confirm bar — strictly causal (§3.4).  ``extra_attr_features``
    optionally copies extra pattern attributes (e.g. ``width_atr``,
    ``convergence_ratio``, ``head_depth_atr``) into the matrix when the
    detector stores them.

    Returns ``(X, feature_names)`` with ``X`` indexed by ``event_id``,
    preprocessed with ``fillna(NA_FILL)``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("build_feature_frame requires a DatetimeIndex")
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    opn = df["open"].to_numpy(dtype=float)
    atr = _atr_series(df)
    has_vol = "volume" in df.columns
    vol = df["volume"].to_numpy(dtype=float) if has_vol else np.zeros(len(df))
    extra = tuple(extra_attr_features)

    rows: list[dict[str, float]] = []
    kept_ids: list[str] = []
    for ev in events:
        a = ev.attributes
        d = int(a.get("pivot_known_at_bar", -1))
        c = int(a.get("confirm_bar", d))
        if d < 0 or d >= len(df) or c < d or c >= len(df):
            continue
        atr_d = float(atr[d]) if not np.isnan(atr[d]) else float("nan")
        atr_c = float(atr[c]) if not np.isnan(atr[c]) else float("nan")
        neck = float(ev.structure_levels.get("neckline", close[d]))
        bear = str(ev.direction) == "bearish"
        sign = -1.0 if bear else 1.0
        left = int(a.get("left_len", 0))
        right = int(a.get("right_len", 0))
        sym = (
            1.0 - min(abs(left - right) / max(left, right), 1.0)
            if max(left, right) > 0
            else 0.0
        )

        # --- detect-bar features (≤ pivot-known bar) ---------------------
        d24 = max(0, d - 24)
        lo24 = float(np.min(low[d24 : d + 1]))
        hi24 = float(np.max(high[d24 : d + 1]))
        span24 = hi24 - lo24
        mom = (close[d] - close[d24]) / atr_d if atr_d > 0 and d >= 24 else 0.0
        vol_win = vol[d24 : d + 1] if has_vol else np.array([1.0])
        vol_mean = float(np.mean(vol_win)) if len(vol_win) > 0 else 0.0
        row: dict[str, float] = {
            "depth_atr": _f(a.get("depth_atr", np.nan)),
            "low_offset_atr": _f(a.get("low_offset_atr", np.nan)),
            "symmetry_ratio": sym,
            "pattern_length": _f(a.get("pattern_length", left + right)),
            "atr_level": atr_d,
            "reclaim_atr": (close[d] - neck) * sign / atr_d if atr_d > 0 else 0.0,
            "range_atr": (high[d] - low[d]) / atr_d if atr_d > 0 else 0.0,
            "mom_24_atr": mom,
            "neck_pos_24": (neck - lo24) / span24 if span24 > 0 else 0.5,
            "close_pos_24": (close[d] - lo24) / span24 if span24 > 0 else 0.5,
            "span_24_atr": span24 / atr_d if atr_d > 0 else 0.0,
            "vol_ratio_24": float(vol[d] / vol_mean) if has_vol and vol_mean > 0 else 1.0,
            # --- confirm-bar features (≤ confirm bar) ---------------------
            "confirm_reclaim_atr": (close[c] - neck) * sign / atr_c if atr_c > 0 else 0.0,
            "confirm_range_atr": (high[c] - low[c]) / atr_c if atr_c > 0 else 0.0,
            "confirm_body_ratio": (
                abs(close[c] - opn[c]) / (high[c] - low[c]) if high[c] > low[c] else 0.0
            ),
            "confirm_close_location": (
                (close[c] - low[c]) / (high[c] - low[c]) if high[c] > low[c] else 0.5
            ),
            "confirm_vol_zscore": _vol_zscore(vol, c, has_vol),
            "bars_confirm_delay": float(c - d),
            # --- context features (detect timestamp) ----------------------
            "atr_ratio_200": _atr_ratio(atr, d),
            "hour_utc": float(df.index[d].hour),
            "day_of_week": float(df.index[d].dayofweek),
            "session_asia": 0.0,
            "session_london": 0.0,
            "session_new_york": 0.0,
            "rule_score": float(ev.rule_score),
        }
        asia, london, ny = _session_flags(df.index[d].hour)
        row["session_asia"] = float(asia)
        row["session_london"] = float(london)
        row["session_new_york"] = float(ny)
        for name in extra:
            if name in a:
                row[name] = _f(a[name])
        rows.append(row)
        kept_ids.append(str(ev.event_id))

    feature_names = list(_DETECT_FEATURES) + list(_CONFIRM_FEATURES) + list(
        _CONTEXT_FEATURES
    ) + list(extra)
    if not rows:
        return pd.DataFrame(columns=feature_names), feature_names
    X = pd.DataFrame(rows, index=kept_ids)
    # enforce the feature-column order even when extra attrs are missing
    X = X.reindex(columns=feature_names)
    return X.fillna(NA_FILL).astype(float), feature_names


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _vol_zscore(vol: np.ndarray[Any, Any], c: int, has_vol: bool) -> float:
    """Causal volume z-score at bar ``c`` (window strictly before ``c``)."""
    if not has_vol or c < 5:
        return 0.0
    win = vol[max(0, c - 24) : c]
    if len(win) < 5:
        return 0.0
    mu = float(np.mean(win))
    sd = float(np.std(win))
    if sd <= 0.0:
        return 0.0
    return float((vol[c] - mu) / sd)


def _atr_ratio(atr: np.ndarray[Any, Any], d: int) -> float:
    """ATR(d) / mean(ATR over the trailing 200 bars) — volatility regime."""
    if d < 15 or np.isnan(atr[d]) or atr[d] <= 0.0:
        return 1.0
    win = atr[max(0, d - 200) : d + 1]
    win = win[~np.isnan(win)]
    if len(win) < 15:
        return 1.0
    mu = float(np.mean(win))
    return float(atr[d] / mu) if mu > 0.0 else 1.0


# ---------------------------------------------------------------------------
# TrainFrame + chronological splits (purge + embargo)
# ---------------------------------------------------------------------------


@dataclass
class TrainFrame:
    """Feature/label matrix for one set of events, ordered by entry bar."""

    X: pd.DataFrame
    y: pd.Series[Any]
    entry_bar: pd.Series[Any]
    horizon_bars: int
    pnl_r: pd.Series[Any] | None = None

    def __len__(self) -> int:
        return len(self.X)

    def ordered(self) -> TrainFrame:
        idx = self.entry_bar.sort_values().index
        return TrainFrame(
            X=self.X.loc[idx],
            y=self.y.loc[idx],
            entry_bar=self.entry_bar.loc[idx],
            horizon_bars=self.horizon_bars,
            pnl_r=self.pnl_r.loc[idx] if self.pnl_r is not None else None,
        )


def sequential_split(
    frame: TrainFrame,
    oos_frac: float = 0.40,
    purge_bars: int = 96,
    embargo_bars: int = 24,
) -> tuple[TrainFrame, TrainFrame]:
    """Chronological train/OOS split with purge + embargo (§6.3 / §3.5).

    OOS = the last ``oos_frac`` events (by entry bar).  The train set ends
    ``purge_bars + embargo_bars`` before the OOS boundary so no train
    forward window [entry+1, entry+H] overlaps an OOS event's entry
    (the same split semantics as ``pattern.dataset.split_train_oos``).
    """
    f = frame.ordered()
    n = len(f)
    if n == 0:
        return frame, frame
    oos_n = max(1, round(n * float(oos_frac)))
    boundary = int(f.entry_bar.iloc[n - oos_n])
    t_keep = (
        f.entry_bar + int(f.horizon_bars) + int(purge_bars) + int(embargo_bars)
        <= boundary
    )
    train = TrainFrame(
        X=f.X.loc[t_keep.to_numpy(dtype=bool)],
        y=f.y.loc[t_keep.to_numpy(dtype=bool)],
        entry_bar=f.entry_bar.loc[t_keep.to_numpy(dtype=bool)],
        horizon_bars=f.horizon_bars,
        pnl_r=f.pnl_r.loc[t_keep.to_numpy(dtype=bool)] if f.pnl_r is not None else None,
    )
    oos = TrainFrame(
        X=f.X.iloc[n - oos_n :],
        y=f.y.iloc[n - oos_n :],
        entry_bar=f.entry_bar.iloc[n - oos_n :],
        horizon_bars=f.horizon_bars,
        pnl_r=f.pnl_r.iloc[n - oos_n :] if f.pnl_r is not None else None,
    )
    return train, oos


def expanding_wf_splits(
    frame: TrainFrame,
    n_folds: int = 5,
    purge_bars: int = 96,
    embargo_bars: int = 24,
) -> list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Expanding-window walk-forward index pairs [(train_idx, eval_idx)].

    Fold ``f`` evaluates the next ``n // n_folds`` events; its train set is
    every earlier event whose forward window (purge + embargo included)
    ends before the fold boundary.  Train/eval are disjoint by construction.
    """
    f = frame.ordered()
    n = len(f)
    if n == 0 or n_folds < 1:
        return []
    fold_size = max(1, n // int(n_folds))
    splits: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    for k in range(int(n_folds)):
        f_start = k * fold_size
        f_end = min(n, f_start + fold_size)
        if f_end <= f_start:
            break
        boundary = int(f.entry_bar.iloc[f_start])
        train_keep = (
            f.entry_bar.iloc[:f_start]
            + int(f.horizon_bars)
            + int(purge_bars)
            + int(embargo_bars)
            <= boundary
        )
        tr_idx = f.X.index[:f_start][train_keep.to_numpy(dtype=bool)]
        ev_idx = f.X.index[f_start:f_end]
        if len(tr_idx) >= 10:
            splits.append(
                (np.asarray(tr_idx, dtype=object), np.asarray(ev_idx, dtype=object))
            )
    return splits


# ---------------------------------------------------------------------------
# Models + metrics
# ---------------------------------------------------------------------------


def make_default_model(seed: int = 42) -> CalibratedClassifierCV:
    """Tier-2 meta-model: RandomForest + isotonic calibration (cv inside the
    train split only — §6.4).  The estimator pipeline scales internally so
    raw (NaN-preprocessed) feature vectors can be scored directly."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
    return CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=3)


def binary_metrics(y_true: pd.Series[Any], y_score: pd.Series[Any]) -> dict[str, float]:
    """Standard OOS metrics; single-class guards return neutral values."""
    y = y_true.to_numpy(dtype=int)
    s = y_score.to_numpy(dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pr_auc = float(average_precision_score(y, s)) if n_pos > 0 and n_neg > 0 else 0.5
    roc = float(roc_auc_score(y, s)) if n_pos > 0 and n_neg > 0 else 0.5
    brier = float(brier_score_loss(y, s)) if len(y) > 0 else float("nan")
    acc = float(accuracy_score(y, (s >= 0.5).astype(int))) if len(y) > 0 else 0.0
    return {
        "n": float(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) > 0 else 0.0,
        "pr_auc": pr_auc,
        "roc_auc": roc,
        "brier": brier,
        "accuracy": acc,
    }


def _profit_factor(pnl: pd.Series[Any]) -> float:
    p = pnl.to_numpy(dtype=float)
    pos = float(p[p > 0].sum())
    neg = abs(float(p[p < 0].sum()))
    if neg <= 0.0:
        return float("inf") if pos > 0 else 0.0
    return pos / neg


@dataclass
class FoldResult:
    """Metrics for one walk-forward fold (model fit on train, scored once)."""

    fold: int
    n_train: int
    n_eval: int
    metrics: dict[str, float]
    rule_pf: float = float("nan")

    def _as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"fold": float(self.fold), "n_train": float(self.n_train), "n_eval": float(self.n_eval)}
        out.update(self.metrics)
        out["rule_pf"] = self.rule_pf
        return out


@dataclass
class WalkForwardReport:
    """Aggregate of all walk-forward folds (no-peek OOS)."""

    folds: list[FoldResult] = field(default_factory=list)

    def aggregate(self, metric: str = "pr_auc") -> dict[str, float]:
        vals = [f.metrics[metric] for f in self.folds]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    def to_rows(self) -> list[dict[str, float]]:
        return [f._as_dict() for f in self.folds]


def run_walk_forward(
    frame: TrainFrame,
    n_folds: int = 5,
    purge_bars: int = 96,
    embargo_bars: int = 24,
    seed: int = 42,
) -> WalkForwardReport:
    """Expanding-window walk-forward validation (purge + embargo enforced).

    Each fold fits a fresh model **on the train split only** and scores the
    fold once.  Returns per-fold + aggregate OOS metrics.
    """
    splits = expanding_wf_splits(frame, n_folds, purge_bars, embargo_bars)
    ordered = frame.ordered()
    report = WalkForwardReport()
    for k, (tr_idx, ev_idx) in enumerate(splits):
        model = make_default_model(seed)
        Xtr = ordered.X.loc[tr_idx]
        ytr = ordered.y.loc[tr_idx]
        model.fit(Xtr.to_numpy(dtype=float), ytr.to_numpy(dtype=int))
        Xev = ordered.X.loc[ev_idx]
        ye = ordered.y.loc[ev_idx]
        prob = pd.Series(model.predict_proba(Xev.to_numpy(dtype=float))[:, 1], index=ev_idx)
        metrics = binary_metrics(ye, prob)
        rule_pf = float("nan")
        if ordered.pnl_r is not None:
            pf_pnl = ordered.pnl_r.loc[ev_idx]
            if len(pf_pnl) > 0:
                rule_pf = _profit_factor(pf_pnl)
        report.folds.append(
            FoldResult(fold=k, n_train=len(tr_idx), n_eval=len(ev_idx), metrics=metrics, rule_pf=rule_pf)
        )
    return report


def fit_final_model(frame: TrainFrame, seed: int = 42) -> CalibratedClassifierCV:
    """Fit the final tier-2 model on a train split (caller supplies a
    purge+embargo'd train frame — OOS is never passed here)."""
    model = make_default_model(seed)
    model.fit(frame.X.to_numpy(dtype=float), frame.y.to_numpy(dtype=int))
    return model


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@dataclass
class TrainSummary:
    """§5.5 registry-ready metadata for one trained model."""

    model_id: str
    pattern_name: str
    symbol: str
    timeframe: str
    feature_schema_version: str
    lifecycle_state: str
    gate_passed: bool
    calibrated: bool
    config_hash: str
    trained_at: str
    horizon_bars: int
    target_r: float
    n_events: int
    n_train: int
    n_oos: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSummary:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def write_model_artifacts(
    out_dir: Path,
    model: Any,
    feature_names: list[str],
    summary: TrainSummary,
) -> None:
    """Persist model.pkl / calibrator.pkl / features.json / train_summary.json.

    ``model.pkl`` uses the dict envelope the live engine
    (``live.engine.signal_engine_v2._load_symbol_model_and_calibrator``)
    expects: ``{"model": ..., "scaler": ..., "feature_names": ...,
    "impute_medians": ...}``; ``calibrator.pkl`` stores the same calibrated
    classifier as the engine fallback.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore[import-untyped]

    envelope = {
        "model": model,
        "scaler": None,  # scaling lives inside the pipeline (StandardScaler)
        "feature_names": list(feature_names),
        "impute_medians": {},
    }
    joblib.dump(envelope, out_dir / "model.pkl")
    joblib.dump(model, out_dir / "calibrator.pkl")

    features_json = {
        "model_id": summary.model_id,
        "pattern_name": summary.pattern_name,
        "symbol": summary.symbol,
        "timeframe": summary.timeframe,
        "feature_schema_version": summary.feature_schema_version,
        "config_hash": summary.config_hash,
        "horizon_bars": summary.horizon_bars,
        "target_r": summary.target_r,
        "n_features": len(feature_names),
        "features": list(feature_names),
        "preprocess": f"fillna({NA_FILL})",
        "trained_at": summary.trained_at,
    }
    (out_dir / "features.json").write_text(json.dumps(features_json, indent=2) + "\n")
    # §5.5-compatible alias (acceptance contract name) — same payload.
    (out_dir / "feature_schema.json").write_text(json.dumps(features_json, indent=2) + "\n")
    (out_dir / "train_summary.json").write_text(json.dumps(summary.to_dict(), indent=2) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def render_walk_forward_report(
    summary: TrainSummary,
    report: WalkForwardReport,
    oos_metrics: dict[str, float],
    gate_md: str = "",
) -> str:
    """Markdown training + walk-forward report (attached to the PR/CI)."""
    lines = [
        "# Walk-Forward Training Report (§6.2/§6.4)",
        "",
        f"- model_id: `{summary.model_id}` · pattern: {summary.pattern_name} · "
        f"symbol: {summary.symbol} · timeframe: {summary.timeframe}",
        f"- feature_schema_version: {summary.feature_schema_version} · "
        f"config_hash: `{summary.config_hash}`",
        f"- trained_at: {summary.trained_at} · lifecycle_state: "
        f"{summary.lifecycle_state} · gate_passed: {summary.gate_passed}",
        f"- label: forward MFE fixed {summary.horizon_bars}-bar horizon, "
        f"target {summary.target_r}R · calibrated: {summary.calibrated}",
        f"- events: {summary.n_events} · train: {summary.n_train} · "
        f"OOS: {summary.n_oos} (purge+embargo applied)",
        "",
        "## Held-out OOS metrics (final model fit on train only)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k, v in sorted(oos_metrics.items()):
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines += [
        "",
        "## Walk-forward folds (expanding window, purge+embargo)",
        "",
        "| fold | n_train | n_eval | PR-AUC | ROC-AUC | Brier | rule PF |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in report.folds:
        m = f.metrics
        lines.append(
            f"| {f.fold} | {f.n_train} | {f.n_eval} | {m['pr_auc']:.4f} | "
            f"{m['roc_auc']:.4f} | {m['brier']:.4f} | {f.rule_pf:.3f} |"
        )
    agg = report.aggregate("pr_auc")
    lines += [
        "",
        f"- PR-AUC across folds: mean {agg['mean']:.4f} ± {agg['std']:.4f} "
        f"(min {agg['min']:.4f}, max {agg['max']:.4f})",
    ]
    if gate_md:
        lines += ["", "## §6.3 research gates", "", gate_md.strip()]
    if summary.notes:
        lines += ["", "## Notes", "", summary.notes]
    return "\n".join(lines)