"""Walk-forward trainer tests (t1 — DB/DT models + FW/HS explore machinery).

Covers, per REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §6.2 / §6.3 / §6.4 and
handoff §3 P0#1:

  1. strict causality of the feature builder — no bar past the event's
     confirm bar is ever read (adversarial corruption of later bars must
     leave the row untouched);
  2. chronological train/OOS split with purge + embargo (no forward-window
     overlap);
  3. walk-forward fold splits are disjoint and purge-gapped by construction;
  4. the tier-2 model actually learns a causal signal (synthetic);
  5. artifact round-trip: model.pkl envelope + calibrator.pkl +
     features.json + train_summary.json in the §5.5 schema, consumable by
     the live engine loader contract;
  6. an end-to-end DB smoke on the real XAUUSD M15 dataset (functions only —
     no repo writes).
"""

# RATIONALE (t5 low finding → resolved at t10): the causality suites in this
# file use the ``no_lookahead`` pytest marker (``CAUSALITY_MARK``) on the
# feature-builder + split test classes instead of inheriting
# ``tests/no_lookahead_base.NoLookaheadTestBase``.  These tests assert
# causality by CONSTRUCTING adversarial inputs (corrupting bars after the
# confirm bar, artificially shifted timestamps) rather than by the shared base
# fixture pattern — the marker (not the base class) is what the CI
# ``-m no_lookahead`` gate selects on, and the corruption-based assertions are
# stronger than the base's structural checks.  Deliberate, not an omission.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_classifier

from research.core.walkforward_trainer import (
    TrainFrame,
    TrainSummary,
    binary_metrics,
    build_feature_frame,
    expanding_wf_splits,
    fit_final_model,
    run_walk_forward,
    sequential_split,
    utc_now_iso,
    write_model_artifacts,
)

#: the causality gate marks the feature-builder + split tests only
#: (no bar past known_at is read; purge+embargo contract).
CAUSALITY_MARK = pytest.mark.no_lookahead


# ---------------------------------------------------------------------------
# Session-scoped real-data fixtures (DB detection is ~0.6 s, shared).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def xauusd_df() -> pd.DataFrame:
    from research.patterns.double_bottom.dataset import load_xauusd_m15

    return load_xauusd_m15()


@pytest.fixture(scope="module")
def db_events(xauusd_df: pd.DataFrame):
    from research.patterns.double_bottom.detector import DoubleBottomDetector

    det = DoubleBottomDetector()
    return det.detect(xauusd_df, det.get_default_config())


@pytest.fixture(scope="module")
def db_labels(xauusd_df: pd.DataFrame, db_events):
    from research.patterns.double_bottom.dataset import label_events

    return label_events(xauusd_df, db_events)


def _synthetic_frame(n: int = 300, seed: int = 7) -> TrainFrame:
    """Random 6-feature frame with a real causal signal in feature 0.

    y = 1 with prob ~0.75 when x0 > its median; ~0.30 otherwise — a signal
    any competent classifier should surface on held-out folds.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.where(
        X[:, 0] > np.median(X[:, 0]),
        (rng.random(n) < 0.75).astype(int),
        (rng.random(n) < 0.30).astype(int),
    )
    idx = pd.Index([f"E{i:05d}" for i in range(n)])
    entry = pd.Series(np.arange(0, n * 10, 10), index=idx, dtype="int64")
    pnl = pd.Series(rng.choice([2.0, -1.0, 0.3], size=n), index=idx, dtype="float64")
    return TrainFrame(
        X=pd.DataFrame(X, index=idx),
        y=pd.Series(y, index=idx, dtype="int64"),
        entry_bar=entry,
        horizon_bars=12,
        pnl_r=pnl,
    )


# ---------------------------------------------------------------------------
# 1. Feature-builder causality
# ---------------------------------------------------------------------------
class TestBuilderCausality:
    """Feature-builder causality (§3.4)."""

    pytestmark = CAUSALITY_MARK
    def test_rows_identical_when_later_bars_corrupted(
        self, xauusd_df: pd.DataFrame, db_events
    ) -> None:
        """Adversarial: corrupt every bar AFTER an event's confirm bar; its
        feature row must not change (no read past known_at, §3.4)."""
        X_full, _ = build_feature_frame(xauusd_df, db_events)
        corrupted = xauusd_df.copy()
        candidates = [
            e
            for e in db_events
            if 60 <= int(e.attributes["confirm_bar"]) <= len(xauusd_df) - 30
        ][:4]
        assert candidates, "expected sampleable events"
        for ev in candidates:
            c = int(ev.attributes["confirm_bar"])
            bad = corrupted.copy()
            bad.iloc[c + 1 :] = bad.iloc[c + 1 :] * 12345.0  # corrupt future bars
            X_one, _ = build_feature_frame(bad, [ev])
            pd.testing.assert_frame_equal(
                X_one, X_full.loc[[str(ev.event_id)]], check_exact=False
            )

    def test_corruption_does_change_other_events(
        self, xauusd_df: pd.DataFrame, db_events
    ) -> None:
        """Sanity: the corruption test above is not vacuous — an event whose
        confirm bar lies AFTER the corruption point must change."""
        evs = sorted(
            db_events, key=lambda e: int(e.attributes["confirm_bar"])
        )
        early, late = evs[0], evs[-1]
        c_early = int(early.attributes["confirm_bar"])
        bad = xauusd_df.copy()
        bad.iloc[c_early + 1 :] = bad.iloc[c_early + 1 :] * 12345.0
        X_full, _ = build_feature_frame(xauusd_df, [early, late])
        X_bad, _ = build_feature_frame(bad, [early, late])
        # early event (confirm before corruption) unchanged; late event changed
        pd.testing.assert_frame_equal(
            X_full.loc[[str(early.event_id)]],
            X_bad.loc[[str(early.event_id)]],
            check_exact=False,
        )
        assert not np.allclose(
            X_full.loc[str(late.event_id)].to_numpy(),
            X_bad.loc[str(late.event_id)].to_numpy(),
        )

    def test_feature_build_deterministic(
        self, xauusd_df: pd.DataFrame, db_events
    ) -> None:
        X1, names1 = build_feature_frame(xauusd_df, db_events)
        X2, names2 = build_feature_frame(xauusd_df, db_events)
        assert names1 == names2
        pd.testing.assert_frame_equal(X1, X2)
        assert X1.isna().sum().sum() == 0  # preprocessing contract fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Purge + embargo splits
# ---------------------------------------------------------------------------
class TestSplits:
    """Purge + embargo chronological splits."""

    pytestmark = CAUSALITY_MARK
    def test_sequential_split_purges_and_embargoes(self) -> None:
        frame = _synthetic_frame()
        train, oos = sequential_split(frame, purge_bars=96, embargo_bars=24)
        h = frame.horizon_bars
        assert (train.entry_bar + h + 96 + 24 <= oos.entry_bar.min()).all()

    def test_wf_splits_disjoint_and_purged(self) -> None:
        frame = _synthetic_frame(n=400)
        splits = expanding_wf_splits(frame, n_folds=5, purge_bars=20, embargo_bars=5)
        assert len(splits) >= 3
        ordered = frame.ordered()
        for tr_idx, ev_idx in splits:
            assert not set(tr_idx) & set(ev_idx)
            tr = ordered.entry_bar.loc[tr_idx]
            ev = ordered.entry_bar.loc[ev_idx]
            assert (tr + frame.horizon_bars + 20 + 5 <= ev.min()).all()

    def test_walk_forward_learns_causal_signal(self) -> None:
        frame = _synthetic_frame(n=400)
        report = run_walk_forward(frame, n_folds=5, seed=42)
        assert len(report.folds) >= 3
        agg = report.aggregate("roc_auc")
        assert agg["mean"] > 0.6, f"model failed to learn signal: {agg}"

    def test_binary_metrics_single_class_neutral(self) -> None:
        y = pd.Series([1, 1, 1], dtype="int64")
        s = pd.Series([0.9, 0.8, 0.7], dtype="float64")
        m = binary_metrics(y, s)
        assert m["pr_auc"] == 0.5 and m["roc_auc"] == 0.5


# ---------------------------------------------------------------------------
# 3. Artifacts (engine-consumable contract)
# ---------------------------------------------------------------------------
class TestArtifacts:
    def test_model_artifact_roundtrip(self, tmp_path: Path) -> None:
        frame = _synthetic_frame(n=240, seed=11)
        train, _ = sequential_split(frame, purge_bars=10, embargo_bars=4)
        model = fit_final_model(train, seed=42)
        assert is_classifier(model)
        assert getattr(model, "estimator", None) is not None  # engine contract
        feature_names = list(train.X.columns)
        summary = TrainSummary(
            model_id="double_bottom_xauusd_m15_v1",
            pattern_name="double_bottom",
            symbol="XAUUSD",
            timeframe="M15",
            feature_schema_version="double-v1.0",
            lifecycle_state="validated",
            gate_passed=True,
            calibrated=True,
            config_hash="bb7129921f24",
            trained_at=utc_now_iso(),
            horizon_bars=72,
            target_r=1.25,
            n_events=382,
            n_train=len(train),
            n_oos=153,
            metrics={"pr_auc": 0.55},
            artifact_paths={"model": "x"},
            notes="",
        )
        write_model_artifacts(tmp_path, model, feature_names, summary)

        env = joblib.load(str(tmp_path / "model.pkl"))
        assert set(env.keys()) == {
            "model",
            "scaler",
            "feature_names",
            "impute_medians",
        }
        calib = joblib.load(str(tmp_path / "calibrator.pkl"))
        X = train.X.to_numpy(dtype=float)
        p1 = env["model"].predict_proba(X)
        p2 = calib.predict_proba(X)
        assert p1.shape == (len(train), 2) and np.allclose(p1, p2)
        assert env["model"].calibrated_classifiers_[0].estimator.n_features_in_ == len(
            feature_names
        )
        assert env["feature_names"] == feature_names

        feats = __import__("json").loads((tmp_path / "features.json").read_text())
        assert feats["features"] == feature_names
        assert feats["n_features"] == len(feature_names)
        assert feats["config_hash"] == "bb7129921f24"
        # §5.5-compatible alias emitted alongside the engine contract
        schema = __import__("json").loads((tmp_path / "feature_schema.json").read_text())
        assert schema["features"] == feature_names

        summ = __import__("json").loads((tmp_path / "train_summary.json").read_text())
        for key in (
            "model_id",
            "pattern_name",
            "symbol",
            "timeframe",
            "feature_schema_version",
            "lifecycle_state",
            "gate_passed",
            "calibrated",
            "config_hash",
            "trained_at",
        ):
            assert key in summ, f"train_summary missing §5.5 field {key}"

    def test_train_summary_roundtrip(self) -> None:
        s = TrainSummary(
            model_id="m",
            pattern_name="double_top",
            symbol="XAUUSD",
            timeframe="M15",
            feature_schema_version="double-v1.0",
            lifecycle_state="validated",
            gate_passed=True,
            calibrated=True,
            config_hash="h",
            trained_at=utc_now_iso(),
            horizon_bars=72,
            target_r=1.25,
            n_events=1,
            n_train=1,
            n_oos=1,
        )
        restored = TrainSummary.from_dict(s.to_dict())
        assert restored.model_id == "m" and restored.gate_passed is True


# ---------------------------------------------------------------------------
# 4. Real-data DB smoke (functions only — writes to tmp_path)
# ---------------------------------------------------------------------------
class TestDbTrainSmoke:
    def test_db_pipeline_full_smoke(
        self,
        xauusd_df: pd.DataFrame,
        db_events,
        db_labels,
        tmp_path: Path,
    ) -> None:
        from research.core.walkforward_trainer import binary_metrics

        X, _names = build_feature_frame(xauusd_df, db_events)
        ids = [
            str(le.event_id)
            for le in db_labels
            if str(le.event_id) in X.index
        ]
        X = X.loc[ids]
        y = pd.Series(
            [int(le.label) for le in db_labels if str(le.event_id) in X.index],
            index=ids,
            dtype="int64",
        )
        entry = pd.Series(
            [int(le.entry_bar) for le in db_labels if str(le.event_id) in X.index],
            index=ids,
            dtype="int64",
        )
        frame = TrainFrame(X=X, y=y, entry_bar=entry, horizon_bars=72)
        train, oos = sequential_split(frame, oos_frac=0.40, purge_bars=96, embargo_bars=24)
        assert len(train) >= 100 and len(oos) >= 100

        report = run_walk_forward(frame, n_folds=5, seed=42)
        assert len(report.folds) >= 3
        final = fit_final_model(train, seed=42)
        prob = final.predict_proba(oos.X.to_numpy(dtype=float))[:, 1]
        metrics = binary_metrics(oos.y, pd.Series(prob, index=oos.y.index))
        baseline = float(oos.y.mean())
        # the tier-2 model must at least beat the OOS positive-rate baseline
        assert metrics["pr_auc"] > baseline, f"PR-AUC {metrics['pr_auc']:.3f} <= {baseline:.3f}"
        assert metrics["roc_auc"] > 0.5
        write_model_artifacts(
            tmp_path / "db_model", final, list(X.columns), TrainSummary(
                model_id="smoke",
                pattern_name="double_bottom",
                symbol="XAUUSD",
                timeframe="M15",
                feature_schema_version="double-v1.0",
                lifecycle_state="validated",
                gate_passed=True,
                calibrated=True,
                config_hash="bb7129921f24",
                trained_at=utc_now_iso(),
                horizon_bars=72,
                target_r=1.25,
                n_events=len(ids),
                n_train=len(train),
                n_oos=len(oos),
                metrics=metrics,
                artifact_paths={},
                notes="smoke",
            ),
        )
        assert (tmp_path / "db_model" / "model.pkl").exists()