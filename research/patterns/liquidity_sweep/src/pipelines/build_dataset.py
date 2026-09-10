"""Pipeline 3 — Build labeled dataset CLI (guide section 26.3).

Owned by Agent 0 (integration).  Orchestrates:

    event table + confirmation
      → registry levels
      → event features (Agent 4, t10)
      → entry/stop/target + triple-barrier labels + MFE/MAE + costs (Agent 5, t6)
      → rule score (Agent 6, t13)
      → final labeled event dataset (Parquet)

The output is one row per valid event carrying identity + features + labels +
cost columns + rule score, ready for Pipeline 4 (train).
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config, resolve_path, set_global_config


def build_dataset_cli(argv: list[str] | None = None) -> int:
    """``xauusd-dataset``: events → labeled feature dataset (Pipeline 3).

    Reads the processed OHLCV Parquet and the event table written by
    ``xauusd-events``, attaches features, labels, costs and rule scores, and
    writes the merged dataset.  Exits 0 on success, 1 on any error.
    """
    parser = argparse.ArgumentParser(
        prog="xauusd-dataset",
        description="Build the labeled event dataset (features + labels + "
        "costs + rule score).",
    )
    parser.add_argument(
        "--config",
        default="baseline.yaml",
        help="Base config file under configs/ (default: baseline.yaml).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional config file(s) to deep-merge (repeatable).",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Event table Parquet (default: from config data.events_path).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Labeled dataset output path (default: artifacts/datasets/).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.override or None)
        set_global_config(cfg)

        from ..data.loader import read_parquet
        from ..events.confirmation import attach_confirmations
        from ..events.sweep_detector import build_sweep_events
        from ..features.feature_pipeline import build_event_features
        from ..labeling.outcome_builder import build_event_labels
        from ..liquidity.level_registry import build_liquidity_levels
        from ..scoring.rule_score import compute_rule_scores

        candles_path = resolve_path(cfg, cfg["data"]["output_path"])
        candles = read_parquet(candles_path)
        if "timestamp" in candles.columns:
            candles = candles.set_index("timestamp")

        events_path = args.events or resolve_path(
            cfg, cfg["data"].get("events_path", "data/processed/events.parquet")
        )
        if os.path.exists(events_path):
            events = read_parquet(events_path)
        else:
            print(f"[xauusd-dataset] events not found at {events_path}; rebuilding", file=sys.stderr)
            events = build_sweep_events(candles, config=cfg)
            events = attach_confirmations(candles, events, cfg)

        registry = build_liquidity_levels(candles, cfg)

        feats = build_event_features(candles, registry, events, cfg)
        labels = build_event_labels(candles, events, cfg)

        # Merge on event_id: features (registry columns) + labels (wide schema).
        merged = feats.merge(
            labels,
            on="event_id",
            how="inner",
            suffixes=("", "_label"),
        )
        # Duplicate columns under the _label suffix are identical by
        # construction (event_time/level_id); drop the suffix copies.
        dupe_cols = [c for c in merged.columns if c.endswith("_label")]
        if dupe_cols:
            merged = merged.drop(columns=dupe_cols)

        scored = compute_rule_scores(merged, cfg, registry)

        out_path = args.output or resolve_path(
            cfg, "artifacts/datasets/liquidity_sweep_events.parquet"
        )
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        scored.to_parquet(out_path, index=False)

        summary = dict(scored.attrs.get("labeling_summary", {}))
        print(
            f"[xauusd-dataset] OK: {len(scored)} labeled events "
            f"(invalid dropped: {summary.get('n_invalid', 'n/a')})"
        )
        print(f"[xauusd-dataset] columns: {len(scored.columns)}")
        print(f"[xauusd-dataset] dataset -> {out_path}")
        return 0
    except Exception as exc:  # CLI boundary: report and exit
        print(f"[xauusd-dataset] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(build_dataset_cli())