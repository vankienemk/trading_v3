"""Pipeline 2 — Build events, and its public CLIs.

Owned by Agent 0 (integration) but the detector/confirmation modules
themselves are owned by Agents 2-3; this module only orchestrates them.

Two console entry points (pyproject.toml):

- ``xauusd-audit``  → Pipeline 1 style data audit (see :func:`data_audit_cli`).
- ``xauusd-events`` → Pipeline 2 event build (requires sweep detector modules).
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import load_config, resolve_path, set_global_config
from ..data.loader import run_data_pipeline
from ..data.validator import build_data_quality_report, save_report


def data_audit_cli(argv: list[str] | None = None) -> int:
    """``xauusd-audit``: raw CSV → normalized Parquet + data-quality JSON.

    Mirrors Pipeline 1 (guide section 26.1).  Exits 0 on success, 1 on any
    validation/config error.
    """
    parser = argparse.ArgumentParser(
        prog="xauusd-audit",
        description="Load raw MT5 CSV, normalize, validate, save Parquet and a "
        "data-quality report.",
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
        "--report-path",
        default=None,
        help="Override the report output path (default: from config data section).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override or None)
    set_global_config(cfg)

    try:
        df = run_data_pipeline(cfg)
    except Exception as exc:  # CLI boundary: report and exit
        print(f"[xauusd-audit] ERROR: {exc}", file=sys.stderr)
        return 1

    volume_kind = cfg["data"].get("volume_kind", "tick")
    timezone = cfg["project"].get("timezone", "UTC")
    report = build_data_quality_report(df, volume_kind, timezone)
    report_path = args.report_path or resolve_path(
        cfg, "reports/data_quality/xauusd_m15_quality.json"
    )
    save_report(report, report_path)

    print(f"[xauusd-audit] OK: {len(df)} rows, {report['suspected_gap_count']} gaps")
    print(f"[xauusd-audit] parquet -> {resolve_path(cfg, cfg['data']['output_path'])}")
    print(f"[xauusd-audit] quality -> {report_path}")
    return 0


def build_events_cli(argv: list[str] | None = None) -> int:
    """``xauusd-events``: processed OHLCV → event + confirmation table (Pipeline 2).

    Runs the sweep detector (t4), confirmation (t9) and deduplication with
    every parameter read from the merged config — in particular
    ``sweep.group_rule`` (baseline ``"first"``), so the event flow the CLI
    produces is strictly causal by default (QA finding F1).  Writes the event
    table (event columns + schema §6 confirmation columns) to Parquet and
    prints a summary.  Exits 0 on success, 1 on any config/data/validation
    error.
    """
    parser = argparse.ArgumentParser(
        prog="xauusd-events",
        description="Build the liquidity-sweep event table from processed OHLCV.",
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
        "--output",
        default=None,
        help="Event table output path (default: data/processed/events.parquet).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, args.override or None)
        set_global_config(cfg)

        from ..events.confirmation import attach_confirmations
        from ..events.sweep_detector import build_sweep_events

        df = run_data_pipeline(cfg)
        events = build_sweep_events(df, config=cfg)
        confirmed = attach_confirmations(df, events, cfg)

        out_path = args.output or resolve_path(cfg, "data/processed/events.parquet")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        confirmed.to_parquet(out_path, index=False)
    except Exception as exc:  # CLI boundary: report and exit
        print(f"[xauusd-events] ERROR: {exc}", file=sys.stderr)
        return 1

    n_confirmed = int(confirmed["is_confirmed"].sum()) if not confirmed.empty else 0
    group_rule = cfg["sweep"].get("group_rule", "first")
    print(
        f"[xauusd-events] OK: {len(events)} events "
        f"({n_confirmed} confirmed), group_rule={group_rule}"
    )
    print(f"[xauusd-events] events -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(data_audit_cli())