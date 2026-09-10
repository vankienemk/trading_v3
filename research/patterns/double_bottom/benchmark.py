"""
benchmark.py -- §8.3 Synthesizer acceptance harness for the double plugins.

Bridges a real ``BasePatternDetector`` plugin into the duck-typed
``PatternDetector`` protocol consumed by ``run_benchmark`` (spec §8.3):

    recall ≥ 80 % (≤2-bar position tolerance), robustness drop ≤ 15 pts
    when noise doubles, FP ≤ 5 % on pure-noise series.

The adapter maps each ``PatternEvent``'s confirm bar (the honest, causal
breakout bar) to the benchmark's ``breakout_bar``.  It never touches the
ground truth -- only OHLCV -- so the score is unbiased (spec §8.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.core.contracts import BasePatternDetector
from research.core.pattern_synthesizer import (
    BenchmarkResult,
    DetectedPattern,
    PatternSynthesizer,
    assert_acceptance,
    run_benchmark,
)
from research.patterns.double_bottom.detector import DoubleBottomDetector
from research.patterns.double_top.detector import DoubleTopDetector


class PluginDetectorAdapter:
    """Wrap a ``BasePatternDetector`` into the ``PatternDetector`` protocol."""

    def __init__(self, detector: BasePatternDetector) -> None:
        self.detector = detector

    def detect(self, df: pd.DataFrame, pattern_name: str) -> list[DetectedPattern]:
        events = self.detector.detect(df, self.detector.get_default_config())
        out: list[DetectedPattern] = []
        for ev in events:
            loc = df.index.get_loc(ev.confirm_time)
            bar = int(loc) if isinstance(loc, (int,)) else int(np.asarray(loc).item())
            out.append(
                DetectedPattern(
                    pattern_name=ev.pattern_name,
                    direction=ev.direction,
                    breakout_bar=int(bar),
                )
            )
        return out


def run_double_benchmark(
    pattern_name: str,
    n_series: int = 150,
    noise_sigma: float = 0.05,
    n_noise_series: int = 300,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the §8.3 acceptance benchmark for one double-pattern plugin."""
    if pattern_name == "double_bottom":
        detector: BasePatternDetector = DoubleBottomDetector()
    elif pattern_name == "double_top":
        detector = DoubleTopDetector()
    else:
        raise ValueError(f"run_double_benchmark supports double_bottom|double_top, got {pattern_name!r}")
    return run_benchmark(
        PluginDetectorAdapter(detector),
        [pattern_name],
        n_series=n_series,
        noise_sigma=noise_sigma,
        n_noise_series=n_noise_series,
        seed=seed,
    )


def render_benchmark_report(result: BenchmarkResult) -> str:
    """Markdown report of a benchmark run (attached to the PR as DoD)."""
    lines = [
        "# Double-Pattern Synthesizer Benchmark Report (§8.3)",
        "",
        f"- noise_sigma base: 0.05 · noise series: {result.n_noise_series}",
        f"- FP rate on pure-noise series: **{result.false_positive_rate:.3f}** "
        f"(gate ≤ 0.05)",
        "",
        "| pattern | n_series | recall | recall@2x-noise | robustness drop | gate (≤0.15) |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r.pattern_name} | {r.n_series} | {r.recall:.3f} | "
        f"{r.recall_at_2x_noise:.3f} | {r.robustness_drop:.3f} | "
        f"{'PASS' if r.robustness_drop <= 0.15 and r.recall >= 0.80 else 'FAIL'} |"
        for r in result.per_pattern
    )
    lines.append("")
    lines.append("Gates: recall ≥ 0.80, robustness drop ≤ 0.15, FP ≤ 0.05.")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Quick self-check when run directly.
    for pat in ("double_bottom", "double_top"):
        res = run_double_benchmark(pat, n_series=40, n_noise_series=40, seed=1)
        assert_acceptance(res)
        print(render_benchmark_report(res))
        print()

    # Also prove the synthesizer itself is still coherent with the plugins.
    syn = PatternSynthesizer()
    for pat in ("double_bottom", "double_top"):
        df, gt = syn.generate(pat, 120, noise_sigma=0.05, seed=3)
        det = DoubleBottomDetector() if pat == "double_bottom" else DoubleTopDetector()
        print(pat, "gt.breakout_bar", gt.breakout_bar,
              "detected", [d.breakout_bar for d in PluginDetectorAdapter(det).detect(df, pat)])