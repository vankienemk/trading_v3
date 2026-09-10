"""
benchmark.py -- §8.3 Synthesizer acceptance harness for the wedge plugins.

Bridges a real ``BasePatternDetector`` plugin into the duck-typed
``PatternDetector`` protocol consumed by ``run_benchmark`` (spec §8.3):

    recall ≥ 80 % (≤2-bar position tolerance), robustness drop ≤ 15 pts
    when noise doubles, FP ≤ 5 % on pure-noise series.

The adapter (reused from the double-pattern family) maps each
``PatternEvent``'s confirm bar -- the honest, causal breakout bar -- to the
benchmark's ``breakout_bar``.  It never touches the ground truth -- only
OHLCV -- so the score is unbiased (spec §8.1).
"""

from __future__ import annotations

from research.core.pattern_synthesizer import (
    BenchmarkResult,
    assert_acceptance,
    run_benchmark,
)
from research.patterns.double_bottom.benchmark import PluginDetectorAdapter
from research.patterns.falling_wedge.detector import FallingWedgeDetector
from research.patterns.rising_wedge.detector import RisingWedgeDetector


def run_wedge_benchmark(
    pattern_name: str,
    n_series: int = 150,
    noise_sigma: float = 0.05,
    n_noise_series: int = 300,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the §8.3 acceptance benchmark for one wedge plugin."""
    from research.core.contracts import BasePatternDetector

    detector: BasePatternDetector
    if pattern_name == "rising_wedge":
        detector = RisingWedgeDetector()
    elif pattern_name == "falling_wedge":
        detector = FallingWedgeDetector()
    else:
        raise ValueError(
            f"run_wedge_benchmark supports rising_wedge|falling_wedge, got {pattern_name!r}"
        )
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
        "# Wedge Synthesizer Benchmark Report (§8.3)",
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
    for pat in ("rising_wedge", "falling_wedge"):
        res = run_wedge_benchmark(pat, n_series=40, n_noise_series=40, seed=1)
        assert_acceptance(res)
        print(render_benchmark_report(res))
        print()