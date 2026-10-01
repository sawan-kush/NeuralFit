"""Baseline vs optimized comparison. 'worse' is reported whenever a metric moves the wrong way."""
from __future__ import annotations

from ..schemas import ComparisonEntry, Metrics


def _entry(baseline: float, optimized: float, higher_is_better: bool) -> ComparisonEntry:
    change = optimized - baseline
    pct = (change / baseline * 100.0) if baseline != 0 else None
    if change == 0:
        direction = "unchanged"
    elif (change > 0) == higher_is_better:
        direction = "improved"
    else:
        direction = "worse"
    return ComparisonEntry(
        baseline=baseline, optimized=optimized, absolute_change=change, percent_change=pct, direction=direction
    )


def compare(baseline: Metrics, optimized: Metrics) -> dict[str, ComparisonEntry]:
    return {
        "top1_accuracy_pct": _entry(baseline.top1_accuracy_pct, optimized.top1_accuracy_pct, True),
        "size_bytes": _entry(baseline.size_bytes, optimized.size_bytes, False),
        "latency_median_ms": _entry(baseline.latency.median_ms, optimized.latency.median_ms, False),
        "param_count": _entry(baseline.param_count, optimized.param_count, False),
    }
