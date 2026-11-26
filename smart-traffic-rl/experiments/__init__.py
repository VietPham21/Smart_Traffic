"""Experiments package for SmartTraffic RL Studio."""

from __future__ import annotations

from typing import Any

__all__ = ["BenchmarkRunner", "run_full_benchmark"]


def __getattr__(name: str) -> Any:
    if name in ("BenchmarkRunner", "run_full_benchmark"):
        from . import benchmark
        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
