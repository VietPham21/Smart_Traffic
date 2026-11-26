"""Unified RL Training Pipeline & TensorBoard Logging package for SmartTraffic RL Studio."""

from __future__ import annotations

from typing import Any

__all__ = ["train_agent"]


def __getattr__(name: str) -> Any:
    if name == "train_agent":
        from .train import train_agent
        return train_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
