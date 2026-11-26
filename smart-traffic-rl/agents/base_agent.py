"""Abstract Base Class locking Interface Contracts for all Traffic Light Agents.

Guarantees identical method signatures, type-safety, and action masking compliance
across all baseline and reinforcement learning implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseAgent(ABC):
    """Abstract Base Class for traffic light control agents.

    Every algorithm developed by the team (FixedTimeAgent, MaxPressureAgent,
    WebsterAgent, QLearningAgent, and DQNAgent) MUST inherit from this class
    and adhere strictly to this interface contract.
    """

    def __init__(self, name: str) -> None:
        """Initialize agent with an identifier name.

        Args:
            name: Human-readable identifier of the agent.
        """
        self.name: str = name

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select a valid action adhering strictly to info['action_mask'].

        Args:
            obs: Normalized 10-dimensional state vector from TrafficLightEnv:
                 [qB, qN, qE, qW, wB, wN, wE, wW, p_t, tau_t].
            info: Telemetry dict from env containing 'action_mask', 'raw_queues', etc.
            eval_mode: If True, disables exploration (e.g. epsilon = 0.0) for pure greedy evaluation.

        Returns:
            int: Discrete action in {0: KEEP, 1: SWITCH}.
        """
        raise NotImplementedError("Subclasses must implement select_action()")

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, Any],
        next_info: dict[str, Any],
        truncated: bool = False,
    ) -> dict[str, Any] | None:
        """Perform policy update or replay buffer insertion based on transition experience.

        Rule-based baseline agents (FixedTime, Webster, MaxPressure) do not learn,
        so this default stub returns None. Learning agents (Q-Learning, DQN) override
        this method to minimize Bellman error.

        Args:
            obs: Current state vector s_t.
            action: Action executed a_t.
            reward: Immediate scalar reward r_t.
            next_obs: Resulting state vector s_{t+1}.
            done: Termination/truncation flag.
            info: Telemetry dict at current step (containing action_mask, raw_queues, etc.).
            next_info: Telemetry dict at next step (containing next_action_mask, etc.).

        Returns:
            dict[str, Any] | None: Training telemetry metrics (e.g. {'loss': ...}) or None.
        """
        return None

    def reset(self) -> None:
        """Reset internal agent states at the start of a new episode."""
        pass

    def on_episode_end(self) -> None:
        """Optional lifecycle hook invoked at the end of an episode (e.g., for epsilon decay)."""
        pass

    def save(self, filepath: str) -> None:
        """Serialize agent weights / Q-table to disk.

        Args:
            filepath: Target file path.
        """
        pass

    def load(self, filepath: str) -> None:
        """Load agent weights / Q-table from disk.

        Args:
            filepath: Source file path.
        """
        pass

    def export_onnx(self, filepath: str) -> None:
        """Export Deep RL inference network to ONNX format (optional for Deep RL agents).

        Args:
            filepath: Target .onnx file path.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support ONNX export.")
