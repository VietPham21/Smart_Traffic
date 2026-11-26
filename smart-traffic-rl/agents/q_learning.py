"""Tabular Q-Learning Agent with StateDiscretizer and Action Masking (Tier 2 RL).

Maps continuous 10-dimensional state vector into 972 discrete states:
81 queue combinations x 2 starvation states x 2 phase states x 3 tau levels = 972.
Maintainer: Dương Văn Hưng (MSV: 2351170597)
"""

from __future__ import annotations

import os
import pickle
from typing import Any
import numpy as np

from core.config import (
    STATE_DIM,
    ACTION_DIM,
    ACTION_KEEP,
    ACTION_SWITCH,
    PHASE_NS,
    PHASE_EW,
    PHASE_APPROACHES,
    OPPOSITE_PHASE,
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    Q_MAX,
    W_MAX,
    T_MIN,
    T_MAX,
    STARVATION_THRESHOLD,
    QUEUE_BIN_LOW_MAX,
    QUEUE_BIN_MED_MAX,
    TOTAL_DISCRETE_STATES,
    QL_LEARNING_RATE,
    QL_EPSILON_START,
    QL_EPSILON_MIN,
    QL_EPSILON_DECAY,
    GAMMA,
)
from .base_agent import BaseAgent


class StateDiscretizer:
    """Discretizes continuous observation vector into structured discrete state representation.

    Discrete State Space Construction (Total: 972 states):
    1. Queues across 4 approaches (b_B, b_N, b_E, b_W):
       Each approach is partitioned into 3 bins:
       - Bin 0 (Low / Empty):      Q <= 3 vehicles (<= QUEUE_BIN_LOW_MAX)
       - Bin 1 (Medium / Normal):   4 <= Q <= 10 vehicles (QUEUE_BIN_MED_MAX)
       - Bin 2 (Heavy / Jammed):    Q > 10 vehicles
       Total combinations: 3^4 = 81 queue states.
    2. Starvation Alert Flag (F_starve):
       - 0: Normal (no red approach waiting >= STARVATION_THRESHOLD = 40s)
       - 1: Starvation alert (at least one red approach has lead vehicle wait >= 40s)
       Total: 2 states.
    3. Active Green Phase (p_t):
       - 0: North-South green
       - 1: East-West green
       Total: 2 states.
    4. Phase Duration Level (tau_level - Mục I.5.2):
       - Level 0: tau < T_MIN (10s)
       - Level 1: T_MIN <= tau < T_MAX (10s <= tau < 60s)
       - Level 2: tau >= T_MAX (60s)
       Total: 3 states.

    Total discrete state size: 81 * 2 * 2 * 3 = 972 distinct states.
    """

    def __init__(
        self,
        queue_bin_low_max: int = QUEUE_BIN_LOW_MAX,
        queue_bin_med_max: int = QUEUE_BIN_MED_MAX,
        starvation_threshold: float = STARVATION_THRESHOLD,
    ) -> None:
        """Initialize discretizer thresholds.

        Args:
            queue_bin_low_max: Max queue length for Low bin 0 (default: 3).
            queue_bin_med_max: Max queue length for Medium bin 1 (default: 10).
            starvation_threshold: Wait time threshold in seconds for starvation alert flag (default: 40s).
        """
        self.queue_bin_low_max: int = queue_bin_low_max
        self.queue_bin_med_max: int = queue_bin_med_max
        self.starvation_threshold: float = starvation_threshold

    def discretize_queue(self, q_count: float) -> int:
        """Partition queue length into 3 discrete bins: 0 (Low), 1 (Med), 2 (High)."""
        if q_count <= self.queue_bin_low_max:
            return 0
        elif q_count <= self.queue_bin_med_max:
            return 1
        return 2

    def discretize_tau(self, tau_seconds: float) -> int:
        """Partition green duration into 3 discrete levels (Mục I.5.2):
        - Level 0: tau < T_MIN (10s)
        - Level 1: T_MIN <= tau < T_MAX (10s <= tau < 60s)
        - Level 2: tau >= T_MAX (60s)
        """
        if tau_seconds < float(T_MIN):
            return 0
        elif tau_seconds < float(T_MAX):
            return 1
        return 2

    def check_starvation(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        """Return 1 if any lead vehicle on RED approaches has waited >= 40s, else 0."""
        phase = int(info.get("phase", int(round(obs[8]))))
        red_approaches = PHASE_APPROACHES[OPPOSITE_PHASE[phase]]
        max_waits = info.get("max_waits", None)

        if max_waits is not None:
            for app in red_approaches:
                if max_waits[app] >= self.starvation_threshold:
                    return 1
        else:
            for app in red_approaches:
                wait_sec = obs[4 + app] * W_MAX
                if wait_sec >= self.starvation_threshold:
                    return 1
        return 0

    def transform(self, obs: np.ndarray, info: dict[str, Any]) -> tuple[int, ...]:
        """Transform continuous 10-D observation into a discrete state tuple (Mục I.5.2).

        Args:
            obs: Normalized 10-dimensional observation vector.
            info: Telemetry dict from environment.

        Returns:
            tuple[int, ...]: (b_B, b_N, b_E, b_W, starve_flag, phase, tau_level)
        """
        raw_queues = info.get("raw_queues", None)
        if raw_queues is not None:
            b_b = self.discretize_queue(raw_queues[0])
            b_n = self.discretize_queue(raw_queues[1])
            b_e = self.discretize_queue(raw_queues[2])
            b_w = self.discretize_queue(raw_queues[3])
        else:
            b_b = self.discretize_queue(obs[0] * Q_MAX)
            b_n = self.discretize_queue(obs[1] * Q_MAX)
            b_e = self.discretize_queue(obs[2] * Q_MAX)
            b_w = self.discretize_queue(obs[3] * Q_MAX)

        starve_flag = self.check_starvation(obs, info)
        phase = int(info.get("phase", int(round(obs[8]))))
        phase_duration = float(info.get("phase_duration", obs[9] * float(T_MAX)))
        tau_level = self.discretize_tau(phase_duration)

        return (b_b, b_n, b_e, b_w, starve_flag, phase, tau_level)

    def to_index(self, state_tuple: tuple[int, ...]) -> int:
        """Map discrete state tuple into a flat integer in [0, 971] for Q-table indexing."""
        b_b, b_n, b_e, b_w, starve_flag, phase, tau_level = state_tuple
        queue_idx = b_b * 27 + b_n * 9 + b_e * 3 + b_w
        flat_idx = ((queue_idx * 2 + starve_flag) * 2 + phase) * 3 + tau_level
        return int(flat_idx)


class QLearningAgent(BaseAgent):
    """Tier 2 Benchmark: Tabular Q-Learning Agent with Action Masking."""

    def __init__(
        self,
        lr: float = QL_LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_start: float = QL_EPSILON_START,
        epsilon_min: float = QL_EPSILON_MIN,
        epsilon_decay: float = QL_EPSILON_DECAY,
        decay_on_episode_end: bool = True,
        name: str = "Tabular-QLearning",
    ) -> None:
        """Initialize Q-Learning agent.

        Args:
            lr: Learning rate alpha (default: 0.1).
            gamma: Discount factor gamma (default: 0.95).
            epsilon_start: Initial exploration rate (default: 1.0).
            epsilon_min: Floor exploration rate (default: 0.02).
            epsilon_decay: Multiplicative decay rate (default: 0.992).
            decay_on_episode_end: If True, decays epsilon at episode end via on_episode_end().
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.lr: float = lr
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.decay_on_episode_end: bool = decay_on_episode_end

        self.discretizer: StateDiscretizer = StateDiscretizer()
        self.q_table: np.ndarray = np.zeros(
            (TOTAL_DISCRETE_STATES, ACTION_DIM),
            dtype=np.float32,
        )

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select a legal control action using epsilon-greedy policy with Action Masking & random tie-breaking."""
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return ACTION_KEEP

        state_tuple = self.discretizer.transform(obs, info)
        s_idx = self.discretizer.to_index(state_tuple)

        # In eval_mode, always act purely greedily
        current_eps = 0.0 if eval_mode else self.epsilon

        # Exploration
        if np.random.rand() < current_eps:
            return int(np.random.choice(valid_actions))

        # Exploitation among valid actions with random tie-breaking to eliminate zero-table KEEP bias
        valid_q = self.q_table[s_idx, valid_actions]
        max_q = float(np.max(valid_q))
        best_candidates = valid_actions[np.isclose(valid_q, max_q, atol=1e-9, rtol=0.0)]
        return int(np.random.choice(best_candidates))

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
        """Perform Bellman tabular update using masked future return and proper truncation bootstrapping."""
        s_tuple = self.discretizer.transform(obs, info)
        s_idx = self.discretizer.to_index(s_tuple)

        s_next_tuple = self.discretizer.transform(next_obs, next_info)
        s_next_idx = self.discretizer.to_index(s_next_tuple)

        next_mask = next_info.get("action_mask", np.array([1, 1], dtype=np.int8))
        valid_next = np.where(next_mask == 1)[0]

        is_truncated = bool(truncated or next_info.get("is_truncated", False))
        is_terminal = bool(done and not is_truncated)

        if is_terminal or len(valid_next) == 0:
            max_future_q = 0.0
        else:
            max_future_q = float(np.max(self.q_table[s_next_idx, valid_next]))

        td_target = reward + self.gamma * max_future_q
        td_error = td_target - self.q_table[s_idx, action]

        # Update rule: Q(s, a) <- Q(s, a) + alpha * TD_Error
        self.q_table[s_idx, action] += self.lr * td_error

        if not self.decay_on_episode_end:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        td_loss = float(td_error ** 2)
        q_val = float(self.q_table[s_idx, action])
        return {
            "loss": td_loss,
            "td_loss": td_loss,
            "td_error": float(td_error),
            "q_val": q_val,
            "mean_q": q_val,
            "epsilon": float(self.epsilon),
        }

    def on_episode_end(self) -> None:
        """Decay exploration rate epsilon at the end of an episode."""
        if self.decay_on_episode_end:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table to pickle (.pkl / .pth) or numpy (.npy) file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if filepath.lower().endswith(".npy"):
            with open(filepath, "wb") as f:
                np.save(f, self.q_table)
        else:
            with open(filepath, "wb") as f:
                pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, filepath: str) -> None:
        """Load Q-table from pickle or numpy file."""
        if not os.path.exists(filepath):
            return
        if filepath.lower().endswith(".npy"):
            with open(filepath, "rb") as f:
                self.q_table = np.load(f, allow_pickle=False)
        else:
            try:
                with open(filepath, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "q_table" in payload:
                    self.q_table = np.asarray(payload["q_table"], dtype=np.float32)
                    self.epsilon = float(payload.get("epsilon", self.epsilon))
                else:
                    self.q_table = np.asarray(payload, dtype=np.float32)
            except Exception:
                with open(filepath, "rb") as f:
                    self.q_table = np.load(f, allow_pickle=False)
