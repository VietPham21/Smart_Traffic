"""Non-Reinforcement Learning Baseline Agents (Tier 1 Benchmarks).

Includes:
1. FixedTimeAgent: Traditional cyclic signal controller.
   Maintainer: Trương Hữu Quang (MSV: 2351170614)
2. WebsterAgent: Webster's analytical formula signal controller.
   Maintainer: Trương Hữu Quang (MSV: 2351170614)
3. MaxPressureAgent: Actuated Max-Pressure feedback controller.
   Maintainers: Cao Đức Đạo (MSV: 2351170581) & Trương Hữu Quang (MSV: 2351170614)
"""

from __future__ import annotations

from typing import Any
import numpy as np

from core.config import (
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PHASE_NS,
    PHASE_EW,
    ACTION_KEEP,
    ACTION_SWITCH,
    FIXED_TIME_GREEN,
    WEBSTER_LOST_TIME,
    MAX_PRESSURE_THRESHOLD,
    DELTA_T_GREEN,
    T_MAX,
)
from .base_agent import BaseAgent


class FixedTimeAgent(BaseAgent):
    """Tier 1 Baseline: Fixed-time cyclic signal controller.

    Switches phase periodically every `green_duration` seconds whenever the
    environment action mask permits switching.
    """

    def __init__(
        self,
        green_duration: float = FIXED_TIME_GREEN,
        cycle_green_time: float | None = None,
        name: str = "Fixed-Time",
    ) -> None:
        """Initialize Fixed-Time agent.

        Args:
            green_duration: Fixed duration of green signal per phase (seconds).
            cycle_green_time: Alias for green_duration (default 30.0s per spec).
            name: Agent identifier.
        """
        super().__init__(name=name)
        duration = float(cycle_green_time if cycle_green_time is not None else green_duration)
        self.green_duration: float = duration
        self.cycle_green_time: float = duration
        self._elapsed_in_phase: float = 0.0
        self._last_phase: int = PHASE_NS

    def reset(self) -> None:
        """Reset internal phase timer."""
        self._elapsed_in_phase = 0.0
        self._last_phase = PHASE_NS

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select action adhering strictly to info['action_mask'] and fixed green duration."""
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        phase_duration = float(info.get("phase_duration", float(obs[9]) * float(T_MAX)))

        # Ràng buộc bắt buộc từ action_mask
        if action_mask[ACTION_KEEP] == 0:
            return ACTION_SWITCH
        if action_mask[ACTION_SWITCH] == 0:
            return ACTION_KEEP

        # Chu kỳ cố định: giữ pha khi tau < cycle_green_time, đổi pha khi tau >= cycle_green_time
        if phase_duration >= self.cycle_green_time and action_mask[ACTION_SWITCH] == 1:
            return ACTION_SWITCH

        return ACTION_KEEP


class WebsterAgent(BaseAgent):
    """Tier 1 Baseline: Webster's analytical formula signal controller.

    Calculates optimal cycle length C_0 = (1.5 * L + 5) / (1 - Y) and proportional
    green splits based on traffic arrival ratios across the two axes.
    """

    def __init__(
        self,
        lost_time: float = WEBSTER_LOST_TIME,
        name: str = "Webster-Opt",
    ) -> None:
        """Initialize Webster agent.

        Args:
            lost_time: Total lost time per cycle L (seconds).
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.lost_time: float = lost_time
        self.green_splits: dict[int, float] = {
            PHASE_NS: FIXED_TIME_GREEN,
            PHASE_EW: FIXED_TIME_GREEN,
        }
        self._elapsed_in_phase: float = 0.0
        self._last_phase: int = PHASE_NS

    def reset(self) -> None:
        """Reset internal phase timer."""
        self._elapsed_in_phase = 0.0
        self._last_phase = PHASE_NS

    def update_splits(self, flow_ns: float, flow_ew: float, sat_flow: float = 1.0) -> None:
        """Recompute Webster optimal cycle and green splits."""
        y_ns = min(0.45, max(0.05, flow_ns / sat_flow))
        y_ew = min(0.45, max(0.05, flow_ew / sat_flow))
        y_total = y_ns + y_ew

        if y_total >= 0.95:
            y_total = 0.95

        cycle = (1.5 * self.lost_time + 5.0) / (1.0 - y_total)
        cycle = float(np.clip(cycle, 30.0, 120.0))
        effective_green = cycle - self.lost_time

        self.green_splits[PHASE_NS] = max(10.0, effective_green * (y_ns / y_total))
        self.green_splits[PHASE_EW] = max(10.0, effective_green * (y_ew / y_total))

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select action adhering to computed Webster green split and info['action_mask']."""
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        current_phase = int(info.get("phase", int(round(obs[8]))))
        phase_duration = float(info.get("phase_duration", float(obs[9]) * float(T_MAX)))

        if action_mask[ACTION_KEEP] == 0:
            return ACTION_SWITCH
        if action_mask[ACTION_SWITCH] == 0:
            return ACTION_KEEP

        target_green = self.green_splits.get(current_phase, FIXED_TIME_GREEN)
        if phase_duration >= target_green and action_mask[ACTION_SWITCH] == 1:
            return ACTION_SWITCH

        return ACTION_KEEP


class MaxPressureAgent(BaseAgent):
    """Tier 1 Heavy Baseline: Actuated Max-Pressure feedback controller.

    Calculates queue pressure:
    - P_NS = Q_North + Q_South
    - P_EW = Q_East + Q_West

    Switches phase if:
    1. Minimum green time is satisfied (action_mask[SWITCH] == 1).
    2. Pressure on red axis exceeds active green axis by threshold theta_pressure,
       OR active green axis queue is completely cleared while red has waiting vehicles.
    """

    def __init__(
        self,
        pressure_threshold: float = MAX_PRESSURE_THRESHOLD,
        theta_diff: float | None = None,
        name: str = "Max-Pressure",
    ) -> None:
        """Initialize Max-Pressure agent.

        Args:
            pressure_threshold: Threshold theta_diff for pressure-based phase switching (default: 3).
            theta_diff: Alias for pressure_threshold per specification Algorithm 1.
            name: Agent identifier.
        """
        super().__init__(name=name)
        thresh = float(theta_diff if theta_diff is not None else pressure_threshold)
        self.pressure_threshold: float = thresh
        self.theta_diff: float = thresh

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Decide control action based on dynamic pressure differential and info['action_mask']."""
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))

        # Ràng buộc bắt buộc từ mặt nạ an toàn
        if action_mask[ACTION_KEEP] == 0:
            return ACTION_SWITCH
        if action_mask[ACTION_SWITCH] == 0:
            return ACTION_KEEP

        # Đọc trực tiếp số lượng xe nguyên bản từ info["raw_queues"] (không sai số làm tròn)
        raw_queues = info.get("raw_queues", [0, 0, 0, 0])
        q_b, q_n, q_e, q_w = raw_queues[0], raw_queues[1], raw_queues[2], raw_queues[3]

        pressure_ns = float(q_b + q_n)
        pressure_ew = float(q_e + q_w)

        current_phase = int(info.get("phase", int(round(obs[8]))))
        if current_phase == PHASE_NS:
            p_active = pressure_ns
            p_red = pressure_ew
        else:
            p_active = pressure_ew
            p_red = pressure_ns

        # Thuật toán 1: (P_green = 0 and P_red > 0) hoặc (P_red - P_green >= theta_diff) với theta_diff = 3
        pressure_delta = p_red - p_active
        should_switch = (p_active == 0 and p_red > 0) or (pressure_delta >= self.theta_diff)

        if should_switch and action_mask[ACTION_SWITCH] == 1:
            return ACTION_SWITCH

        return ACTION_KEEP
