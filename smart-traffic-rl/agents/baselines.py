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
        name: str = "Fixed-Time",
    ) -> None:
        """Initialize Fixed-Time agent.

        Args:
            green_duration: Fixed duration of green signal per phase (seconds).
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.green_duration: float = green_duration
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
        """Select action adhering strictly to info['action_mask']."""
        # TODO [Trương Hữu Quang]: Hoàn thiện thuật toán Fixed-Time Signal Control
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        current_phase = int(info.get("phase", int(round(obs[8]))))

        if current_phase != self._last_phase:
            self._elapsed_in_phase = 0.0
            self._last_phase = current_phase
        else:
            self._elapsed_in_phase += float(DELTA_T_GREEN)

        # Ràng buộc ép buộc đổi pha từ mask
        if action_mask[ACTION_KEEP] == 0:
            self._elapsed_in_phase = 0.0
            return ACTION_SWITCH

        # Hết thời gian xanh chu kỳ và mask cho phép đổi pha
        if self._elapsed_in_phase >= self.green_duration and action_mask[ACTION_SWITCH] == 1:
            self._elapsed_in_phase = 0.0
            return ACTION_SWITCH

        # Mặc định giữ pha nếu được phép
        if action_mask[ACTION_KEEP] == 1:
            return ACTION_KEEP

        valid_actions = np.where(action_mask == 1)[0]
        return int(valid_actions[0]) if len(valid_actions) > 0 else ACTION_KEEP


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
        # TODO [Trương Hữu Quang]: Hoàn thiện thuật toán Webster Optimal Split
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        current_phase = int(info.get("phase", int(round(obs[8]))))

        if current_phase != self._last_phase:
            self._elapsed_in_phase = 0.0
            self._last_phase = current_phase
        else:
            self._elapsed_in_phase += float(DELTA_T_GREEN)

        if action_mask[ACTION_KEEP] == 0:
            self._elapsed_in_phase = 0.0
            return ACTION_SWITCH

        target_green = self.green_splits.get(current_phase, FIXED_TIME_GREEN)
        if self._elapsed_in_phase >= target_green and action_mask[ACTION_SWITCH] == 1:
            self._elapsed_in_phase = 0.0
            return ACTION_SWITCH

        if action_mask[ACTION_KEEP] == 1:
            return ACTION_KEEP

        valid_actions = np.where(action_mask == 1)[0]
        return int(valid_actions[0]) if len(valid_actions) > 0 else ACTION_KEEP


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
        name: str = "Max-Pressure",
    ) -> None:
        """Initialize Max-Pressure agent.

        Args:
            pressure_threshold: Threshold theta_pressure for pressure-based phase switching.
            name: Agent identifier.
        """
        super().__init__(name=name)
        self.pressure_threshold: float = pressure_threshold

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Decide control action based on dynamic pressure differential and info['action_mask']."""
        # TODO [Cao Đức Đạo] / [Trương Hữu Quang]: Hoàn thiện thuật toán Actuated Max-Pressure Control
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

        # Luật chuyển pha cảm biến áp lực cực đại
        pressure_delta = p_red - p_active
        should_switch = (pressure_delta > self.pressure_threshold) or (p_active == 0 and p_red > 0)

        if should_switch and action_mask[ACTION_SWITCH] == 1:
            return ACTION_SWITCH

        return ACTION_KEEP
