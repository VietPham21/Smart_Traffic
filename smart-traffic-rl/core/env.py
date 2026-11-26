"""Gymnasium-compliant Traffic Signal Control Environment (MDP v2.0).

Môi trường mô phỏng điều khiển tín hiệu đèn giao thông ngã tư đơn 4 hướng tuân thủ chuẩn Gymnasium API.
Tích hợp:
- Không gian trạng thái 10 chiều chuẩn hóa trong [0.0, 1.0].
- Mặt nạ hành động an toàn (Action Masking) chống nhấp nháy đèn và chống đói đèn.
- Phân rã tường minh pha con 3s Đèn Vàng và 5s Đèn Xanh khi chuyển pha.
- Hàm phần thưởng đa mục tiêu chuẩn hóa theo thời gian bước (chống Reward Hacking Semi-MDP).
- Bộ sinh dòng xe Poisson theo 1-second buckets cho kết quả chờ thực tế chính xác.

Tác giả / Người phụ trách: Phạm Văn Việt (MSV: 2351170633)
"""

from __future__ import annotations

from collections import deque
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import (
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PHASE_NS,
    PHASE_EW,
    PHASE_APPROACHES,
    OPPOSITE_PHASE,
    ACTION_KEEP,
    ACTION_SWITCH,
    STATE_DIM,
    ACTION_DIM,
    Q_MAX,
    W_MAX,
    T_MIN,
    T_MAX,
    DELTA_T_GREEN,
    DELTA_T_YELLOW,
    SATURATION_FLOW_RATE,
    EPISODE_DURATION,
    MAX_STEPS,
    ALPHA_QUEUE,
    BETA_WAIT_SQ,
    C_SWITCH,
    ETA_PASSED,
    DELTA_V_MAX,
    STARTUP_DELAY,
)
from .traffic_generator import TrafficGenerator, Vehicle


class TrafficLightEnv(gym.Env):
    """Môi trường mô phỏng điều khiển đèn tín hiệu ngã tư 4 hướng chuẩn Gymnasium.

    1. Không gian quan sát (Observation Space):
       spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
       Vector 10 chiều chuẩn hóa: s_t = [q_B, q_N, q_Đ, q_T, w_B, w_N, w_Đ, w_T, p_t, tau_t]

    2. Không gian hành động (Action Space):
       spaces.Discrete(2)
       - 0: Giữ nguyên pha (KEEP) thêm 5 giây đèn xanh.
       - 1: Chuyển pha (SWITCH) - đệm 3 giây đèn vàng (0 xả xe), sau đó bật 5 giây xanh cho hướng mới.

    3. Mặt nạ hành động an toàn (Action Masking):
       - tau < T_MIN (10s): mask = [1, 0] (Khóa đổi pha, bảo đảm an toàn).
       - tau >= T_MAX (60s) và hướng đỏ có xe: mask = [0, 1] (Bắt buộc đổi pha, chống đói đèn).
       - Ngược lại: mask = [1, 1] (Tác tử toàn quyền quyết định).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenario_id: str = "SC1_LOW",
        seed: int | None = None,
        traffic_generator: TrafficGenerator | None = None,
    ) -> None:
        """Khởi tạo môi trường điều khiển tín hiệu đèn giao thông."""
        super().__init__()

        self.scenario_id: str = scenario_id
        self._initial_seed: int | None = seed

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        if traffic_generator is not None:
            self.traffic_generator: TrafficGenerator = traffic_generator
        else:
            self.traffic_generator = TrafficGenerator(scenario_id=scenario_id, seed=seed)

        self.phase: int = PHASE_NS
        self.phase_duration: float = 0.0
        self.current_time: float = 0.0
        self.step_count: int = 0

        # Hàng đợi FIFO O(1) bằng collections.deque cho 4 hướng
        self.queues: dict[int, deque[Vehicle]] = {app: deque() for app in APPROACHES}

        self.total_passed_vehicles: int = 0
        self.total_wait_time_accumulated: float = 0.0
        self.max_wait_recorded: float = 0.0

        self._passed_window: deque[tuple[float, int]] = deque()
        self.last_step_switched: bool = False
        self.light_states: dict[str, str] = self._compute_light_states(self.phase)

    @property
    def sim_time(self) -> float:
        """Thời gian mô phỏng hiện tại (alias chuẩn hóa cho current_time)."""
        return float(self.current_time)

    @staticmethod
    def _compute_light_states(phase: int) -> dict[str, str]:
        """Sinh bản đồ trạng thái màu đèn tường minh của 4 cột đèn."""
        if phase == PHASE_NS:
            return {"N": "GREEN", "S": "GREEN", "E": "RED", "W": "RED"}
        return {"N": "RED", "S": "RED", "E": "GREEN", "W": "GREEN"}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Khởi tạo lại toàn bộ ngã tư về trạng thái rỗng ban đầu (chuẩn Gymnasium API)."""
        super().reset(seed=seed)

        if seed is not None:
            self._initial_seed = seed
            self.traffic_generator.reset(seed=seed)
        else:
            self.traffic_generator.reset()

        if options and "scenario_id" in options:
            self.scenario_id = str(options["scenario_id"])
            self.traffic_generator.set_scenario(self.scenario_id)

        self.phase = PHASE_NS
        self.phase_duration = 0.0
        self.current_time = 0.0
        self.step_count = 0
        self.queues = {app: deque() for app in APPROACHES}
        self.total_passed_vehicles = 0
        self.total_wait_time_accumulated = 0.0
        self.max_wait_recorded = 0.0
        self._passed_window.clear()
        self.last_step_switched = False
        self.light_states = self._compute_light_states(self.phase)

        observation = self._compute_observation()
        info = self._build_info(passed_vehicles=0, is_truncated=False)

        return observation, info

    def get_action_mask(self) -> np.ndarray:
        """Tính toán Mặt nạ hành động an toàn (Action Mask) tại bước hiện tại."""
        mask = np.ones(ACTION_DIM, dtype=np.int8)

        if self.phase_duration < T_MIN:
            mask[ACTION_SWITCH] = 0
        elif self.phase_duration >= T_MAX:
            red_approaches = PHASE_APPROACHES[OPPOSITE_PHASE[self.phase]]
            red_waiting_count = sum(len(self.queues[app]) for app in red_approaches)
            if red_waiting_count > 0:
                mask[ACTION_KEEP] = 0

        return mask

    def _compute_observation(self) -> np.ndarray:
        """Xây dựng vector quan sát 10 chiều s_t đã chuẩn hóa về đoạn [0.0, 1.0]."""
        obs = np.zeros(STATE_DIM, dtype=np.float32)

        for i, app in enumerate(APPROACHES):
            q_len = len(self.queues[app])
            obs[i] = float(np.clip(q_len / Q_MAX, 0.0, 1.0))

        for i, app in enumerate(APPROACHES):
            if self.queues[app]:
                lead_vehicle = self.queues[app][0]
                wait = lead_vehicle.get_wait_time(self.current_time)
                obs[4 + i] = float(np.clip(wait / W_MAX, 0.0, 1.0))
            else:
                obs[4 + i] = 0.0

        obs[8] = float(self.phase)
        obs[9] = float(np.clip(self.phase_duration / float(T_MAX), 0.0, 1.0))

        return obs

    def _compute_reward(
        self,
        action_switched: bool,
        passed_vehicles: int,
        dt_step: float = float(DELTA_T_GREEN),
    ) -> float:
        """Tính toán hàm phần thưởng tức thời đa mục tiêu r_t chuẩn hóa Semi-MDP.

        Công thức toán học:
            r_t = -( (dt_step / DELTA_T_GREEN) * [alpha * sum(q_i) + beta * sum((w_i)^2)]
                     + c * 1(switch) )
                  + eta * (passed_vehicles / DELTA_V_MAX)

        Hệ số tỷ lệ thời gian (dt_step / DELTA_T_GREEN) đảm bảo chi phí giữ hàng đợi trên
        bước 8s (khi đổi pha) chịu đúng 1.6x chi phí của bước 5s, loại bỏ Reward Hacking.
        """
        time_scale = float(dt_step) / float(DELTA_T_GREEN)

        norm_queues = [min(1.0, len(self.queues[app]) / float(Q_MAX)) for app in APPROACHES]
        penalty_queue = ALPHA_QUEUE * float(sum(norm_queues))

        penalty_wait = 0.0
        for app in APPROACHES:
            if self.queues[app]:
                lead_wait = self.queues[app][0].get_wait_time(self.current_time)
                norm_wait = min(1.0, lead_wait / float(W_MAX))
                penalty_wait += (norm_wait ** 2)
        penalty_wait *= BETA_WAIT_SQ

        holding_cost = time_scale * (penalty_queue + penalty_wait)
        penalty_switch = C_SWITCH if action_switched else 0.0
        reward_passed = ETA_PASSED * (float(passed_vehicles) / float(DELTA_V_MAX))

        return float(-(holding_cost + penalty_switch) + reward_passed)

    def _build_info(
        self,
        passed_vehicles: int,
        is_truncated: bool = False,
    ) -> dict[str, Any]:
        """Đóng gói từ điển dữ liệu gốc (Telemetry dictionary) chuẩn Hợp đồng giao diện.

        Bao gồm cả xe đang kẹt trong hàng đợi khi tính avg_wait_time nhằm phản ánh chính xác
        độ trễ toàn cục của hệ thống và bịt lỗ hổng đánh giá.
        """
        raw_queues = [int(len(self.queues[app])) for app in APPROACHES]
        max_waits = [
            float(self.queues[app][0].get_wait_time(self.current_time) if self.queues[app] else 0.0)
            for app in APPROACHES
        ]

        active_wait_sum = sum(
            v.get_wait_time(self.current_time)
            for app in APPROACHES
            for v in self.queues[app]
        )
        total_seen = self.total_passed_vehicles + sum(raw_queues)
        avg_wait = float((self.total_wait_time_accumulated + active_wait_sum) / max(1, total_seen))
        avg_wait_passed_only = float(
            self.total_wait_time_accumulated / max(1, self.total_passed_vehicles)
        )

        current_max_wait = max([self.max_wait_recorded, *max_waits])
        passed_last_60s = int(sum(cnt for _, cnt in self._passed_window))
        rates = self.traffic_generator.get_arrival_rates(self.current_time)
        current_rates = {
            "N": float(rates.get(NORTH, 0.0)),
            "S": float(rates.get(SOUTH, 0.0)),
            "E": float(rates.get(EAST, 0.0)),
            "W": float(rates.get(WEST, 0.0)),
        }

        return {
            # 8 khóa bắt buộc tương thích ngược 100%
            "raw_queues": raw_queues,
            "max_waits": max_waits,
            "phase": int(self.phase),
            "phase_duration": float(self.phase_duration),
            "action_mask": self.get_action_mask(),
            "step_passed_vehicles": int(passed_vehicles),
            "total_passed": int(self.total_passed_vehicles),
            "avg_wait_time": float(avg_wait),
            # Các khóa bổ sung phục vụ Pygame HUD, TensorBoard & Truncation Bootstrap
            "avg_wait_passed_only": float(avg_wait_passed_only),
            "max_wait_recorded": float(current_max_wait),
            "sim_time": float(self.sim_time),
            "passed_last_60s": passed_last_60s,
            "last_step_switched": bool(self.last_step_switched),
            "light_states": dict(self.light_states),
            "current_rates": current_rates,
            "spawner_enabled": bool(self.traffic_generator.spawner_enabled),
            "is_truncated": bool(is_truncated),
        }

    def _enqueue_arrivals(self, arrivals: dict[int, list[Vehicle]]) -> None:
        """Nạp xe mới vào cuối hàng đợi FIFO theo thứ tự arrival_time tăng dần (tối đa Q_MAX)."""
        for app in APPROACHES:
            for veh in arrivals[app]:
                if len(self.queues[app]) < Q_MAX:
                    self.queues[app].append(veh)

    def _discharge_green_phase(
        self,
        t_green_start: float,
        discharge_cap: int,
        startup_delay: float,
    ) -> int:
        """Giải phóng xe ở các nhánh xanh theo thứ tự FIFO và tính thời gian chờ thực."""
        passed = 0
        t_eff_start = t_green_start + startup_delay
        green_approaches = PHASE_APPROACHES[self.phase]

        for app in green_approaches:
            num_discharge = min(discharge_cap, len(self.queues[app]))
            for k in range(1, num_discharge + 1):
                discharged_veh = self.queues[app].popleft()
                t_scheduled = t_eff_start + (float(k) / float(SATURATION_FLOW_RATE))
                t_depart = max(discharged_veh.arrival_time, t_scheduled)
                wait_t = max(0.0, t_depart - discharged_veh.arrival_time)

                if wait_t > self.max_wait_recorded:
                    self.max_wait_recorded = float(wait_t)
                self.total_wait_time_accumulated += wait_t
                passed += 1
                self.total_passed_vehicles += 1

        return passed

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Thực thi một bước mô phỏng (Decision Step) phân rã chính xác pha Vàng / Xanh."""
        action = int(action)
        mask = self.get_action_mask()
        applied_action = action
        if mask[action] == 0:
            applied_action = int(np.argmax(mask))

        switched = (applied_action == ACTION_SWITCH)
        self.last_step_switched = switched

        if switched:
            # Pha con 1: 3 giây Đèn Vàng (không xả xe)
            yellow_dur = float(DELTA_T_YELLOW)
            yellow_arrivals = self.traffic_generator.generate_arrivals(self.current_time, yellow_dur)
            self._enqueue_arrivals(yellow_arrivals)
            self.current_time += yellow_dur

            # Pha con 2: 5 giây Đèn Xanh cho hướng mới (có 1s độ trễ khởi hành)
            self.phase = OPPOSITE_PHASE[self.phase]
            green_dur = float(DELTA_T_GREEN)
            self.phase_duration = green_dur
            t_green_start = self.current_time

            green_arrivals = self.traffic_generator.generate_arrivals(t_green_start, green_dur)
            self._enqueue_arrivals(green_arrivals)

            green_effective = max(0.0, green_dur - STARTUP_DELAY)
            discharge_cap = int(np.floor(green_effective * SATURATION_FLOW_RATE))
            self.current_time += green_dur

            step_passed = self._discharge_green_phase(
                t_green_start=t_green_start,
                discharge_cap=discharge_cap,
                startup_delay=STARTUP_DELAY,
            )
            dt_step = yellow_dur + green_dur
        else:
            # Giữ nguyên pha: tiếp tục 5 giây Đèn Xanh (không có độ trễ khởi hành)
            t_green_start = self.current_time
            green_dur = float(DELTA_T_GREEN)
            self.phase_duration += green_dur

            green_arrivals = self.traffic_generator.generate_arrivals(t_green_start, green_dur)
            self._enqueue_arrivals(green_arrivals)

            discharge_cap = int(np.floor(green_dur * SATURATION_FLOW_RATE))
            self.current_time += green_dur

            step_passed = self._discharge_green_phase(
                t_green_start=t_green_start,
                discharge_cap=discharge_cap,
                startup_delay=0.0,
            )
            dt_step = green_dur

        self.step_count += 1
        self.light_states = self._compute_light_states(self.phase)

        for app in APPROACHES:
            if self.queues[app]:
                lead_w = self.queues[app][0].get_wait_time(self.current_time)
                if lead_w > self.max_wait_recorded:
                    self.max_wait_recorded = float(lead_w)

        if step_passed > 0:
            self._passed_window.append((float(self.current_time), int(step_passed)))
        while self._passed_window and (self.sim_time - self._passed_window[0][0] > 60.0):
            self._passed_window.popleft()

        reward = self._compute_reward(
            action_switched=switched,
            passed_vehicles=step_passed,
            dt_step=dt_step,
        )

        terminated = False
        truncated = (self.step_count >= MAX_STEPS) or (self.current_time >= float(EPISODE_DURATION))

        observation = self._compute_observation()
        info = self._build_info(passed_vehicles=step_passed, is_truncated=truncated)

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Kết xuất đồ họa khung hình. Được ủy quyền xử lý trực quan độc lập bởi ui/studio_app.py và demo_env.py."""
        pass

    def close(self) -> None:
        """Dọn dẹp và giải phóng tài nguyên môi trường khi kết thúc phiên mô phỏng."""
        pass
