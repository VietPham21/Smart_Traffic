"""Gymnasium-compliant Traffic Signal Control Environment (MDP v2.0).

Môi trường mô phỏng điều khiển tín hiệu đèn giao thông ngã tư đơn 4 hướng tuân thủ chuẩn Gymnasium API.
Tích hợp:
- Không gian trạng thái 10 chiều chuẩn hóa trong [0.0, 1.0].
- Mặt nạ hành động an toàn (Action Masking) chống nhấp nháy đèn và chống đói đèn.
- Động lực học xả xe với tốc độ dòng bão hòa và độ trễ khởi hành.
- Bộ sinh dòng xe ngẫu nhiên Poisson theo 4 kịch bản chuẩn hóa.
- Hàm phần thưởng đa mục tiêu (Multi-objective Reward) cân bằng giải phóng hàng đợi và thời gian chờ.

Tác giả / Người phụ trách: Phạm Văn Việt (MSV: 2351170633)
"""

from __future__ import annotations

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
    AXIS_NS,
    AXIS_EW,
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
       - q_i: Tỷ lệ lấp đầy hàng đợi từng nhánh (Q_i / Q_MAX)
       - w_i: Thời gian chờ chuẩn hóa của xe đầu hàng đợi từng nhánh (W_i / W_MAX)
       - p_t: Pha đèn xanh hiện tại (0.0: Bắc-Nam xanh, 1.0: Đông-Tây xanh)
       - tau_t: Thời gian duy trì pha hiện tại đã chuẩn hóa (tau / T_MAX)

    2. Không gian hành động (Action Space):
       spaces.Discrete(2)
       - 0: Giữ nguyên pha (KEEP) thêm 5 giây đèn xanh.
       - 1: Chuyển pha (SWITCH) - đệm 3 giây đèn vàng cho pha cũ, sau đó bật 5 giây xanh cho hướng mới.

    3. Mặt nạ hành động an toàn (Action Masking):
       - tau < T_MIN (10s): mask = [1, 0] (Khóa đổi pha, bảo đảm an toàn, chống nhấp nháy đèn).
       - tau >= T_MAX (60s) và hướng đỏ có xe: mask = [0, 1] (Bắt buộc đổi pha, chống đói đèn vô hạn).
       - Ngược lại: mask = [1, 1] (Tác tử toàn quyền quyết định).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        scenario_id: str = "SC1_LOW",
        seed: int | None = None,
        traffic_generator: TrafficGenerator | None = None,
    ) -> None:
        """Khởi tạo môi trường điều khiển tín hiệu đèn giao thông.

        Args:
            scenario_id: Mã kịch bản lưu lượng trong SCENARIOS ('SC1_LOW', 'SC2_HIGH', v.v.).
            seed: Hạt giống ngẫu nhiên khởi tạo để tái lập thực nghiệm.
            traffic_generator: Bộ sinh xe Poisson cấu hình sẵn (nếu có).
        """
        super().__init__()

        self.scenario_id: str = scenario_id
        self._initial_seed: int | None = seed

        # ======================================================================
        # 1. KHAI BÁO KHÔNG GIAN THEO CHUẨN GYMNASIUM API
        # ======================================================================
        # observation_space: Không gian trạng thái liên tục 10 chiều trong đoạn [0.0, 1.0]
        # Giúp mạng nơ-ron học sâu (DQN/DDQN) nhận vector đầu vào có độ dốc ổn định, không bão hòa gradient.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        # action_space: Không gian hành động rời rạc gồm 2 hành động: 0 (Giữ pha) hoặc 1 (Chuyển pha)
        self.action_space = spaces.Discrete(ACTION_DIM)

        # ======================================================================
        # 2. KHỞI TẠO BỘ SINH DÒNG XE POISSON (TrafficGenerator)
        # ======================================================================
        if traffic_generator is not None:
            self.traffic_generator: TrafficGenerator = traffic_generator
        else:
            self.traffic_generator = TrafficGenerator(scenario_id=scenario_id, seed=seed)

        # ======================================================================
        # 3. CÁC BIẾN TRẠNG THÁI NỘI TẠI CỦA NGÃ TƯ MÔ PHỎNG
        # ======================================================================
        # Pha đèn xanh hiện tại: PHASE_NS (0: Bắc-Nam) hoặc PHASE_EW (1: Đông-Tây)
        self.phase: int = PHASE_NS

        # tau_t: Số giây liên tục mà pha hiện tại đã bật đèn xanh (giây)
        self.phase_duration: float = 0.0

        # current_time: Tổng thời gian mô phỏng đã trôi qua kể từ khi reset (giây)
        self.current_time: float = 0.0

        # step_count: Đếm số bước quyết định (step) đã thực hiện trong tập hiện tại
        self.step_count: int = 0

        # queues: Danh sách các đối tượng Vehicle đang dừng chờ tại 4 nhánh (FIFO: xe đến trước đứng đầu)
        self.queues: dict[int, list[Vehicle]] = {app: [] for app in APPROACHES}

        # Các biến đo lường tích lũy (Telemetry & Metrics)
        self.total_passed_vehicles: int = 0         # Tổng số xe đã thoát thành công qua nút giao
        self.total_wait_time_accumulated: float = 0.0 # Tổng thời gian chờ tích lũy của tất cả các xe đã thoát (giây)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Khởi tạo lại toàn bộ ngã tư về trạng thái rỗng ban đầu (chuẩn Gymnasium API).

        Quy trình xử lý:
        1. Cập nhật seed ngẫu nhiên cho môi trường và đồng bộ seed sang TrafficGenerator.
        2. Nếu options có chứa 'scenario_id', chuyển đổi kịch bản lưu lượng ngay lập tức.
        3. Đặt lại toàn bộ biến mô phỏng: hàng đợi rỗng, đồng hồ = 0, pha = Bắc-Nam xanh.
        4. Trích xuất vector quan sát ban đầu s_0 và từ điển info ban đầu.

        Args:
            seed: Hạt giống ngẫu nhiên mới (nếu muốn thay đổi seed giữa các tập).
            options: Tùy chọn bổ sung lúc runtime (ví dụ: {'scenario_id': 'SC2_HIGH'}).

        Returns:
            Tuple gồm (observation_ban_đầu, info_dict_ban_đầu).
        """
        super().reset(seed=seed)

        # Đồng bộ hạt giống ngẫu nhiên nếu được truyền vào
        if seed is not None:
            self._initial_seed = seed
            self.traffic_generator.reset(seed=seed)

        # Thay đổi kịch bản lúc runtime nếu có chỉ định trong options
        if options and "scenario_id" in options:
            self.scenario_id = str(options["scenario_id"])
            self.traffic_generator.set_scenario(self.scenario_id)

        # Đặt lại các biến trạng thái về mặc định ban đầu
        self.phase = PHASE_NS
        self.phase_duration = 0.0
        self.current_time = 0.0
        self.step_count = 0
        self.queues = {app: [] for app in APPROACHES}
        self.total_passed_vehicles = 0
        self.total_wait_time_accumulated = 0.0

        # Tạo vector quan sát s_0 và từ điển info ban đầu
        observation = self._compute_observation()
        info = self._build_info(passed_vehicles=0)

        return observation, info

    def get_action_mask(self) -> np.ndarray:
        """Tính toán Mặt nạ hành động an toàn (Action Mask) tại bước hiện tại (Mục I.2.2).

        Cơ chế Mặt nạ an toàn A_valid(s_t):
        - Mặc định: mask = [1, 1] (Cả 2 hành động Giữ pha và Đổi pha đều hợp lệ).
        - Ràng buộc xanh tối thiểu: Nếu thời gian pha liên tục tau < T_MIN (10s),
          khóa hành động chuyển pha (mask[1] = 0 -> [1, 0]).
          -> Ý nghĩa thực tế: Ngăn đèn nhấp nháy đổi pha liên tục, bảo đảm an toàn cho tài xế.
        - Ràng buộc xanh tối đa: Nếu thời gian pha liên tục tau >= T_MAX (60s) VÀ
          ở hướng đang đèn đỏ có xe đang chờ, ép buộc chuyển pha (mask[0] = 0 -> [0, 1]).
          -> Ý nghĩa thực tế: Chống hiện tượng 'đói đèn xanh' (starvation) ở hướng phụ.

        Returns:
            Mảng nhị phân numpy shape (2,) kiểu np.int8: [mask_keep, mask_switch].
            Ví dụ: [1, 0] nghĩa là chỉ được giữ pha; [0, 1] là bắt buộc chuyển pha; [1, 1] là tự do.
        """
        mask = np.ones(ACTION_DIM, dtype=np.int8)

        # 1. Ràng buộc thời gian xanh tối thiểu: tau < T_MIN (10s) -> Cấm đổi pha
        if self.phase_duration < T_MIN:
            mask[ACTION_SWITCH] = 0

        # 2. Ràng buộc thời gian xanh tối đa: tau >= T_MAX (60s) và hướng đỏ có xe -> Bắt buộc đổi pha
        elif self.phase_duration >= T_MAX:
            red_approaches = PHASE_APPROACHES[OPPOSITE_PHASE[self.phase]]
            red_waiting_count = sum(len(self.queues[app]) for app in red_approaches)
            if red_waiting_count > 0:
                mask[ACTION_KEEP] = 0

        return mask

    def _compute_observation(self) -> np.ndarray:
        """Xây dựng vector quan sát 10 chiều s_t đã chuẩn hóa về đoạn [0.0, 1.0] (Mục I.2.1).

        Cấu trúc vector s_t:
        s_t = [q_B, q_N, q_Đ, q_T, w_B, w_N, w_Đ, w_T, p_t, tau_t]^T
        - q_i = min(Q_i / Q_MAX, 1.0): Tỷ lệ lấp đầy hàng đợi 4 nhánh Bắc, Nam, Đông, Tây (Q_MAX = 30 xe).
        - w_i = min(W_i / W_MAX, 1.0): Thời gian chờ chuẩn hóa của xe đầu hàng tại 4 nhánh (W_MAX = 120s).
          * Ghi chú: Việc đưa w_i vào trạng thái là bắt buộc để tác tử nhận diện các nhánh phụ có ít xe nhưng đã chờ quá lâu.
        - p_t: Pha đèn hiện tại (0.0: Bắc-Nam bật xanh; 1.0: Đông-Tây bật xanh).
        - tau_t = min(phase_duration / T_MAX, 1.0): Thời gian đã duy trì pha hiện tại (T_MAX = 60s).
          * Ghi chú: tau_t bảo toàn tính chất Markov khi áp dụng ràng buộc T_MIN và T_MAX.

        Returns:
            np.ndarray shape (10,) kiểu float32 với toàn bộ giá trị thuộc [0.0, 1.0].
        """
        obs = np.zeros(STATE_DIM, dtype=np.float32)

        # 1. Tỷ lệ lấp đầy hàng đợi 4 nhánh: q_B (idx 0), q_N (idx 1), q_Đ (idx 2), q_T (idx 3)
        for i, app in enumerate(APPROACHES):
            q_len = len(self.queues[app])
            obs[i] = float(np.clip(q_len / Q_MAX, 0.0, 1.0))

        # 2. Thời gian chờ chuẩn hóa của xe đầu hàng: w_B (idx 4), w_N (idx 5), w_Đ (idx 6), w_T (idx 7)
        for i, app in enumerate(APPROACHES):
            if len(self.queues[app]) > 0:
                lead_vehicle = self.queues[app][0]
                wait = lead_vehicle.get_wait_time(self.current_time)
                obs[4 + i] = float(np.clip(wait / W_MAX, 0.0, 1.0))
            else:
                obs[4 + i] = 0.0

        # 3. Pha đèn xanh hiện tại: p_t (idx 8: 0.0 là Bắc-Nam, 1.0 là Đông-Tây)
        obs[8] = float(self.phase)

        # 4. Thời gian duy trì pha hiện tại đã chuẩn hóa: tau_t (idx 9)
        obs[9] = float(np.clip(self.phase_duration / float(T_MAX), 0.0, 1.0))

        return obs

    def _compute_reward(self, action_switched: bool, passed_vehicles: int) -> float:
        """Tính toán hàm phần thưởng tức thời đa mục tiêu r_t theo công thức Mục I.2.4.

        Công thức toán học:
            r_t = - [ alpha * sum(q_i) + beta * sum((w_i)^2) + c * 1(switch) ]
                  + eta * (passed_vehicles / DELTA_V_MAX)

        Ý nghĩa các thành phần:
        1. Phạt độ dài hàng đợi (- alpha * sum(q_i), alpha = 1.0):
           q_i in [0, 1] là tỷ lệ lấp đầy hàng đợi. Thúc đẩy tác tử ưu tiên bật đèn xanh cho trục
           có tổng lượng xe ùn ứ lớn hơn để giảm nhanh áp lực nút giao.
        2. Phạt bình phương thời gian chờ (- beta * sum((w_i)^2), beta = 1.5):
           w_i in [0, 1] là thời gian chờ chuẩn hóa của xe đầu hàng. Đây là cơ chế phi tuyến cốt lõi
           giúp chống 'đói đèn xanh': khi xe mới chờ ít, (w_i)^2 rất nhỏ; nhưng khi xe chờ lâu (40-50s),
           mức phạt tăng vọt theo hàm bậc hai, ép tác tử buộc phải chuyển pha giải phóng nhánh phụ.
        3. Phạt chuyển pha (- c * 1(switch), c = 0.25):
           Chỉ phạt khi tác tử chọn chuyển pha (action_switched = True) để tính đến chi phí cơ hội
           của 3 giây đèn vàng hoàn toàn không xả được xe.
        4. Thưởng thông lượng xe thoát (+ eta * (passed_vehicles / DELTA_V_MAX), eta = 0.4, DELTA_V_MAX = 10):
           Thưởng chuẩn hóa dựa trên số lượng xe thực tế đã thoát khỏi ngã tư trong bước vừa qua,
           giúp tín hiệu phần thưởng định hướng hội tụ nhanh hơn.

        Args:
            action_switched: True nếu bước này thực hiện chuyển pha, False nếu giữ nguyên pha.
            passed_vehicles: Số lượng xe đã xả thành công qua ngã tư trong bước này.

        Returns:
            Giá trị phần thưởng thực r_t (float).
        """
        # Thành phần 1: Phạt độ dài hàng đợi chuẩn hóa q_i = min(Q_i / Q_MAX, 1.0)
        norm_queues = [min(1.0, len(self.queues[app]) / float(Q_MAX)) for app in APPROACHES]
        penalty_queue = ALPHA_QUEUE * float(sum(norm_queues))

        # Thành phần 2: Phạt bình phương thời gian chờ của xe đầu hàng đợi (w_i)^2
        penalty_wait = 0.0
        for app in APPROACHES:
            if len(self.queues[app]) > 0:
                lead_wait = self.queues[app][0].get_wait_time(self.current_time)
                norm_wait = min(1.0, lead_wait / float(W_MAX))
                penalty_wait += (norm_wait ** 2)
        penalty_wait *= BETA_WAIT_SQ

        # Thành phần 3: Phạt chi phí chuyển pha (chi phí 3 giây đệm vàng)
        penalty_switch = C_SWITCH if action_switched else 0.0

        # Thành phần 4: Thưởng thông lượng xe thoát đã chuẩn hóa trên DELTA_V_MAX = 10 xe
        reward_passed = ETA_PASSED * (float(passed_vehicles) / float(DELTA_V_MAX))

        # Tổng hợp giá trị reward
        return float(- penalty_queue - penalty_wait - penalty_switch + reward_passed)

    def _build_info(self, passed_vehicles: int) -> dict[str, Any]:
        """Đóng gói từ điển dữ liệu gốc (Telemetry dictionary) theo chuẩn Hợp đồng giao diện (Mục I.4.2).

        Chứa đầy đủ 8 trường dữ liệu thô (chưa chuẩn hóa) phục vụ giám sát đồ họa và thuật toán:
        - raw_queues: Số lượng xe thực tế đang chờ tại 4 nhánh [B, N, Đ, T].
        - max_waits: Số giây chờ thực tế của xe đợi lâu nhất tại 4 nhánh.
        - phase: Pha hiện tại (0: Bắc-Nam, 1: Đông-Tây).
        - phase_duration: Số giây liên tục pha hiện tại đã bật xanh.
        - action_mask: Mảng nhị phân [mask_keep, mask_switch].
        - step_passed_vehicles: Số xe thoát trong bước vừa qua.
        - total_passed: Tổng số xe đã thoát kể từ đầu tập.
        - avg_wait_time: Thời gian chờ trung bình của toàn bộ các xe đã thoát (giây/xe).

        Returns:
            dict chứa đúng 8 khóa bắt buộc.
        """
        raw_queues = [int(len(self.queues[app])) for app in APPROACHES]
        max_waits = [
            float(self.queues[app][0].get_wait_time(self.current_time) if self.queues[app] else 0.0)
            for app in APPROACHES
        ]
        avg_wait = float(
            self.total_wait_time_accumulated / max(1, self.total_passed_vehicles)
        )

        return {
            "raw_queues": raw_queues,
            "max_waits": max_waits,
            "phase": int(self.phase),
            "phase_duration": float(self.phase_duration),
            "action_mask": self.get_action_mask(),
            "step_passed_vehicles": int(passed_vehicles),
            "total_passed": int(self.total_passed_vehicles),
            "avg_wait_time": float(avg_wait),
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Thực thi một bước mô phỏng (Decision Step) theo chuẩn Gymnasium API (Mục I.2.2 & I.2.3).

        Quy trình xử lý 7 bước:
        1. Kiểm tra an toàn: Nếu tác tử chọn hành động bị cấm bởi action_mask,
           môi trường tự động cưỡng bức hành động hợp lệ (Safety Override).
        2. Cập nhật pha đèn và thời gian bước:
           - Giữ pha (action = 0): dt_step = 5s xanh; tau_{t+1} = tau_t + 5s;
             xả tối đa floor(5s * 1.0 xe/s) = 5 xe/làn.
           - Đổi pha (action = 1): Đệm 3s vàng cho hướng cũ (0 xe xả) + 5s xanh cho hướng mới;
             dt_step = 3s + 5s = 8s; đặt lại tau_{t+1} = 5s;
             trừ độ trễ khởi hành 1s -> xả tối đa floor((5s - 1s) * 1.0) = 4 xe/làn.
        3. Sinh dòng xe mới theo phân phối Poisson trong khoảng thời gian dt_step
           và nạp vào hàng đợi 4 nhánh (tối đa Q_MAX = 30 xe/nhánh).
        4. Xả xe từ các nhánh đang bật đèn xanh (theo nguyên tắc FIFO: xe đầu hàng thoát trước).
        5. Cập nhật đồng hồ mô phỏng (current_time += dt_step) và tăng biến đếm bước.
        6. Tính toán hàm phần thưởng tức thời r_t.
        7. Kiểm tra điều kiện kết thúc tập (1800 giây = 30 phút thực tế).

        Args:
            action: Hành động tác tử lựa chọn (0: Giữ pha, 1: Chuyển pha).

        Returns:
            Tuple chuẩn Gymnasium: (next_observation, reward, terminated, truncated, info).
        """
        # ======================================================================
        # BƯỚC 1: KIỂM SOÁT AN TOÀN BẰNG ACTION MASK
        # ======================================================================
        mask = self.get_action_mask()
        applied_action = action

        # Nếu tác tử vô tình chọn hành động bị cấm, môi trường ép buộc hành động hợp lệ
        if mask[action] == 0:
            applied_action = int(np.argmax(mask))

        switched = (applied_action == ACTION_SWITCH)
        step_passed = 0

        # ======================================================================
        # BƯỚC 2: CẬP NHẬT PHA ĐÈN VÀ TÍNH KHẢ NĂNG XẢ PHƯƠNG TIỆN
        # ======================================================================
        if switched:
            # Khi đổi pha: mất 3s đệm vàng (không xả xe) + 5s đèn xanh cho hướng vuông góc mới
            dt_step = float(DELTA_T_YELLOW + DELTA_T_GREEN)  # Tổng thời gian bước là 3s + 5s = 8s
            self.phase = OPPOSITE_PHASE[self.phase]          # Đổi sang trục vuông góc
            self.phase_duration = float(DELTA_T_GREEN)        # Pha mới bắt đầu duy trì được 5s xanh (tau = 5s)

            # Do vừa chuyển từ đỏ sang xanh, tài xế mất độ trễ khởi hành STARTUP_DELAY = 1.0s
            green_effective = max(0.0, float(DELTA_T_GREEN) - STARTUP_DELAY)  # 5s - 1s = 4s
            discharge_cap = int(np.floor(green_effective * SATURATION_FLOW_RATE))  # 4s * 1.0 = 4 xe/làn
        else:
            # Khi giữ nguyên pha: tiếp tục duy trì đèn xanh thêm 5s
            dt_step = float(DELTA_T_GREEN)                    # Thời gian bước là 5s
            self.phase_duration += float(DELTA_T_GREEN)       # Tích lũy thời gian xanh: tau_{t+1} = tau_t + 5s
            discharge_cap = int(np.floor(float(DELTA_T_GREEN) * SATURATION_FLOW_RATE))  # 5s * 1.0 = 5 xe/làn

        # ======================================================================
        # BƯỚC 3: SINH PHƯƠNG TIỆN MỚI THEO QUÁ TRÌNH POISSON
        # ======================================================================
        # Lấy mẫu số lượng xe mới đi vào ngã tư trong khoảng thời gian dt_step (5s hoặc 8s)
        arrivals = self.traffic_generator.generate_arrivals(self.current_time, dt_step)
        for app in APPROACHES:
            for veh in arrivals[app]:
                # Chỉ nhận thêm xe nếu hàng đợi chưa vượt sức chứa tối đa Q_MAX = 30 xe
                if len(self.queues[app]) < Q_MAX:
                    self.queues[app].append(veh)

        # ======================================================================
        # BƯỚC 4: GIẢI PHÓNG PHƯƠNG TIỆN TỪ CÁC HƯỚNG ĐANG ĐÈN XANH
        # ======================================================================
        green_approaches = PHASE_APPROACHES[self.phase]
        for app in green_approaches:
            # Số xe xả tối đa bằng giới hạn khả năng xả hoặc bằng toàn bộ số xe đang chờ
            num_discharge = min(discharge_cap, len(self.queues[app]))
            for _ in range(num_discharge):
                # Xe đến trước được xả trước (FIFO)
                discharged_veh = self.queues[app].pop(0)

                # Thời điểm xe vượt vạch dừng là cuối bước thời gian hiện tại
                wait_t = discharged_veh.get_wait_time(self.current_time + dt_step)
                self.total_wait_time_accumulated += wait_t
                step_passed += 1
                self.total_passed_vehicles += 1

        # ======================================================================
        # BƯỚC 5: CẬP NHẬT ĐỒNG HỒ MÔ PHỎNG VÀ TÍNH PHẦN THƯỞNG
        # ======================================================================
        self.current_time += dt_step
        self.step_count += 1

        # Tính toán reward tức thời dựa trên trạng thái hàng đợi và số xe đã xả
        reward = self._compute_reward(action_switched=switched, passed_vehicles=step_passed)

        # ======================================================================
        # BƯỚC 6: KIỂM TRA ĐIỀU KIỆN KẾT THÚC TẬP MÔ PHỎNG
        # ======================================================================
        # terminated: Kết thúc do đạt điều kiện đích (bài toán này không có trạng thái đích hấp thụ)
        terminated = False

        # truncated: Hết thời lượng mô phỏng 1 tập (1800 giây = 30 phút thực tế hoặc đạt MAX_STEPS = 360 bước)
        truncated = (self.step_count >= MAX_STEPS) or (self.current_time >= float(EPISODE_DURATION))

        # ======================================================================
        # BƯỚC 7: ĐÓNG GÓI QUAN SÁT TIẾP THEO s_{t+1} VÀ TỪ ĐIỂN TELEMETRY INFO
        # ======================================================================
        observation = self._compute_observation()
        info = self._build_info(passed_vehicles=step_passed)

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Kết xuất đồ họa khung hình. Được ủy quyền xử lý trực quan độc lập bởi ui/studio_app.py và demo_env.py."""
        pass

    def close(self) -> None:
        """Dọn dẹp và giải phóng tài nguyên môi trường khi kết thúc phiên mô phỏng."""
        pass
