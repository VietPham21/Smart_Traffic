"""Bộ sinh dòng xe ngẫu nhiên theo phân phối Poisson (TrafficGenerator).

Mô phỏng dòng phương tiện đi vào ngã tư 4 hướng (Bắc, Nam, Đông, Tây) tuân theo
quá trình Poisson ngẫu nhiên độc lập dựa trên 4 kịch bản chuẩn hóa (SC1_LOW - SC4_DYNAMIC_SURGE).
Cung cấp khả năng kiểm soát Random Seed để nhân bản 2 luồng xe giống hệt nhau phục vụ A/B testing công bằng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from .config import (
    APPROACHES,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    SCENARIOS,
    ScenarioConfig,
    DELTA_T_GREEN,
)


@dataclass
class Vehicle:
    """Lớp biểu diễn một phương tiện giao thông cá thể trong ngã tư.

    Thuộc tính:
        vehicle_id: Mã định danh duy nhất của xe (tăng dần).
        approach: Hướng xe đi vào (0: Bắc, 1: Nam, 2: Đông, 3: Tây).
        arrival_time: Thời điểm xe bắt đầu xuất hiện và xếp vào hàng đợi (tính bằng giây).
    """
    vehicle_id: int
    approach: int
    arrival_time: float

    def get_wait_time(self, current_time: float) -> float:
        """Tính số giây xe đã phải dừng chờ đèn đỏ tính từ khi đến cho tới current_time.

        Công thức:
            wait_time = max(0.0, current_time - arrival_time)
        """
        return max(0.0, current_time - self.arrival_time)


class TrafficGenerator:
    """Bộ tạo dòng xe ngẫu nhiên theo phân phối Poisson độc lập cho 4 nhánh.

    Đặc điểm kỹ thuật:
    - Sử dụng bộ sinh số ngẫu nhiên độc lập np.random.default_rng(seed) đảm bảo tính tái lập.
    - Dễ dàng chuyển đổi linh hoạt giữa 4 kịch bản chuẩn (SC1_LOW -> SC4_DYNAMIC_SURGE).
    - Hỗ trợ hàm clone_synchronized_generator() để sinh ra 2 luồng xe giống nhau 100%
      chạy song song cho giao diện đối sánh Split-Screen A/B (AI vs Baseline).
    """

    def __init__(
        self,
        scenario_id: str = "SC1_LOW",
        seed: Optional[int] = None,
    ) -> None:
        """Khởi tạo bộ sinh xe với kịch bản và hạt giống ngẫu nhiên ban đầu.

        Args:
            scenario_id: Mã kịch bản lưu lượng trong SCENARIOS.
            seed: Hạt giống ngẫu nhiên master PRNG seed.
        """
        self.scenario_id: str = scenario_id
        self._seed: Optional[int] = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._vehicle_counter: int = 0
        self.spawner_enabled: bool = True
        self._live_rates: Optional[dict[int, float]] = None

        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Không tìm thấy scenario_id '{scenario_id}'. Danh sách hợp lệ: {list(SCENARIOS.keys())}"
            )
        self.scenario: ScenarioConfig = SCENARIOS[scenario_id]

    def reset(self, seed: Optional[int] = None) -> None:
        """Đặt lại trạng thái bộ sinh và biến đếm số xe về 0.

        Args:
            seed: Seed mới nếu muốn thay đổi, hoặc None để tái sử dụng seed ban đầu.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._vehicle_counter = 0

    def set_spawner_enabled(self, enabled: bool) -> None:
        """Bật hoặc tắt sinh phương tiện mới (phục vụ nút SpawnerCar: On/Off)."""
        self.spawner_enabled = bool(enabled)

    def set_live_rates(
        self,
        rates: Optional[dict[Any, float] | Sequence[float]],
    ) -> None:
        """Cập nhật nóng cường độ dòng xe lambda_i (xe/giây) từ giao diện Live Injection.

        Truyền None để khôi phục về hàm cường độ mặc định của kịch bản.
        """
        if rates is None:
            self._live_rates = None
            return

        if isinstance(rates, (list, tuple)):
            if len(rates) != len(APPROACHES):
                raise ValueError(f"Expected 4 arrival rates for [N, S, E, W], got {len(rates)}")
            self._live_rates = {
                app: max(0.0, float(rates[idx]))
                for idx, app in enumerate(APPROACHES)
            }
            return

        # Hỗ trợ khóa int (NORTH..WEST) hoặc khóa chuỗi ("N","S","E","W")
        str_map = {"N": NORTH, "S": SOUTH, "E": EAST, "W": WEST,
                   "NORTH": NORTH, "SOUTH": SOUTH, "EAST": EAST, "WEST": WEST}
        resolved: dict[int, float] = dict(self.scenario.rate_func(0.0))
        if self._live_rates is not None:
            resolved.update(self._live_rates)

        for k, v in rates.items():
            if isinstance(k, str):
                app_id = str_map.get(k.upper())
                if app_id is None:
                    raise ValueError(f"Invalid approach key: {k}")
            else:
                app_id = int(k)
            resolved[app_id] = max(0.0, float(v))

        self._live_rates = resolved

    def set_scenario(self, scenario_id: str) -> None:
        """Chuyển đổi kịch bản lưu lượng xe lúc runtime.

        Args:
            scenario_id: Khóa kịch bản trong SCENARIOS ('SC1_LOW', 'SC2_HIGH', v.v.).
        """
        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Không tìm thấy scenario_id '{scenario_id}'. Danh sách hợp lệ: {list(SCENARIOS.keys())}"
            )
        self.scenario_id = scenario_id
        self.scenario = SCENARIOS[scenario_id]
        self._live_rates = None

    def get_arrival_rates(self, current_time: float) -> dict[int, float]:
        """Truy vấn cường độ dòng xe lambda_i (xe/giây) của 4 nhánh tại thời điểm t.

        Args:
            current_time: Thời điểm mô phỏng hiện tại (giây).

        Returns:
            Dict ánh xạ Approach ID -> lambda (xe/giây).
        """
        if self._live_rates is not None:
            return dict(self._live_rates)
        return self.scenario.rate_func(current_time)

    def generate_arrivals(
        self,
        current_time: float,
        delta_t: float = float(DELTA_T_GREEN),
    ) -> dict[int, list[Vehicle]]:
        """Lấy mẫu số lượng xe đến trên 4 nhánh trong khoảng thời gian [t, t + delta_t].

        Cơ sở toán học (Mục I.2.3):
            Số xe mới đến nhánh i tuân theo phân phối Poisson:
            A_i(delta_t) ~ Poisson(lambda_i(t) * delta_t)

        Args:
            current_time: Thời điểm bắt đầu khoảng quyết định (giây).
            delta_t: Độ dài khoảng thời gian của bước (5s nếu giữ pha, 8s nếu chuyển pha).

        Returns:
            Dict ánh xạ Approach ID -> danh sách các đối tượng Vehicle mới xuất hiện.
        """
        new_vehicles: dict[int, list[Vehicle]] = {app: [] for app in APPROACHES}
        if not self.spawner_enabled:
            return new_vehicles

        rates = self.get_arrival_rates(current_time)
        for app in APPROACHES:
            # Kỳ vọng số xe đến trong khoảng delta_t: E[A_i] = lambda_i * delta_t
            lam = rates.get(app, 0.0) * delta_t
            # Lấy mẫu ngẫu nhiên từ phân phối Poisson
            count = int(self._rng.poisson(lam=max(0.0, lam)))
            for _ in range(count):
                self._vehicle_counter += 1
                veh = Vehicle(
                    vehicle_id=self._vehicle_counter,
                    approach=app,
                    arrival_time=current_time,
                )
                new_vehicles[app].append(veh)

        return new_vehicles

    def sample_arrivals(
        self,
        current_time: float = 0.0,
        delta_t: float = float(DELTA_T_GREEN),
    ) -> list[list[Vehicle]]:
        """Lấy mẫu xe đến dưới dạng danh sách 4 nhánh [[N], [S], [E], [W]].

        Khi self.spawner_enabled == False, trả về [[], [], [], []].
        """
        if not self.spawner_enabled:
            return [[], [], [], []]
        arrivals_dict = self.generate_arrivals(current_time=current_time, delta_t=delta_t)
        return [arrivals_dict[app] for app in APPROACHES]

    def clone_synchronized_generator(self) -> TrafficGenerator:
        """Tạo một bản sao độc lập nhưng cùng seed ban đầu để sinh 2 luồng xe giống hệt nhau (A/B testing)."""
        clone = TrafficGenerator(scenario_id=self.scenario_id, seed=self._seed)
        clone.spawner_enabled = self.spawner_enabled
        clone._live_rates = dict(self._live_rates) if self._live_rates is not None else None
        return clone
