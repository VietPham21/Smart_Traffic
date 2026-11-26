"""Bộ sinh dòng xe ngẫu nhiên theo phân phối Poisson (TrafficGenerator).

Mô phỏng dòng phương tiện đi vào ngã tư 4 hướng (Bắc, Nam, Đông, Tây) tuân theo
quá trình Poisson ngẫu nhiên độc lập dựa trên 4 kịch bản chuẩn hóa (SC1_LOW - SC4_DYNAMIC_SURGE).
Cung cấp cơ chế bộ đệm theo từng giây (1-second buckets) đảm bảo hai bộ sinh dùng
chung seed tạo ra luồng phương tiện và thời điểm đến (arrival_time) giống hệt nhau 100%,
bất biến với việc hai thuật toán chọn bước 5s hay 8s trong Split-Screen A/B Testing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence
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
    - Sinh xe theo từng giây rời rạc (1-second buckets) với thời điểm đến phân bố đều
      trên [sec, sec + 1) để đảm bảo tính bất biến thời gian (step-size invariance).
    - Sử dụng bộ sinh số ngẫu nhiên độc lập np.random.default_rng đảm bảo tái lập thực nghiệm.
    - Hỗ trợ hàm clone_synchronized_generator() và đồng bộ trực tiếp trong Split-Screen A/B.
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
        self._effective_seed: int = (
            int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31 - 1))
        )
        self._rng: np.random.Generator = np.random.default_rng(self._effective_seed)
        self._vehicle_counter: int = 0
        self.spawner_enabled: bool = True
        self._live_rates: Optional[dict[int, float]] = None

        # Bộ đệm theo từng giây nguyên sec -> {approach: [arrival_time_1, ...]}
        self._bucket_cache: dict[int, dict[int, list[float]]] = {}
        self._next_gen_sec: int = 0
        self._epoch: int = 0

        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Không tìm thấy scenario_id '{scenario_id}'. Danh sách hợp lệ: {list(SCENARIOS.keys())}"
            )
        self.scenario: ScenarioConfig = SCENARIOS[scenario_id]

    def reset(self, seed: Optional[int] = None) -> None:
        """Đặt lại trạng thái bộ sinh, bộ đệm từng giây và biến đếm số xe về 0.

        Args:
            seed: Seed mới nếu muốn thay đổi, hoặc None để tái sử dụng seed ban đầu.
        """
        if seed is not None:
            self._seed = seed
            self._effective_seed = int(seed)
        self._rng = np.random.default_rng(self._effective_seed)
        self._vehicle_counter = 0
        self._bucket_cache.clear()
        self._next_gen_sec = 0
        self._epoch = 0

    def _invalidate_schedule_from(self, from_sec: int) -> None:
        """Xóa bộ đệm các giây >= from_sec và đồng bộ lại PRNG nếu đổi cấu hình nóng."""
        from_sec = max(0, int(from_sec))
        for sec_k in [k for k in self._bucket_cache if k >= from_sec]:
            del self._bucket_cache[sec_k]
        self._next_gen_sec = from_sec
        self._epoch += 1
        self._rng = np.random.default_rng([self._effective_seed, self._epoch, from_sec])

    def pregenerate_until(self, up_to_time: float) -> None:
        """Sinh trước bộ đệm tới mốc giây ceil(up_to_time) để hai clone đồng bộ tuyệt đối trước khi đổi cấu hình."""
        target_sec = max(0, int(math.ceil(up_to_time)))
        while self._next_gen_sec < target_sec:
            self._generate_one_second_bucket(self._next_gen_sec)
            self._next_gen_sec += 1

    def set_spawner_enabled(self, enabled: bool, sync_from_time: Optional[float] = None) -> None:
        """Bật hoặc tắt sinh phương tiện mới (phục vụ nút SpawnerCar: On/Off)."""
        enabled_bool = bool(enabled)
        if sync_from_time is not None:
            sync_sec = max(0, int(math.ceil(sync_from_time)))
            self.pregenerate_until(float(sync_sec))
            self.spawner_enabled = enabled_bool
            self._invalidate_schedule_from(sync_sec)
        else:
            self.spawner_enabled = enabled_bool
            # Clear cached buckets at/after _next_gen_sec only if turning ON after being OFF
            if enabled_bool and self._next_gen_sec == 0:
                self._bucket_cache.clear()

    def set_live_rates(
        self,
        rates: Optional[dict[Any, float] | Sequence[float]],
        sync_from_time: Optional[float] = None,
    ) -> None:
        """Cập nhật nóng cường độ dòng xe lambda_i (xe/giây) từ giao diện Live Injection.

        Truyền None để khôi phục về hàm cường độ mặc định của kịch bản.
        """
        if sync_from_time is not None:
            self.pregenerate_until(sync_from_time)

        if rates is None:
            self._live_rates = None
        elif isinstance(rates, (list, tuple)):
            if len(rates) != len(APPROACHES):
                raise ValueError(f"Expected 4 arrival rates for [N, S, E, W], got {len(rates)}")
            self._live_rates = {
                app: max(0.0, float(rates[idx]))
                for idx, app in enumerate(APPROACHES)
            }
        else:
            # Hỗ trợ khóa int (NORTH..WEST) hoặc khóa chuỗi ("N","S","E","W")
            str_map = {
                "N": NORTH,
                "S": SOUTH,
                "E": EAST,
                "W": WEST,
                "NORTH": NORTH,
                "SOUTH": SOUTH,
                "EAST": EAST,
                "WEST": WEST,
            }
            ref_t = float(sync_from_time) if sync_from_time is not None else 0.0
            resolved: dict[int, float] = dict(self.scenario.rate_func(ref_t))
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

        if sync_from_time is not None:
            self._invalidate_schedule_from(int(math.ceil(sync_from_time)))

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

    def _generate_one_second_bucket(self, sec: int) -> dict[int, list[float]]:
        """Sinh ngẫu nhiên danh sách thời điểm xe đến trong khoảng giây [sec, sec + 1)."""
        if not self.spawner_enabled:
            empty = {app: [] for app in APPROACHES}
            self._bucket_cache[sec] = empty
            return empty

        rates = self.get_arrival_rates(float(sec) + 0.5)
        sec_arrivals: dict[int, list[float]] = {}
        for app in APPROACHES:
            lam = max(0.0, float(rates.get(app, 0.0)))
            count = int(self._rng.poisson(lam=lam))
            if count > 0:
                offsets = np.sort(self._rng.uniform(0.0, 0.999999, size=count))
                sec_arrivals[app] = [float(sec) + float(off) for off in offsets]
            else:
                sec_arrivals[app] = []

        self._bucket_cache[sec] = sec_arrivals
        return sec_arrivals

    def _get_or_create_bucket(self, sec: int) -> dict[int, list[float]]:
        """Truy xuất hoặc sinh lười bộ đệm cho giây nguyên `sec` theo đúng thứ tự."""
        if sec in self._bucket_cache:
            return self._bucket_cache[sec]

        while self._next_gen_sec <= sec:
            self._generate_one_second_bucket(self._next_gen_sec)
            self._next_gen_sec += 1

        return self._bucket_cache[sec]

    def generate_arrivals(
        self,
        current_time: float,
        delta_t: float = float(DELTA_T_GREEN),
    ) -> dict[int, list[Vehicle]]:
        """Lấy mẫu số lượng xe đến trên 4 nhánh trong khoảng thời gian [t, t + delta_t).

        Cơ sở toán học (Quá trình Poisson theo từng giây rời rạc):
            Mỗi giây nguyên s được sinh độc lập A_i(s) ~ Poisson(lambda_i(s + 0.5)),
            thời điểm đến của từng xe được lấy mẫu đều trong [s, s + 1) và sắp xếp tăng dần.
            Do đó mọi cách phân hoạch bước thời gian (5s hay 3s+5s) đều trả về cùng tập xe.

        Args:
            current_time: Thời điểm bắt đầu khoảng (giây).
            delta_t: Độ dài khoảng thời gian (giây).

        Returns:
            Dict ánh xạ Approach ID -> danh sách đối tượng Vehicle sắp xếp theo arrival_time.
        """
        new_vehicles: dict[int, list[Vehicle]] = {app: [] for app in APPROACHES}
        if not self.spawner_enabled or delta_t <= 0.0:
            return new_vehicles

        t_start = float(current_time)
        t_end = t_start + float(delta_t)
        sec_start = max(0, int(math.floor(t_start)))
        sec_end = max(sec_start, int(math.ceil(t_end)))

        for sec in range(sec_start, sec_end):
            bucket = self._get_or_create_bucket(sec)
            for app in APPROACHES:
                for arr_t in bucket[app]:
                    if t_start - 1e-9 <= arr_t < t_end - 1e-9:
                        self._vehicle_counter += 1
                        new_vehicles[app].append(
                            Vehicle(
                                vehicle_id=self._vehicle_counter,
                                approach=app,
                                arrival_time=arr_t,
                            )
                        )

        for app in APPROACHES:
            if len(new_vehicles[app]) > 1:
                new_vehicles[app].sort(key=lambda v: v.arrival_time)

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
        clone = TrafficGenerator(scenario_id=self.scenario_id, seed=self._effective_seed)
        clone.spawner_enabled = self.spawner_enabled
        clone._live_rates = dict(self._live_rates) if self._live_rates is not None else None
        return clone
