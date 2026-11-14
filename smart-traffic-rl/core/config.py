"""Configuration constants and hyperparameter specifications for SmartTraffic RL Studio.

Standardized 100% according to Yeu_Cau.txt and Technical Specification:
- MDP v2.0 parameters (STATE_DIM, ACTION_DIM, Q_MAX, W_MAX, T_MIN, T_MAX, etc.)
- Multi-objective reward coefficients (ALPHA_QUEUE, BETA_WAIT_SQ, C_SWITCH, ETA_PASSED)
- Benchmark traffic scenarios (SC1_LOW, SC2_HIGH, SC3_ASYMMETRIC, SC4_DYNAMIC_SURGE)
- Hyperparameters for Tabular Q-Learning and Deep RL (DQN / Double DQN)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Final


# ==============================================================================
# 1. INTERSECTION TOPOLOGY & APPROACHES
# ==============================================================================

class Approach(IntEnum):
    """Four intersection approaches: North (B), South (N), East (E), West (W)."""
    NORTH = 0  # B - Bắc
    SOUTH = 1  # N - Nam
    EAST = 2   # E - Đông
    WEST = 3   # W - Tây


NORTH: Final[int] = Approach.NORTH
SOUTH: Final[int] = Approach.SOUTH
EAST: Final[int] = Approach.EAST
WEST: Final[int] = Approach.WEST

APPROACHES: Final[tuple[int, ...]] = (NORTH, SOUTH, EAST, WEST)
APPROACH_NAMES: Final[dict[int, str]] = {
    NORTH: "North (Bắc)",
    SOUTH: "South (Nam)",
    EAST: "East (Đông)",
    WEST: "West (Tây)",
}

AXIS_NS: Final[tuple[int, int]] = (NORTH, SOUTH)
AXIS_EW: Final[tuple[int, int]] = (EAST, WEST)


# ==============================================================================
# 2. MDP v2.0 CORE CONSTANTS (Yeu_Cau.txt Exact Specifications)
# ==============================================================================

STATE_DIM: Final[int] = 10                  # [qB, qN, qE, qW, wB, wN, wE, wW, p_t, tau_t]
ACTION_DIM: Final[int] = 2                 # 0: KEEP current phase, 1: SWITCH phase
Q_MAX: Final[int] = 30                     # Sức chứa hàng đợi tối đa mỗi nhánh (xe)
W_MAX: Final[float] = 120.0                # Ngưỡng chuẩn hóa thời gian chờ (giây)
T_MIN: Final[int] = 10                     # Thời gian xanh tối thiểu bắt buộc (giây)
T_MAX: Final[int] = 60                     # Thời gian xanh tối đa trước khi ép đổi pha (giây)
DELTA_T_GREEN: Final[int] = 5              # Bước thời gian quyết định khi giữ pha (giây)
DELTA_T_YELLOW: Final[int] = 3             # Thời gian đệm đèn vàng khi đổi pha (giây)
SATURATION_FLOW_RATE: Final[float] = 1.0   # Tốc độ xả bão hòa (xe/giây)
EPISODE_DURATION: Final[int] = 1800        # Thời lượng 1 tập mô phỏng (30 phút = 1800 giây)
MAX_STEPS: Final[int] = int(EPISODE_DURATION / DELTA_T_GREEN)  # 360 decision steps

# Backward-compatibility aliases
OBS_DIM: Final[int] = STATE_DIM
MAX_QUEUE_CAPACITY: Final[int] = Q_MAX
MAX_WAIT_TIME: Final[float] = W_MAX
MIN_GREEN_TIME: Final[float] = float(T_MIN)
MAX_GREEN_TIME: Final[float] = float(T_MAX)
DELTA_T: Final[float] = float(DELTA_T_GREEN)
YELLOW_TIME: Final[float] = float(DELTA_T_YELLOW)
STARVATION_THRESHOLD: Final[float] = 40.0  # Ngưỡng kích hoạt cờ cảnh báo đói đèn (Mục I.5.2: w_i >= 40s)
STARTUP_DELAY: Final[float] = 1.0         # Độ trễ khởi hành khi mới chuyển sang xanh (giây)


# ==============================================================================
# 3. PHASES & ACTIONS
# ==============================================================================

class Phase(IntEnum):
    """Active green phase: 0 = North-South green, 1 = East-West green."""
    NS_GREEN = 0  # Trục Bắc - Nam đèn xanh, Đông - Tây đèn đỏ
    EW_GREEN = 1  # Trục Đông - Tây đèn xanh, Bắc - Nam đèn đỏ


PHASE_NS: Final[int] = Phase.NS_GREEN
PHASE_EW: Final[int] = Phase.EW_GREEN

PHASE_APPROACHES: Final[dict[int, tuple[int, int]]] = {
    PHASE_NS: AXIS_NS,
    PHASE_EW: AXIS_EW,
}

OPPOSITE_PHASE: Final[dict[int, int]] = {
    PHASE_NS: PHASE_EW,
    PHASE_EW: PHASE_NS,
}


class Action(IntEnum):
    """Discrete action set for signal controller."""
    KEEP = 0    # Duy trì pha hiện tại thêm DELTA_T_GREEN giây
    SWITCH = 1  # Chuyển pha (3s đệm vàng cho pha cũ, sau đó bật xanh pha mới)


ACTION_KEEP: Final[int] = Action.KEEP
ACTION_SWITCH: Final[int] = Action.SWITCH


# ==============================================================================
# 4. REWARD FUNCTION COEFFICIENTS (Yeu_Cau.txt Exact Specifications)
# ==============================================================================
# Formula:
# R_t = - ALPHA_QUEUE * sum(Q_i)
#       - BETA_WAIT_SQ * sum((W_i / W_MAX)^2)
#       - C_SWITCH * 1(action_switched)
#       + ETA_PASSED * passed_vehicles

ALPHA_QUEUE: Final[float] = 1.0            # Trọng số phạt tổng độ dài hàng đợi ùn ứ
BETA_WAIT_SQ: Final[float] = 1.5           # Trọng số phạt bình phương thời gian chờ (chống đói đèn)
C_SWITCH: Final[float] = 0.25              # Phạt chi phí cơ hội chuyển pha (3 giây đèn vàng)
ETA_PASSED: Final[float] = 0.4             # Thưởng thông lượng xe thoát nút giao
DELTA_V_MAX: Final[float] = 10.0           # Giới hạn biến thiên vận tốc / áp lực
GAMMA: Final[float] = 0.95                 # Hệ số chiết khấu Bellman (tầm nhìn ~100s)

# Backward-compatibility aliases
WEIGHT_QUEUE: Final[float] = ALPHA_QUEUE
WEIGHT_WAIT: Final[float] = BETA_WAIT_SQ
WEIGHT_SWITCH: Final[float] = C_SWITCH
WEIGHT_PASSED: Final[float] = ETA_PASSED


# ==============================================================================
# 5. STATE DISCRETIZER CONSTANTS (972 DISCRETE STATES)
# ==============================================================================
# Bins: 4 approaches x 3 bins (3^4 = 81) x 2 starve states x 2 phases x 3 tau levels = 972

QUEUE_BIN_LOW_MAX: Final[int] = 3       # Mức 0 (Vắng): <= 3 xe
QUEUE_BIN_MED_MAX: Final[int] = 10      # Mức 1 (Trung bình): 4 - 10 xe
                                        # Mức 2 (Ùn ứ): > 10 xe
NUM_QUEUE_BINS: Final[int] = 3
NUM_STARVATION_STATES: Final[int] = 2   # 0: Bình thường, 1: Có xe chờ >= 40s ở hướng đèn đỏ (Mục I.5.2)
NUM_PHASE_STATES: Final[int] = 2        # 0: Bắc-Nam, 1: Đông-Tây
NUM_TAU_STATES: Final[int] = 3          # 0: tau < T_MIN (10s), 1: T_MIN <= tau < T_MAX (10-60s), 2: tau >= T_MAX (60s)

TOTAL_DISCRETE_STATES: Final[int] = (
    (NUM_QUEUE_BINS ** 4) * NUM_STARVATION_STATES * NUM_PHASE_STATES * NUM_TAU_STATES
)  # 81 * 2 * 2 * 3 = 972 states


# ==============================================================================
# 6. STANDARDIZED TRAFFIC SCENARIOS (SC1_LOW - SC4_DYNAMIC_SURGE)
# ==============================================================================
# Thông số chuẩn hóa theo Mục I.5.1 Báo cáo N3_Học TC & UD:
# SC1_LOW: lambda = 0.10 xe/s cả 4 nhánh
# SC2_HIGH: lambda = 0.35 xe/s cả 4 nhánh
# SC3_ASYMMETRIC: lambda_NS = 0.42 (Trục chính), lambda_EW = 0.08 (Trục phụ)
# SC4_DYNAMIC_SURGE: Phút 0-10 NS đông (0.45), Phút 10-20 EW đông (0.45), Phút 20-30 Cân bằng (0.20)

@dataclass(frozen=True)
class ScenarioConfig:
    """Specification of a standardized benchmark traffic scenario."""
    scenario_id: str
    name: str
    description: str
    rate_func: Callable[[float], dict[int, float]]
    tags: list[str] = field(default_factory=list)


def _sc1_rates(t: float) -> dict[int, float]:
    """SC1_LOW: 0.10 veh/s on all approaches (~360 veh/h/lane)."""
    return {NORTH: 0.10, SOUTH: 0.10, EAST: 0.10, WEST: 0.10}


def _sc2_rates(t: float) -> dict[int, float]:
    """SC2_HIGH: 0.35 veh/s on all approaches (~1260 veh/h/lane)."""
    return {NORTH: 0.35, SOUTH: 0.35, EAST: 0.35, WEST: 0.35}


def _sc3_rates(t: float) -> dict[int, float]:
    """SC3_ASYMMETRIC: NS heavy (0.42 veh/s), EW light (0.08 veh/s)."""
    return {NORTH: 0.42, SOUTH: 0.42, EAST: 0.08, WEST: 0.08}


def _sc4_rates(t: float) -> dict[int, float]:
    """SC4_DYNAMIC_SURGE:
    - 0-600s: NS heavy (0.45), EW light (0.08)
    - 600-1200s: EW heavy (0.45), NS light (0.08)
    - 1200-1800s: Balanced recovery (0.20 on both)
    """
    if t < 600.0:
        return {NORTH: 0.45, SOUTH: 0.45, EAST: 0.08, WEST: 0.08}
    elif t < 1200.0:
        return {NORTH: 0.08, SOUTH: 0.08, EAST: 0.45, WEST: 0.45}
    else:
        return {NORTH: 0.20, SOUTH: 0.20, EAST: 0.20, WEST: 0.20}


SC1_LOW: Final[ScenarioConfig] = ScenarioConfig(
    scenario_id="SC1_LOW",
    name="Low Traffic",
    description="Lưu lượng thấp đồng đều (0.10 xe/s trên cả 4 nhánh)",
    rate_func=_sc1_rates,
    tags=["baseline", "sparse", "quick-response"],
)

SC2_HIGH: Final[ScenarioConfig] = ScenarioConfig(
    scenario_id="SC2_HIGH",
    name="High Peak",
    description="Cao điểm đồng đều (0.35 xe/s trên cả 4 nhánh)",
    rate_func=_sc2_rates,
    tags=["peak", "congestion", "saturation"],
)

SC3_ASYMMETRIC: Final[ScenarioConfig] = ScenarioConfig(
    scenario_id="SC3_ASYMMETRIC",
    name="Asymmetric Axis",
    description="Lệch pha trục chính - phụ (Bắc-Nam: 0.42 xe/s, Đông-Tây: 0.08 xe/s)",
    rate_func=_sc3_rates,
    tags=["asymmetric", "starvation-test", "priority"],
)

SC4_DYNAMIC_SURGE: Final[ScenarioConfig] = ScenarioConfig(
    scenario_id="SC4_DYNAMIC_SURGE",
    name="Dynamic Surge",
    description="Biến thiên làn sóng cao điểm (Phút 0-10: NS 0.45, Phút 10-20: EW 0.45, Phút 20-30: Cân bằng 0.20)",
    rate_func=_sc4_rates,
    tags=["surge", "wave", "dynamic-adaptation"],
)

SCENARIOS: Final[dict[str, ScenarioConfig]] = {
    "SC1_LOW": SC1_LOW,
    "SC2_HIGH": SC2_HIGH,
    "SC3_ASYMMETRIC": SC3_ASYMMETRIC,
    "SC4_DYNAMIC_SURGE": SC4_DYNAMIC_SURGE,
    # Short name aliases for convenience
    "SC1": SC1_LOW,
    "SC2": SC2_HIGH,
    "SC3": SC3_ASYMMETRIC,
    "SC4": SC4_DYNAMIC_SURGE,
}


# ==============================================================================
# 7. AGENT HYPERPARAMETERS (Section III.2)
# ==============================================================================

# Baseline: Fixed Time Control
FIXED_TIME_GREEN: Final[float] = 30.0

# Baseline: Webster Optimal
WEBSTER_LOST_TIME: Final[float] = 4.0

# Baseline: Actuated Max-Pressure Control
MAX_PRESSURE_THRESHOLD: Final[float] = 3.0

# Tabular Q-Learning Hyperparameters
QL_LEARNING_RATE: Final[float] = 0.1
QL_EPSILON_START: Final[float] = 1.0
QL_EPSILON_MIN: Final[float] = 0.01
QL_EPSILON_DECAY: Final[float] = 0.995

# Double Deep Q-Network (Double DQN) Hyperparameters
DQN_HIDDEN_DIMS: Final[tuple[int, int]] = (64, 64)  # Architecture: 10 -> 64 -> 64 -> 2
DQN_BUFFER_CAPACITY: Final[int] = 50_000
DQN_BATCH_SIZE: Final[int] = 64
DQN_LEARNING_RATE: Final[float] = 1e-3
DQN_TARGET_UPDATE_FREQ: Final[int] = 200
DQN_MIN_REPLAY_SIZE: Final[int] = 1_000
DQN_EPSILON_START: Final[float] = 1.0
DQN_EPSILON_MIN: Final[float] = 0.02
DQN_EPSILON_DECAY: Final[float] = 0.995


# ==============================================================================
# 8. UI & VISUALIZATION (PYGAME 1280x720)
# ==============================================================================

WINDOW_WIDTH: Final[int] = 1280
WINDOW_HEIGHT: Final[int] = 720
FPS: Final[int] = 60

LEFT_PANE_WIDTH: Final[int] = 500   # Khung Trái (Mục I.5.3: 500x720px)
MID_PANE_WIDTH: Final[int] = 500    # Khung Giữa (Mục I.5.3: 500x720px)
RIGHT_PANE_WIDTH: Final[int] = 280  # Bảng Điều khiển Phải (Mục I.5.3: 280x720px)

COLOR_BG: Final[tuple[int, int, int]] = (18, 22, 34)
COLOR_PANE_BORDER: Final[tuple[int, int, int]] = (45, 55, 75)
COLOR_ROAD: Final[tuple[int, int, int]] = (35, 40, 52)
COLOR_ROAD_MARKING: Final[tuple[int, int, int]] = (200, 205, 215)
COLOR_GREEN_LIGHT: Final[tuple[int, int, int]] = (34, 197, 94)
COLOR_YELLOW_LIGHT: Final[tuple[int, int, int]] = (234, 179, 8)
COLOR_RED_LIGHT: Final[tuple[int, int, int]] = (239, 68, 68)
COLOR_CAR_BODY: Final[tuple[int, int, int]] = (56, 189, 248)
COLOR_TEXT_PRIMARY: Final[tuple[int, int, int]] = (248, 250, 252)
COLOR_TEXT_MUTED: Final[tuple[int, int, int]] = (148, 163, 184)
COLOR_ACCENT_RL: Final[tuple[int, int, int]] = (168, 85, 247)
COLOR_ACCENT_BASELINE: Final[tuple[int, int, int]] = (59, 130, 246)
