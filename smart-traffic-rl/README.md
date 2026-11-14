# SmartTraffic RL Studio 🚦🤖

> **Nền tảng Mô phỏng, Huấn luyện & Đối sánh Đa tầng Điều khiển Tín hiệu Đèn Giao thông Thông minh**  
> Môn học: *Học Tăng Cường và Ứng Dụng* — Khoa Công nghệ Thông tin, Đại học Thủy Lợi (Nhóm 3).

---

## 📌 1. Giới thiệu Dự án

**SmartTraffic RL Studio** giải quyết bài toán tối ưu hóa điều khiển tín hiệu đèn giao thông tại ngã tư đơn 4 hướng (Bắc, Nam, Đông, Tây) bằng các thuật toán Học Tăng Cường (Reinforcement Learning - Q-Learning, Deep Q-Network, Double DQN) đối sánh với các phương pháp điều khiển truyền thống (Fixed-Time, Webster, Actuated Max-Pressure).

Hệ thống được thiết kế theo nguyên lý **Phân tầng (Layered Architecture)** và tuân thủ nghiêm ngặt chuẩn **Gymnasium API** kết hợp cơ chế **Mặt nạ An toàn (Action Masking)**, chống triệt để lỗi "Đói đèn xanh" (Starvation) và đổi pha nhấp nháy (Anti-flicker).

---

## 👥 2. Phân công Trách nhiệm & Nhánh Phát triển (5 Thành viên)

Toàn bộ Interface Contract (chữ ký hàm `select_action(obs, info, eval_mode)`, kiểu dữ liệu, 100% hằng số cấu hình) đã được khóa chặt theo đúng tài liệu thiết kế và yêu cầu kỹ thuật:

| Thành viên | MSV | Vai trò & Trách nhiệm Code | Thẻ TODO trong Code | Nhánh Git khuyến nghị |
| :--- | :---: | :--- | :--- | :--- |
| **Phạm Văn Việt** | 2351170633 | **Lead Architect / Core Env**: Hoàn thiện động lực học ngã tư, tính toán thời gian xả bão hòa, hàm reward và bảo đảm chuẩn Gymnasium trong [core/env.py](file:///f:/Smart_Traffic/smart-traffic-rl/core/env.py). | `# TODO [Phạm Văn Việt]` | `feature/core-env` |
| **Vũ Bá Đài** | 2351170577 | **Deep RL Engineer**: Hoàn thiện thuật toán Double DQN (`use_double_dqn: bool = True`), cơ chế ReplayBuffer và xử lý Action Masking trên PyTorch trong [agents/dqn_agent.py](file:///f:/Smart_Traffic/smart-traffic-rl/agents/dqn_agent.py). | `# TODO [Vũ Bá Đài]` | `feature/double-dqn` |
| **Dương Văn Hưng** | 2351170597 | **Tabular RL Engineer**: Tinh chỉnh bộ lượng tử hóa 972 trạng thái (`transform(obs, info)`), thuật toán Q-Learning và kiểm định 4 kịch bản lưu lượng trong [agents/q_learning.py](file:///f:/Smart_Traffic/smart-traffic-rl/agents/q_learning.py). | `# TODO [Dương Văn Hưng]` | `feature/tabular-qlearning` |
| **Cao Đức Đạo** | 2351170581 | **UI/UX & Graphics Engineer**: Phát triển đồ họa 2D Split-Screen 60 FPS ($1280 \times 720$), vẽ ngã tư, hoạt ảnh xe và các nút bấm tương tác thời gian thực trong [ui/studio_app.py](file:///f:/Smart_Traffic/smart-traffic-rl/ui/studio_app.py). | `# TODO [Cao Đức Đạo]` | `feature/split-screen-ui` |
| **Trương Hữu Quang** | 2351170614 | **Benchmark & Baselines Engineer**: Hoàn thiện các thuật toán đối chứng (Fixed-Time, Webster, Max-Pressure) trong [agents/baselines.py](file:///f:/Smart_Traffic/smart-traffic-rl/agents/baselines.py) và pipeline benchmark xuất biểu đồ trong [experiments/benchmark.py](file:///f:/Smart_Traffic/smart-traffic-rl/experiments/benchmark.py). | `# TODO [Trương Hữu Quang]` | `feature/benchmarks-baselines` |

---

## 🏗️ 3. Cấu trúc Thư mục Dự án

```text
smart-traffic-rl/
├── requirements.txt              # Thư viện phụ thuộc: gymnasium, torch, pygame, pandas, seaborn...
├── README.md                     # Tài liệu kiến trúc, Interface Contract & hướng dẫn sử dụng
├── verify_contract.py            # Test suite tự động kiểm tra 100% Interface Contracts (Smoke Test)
├── core/
│   ├── __init__.py
│   ├── config.py                 # Tập trung 100% hằng số MDP v2.0, trọng số reward, 4 kịch bản chuẩn
│   ├── traffic_generator.py      # Sinh luồng xe Poisson ngẫu nhiên có kiểm soát hạt giống (Seed)
│   └── env.py                    # Gymnasium TrafficLightEnv (Box(10), Discrete(2), Action Masking)
├── agents/
│   ├── __init__.py
│   ├── base_agent.py             # Lớp trừu tượng BaseAgent khóa chặt select_action(obs, info) & update()
│   ├── baselines.py              # FixedTimeAgent, WebsterAgent, Actuated MaxPressureAgent
│   ├── q_learning.py             # StateDiscretizer (972 states) & Tabular QLearningAgent
│   └── dqn_agent.py              # PyTorch QNetwork (10->64->64->2), ReplayBuffer, DQNAgent (use_double_dqn)
├── ui/
│   ├── __init__.py
│   └── studio_app.py             # Pygame 1280x720 Split-Screen Studio (Baseline vs AI + Live Telemetry)
└── experiments/
    ├── __init__.py
    └── benchmark.py              # BenchmarkRunner tự động chạy 4 kịch bản, xuất CSV & vẽ Boxplot
```

---

## 📐 4. Đặc tả Toán học MDP v2.0 & Interface Contracts

### 4.1. Không gian Trạng thái $S$ (10 chiều, chuẩn hóa $[0.0, 1.0]$)
```python
observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
```
Vector trạng thái $s_t = [q_B, q_N, q_E, q_W, w_B, w_N, w_E, w_W, p_t, \tau_t]$:
- $q_i = \min(1.0, Q_i / Q_{MAX})$: Tỷ lệ lấp đầy hàng đợi 4 nhánh ($Q_{MAX} = 30$ xe).
- $w_i = \min(1.0, W_i / W_{MAX})$: Thời gian chờ chuẩn hóa của xe đầu hàng đợi ($W_{MAX} = 120.0$s).
- $p_t \in \{0.0, 1.0\}$: Pha đèn hiện tại ($0.0$: Bắc-Nam xanh; $1.0$: Đông-Tây xanh).
- $\tau_t = \min(1.0, \tau / T_{MAX})$: Thời gian xanh liên tục của pha hiện tại ($T_{MAX} = 60$s).

### 4.2. Không gian Hành động $A$ và Mặt nạ An toàn (Action Masking)
```python
action_space = spaces.Discrete(2)  # 0: KEEP current phase, 1: SWITCH phase
```
Môi trường trả về mặt nạ nhị phân `get_action_mask() -> np.ndarray(shape=(2,), dtype=np.int8)`:
- **Ràng buộc xanh tối thiểu ($T_{MIN} = 10$s)**: Nếu $\tau < 10$s $\rightarrow$ `mask = [1, 0]` (cấm đổi pha, chống nhấp nháy đèn).
- **Ràng buộc xanh tối đa ($T_{MAX} = 60$s)**: Nếu $\tau \ge 60$s và hướng đèn đỏ có xe $\rightarrow$ `mask = [0, 1]` (bắt buộc đổi pha để chống đói đèn).
- **Khoảng tự do ($10\text{s} \le \tau < 60\text{s}$)**: `mask = [1, 1]` (tác tử toàn quyền lựa chọn).

### 4.3. Hàm Phần thưởng Đa mục tiêu $R_t$ (Multi-Objective Reward)
$$R_t = - \alpha \sum_{i=1}^{4} Q_i - \beta \sum_{i=1}^{4} \left(\frac{W_i}{W_{MAX}}\right)^2 - c_{switch} \cdot \mathbf{1}(a_t = 1) + \eta \cdot N_{passed}$$
- $\alpha = 1.0$ (`ALPHA_QUEUE`): Phạt ùn ứ hàng đợi.
- $\beta = 1.5$ (`BETA_WAIT_SQ`): Phạt bình phương thời gian chờ (càng chờ lâu phạt càng nặng, chống bỏ rơi nhánh phụ).
- $c_{switch} = 0.25$ (`C_SWITCH`): Phạt chuyển pha (chi phí 3 giây đèn vàng không xả được xe).
- $\eta = 0.4$ (`ETA_PASSED`): Thưởng số lượng xe đã giải phóng thành công qua nút giao.
- Hệ số chiết khấu $\gamma = 0.95$ (`GAMMA`): Tầm nhìn quy hoạch ~100 giây.

### 4.4. Chuẩn Trả về của Hàm `step(action)` và Từ điển `info`
```python
observation, reward, terminated, truncated, info = env.step(action)
```
Từ điển `info` đảm bảo 100% chứa đầy đủ 8 keys bắt buộc:
```python
info = {
    "raw_queues": [qB, qN, qE, qW],      # Số lượng xe thực tế tại 4 nhánh (list[int])
    "max_waits": [wB, wN, wE, wW],        # Số giây chờ thực tế của xe đứng đầu (list[float])
    "phase": 0,                           # 0: Bắc-Nam, 1: Đông-Tây (int)
    "phase_duration": 15.0,               # Số giây xanh liên tục (float)
    "action_mask": np.array([1, 1], dtype=np.int8), # Mặt nạ hành động hợp lệ
    "step_passed_vehicles": 2,            # Số xe vừa thoát trong step này (int)
    "total_passed": 142,                  # Tổng số xe đã thoát trong episode (int)
    "avg_wait_time": 18.4,                # Thời gian chờ trung bình (float, giây/xe)
}
```

### 4.5. Lớp Trừu tượng `BaseAgent`
Tất cả các agent phải tuân thủ nghiêm ngặt chữ ký hàm:
```python
class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, obs: np.ndarray, info: dict, eval_mode: bool = False) -> int:
        """Bắt buộc trả về action in {0, 1} và chỉ được chọn nơi info['action_mask'] == 1."""
        ...

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool, info: dict, next_info: dict) -> dict | None:
        """Tác tử học cập nhật trọng số; baseline trả về None."""
        ...
```

---

## 🚗 5. Danh mục 4 Kịch bản Kiểm thử Chuẩn hóa (SC1_LOW - SC4_DYNAMIC_SURGE)

Mỗi kịch bản mô phỏng 1 tập 30 phút ($T = 1800$ giây = 360 bước với $\Delta t = 5$s):

1. **SC1_LOW**: Lưu lượng thấp đồng đều ($\lambda = 0.05$ xe/s cả 4 hướng $\approx 180$ xe/h/làn).
2. **SC2_HIGH**: Cao điểm đồng đều ($\lambda = 0.25$ xe/s $\approx 900$ xe/h/làn).
3. **SC3_ASYMMETRIC**: Trục chính Bắc-Nam đông ($\lambda_{NS} = 0.35$ xe/s), trục phụ Đông-Tây vắng ($\lambda_{EW} = 0.08$ xe/s).
4. **SC4_DYNAMIC_SURGE**: Làn sóng cao điểm thay đổi:
   - 0 – 10 phút: Bắc - Nam đông ($\lambda_{NS} = 0.35, \lambda_{EW} = 0.08$)
   - 10 – 20 phút: Đột ngột đảo chiều sang Đông - Tây đông ($\lambda_{NS} = 0.08, \lambda_{EW} = 0.35$)
   - 20 – 30 phút: Cân bằng phục hồi ($\lambda = 0.20$)

---

## 🚀 6. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### 6.1. Cài đặt Môi trường
Yêu cầu **Python 3.10+**:
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 6.2. Chạy Kiểm tra Hợp đồng (Smoke Test)
```bash
python verify_contract.py
```
Kết quả hiển thị:
```text
================================================================================
  SMARTTRAFFIC RL STUDIO — INTERFACE CONTRACT VERIFICATION (Yeu_Cau.txt)
================================================================================
  [1/4] Checking core/config.py constants...
        -> OK: All MDP v2.0 constants and reward weights matched exactly.
  [2/4] Checking StateDiscretizer.transform(obs, info) interface...
        -> OK: Discretizer returned tuple (0, 0, 0, 0, 0, 0, 0) mapping to flat state ID 0 (0-971).
  [3/4] Initializing TrafficLightEnv and 5 Candidate Agents...
  [4/4] Executing 10-step simulation loops (reset() -> select_action() -> step())...

    Testing Agent 1/5: FixedTimeAgent...
      [PASS] 10/10 steps verified successfully for FixedTimeAgent.
    Testing Agent 2/5: MaxPressureAgent...
      [PASS] 10/10 steps verified successfully for MaxPressureAgent.
    Testing Agent 3/5: QLearningAgent...
      [PASS] 10/10 steps verified successfully for QLearningAgent.
    Testing Agent 4/5: Standard-DQN (use_double_dqn=False)...
      [PASS] 10/10 steps verified successfully for Standard-DQN.
    Testing Agent 5/5: Double-DQN (use_double_dqn=True)...
      [PASS] 10/10 steps verified successfully for Double-DQN.

================================================================================
  ALL INTERFACE CONTRACT REQUIREMENTS FROM Yeu_Cau.txt VERIFIED SUCCESSFULLY!
================================================================================
```

### 6.3. Khởi chạy Giao diện Split-Screen Studio (Pygame 1280x720)
```bash
python -c "from ui.studio_app import SplitScreenStudio; studio = SplitScreenStudio(); studio.run()"
```
- **Phím `Space`**: Tạm dừng / Tiếp tục mô phỏng.
- **Phím `1`, `2`, `3`, `4`**: Đổi nhanh kịch bản giao thông (SC1_LOW đến SC4_DYNAMIC_SURGE).
- **Bảng điều khiển phải**: Bấm chuột chọn tốc độ (1x, 2x, 5x, 10x) và theo dõi Live Delta thời gian chờ.

### 6.4. Chạy Benchmark Tự động & Xuất Biểu đồ
```bash
python -c "from experiments.benchmark import BenchmarkRunner; runner = BenchmarkRunner(); df = runner.run_benchmark(num_episodes_per_scenario=2); runner.export_csv(df); runner.plot_results(df)"
```

---

## 🔒 7. Quy ước Phối hợp Nhóm (Git Workflow Guidelines)

1. **Interface Freeze**: Tuyệt đối **KHÔNG ĐƯỢC** thay đổi cấu trúc của `observation_space`, `action_space`, các tham số của `core/config.py`, và chữ ký hàm `BaseAgent.select_action()`.
2. **Quy tắc Kiểm thử trước khi tạo Pull Request**: Mỗi thành viên trước khi commit code vào nhánh chính bắt buộc phải chạy `python verify_contract.py`. Chỉ tạo PR khi toàn bộ test pass.
3. **Seed Consistency**: Luôn cố định `seed` khi đánh giá thực nghiệm để đảm bảo luồng phương tiện đối chứng giữa Baseline và AI là hoàn toàn đồng nhất.
