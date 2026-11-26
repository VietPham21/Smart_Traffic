# SmartTraffic RL Studio (Phương án B) 🚦🤖

> **Nền tảng Mô phỏng, Huấn luyện & Đối sánh Đa tầng Điều khiển Tín hiệu Đèn Giao thông Thông minh**  
> Môn học: *Học Tăng Cường và Ứng Dụng* — Khoa Công nghệ Thông tin, Đại học Thủy Lợi (Nhóm 3).

---

## 📌 1. Giới thiệu Dự án

**SmartTraffic RL Studio** giải quyết bài toán tối ưu hóa điều khiển tín hiệu đèn giao thông ngã tư đơn 4 hướng (Bắc, Nam, Đông, Tây) trên mô hình toán học **MDP v2.0** tuân thủ chuẩn **Gymnasium API**, kết hợp **Action Masking** ($T_{\min}=10\text{s}, T_{\max}=60\text{s}$, 3s đệm đèn vàng), chuẩn hóa phần thưởng **Semi-MDP** theo thời lượng bước $\frac{\Delta t_{\text{step}}}{\Delta t_{\text{green}}}$, và hỗ trợ đầy đủ cả 2 nhánh Học tăng cường lớn:
- **Value-Based:** Tabular **Q-Learning** (972 states), **Standard DQN**, **Double DQN** (PyTorch MLP `10->64->64->2`)
- **Policy-Based / Actor-Critic:** Masked **PPO-Clip** + **GAE** ($\gamma=0.95, \lambda=0.95$, PyTorch MLP `10->128->128`)
- **Baselines:** **Fixed-Time** (30s), **Webster Optimal**, **Actuated Max-Pressure** ($\theta_{\text{diff}}=3$)

---

## 🏗️ 2. Cấu trúc Thư mục Dự án

```text
smart-traffic-rl/
├── requirements.txt             # gymnasium, numpy, torch, onnx, onnxruntime, tensorboard, pygame, pandas, matplotlib, seaborn
├── README.md                    # Tài liệu kiến trúc & hướng dẫn sử dụng
├── verify_contract.py           # Kiểm thử tự động 6 Agents, 15+ Info Keys, A/B Poisson, Truncation & ONNX
├── demo_env.py                  # MDP Interactive Inspector (Thực hành 3: kiểm thử thủ công từng bước MDP)
├── core/
│   ├── config.py                # Hằng số MDP v2.0, 4 kịch bản SC1-SC4, Q_LEARNING/DQN/PPO_CONFIG
│   ├── traffic_generator.py     # Bộ sinh Poisson 1-second buckets đồng bộ Seed + Spawner On/Off + Live Injection
│   └── env.py                   # TrafficLightEnv chuẩn Gymnasium (FIFO deque, tách 3s Vàng/5s Xanh, Semi-MDP reward)
├── agents/
│   ├── base_agent.py            # BaseAgent + on_episode_end() + export_onnx() + truncated flag
│   ├── baselines.py             # FixedTimeAgent (30s), WebsterAgent, MaxPressureAgent (theta_diff=3)
│   ├── q_learning.py            # StateDiscretizer (972 states) & Masked QLearningAgent (random tie-breaking)
│   ├── dqn_agent.py             # QNetwork, ReplayBuffer, DQNAgent (DQN / Double DQN + ONNX)
│   └── ppo_agent.py             # ActorCriticNetwork, RolloutBuffer, PPOAgent (Masked PPO + SmoothL1 + GAE + ONNX)
├── training/
│   ├── train.py                 # CLI huấn luyện thống nhất + Deterministic Eval Suite + TensorBoard + .pth/.onnx
│   └── train_all.py             # Script huấn luyện tuần tự cả 4 thuật toán trên cùng kịch bản
├── ui/
│   ├── visual_entities.py       # VisualIntersection, SensorOverlay, TrafficLightWidget, VisualVehicle
│   ├── traffic_renderer.py      # Bộ kết xuất Isometric cho MDP Interactive Inspector (demo_env.py)
│   └── studio_app.py            # Pygame 1360x760 @ 60 FPS (Split A/B đồng bộ sim_time + ONNX/PyTorch + Sliders)
├── experiments/
│   └── benchmark.py             # Đánh giá 6+ thuật toán x 4 kịch bản x 20 seeds, xuất CSV & 5 biểu đồ 300 DPI
├── models/                      # Trọng số đã huấn luyện (.pth, .onnx)
├── runs/                        # Nhật ký huấn luyện TensorBoard & history.csv (bỏ qua trong Git)
└── results/                     # Kết quả benchmark (.csv) & 5 biểu đồ (.png)
```

> **Phân biệt 2 file giao diện:**
> - `python ui/studio_app.py`: **Sản phẩm chính** (*SmartTraffic RL Studio*), đối sánh Split-Screen A/B (Baseline vs AI PPO/DDQN/Q-Learning), nạp `.pth` hoặc `.onnx`, Live Injection và thống kê trượt 1 phút.
> - `python demo_env.py`: **MDP Interactive Inspector** (*Thực hành 3*), chạy độc lập môi trường `TrafficLightEnv` không có AI để kiểm tra vector 10 chiều $s_t$, `action_mask` và các thành phần hàm reward bằng phím `[SPACE]` / `[K]` / `[A]`.

---

## 🚀 3. Hướng dẫn Cài đặt & Chạy Hệ thống

### 3.1. Kích hoạt môi trường & Kiểm thử Hợp đồng
```powershell
cd f:\Smart_Traffic\smart-traffic-rl
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python verify_contract.py
```

### 3.2. Huấn luyện Chính thức cả 4 Mô hình RL (500 Episodes)
```powershell
# Huấn luyện tuần tự cả 4 thuật toán trên kịch bản MIXED (tổng quát hóa SC1-SC4 + xuất ONNX)
python -m training.train_all --scenario MIXED --episodes 500 --seed 42

# Hoặc huấn luyện riêng lẻ từng thuật toán
python -m training.train --agent ppo --scenario MIXED --episodes 500 --seed 42 --export-onnx

# Xem biểu đồ huấn luyện trực quan trên TensorBoard
tensorboard --logdir runs
```

### 3.3. Chạy Giao diện Sản phẩm Trực quan (`1360x760 @ 60 FPS`)
```powershell
python ui/studio_app.py
```
- **Chế độ hiển thị:** **Split A/B** (đồng bộ đồng hồ `sim_time` và luồng xe 100% giữa Baseline & AI) hoặc **Single View** (`[V]`).
- **Tương tác nóng:** Checkbox **`[x] Dùng AI`** (`[I]`), **`Eng: PT / ONNX`** (chuyển đổi trực tiếp suy luận PyTorch `.pth` ↔ ONNXRuntime `.onnx`), **`SpawnerCar: ON/OFF`** (`[S]`), **`ResetCar`** (`[R]`), **`Ẩn/Hiện TT`** (`[H]`), 4 thanh trượt **Live Injection ($\lambda_i$)**, kịch bản `[1]–[4]`, Tạm dừng `[SPACE]`.

### 3.4. Chạy Benchmark Định lượng (4 Kịch bản x 20 Seeds) & Sinh 5 Biểu đồ Báo cáo
```powershell
python -m experiments.benchmark --model-tag MIXED --seeds 20 --scenarios SC1_LOW SC2_HIGH SC3_ASYMMETRIC SC4_DYNAMIC_SURGE
```
Kết quả lưu tại `results/`:
- `benchmark_raw.csv` & `benchmark_summary.csv`
- `fig1_avg_wait_comparison.png`
- `fig2_max_wait_starvation.png`
- `fig3_throughput_comparison.png`
- `fig4_training_curves.png`
- `fig5_dqn_vs_ddqn_qvalue.png`
