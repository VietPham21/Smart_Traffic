# SmartTraffic RL Studio (Phương án B) 🚦🤖

> **Nền tảng Mô phỏng, Huấn luyện & Đối sánh Đa tầng Điều khiển Tín hiệu Đèn Giao thông Thông minh**  
> Môn học: *Học Tăng Cường và Ứng Dụng* — Khoa Công nghệ Thông tin, Đại học Thủy Lợi (Nhóm 3).

---

## 📌 1. Giới thiệu Dự án

**SmartTraffic RL Studio** giải quyết bài toán tối ưu hóa điều khiển tín hiệu đèn giao thông ngã tư đơn 4 hướng (Bắc, Nam, Đông, Tây) trên mô hình toán học **MDP v2.0** tuân thủ chuẩn **Gymnasium API**, kết hợp **Action Masking** ($T_{\min}=10\text{s}, T_{\max}=60\text{s}$, 3s đệm đèn vàng) và hỗ trợ đầy đủ cả 2 nhánh Học tăng cường lớn:
- **Value-Based:** Tabular **Q-Learning** (972 states), **Standard DQN**, **Double DQN** (PyTorch MLP `10->64->64->2`)
- **Policy-Based / Actor-Critic:** Masked **PPO-Clip** + **GAE** ($\gamma=0.95, \lambda=0.95$, PyTorch MLP `10->128->128`)
- **Baselines:** **Fixed-Time** (30s), **Webster Optimal**, **Actuated Max-Pressure** ($\theta_{\text{diff}}=3$)

---

## 🏗️ 2. Cấu trúc Thư mục Dự án

```text
smart-traffic-rl/
├── requirements.txt             # gymnasium, numpy, torch, onnx, onnxruntime, tensorboard, pygame, pandas, matplotlib, seaborn
├── README.md                    # Tài liệu kiến trúc & hướng dẫn sử dụng
├── verify_contract.py           # Kiểm thử tự động 6 Agents, 15 Info Keys, GAE/DQN Update & ONNX
├── core/
│   ├── config.py                # Hằng số MDP v2.0, 4 kịch bản SC1-SC4, Q_LEARNING/DQN/PPO_CONFIG
│   ├── traffic_generator.py     # Bộ sinh Poisson đồng bộ Seed + Spawner On/Off + Live Injection
│   └── env.py                   # TrafficLightEnv chuẩn Gymnasium (FIFO wait, Sliding Window 60s, Light States)
├── agents/
│   ├── base_agent.py            # BaseAgent + on_episode_end() + export_onnx()
│   ├── baselines.py             # FixedTimeAgent (30s), WebsterAgent, MaxPressureAgent (theta_diff=3)
│   ├── q_learning.py            # StateDiscretizer (972 states) & Masked QLearningAgent
│   ├── dqn_agent.py             # QNetwork, ReplayBuffer, DQNAgent (DQN / Double DQN + ONNX)
│   └── ppo_agent.py             # ActorCriticNetwork, RolloutBuffer, PPOAgent (Masked PPO + GAE + ONNX)
├── training/
│   └── train.py                 # CLI huấn luyện thống nhất + TensorBoard Logger + Save Best .pth/.onnx
├── ui/
│   ├── visual_entities.py       # VisualIntersection, SensorOverlay, TrafficLightWidget, VisualVehicle
│   └── studio_app.py            # Pygame 1360x760 @ 60 FPS (Split A/B + Single Toggle AI + Live Sliders)
├── experiments/
│   └── benchmark.py             # Đánh giá 6 thuật toán x 4 kịch bản x N seeds, xuất CSV & 4 biểu đồ 300 DPI
├── models/                      # Trọng số đã huấn luyện (.pth, .onnx)
├── runs/                        # Nhật ký huấn luyện TensorBoard
└── results/                     # Kết quả benchmark (.csv) & 4 biểu đồ (.png)
```

---

## 🚀 3. Hướng dẫn Cài đặt & Chạy Hệ thống

### 3.1. Kích hoạt môi trường & Cài đặt thư viện
```powershell
cd f:\Smart_Traffic\smart-traffic-rl
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.2. Kiểm thử Hợp đồng Toàn hệ thống (Smoke Test)
```powershell
python verify_contract.py
```

### 3.3. Huấn luyện Mô hình RL & Ghi log TensorBoard
Hỗ trợ `--scenario`: `SC1_LOW`, `SC2_HIGH`, `SC3_ASYMMETRIC`, `SC4_DYNAMIC_SURGE`, và `MIXED`.
```powershell
python -m training.train --agent ppo --scenario SC3_ASYMMETRIC --episodes 500 --seed 42 --export-onnx
python -m training.train --agent ddqn --scenario SC3_ASYMMETRIC --episodes 500 --seed 42 --export-onnx
python -m training.train --agent dqn --scenario SC3_ASYMMETRIC --episodes 500 --seed 42
python -m training.train --agent q_learning --scenario SC3_ASYMMETRIC --episodes 500 --seed 42

# Xem biểu đồ huấn luyện trực quan trên TensorBoard
tensorboard --logdir runs
```

### 3.4. Chạy Giao diện Sản phẩm Trực quan (`1360x760 @ 60 FPS`)
```powershell
python ui/studio_app.py
```
- **Chế độ hiển thị:** Nút **Split A/B** (so sánh song song Baseline vs AI) hoặc **Single View** (`[V]`).
- **Tương tác nóng:** Checkbox **`[x] Dùng AI`** (`[I]`), **`SpawnerCar: ON/OFF`** (`[S]`), **`ResetCar`** (`[R]`), **`Ẩn/Hiện TT`** (`[H]`), 4 thanh trượt **Live Injection ($\lambda_i$)**, kịch bản `[1]–[4]`, Tạm dừng `[SPACE]`.

### 3.5. Chạy Benchmark Định lượng & Sinh Biểu đồ Báo cáo
```powershell
python -m experiments.benchmark --seeds 20 --scenarios SC1_LOW SC2_HIGH SC3_ASYMMETRIC SC4_DYNAMIC_SURGE
```
Kết quả được lưu tại `results/`:
- `benchmark_raw.csv` & `benchmark_summary.csv`
- `fig1_avg_wait_comparison.png`
- `fig2_max_wait_starvation.png`
- `fig3_throughput_comparison.png`
- `fig4_training_curves.png`
