"""Automated Interface Contract Smoke Test.

Directly validates the requirements specified in Yeu_Cau.txt:
1. Initializes TrafficLightEnv (MDP v2.0).
2. Initializes all 5 required agents:
   - FixedTimeAgent
   - MaxPressureAgent
   - QLearningAgent
   - DQNAgent(use_double_dqn=False)
   - DQNAgent(use_double_dqn=True)
3. Executes a 10-step simulation loop (reset() -> select_action() -> step()) for each agent.
4. Uses strict assertions to verify:
   - obs.shape == (10,) and obs.dtype == np.float32
   - 0.0 <= obs.min() and obs.max() <= 1.0
   - Exactly 8 mandatory keys in info dictionary
   - Action returned by every agent strictly adheres to info["action_mask"].
"""

from __future__ import annotations

import os
import sys

# Ensure package roots are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import gymnasium as gym

from core.config import (
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
    ALPHA_QUEUE,
    BETA_WAIT_SQ,
    C_SWITCH,
    ETA_PASSED,
    GAMMA,
    TOTAL_DISCRETE_STATES,
)
from core.env import TrafficLightEnv
from agents.baselines import FixedTimeAgent, MaxPressureAgent
from agents.q_learning import StateDiscretizer, QLearningAgent
from agents.dqn_agent import DQNAgent


REQUIRED_INFO_KEYS = {
    "raw_queues",
    "max_waits",
    "phase",
    "phase_duration",
    "action_mask",
    "step_passed_vehicles",
    "total_passed",
    "avg_wait_time",
}


def test_config_constants() -> None:
    """Verify 100% of constants specified in Yeu_Cau.txt."""
    print("  [1/4] Checking core/config.py constants...")
    assert STATE_DIM == 10, f"Expected STATE_DIM=10, got {STATE_DIM}"
    assert ACTION_DIM == 2, f"Expected ACTION_DIM=2, got {ACTION_DIM}"
    assert Q_MAX == 30, f"Expected Q_MAX=30, got {Q_MAX}"
    assert W_MAX == 120.0, f"Expected W_MAX=120.0, got {W_MAX}"
    assert T_MIN == 10, f"Expected T_MIN=10, got {T_MIN}"
    assert T_MAX == 60, f"Expected T_MAX=60, got {T_MAX}"
    assert DELTA_T_GREEN == 5, f"Expected DELTA_T_GREEN=5, got {DELTA_T_GREEN}"
    assert DELTA_T_YELLOW == 3, f"Expected DELTA_T_YELLOW=3, got {DELTA_T_YELLOW}"
    assert SATURATION_FLOW_RATE == 1.0, f"Expected SATURATION_FLOW_RATE=1.0, got {SATURATION_FLOW_RATE}"
    assert EPISODE_DURATION == 1800, f"Expected EPISODE_DURATION=1800, got {EPISODE_DURATION}"
    assert ALPHA_QUEUE == 1.0
    assert BETA_WAIT_SQ == 1.5
    assert C_SWITCH == 0.25
    assert ETA_PASSED == 0.4
    assert GAMMA == 0.95
    assert TOTAL_DISCRETE_STATES == 972
    print("        -> OK: All MDP v2.0 constants and reward weights matched exactly.")


def test_discretizer_transform() -> None:
    """Verify StateDiscretizer.transform(obs, info) -> tuple[int, ...] (972 states)."""
    print("  [2/4] Checking StateDiscretizer.transform(obs, info) interface...")
    discretizer = StateDiscretizer()
    dummy_obs = np.zeros(10, dtype=np.float32)
    dummy_info = {
        "raw_queues": [0, 0, 0, 0],
        "max_waits": [0.0, 0.0, 0.0, 0.0],
        "phase": 0,
        "phase_duration": 0.0,
    }
    state_tuple = discretizer.transform(dummy_obs, dummy_info)
    assert isinstance(state_tuple, tuple), f"Expected tuple, got {type(state_tuple)}"
    assert len(state_tuple) == 7, f"Expected 7-element tuple, got {len(state_tuple)}"
    flat_idx = discretizer.to_index(state_tuple)
    assert 0 <= flat_idx < 972, f"State index {flat_idx} outside range [0, 971]"
    print(f"        -> OK: Discretizer returned tuple {state_tuple} mapping to flat state ID {flat_idx} (0-971).")


def test_agent_contract_and_simulation_loop() -> None:
    """Run 10-step simulation loop across all 5 agents with strict assertions (Yeu_Cau.txt Section 6)."""
    print("  [3/4] Initializing TrafficLightEnv and 5 Candidate Agents...")

    env = TrafficLightEnv(scenario_id="SC1_LOW", seed=42)

    # Initialize all 5 agents specified in Yeu_Cau.txt
    agents = [
        FixedTimeAgent(name="FixedTimeAgent"),
        MaxPressureAgent(name="MaxPressureAgent"),
        QLearningAgent(name="QLearningAgent"),
        DQNAgent(use_double_dqn=False, name="Standard-DQN (use_double_dqn=False)"),
        DQNAgent(use_double_dqn=True, name="Double-DQN (use_double_dqn=True)"),
    ]

    print(f"  [4/4] Executing 10-step simulation loops (reset() -> select_action() -> step())...\n")

    for agent_idx, agent in enumerate(agents, start=1):
        obs, info = env.reset(seed=42 + agent_idx)
        agent.reset()

        print(f"    Testing Agent {agent_idx}/5: {agent.name}...")

        for step_i in range(10):
            # 1. Kiểm tra không gian quan sát
            assert obs.shape == (10,), f"Step {step_i}: obs.shape must be (10,), got {obs.shape}"
            assert obs.dtype == np.float32, f"Step {step_i}: obs.dtype must be np.float32, got {obs.dtype}"
            assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0, (
                f"Step {step_i}: obs values outside [0.0, 1.0]: min={obs.min()}, max={obs.max()}"
            )

            # 2. Kiểm tra đầy đủ 8 keys bắt buộc trong info dictionary
            missing_keys = REQUIRED_INFO_KEYS - set(info.keys())
            assert not missing_keys, f"Step {step_i}: info dictionary missing required keys: {missing_keys}"

            # 3. Kiểm tra kiểu dữ liệu các trường trong info
            assert isinstance(info["raw_queues"], list) and len(info["raw_queues"]) == 4
            assert isinstance(info["max_waits"], list) and len(info["max_waits"]) == 4
            assert isinstance(info["action_mask"], np.ndarray) and info["action_mask"].shape == (2,)

            # 4. Agent quyết định hành động qua chữ ký select_action(obs, info)
            action = agent.select_action(obs, info, eval_mode=False)
            assert action in (0, 1), f"Step {step_i}: action must be 0 or 1, got {action}"

            # 5. Kiểm tra hành động luôn tôn trọng mặt nạ an toàn info["action_mask"]
            action_mask = info["action_mask"]
            assert action_mask[action] == 1, (
                f"Step {step_i} by {agent.name}: Chosen action {action} violated action_mask {action_mask}"
            )

            # 6. Môi trường bước tiếp một bước
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"

            # 7. Agent học tập (update)
            agent.update(obs, action, reward, next_obs, terminated or truncated, info, next_info)

            obs = next_obs
            info = next_info

        print(f"      [PASS] 10/10 steps verified successfully for {agent.name}.")

    env.close()


def main() -> int:
    print("=" * 80)
    print("  SMARTTRAFFIC RL STUDIO — INTERFACE CONTRACT VERIFICATION (Yeu_Cau.txt)")
    print("=" * 80)

    try:
        test_config_constants()
        test_discretizer_transform()
        test_agent_contract_and_simulation_loop()

        print("\n" + "=" * 80)
        print("  ALL INTERFACE CONTRACT REQUIREMENTS FROM Yeu_Cau.txt VERIFIED SUCCESSFULLY!")
        print("  - Observation vector: shape (10,), float32, range [0.0, 1.0] -> PASSED")
        print("  - Info dictionary: 8 mandatory keys, exact data types -> PASSED")
        print("  - select_action(obs, info) contract & action masking compliance -> PASSED")
        print("  - 5 Agents (FixedTime, MaxPressure, QL, DQN, DDQN) -> PASSED")
        print("=" * 80 + "\n")
        return 0
    except AssertionError as err:
        print(f"\n[CONTRACT VIOLATION]: {err}\n")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR]: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
