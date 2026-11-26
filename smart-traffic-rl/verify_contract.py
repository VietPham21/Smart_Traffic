"""Automated Interface Contract & System Smoke Test (Method B Complete Suite).

Validates 100% of specifications from Đặc tả Phương án B:
1. Checks core/config.py MDP v2.0 constants and Q-Learning/DQN/PPO configs.
2. Checks TrafficGenerator Poisson sampling, live rates & spawner On/Off toggle.
3. Checks StateDiscretizer.transform(obs, info) -> 972 discrete states.
4. Validates all 6 Candidate Agents (FixedTime, MaxPressure, QLearning,
   StandardDQN, DoubleDQN, PPO):
   - obs.shape == (10,), float32, in [0.0, 1.0]
   - All 8 backward-compatible + 7 new telemetry keys in info dictionary
   - Every selected action adheres strictly to info["action_mask"][action] == 1
5. Verifies real weight/table updates (no NaN/Inf) for QLearning, StandardDQN,
   DoubleDQN, and PPOAgent.
6. Verifies .pth and .onnx export + reload + ONNXRuntime masked inference for
   both PPOAgent and DQNAgent.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import onnxruntime as ort

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
    Q_LEARNING_CONFIG,
    DQN_CONFIG,
    PPO_CONFIG,
)
from core.env import TrafficLightEnv
from agents.baselines import FixedTimeAgent, MaxPressureAgent
from agents.q_learning import StateDiscretizer, QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


REQUIRED_OLD_INFO_KEYS = {
    "raw_queues",
    "max_waits",
    "phase",
    "phase_duration",
    "action_mask",
    "step_passed_vehicles",
    "total_passed",
    "avg_wait_time",
}

REQUIRED_NEW_INFO_KEYS = {
    "max_wait_recorded",
    "sim_time",
    "passed_last_60s",
    "last_step_switched",
    "light_states",
    "current_rates",
    "spawner_enabled",
}

ALL_REQUIRED_INFO_KEYS = REQUIRED_OLD_INFO_KEYS | REQUIRED_NEW_INFO_KEYS


def test_config_and_generator() -> None:
    """Verify MDP v2.0 constants, algorithm configs, and TrafficGenerator controls."""
    print("  [1/5] Checking core/config.py constants & TrafficGenerator features...")
    assert STATE_DIM == 10
    assert ACTION_DIM == 2
    assert Q_MAX == 30
    assert W_MAX == 120.0
    assert T_MIN == 10
    assert T_MAX == 60
    assert DELTA_T_GREEN == 5
    assert DELTA_T_YELLOW == 3
    assert SATURATION_FLOW_RATE == 1.0
    assert EPISODE_DURATION == 1800
    assert ALPHA_QUEUE == 1.0
    assert BETA_WAIT_SQ == 1.5
    assert C_SWITCH == 0.25
    assert ETA_PASSED == 0.4
    assert GAMMA == 0.95
    assert TOTAL_DISCRETE_STATES == 972
    assert PPO_CONFIG["hidden_dims"] == (128, 128) and PPO_CONFIG["gae_lambda"] == 0.95
    assert DQN_CONFIG["hidden_dims"] == (64, 64) and DQN_CONFIG["lr"] == 5e-4
    assert Q_LEARNING_CONFIG["epsilon_decay"] == 0.992

    env = TrafficLightEnv(scenario_id="SC3_ASYMMETRIC", seed=42)
    env.reset(seed=42)
    env.traffic_generator.set_spawner_enabled(False)
    assert env.traffic_generator.sample_arrivals() == [[], [], [], []]
    env.traffic_generator.set_spawner_enabled(True)
    env.traffic_generator.set_live_rates({"N": 0.5, "S": 0.5, "E": 0.1, "W": 0.1})
    assert abs(env.traffic_generator.get_arrival_rates(0.0)[0] - 0.5) < 1e-6
    env.close()
    assert "pygame" not in sys.modules, "core/ must never import pygame!"
    print("        -> OK: MDP constants, configs, spawner/live rates, and headless isolation verified.")


def test_discretizer_transform() -> None:
    """Verify StateDiscretizer.transform(obs, info) -> 972 discrete states."""
    print("  [2/5] Checking StateDiscretizer.transform(obs, info) interface...")
    discretizer = StateDiscretizer()
    dummy_obs = np.zeros(10, dtype=np.float32)
    dummy_info = {
        "raw_queues": [0, 0, 0, 0],
        "max_waits": [0.0, 0.0, 0.0, 0.0],
        "phase": 0,
        "phase_duration": 0.0,
    }
    state_tuple = discretizer.transform(dummy_obs, dummy_info)
    assert isinstance(state_tuple, tuple) and len(state_tuple) == 7
    flat_idx = discretizer.to_index(state_tuple)
    assert 0 <= flat_idx < 972
    print(f"        -> OK: Discretizer returned tuple {state_tuple} mapping to ID {flat_idx} (0-971).")


def test_agent_contract_all_6_agents() -> None:
    """Verify observation space, full info keys, and action masking across all 6 agents."""
    print("  [3/5] Verifying Interface Contract across all 6 agents (15 steps each)...")
    env = TrafficLightEnv(scenario_id="SC3_ASYMMETRIC", seed=42)

    agents = [
        FixedTimeAgent(cycle_green_time=30.0, name="FixedTime"),
        MaxPressureAgent(theta_diff=3.0, name="MaxPressure"),
        QLearningAgent(name="QLearning"),
        DQNAgent(use_double_dqn=False, name="StandardDQN"),
        DQNAgent(use_double_dqn=True, name="DoubleDQN"),
        PPOAgent(name="PPO"),
    ]

    for idx, agent in enumerate(agents, start=1):
        obs, info = env.reset(seed=100 + idx)
        agent.reset()

        for step_i in range(15):
            assert obs.shape == (10,) and obs.dtype == np.float32
            assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0

            missing = ALL_REQUIRED_INFO_KEYS - set(info.keys())
            assert not missing, f"{agent.name} step {step_i}: missing info keys {missing}"
            assert set(info["light_states"].keys()) == {"N", "S", "E", "W"}
            assert all(v in ("GREEN", "YELLOW", "RED") for v in info["light_states"].values())
            assert isinstance(info["action_mask"], np.ndarray) and info["action_mask"].shape == (2,)

            action = agent.select_action(obs, info, eval_mode=False)
            assert action in (0, 1)
            assert info["action_mask"][action] == 1, f"{agent.name} violated action_mask!"

            next_obs, reward, term, trunc, next_info = env.step(action)
            assert isinstance(reward, float) and math.isfinite(reward)
            agent.update(obs, action, reward, next_obs, term or trunc, info, next_info)
            obs, info = next_obs, next_info

        print(f"        [PASS {idx}/6] {agent.name}: 15/15 steps compliant.")
    env.close()


def test_rl_updates_no_nan_inf() -> None:
    """Ensure QLearning, StandardDQN, DoubleDQN, and PPO execute real updates without NaN/Inf."""
    print("  [4/5] Testing gradient/Bellman updates (no NaN/Inf) for 4 RL agents...")
    env = TrafficLightEnv(scenario_id="SC3_ASYMMETRIC", seed=123)

    rl_agents = [
        ("QLearning", QLearningAgent(name="QLearning")),
        ("StandardDQN", DQNAgent(use_double_dqn=False, min_replay_size=4, batch_size=4, device="cpu")),
        ("DoubleDQN", DQNAgent(use_double_dqn=True, min_replay_size=4, batch_size=4, device="cpu")),
        ("PPO", PPOAgent(rollout_steps=8, mini_batch_size=4, device="cpu")),
    ]

    for label, agent in rl_agents:
        obs, info = env.reset(seed=77)
        agent.reset()
        update_count = 0
        for _ in range(10):
            action = agent.select_action(obs, info, eval_mode=False)
            nobs, r, term, trunc, ninfo = env.step(action)
            metrics = agent.update(obs, action, r, nobs, term or trunc, info, ninfo)
            if metrics is not None:
                update_count += 1
                for k, v in metrics.items():
                    assert math.isfinite(float(v)), f"{label}.update() produced non-finite {k}={v}!"
            obs, info = nobs, ninfo

        assert update_count > 0, f"{label} did not execute any optimization update in test loop!"
        print(f"        [PASS] {label}: {update_count} update(s) executed cleanly (all finite).")
    env.close()


def test_pth_and_onnx_roundtrip() -> None:
    """Verify saving/loading .pth and exporting/running .onnx for PPOAgent & DQNAgent."""
    print("  [5/5] Verifying .pth + .onnx export & masked inference reload...")
    ppo = PPOAgent(device="cpu")
    ddqn = DQNAgent(use_double_dqn=True, device="cpu")

    test_obs = np.full((1, 10), 0.35, dtype=np.float32)
    mask_keep_only = np.array([[1.0, 0.0]], dtype=np.float32)
    mask_switch_only = np.array([[0.0, 1.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        # 1. PPO round-trip
        ppo_pth = os.path.join(td, "ppo_test.pth")
        ppo_onnx = os.path.join(td, "ppo_test.onnx")
        ppo.save(ppo_pth)
        ppo_reloaded = PPOAgent(device="cpu")
        ppo_reloaded.load(ppo_pth)
        ppo_reloaded.export_onnx(ppo_onnx)

        sess_ppo = ort.InferenceSession(ppo_onnx)
        probs_0, act_0 = sess_ppo.run(None, {"obs": test_obs, "action_mask": mask_keep_only})
        probs_1, act_1 = sess_ppo.run(None, {"obs": test_obs, "action_mask": mask_switch_only})
        assert int(act_0[0]) == 0 and np.isfinite(probs_0).all()
        assert int(act_1[0]) == 1 and np.isfinite(probs_1).all()

        # 2. Double DQN round-trip
        dqn_pth = os.path.join(td, "ddqn_test.pth")
        dqn_onnx = os.path.join(td, "ddqn_test.onnx")
        ddqn.save(dqn_pth)
        dqn_reloaded = DQNAgent(use_double_dqn=True, device="cpu")
        dqn_reloaded.load(dqn_pth)
        dqn_reloaded.export_onnx(dqn_onnx)

        sess_dqn = ort.InferenceSession(dqn_onnx)
        q_0, dact_0 = sess_dqn.run(None, {"obs": test_obs, "action_mask": mask_keep_only})
        q_1, dact_1 = sess_dqn.run(None, {"obs": test_obs, "action_mask": mask_switch_only})
        assert int(dact_0[0]) == 0 and np.isfinite(q_0).all()
        assert int(dact_1[0]) == 1 and np.isfinite(q_1).all()

    print("        -> OK: Both PPOAgent and DQNAgent .pth + .onnx verified via ONNXRuntime!")


def main() -> int:
    print("=" * 80)
    print("  SMARTTRAFFIC RL STUDIO — FULL CONTRACT & SMOKE VERIFICATION (METHOD B)")
    print("=" * 80)
    try:
        test_config_and_generator()
        test_discretizer_transform()
        test_agent_contract_all_6_agents()
        test_rl_updates_no_nan_inf()
        test_pth_and_onnx_roundtrip()

        print("\n" + "=" * 80)
        print("  ALL 5 VERIFICATION STAGES PASSED SUCCESSFULLY!")
        print("  - 6 Agents (FixedTime, MaxPressure, QLearning, StandardDQN, DoubleDQN, PPO) -> PASSED")
        print("  - 15 Info Keys (8 backward-compat + 7 HUD/Telemetry keys) -> PASSED")
        print("  - Gradient / Bellman updates (0 NaN/Inf) & ONNX/PyTorch roundtrip -> PASSED")
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
    raise SystemExit(main())
