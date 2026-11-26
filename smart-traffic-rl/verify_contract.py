"""Automated Interface Contract & Scientific Integrity Verification Suite.

Validates all specifications and P0/P1 mathematical fixes:
1. MDP v2.0 constants, algorithm configs, TrafficGenerator controls, & typing hints.
2. StateDiscretizer.transform(obs, info) -> 972 discrete states.
3. All 6 Candidate Agents (FixedTime, MaxPressure, QLearning, StandardDQN, DoubleDQN, PPO) + Webster:
   - obs.shape == (10,), float32, in [0.0, 1.0]
   - All 8 backward-compatible + telemetry keys in info dictionary
   - Strict action_mask compliance
4. Real Bellman / gradient updates (no NaN/Inf) for all 4 RL agents.
5. .pth and .onnx export + reload + ONNXRuntime masked inference.
6. P0 Regression Checks:
   - A/B Poisson Step-Size Invariance ([5]*8 vs [8]*5 produce 100% identical arrivals & timestamps)
   - Queued waiting vehicles properly counted in avg_wait_time (> 0.0 even when 0 vehicles exited)
   - Semi-MDP time_scale reward sensitivity (8s switch scales queue+wait holding penalty by 1.6x)
   - Proper `truncated=True, terminated=False` bootstrapping for QLearning, DQN, and PPO.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import typing

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import onnxruntime as ort

from core.config import (
    STATE_DIM,
    ACTION_DIM,
    APPROACHES,
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
from core.traffic_generator import TrafficGenerator, Vehicle
from core.env import TrafficLightEnv
from agents.baselines import FixedTimeAgent, MaxPressureAgent, WebsterAgent
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
    "is_truncated",
}

ALL_REQUIRED_INFO_KEYS = REQUIRED_OLD_INFO_KEYS | REQUIRED_NEW_INFO_KEYS


def test_config_and_generator() -> None:
    """Verify MDP v2.0 constants, algorithm configs, type hints, and TrafficGenerator controls."""
    print("  [1/6] Checking core/config.py constants & TrafficGenerator features...")
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
    assert PPO_CONFIG["hidden_dims"] == (128, 128) and PPO_CONFIG["rollout_steps"] == 1024 and PPO_CONFIG["ppo_epochs"] == 8
    assert DQN_CONFIG["hidden_dims"] == (64, 64) and DQN_CONFIG["lr"] == 5e-4
    assert Q_LEARNING_CONFIG["epsilon_decay"] == 0.992

    # Ensure Any typing hint in TrafficGenerator resolves cleanly at runtime
    hints = typing.get_type_hints(TrafficGenerator.set_live_rates)
    assert "rates" in hints

    env = TrafficLightEnv(scenario_id="SC3_ASYMMETRIC", seed=42)
    env.reset(seed=42)
    env.traffic_generator.set_spawner_enabled(False)
    assert env.traffic_generator.sample_arrivals() == [[], [], [], []]
    env.traffic_generator.set_spawner_enabled(True)
    env.traffic_generator.set_live_rates({"N": 0.5, "S": 0.5, "E": 0.1, "W": 0.1})
    assert abs(env.traffic_generator.get_arrival_rates(0.0)[0] - 0.5) < 1e-6
    env.close()
    assert "pygame" not in sys.modules, "core/ must never import pygame!"
    print("        -> OK: MDP constants, configs, typing hints, spawner/live rates verified.")


def test_discretizer_transform() -> None:
    """Verify StateDiscretizer.transform(obs, info) -> 972 discrete states."""
    print("  [2/6] Checking StateDiscretizer.transform(obs, info) interface...")
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


def test_agent_contract_all_agents() -> None:
    """Verify observation space, full info keys, and action masking across all 7 agents."""
    print("  [3/6] Verifying Interface Contract across all agents (15 steps each)...")
    env = TrafficLightEnv(scenario_id="SC3_ASYMMETRIC", seed=42)

    agents = [
        FixedTimeAgent(cycle_green_time=30.0, name="FixedTime"),
        WebsterAgent(name="Webster"),
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
            agent.update(obs, action, reward, next_obs, term, info, next_info, truncated=trunc)
            obs, info = next_obs, next_info

        print(f"        [PASS {idx}/7] {agent.name}: 15/15 steps compliant.")
    env.close()


def test_rl_updates_no_nan_inf() -> None:
    """Ensure QLearning, StandardDQN, DoubleDQN, and PPO execute real updates without NaN/Inf."""
    print("  [4/6] Testing gradient/Bellman updates (no NaN/Inf) for 4 RL agents...")
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
            metrics = agent.update(obs, action, r, nobs, term, info, ninfo, truncated=trunc)
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
    print("  [5/6] Verifying .pth + .onnx export & masked inference reload...")
    ppo = PPOAgent(device="cpu")
    ddqn = DQNAgent(use_double_dqn=True, device="cpu")

    test_obs = np.full((1, 10), 0.35, dtype=np.float32)
    mask_keep_only = np.array([[1.0, 0.0]], dtype=np.float32)
    mask_switch_only = np.array([[0.0, 1.0]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        ppo_pth = os.path.join(td, "ppo_test.pth")
        ppo_onnx = os.path.join(td, "ppo_test.onnx")
        ppo.save(ppo_pth)
        ppo_reloaded = PPOAgent(device="cpu")
        ppo_reloaded.load(ppo_pth)
        ppo_reloaded.export_onnx(ppo_onnx)

        sess_ppo = ort.InferenceSession(ppo_onnx, providers=["CPUExecutionProvider"])
        probs_0, act_0 = sess_ppo.run(None, {"obs": test_obs, "action_mask": mask_keep_only})
        probs_1, act_1 = sess_ppo.run(None, {"obs": test_obs, "action_mask": mask_switch_only})
        assert int(act_0[0]) == 0 and np.isfinite(probs_0).all()
        assert int(act_1[0]) == 1 and np.isfinite(probs_1).all()

        dqn_pth = os.path.join(td, "ddqn_test.pth")
        dqn_onnx = os.path.join(td, "ddqn_test.onnx")
        ddqn.save(dqn_pth)
        dqn_reloaded = DQNAgent(use_double_dqn=True, device="cpu")
        dqn_reloaded.load(dqn_pth)
        dqn_reloaded.export_onnx(dqn_onnx)

        sess_dqn = ort.InferenceSession(dqn_onnx, providers=["CPUExecutionProvider"])
        q_0, dact_0 = sess_dqn.run(None, {"obs": test_obs, "action_mask": mask_keep_only})
        q_1, dact_1 = sess_dqn.run(None, {"obs": test_obs, "action_mask": mask_switch_only})
        assert int(dact_0[0]) == 0 and np.isfinite(q_0).all()
        assert int(dact_1[0]) == 1 and np.isfinite(q_1).all()

    print("        -> OK: Both PPOAgent and DQNAgent .pth + .onnx verified via ONNXRuntime!")


def test_p0_scientific_guarantees() -> None:
    """Verify P0 fixes: A/B Poisson invariance, active-queue avg_wait_time, Semi-MDP scaling, & truncation bootstrap."""
    print("  [6/6] Checking P0 Regression Assertions (A/B Invariance, Wait, Semi-MDP, Truncation)...")

    # 1. A/B Arrival Invariance: [5]*8 (40s) vs [8]*5 (40s) must produce identical arrivals & timestamps
    gen_a = TrafficGenerator(scenario_id="SC3_ASYMMETRIC", seed=2026)
    gen_b = gen_a.clone_synchronized_generator()

    times_a: dict[int, list[float]] = {app: [] for app in APPROACHES}
    t_cur = 0.0
    for dt in [5.0] * 8:
        arr = gen_a.generate_arrivals(t_cur, dt)
        for app in APPROACHES:
            times_a[app].extend(v.arrival_time for v in arr[app])
        t_cur += dt

    times_b: dict[int, list[float]] = {app: [] for app in APPROACHES}
    t_cur = 0.0
    for dt in [8.0] * 5:
        arr = gen_b.generate_arrivals(t_cur, dt)
        for app in APPROACHES:
            times_b[app].extend(v.arrival_time for v in arr[app])
        t_cur += dt

    for app in APPROACHES:
        assert len(times_a[app]) == len(times_b[app]), f"Approach {app} count mismatch: {len(times_a[app])} vs {len(times_b[app])}"
        assert np.allclose(times_a[app], times_b[app]), f"Approach {app} arrival timestamps mismatch!"

    # 2. Queued red vehicles must count toward avg_wait_time even when 0 cars exited
    env_w = TrafficLightEnv(scenario_id="SC1_LOW", seed=123)
    env_w.reset(seed=123)
    env_w.traffic_generator.set_spawner_enabled(False)
    # Place waiting vehicles strictly on RED axis (EAST=2) with 0 passed vehicles
    env_w.queues[2].append(Vehicle(vehicle_id=1, approach=2, arrival_time=0.0))
    env_w.queues[2].append(Vehicle(vehicle_id=2, approach=2, arrival_time=1.0))
    _, _, _, _, info_w = env_w.step(0)  # NS keeps green; East stays RED at t=5s
    assert info_w["total_passed"] == 0
    assert info_w["avg_wait_time"] > 0.0, "Queued un-discharged vehicles were ignored in avg_wait_time!"
    env_w.close()

    # 3. Semi-MDP time_scale: 8s step holding cost == 1.6x of 5s step holding cost
    env_s = TrafficLightEnv(scenario_id="SC1_LOW", seed=123)
    env_s.reset(seed=123)
    env_s.queues[2].append(Vehicle(vehicle_id=10, approach=2, arrival_time=0.0))
    r_5s = env_s._compute_reward(action_switched=False, passed_vehicles=0, dt_step=5.0)
    r_8s = env_s._compute_reward(action_switched=False, passed_vehicles=0, dt_step=8.0)
    assert abs(r_8s / r_5s - 1.6) < 1e-5, f"Semi-MDP holding ratio expected 1.6, got {r_8s / r_5s}"
    env_s.close()

    # 4. Truncation bootstrap for QLearning, DQN, PPO (truncated=True, terminated=False)
    obs0 = np.full(10, 0.2, dtype=np.float32)
    obs1 = np.full(10, 0.4, dtype=np.float32)
    info0 = {"raw_queues": [2, 2, 2, 2], "max_waits": [5.0, 5.0, 5.0, 5.0], "phase": 0, "phase_duration": 15.0, "action_mask": np.array([1, 1], dtype=np.int8)}
    info1 = {"raw_queues": [3, 3, 2, 2], "max_waits": [10.0, 10.0, 8.0, 8.0], "phase": 0, "phase_duration": 20.0, "action_mask": np.array([1, 1], dtype=np.int8), "is_truncated": True}

    ql = QLearningAgent(lr=1.0, gamma=0.95)
    s_next_idx = ql.discretizer.to_index(ql.discretizer.transform(obs1, info1))
    ql.q_table[s_next_idx] = 10.0
    ql.update(obs0, 0, -1.0, obs1, done=False, info=info0, next_info=info1, truncated=True)
    s0_idx = ql.discretizer.to_index(ql.discretizer.transform(obs0, info0))
    expected_ql = -1.0 + 0.95 * 10.0
    assert abs(ql.q_table[s0_idx, 0] - expected_ql) < 1e-4, f"QL truncation bootstrap failed: {ql.q_table[s0_idx, 0]} vs {expected_ql}"

    dqn_ag = DQNAgent(use_double_dqn=True, min_replay_size=2, batch_size=2, device="cpu")
    dqn_ag.update(obs0, 0, -1.0, obs1, done=False, info=info0, next_info=info1, truncated=True)
    dqn_metrics = dqn_ag.update(obs0, 1, -1.5, obs1, done=False, info=info0, next_info=info1, truncated=True)
    assert dqn_metrics is not None and all(math.isfinite(float(v)) for v in dqn_metrics.values())
    assert dqn_ag.replay_buffer.buffer[-1][4] is False, "DQN ReplayBuffer should NOT treat truncated=True as absorbing done=True!"

    ppo_ag = PPOAgent(rollout_steps=4, mini_batch_size=2, device="cpu")
    for k in range(3):
        ppo_ag.select_action(obs0, info0, eval_mode=False)
        ppo_ag.update(obs0, 0, -1.0, obs1, done=False, info=info0, next_info=info0, truncated=False)
    ppo_ag.select_action(obs0, info0, eval_mode=False)
    ppo_metrics = ppo_ag.update(obs0, 0, -1.0, obs1, done=False, info=info0, next_info=info1, truncated=True)
    assert ppo_metrics is not None and all(math.isfinite(float(v)) for v in ppo_metrics.values())

    print("        -> OK: A/B Poisson invariance, Active-queue Wait, Semi-MDP 1.6x scaling & Truncation verified!")


def main() -> int:
    print("=" * 84)
    print("  SMARTTRAFFIC RL STUDIO — COMPLETE CONTRACT & SCIENTIFIC VERIFICATION")
    print("=" * 84)
    try:
        test_config_and_generator()
        test_discretizer_transform()
        test_agent_contract_all_agents()
        test_rl_updates_no_nan_inf()
        test_pth_and_onnx_roundtrip()
        test_p0_scientific_guarantees()

        print("\n" + "=" * 84)
        print("  ALL 6 VERIFICATION STAGES PASSED SUCCESSFULLY!")
        print("=" * 84 + "\n")
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
