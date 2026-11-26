"""Unified RL Training & TensorBoard Logging CLI for SmartTraffic RL Studio.

Supports all 4 trainable RL agents:
- q_learning : Tabular Q-Learning with 972 discrete states
- dqn        : Standard Deep Q-Network (10->64->64->2)
- ddqn       : Double Deep Q-Network (10->64->64->2)
- ppo        : Masked Proximal Policy Optimization with GAE (10->128->128)

Supports 4 benchmark scenarios (SC1_LOW - SC4_DYNAMIC_SURGE) plus MIXED mode
which randomly rotates across all 4 scenarios per episode for generalization.
Logs diagnostics to TensorBoard (runs/) and saves best + final checkpoints
to models/ (with optional .onnx inference graph export).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from typing import Any, Optional

# Ensure project root is on sys.path when executed as module or script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.config import (
    DQN_CONFIG,
    MAX_STEPS,
    PPO_CONFIG,
    Q_LEARNING_CONFIG,
    SCENARIOS,
)
from core.env import TrafficLightEnv
from agents.base_agent import BaseAgent
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


STANDARD_SCENARIOS = ("SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE")

SCENARIO_ALIASES = {
    "SC1": "SC1_LOW",
    "SC1_LOW": "SC1_LOW",
    "SC2": "SC2_HIGH",
    "SC2_HIGH": "SC2_HIGH",
    "SC3": "SC3_ASYMMETRIC",
    "SC3_ASYMMETRIC": "SC3_ASYMMETRIC",
    "SC4": "SC4_DYNAMIC_SURGE",
    "SC4_DYNAMIC_SURGE": "SC4_DYNAMIC_SURGE",
    "MIXED": "MIXED",
}


def _create_agent(
    agent_key: str,
    device: Optional[str] = None,
    min_replay_override: Optional[int] = None,
) -> BaseAgent:
    """Instantiate RL agent parameterized from unified config blocks."""
    key = agent_key.lower()
    if key == "q_learning":
        return QLearningAgent(
            lr=float(Q_LEARNING_CONFIG["alpha_lr"]),
            gamma=float(Q_LEARNING_CONFIG["gamma"]),
            epsilon_start=float(Q_LEARNING_CONFIG["epsilon_start"]),
            epsilon_min=float(Q_LEARNING_CONFIG["epsilon_min"]),
            epsilon_decay=float(Q_LEARNING_CONFIG["epsilon_decay"]),
            decay_on_episode_end=True,
            name="Q-Learning",
        )
    if key in ("dqn", "ddqn"):
        use_double = (key == "ddqn")
        extra_kwargs: dict[str, Any] = {}
        if min_replay_override is not None:
            extra_kwargs["min_replay_size"] = min_replay_override
        return DQNAgent(
            use_double_dqn=use_double,
            lr=float(DQN_CONFIG["lr"]),
            gamma=float(DQN_CONFIG["gamma"]),
            buffer_capacity=int(DQN_CONFIG["buffer_size"]),
            batch_size=int(DQN_CONFIG["batch_size"]),
            target_update_freq=int(DQN_CONFIG["target_update_freq"]),
            epsilon_start=float(DQN_CONFIG["epsilon_start"]),
            epsilon_min=float(DQN_CONFIG["epsilon_min"]),
            epsilon_decay=float(DQN_CONFIG["epsilon_decay"]),
            decay_on_episode_end=True,
            device=device,
            name="Double-DQN" if use_double else "Standard-DQN",
            **extra_kwargs,
        )
    if key == "ppo":
        return PPOAgent(
            hidden_dims=tuple(PPO_CONFIG["hidden_dims"]),
            lr=float(PPO_CONFIG["lr"]),
            gamma=float(PPO_CONFIG["gamma"]),
            gae_lambda=float(PPO_CONFIG["gae_lambda"]),
            clip_eps=float(PPO_CONFIG["clip_eps"]),
            entropy_coef=float(PPO_CONFIG["entropy_coef"]),
            value_coef=float(PPO_CONFIG["value_coef"]),
            max_grad_norm=float(PPO_CONFIG["max_grad_norm"]),
            ppo_epochs=int(PPO_CONFIG["ppo_epochs"]),
            mini_batch_size=int(PPO_CONFIG["mini_batch_size"]),
            rollout_steps=int(PPO_CONFIG["rollout_steps"]),
            device=device,
            name="PPO-Clip",
        )
    raise ValueError(f"Unsupported agent '{agent_key}'. Choose from: q_learning, dqn, ddqn, ppo")


def _log_update_metrics(
    writer: SummaryWriter,
    agent_key: str,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Write algorithm-specific diagnostic losses and metrics to TensorBoard."""
    if agent_key == "ppo":
        if "policy_loss" in metrics:
            writer.add_scalar("Losses/Policy_Loss", float(metrics["policy_loss"]), step)
        if "value_loss" in metrics:
            writer.add_scalar("Losses/Value_Loss", float(metrics["value_loss"]), step)
        if "entropy" in metrics:
            writer.add_scalar("Losses/Entropy", float(metrics["entropy"]), step)
    else:
        if "td_loss" in metrics:
            writer.add_scalar("Losses/TD_Loss", float(metrics["td_loss"]), step)
        elif "loss" in metrics:
            writer.add_scalar("Losses/TD_Loss", float(metrics["loss"]), step)
        if "mean_q" in metrics:
            writer.add_scalar("Metrics/Mean_Q_Value", float(metrics["mean_q"]), step)
        elif "q_val" in metrics:
            writer.add_scalar("Metrics/Mean_Q_Value", float(metrics["q_val"]), step)
        if "epsilon" in metrics:
            writer.add_scalar("Metrics/Epsilon", float(metrics["epsilon"]), step)


def train_agent(
    agent_key: str,
    scenario: str = "SC3_ASYMMETRIC",
    episodes: Optional[int] = None,
    seed: int = 42,
    export_onnx: bool = False,
    models_dir: str = os.path.join(PROJECT_ROOT, "models"),
    runs_dir: str = os.path.join(PROJECT_ROOT, "runs"),
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Train specified RL agent and log telemetry to TensorBoard and disk."""
    agent_key = agent_key.lower()
    scenario_upper = scenario.upper()
    resolved_scenario = SCENARIO_ALIASES.get(scenario_upper)
    if resolved_scenario is None:
        raise ValueError(f"Invalid scenario '{scenario}'. Valid: {list(SCENARIO_ALIASES.keys())}")

    if episodes is None:
        if agent_key == "q_learning":
            episodes = int(Q_LEARNING_CONFIG["num_episodes"])
        elif agent_key in ("dqn", "ddqn"):
            episodes = int(DQN_CONFIG["num_episodes"])
        else:
            episodes = int(PPO_CONFIG["num_episodes"])

    # Seed global RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    py_rng = random.Random(seed)

    os.makedirs(models_dir, exist_ok=True)
    run_id = f"{agent_key}_{resolved_scenario}_seed{seed}"
    run_log_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_log_dir)
    min_replay_override = (
        min(1000, max(int(DQN_CONFIG["batch_size"]), episodes * 100))
        if episodes < 10
        else None
    )
    agent = _create_agent(
        agent_key=agent_key,
        device=device,
        min_replay_override=min_replay_override,
    )

    best_pth_path = os.path.join(models_dir, f"{agent_key}_{resolved_scenario}_best.pth")
    final_pth_path = os.path.join(models_dir, f"{agent_key}_{resolved_scenario}.pth")
    best_onnx_path = os.path.join(models_dir, f"{agent_key}_{resolved_scenario}_best.onnx")
    final_onnx_path = os.path.join(models_dir, f"{agent_key}_{resolved_scenario}.onnx")

    history_rows: list[dict[str, Any]] = []
    rolling_rewards: deque[float] = deque(maxlen=20)
    rolling_waits: deque[float] = deque(maxlen=20)
    rolling_max_waits: deque[float] = deque(maxlen=20)
    rolling_throughputs: deque[int] = deque(maxlen=20)

    best_rolling_reward = -float("inf")
    global_step = 0
    next_step_log = 1000
    last_update_occurred = False
    start_wall = time.time()

    initial_sc = py_rng.choice(STANDARD_SCENARIOS) if resolved_scenario == "MIXED" else resolved_scenario
    env = TrafficLightEnv(scenario_id=initial_sc, seed=seed)

    print("=" * 96)
    print(f"  TRAINING START | Agent: {agent.name} ({agent_key}) | Scenario: {resolved_scenario} | Episodes: {episodes} | Seed: {seed}")
    print("=" * 96)
    print(f"{'Step':>8} | {'Episode':>7} | {'Mean Reward':>11} | {'Std Reward':>10} | {'Avg Wait (s)':>12} | {'Max Wait (s)':>12} | {'Throughput':>10}")
    print("-" * 96)

    last_obs: Optional[np.ndarray] = None
    last_done = True

    try:
        for ep in range(1, episodes + 1):
            ep_scenario = (
                py_rng.choice(STANDARD_SCENARIOS)
                if resolved_scenario == "MIXED"
                else resolved_scenario
            )
            obs, info = env.reset(seed=seed + ep, options={"scenario_id": ep_scenario})
            agent.reset()

            ep_reward = 0.0
            ep_len = 0
            done = False

            while not done and ep_len < MAX_STEPS:
                action = agent.select_action(obs, info, eval_mode=False)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)

                update_metrics = agent.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    info=info,
                    next_info=next_info,
                )

                global_step += 1
                ep_len += 1
                ep_reward += float(reward)

                if update_metrics is not None:
                    last_update_occurred = True
                    _log_update_metrics(writer, agent_key, update_metrics, global_step)

                # Periodic 1,000-step diagnostic print
                if global_step >= next_step_log:
                    m_rew = float(np.mean(rolling_rewards)) if rolling_rewards else ep_reward
                    s_rew = float(np.std(rolling_rewards)) if len(rolling_rewards) > 1 else 0.0
                    m_wait = float(np.mean(rolling_waits)) if rolling_waits else float(next_info["avg_wait_time"])
                    m_max_w = float(np.mean(rolling_max_waits)) if rolling_max_waits else float(next_info["max_wait_recorded"])
                    m_tp = int(np.mean(rolling_throughputs)) if rolling_throughputs else int(next_info["total_passed"])
                    print(
                        f"{global_step:>8d} | {ep:>7d} | {m_rew:>11.2f} | {s_rew:>10.2f} | "
                        f"{m_wait:>12.2f} | {m_max_w:>12.2f} | {m_tp:>10d}"
                    )
                    next_step_log += 1000

                obs = next_obs
                info = next_info
                last_obs = next_obs
                last_done = done

            agent.on_episode_end()

            avg_wait = float(info.get("avg_wait_time", 0.0))
            max_wait = float(info.get("max_wait_recorded", 0.0))
            total_passed = int(info.get("total_passed", 0))

            rolling_rewards.append(ep_reward)
            rolling_waits.append(avg_wait)
            rolling_max_waits.append(max_wait)
            rolling_throughputs.append(total_passed)

            mean_20_rew = float(np.mean(rolling_rewards))
            std_20_rew = float(np.std(rolling_rewards)) if len(rolling_rewards) > 1 else 0.0

            # Log episode metrics to TensorBoard
            writer.add_scalar("Environment/Cumulative_Reward", ep_reward, ep)
            writer.add_scalar("Environment/Episode_Length", ep_len, ep)
            writer.add_scalar("Environment/Avg_Wait_Time", avg_wait, ep)
            writer.add_scalar("Environment/Max_Wait_Time", max_wait, ep)
            writer.add_scalar("Environment/Total_Throughput", total_passed, ep)

            history_rows.append(
                {
                    "agent": agent_key,
                    "scenario": resolved_scenario,
                    "episode": ep,
                    "global_step": global_step,
                    "cumulative_reward": ep_reward,
                    "episode_length": ep_len,
                    "avg_wait_time": avg_wait,
                    "max_wait_time": max_wait,
                    "total_throughput": total_passed,
                }
            )

            # Save best rolling-20 checkpoint
            if mean_20_rew > best_rolling_reward:
                best_rolling_reward = mean_20_rew
                agent.save(best_pth_path)

            # Periodic per-episode summary print
            if ep == 1 or ep == episodes or ep % max(1, min(10, episodes // 5 or 1)) == 0:
                print(
                    f"{global_step:>8d} | {ep:>7d} | {mean_20_rew:>11.2f} | {std_20_rew:>10.2f} | "
                    f"{avg_wait:>12.2f} | {max_wait:>12.2f} | {total_passed:>10d}"
                )

        # Ensure PPO executes an optimization step even on very short smoke runs (< rollout_steps)
        if agent_key == "ppo" and not last_update_occurred:
            flush_metrics = agent.flush_update(last_next_obs=last_obs, last_done=last_done)
            if flush_metrics is not None:
                _log_update_metrics(writer, agent_key, flush_metrics, global_step)

        # Save final checkpoint
        agent.save(final_pth_path)

        # Export ONNX graphs if requested and supported
        if export_onnx and agent_key in ("ppo", "dqn", "ddqn"):
            agent.export_onnx(final_onnx_path)
            if os.path.exists(best_pth_path):
                best_agent = _create_agent(agent_key=agent_key, device="cpu")
                best_agent.load(best_pth_path)
                best_agent.export_onnx(best_onnx_path)

        # Persist history CSV alongside TensorBoard event files for fig4 plotting
        csv_history_path = os.path.join(run_log_dir, "history.csv")
        with open(csv_history_path, "w", newline="", encoding="utf-8") as f:
            if history_rows:
                fieldnames = list(history_rows[0].keys())
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(history_rows)

        elapsed = time.time() - start_wall
        print("-" * 96)
        print(f"  TRAINING COMPLETE in {elapsed:.2f}s | Best Rolling-20 Reward: {best_rolling_reward:.2f}")
        print(f"  Saved Best Model  : {best_pth_path}")
        print(f"  Saved Final Model : {final_pth_path}")
        if export_onnx and agent_key in ("ppo", "dqn", "ddqn"):
            print(f"  Exported ONNX Best: {best_onnx_path}")
            print(f"  Exported ONNX Fin : {final_onnx_path}")
        print(f"  TensorBoard Logs  : {run_log_dir}")
        print("=" * 96)

        return {
            "agent": agent_key,
            "scenario": resolved_scenario,
            "episodes": episodes,
            "global_steps": global_step,
            "best_rolling_reward": best_rolling_reward,
            "best_pth_path": best_pth_path,
            "final_pth_path": final_pth_path,
            "best_onnx_path": best_onnx_path if export_onnx else None,
            "final_onnx_path": final_onnx_path if export_onnx else None,
            "run_log_dir": run_log_dir,
        }
    finally:
        env.close()
        writer.flush()
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified RL Training CLI with TensorBoard & ONNX Export"
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["q_learning", "dqn", "ddqn", "ppo"],
        help="RL algorithm to train",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="SC3_ASYMMETRIC",
        choices=["SC1", "SC2", "SC3", "SC4", "SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE", "MIXED"],
        help="Traffic scenario or MIXED across SC1-SC4",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (defaults to config block value, usually 500)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export inference model (.onnx) alongside PyTorch checkpoint (.pth)",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_agent(
        agent_key=args.agent,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        export_onnx=args.export_onnx,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
