"""Unified RL Training & TensorBoard Logging CLI for SmartTraffic RL Studio.

Supports all 4 trainable RL agents:
- q_learning : Tabular Q-Learning with 972 discrete states
- dqn        : Standard Deep Q-Network (10->64->64->2)
- ddqn       : Double Deep Q-Network (10->64->64->2)
- ppo        : Masked Proximal Policy Optimization with GAE (10->128->128)

Key Safeguards:
- Strictly separates `terminated` vs `truncated` for unbiased bootstrapping.
- Cleans overlapping TensorBoard event files by default (--clean-run).
- Evaluates checkpoints on a fixed Deterministic Eval Suite (eval_mode=True)
  every `--eval-freq` episodes and only saves `*_best.pth` after real optimization updates.
- Exports rich diagnostics (mean_q, td_loss, policy_loss, value_loss, entropy,
  approx_kl, eval_reward) to TensorBoard and `runs/<run_id>/history.csv`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
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
        if "approx_kl" in metrics:
            writer.add_scalar("Metrics/Approx_KL", float(metrics["approx_kl"]), step)
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


def _evaluate_deterministic(
    agent: BaseAgent,
    eval_env: TrafficLightEnv,
    resolved_scenario: str,
    base_eval_seed: int = 9000,
) -> dict[str, float]:
    """Run deterministic policy evaluation across fixed validation seeds (zero exploration)."""
    if resolved_scenario == "MIXED":
        eval_cases = [(sc, base_eval_seed + i) for i, sc in enumerate(STANDARD_SCENARIOS)]
    else:
        eval_cases = [(resolved_scenario, base_eval_seed + i) for i in range(3)]

    rewards: list[float] = []
    avg_waits: list[float] = []
    max_waits: list[float] = []

    for sc_id, sd in eval_cases:
        obs, info = eval_env.reset(seed=sd, options={"scenario_id": sc_id})
        agent.reset()
        ep_rew = 0.0
        for _ in range(MAX_STEPS):
            action = agent.select_action(obs, info, eval_mode=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_rew += float(reward)
            if terminated or truncated:
                break
        rewards.append(ep_rew)
        avg_waits.append(float(info.get("avg_wait_time", 0.0)))
        max_waits.append(float(info.get("max_wait_recorded", 0.0)))

    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_avg_wait": float(np.mean(avg_waits)),
        "eval_max_wait": float(np.mean(max_waits)),
    }


def _clean_previous_tb_events(run_log_dir: str) -> int:
    """Remove old TensorBoard event files in run_log_dir to avoid overlapping chart traces."""
    removed = 0
    for old_ev in glob.glob(os.path.join(run_log_dir, "events.out.tfevents*")):
        try:
            os.remove(old_ev)
            removed += 1
        except OSError:
            pass
    return removed


def train_agent(
    agent_key: str,
    scenario: str = "SC3_ASYMMETRIC",
    episodes: Optional[int] = None,
    seed: int = 42,
    eval_freq: int = 10,
    export_onnx: bool = False,
    clean_run: bool = True,
    models_dir: str = os.path.join(PROJECT_ROOT, "models"),
    runs_dir: str = os.path.join(PROJECT_ROOT, "runs"),
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Train specified RL agent with unbiased truncation handling & deterministic checkpoint selection."""
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    py_rng = random.Random(seed)

    os.makedirs(models_dir, exist_ok=True)
    run_id = f"{agent_key}_{resolved_scenario}_seed{seed}"
    run_log_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_log_dir, exist_ok=True)
    if clean_run:
        _clean_previous_tb_events(run_log_dir)

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

    best_eval_reward = -float("inf")
    has_updated_once = False
    global_step = 0
    next_step_log = 1000
    start_wall = time.time()

    initial_sc = py_rng.choice(STANDARD_SCENARIOS) if resolved_scenario == "MIXED" else resolved_scenario
    env = TrafficLightEnv(scenario_id=initial_sc, seed=seed)
    eval_env = TrafficLightEnv(scenario_id=initial_sc, seed=9000)

    print("=" * 104)
    print(f"  TRAINING START | Agent: {agent.name} ({agent_key}) | Scenario: {resolved_scenario} | Episodes: {episodes} | Seed: {seed}")
    print("=" * 104)
    print(
        f"{'Step':>8} | {'Episode':>7} | {'Roll Rew':>10} | {'Eval Rew':>10} | "
        f"{'Avg Wait(s)':>11} | {'Max Wait(s)':>11} | {'Throughput':>10}"
    )
    print("-" * 104)

    last_obs: Optional[np.ndarray] = None
    last_terminated = False
    last_truncated = False

    try:
        for ep in range(1, episodes + 1):
            if agent_key == "ppo" and hasattr(agent, "set_lr_progress"):
                agent.set_lr_progress((ep - 1) / max(1, episodes))

            ep_scenario = (
                py_rng.choice(STANDARD_SCENARIOS)
                if resolved_scenario == "MIXED"
                else resolved_scenario
            )
            obs, info = env.reset(seed=seed + ep, options={"scenario_id": ep_scenario})
            agent.reset()

            ep_reward = 0.0
            ep_len = 0
            ep_ended = False

            ep_mean_qs: list[float] = []
            ep_td_losses: list[float] = []
            ep_policy_losses: list[float] = []
            ep_value_losses: list[float] = []
            ep_entropies: list[float] = []
            ep_approx_kls: list[float] = []

            while not ep_ended and ep_len < MAX_STEPS:
                action = agent.select_action(obs, info, eval_mode=False)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                ep_ended = bool(terminated or truncated)

                update_metrics = agent.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=bool(terminated),
                    info=info,
                    next_info=next_info,
                    truncated=bool(truncated),
                )

                global_step += 1
                ep_len += 1
                ep_reward += float(reward)

                if update_metrics is not None:
                    has_updated_once = True
                    _log_update_metrics(writer, agent_key, update_metrics, global_step)
                    if "mean_q" in update_metrics:
                        ep_mean_qs.append(float(update_metrics["mean_q"]))
                    elif "q_val" in update_metrics:
                        ep_mean_qs.append(float(update_metrics["q_val"]))
                    if "td_loss" in update_metrics:
                        ep_td_losses.append(float(update_metrics["td_loss"]))
                    elif "loss" in update_metrics:
                        ep_td_losses.append(float(update_metrics["loss"]))
                    if "policy_loss" in update_metrics:
                        ep_policy_losses.append(float(update_metrics["policy_loss"]))
                    if "value_loss" in update_metrics:
                        ep_value_losses.append(float(update_metrics["value_loss"]))
                    if "entropy" in update_metrics:
                        ep_entropies.append(float(update_metrics["entropy"]))
                    if "approx_kl" in update_metrics:
                        ep_approx_kls.append(float(update_metrics["approx_kl"]))

                if global_step >= next_step_log:
                    m_rew = float(np.mean(rolling_rewards)) if rolling_rewards else ep_reward
                    m_wait = float(np.mean(rolling_waits)) if rolling_waits else float(next_info["avg_wait_time"])
                    m_max_w = float(np.mean(rolling_max_waits)) if rolling_max_waits else float(next_info["max_wait_recorded"])
                    m_tp = int(np.mean(rolling_throughputs)) if rolling_throughputs else int(next_info["total_passed"])
                    eval_str = f"{best_eval_reward:>10.2f}" if math.isfinite(best_eval_reward) else f"{'N/A':>10}"
                    print(
                        f"{global_step:>8d} | {ep:>7d} | {m_rew:>10.2f} | {eval_str} | "
                        f"{m_wait:>11.2f} | {m_max_w:>11.2f} | {m_tp:>10d}"
                    )
                    next_step_log += 1000

                obs = next_obs
                info = next_info
                last_obs = next_obs
                last_terminated = bool(terminated)
                last_truncated = bool(truncated)

            agent.on_episode_end()

            avg_wait = float(info.get("avg_wait_time", 0.0))
            max_wait = float(info.get("max_wait_recorded", 0.0))
            total_passed = int(info.get("total_passed", 0))

            rolling_rewards.append(ep_reward)
            rolling_waits.append(avg_wait)
            rolling_max_waits.append(max_wait)
            rolling_throughputs.append(total_passed)

            mean_20_rew = float(np.mean(rolling_rewards))

            writer.add_scalar("Environment/Cumulative_Reward", ep_reward, ep)
            writer.add_scalar("Environment/Episode_Length", ep_len, ep)
            writer.add_scalar("Environment/Avg_Wait_Time", avg_wait, ep)
            writer.add_scalar("Environment/Max_Wait_Time", max_wait, ep)
            writer.add_scalar("Environment/Total_Throughput", total_passed, ep)

            # Deterministic Checkpoint Evaluation every eval_freq episodes ONLY after real training updates
            eval_reward_this_ep: Optional[float] = None
            if has_updated_once and (ep % max(1, eval_freq) == 0):
                eval_res = _evaluate_deterministic(agent, eval_env, resolved_scenario)
                eval_reward_this_ep = eval_res["eval_reward"]
                writer.add_scalar("Evaluation/Mean_Reward", eval_reward_this_ep, ep)
                writer.add_scalar("Evaluation/Avg_Wait_Time", eval_res["eval_avg_wait"], ep)
                writer.add_scalar("Evaluation/Max_Wait_Time", eval_res["eval_max_wait"], ep)
                if eval_reward_this_ep > best_eval_reward:
                    best_eval_reward = eval_reward_this_ep
                    agent.save(best_pth_path)

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
                    "mean_q": float(np.mean(ep_mean_qs)) if ep_mean_qs else "",
                    "td_loss": float(np.mean(ep_td_losses)) if ep_td_losses else "",
                    "policy_loss": float(np.mean(ep_policy_losses)) if ep_policy_losses else "",
                    "value_loss": float(np.mean(ep_value_losses)) if ep_value_losses else "",
                    "entropy": float(np.mean(ep_entropies)) if ep_entropies else "",
                    "approx_kl": float(np.mean(ep_approx_kls)) if ep_approx_kls else "",
                    "eval_reward": eval_reward_this_ep if eval_reward_this_ep is not None else "",
                }
            )

            if ep == 1 or ep == episodes or ep % max(1, min(10, episodes // 5 or 1)) == 0:
                eval_disp = (
                    f"{eval_reward_this_ep:>10.2f}"
                    if eval_reward_this_ep is not None
                    else (f"{best_eval_reward:>10.2f}" if math.isfinite(best_eval_reward) else f"{'N/A':>10}")
                )
                print(
                    f"{global_step:>8d} | {ep:>7d} | {mean_20_rew:>10.2f} | {eval_disp} | "
                    f"{avg_wait:>11.2f} | {max_wait:>11.2f} | {total_passed:>10d}"
                )

        # Ensure PPO executes update on any remaining buffer (especially short smoke tests < rollout_steps)
        if agent_key == "ppo" and len(agent.rollout_buffer) > 0:
            flush_metrics = agent.flush_update(
                last_next_obs=last_obs,
                last_done=last_terminated,
                last_truncated=last_truncated,
            )
            if flush_metrics is not None:
                has_updated_once = True
                _log_update_metrics(writer, agent_key, flush_metrics, global_step)
                if history_rows:
                    history_rows[-1]["policy_loss"] = float(flush_metrics.get("policy_loss", 0.0))
                    history_rows[-1]["value_loss"] = float(flush_metrics.get("value_loss", 0.0))
                    history_rows[-1]["entropy"] = float(flush_metrics.get("entropy", 0.0))
                    history_rows[-1]["approx_kl"] = float(flush_metrics.get("approx_kl", 0.0))

        # Final deterministic evaluation pass (guaranteed after flush_update / short runs)
        if has_updated_once:
            final_eval = _evaluate_deterministic(agent, eval_env, resolved_scenario)
            final_eval_rew = final_eval["eval_reward"]
            writer.add_scalar("Evaluation/Mean_Reward", final_eval_rew, episodes)
            if history_rows and history_rows[-1]["eval_reward"] == "":
                history_rows[-1]["eval_reward"] = final_eval_rew
            if final_eval_rew > best_eval_reward:
                best_eval_reward = final_eval_rew
                agent.save(best_pth_path)

        # Save final checkpoint (and ensure best_pth exists if updates occurred)
        agent.save(final_pth_path)
        if has_updated_once and not os.path.exists(best_pth_path):
            agent.save(best_pth_path)

        # Export ONNX graphs if requested and supported
        if export_onnx and agent_key in ("ppo", "dqn", "ddqn"):
            agent.export_onnx(final_onnx_path)
            if os.path.exists(best_pth_path):
                best_agent = _create_agent(agent_key=agent_key, device="cpu")
                best_agent.load(best_pth_path)
                best_agent.export_onnx(best_onnx_path)

        csv_history_path = os.path.join(run_log_dir, "history.csv")
        with open(csv_history_path, "w", newline="", encoding="utf-8") as f:
            if history_rows:
                fieldnames = list(history_rows[0].keys())
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(history_rows)

        elapsed = time.time() - start_wall
        print("-" * 104)
        print(f"  TRAINING COMPLETE in {elapsed:.2f}s | Best Deterministic Eval Reward: {best_eval_reward:.2f}")
        print(f"  Saved Best Model  : {best_pth_path}")
        print(f"  Saved Final Model : {final_pth_path}")
        if export_onnx and agent_key in ("ppo", "dqn", "ddqn"):
            print(f"  Exported ONNX Best: {best_onnx_path}")
            print(f"  Exported ONNX Fin : {final_onnx_path}")
        print(f"  TensorBoard Logs  : {run_log_dir}")
        print("=" * 104)

        return {
            "agent": agent_key,
            "scenario": resolved_scenario,
            "episodes": episodes,
            "global_steps": global_step,
            "elapsed_sec": elapsed,
            "best_eval_reward": best_eval_reward,
            "best_pth_path": best_pth_path,
            "final_pth_path": final_pth_path,
            "best_onnx_path": best_onnx_path if export_onnx else None,
            "final_onnx_path": final_onnx_path if export_onnx else None,
            "run_log_dir": run_log_dir,
        }
    finally:
        env.close()
        eval_env.close()
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
    parser.add_argument("--eval-freq", type=int, default=10, help="Deterministic eval frequency in episodes")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export inference model (.onnx) alongside PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--no-clean-run",
        action="store_true",
        help="Keep previous TensorBoard tfevents files in run directory instead of replacing",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_agent(
        agent_key=args.agent,
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        eval_freq=args.eval_freq,
        export_onnx=args.export_onnx,
        clean_run=not args.no_clean_run,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
