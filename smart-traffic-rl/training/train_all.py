"""Sequential Multi-Agent Training Runner for SmartTraffic RL Studio.

Runs all 4 RL algorithms (q_learning, dqn, ddqn, ppo) sequentially on the same
training scenario and random seed, logging wall-clock timings and printing a unified summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.train import train_agent


ALL_RL_AGENTS = ("q_learning", "dqn", "ddqn", "ppo")


def train_all_agents(
    scenario: str = "MIXED",
    episodes: int = 500,
    seed: int = 42,
    eval_freq: int = 10,
    export_onnx: bool = True,
    clean_run: bool = True,
    device: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train all 4 RL agents sequentially and print consolidated timing/eval summary."""
    total_start = time.time()
    results: list[dict[str, Any]] = []

    print("=" * 104)
    print(f"  TRAIN ALL RL AGENTS | Scenario: {scenario} | Episodes: {episodes} | Seed: {seed} | ONNX: {export_onnx}")
    print("=" * 104)

    for ag in ALL_RL_AGENTS:
        res = train_agent(
            agent_key=ag,
            scenario=scenario,
            episodes=episodes,
            seed=seed,
            eval_freq=eval_freq,
            export_onnx=(export_onnx and ag in ("dqn", "ddqn", "ppo")),
            clean_run=clean_run,
            device=device,
        )
        results.append(res)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 104)
    print(f"  ALL 4 AGENTS FINISHED in {total_elapsed:.2f}s")
    print("-" * 104)
    print(f"{'Agent':<14} | {'Scenario':<18} | {'Best Eval Rew':>14} | {'Time (s)':>10} | {'Best Checkpoint'}")
    print("-" * 104)
    for r in results:
        print(
            f"{r['agent']:<14} | {r['scenario']:<18} | {r['best_eval_reward']:>14.2f} | "
            f"{r['elapsed_sec']:>10.2f} | {os.path.basename(r['best_pth_path'])}"
        )
    print("=" * 104)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all 4 RL algorithms sequentially on the same scenario")
    parser.add_argument(
        "--scenario",
        type=str,
        default="MIXED",
        choices=["SC1", "SC2", "SC3", "SC4", "SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE", "MIXED"],
        help="Target scenario or MIXED",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--eval-freq", type=int, default=10, help="Deterministic eval frequency")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export for Deep RL models")
    parser.add_argument("--no-clean-run", action="store_true", help="Keep previous TensorBoard event files")
    parser.add_argument("--device", type=str, default=None, help="Torch device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_all_agents(
        scenario=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        eval_freq=args.eval_freq,
        export_onnx=not args.no_onnx,
        clean_run=not args.no_clean_run,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
