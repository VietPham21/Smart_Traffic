"""Automated Multi-Tier Algorithmic Benchmark Runner.

Executes quantitative evaluations across 4 standardized traffic scenarios (SC1_LOW - SC4_DYNAMIC_SURGE),
logging metrics (Average Wait Time, Max Wait Time, Throughput) to CSV,
and rendering comparison Boxplots & throughput charts via Matplotlib/Seaborn.

Maintainer: Trương Hữu Quang (MSV: 2351170614)
"""

from __future__ import annotations

import os
from typing import Any, Optional
import numpy as np
import pandas as pd

from core.config import (
    SCENARIOS,
    MAX_STEPS,
    DELTA_T_GREEN,
    EPISODE_DURATION,
)
from core.env import TrafficLightEnv
from agents.base_agent import BaseAgent
from agents.baselines import FixedTimeAgent, WebsterAgent, MaxPressureAgent
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent


class BenchmarkRunner:
    """Orchestrates multi-scenario evaluation and comparative statistical analysis."""

    def __init__(
        self,
        agents: Optional[dict[str, BaseAgent]] = None,
        scenarios: Optional[list[str]] = None,
        output_dir: str = "benchmark_results",
    ) -> None:
        """Initialize benchmark suite.

        Args:
            agents: Dict mapping Agent Name -> BaseAgent instance.
            scenarios: List of scenario keys to test (default: SC1_LOW - SC4_DYNAMIC_SURGE).
            output_dir: Directory where CSV reports and PNG figures are exported.
        """
        # TODO [Trương Hữu Quang]: Khởi tạo benchmark suite và thư mục kết quả
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.scenarios: list[str] = scenarios or ["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"]

        if agents is not None:
            self.agents: dict[str, BaseAgent] = agents
        else:
            self.agents = {
                "Fixed-Time": FixedTimeAgent(name="Fixed-Time"),
                "Webster-Opt": WebsterAgent(name="Webster-Opt"),
                "Max-Pressure": MaxPressureAgent(name="Max-Pressure"),
                "Tabular-QL": QLearningAgent(name="Tabular-QL"),
                "Standard-DQN": DQNAgent(use_double_dqn=False, name="Standard-DQN"),
                "Double-DQN": DQNAgent(use_double_dqn=True, name="Double-DQN"),
            }

    def run_single_episode(
        self,
        agent: BaseAgent,
        scenario_id: str,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute a single episode simulation and return summary metrics.

        Uses select_action(obs, info, eval_mode=True) adhering to Interface Contract.
        """
        # TODO [Trương Hữu Quang]: Thực thi 1 episode đánh giá và thu thập telemetry
        env = TrafficLightEnv(scenario_id=scenario_id, seed=seed)
        obs, info = env.reset(seed=seed)
        agent.reset()

        total_reward = 0.0
        step_limit = max_steps or MAX_STEPS
        step_idx = 0
        peak_observed_wait = 0.0

        while step_idx < step_limit:
            action = agent.select_action(obs, info, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_idx += 1

            for w in info.get("max_waits", []):
                if w > peak_observed_wait:
                    peak_observed_wait = w

            if terminated or truncated:
                break

        env.close()

        total_passed = int(info.get("total_passed", 0))
        sim_hours = (step_idx * float(DELTA_T_GREEN)) / 3600.0
        throughput_vph = total_passed / max(1e-4, sim_hours)
        avg_wait = float(info.get("avg_wait_time", 0.0))

        return {
            "agent": agent.name,
            "scenario": scenario_id,
            "seed": seed,
            "avg_wait_time": float(avg_wait),
            "max_wait_time": float(peak_observed_wait),
            "total_passed": int(total_passed),
            "throughput_vph": float(throughput_vph),
            "total_reward": float(total_reward),
            "sim_steps": step_idx,
        }

    def run_benchmark(
        self,
        num_episodes_per_scenario: int = 3,
        base_seed: int = 100,
        max_steps_per_episode: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run full evaluation matrix across all registered agents and scenarios."""
        # TODO [Trương Hữu Quang]: Chạy ma trận đánh giá lặp lại nhiều episode
        records: list[dict[str, Any]] = []

        for scenario_id in self.scenarios:
            print(f"[BenchmarkRunner] Evaluating Scenario: {scenario_id}...")
            for agent_name, agent in self.agents.items():
                for ep in range(num_episodes_per_scenario):
                    seed = base_seed + ep
                    res = self.run_single_episode(
                        agent=agent,
                        scenario_id=scenario_id,
                        seed=seed,
                        max_steps=max_steps_per_episode,
                    )
                    records.append(res)

        df = pd.DataFrame(records)
        return df

    def export_csv(self, df: pd.DataFrame, filename: str = "benchmark_summary.csv") -> str:
        """Export benchmark DataFrame to CSV file in output_dir."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"[BenchmarkRunner] Exported results to: {filepath}")
        return filepath

    def plot_results(
        self,
        df: pd.DataFrame,
        filename_prefix: str = "benchmark",
    ) -> list[str]:
        """Generate Boxplots of average wait times and Throughput charts via Seaborn/Matplotlib."""
        # TODO [Trương Hữu Quang]: Vẽ biểu đồ Boxplot phân bố thời gian chờ và biểu đồ cột thông lượng
        generated_files: list[str] = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="darkgrid")

            # 1. Boxplot: Average Wait Time by Scenario & Agent
            plt.figure(figsize=(10, 6))
            ax1 = sns.boxplot(
                data=df,
                x="scenario",
                y="avg_wait_time",
                hue="agent",
                palette="Set2",
            )
            plt.title("Average Vehicle Wait Time by Scenario (Lower is Better)", fontsize=14, fontweight="bold")
            plt.xlabel("Traffic Scenario", fontsize=12)
            plt.ylabel("Avg Wait Time (seconds / vehicle)", fontsize=12)
            plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            box_path = os.path.join(self.output_dir, f"{filename_prefix}_wait_time_boxplot.png")
            plt.savefig(box_path, dpi=300)
            plt.close()
            generated_files.append(box_path)

            # 2. Barplot: Total Throughput (Vehicles / Hour)
            plt.figure(figsize=(10, 6))
            ax2 = sns.barplot(
                data=df,
                x="scenario",
                y="throughput_vph",
                hue="agent",
                palette="viridis",
                errorbar=None,
            )
            plt.title("Intersection Throughput (Higher is Better)", fontsize=14, fontweight="bold")
            plt.xlabel("Traffic Scenario", fontsize=12)
            plt.ylabel("Throughput (vehicles / hour)", fontsize=12)
            plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            bar_path = os.path.join(self.output_dir, f"{filename_prefix}_throughput_bar.png")
            plt.savefig(bar_path, dpi=300)
            plt.close()
            generated_files.append(bar_path)

            print(f"[BenchmarkRunner] Plots saved successfully to: {generated_files}")
        except Exception as e:
            print(f"[BenchmarkRunner] Plot generation skipped or failed: {e}")

        return generated_files
