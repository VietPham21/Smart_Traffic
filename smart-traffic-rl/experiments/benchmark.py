"""Quantitative Benchmark & Automated Academic Figure Generator.

Evaluates 6+ signal control algorithms (FixedTime, [optional Webster], MaxPressure,
QLearning, StandardDQN, DoubleDQN, PPO) across all 4 benchmark traffic scenarios
(SC1_LOW - SC4_DYNAMIC_SURGE) repeated across deterministic Random Seeds
[1000, 1001, ..., 1000 + N - 1] (default N=20).

Key Scientific Safeguards:
- Supports `--model-tag MIXED` (default) or `AUTO` / specific scenario tag to guarantee
  fair generalization evaluation from the exact same training regime.
- Emits explicit [WARNING] (or raises error under `--strict`) if any checkpoint is missing.
- Outputs:
  1. benchmark_raw.csv
  2. benchmark_summary.csv
  3. fig1_avg_wait_comparison.png     (300 DPI)
  4. fig2_max_wait_starvation.png     (300 DPI)
  5. fig3_throughput_comparison.png   (300 DPI)
  6. fig4_training_curves.png         (300 DPI, unified scenario + integer ticks + outside legend)
  7. fig5_dqn_vs_ddqn_qvalue.png      (300 DPI, Overestimation Bias proof)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from core.config import AXIS_EW, AXIS_NS, MAX_STEPS, SCENARIOS, STARVATION_THRESHOLD, T_MAX
from core.env import TrafficLightEnv
from agents.base_agent import BaseAgent
from agents.baselines import FixedTimeAgent, MaxPressureAgent, WebsterAgent
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


ALL_SCENARIOS = ["SC1_LOW", "SC2_HIGH", "SC3_ASYMMETRIC", "SC4_DYNAMIC_SURGE"]

DEFAULT_AGENT_SPECS = [
    ("FixedTime", "fixed_time"),
    ("MaxPressure", "max_pressure"),
    ("QLearning", "q_learning"),
    ("StandardDQN", "dqn"),
    ("DoubleDQN", "ddqn"),
    ("PPO", "ppo"),
]

RL_MODEL_KEYS = {"q_learning", "dqn", "ddqn", "ppo"}


def _resolve_checkpoint_candidates(
    model_key: str,
    scenario_id: str,
    models_dir: str,
    model_tag: str = "MIXED",
) -> list[str]:
    tag = model_tag.upper()
    if tag == "AUTO":
        return [
            os.path.join(models_dir, f"{model_key}_{scenario_id}_best.pth"),
            os.path.join(models_dir, f"{model_key}_MIXED_best.pth"),
            os.path.join(models_dir, f"{model_key}_{scenario_id}.pth"),
            os.path.join(models_dir, f"{model_key}_MIXED.pth"),
        ]
    return [
        os.path.join(models_dir, f"{model_key}_{tag}_best.pth"),
        os.path.join(models_dir, f"{model_key}_{tag}.pth"),
    ]


def _load_best_checkpoint(
    agent: BaseAgent,
    model_key: str,
    scenario_id: str,
    models_dir: str,
    model_tag: str = "MIXED",
    strict: bool = False,
) -> Optional[str]:
    """Load trained weights from models/ according to model_tag; warn or fail loudly if missing."""
    candidates = _resolve_checkpoint_candidates(
        model_key=model_key,
        scenario_id=scenario_id,
        models_dir=models_dir,
        model_tag=model_tag,
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                agent.load(path)
                return path
            except Exception as exc:
                print(f"[WARNING] Failed loading checkpoint '{path}': {exc}", file=sys.stderr)

    msg = (
        f"[WARNING] No trained checkpoint found for agent='{model_key}' "
        f"(model_tag='{model_tag}', scenario='{scenario_id}'). Searched: {candidates}"
    )
    if strict:
        raise FileNotFoundError(msg)
    print(msg + " -> Running UNTRAINED weights!", file=sys.stderr)
    return None


class BenchmarkRunner:
    """Multi-scenario evaluation & academic visualization orchestrator."""

    def __init__(
        self,
        agents: Optional[dict[str, BaseAgent]] = None,
        scenarios: Optional[list[str]] = None,
        model_tag: str = "MIXED",
        train_scenario: Optional[str] = None,
        strict: bool = False,
        include_webster: bool = False,
        output_dir: str = os.path.join(PROJECT_ROOT, "results"),
        models_dir: str = os.path.join(PROJECT_ROOT, "models"),
        runs_dir: str = os.path.join(PROJECT_ROOT, "runs"),
    ) -> None:
        self.output_dir: str = output_dir
        self.models_dir: str = models_dir
        self.runs_dir: str = runs_dir
        self.model_tag: str = model_tag.upper()
        self.train_scenario: str = (
            train_scenario.upper()
            if train_scenario
            else ("MIXED" if self.model_tag == "AUTO" else self.model_tag)
        )
        self.strict: bool = bool(strict)
        self.include_webster: bool = bool(include_webster)
        os.makedirs(self.output_dir, exist_ok=True)

        self.scenarios: list[str] = scenarios or list(ALL_SCENARIOS)

        if agents is not None:
            self.agents: dict[str, BaseAgent] = agents
            self._custom_agents = True
        else:
            self._custom_agents = False
            self.agents = {
                "FixedTime": FixedTimeAgent(cycle_green_time=30.0, name="FixedTime"),
                "MaxPressure": MaxPressureAgent(theta_diff=3.0, name="MaxPressure"),
                "QLearning": QLearningAgent(name="QLearning"),
                "StandardDQN": DQNAgent(use_double_dqn=False, device="cpu", name="StandardDQN"),
                "DoubleDQN": DQNAgent(use_double_dqn=True, device="cpu", name="DoubleDQN"),
                "PPO": PPOAgent(device="cpu", name="PPO"),
            }
            if self.include_webster:
                self.agents["Webster"] = WebsterAgent(name="Webster")

    def _prepare_agent_for_scenario(self, label: str, model_key: str, scenario_id: str) -> BaseAgent:
        if self._custom_agents and label in self.agents:
            return self.agents[label]
        if model_key == "fixed_time":
            return FixedTimeAgent(cycle_green_time=30.0, name=label)
        if model_key == "webster":
            wb = WebsterAgent(name=label)
            rates = SCENARIOS[scenario_id].rate_func(0.0)
            flow_ns = max(rates[AXIS_NS[0]], rates[AXIS_NS[1]])
            flow_ew = max(rates[AXIS_EW[0]], rates[AXIS_EW[1]])
            wb.update_splits(flow_ns=flow_ns, flow_ew=flow_ew)
            return wb
        if model_key == "max_pressure":
            return MaxPressureAgent(theta_diff=3.0, name=label)
        if model_key == "q_learning":
            ag: BaseAgent = QLearningAgent(name=label)
        elif model_key == "dqn":
            ag = DQNAgent(use_double_dqn=False, device="cpu", name=label)
        elif model_key == "ddqn":
            ag = DQNAgent(use_double_dqn=True, device="cpu", name=label)
        else:
            ag = PPOAgent(device="cpu", name=label)

        loaded = _load_best_checkpoint(
            agent=ag,
            model_key=model_key,
            scenario_id=scenario_id,
            models_dir=self.models_dir,
            model_tag=self.model_tag,
            strict=self.strict,
        )
        if loaded and scenario_id == self.scenarios[0]:
            print(f"  [Loaded] {label:<12} <- {os.path.basename(loaded)}")
        return ag

    def run_single_episode(
        self,
        agent: BaseAgent,
        scenario_id: str,
        seed: int = 1000,
        max_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        env = TrafficLightEnv(scenario_id=scenario_id, seed=seed)
        obs, info = env.reset(seed=seed)
        agent.reset()

        total_reward = 0.0
        step_limit = max_steps or MAX_STEPS
        step_idx = 0

        while step_idx < step_limit:
            action = agent.select_action(obs, info, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_idx += 1
            if terminated or truncated:
                break

        sim_time = max(1.0, float(info.get("sim_time", env.current_time)))
        total_passed = int(info.get("total_passed", 0))
        passed_per_min = float(total_passed) / (sim_time / 60.0)
        avg_wait = float(info.get("avg_wait_time", 0.0))
        max_wait = float(info.get("max_wait_recorded", 0.0))
        env.close()

        return {
            "agent": agent.name,
            "scenario": scenario_id,
            "seed": int(seed),
            "avg_wait_time": avg_wait,
            "max_wait_time": max_wait,
            "total_passed": total_passed,
            "passed_per_min": passed_per_min,
            "total_reward": float(total_reward),
            "sim_steps": step_idx,
        }

    def run_benchmark(
        self,
        num_seeds: int = 20,
        base_seed: int = 1000,
        max_steps_per_episode: Optional[int] = None,
    ) -> pd.DataFrame:
        seeds = [base_seed + i for i in range(num_seeds)]
        records: list[dict[str, Any]] = []

        if self._custom_agents:
            specs = [(name, name.lower()) for name in self.agents.keys()]
        else:
            specs = list(DEFAULT_AGENT_SPECS)
            if self.include_webster:
                specs.insert(1, ("Webster", "webster"))

        for scenario_id in self.scenarios:
            print(f"[BenchmarkRunner] Scenario: {scenario_id} (tag={self.model_tag}, seeds: {seeds[0]}..{seeds[-1]})")
            for label, model_key in specs:
                agent = self._prepare_agent_for_scenario(label, model_key, scenario_id)
                for sd in seeds:
                    res = self.run_single_episode(
                        agent=agent,
                        scenario_id=scenario_id,
                        seed=sd,
                        max_steps=max_steps_per_episode,
                    )
                    records.append(res)

        return pd.DataFrame(records)

    def build_summary(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Mean +/- Std aggregated summary grouped by scenario and agent."""
        grouped = raw_df.groupby(["scenario", "agent"], sort=False).agg(
            avg_wait_mean=("avg_wait_time", "mean"),
            avg_wait_std=("avg_wait_time", "std"),
            max_wait_mean=("max_wait_time", "mean"),
            max_wait_std=("max_wait_time", "std"),
            total_passed_mean=("total_passed", "mean"),
            total_passed_std=("total_passed", "std"),
            passed_per_min_mean=("passed_per_min", "mean"),
            passed_per_min_std=("passed_per_min", "std"),
            total_reward_mean=("total_reward", "mean"),
            total_reward_std=("total_reward", "std"),
        ).reset_index()

        std_cols = [c for c in grouped.columns if c.endswith("_std")]
        grouped[std_cols] = grouped[std_cols].fillna(0.0)

        grouped["avg_wait_str"] = grouped.apply(
            lambda r: f"{r['avg_wait_mean']:.2f} +/- {r['avg_wait_std']:.2f}", axis=1
        )
        grouped["max_wait_str"] = grouped.apply(
            lambda r: f"{r['max_wait_mean']:.2f} +/- {r['max_wait_std']:.2f}", axis=1
        )
        grouped["throughput_min_str"] = grouped.apply(
            lambda r: f"{r['passed_per_min_mean']:.2f} +/- {r['passed_per_min_std']:.2f}", axis=1
        )
        return grouped

    def export_csvs(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[str, str]:
        raw_path = os.path.join(self.output_dir, "benchmark_raw.csv")
        sum_path = os.path.join(self.output_dir, "benchmark_summary.csv")
        raw_df.to_csv(raw_path, index=False, encoding="utf-8")
        summary_df.to_csv(sum_path, index=False, encoding="utf-8")
        print(f"[BenchmarkRunner] Saved raw CSV     -> {raw_path}")
        print(f"[BenchmarkRunner] Saved summary CSV -> {sum_path}")
        return raw_path, sum_path

    def _find_run_dir(self, key: str) -> Optional[str]:
        exact_matches = sorted(glob.glob(os.path.join(self.runs_dir, f"{key}_{self.train_scenario}_seed*")))
        if exact_matches:
            return exact_matches[-1]
        any_matches = sorted(glob.glob(os.path.join(self.runs_dir, f"{key}_*")))
        if any_matches:
            print(
                f"[WARNING] Log for {key}_{self.train_scenario} not found; falling back to {os.path.basename(any_matches[-1])}",
                file=sys.stderr,
            )
            return any_matches[-1]
        return None

    def _collect_training_curves(self) -> dict[str, pd.DataFrame]:
        """Load per-episode curves cleanly from runs/<agent>_<train_scenario>_*/history.csv or TensorBoard."""
        target_map = {
            "q_learning": "Q-Learning",
            "dqn": "Standard DQN",
            "ddqn": "Double DQN",
            "ppo": "PPO",
        }
        curves: dict[str, pd.DataFrame] = {}

        for key, label in target_map.items():
            run_dir = self._find_run_dir(key)
            if not run_dir:
                continue
            csv_path = os.path.join(run_dir, "history.csv")
            if os.path.exists(csv_path):
                df_h = pd.read_csv(csv_path)
                if "episode" in df_h.columns and "cumulative_reward" in df_h.columns:
                    curves[label] = df_h.copy()
                    continue

            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

                ea = EventAccumulator(run_dir)
                ea.Reload()
                if "Environment/Cumulative_Reward" in ea.Tags().get("scalars", []):
                    evs = ea.Scalars("Environment/Cumulative_Reward")
                    curves[label] = pd.DataFrame(
                        [{"episode": e.step, "cumulative_reward": e.value} for e in evs]
                    )
            except Exception:
                pass

        return curves

    def _collect_qvalue_curves(self) -> dict[str, pd.DataFrame]:
        """Load mean_q training diagnostics for Standard DQN vs Double DQN (fig5 Overestimation Bias)."""
        target_map = {
            "dqn": "Standard DQN",
            "ddqn": "Double DQN",
        }
        q_curves: dict[str, pd.DataFrame] = {}

        for key, label in target_map.items():
            run_dir = self._find_run_dir(key)
            if not run_dir:
                continue
            csv_path = os.path.join(run_dir, "history.csv")
            if os.path.exists(csv_path):
                df_h = pd.read_csv(csv_path)
                if "episode" in df_h.columns and "mean_q" in df_h.columns:
                    sub = df_h[["episode", "mean_q"]].dropna()
                    sub = sub[pd.to_numeric(sub["mean_q"], errors="coerce").notna()]
                    if not sub.empty:
                        sub["mean_q"] = sub["mean_q"].astype(float)
                        q_curves[label] = sub.rename(columns={"episode": "step"})
                        continue

            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

                ea = EventAccumulator(run_dir)
                ea.Reload()
                if "Metrics/Mean_Q_Value" in ea.Tags().get("scalars", []):
                    evs = ea.Scalars("Metrics/Mean_Q_Value")
                    q_curves[label] = pd.DataFrame(
                        [{"step": e.step, "mean_q": e.value} for e in evs]
                    )
            except Exception:
                pass

        return q_curves

    def plot_all_figures(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[str]:
        """Generate all 5 high-DPI (300 DPI) academic comparison figures."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plt.rcParams["axes.unicode_minus"] = False

        generated: list[str] = []

        scenarios = list(dict.fromkeys(raw_df["scenario"].tolist()))
        agents = list(dict.fromkeys(raw_df["agent"].tolist()))
        palette = sns.color_palette("tab10", n_colors=max(len(agents), 6))

        # ------------------------------------------------------------------
        # FIG 1: Grouped Bar Chart with Error Bars — Average Wait Time (s)
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(scenarios))
        bar_w = 0.80 / max(1, len(agents))

        for i, ag in enumerate(agents):
            means = []
            stds = []
            for sc in scenarios:
                sub = summary_df[(summary_df["scenario"] == sc) & (summary_df["agent"] == ag)]
                means.append(float(sub["avg_wait_mean"].iloc[0]) if not sub.empty else 0.0)
                stds.append(float(sub["avg_wait_std"].iloc[0]) if not sub.empty else 0.0)
            offset = (i - (len(agents) - 1) / 2.0) * bar_w
            ax.bar(
                x + offset,
                means,
                yerr=stds,
                width=bar_w,
                capsize=3,
                label=ag,
                color=palette[i],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=11)
        ax.set_title("So sánh Thời gian chờ trung bình (s) theo Thuật toán & Kịch bản", fontsize=13, fontweight="bold")
        ax.set_xlabel("Kịch bản giao thông", fontsize=11)
        ax.set_ylabel("Thời gian chờ trung bình (giây / xe)", fontsize=11)
        ax.legend(title="Thuật toán", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        p1 = os.path.join(self.output_dir, "fig1_avg_wait_comparison.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)
        generated.append(p1)

        # ------------------------------------------------------------------
        # FIG 2: Max Wait Time Starvation Suppression Boxplot + Threshold
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=raw_df, x="scenario", y="max_wait_time", hue="agent", palette="Set2", ax=ax)
        ax.axhline(
            y=STARVATION_THRESHOLD,
            color="orange",
            linestyle="--",
            linewidth=1.3,
            label=f"Ngưỡng cảnh báo ({STARVATION_THRESHOLD:.0f}s)",
        )
        ax.axhline(
            y=float(T_MAX),
            color="red",
            linestyle=":",
            linewidth=1.3,
            label=f"Ngưỡng T_max ({T_MAX:.0f}s)",
        )
        ax.set_title("Phân bố Thời gian chờ cực đại (s) & Chống đói đèn xanh (MDP v2.0)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Kịch bản giao thông", fontsize=11)
        ax.set_ylabel("Thời gian chờ cực đại ghi nhận (giây)", fontsize=11)
        ax.legend(title="Thuật toán / Ngưỡng", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        p2 = os.path.join(self.output_dir, "fig2_max_wait_starvation.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)
        generated.append(p2)

        # ------------------------------------------------------------------
        # FIG 3: Throughput Comparison (veh/min and total veh / 30 min)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.barplot(data=raw_df, x="scenario", y="passed_per_min", hue="agent", palette="viridis", errorbar="sd", ax=axes[0])
        axes[0].set_title("Thông lượng trung bình (xe / phút)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Kịch bản", fontsize=10)
        axes[0].set_ylabel("Xe qua / phút", fontsize=10)
        axes[0].tick_params(axis="x", rotation=15)
        if axes[0].legend_:
            axes[0].legend_.remove()

        sns.barplot(data=raw_df, x="scenario", y="total_passed", hue="agent", palette="viridis", errorbar="sd", ax=axes[1])
        axes[1].set_title("Tổng lượng xe giải phóng (30 phút / tập)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Kịch bản", fontsize=10)
        axes[1].set_ylabel("Tổng xe thoát (xe)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].legend(title="Thuật toán", bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.suptitle("Đối sánh Thông lượng giải phóng nút giao giữa các thuật toán", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p3 = os.path.join(self.output_dir, "fig3_throughput_comparison.png")
        plt.savefig(p3, dpi=300)
        plt.close(fig)
        generated.append(p3)

        # ------------------------------------------------------------------
        # FIG 4: Training Convergence Curves from runs/ Logs (Unified Scenario)
        # ------------------------------------------------------------------
        curves = self._collect_training_curves()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        if curves:
            for idx, (label, df_c) in enumerate(curves.items()):
                ep_vals = df_c["episode"].values
                rew_vals = df_c["cumulative_reward"].values
                ax.plot(ep_vals, rew_vals, alpha=0.25, color=palette[idx])
                win = max(1, min(15, len(rew_vals) // 4 or 1))
                smooth = pd.Series(rew_vals).rolling(window=win, min_periods=1).mean().values
                ax.plot(ep_vals, smooth, linewidth=2.2, label=f"{label} (Train)", color=palette[idx])

                # Also overlay deterministic eval_reward markers if present (especially helpful for MIXED)
                if "eval_reward" in df_c.columns:
                    ev_sub = df_c[["episode", "eval_reward"]].copy()
                    ev_sub["eval_reward"] = pd.to_numeric(ev_sub["eval_reward"], errors="coerce")
                    ev_sub = ev_sub.dropna()
                    if len(ev_sub) > 1:
                        ax.plot(
                            ev_sub["episode"].values,
                            ev_sub["eval_reward"].values,
                            linestyle="--",
                            marker="o",
                            markersize=3.0,
                            linewidth=1.4,
                            alpha=0.85,
                            color=palette[idx],
                            label=f"{label} (Det. Eval)",
                        )
            ax.legend(title="Thuật toán RL", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.text(
                0.5,
                0.5,
                "Chưa có dữ liệu nhật ký trong runs/\nHãy chạy `python -m training.train_all` trước.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"Đường cong hội tụ Phần thưởng ({self.train_scenario})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Tập huấn luyện (Episode)", fontsize=11)
        ax.set_ylabel("Phần thưởng tích lũy (Cumulative Reward)", fontsize=11)
        plt.tight_layout()
        p4 = os.path.join(self.output_dir, "fig4_training_curves.png")
        plt.savefig(p4, dpi=300)
        plt.close(fig)
        generated.append(p4)

        # ------------------------------------------------------------------
        # FIG 5: Standard DQN vs Double DQN Q-Value Overestimation Bias
        # ------------------------------------------------------------------
        q_curves = self._collect_qvalue_curves()
        fig, ax = plt.subplots(figsize=(10.5, 5.5))
        if q_curves:
            q_colors = {"Standard DQN": "#ef4444", "Double DQN": "#2563eb"}
            for label, df_q in q_curves.items():
                x_vals = df_q["step"].values
                q_vals = df_q["mean_q"].values
                col = q_colors.get(label, palette[0])
                ax.plot(x_vals, q_vals, alpha=0.25, color=col)
                win = max(1, min(15, len(q_vals) // 5 or 1))
                smooth_q = pd.Series(q_vals).rolling(window=win, min_periods=1).mean().values
                ax.plot(x_vals, smooth_q, linewidth=2.3, label=label, color=col)
            ax.legend(title="Ước lượng Q-Value", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.text(
                0.5,
                0.5,
                "Chưa có dữ liệu mean_q cho Standard DQN & Double DQN.\nHãy chạy huấn luyện chính thức trước.",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(
            f"So sánh Giá trị Q trung bình: Standard DQN vs Double DQN ({self.train_scenario})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Tập / Bước huấn luyện", fontsize=11)
        ax.set_ylabel("Giá trị Q trung bình (Mean Q-Value)", fontsize=11)
        plt.tight_layout()
        p5 = os.path.join(self.output_dir, "fig5_dqn_vs_ddqn_qvalue.png")
        plt.savefig(p5, dpi=300)
        plt.close(fig)
        generated.append(p5)

        print(f"[BenchmarkRunner] Saved {len(generated)} figures (300 DPI) -> {self.output_dir}")
        return generated


def run_full_benchmark(
    num_seeds: int = 20,
    scenarios: Optional[list[str]] = None,
    base_seed: int = 1000,
    max_steps: Optional[int] = None,
    model_tag: str = "MIXED",
    train_scenario: Optional[str] = None,
    strict: bool = False,
    include_webster: bool = False,
    output_dir: str = os.path.join(PROJECT_ROOT, "results"),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    runner = BenchmarkRunner(
        scenarios=scenarios,
        model_tag=model_tag,
        train_scenario=train_scenario,
        strict=strict,
        include_webster=include_webster,
        output_dir=output_dir,
    )
    raw_df = runner.run_benchmark(
        num_seeds=num_seeds,
        base_seed=base_seed,
        max_steps_per_episode=max_steps,
    )
    summary_df = runner.build_summary(raw_df)
    runner.export_csvs(raw_df, summary_df)
    figures = runner.plot_all_figures(raw_df, summary_df)
    return raw_df, summary_df, figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Scenario RL Benchmark & Figure Export")
    parser.add_argument("--seeds", type=int, default=20, help="Number of fixed random seeds [1000..1000+N-1]")
    parser.add_argument("--base-seed", type=int, default=1000, help="Starting seed integer")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(ALL_SCENARIOS),
        choices=ALL_SCENARIOS,
        help="List of benchmark scenarios to evaluate",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="MIXED",
        help="Checkpoint tag to evaluate across scenarios ('MIXED', 'AUTO', or specific scenario e.g. 'SC3_ASYMMETRIC')",
    )
    parser.add_argument(
        "--train-scenario",
        type=str,
        default=None,
        help="Training scenario log to display in fig4 & fig5 (defaults to model-tag or MIXED)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise FileNotFoundError if any RL model checkpoint is missing",
    )
    parser.add_argument(
        "--include-webster",
        action="store_true",
        help="Include calibrated WebsterAgent alongside FixedTime and MaxPressure",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap per episode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Destination directory for CSVs and 300 DPI PNG figures",
    )
    args = parser.parse_args()

    print("=" * 90)
    print(
        f"  BENCHMARK RUNNER | ModelTag: {args.model_tag} | Scenarios: {args.scenarios} | "
        f"Seeds: {args.seeds} ([{args.base_seed}..{args.base_seed + args.seeds - 1}])"
    )
    print("=" * 90)
    _, summary_df, figs = run_full_benchmark(
        num_seeds=args.seeds,
        scenarios=args.scenarios,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        model_tag=args.model_tag,
        train_scenario=args.train_scenario,
        strict=args.strict,
        include_webster=args.include_webster,
        output_dir=args.output_dir,
    )
    print("\nSummary Preview:")
    print(summary_df[["scenario", "agent", "avg_wait_str", "max_wait_str", "throughput_min_str"]].to_string(index=False))
    print("\nGenerated Figures:")
    for f in figs:
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
