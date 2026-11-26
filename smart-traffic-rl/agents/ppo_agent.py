"""Masked Proximal Policy Optimization (PPO) Agent with GAE & ONNX Export.

Implements Policy-Based / Actor-Critic RL algorithm tailored for TrafficLightEnv:
1. ActorCriticNetwork: Separate Actor (10->128->128->2) & Critic (10->128->128->1)
   MLPs with Tanh activations and Orthogonal Weight Initialization.
2. Masked Categorical Policy: Strictly applies action masking via
   masked_logits = logits + (1 - m_t) * (-1e9).
3. RolloutBuffer & Generalized Advantage Estimation (GAE, gamma=0.95, lambda=0.95).
4. PPO-Clip Clipped Surrogate Objective + Value Function Loss + Entropy Bonus.
5. Serialization (.pth) and Inference ONNX Export ((obs, action_mask) -> (action_probs, deterministic_action)).
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.config import (
    STATE_DIM,
    ACTION_DIM,
    ACTION_KEEP,
    PPO_CONFIG,
)
from .base_agent import BaseAgent


def _orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    """Apply Orthogonal initialization to weights and zero initialization to biases."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCriticNetwork(nn.Module):
    """Decoupled Actor-Critic MLP architecture with Tanh activations and Orthogonal init.

    Actor:  Linear(10, 128) -> Tanh -> Linear(128, 128) -> Tanh -> Linear(128, 2)
    Critic: Linear(10, 128) -> Tanh -> Linear(128, 128) -> Tanh -> Linear(128, 1)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dims: tuple[int, int] = tuple(PPO_CONFIG["hidden_dims"]),
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        hidden_gain = float(np.sqrt(2.0))

        self.actor = nn.Sequential(
            _orthogonal_init(nn.Linear(state_dim, h1), gain=hidden_gain),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(h1, h2), gain=hidden_gain),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(h2, action_dim), gain=0.01),
        )

        self.critic = nn.Sequential(
            _orthogonal_init(nn.Linear(state_dim, h1), gain=hidden_gain),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(h1, h2), gain=hidden_gain),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(h2, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute unmasked action logits (B, 2) and state value estimates V_phi(s) (B, 1)."""
        action_logits = self.actor(obs)
        state_values = self.critic(obs)
        return action_logits, state_values


class PPOOnnxWrapper(nn.Module):
    """ONNX Export Wrapper accepting (obs, action_mask) and outputting (action_probs, deterministic_action)."""

    def __init__(self, actor_critic: ActorCriticNetwork) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.actor_critic(obs)
        masked_logits = logits + (1.0 - action_mask) * (-1e9)
        action_probs = torch.softmax(masked_logits, dim=-1)
        deterministic_action = torch.argmax(masked_logits, dim=-1)
        return action_probs, deterministic_action


class RolloutBuffer:
    """On-policy trajectory storage supporting Generalized Advantage Estimation (GAE)."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.action_masks: list[np.ndarray] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: np.ndarray,
    ) -> None:
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.action_masks.append(np.array(action_mask, dtype=np.float32))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.states)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute discounted returns R_hat_t and GAE advantages A_hat_t."""
        n = len(self.states)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            non_terminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * non_terminal * next_value - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        values_arr = np.asarray(self.values, dtype=np.float32)
        returns = advantages + values_arr
        return returns, advantages


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO-Clip) Agent with Masked Categorical Policy & GAE."""

    def __init__(
        self,
        hidden_dims: tuple[int, int] = tuple(PPO_CONFIG["hidden_dims"]),
        lr: float = float(PPO_CONFIG["lr"]),
        gamma: float = float(PPO_CONFIG["gamma"]),
        gae_lambda: float = float(PPO_CONFIG["gae_lambda"]),
        clip_eps: float = float(PPO_CONFIG["clip_eps"]),
        entropy_coef: float = float(PPO_CONFIG["entropy_coef"]),
        value_coef: float = float(PPO_CONFIG["value_coef"]),
        max_grad_norm: float = float(PPO_CONFIG["max_grad_norm"]),
        ppo_epochs: int = int(PPO_CONFIG["ppo_epochs"]),
        mini_batch_size: int = int(PPO_CONFIG["mini_batch_size"]),
        rollout_steps: int = int(PPO_CONFIG["rollout_steps"]),
        device: Optional[str] = None,
        name: str = "PPO-Clip",
    ) -> None:
        super().__init__(name=name)

        self.gamma: float = float(gamma)
        self.gae_lambda: float = float(gae_lambda)
        self.clip_eps: float = float(clip_eps)
        self.entropy_coef: float = float(entropy_coef)
        self.value_coef: float = float(value_coef)
        self.max_grad_norm: float = float(max_grad_norm)
        self.ppo_epochs: int = int(ppo_epochs)
        self.mini_batch_size: int = int(mini_batch_size)
        self.rollout_steps: int = int(rollout_steps)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network: ActorCriticNetwork = ActorCriticNetwork(
            state_dim=STATE_DIM,
            hidden_dims=hidden_dims,
            action_dim=ACTION_DIM,
        ).to(self.device)

        self.optimizer: optim.Optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.rollout_buffer: RolloutBuffer = RolloutBuffer()

        # Temporary cache bridging select_action() -> update()
        self._last_log_prob: Optional[float] = None
        self._last_value: Optional[float] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_obs_cache: Optional[np.ndarray] = None
        self.total_steps: int = 0

    def reset(self) -> None:
        """Reset transient per-step action cache at episode boundaries."""
        self._last_log_prob = None
        self._last_value = None
        self._last_mask = None
        self._last_obs_cache = None

    def _compute_masked_dist(
        self,
        obs_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """Compute masked Categorical policy distribution, masked logits, and state value."""
        logits, values = self.network(obs_tensor)
        masked_logits = logits + (1.0 - mask_tensor) * (-1e9)
        dist = Categorical(logits=masked_logits)
        return dist, masked_logits, values.squeeze(-1)

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select action using Masked Categorical Policy (deterministic argmax if eval_mode=True)."""
        action_mask = np.asarray(
            info.get("action_mask", np.array([1, 1], dtype=np.int8)),
            dtype=np.float32,
        )
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            action_mask = np.ones(ACTION_DIM, dtype=np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            dist, masked_logits, value = self._compute_masked_dist(obs_t, mask_t)
            if eval_mode:
                action_t = torch.argmax(masked_logits, dim=-1)
                self._last_log_prob = None
                self._last_value = None
                self._last_mask = None
                self._last_obs_cache = None
            else:
                action_t = dist.sample()
                log_prob_t = dist.log_prob(action_t)
                self._last_log_prob = float(log_prob_t.item())
                self._last_value = float(value.item())
                self._last_mask = action_mask.copy()
                self._last_obs_cache = np.array(obs, dtype=np.float32)

        chosen = int(action_t.item())
        if len(valid_actions) > 0 and action_mask[chosen] < 0.5:
            chosen = int(valid_actions[0])
        return chosen

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, Any],
        next_info: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Store step transition into RolloutBuffer and trigger PPO optimization when buffer is full."""
        action_mask = np.asarray(
            info.get("action_mask", np.array([1, 1], dtype=np.int8)),
            dtype=np.float32,
        )

        # Reuse cached log_prob / value from select_action if available, otherwise compute cleanly
        if (
            self._last_log_prob is not None
            and self._last_value is not None
            and self._last_obs_cache is not None
            and np.allclose(self._last_obs_cache, obs)
        ):
            log_prob = self._last_log_prob
            value = self._last_value
            stored_mask = self._last_mask if self._last_mask is not None else action_mask
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_t = torch.tensor([int(action)], dtype=torch.int64, device=self.device)
            self.network.eval()
            with torch.no_grad():
                dist, _, val_t = self._compute_masked_dist(obs_t, mask_t)
                log_prob = float(dist.log_prob(act_t).item())
                value = float(val_t.item())
            stored_mask = action_mask

        self.rollout_buffer.push(
            state=obs,
            action=int(action),
            log_prob=log_prob,
            reward=float(reward),
            done=bool(done),
            value=value,
            action_mask=stored_mask,
        )
        self.total_steps += 1

        # Clear transient single-step cache
        self._last_log_prob = None
        self._last_value = None
        self._last_mask = None
        self._last_obs_cache = None

        if len(self.rollout_buffer) >= self.rollout_steps:
            return self._optimize(last_next_obs=next_obs, last_done=done)
        return None

    def flush_update(
        self,
        last_next_obs: Optional[np.ndarray] = None,
        last_done: bool = True,
    ) -> dict[str, Any] | None:
        """Force a PPO update on remaining buffered transitions (useful at end of short training runs)."""
        if len(self.rollout_buffer) == 0:
            return None
        return self._optimize(last_next_obs=last_next_obs, last_done=last_done)

    def _optimize(
        self,
        last_next_obs: Optional[np.ndarray],
        last_done: bool,
    ) -> dict[str, float]:
        """Compute GAE returns and run PPO-Clip gradient updates over collected rollout."""
        if last_next_obs is None or last_done:
            last_value = 0.0
        else:
            next_t = torch.tensor(last_next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.network.eval()
            with torch.no_grad():
                _, val_t = self.network(next_t)
                last_value = float(val_t.squeeze().item())

        returns_np, adv_np = self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        states_t = torch.tensor(np.array(self.rollout_buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.rollout_buffer.actions), dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.tensor(np.array(self.rollout_buffer.log_probs), dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(np.array(self.rollout_buffer.action_masks), dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(adv_np, dtype=torch.float32, device=self.device)

        n_samples = len(self.rollout_buffer)
        batch_size = min(self.mini_batch_size, n_samples)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        self.network.train()
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                mb_idx = indices[start : start + batch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_masks = masks_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]

                # Normalize advantages per mini-batch
                if mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

                dist, _, values = self._compute_masked_dist(mb_states, mb_masks)
                new_lp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - mb_returns) ** 2).mean()
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.rollout_buffer.clear()

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, filepath: str) -> None:
        """Save ActorCriticNetwork state_dict and optimizer metadata to .pth file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load ActorCriticNetwork weights from .pth file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
                self.network.load_state_dict(checkpoint["network_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.total_steps = int(checkpoint.get("total_steps", self.total_steps))
            else:
                self.network.load_state_dict(checkpoint)

    def export_onnx(self, filepath: str) -> None:
        """Export masked Actor policy to ONNX accepting (obs, action_mask) -> (action_probs, deterministic_action)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        cpu_net = copy.deepcopy(self.network).to("cpu").eval()
        wrapper = PPOOnnxWrapper(cpu_net).eval()

        dummy_obs = torch.zeros(1, STATE_DIM, dtype=torch.float32)
        dummy_mask = torch.ones(1, ACTION_DIM, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                (dummy_obs, dummy_mask),
                filepath,
                input_names=["obs", "action_mask"],
                output_names=["action_probs", "deterministic_action"],
                dynamic_axes={
                    "obs": {0: "batch_size"},
                    "action_mask": {0: "batch_size"},
                    "action_probs": {0: "batch_size"},
                    "deterministic_action": {0: "batch_size"},
                },
                opset_version=17,
                dynamo=False,
            )
