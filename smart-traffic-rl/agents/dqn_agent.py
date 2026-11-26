"""Deep Q-Network (DQN) & Double DQN Agent with Action Masking (Tier 3 Deep RL).

Implements:
1. QNetwork: PyTorch Multi-Layer Perceptron (MLP: 10 -> 64 -> 64 -> 2).
2. ReplayBuffer: Experience replay with action mask preservation.
3. DQNAgent: Unified agent supporting both Standard DQN and Double DQN
   with parameter `use_double_dqn: bool = True`.
Maintainer: Vũ Bá Đài (MSV: 2351170577)
"""

from __future__ import annotations

import copy
import os
import warnings
from collections import deque
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.config import (
    STATE_DIM,
    ACTION_DIM,
    ACTION_KEEP,
    ACTION_SWITCH,
    GAMMA,
    DQN_HIDDEN_DIMS,
    DQN_BUFFER_CAPACITY,
    DQN_BATCH_SIZE,
    DQN_LEARNING_RATE,
    DQN_TARGET_UPDATE_FREQ,
    DQN_MIN_REPLAY_SIZE,
    DQN_EPSILON_START,
    DQN_EPSILON_MIN,
    DQN_EPSILON_DECAY,
)
from .base_agent import BaseAgent


class QNetwork(nn.Module):
    """Multi-Layer Perceptron Q-Network mapping 10-D state to 2 discrete action Q-values.

    Architecture: Input(10) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Output(2)
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden_dims: tuple[int, int] = DQN_HIDDEN_DIMS,
        output_dim: int = ACTION_DIM,
    ) -> None:
        """Initialize MLP layers.

        Args:
            input_dim: State vector dimension (10).
            hidden_dims: Tuple of hidden layer units (64, 64).
            output_dim: Number of discrete actions (2).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unconstrained Q-values for all actions."""
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity experience replay buffer preserving action safety masks."""

    def __init__(self, capacity: int = DQN_BUFFER_CAPACITY) -> None:
        """Initialize circular buffer."""
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        """Store experience tuple into buffer."""
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
                np.array(next_action_mask, dtype=np.float32),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions converted to PyTorch tensors."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        t_states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        t_actions = torch.tensor(np.array(actions), dtype=torch.int64, device=device).unsqueeze(1)
        t_rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        t_next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        t_dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)
        t_next_masks = torch.tensor(np.array(next_masks), dtype=torch.float32, device=device)

        return t_states, t_actions, t_rewards, t_next_states, t_dones, t_next_masks

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNOnnxWrapper(nn.Module):
    """Inference-ready ONNX wrapper mapping (obs, action_mask) -> (masked_q_values, best_action)."""

    def __init__(self, q_net: QNetwork) -> None:
        super().__init__()
        self.q_net = q_net

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_values = self.q_net(obs)
        masked_q_values = q_values + (1.0 - action_mask) * (-1e9)
        best_action = torch.argmax(masked_q_values, dim=-1)
        return masked_q_values, best_action


class DQNAgent(BaseAgent):
    """Tier 3 Benchmark: Deep Q-Network Agent supporting both Standard DQN and Double DQN."""

    def __init__(
        self,
        use_double_dqn: bool = True,
        lr: float = DQN_LEARNING_RATE,
        gamma: float = GAMMA,
        buffer_capacity: int = DQN_BUFFER_CAPACITY,
        batch_size: int = DQN_BATCH_SIZE,
        target_update_freq: int = DQN_TARGET_UPDATE_FREQ,
        min_replay_size: int = DQN_MIN_REPLAY_SIZE,
        epsilon_start: float = DQN_EPSILON_START,
        epsilon_min: float = DQN_EPSILON_MIN,
        epsilon_decay: float = DQN_EPSILON_DECAY,
        decay_on_episode_end: bool = True,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize DQN / Double DQN agent.

        Args:
            use_double_dqn: If True, uses Double DQN target evaluation to eliminate overestimation bias.
            lr: Learning rate for Adam optimizer.
            gamma: Discount factor.
            buffer_capacity: Maximum size of experience replay memory.
            batch_size: Minibatch size for gradient descent.
            target_update_freq: Steps between target network weight synchronizations.
            min_replay_size: Warm-up steps before gradient descent starts.
            epsilon_start: Initial epsilon.
            epsilon_min: Floor epsilon.
            epsilon_decay: Epsilon multiplicative decay rate.
            decay_on_episode_end: If True, decays epsilon at episode end via on_episode_end().
            device: 'cuda', 'cpu', or None (auto-detect).
            name: Custom agent name.
        """
        agent_name = name or ("Double-DQN" if use_double_dqn else "Standard-DQN")
        super().__init__(name=agent_name)

        self.use_double_dqn: bool = use_double_dqn
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.target_update_freq: int = target_update_freq
        self.min_replay_size: int = min_replay_size
        self.epsilon: float = epsilon_start
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay
        self.decay_on_episode_end: bool = decay_on_episode_end

        # Select computation device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks: Online Q_theta and Target Q_theta_minus
        self.online_net: QNetwork = QNetwork().to(self.device)
        self.target_net: QNetwork = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer: optim.Optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn: nn.Module = nn.SmoothL1Loss()
        self.replay_buffer: ReplayBuffer = ReplayBuffer(capacity=buffer_capacity)

        self.total_steps: int = 0

    def select_action(
        self,
        obs: np.ndarray,
        info: dict[str, Any],
        eval_mode: bool = False,
    ) -> int:
        """Select action using epsilon-greedy policy with Action Masking."""
        action_mask = np.asarray(
            info.get("action_mask", np.array([1, 1], dtype=np.int8)),
            dtype=np.float32,
        )
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            return ACTION_KEEP

        current_eps = 0.0 if eval_mode else self.epsilon

        # Exploration
        if np.random.rand() < current_eps:
            return int(np.random.choice(valid_actions))

        # Exploitation via Online Q-Network
        state_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor).squeeze(0).cpu().numpy()

        # Áp dụng Action Masking bằng cách cộng (1 - m) * (-1e9)
        masked_q = q_values + (1.0 - action_mask) * np.float32(-1e9)
        return int(np.argmax(masked_q))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: dict[str, Any],
        next_info: dict[str, Any],
        truncated: bool = False,
    ) -> dict[str, Any] | None:
        """Store experience and train network on mini-batch when buffer is ready."""
        next_action_mask = next_info.get("action_mask", np.array([1, 1], dtype=np.int8))
        is_truncated = bool(truncated or next_info.get("is_truncated", False))
        # Only absorbing terminal transitions zero out future bootstrap value; time-limit truncations bootstrap V/Q!
        bootstrap_done = bool(done and not is_truncated)

        # 1. Store transition in replay buffer
        self.replay_buffer.push(obs, action, reward, next_obs, bootstrap_done, next_action_mask)
        self.total_steps += 1

        # Check warm-up condition
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # 2. Sample mini-batch
        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            self.batch_size,
            self.device,
        )

        # 3. Current Q-value estimation: Q_online(s, a)
        self.online_net.train()
        current_q = self.online_net(states).gather(1, actions)

        # 4. Target Q calculation with Action Masking via (1 - m) * (-1e9)
        with torch.no_grad():
            mask_penalty = (1.0 - next_masks) * (-1e9)
            online_next_q = self.online_net(next_states) + mask_penalty

            if self.use_double_dqn:
                # Double DQN: action selected by Online Net, evaluated by Target Net
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.target_net(next_states).gather(1, best_next_actions)
            else:
                # Standard DQN: max action evaluated by Target Net
                target_all_next_q = self.target_net(next_states) + mask_penalty
                target_next_q = target_all_next_q.max(dim=1, keepdim=True)[0]

            expected_q = rewards + (1.0 - dones) * self.gamma * target_next_q

        # 5. Gradient descent step
        loss = self.loss_fn(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 6. Periodic target network update
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # 7. Optional per-step epsilon decay if episodic decay is disabled
        if not self.decay_on_episode_end:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        td_loss_val = float(loss.item())
        mean_q_val = float(current_q.mean().item())
        return {
            "td_loss": td_loss_val,
            "mean_q": mean_q_val,
            "epsilon": float(self.epsilon),
            "loss": td_loss_val,
            "q_val": mean_q_val,
        }

    def on_episode_end(self) -> None:
        """Decay epsilon multiplicatively at the end of an episode."""
        if self.decay_on_episode_end:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save PyTorch network weights."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "use_double_dqn": self.use_double_dqn,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load PyTorch network weights."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "online_state_dict" in checkpoint:
                self.online_net.load_state_dict(checkpoint["online_state_dict"])
                self.target_net.load_state_dict(
                    checkpoint.get("target_state_dict", checkpoint["online_state_dict"])
                )
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
                self.total_steps = int(checkpoint.get("total_steps", self.total_steps))
            else:
                self.online_net.load_state_dict(checkpoint)
                self.target_net.load_state_dict(checkpoint)

    def export_onnx(self, filepath: str) -> None:
        """Export masked Q-Network to ONNX taking (obs, action_mask) -> (masked_q_values, best_action)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        cpu_net = copy.deepcopy(self.online_net).to("cpu").eval()
        wrapper = DQNOnnxWrapper(cpu_net).eval()

        dummy_obs = torch.zeros(1, STATE_DIM, dtype=torch.float32)
        dummy_mask = torch.ones(1, ACTION_DIM, dtype=torch.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                (dummy_obs, dummy_mask),
                filepath,
                input_names=["obs", "action_mask"],
                output_names=["masked_q_values", "best_action"],
                dynamic_axes={
                    "obs": {0: "batch_size"},
                    "action_mask": {0: "batch_size"},
                    "masked_q_values": {0: "batch_size"},
                    "best_action": {0: "batch_size"},
                },
                opset_version=17,
                dynamo=False,
            )
