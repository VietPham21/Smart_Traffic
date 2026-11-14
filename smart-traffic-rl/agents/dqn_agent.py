"""Deep Q-Network (DQN) & Double DQN Agent with Action Masking (Tier 3 Deep RL).

Implements:
1. QNetwork: PyTorch Multi-Layer Perceptron (MLP: 10 -> 64 -> 64 -> 2).
2. ReplayBuffer: Experience replay with action mask preservation.
3. DQNAgent: Unified agent supporting both Standard DQN and Double DQN
   with parameter `use_double_dqn: bool = True`.
Maintainer: Vũ Bá Đài (MSV: 2351170577)
"""

from __future__ import annotations

import os
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
            epsilon_decay: Epsilon multiplicative decay per update.
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
        # TODO [Vũ Bá Đài]: Hoàn thiện thuật toán Double DQN với Action Masking trên PyTorch
        action_mask = info.get("action_mask", np.array([1, 1], dtype=np.int8))
        valid_actions = np.where(action_mask == 1)[0]
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

        # Mask invalid actions with large negative value (-1e9)
        masked_q = np.full_like(q_values, -1e9)
        masked_q[valid_actions] = q_values[valid_actions]

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
    ) -> dict[str, Any] | None:
        """Store experience and train network on mini-batch when buffer is ready."""
        # TODO [Vũ Bá Đài]: Cập nhật mạng nơ-ron qua Mini-batch và Decoupled Double DQN target
        next_action_mask = next_info.get("action_mask", np.array([1, 1], dtype=np.int8))

        # 1. Store transition in replay buffer
        self.replay_buffer.push(obs, action, reward, next_obs, done, next_action_mask)
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

        # 4. Target Q calculation with Action Masking
        with torch.no_grad():
            online_next_q = self.online_net(next_states)
            # Mask invalid actions in next states with -1e9
            invalid_mask = (next_masks == 0.0)
            online_next_q[invalid_mask] = -1e9

            if self.use_double_dqn:
                # Double DQN: action selected by Online Net, evaluated by Target Net
                best_next_actions = online_next_q.argmax(dim=1, keepdim=True)
                target_next_q = self.target_net(next_states).gather(1, best_next_actions)
            else:
                # Standard DQN: max action evaluated by Target Net
                target_all_next_q = self.target_net(next_states)
                target_all_next_q[invalid_mask] = -1e9
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

        # 7. Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "loss": float(loss.item()),
            "q_val": float(current_q.mean().item()),
            "epsilon": float(self.epsilon),
        }

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
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load PyTorch network weights."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.online_net.load_state_dict(checkpoint["online_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.total_steps = checkpoint.get("total_steps", self.total_steps)
