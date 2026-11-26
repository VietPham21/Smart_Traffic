"""Agents package for SmartTraffic RL Studio.

Includes:
- BaseAgent: Interface contract for all signal control agents.
- Baselines: FixedTimeAgent, WebsterAgent, MaxPressureAgent.
- Tabular RL: StateDiscretizer (972 states), QLearningAgent.
- Deep RL: QNetwork (MLP 10->64->64->2), ReplayBuffer, DQNAgent (DQN & Double DQN).
"""

from .base_agent import BaseAgent
from .baselines import FixedTimeAgent, WebsterAgent, MaxPressureAgent
from .q_learning import StateDiscretizer, QLearningAgent
from .dqn_agent import QNetwork, ReplayBuffer, DQNAgent
from .ppo_agent import ActorCriticNetwork, RolloutBuffer, PPOAgent

__all__ = [
    "BaseAgent",
    "FixedTimeAgent",
    "WebsterAgent",
    "MaxPressureAgent",
    "StateDiscretizer",
    "QLearningAgent",
    "QNetwork",
    "ReplayBuffer",
    "DQNAgent",
    "ActorCriticNetwork",
    "RolloutBuffer",
    "PPOAgent",
]
