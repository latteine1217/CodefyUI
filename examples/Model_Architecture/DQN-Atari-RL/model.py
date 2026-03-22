"""
Deep Q-Network (DQN) with CNN Feature Extractor -- Atari-Style Agent

DQN, introduced by Mnih et al. (2015) at DeepMind, was the first deep reinforcement
learning algorithm to achieve human-level performance on Atari 2600 games directly
from raw pixel inputs. It combines Q-learning with deep convolutional neural networks
and two critical stabilization techniques:

  1. Experience Replay: Stores transitions in a buffer and samples random mini-batches,
     breaking temporal correlations and improving data efficiency.
  2. Target Network: A slowly-updated copy of the Q-network used to compute stable
     TD targets, preventing the "moving target" problem in bootstrapped updates.

Significance:
  - Sparked the modern deep RL revolution and led directly to AlphaGo / AlphaZero.
  - Foundation for game-playing AI (Atari, StarCraft, Dota 2).
  - Core building block in offline RL and RLHF pipelines that align large language
    models with human preferences (the "reward model" in RLHF is trained similarly).

This script implements a full DQN training loop with a CNN backbone designed for
84x84x4 stacked grayscale frames (standard Atari preprocessing). When gymnasium
is available it trains on CartPole-v1 with synthetic frame observations; otherwise
it falls back to a lightweight synthetic environment.

CodefyUI -- visual node-graph builder for ML pipelines.
"""

import math
import random
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# CNN Feature Extractor (Nature-DQN architecture)
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """Three-layer CNN that maps 84x84x4 frames to a 3136-dim feature vector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ---------------------------------------------------------------------------
# DQN Model
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """Deep Q-Network: CNN backbone followed by two fully-connected layers."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.features = CNNFeatureExtractor()
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action given a batch of states."""
        return self.head(self.features(x))

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size FIFO buffer that stores (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)

# ---------------------------------------------------------------------------
# Simple Environment Wrapper
# ---------------------------------------------------------------------------

class SimpleEnv:
    """Wraps gymnasium CartPole or provides a synthetic fallback.

    Observations are projected into 4x84x84 pseudo-frames so the CNN
    backbone receives the tensor shape it expects.
    """

    def __init__(self) -> None:
        self.gym_env = None
        try:
            import gymnasium as gym
            self.gym_env = gym.make("CartPole-v1")
            self.n_actions = self.gym_env.action_space.n
        except Exception:
            self.n_actions = 2
            self._state = None

    def _to_frame(self, obs: np.ndarray) -> np.ndarray:
        """Project a low-dim observation into a 4x84x84 pseudo-image."""
        frame = np.zeros((4, 84, 84), dtype=np.float32)
        normed = obs.astype(np.float32)
        for c in range(4):
            idx = c % len(normed)
            frame[c] = normed[idx]
        return frame

    def reset(self) -> np.ndarray:
        if self.gym_env is not None:
            obs, _ = self.gym_env.reset()
        else:
            obs = np.random.randn(4).astype(np.float32) * 0.05
            self._state = obs.copy()
        return self._to_frame(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.gym_env is not None:
            obs, reward, terminated, truncated, _ = self.gym_env.step(action)
            done = terminated or truncated
        else:
            self._state += np.random.randn(4).astype(np.float32) * 0.1
            obs = self._state.copy()
            reward = 1.0
            done = bool(np.abs(self._state).max() > 2.5)
        return self._to_frame(obs), reward, done

# ---------------------------------------------------------------------------
# Soft (Polyak) target-network update
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    device = get_device()
    print(f"[DQN] Using device: {device}")

    env = SimpleEnv()
    n_actions = env.n_actions

    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=2.5e-4)
    loss_fn = nn.MSELoss()
    replay = ReplayBuffer(capacity=50_000)

    # Hyperparameters
    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 500
    batch_size = 64
    num_episodes = 300
    max_steps = 500
    learning_starts = 256
    tau = 0.005

    episode_rewards: list = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # Epsilon-greedy exploration with exponential decay
            eps = eps_end + (eps_start - eps_end) * math.exp(-episode / eps_decay)
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = int(policy_net(s_t).argmax(dim=1).item())

            next_state, reward, done = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            # Train once we have enough experience
            if len(replay) >= learning_starts:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                # Current Q-values for chosen actions
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # TD target using the frozen target network
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(dim=1).values
                    td_target = rewards + gamma * max_next_q * (1.0 - dones)

                loss = loss_fn(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()

                # Soft-update target network every step
                soft_update(target_net, policy_net, tau=tau)

            if done:
                break

        episode_rewards.append(total_reward)
        avg_10 = np.mean(episode_rewards[-10:])
        if episode % 10 == 0 or episode == 1:
            print(f"  Episode {episode:>4d}  |  Reward: {total_reward:6.1f}  |  "
                  f"Avg(10): {avg_10:6.1f}  |  eps: {eps:.3f}")

    print("[DQN] Training complete.")
    print(f"  Final 10-episode average reward: {np.mean(episode_rewards[-10:]):.1f}")


if __name__ == "__main__":
    train()
