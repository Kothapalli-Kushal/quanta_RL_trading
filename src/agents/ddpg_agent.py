"""
DDPG Agent with Safety-Critic and Conditional Safety Penalty (CSP).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional
import logging

from .networks import Actor, Critic, SafetyCritic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG Agent with:
    - Actor network
    - Critic network (Q-function)
    - Safety-Critic network (C-function)
    - Conditional Safety Penalty (CSP) in actor update
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 agent_id: int,
                 risk_threshold: float = 0.5,
                 risk_penalty: float = 1.0,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 lr_safety: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            agent_id: Agent identifier
            risk_threshold: Risk threshold θ_i
            risk_penalty: Risk penalty coefficient λ_i
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            lr_safety: Learning rate for safety-critic
            gamma: Discount factor
            tau: Soft update coefficient
            device: Device (cuda/cpu)
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.risk_threshold = risk_threshold
        self.risk_penalty = risk_penalty
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.safety_critic = SafetyCritic(state_dim, action_dim).to(device)
        self.safety_critic_target = SafetyCritic(state_dim, action_dim).to(device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.safety_optimizer = optim.Adam(self.safety_critic.parameters(), lr=lr_safety)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Noise for exploration
        self.noise_scale = 0.1
        
        logger.info(f"Initialized Agent {agent_id} with risk_threshold={risk_threshold}, risk_penalty={risk_penalty}")
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using actor network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def compute_env_risk_score(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Compute environment risk score C_env(s, a) ∈ [0, 1].
        
        Combines:
        - Portfolio concentration (from state)
        - Herfindahl-Hirschman Index
        - Leverage
        - Exposure relative to equity
        - Simulated volatility
        - Impact of proposed trade on recent realized volatility
        """
        # Extract holdings from state (normalized allocations)
        # State structure: [cash_norm, holdings_norm (n_assets), features (n_assets * 5)]
        n_assets = self.action_dim
        holdings_start_idx = 1
        holdings = state[holdings_start_idx:holdings_start_idx + n_assets]
        
        # Concentration (HHI)
        hhi = np.sum(holdings ** 2)
        
        # Max position concentration
        max_position = np.max(holdings) if len(holdings) > 0 else 0.0
        
        # Leverage (sum of absolute positions)
        leverage = np.sum(np.abs(holdings))
        
        # Action impact (magnitude of proposed trade)
        action_magnitude = np.linalg.norm(action)
        
        # Combined risk score (normalized to [0, 1])
        risk_score = (
            0.3 * hhi +  # Concentration
            0.3 * (max_position / 0.2) +  # Max position (normalized by 20% limit)
            0.2 * min(leverage, 1.0) +  # Leverage
            0.2 * min(action_magnitude / np.sqrt(n_assets), 1.0)  # Trade impact
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def update(self, batch_size: int = 64):
        """Update networks using batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Critic (Q-function)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Safety-Critic (C-function)
        # Compute environment risk scores for batch
        env_risk_scores = []
        for i in range(batch_size):
            state_np = states[i].cpu().numpy()
            action_np = actions[i].cpu().numpy()
            risk_score = self.compute_env_risk_score(state_np, action_np)
            env_risk_scores.append(risk_score)
        
        env_risk_scores = torch.FloatTensor(env_risk_scores).unsqueeze(1).to(self.device)
        
        predicted_risk = self.safety_critic(states, actions)
        safety_loss = nn.MSELoss()(predicted_risk, env_risk_scores)
        
        self.safety_optimizer.zero_grad()
        safety_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_critic.parameters(), 1.0)
        self.safety_optimizer.step()
        
        # Update Actor with Conditional Safety Penalty (CSP)
        # ∇J = ∇Q - λ_i * ∇ReLU(C_ξ - θ_i)
        actor_actions = self.actor(states)
        q_values = self.critic(states, actor_actions)
        
        # Q-value gradient (maximize)
        actor_loss_q = -q_values.mean()
        
        # Safety penalty gradient (only when C > θ)
        safety_scores = self.safety_critic(states, actor_actions)
        safety_violation = torch.clamp(safety_scores - self.risk_threshold, min=0.0)
        actor_loss_safety = self.risk_penalty * safety_violation.mean()
        
        # Total actor loss
        actor_loss = actor_loss_q + actor_loss_safety
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
        self._soft_update(self.safety_critic_target, self.safety_critic, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'safety_loss': safety_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': q_values.mean().item(),
            'safety_score': safety_scores.mean().item()
        }
    
    def _soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'safety_critic': self.safety_critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'safety_critic_target': self.safety_critic_target.state_dict(),
            'risk_threshold': self.risk_threshold,
            'risk_penalty': self.risk_penalty
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.safety_critic.load_state_dict(checkpoint['safety_critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.safety_critic_target.load_state_dict(checkpoint['safety_critic_target'])

