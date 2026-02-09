"""
DDPG (Deep Deterministic Policy Gradient) agent implementation.

Includes Actor, Critic, and Safety-Critic networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import copy

from .base_agent import BaseAgent
from .safety_critic import SafetyCritic, SafetyCriticTrainer
from ..config import SignalMARSConfig


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize actor.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        super(Actor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (action logits)
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))  # Portfolio weights sum to 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch, state_dim)
        
        Returns:
            Action tensor (batch, action_dim) - portfolio weights
        """
        return self.network(state)


class Critic(nn.Module):
    """Critic network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize critic.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (Q-value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (batch, state_dim)
            action: Action tensor (batch, action_dim)
        
        Returns:
            Q-value tensor (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent(BaseAgent):
    """
    DDPG agent with safety critic.
    
    Implements Deep Deterministic Policy Gradient with:
    - Actor network (policy)
    - Critic network (Q-function)
    - Safety critic network (risk prediction)
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_type: str,
                 config: SignalMARSConfig, device: str = "cpu"):
        """
        Initialize DDPG agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            agent_type: Agent type identifier
            config: Configuration object
            device: Device to run on
        """
        super().__init__(state_dim, action_dim, agent_type, device)
        
        self.config = config
        agent_config = config.agent
        
        # Networks
        self.actor = Actor(state_dim, action_dim, agent_config.actor_hidden_dims).to(device)
        self.critic = Critic(state_dim, action_dim, agent_config.critic_hidden_dims).to(device)
        self.safety_critic = SafetyCritic(
            state_dim, action_dim, agent_config.safety_critic_hidden_dims
        ).to(device)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_config.critic_lr
        )
        self.safety_trainer = SafetyCriticTrainer(
            self.safety_critic, agent_config.safety_critic_lr
        )
        
        # Exploration
        self.noise_std = agent_config.noise_std
        self.noise_decay = agent_config.noise_decay
        
        # Feature masking (for signal specialization)
        self.use_feature_masking = agent_config.use_feature_masking
        self.mask_probability = agent_config.mask_probability
        
        # Risk threshold for this agent type
        self.risk_threshold = config.risk.risk_thresholds.get(agent_type, 0.3)
        self.risk_lambda = config.risk.risk_lambdas.get(agent_type, 0.5)
    
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: State vector
            explore: Whether to add exploration noise
        
        Returns:
            Action vector (portfolio weights)
        """
        # Apply feature masking if enabled
        if self.use_feature_masking and self.training:
            state = self._apply_feature_mask(state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from actor
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        if explore and self.training:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, 0.0, 1.0)  # Clip to valid range
            action = action / (np.sum(action) + 1e-8)  # Renormalize
        
        return action
    
    def _apply_feature_mask(self, state: np.ndarray) -> np.ndarray:
        """
        Apply random feature masking for signal specialization.
        
        Args:
            state: Original state
        
        Returns:
            Masked state
        """
        mask = np.random.binomial(1, 1 - self.mask_probability, size=state.shape)
        return state * mask
    
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        Update agent from batch of transitions.
        
        Args:
            batch: (states, actions, rewards, next_states, dones)
        
        Returns:
            Dictionary of training metrics
        """
        states, actions, rewards, next_states, dones = batch
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * 0.99 * next_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_q = self.critic(states, actor_actions)
        
        # Risk penalty
        risk_scores = self.safety_critic(states, actor_actions)
        risk_penalty = self.risk_lambda * risk_scores
        
        actor_loss = -(actor_q - risk_penalty).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Decay noise
        self.noise_std *= self.noise_decay
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": actor_q.mean().item()
        }
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Soft update target networks.
        
        Args:
            tau: Soft update coefficient
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def predict_risk(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Predict risk score for state-action pair.
        
        Args:
            state: State vector
            action: Action vector
        
        Returns:
            Risk score in [0, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        return self.safety_critic.predict_risk(state_tensor, action_tensor)
    
    def save(self, filepath: str):
        """Save agent parameters."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "safety_critic": self.safety_critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.safety_critic.load_state_dict(checkpoint["safety_critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
