"""
Neural networks for DDPG: Actor, Critic, and Safety-Critic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(Actor, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Output action in [-1, 1]."""
        return self.network(state)


class Critic(nn.Module):
    """Critic network (Q-function) for DDPG."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(Critic, self).__init__()
        
        # State pathway
        state_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            state_layers.append(nn.Linear(input_dim, hidden_dim))
            state_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.state_network = nn.Sequential(*state_layers)
        
        # Action pathway
        self.action_network = nn.Linear(action_dim, hidden_dims[-1])
        
        # Combined pathway
        combined_layers = []
        combined_layers.append(nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]))
        combined_layers.append(nn.ReLU())
        combined_layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.combined_network = nn.Sequential(*combined_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Output Q-value."""
        state_out = self.state_network(state)
        action_out = F.relu(self.action_network(action))
        combined = torch.cat([state_out, action_out], dim=-1)
        return self.combined_network(combined)


class SafetyCritic(nn.Module):
    """
    Safety-Critic network: C_ξ(s, a) ≈ C_env(s, a) ∈ [0, 1]
    
    Learns to predict environment risk score combining:
    - Portfolio concentration
    - Herfindahl-Hirschman Index
    - Leverage
    - Exposure relative to equity
    - Simulated volatility
    - Impact of proposed trade on recent realized volatility
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super(SafetyCritic, self).__init__()
        
        # Same architecture as Critic
        # State pathway
        state_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            state_layers.append(nn.Linear(input_dim, hidden_dim))
            state_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.state_network = nn.Sequential(*state_layers)
        
        # Action pathway
        self.action_network = nn.Linear(action_dim, hidden_dims[-1])
        
        # Combined pathway
        combined_layers = []
        combined_layers.append(nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]))
        combined_layers.append(nn.ReLU())
        combined_layers.append(nn.Linear(hidden_dims[-1], 1))
        combined_layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.combined_network = nn.Sequential(*combined_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Output risk score in [0, 1]."""
        state_out = self.state_network(state)
        action_out = F.relu(self.action_network(action))
        combined = torch.cat([state_out, action_out], dim=-1)
        return self.combined_network(combined)

