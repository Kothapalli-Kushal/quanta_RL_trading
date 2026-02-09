"""
Safety critic network for risk constraint enforcement.

Predicts risk scores in [0, 1] for given state-action pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SafetyCritic(nn.Module):
    """
    Safety critic network that predicts risk scores.
    
    Outputs a scalar risk score in [0, 1] where:
    - 0 = low risk (safe)
    - 1 = high risk (unsafe)
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: list = None):
        """
        Initialize safety critic.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dims: List of hidden layer dimensions
        """
        super(SafetyCritic, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build network
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer (single scalar risk score)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output in [0, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict risk score for state-action pair.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)
        
        Returns:
            Risk score tensor of shape (batch, 1) in [0, 1]
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)
    
    def predict_risk(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Predict risk score (single value, not batched).
        
        Args:
            state: State tensor of shape (state_dim,)
            action: Action tensor of shape (action_dim,)
        
        Returns:
            Risk score as float in [0, 1]
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            risk = self.forward(state, action)
            return risk.item()


class SafetyCriticTrainer:
    """
    Trainer for safety critic network.
    
    Trains the safety critic to predict risk based on observed outcomes.
    """
    
    def __init__(self, safety_critic: SafetyCritic, lr: float = 5e-4):
        """
        Initialize safety critic trainer.
        
        Args:
            safety_critic: Safety critic network
            lr: Learning rate
        """
        self.safety_critic = safety_critic
        self.optimizer = torch.optim.Adam(safety_critic.parameters(), lr=lr)
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
               risk_labels: torch.Tensor) -> float:
        """
        Update safety critic with risk labels.
        
        Args:
            states: State tensors (batch, state_dim)
            actions: Action tensors (batch, action_dim)
            risk_labels: True risk labels (batch, 1) in [0, 1]
        
        Returns:
            Loss value
        """
        # Predict risk
        risk_pred = self.safety_critic(states, actions)
        
        # Compute loss (MSE)
        loss = F.mse_loss(risk_pred, risk_labels)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
