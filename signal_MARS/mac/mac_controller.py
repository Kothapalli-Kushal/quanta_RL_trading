"""
Meta-Adaptive Controller (MAC) for agent weight allocation.

Neural network that outputs agent weights via softmax based on macro + signal summaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple


class MACController(nn.Module):
    """
    Meta-Adaptive Controller network.
    
    Takes macro features and signal summaries as input,
    outputs softmax weights over agents.
    """
    
    def __init__(self, input_dim: int, num_agents: int, hidden_dims: list = None):
        """
        Initialize MAC controller.
        
        Args:
            input_dim: Dimension of MAC input (macro + signal summaries)
            num_agents: Number of agents to allocate weights to
            hidden_dims: Hidden layer dimensions
        """
        super(MACController, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # Build network
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer (agent weights)
        layers.append(nn.Linear(current_dim, num_agents))
        # Softmax will be applied in forward()
        
        self.network = nn.Sequential(*layers)
        self.num_agents = num_agents
    
    def forward(self, mac_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            mac_state: MAC state tensor (batch, input_dim)
        
        Returns:
            Agent weights tensor (batch, num_agents) - softmax over agents
        """
        logits = self.network(mac_state)
        return F.softmax(logits, dim=-1)
    
    def get_weights(self, mac_state: np.ndarray) -> np.ndarray:
        """
        Get agent weights for a single MAC state.
        
        Args:
            mac_state: MAC state vector (input_dim,)
        
        Returns:
            Agent weights (num_agents,) - normalized to sum to 1
        """
        with torch.no_grad():
            if isinstance(mac_state, np.ndarray):
                mac_state = torch.FloatTensor(mac_state).unsqueeze(0)
            
            weights = self.forward(mac_state)
            return weights.cpu().numpy()[0]


class MACTrainer:
    """
    Trainer for MAC controller.
    
    Updates MAC to optimize risk-adjusted Sharpe-like objective.
    """
    
    def __init__(self, mac_controller: MACController, config, device: str = "cpu"):
        """
        Initialize MAC trainer.
        
        Args:
            mac_controller: MAC controller network
            config: Configuration object (for loss weights)
            device: Device to run on
        """
        self.mac_controller = mac_controller
        self.config = config.mac
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            mac_controller.parameters(), lr=config.mac.mac_lr
        )
    
    def compute_loss(self, mac_states: torch.Tensor, q_values: torch.Tensor,
                    risk_values: torch.Tensor, mac_weights: torch.Tensor,
                    returns: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MAC loss.
        
        Loss components:
        1. Q-value loss (weighted by MAC weights)
        2. Risk penalty (weighted by MAC weights)
        3. Sharpe-like loss (mean / std of returns)
        
        Args:
            mac_states: MAC state tensors (batch, input_dim)
            q_values: Q-values from agents (batch, num_agents)
            risk_values: Risk scores from agents (batch, num_agents)
            mac_weights: MAC weights used (batch, num_agents)
            returns: Actual returns (batch,)
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Predict MAC weights
        pred_weights = self.mac_controller(mac_states)
        
        # Q-value component: weighted Q-values
        weighted_q = torch.sum(pred_weights * q_values, dim=1)
        q_loss = -torch.mean(weighted_q)  # Negative because we want to maximize
        
        # Risk component: weighted risk scores
        weighted_risk = torch.sum(pred_weights * risk_values, dim=1)
        risk_loss = torch.mean(weighted_risk)
        
        # Sharpe-like component: mean / std of returns
        # Use predicted weights to compute expected return
        # (simplified: use actual returns weighted by predicted weights)
        mean_return = torch.mean(returns)
        std_return = torch.std(returns)
        
        if std_return > 1e-8:
            sharpe = mean_return / std_return
            sharpe_loss = -sharpe  # Negative because we want to maximize
        else:
            sharpe_loss = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total_loss = (
            self.config.q_weight * q_loss +
            self.config.risk_weight * risk_loss +
            self.config.sharpe_weight * sharpe_loss
        )
        
        metrics = {
            "q_loss": q_loss.item(),
            "risk_loss": risk_loss.item(),
            "sharpe_loss": sharpe_loss.item(),
            "total_loss": total_loss.item(),
            "mean_weighted_q": weighted_q.mean().item(),
            "mean_weighted_risk": weighted_risk.mean().item()
        }
        
        return total_loss, metrics
    
    def update(self, mac_states: torch.Tensor, q_values: torch.Tensor,
              risk_values: torch.Tensor, mac_weights: torch.Tensor,
              returns: torch.Tensor) -> Dict[str, float]:
        """
        Update MAC controller.
        
        Args:
            mac_states: MAC state tensors
            q_values: Q-values from agents
            risk_values: Risk scores from agents
            mac_weights: MAC weights used
            returns: Actual returns
        
        Returns:
            Dictionary of training metrics
        """
        loss, metrics = self.compute_loss(
            mac_states, q_values, risk_values, mac_weights, returns
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return metrics
