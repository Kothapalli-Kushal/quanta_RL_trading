"""
Buffer for storing MAC (Meta-Adaptive Controller) training data.

Stores predicted Q-values, risk values, and actual outcomes for MAC updates.
"""

from typing import List, Tuple
import numpy as np
import torch
from collections import deque


class MACBuffer:
    """
    Buffer for MAC training data.
    
    Stores per-step:
    - Predicted Q-values from each agent
    - Predicted risk values from each agent
    - MAC weights used
    - Actual portfolio returns
    """
    
    def __init__(self, capacity: int, num_agents: int, device: str = "cpu"):
        """
        Initialize MAC buffer.
        
        Args:
            capacity: Maximum number of steps to store
            num_agents: Number of agents
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.device = device
        
        # Storage
        self.q_values = deque(maxlen=capacity)  # List of (num_agents,) arrays
        self.risk_values = deque(maxlen=capacity)  # List of (num_agents,) arrays
        self.mac_weights = deque(maxlen=capacity)  # List of (num_agents,) arrays
        self.returns = deque(maxlen=capacity)  # List of floats
        self.mac_states = deque(maxlen=capacity)  # List of MAC state vectors
    
    def add(self, q_values: np.ndarray, risk_values: np.ndarray,
            mac_weights: np.ndarray, returns: float, mac_state: np.ndarray):
        """
        Add a step to the buffer.
        
        Args:
            q_values: Q-values from all agents (num_agents,)
            risk_values: Risk scores from all agents (num_agents,)
            mac_weights: MAC weights used (num_agents,)
            returns: Actual portfolio return
            mac_state: MAC state vector
        """
        self.q_values.append(q_values.copy())
        self.risk_values.append(risk_values.copy())
        self.mac_weights.append(mac_weights.copy())
        self.returns.append(returns)
        self.mac_states.append(mac_state.copy())
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a batch of data for MAC training.
        
        Args:
            batch_size: Number of samples to return
        
        Returns:
            Tuple of (mac_states, q_values, risk_values, mac_weights, returns)
        """
        if len(self.q_values) < batch_size:
            batch_size = len(self.q_values)
        
        indices = np.random.randint(0, len(self.q_values), size=batch_size)
        
        mac_states = torch.FloatTensor(
            np.array([self.mac_states[i] for i in indices])
        ).to(self.device)
        
        q_values = torch.FloatTensor(
            np.array([self.q_values[i] for i in indices])
        ).to(self.device)
        
        risk_values = torch.FloatTensor(
            np.array([self.risk_values[i] for i in indices])
        ).to(self.device)
        
        mac_weights = torch.FloatTensor(
            np.array([self.mac_weights[i] for i in indices])
        ).to(self.device)
        
        returns = torch.FloatTensor(
            np.array([self.returns[i] for i in indices])
        ).to(self.device)
        
        return mac_states, q_values, risk_values, mac_weights, returns
    
    def get_episode_data(self) -> Tuple[np.ndarray, ...]:
        """
        Get all data from current episode.
        
        Returns:
            Tuple of (q_values, risk_values, mac_weights, returns, mac_states)
        """
        return (
            np.array(self.q_values),
            np.array(self.risk_values),
            np.array(self.mac_weights),
            np.array(self.returns),
            np.array(self.mac_states)
        )
    
    def clear(self):
        """Clear the buffer."""
        self.q_values.clear()
        self.risk_values.clear()
        self.mac_weights.clear()
        self.returns.clear()
        self.mac_states.clear()
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.q_values)
