"""
Base agent interface for RL agents.

All agents must inherit from BaseAgent and implement act() and update().
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    
    Defines the interface that all agents must implement.
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_type: str, 
                 device: str = "cpu"):
        """
        Initialize agent.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            agent_type: Type identifier (e.g., "trend", "meanrev")
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type
        self.device = device
        self.training = True
    
    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select an action given a state.
        
        Args:
            state: Current state vector
            explore: Whether to add exploration noise
        
        Returns:
            Action vector
        """
        pass
    
    @abstractmethod
    def update(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """
        Update agent parameters from a batch of transitions.
        
        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save agent parameters to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent parameters from file."""
        pass
    
    def train(self):
        """Set agent to training mode."""
        self.training = True
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.training = False
    
    def get_agent_type(self) -> str:
        """Get agent type identifier."""
        return self.agent_type
