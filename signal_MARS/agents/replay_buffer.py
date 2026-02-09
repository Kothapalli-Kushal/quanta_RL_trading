"""
Replay buffer for off-policy reinforcement learning.

Stores transitions for experience replay.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    Standard replay buffer for off-policy RL algorithms.
    
    Stores (state, action, reward, next_state, done) transitions.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if self.size < batch_size:
            batch_size = self.size
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size
