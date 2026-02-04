"""
Meta-Adaptive Controller (MAC): Neural network that adaptively weights agent actions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MACNetwork(nn.Module):
    """Neural network for Meta-Adaptive Controller."""
    
    def __init__(self, state_dim: int, n_agents: int, hidden_dims: list = [128, 64]):
        super(MACNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, n_agents))
        # No activation - will apply softmax externally
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Output agent logits."""
        return self.network(state)


class MACReplayBuffer:
    """Replay buffer for MAC training."""
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, weights, q_bar, c_bar, reward):
        self.buffer.append((state, weights, q_bar, c_bar, reward))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, weights, q_bars, c_bars, rewards = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        weights = torch.FloatTensor(np.array(weights))
        q_bars = torch.FloatTensor(q_bars).unsqueeze(1)
        c_bars = torch.FloatTensor(c_bars).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        
        return states, weights, q_bars, c_bars, rewards
    
    def __len__(self):
        return len(self.buffer)


class MetaAdaptiveController:
    """
    Meta-Adaptive Controller (MAC).
    
    Learns to weight agent actions based on state to maximize:
    L(ω) = -(E[Q̄_t] / (Std(Q̄_t) + ε) - 0.5 * E[C̄_t])
    
    Where:
    - Q̄_t = Σ_i w_t^i * Q_i(s_t, a_t^i)
    - C̄_t = Σ_i w_t^i * C_i(s_t, a_t^i)
    """
    
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 lr: float = 1e-4,
                 epsilon: float = 1e-6,
                 device: str = 'cuda'):
        """
        Args:
            state_dim: State dimension
            n_agents: Number of agents
            lr: Learning rate
            epsilon: Small constant for numerical stability
            device: Device for computation
        """
        self.state_dim = state_dim
        self.n_agents = n_agents
        # Ensure epsilon is a float (in case it comes from YAML as string)
        self.epsilon = float(epsilon)
        self.device = device
        
        # Network
        self.network = MACNetwork(state_dim, n_agents).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = MACReplayBuffer()
        
        logger.info(f"Initialized MAC with {n_agents} agents on {device}")
        if device == 'cuda' and torch.cuda.is_available():
            logger.info(f"MAC using GPU: {torch.cuda.get_device_name(0)}")
    
    def compute_weights(self, state: np.ndarray) -> np.ndarray:
        """
        Compute agent weights from state.
        
        Args:
            state: State vector
            
        Returns:
            Weights (n_agents,) that sum to 1.0
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.network(state_tensor)
            weights = torch.softmax(logits, dim=-1)
        return weights.cpu().numpy().flatten()
    
    def aggregate_actions(self, state: np.ndarray, agent_actions: np.ndarray) -> np.ndarray:
        """
        Aggregate agent actions using MAC weights.
        
        Args:
            state: Current state
            agent_actions: Actions from all agents (n_agents, action_dim)
            
        Returns:
            Aggregated action (action_dim,)
        """
        weights = self.compute_weights(state)
        aggregated_action = np.sum(weights[:, np.newaxis] * agent_actions, axis=0)
        return aggregated_action
    
    def compute_mac_objective(self, q_bars: torch.Tensor, c_bars: torch.Tensor) -> torch.Tensor:
        """
        Compute MAC objective:
        L(ω) = -(E[Q̄_t] / (Std(Q̄_t) + ε) - 0.5 * E[C̄_t])
        
        Args:
            q_bars: Weighted Q-values (batch_size, 1)
            c_bars: Weighted safety scores (batch_size, 1)
            
        Returns:
            Loss value
        """
        # Sharpe-like term: E[Q̄] / (Std(Q̄) + ε)
        q_mean = q_bars.mean()
        q_std = q_bars.std()
        # Ensure epsilon is a tensor on the same device
        epsilon_tensor = torch.tensor(self.epsilon, dtype=q_std.dtype, device=q_std.device)
        sharpe_term = q_mean / (q_std + epsilon_tensor)
        
        # Safety term: 0.5 * E[C̄]
        safety_term = 0.5 * c_bars.mean()
        
        # Negative (to maximize via gradient descent)
        loss = -(sharpe_term - safety_term)
        
        return loss
    
    def update(self, batch_size: int = 64):
        """Update MAC network."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, weights, q_bars, c_bars, rewards = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        weights = weights.to(self.device)
        q_bars = q_bars.to(self.device)
        c_bars = c_bars.to(self.device)
        
        # Forward pass: compute predicted weights
        logits = self.network(states)
        predicted_weights = torch.softmax(logits, dim=-1)
        
        # Compute weighted Q and C using predicted weights
        # We need to recompute Q_bar and C_bar with predicted weights
        # For now, use the stored values and add a consistency term
        
        # Compute MAC objective on stored Q_bar and C_bar
        loss = self.compute_mac_objective(q_bars, c_bars)
        
        # Add consistency term: predicted weights should match stored weights
        consistency_loss = nn.MSELoss()(predicted_weights, weights)
        
        # Total loss
        total_loss = loss + 0.1 * consistency_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'mac_loss': total_loss.item(),
            'sharpe_term': (q_bars.mean() / (q_bars.std() + self.epsilon)).item(),
            'safety_term': c_bars.mean().item()
        }
    
    def save(self, filepath: str):
        """Save MAC state."""
        torch.save({
            'network': self.network.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load MAC state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])

