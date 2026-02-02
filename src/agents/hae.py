"""
Heterogeneous Agent Ensemble (HAE): N=10 DDPG agents with varying risk parameters.
"""
import numpy as np
from typing import List
import logging

from .ddpg_agent import DDPGAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeterogeneousAgentEnsemble:
    """
    Ensemble of N=10 DDPG agents with heterogeneous risk parameters.
    
    Agents span from ultra-conservative to aggressive across the risk spectrum.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int = 10,
                 device: str = 'cuda'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            n_agents: Number of agents (default 10)
            device: Device for computation
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Generate heterogeneous risk parameters
        # Risk thresholds: from 0.2 (conservative) to 0.8 (aggressive)
        # Risk penalties: from 2.0 (conservative) to 0.5 (aggressive)
        risk_thresholds = np.linspace(0.2, 0.8, n_agents)
        risk_penalties = np.linspace(2.0, 0.5, n_agents)
        
        # Create agents
        self.agents: List[DDPGAgent] = []
        for i in range(n_agents):
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_id=i,
                risk_threshold=risk_thresholds[i],
                risk_penalty=risk_penalties[i],
                device=device
            )
            self.agents.append(agent)
        
        logger.info(f"Initialized HAE with {n_agents} agents")
        logger.info(f"Risk thresholds: {risk_thresholds}")
        logger.info(f"Risk penalties: {risk_penalties}")
    
    def select_actions(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Get actions from all agents.
        
        Returns:
            Array of shape (n_agents, action_dim)
        """
        actions = []
        for agent in self.agents:
            action = agent.select_action(state, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)
    
    def get_q_values(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Get Q-values from all agents for given state-action pairs.
        
        Args:
            state: State vector
            actions: Actions from agents (n_agents, action_dim)
            
        Returns:
            Q-values (n_agents,)
        """
        import torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = []
        
        for i, agent in enumerate(self.agents):
            action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_value = agent.critic(state_tensor, action_tensor).cpu().item()
            q_values.append(q_value)
        
        return np.array(q_values)
    
    def get_safety_scores(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Get safety scores from all agents for given state-action pairs.
        
        Args:
            state: State vector
            actions: Actions from agents (n_agents, action_dim)
            
        Returns:
            Safety scores (n_agents,)
        """
        import torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        safety_scores = []
        
        for i, agent in enumerate(self.agents):
            action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                safety_score = agent.safety_critic(state_tensor, action_tensor).cpu().item()
            safety_scores.append(safety_score)
        
        return np.array(safety_scores)
    
    def update_all(self, batch_size: int = 64):
        """Update all agents."""
        update_stats = []
        for agent in self.agents:
            stats = agent.update(batch_size)
            if stats is not None:
                update_stats.append(stats)
        return update_stats
    
    def save_all(self, base_path: str):
        """Save all agents."""
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}/agent_{i}.pth"
            agent.save(filepath)
    
    def load_all(self, base_path: str):
        """Load all agents."""
        for i, agent in enumerate(self.agents):
            filepath = f"{base_path}/agent_{i}.pth"
            agent.load(filepath)

