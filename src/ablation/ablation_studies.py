"""
Ablation studies: MARS-Static, MARS-Homogeneous, MARS-Div5, MARS-Div15
"""
import numpy as np
import torch
from typing import List
import logging

from ..agents.ddpg_agent import DDPGAgent
from ..mac.meta_controller import MetaAdaptiveController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MARSStatic:
    """
    MARS-Static: Uniform weights (no adaptation).
    """
    
    def __init__(self, n_agents: int = 10):
        self.n_agents = n_agents
        self.weights = np.ones(n_agents) / n_agents  # Uniform weights
    
    def compute_weights(self, state: np.ndarray) -> np.ndarray:
        """Return uniform weights."""
        return self.weights
    
    def aggregate_actions(self, state: np.ndarray, agent_actions: np.ndarray) -> np.ndarray:
        """Aggregate with uniform weights."""
        return np.mean(agent_actions, axis=0)


class MARSHomogeneous:
    """
    MARS-Homogeneous: All agents have same risk parameters.
    """
    
    @staticmethod
    def create_homogeneous_ensemble(state_dim: int, action_dim: int, n_agents: int = 10,
                                   risk_threshold: float = 0.5, risk_penalty: float = 1.0,
                                   device: str = 'cuda'):
        """Create ensemble with identical agents."""
        from ..agents.hae import HeterogeneousAgentEnsemble
        
        agents = []
        for i in range(n_agents):
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_id=i,
                risk_threshold=risk_threshold,
                risk_penalty=risk_penalty,
                device=device
            )
            agents.append(agent)
        
        # Create wrapper similar to HAE
        class HomogeneousEnsemble:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)
            
            def select_actions(self, state, add_noise=True):
                actions = []
                for agent in self.agents:
                    action = agent.select_action(state, add_noise=add_noise)
                    actions.append(action)
                return np.array(actions)
            
            def get_q_values(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_value = agent.critic(state_tensor, action_tensor).cpu().item()
                    q_values.append(q_value)
                return np.array(q_values)
            
            def get_safety_scores(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                safety_scores = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        safety_score = agent.safety_critic(state_tensor, action_tensor).cpu().item()
                    safety_scores.append(safety_score)
                return np.array(safety_scores)
            
            def update_all(self, batch_size=64):
                update_stats = []
                for agent in self.agents:
                    stats = agent.update(batch_size)
                    if stats is not None:
                        update_stats.append(stats)
                return update_stats
        
        return HomogeneousEnsemble(agents)


class MARSDiv5:
    """
    MARS-Div5: 5 agents instead of 10.
    """
    
    @staticmethod
    def create_ensemble(state_dim: int, action_dim: int, n_agents: int = 5,
                       device: str = 'cuda'):
        """Create ensemble with 5 agents."""
        from ..agents.hae import HeterogeneousAgentEnsemble
        
        # Use same risk parameter distribution but with 5 agents
        risk_thresholds = np.linspace(0.2, 0.8, n_agents)
        risk_penalties = np.linspace(2.0, 0.5, n_agents)
        
        agents = []
        for i in range(n_agents):
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_id=i,
                risk_threshold=risk_thresholds[i],
                risk_penalty=risk_penalties[i],
                device=device
            )
            agents.append(agent)
        
        # Create wrapper
        class Div5Ensemble:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)
            
            def select_actions(self, state, add_noise=True):
                actions = []
                for agent in self.agents:
                    action = agent.select_action(state, add_noise=add_noise)
                    actions.append(action)
                return np.array(actions)
            
            def get_q_values(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_value = agent.critic(state_tensor, action_tensor).cpu().item()
                    q_values.append(q_value)
                return np.array(q_values)
            
            def get_safety_scores(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                safety_scores = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        safety_score = agent.safety_critic(state_tensor, action_tensor).cpu().item()
                    safety_scores.append(safety_score)
                return np.array(safety_scores)
            
            def update_all(self, batch_size=64):
                update_stats = []
                for agent in self.agents:
                    stats = agent.update(batch_size)
                    if stats is not None:
                        update_stats.append(stats)
                return update_stats
        
        return Div5Ensemble(agents)


class MARSDiv15:
    """
    MARS-Div15: 15 agents instead of 10.
    """
    
    @staticmethod
    def create_ensemble(state_dim: int, action_dim: int, n_agents: int = 15,
                       device: str = 'cuda'):
        """Create ensemble with 15 agents."""
        from ..agents.hae import HeterogeneousAgentEnsemble
        
        # Use same risk parameter distribution but with 15 agents
        risk_thresholds = np.linspace(0.2, 0.8, n_agents)
        risk_penalties = np.linspace(2.0, 0.5, n_agents)
        
        agents = []
        for i in range(n_agents):
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                agent_id=i,
                risk_threshold=risk_thresholds[i],
                risk_penalty=risk_penalties[i],
                device=device
            )
            agents.append(agent)
        
        # Create wrapper (same as Div5)
        class Div15Ensemble:
            def __init__(self, agents):
                self.agents = agents
                self.n_agents = len(agents)
            
            def select_actions(self, state, add_noise=True):
                actions = []
                for agent in self.agents:
                    action = agent.select_action(state, add_noise=add_noise)
                    actions.append(action)
                return np.array(actions)
            
            def get_q_values(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_value = agent.critic(state_tensor, action_tensor).cpu().item()
                    q_values.append(q_value)
                return np.array(q_values)
            
            def get_safety_scores(self, state, actions):
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                safety_scores = []
                for i, agent in enumerate(self.agents):
                    action_tensor = torch.FloatTensor(actions[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        safety_score = agent.safety_critic(state_tensor, action_tensor).cpu().item()
                    safety_scores.append(safety_score)
                return np.array(safety_scores)
            
            def update_all(self, batch_size=64):
                update_stats = []
                for agent in self.agents:
                    stats = agent.update(batch_size)
                    if stats is not None:
                        update_stats.append(stats)
                return update_stats
        
        return Div15Ensemble(agents)

