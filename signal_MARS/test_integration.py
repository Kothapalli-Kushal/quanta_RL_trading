"""
Integration tests for Signal-MARS training and evaluation.

Tests the complete training and evaluation workflows.
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from signal_MARS.config import get_default_config
from signal_MARS.signals import SignalFactory
from signal_MARS.macro import MacroLoader
from signal_MARS.env import PortfolioEnv, StateBuilder
from signal_MARS.agents import DDPGAgent, ReplayBuffer
from signal_MARS.mac import MACController, MACTrainer, MACBuffer
from signal_MARS.risk import RiskOverlay
from signal_MARS.utils import set_seed


class TestTrainingIntegration(unittest.TestCase):
    """Test complete training workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        # Use small config for fast testing
        self.config.env.num_assets = 3
        self.config.env.max_steps = 5
        self.config.training.batch_size = 4
        self.config.training.buffer_size = 50
        self.config.agent.num_agents = 2
        self.config.agent.agent_types = ["trend", "meanrev"]
    
    def test_mini_training_loop(self):
        """Test a mini training loop."""
        # Setup components
        signal_factory = SignalFactory()
        macro_loader = MacroLoader()
        macro_loader.load_macro_data(None)
        state_builder = StateBuilder(self.config, signal_factory, macro_loader)
        env = PortfolioEnv(self.config, state_builder)
        
        # Create agents
        agents = []
        buffers = []
        state_dim = state_builder.get_state_dim()
        action_dim = self.config.env.num_assets
        
        for agent_type in self.config.agent.agent_types:
            agent = DDPGAgent(state_dim, action_dim, agent_type, self.config)
            agents.append(agent)
            buffers.append(ReplayBuffer(
                self.config.training.buffer_size, state_dim, action_dim
            ))
        
        # Create MAC
        mac = MACController(self.config.mac.mac_input_dim, len(agents))
        mac_trainer = MACTrainer(mac, self.config, self.config.training.device)
        mac_buffer = MACBuffer(100, len(agents))
        
        # Create risk overlay
        risk_overlay = RiskOverlay(self.config)
        
        # Mini training loop
        initial_prices = np.ones(self.config.env.num_assets)
        state = env.reset(initial_prices)
        
        episode_reward = 0.0
        step = 0
        
        while step < self.config.env.max_steps:
            # Get actions from agents
            agent_actions = []
            agent_q_values = []
            agent_risk_scores = []
            
            for agent in agents:
                action = agent.act(state, explore=True)
                agent_actions.append(action)
                
                # Get Q-value and risk
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.training.device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.config.training.device)
                
                with torch.no_grad():
                    q_value = agent.critic(state_tensor, action_tensor).item()
                    risk_score = agent.predict_risk(state, action)
                
                agent_q_values.append(q_value)
                agent_risk_scores.append(risk_score)
            
            # MAC weighting
            mac_state = state_builder.build_mac_state(env._get_market_data())
            mac_weights = mac.get_weights(mac_state)
            
            # Aggregate actions
            aggregated_action = np.sum(
                [w * a for w, a in zip(mac_weights, agent_actions)], axis=0
            )
            
            # Apply risk overlay
            portfolio_state = env._get_portfolio_state()
            portfolio_state["prices"] = env.current_prices
            constrained_action = risk_overlay.apply_constraints(aggregated_action, portfolio_state)
            
            # Step environment
            new_prices = env.current_prices * (1 + np.random.randn(self.config.env.num_assets) * 0.01)
            next_state, reward, done, info = env.step(constrained_action, new_prices)
            
            # Store in MAC buffer
            mac_buffer.add(
                np.array(agent_q_values),
                np.array(agent_risk_scores),
                mac_weights,
                reward,
                mac_state
            )
            
            # Store transitions
            for i, (agent, buffer) in enumerate(zip(agents, buffers)):
                buffer.add(state, agent_actions[i], reward, next_state, done)
            
            # Update agents
            for agent, buffer in zip(agents, buffers):
                if len(buffer) >= self.config.training.batch_size:
                    batch = buffer.sample(self.config.training.batch_size)
                    metrics = agent.update(batch)
                    self.assertIn("critic_loss", metrics)
            
            # Update MAC
            if len(mac_buffer) >= self.config.training.batch_size:
                mac_batch = mac_buffer.get_batch(self.config.training.batch_size)
                mac_metrics = mac_trainer.update(*mac_batch)
                self.assertIn("total_loss", mac_metrics)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Verify we completed the loop
        self.assertGreater(step, 0)
        self.assertGreater(len(env.returns_history), 0)


class TestEvaluationIntegration(unittest.TestCase):
    """Test evaluation workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        self.config.env.num_assets = 3
        self.config.env.max_steps = 5
        self.config.agent.num_agents = 2
        self.config.agent.agent_types = ["trend", "meanrev"]
    
    def test_evaluation_loop(self):
        """Test evaluation workflow."""
        # Setup components
        signal_factory = SignalFactory()
        macro_loader = MacroLoader()
        macro_loader.load_macro_data(None)
        state_builder = StateBuilder(self.config, signal_factory, macro_loader)
        env = PortfolioEnv(self.config, state_builder)
        
        # Create agents
        agents = []
        state_dim = state_builder.get_state_dim()
        action_dim = self.config.env.num_assets
        
        for agent_type in self.config.agent.agent_types:
            agent = DDPGAgent(state_dim, action_dim, agent_type, self.config)
            agent.eval()  # Set to eval mode
            agents.append(agent)
        
        # Create MAC
        mac = MACController(self.config.mac.mac_input_dim, len(agents))
        mac.eval()
        
        # Create risk overlay
        risk_overlay = RiskOverlay(self.config)
        
        # Evaluation loop
        initial_prices = np.ones(self.config.env.num_assets)
        state = env.reset(initial_prices)
        
        done = False
        step = 0
        
        while not done and step < self.config.env.max_steps:
            # Get actions (no exploration)
            agent_actions = []
            for agent in agents:
                action = agent.act(state, explore=False)
                agent_actions.append(action)
            
            # MAC weighting
            mac_state = state_builder.build_mac_state(env._get_market_data())
            mac_weights = mac.get_weights(mac_state)
            
            # Aggregate
            aggregated_action = np.sum(
                [w * a for w, a in zip(mac_weights, agent_actions)], axis=0
            )
            
            # Apply risk overlay
            portfolio_state = env._get_portfolio_state()
            portfolio_state["prices"] = env.current_prices
            constrained_action = risk_overlay.apply_constraints(aggregated_action, portfolio_state)
            
            # Step
            new_prices = env.current_prices * (1 + np.random.randn(self.config.env.num_assets) * 0.01)
            next_state, reward, done, info = env.step(constrained_action, new_prices)
            
            state = next_state
            step += 1
        
        # Verify evaluation completed
        self.assertGreater(step, 0)
        self.assertGreater(len(env.returns_history), 0)
        
        # Check metrics
        sharpe = env.get_sharpe_ratio()
        max_dd = env.get_max_drawdown()
        self.assertIsInstance(sharpe, (float, np.floating))
        self.assertIsInstance(max_dd, (float, np.floating))


if __name__ == "__main__":
    import torch
    unittest.main()
