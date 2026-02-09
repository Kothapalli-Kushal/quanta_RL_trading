"""
Comprehensive test suite for Signal-MARS framework.

Tests all components including training and evaluation processes.
"""

import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path for absolute imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Now use absolute imports
from signal_MARS.config import SignalMARSConfig, get_default_config
from signal_MARS.signals import SignalFactory, SignalRegistry, BaseSignal, PlaceholderSignal
from signal_MARS.macro import MacroLoader
from signal_MARS.env import PortfolioEnv, StateBuilder, RollingNormalizer
from signal_MARS.agents import DDPGAgent, ReplayBuffer, SafetyCritic, SafetyCriticTrainer
from signal_MARS.mac import MACController, MACTrainer, MACBuffer
from signal_MARS.risk import RiskOverlay, RiskMetrics
from signal_MARS.utils import set_seed, sharpe_ratio


class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.signal)
        self.assertIsNotNone(config.macro)
        self.assertIsNotNone(config.agent)
        self.assertIsNotNone(config.mac)
        self.assertIsNotNone(config.risk)
        self.assertIsNotNone(config.env)
        self.assertIsNotNone(config.training)
    
    def test_config_dimensions(self):
        """Test that dimensions are computed correctly."""
        config = get_default_config()
        self.assertGreater(config.env.state_dim, 0)
        self.assertGreater(config.mac.mac_input_dim, 0)


class TestSignals(unittest.TestCase):
    """Test signal generation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
    
    def test_base_signal_interface(self):
        """Test BaseSignal abstract interface."""
        signal = PlaceholderSignal("test", output_dim=10, window=20)
        self.assertEqual(signal.name, "test")
        self.assertEqual(signal.output_dim, 10)
        self.assertEqual(signal.window, 20)
    
    def test_placeholder_signal_compute(self):
        """Test placeholder signal computation."""
        signal = PlaceholderSignal("test", output_dim=5, window=20)
        raw_data = {"prices": np.random.randn(100, 10)}
        output = signal.compute(raw_data)
        self.assertEqual(output.shape, (5,))
        self.assertTrue(np.all(output >= -1.0))
        self.assertTrue(np.all(output <= 1.0))
    
    def test_signal_registry(self):
        """Test signal registry."""
        self.assertTrue(SignalRegistry.is_registered("trend"))
        self.assertTrue(SignalRegistry.is_registered("meanrev"))
        
        signal = SignalRegistry.create("trend", output_dim=16)
        self.assertIsInstance(signal, BaseSignal)
    
    def test_signal_factory(self):
        """Test signal factory."""
        factory = SignalFactory()
        self.assertGreater(factory.get_signal_dim(), 0)
        
        raw_data = {"prices": np.random.randn(100, 10)}
        output = factory.compute(raw_data)
        self.assertEqual(output.shape, (factory.get_signal_dim(),))
    
    def test_signal_factory_enable_disable(self):
        """Test enabling/disabling signals."""
        factory = SignalFactory()
        initial_dim = factory.get_signal_dim()
        
        if len(factory.signals) > 0:
            first_signal = list(factory.signals.keys())[0]
            factory.disable_signal(first_signal)
            self.assertLess(factory.get_signal_dim(), initial_dim)
            
            factory.enable_signal(first_signal)
            self.assertEqual(factory.get_signal_dim(), initial_dim)


class TestMacro(unittest.TestCase):
    """Test macro feature module."""
    
    def test_macro_loader(self):
        """Test macro loader."""
        loader = MacroLoader(forward_fill_days=5)
        
        # Load data first
        loader.load_macro_data(None)
        self.assertIsNotNone(loader.macro_data)
        self.assertGreater(loader.get_feature_dim(), 0)
    
    def test_macro_alignment(self):
        """Test macro feature alignment."""
        loader = MacroLoader(forward_fill_days=5)
        loader.load_macro_data(None)
        
        dates = np.array(['2022-01-01', '2022-01-02', '2022-01-03'], dtype='datetime64')
        aligned = loader.align_by_date(dates)
        self.assertEqual(aligned.shape[0], len(dates))
        self.assertEqual(aligned.shape[1], loader.get_feature_dim())


class TestNormalizer(unittest.TestCase):
    """Test normalization utilities."""
    
    def test_rolling_normalizer_zscore(self):
        """Test rolling z-score normalization."""
        normalizer = RollingNormalizer(window_size=10, method="zscore")
        
        values = np.random.randn(20)
        normalized = []
        for val in values:
            norm_val = normalizer.normalize(val, update=True)
            normalized.append(norm_val)
        
        self.assertEqual(len(normalized), len(values))
    
    def test_rolling_normalizer_percentile(self):
        """Test rolling percentile normalization."""
        normalizer = RollingNormalizer(window_size=10, method="percentile")
        
        values = np.random.randn(20)
        normalized = []
        for val in values:
            norm_val = normalizer.normalize(val, update=True)
            normalized.append(norm_val)
        
        self.assertEqual(len(normalized), len(values))


class TestStateBuilder(unittest.TestCase):
    """Test state builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.signal_factory = SignalFactory()
        self.macro_loader = MacroLoader()
        self.macro_loader.load_macro_data(None)
        self.state_builder = StateBuilder(
            self.config, self.signal_factory, self.macro_loader
        )
    
    def test_state_builder_initialization(self):
        """Test state builder initialization."""
        self.assertIsNotNone(self.state_builder)
        self.assertGreater(self.state_builder.get_state_dim(), 0)
    
    def test_build_full_state(self):
        """Test building full state."""
        portfolio_state = {
            "positions": np.zeros(self.config.env.num_assets),
            "cash": self.config.env.initial_cash,
            "returns": np.zeros(self.config.env.num_assets)
        }
        market_data = {"prices": np.random.randn(100, self.config.env.num_assets)}
        
        state = self.state_builder.build_full_state(portfolio_state, market_data)
        self.assertEqual(state.shape[0], self.state_builder.get_state_dim())
    
    def test_build_mac_state(self):
        """Test building MAC state."""
        market_data = {"prices": np.random.randn(100, self.config.env.num_assets)}
        mac_state = self.state_builder.build_mac_state(market_data)
        self.assertGreater(len(mac_state), 0)


class TestPortfolioEnv(unittest.TestCase):
    """Test portfolio environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        self.config.env.num_assets = 5  # Smaller for testing
        self.signal_factory = SignalFactory()
        self.macro_loader = MacroLoader()
        self.macro_loader.load_macro_data(None)
        self.state_builder = StateBuilder(
            self.config, self.signal_factory, self.macro_loader
        )
        self.env = PortfolioEnv(self.config, self.state_builder)
    
    def test_env_reset(self):
        """Test environment reset."""
        initial_prices = np.ones(self.config.env.num_assets)
        state = self.env.reset(initial_prices)
        self.assertEqual(len(state), self.state_builder.get_state_dim())
        self.assertEqual(self.env.portfolio_value, self.config.env.initial_cash)
    
    def test_env_step(self):
        """Test environment step."""
        initial_prices = np.ones(self.config.env.num_assets)
        state = self.env.reset(initial_prices)
        
        action = np.ones(self.config.env.num_assets) / self.config.env.num_assets
        new_prices = initial_prices * 1.01
        next_state, reward, done, info = self.env.step(action, new_prices)
        
        self.assertEqual(len(next_state), self.state_builder.get_state_dim())
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, bool)
        self.assertIn("portfolio_value", info)
    
    def test_env_multiple_steps(self):
        """Test multiple environment steps."""
        initial_prices = np.ones(self.config.env.num_assets)
        state = self.env.reset(initial_prices)
        
        for _ in range(10):
            action = np.random.dirichlet(np.ones(self.config.env.num_assets))
            new_prices = self.env.current_prices * (1 + np.random.randn(self.config.env.num_assets) * 0.01)
            state, reward, done, info = self.env.step(action, new_prices)
            if done:
                break
        
        self.assertGreater(len(self.env.returns_history), 0)


class TestReplayBuffer(unittest.TestCase):
    """Test replay buffer."""
    
    def test_replay_buffer_add(self):
        """Test adding transitions to buffer."""
        buffer = ReplayBuffer(1000, 10, 5)
        state = np.random.randn(10)
        action = np.random.randn(5)
        reward = 0.1
        next_state = np.random.randn(10)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 1)
    
    def test_replay_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(1000, 10, 5)
        
        # Add multiple transitions
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randn(5)
            buffer.add(state, action, 0.1, state, False)
        
        # Sample batch
        batch = buffer.sample(32)
        self.assertEqual(len(batch), 5)
        states, actions, rewards, next_states, dones = batch
        self.assertEqual(states.shape[0], 32)
        self.assertEqual(actions.shape[0], 32)


class TestSafetyCritic(unittest.TestCase):
    """Test safety critic."""
    
    def test_safety_critic_forward(self):
        """Test safety critic forward pass."""
        critic = SafetyCritic(state_dim=10, action_dim=5)
        state = torch.randn(32, 10)
        action = torch.randn(32, 5)
        
        risk = critic(state, action)
        self.assertEqual(risk.shape, (32, 1))
        self.assertTrue(torch.all(risk >= 0.0))
        self.assertTrue(torch.all(risk <= 1.0))
    
    def test_safety_critic_trainer(self):
        """Test safety critic trainer."""
        critic = SafetyCritic(state_dim=10, action_dim=5)
        trainer = SafetyCriticTrainer(critic, lr=1e-3)
        
        states = torch.randn(32, 10)
        actions = torch.randn(32, 5)
        risk_labels = torch.rand(32, 1)
        
        loss = trainer.update(states, actions, risk_labels)
        self.assertIsInstance(loss, float)


class TestDDPGAgent(unittest.TestCase):
    """Test DDPG agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        self.config.env.num_assets = 5
        self.state_dim = 50
        self.action_dim = 5
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = DDPGAgent(
            self.state_dim, self.action_dim, "trend", self.config
        )
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.agent_type, "trend")
    
    def test_agent_act(self):
        """Test agent action selection."""
        agent = DDPGAgent(
            self.state_dim, self.action_dim, "trend", self.config
        )
        state = np.random.randn(self.state_dim)
        
        action = agent.act(state, explore=True)
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= 0))
        self.assertTrue(np.allclose(np.sum(action), 1.0, atol=0.1))
    
    def test_agent_update(self):
        """Test agent update."""
        agent = DDPGAgent(
            self.state_dim, self.action_dim, "trend", self.config
        )
        
        # Create batch
        states = torch.randn(32, self.state_dim).to(self.config.training.device)
        actions = torch.randn(32, self.action_dim).to(self.config.training.device)
        rewards = torch.randn(32).to(self.config.training.device)
        next_states = torch.randn(32, self.state_dim).to(self.config.training.device)
        dones = torch.zeros(32).to(self.config.training.device)
        
        batch = (states, actions, rewards, next_states, dones)
        metrics = agent.update(batch)
        
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
    
    def test_agent_predict_risk(self):
        """Test agent risk prediction."""
        agent = DDPGAgent(
            self.state_dim, self.action_dim, "trend", self.config
        )
        state = np.random.randn(self.state_dim)
        action = np.random.dirichlet(np.ones(self.action_dim))
        
        risk = agent.predict_risk(state, action)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)


class TestMAC(unittest.TestCase):
    """Test MAC controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        self.input_dim = self.config.mac.mac_input_dim
        self.num_agents = self.config.agent.num_agents
    
    def test_mac_controller_forward(self):
        """Test MAC controller forward pass."""
        mac = MACController(self.input_dim, self.num_agents)
        mac_state = torch.randn(32, self.input_dim)
        
        weights = mac(mac_state)
        self.assertEqual(weights.shape, (32, self.num_agents))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(32), atol=1e-5))
    
    def test_mac_controller_get_weights(self):
        """Test MAC controller get weights."""
        mac = MACController(self.input_dim, self.num_agents)
        mac_state = np.random.randn(self.input_dim)
        
        weights = mac.get_weights(mac_state)
        self.assertEqual(weights.shape, (self.num_agents,))
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
    
    def test_mac_buffer(self):
        """Test MAC buffer."""
        buffer = MACBuffer(1000, self.num_agents)
        
        q_values = np.random.randn(self.num_agents)
        risk_values = np.random.rand(self.num_agents)
        mac_weights = np.random.dirichlet(np.ones(self.num_agents))
        returns = 0.01
        mac_state = np.random.randn(self.input_dim)
        
        buffer.add(q_values, risk_values, mac_weights, returns, mac_state)
        self.assertEqual(len(buffer), 1)
        
        batch = buffer.get_batch(1)
        self.assertEqual(len(batch), 5)
    
    def test_mac_trainer(self):
        """Test MAC trainer."""
        mac = MACController(self.input_dim, self.num_agents)
        trainer = MACTrainer(mac, self.config, "cpu")
        
        mac_states = torch.randn(32, self.input_dim)
        q_values = torch.randn(32, self.num_agents)
        risk_values = torch.rand(32, self.num_agents)
        mac_weights = torch.rand(32, self.num_agents)
        mac_weights = mac_weights / mac_weights.sum(dim=1, keepdim=True)
        returns = torch.randn(32)
        
        metrics = trainer.update(mac_states, q_values, risk_values, mac_weights, returns)
        self.assertIn("total_loss", metrics)


class TestRisk(unittest.TestCase):
    """Test risk management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_default_config()
        self.risk_overlay = RiskOverlay(self.config)
    
    def test_risk_metrics_concentration(self):
        """Test concentration metric."""
        weights = np.array([0.5, 0.3, 0.2])
        concentration = RiskMetrics.compute_concentration(weights)
        self.assertGreaterEqual(concentration, 0.0)
        self.assertLessEqual(concentration, 1.0)
    
    def test_risk_metrics_leverage(self):
        """Test leverage metric."""
        positions = np.array([100, 50, 25])
        prices = np.array([10, 20, 40])
        cash = 1000
        portfolio_value = 5000
        
        leverage = RiskMetrics.compute_leverage(positions, prices, cash, portfolio_value)
        self.assertGreaterEqual(leverage, 0.0)
    
    def test_risk_overlay_apply_constraints(self):
        """Test risk overlay constraints."""
        action = np.ones(5) / 5
        portfolio_state = {
            "positions": np.zeros(5),
            "prices": np.ones(5),
            "cash": 100000,
            "portfolio_value": 100000
        }
        
        constrained = self.risk_overlay.apply_constraints(action, portfolio_state)
        self.assertEqual(constrained.shape, action.shape)
        self.assertTrue(np.all(constrained >= 0))
        self.assertAlmostEqual(np.sum(constrained), 1.0, places=3)
    
    def test_risk_overlay_validate_action(self):
        """Test action validation."""
        action = np.ones(5) / 5
        portfolio_state = {
            "positions": np.zeros(5),
            "prices": np.ones(5),
            "cash": 100000,
            "portfolio_value": 100000
        }
        
        is_valid, msg = self.risk_overlay.validate_action(action, portfolio_state)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(msg, str)


class TestTrainingFlow(unittest.TestCase):
    """Test end-to-end training flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.config = get_default_config()
        self.config.env.num_assets = 5
        self.config.env.max_steps = 10  # Short episodes for testing
        self.config.training.batch_size = 8
        self.config.training.buffer_size = 100
    
    def test_training_setup(self):
        """Test training setup without actual training."""
        from signal_MARS.signals import SignalFactory
        from signal_MARS.macro import MacroLoader
        from signal_MARS.env import StateBuilder, PortfolioEnv
        from signal_MARS.agents import DDPGAgent, ReplayBuffer
        from signal_MARS.mac import MACController, MACTrainer, MACBuffer
        from signal_MARS.risk import RiskOverlay
        
        # Create components
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
        
        for agent_type in self.config.agent.agent_types[:2]:  # Just 2 for testing
            agent = DDPGAgent(state_dim, action_dim, agent_type, self.config)
            agents.append(agent)
            buffers.append(ReplayBuffer(100, state_dim, action_dim))
        
        # Create MAC
        mac = MACController(self.config.mac.mac_input_dim, len(agents))
        mac_buffer = MACBuffer(100, len(agents))
        
        # Test one step
        initial_prices = np.ones(self.config.env.num_assets)
        state = env.reset(initial_prices)
        
        # Get actions
        agent_actions = []
        for agent in agents:
            action = agent.act(state, explore=True)
            agent_actions.append(action)
        
        # MAC weighting
        mac_state = state_builder.build_mac_state(env._get_market_data())
        mac_weights = mac.get_weights(mac_state)
        aggregated = np.sum([w * a for w, a in zip(mac_weights, agent_actions)], axis=0)
        
        # Apply risk overlay
        risk_overlay = RiskOverlay(self.config)
        portfolio_state = env._get_portfolio_state()
        portfolio_state["prices"] = env.current_prices
        constrained = risk_overlay.apply_constraints(aggregated, portfolio_state)
        
        # Step
        new_prices = env.current_prices * 1.01
        next_state, reward, done, info = env.step(constrained, new_prices)
        
        # Store transitions
        for agent, buffer, action in zip(agents, buffers, agent_actions):
            buffer.add(state, action, reward, next_state, done)
        
        # Update agents
        for agent, buffer in zip(agents, buffers):
            if len(buffer) >= self.config.training.batch_size:
                batch = buffer.sample(self.config.training.batch_size)
                agent.update(batch)
        
        self.assertTrue(True)  # If we got here, setup works


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        returns = np.random.randn(100) * 0.01
        sharpe = sharpe_ratio(returns)
        self.assertIsInstance(sharpe, (float, np.floating))
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        val1 = np.random.rand()
        set_seed(42)
        val2 = np.random.rand()
        self.assertEqual(val1, val2)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestSignals,
        TestMacro,
        TestNormalizer,
        TestStateBuilder,
        TestPortfolioEnv,
        TestReplayBuffer,
        TestSafetyCritic,
        TestDDPGAgent,
        TestMAC,
        TestRisk,
        TestTrainingFlow,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
