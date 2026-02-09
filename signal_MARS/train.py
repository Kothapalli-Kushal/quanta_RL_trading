"""
Training script for Signal-MARS framework.

End-to-end training loop with environment, agents, MAC, and risk overlay.
"""

import numpy as np
import torch
from typing import Dict, Any, List
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SignalMARSConfig, get_default_config
from signals import SignalFactory
from macro import MacroLoader
from env import PortfolioEnv, StateBuilder
from agents import DDPGAgent, ReplayBuffer
from mac import MACController, MACTrainer, MACBuffer
from risk import RiskOverlay, RiskMetrics
from utils import setup_logger, log_metrics, set_seed


class SignalMARSTrainer:
    """
    Main trainer class for Signal-MARS.
    
    Orchestrates the training loop:
    1. Environment reset
    2. Agents propose actions
    3. MAC weights agents
    4. Aggregate action
    5. Apply risk overlay
    6. Step environment
    7. Store transitions
    8. Update agents
    9. Periodically update MAC
    """
    
    def __init__(self, config: SignalMARSConfig):
        """
        Initialize trainer.
        
        Args:
            config: Signal-MARS configuration
        """
        self.config = config
        
        # Set seed
        set_seed(config.training.seed)
        
        # Setup logging
        self.logger = setup_logger("train", config.training.log_dir)
        self.logger.info("Initializing Signal-MARS trainer...")
        
        # Create signal factory
        self.signal_factory = SignalFactory(
            enabled_signals=config.signal.enabled_signals
        )
        
        # Create macro loader
        self.macro_loader = MacroLoader(
            forward_fill_days=config.macro.forward_fill_days
        )
        # TODO: Load actual macro data
        # self.macro_loader.load_macro_data(data_source)
        
        # Create state builder
        self.state_builder = StateBuilder(
            config, self.signal_factory, self.macro_loader
        )
        
        # Create environment
        self.env = PortfolioEnv(config, self.state_builder)
        
        # Create agents
        self.agents: List[DDPGAgent] = []
        self.replay_buffers: List[ReplayBuffer] = []
        
        state_dim = self.state_builder.get_state_dim()
        action_dim = config.env.num_assets
        
        for i, agent_type in enumerate(config.agent.agent_types):
            agent = DDPGAgent(
                state_dim, action_dim, agent_type, config, config.training.device
            )
            self.agents.append(agent)
            
            buffer = ReplayBuffer(
                config.training.buffer_size, state_dim, action_dim, config.training.device
            )
            self.replay_buffers.append(buffer)
        
        # Create MAC controller
        if config.mac.use_mac:
            self.mac_controller = MACController(
                config.mac.mac_input_dim,
                config.agent.num_agents,
                config.mac.mac_hidden_dims
            ).to(config.training.device)
            self.mac_trainer = MACTrainer(self.mac_controller, config, config.training.device)
            self.mac_buffer = MACBuffer(
                10000, config.agent.num_agents, config.training.device
            )
        else:
            self.mac_controller = None
            self.mac_trainer = None
            self.mac_buffer = None
        
        # Create risk overlay
        self.risk_overlay = RiskOverlay(config)
        
        # Training state
        self.episode = 0
        self.global_step = 0
        
        # Create save directory
        os.makedirs(config.training.save_dir, exist_ok=True)
    
    def train(self):
        """Run training loop."""
        self.logger.info("Starting training...")
        
        for episode in range(self.config.training.num_episodes):
            self.episode = episode
            episode_metrics = self._train_episode()
            
            # Logging
            if episode % self.config.training.log_freq == 0:
                log_metrics(self.logger, episode_metrics, episode, prefix="Episode")
            
            # Evaluation
            if episode % self.config.training.eval_freq == 0:
                self._evaluate(episode)
            
            # Save checkpoints
            if episode % self.config.training.save_freq == 0:
                self._save_checkpoint(episode)
        
        self.logger.info("Training completed!")
    
    def _train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        # Reset environment
        # TODO: Get actual market data
        initial_prices = np.ones(self.config.env.num_assets)  # Placeholder
        state = self.env.reset(initial_prices)
        
        episode_reward = 0.0
        episode_returns = []
        
        done = False
        step = 0
        
        while not done:
            # Get actions from all agents
            agent_actions = []
            agent_q_values = []
            agent_risk_scores = []
            
            for agent in self.agents:
                action = agent.act(state, explore=True)
                agent_actions.append(action)
                
                # Get Q-value and risk score
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.training.device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.config.training.device)
                
                with torch.no_grad():
                    q_value = agent.critic(state_tensor, action_tensor).item()
                    risk_score = agent.predict_risk(state, action)
                
                agent_q_values.append(q_value)
                agent_risk_scores.append(risk_score)
            
            # MAC weighting
            if self.mac_controller is not None:
                mac_state = self.state_builder.build_mac_state(
                    self.env._get_market_data()
                )
                mac_weights = self.mac_controller.get_weights(mac_state)
                
                # Aggregate actions
                aggregated_action = np.sum(
                    [w * a for w, a in zip(mac_weights, agent_actions)], axis=0
                )
                
                # Store in MAC buffer
                self.mac_buffer.add(
                    np.array(agent_q_values),
                    np.array(agent_risk_scores),
                    mac_weights,
                    0.0,  # Return will be updated after step
                    mac_state
                )
            else:
                # Uniform weighting if no MAC
                mac_weights = np.ones(len(self.agents)) / len(self.agents)
                aggregated_action = np.mean(agent_actions, axis=0)
            
            # Apply risk overlay
            portfolio_state = self.env._get_portfolio_state()
            portfolio_state["prices"] = self.env.current_prices
            constrained_action = self.risk_overlay.apply_constraints(
                aggregated_action, portfolio_state
            )
            
            # Step environment
            # TODO: Get actual new prices from data
            new_prices = self.env.current_prices * (1 + np.random.randn(self.config.env.num_assets) * 0.01)
            next_state, reward, done, info = self.env.step(constrained_action, new_prices)
            
            # Update MAC buffer with actual return
            if self.mac_controller is not None:
                # Update last return in buffer
                if len(self.mac_buffer.returns) > 0:
                    self.mac_buffer.returns[-1] = reward
            
            # Store transitions
            for i, (agent, buffer) in enumerate(zip(self.agents, self.replay_buffers)):
                buffer.add(state, agent_actions[i], reward, next_state, done)
            
            # Update agents
            if step % self.config.training.agent_update_freq == 0:
                for agent, buffer in zip(self.agents, self.replay_buffers):
                    if len(buffer) >= self.config.training.batch_size:
                        batch = buffer.sample(self.config.training.batch_size)
                        agent.update(batch)
                        
                        # Update target networks
                        if self.global_step % self.config.training.target_update_freq == 0:
                            agent.update_target_networks(self.config.training.target_update_tau)
            
            # Update MAC
            if (self.mac_controller is not None and 
                step % self.config.mac.mac_update_freq == 0 and
                len(self.mac_buffer) >= self.config.training.batch_size):
                mac_batch = self.mac_buffer.get_batch(self.config.training.batch_size)
                self.mac_trainer.update(*mac_batch)
            
            state = next_state
            episode_reward += reward
            episode_returns.append(reward)
            step += 1
            self.global_step += 1
        
        # Compute episode metrics
        sharpe = self.env.get_sharpe_ratio()
        max_dd = self.env.get_max_drawdown()
        
        metrics = {
            "episode_reward": episode_reward,
            "portfolio_value": self.env.get_portfolio_value(),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "episode_length": step
        }
        
        return metrics
    
    def _evaluate(self, episode: int):
        """Run evaluation."""
        # TODO: Implement evaluation on test set
        self.logger.info(f"Evaluation at episode {episode} (placeholder)")
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.training.save_dir, f"episode_{episode}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save agents
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(checkpoint_dir, f"agent_{i}_{agent.agent_type}.pth")
            agent.save(agent_path)
        
        # Save MAC
        if self.mac_controller is not None:
            mac_path = os.path.join(checkpoint_dir, "mac_controller.pth")
            torch.save(self.mac_controller.state_dict(), mac_path)
        
        self.logger.info(f"Checkpoint saved at episode {episode}")


def main():
    """Main training function."""
    # Load configuration
    config = get_default_config()
    
    # Create trainer
    trainer = SignalMARSTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
