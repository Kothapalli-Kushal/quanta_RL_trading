"""
Two-tier training loop: Agent updates every step, MAC updates every meta_train_freq episodes.
"""
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os

from ..env.portfolio_env import PortfolioEnv
from ..env.risk_overlay import RiskOverlay
from ..agents.hae import HeterogeneousAgentEnsemble
from ..mac.meta_controller import MetaAdaptiveController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Two-tier training loop following Algorithm 1 in the paper.
    
    - Agents update every step using standard DDPG + Safety-Critic
    - MAC updates every meta_train_freq episodes
    """
    
    def __init__(self,
                 env: PortfolioEnv,
                 hae: HeterogeneousAgentEnsemble,
                 mac: MetaAdaptiveController,
                 risk_overlay: RiskOverlay,
                 meta_train_freq: int = 10,
                 batch_size: int = 64,
                 device: str = 'cuda'):
        """
        Args:
            env: Portfolio environment
            hae: Heterogeneous Agent Ensemble
            mac: Meta-Adaptive Controller
            risk_overlay: Risk management overlay
            meta_train_freq: Frequency of MAC updates (in episodes)
            batch_size: Batch size for updates
            device: Device for computation
        """
        self.env = env
        self.hae = hae
        self.mac = mac
        self.risk_overlay = risk_overlay
        self.meta_train_freq = meta_train_freq
        self.batch_size = batch_size
        self.device = device
        self.agent_update_freq = 4  # Update agents every 4 steps instead of every step
        
        # Training statistics
        self.episode_rewards = []
        self.episode_values = []
        self.mac_weights_history = []
        self.step_count = 0
        
    def train_episode(self, episode: int, start_idx: int = 0) -> Dict:
        """Train for one episode."""
        state = self.env.reset(start_idx=start_idx)
        done = False
        episode_reward = 0.0
        episode_steps = 0
        
        episode_mac_weights = []
        episode_q_values = []
        episode_safety_scores = []
        
        while not done:
            # Get actions from all agents
            agent_actions = self.hae.select_actions(state, add_noise=True)
            
            # Get Q-values and safety scores
            q_values = self.hae.get_q_values(state, agent_actions)
            safety_scores = self.hae.get_safety_scores(state, agent_actions)
            
            # Aggregate actions using MAC
            aggregated_action = self.mac.aggregate_actions(state, agent_actions)
            
            # Get MAC weights for logging
            mac_weights = self.mac.compute_weights(state)
            episode_mac_weights.append(mac_weights.copy())
            
            # Apply risk overlay
            current_holdings = self.env.holdings
            current_prices = self.env.prices
            cash = self.env.cash
            portfolio_value = self.env._compute_portfolio_value()
            
            # Convert action to allocation
            current_allocation = (current_holdings * current_prices) / portfolio_value if portfolio_value > 0 else np.zeros(self.env.n_assets)
            constrained_allocation, new_cash = self.risk_overlay.apply(
                aggregated_action, current_holdings, current_prices, cash, portfolio_value
            )
            
            # Convert back to action space (approximate)
            action_for_env = constrained_allocation - current_allocation
            action_for_env = np.clip(action_for_env / self.env.max_trade_size, -1.0, 1.0)
            
            # Step environment
            next_state, reward, done, info = self.env.step(action_for_env)
            
            # Store experience for all agents
            for i, agent in enumerate(self.hae.agents):
                agent.replay_buffer.push(state, agent_actions[i], reward, next_state, done)
            
            # Update all agents (every agent_update_freq steps to speed up training)
            self.step_count += 1
            agent_update_stats = None
            if self.step_count % self.agent_update_freq == 0:
                agent_update_stats = self.hae.update_all(self.batch_size)
            
            # Store MAC experience
            q_bar = np.sum(mac_weights * q_values)
            c_bar = np.sum(mac_weights * safety_scores)
            self.mac.replay_buffer.push(state, mac_weights, q_bar, c_bar, reward)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            episode_q_values.append(q_bar)
            episode_safety_scores.append(c_bar)
        
        # Update MAC (every meta_train_freq episodes)
        mac_update_stats = None
        if (episode + 1) % self.meta_train_freq == 0:
            mac_update_stats = self.mac.update(self.batch_size)
        
        # Collect statistics
        final_value = self.env._compute_portfolio_value()
        
        # Check for NaN values
        if np.isnan(episode_reward) or np.isnan(final_value):
            logger.warning(f"NaN detected in episode {episode}: reward={episode_reward}, final_value={final_value}")
            episode_reward = 0.0 if np.isnan(episode_reward) else episode_reward
            final_value = self.env.initial_cash if np.isnan(final_value) else final_value
        
        return_val = (final_value - self.env.initial_cash) / self.env.initial_cash
        if np.isnan(return_val):
            return_val = 0.0
        
        episode_stats = {
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'final_value': final_value,
            'return': return_val,
            'mac_weights_mean': np.mean(episode_mac_weights, axis=0) if episode_mac_weights else None,
            'q_bar_mean': np.mean(episode_q_values) if episode_q_values and not np.isnan(episode_q_values).any() else 0.0,
            'c_bar_mean': np.mean(episode_safety_scores) if episode_safety_scores and not np.isnan(episode_safety_scores).any() else 0.0,
            'agent_updates': agent_update_stats,
            'mac_update': mac_update_stats
        }
        
        return episode_stats
    
    def train(self, n_episodes: int, start_idx: int = 0, log_dir: str = "logs"):
        """Train for n_episodes."""
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"Starting training for {n_episodes} episodes")
        
        for episode in tqdm(range(n_episodes)):
            stats = self.train_episode(episode, start_idx=start_idx)
            
            self.episode_rewards.append(stats['reward'])
            self.episode_values.append(stats['final_value'])
            if stats['mac_weights_mean'] is not None:
                self.mac_weights_history.append(stats['mac_weights_mean'])
            
            # Logging
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Reward: {stats['reward']:.4f} | "
                    f"Return: {stats['return']:.4f} | "
                    f"Final Value: {stats['final_value']:.2f}"
                )
        
        logger.info("Training completed")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_values': self.episode_values,
            'mac_weights_history': self.mac_weights_history
        }
    
    def save_models(self, save_dir: str):
        """Save all models."""
        os.makedirs(save_dir, exist_ok=True)
        self.hae.save_all(f"{save_dir}/agents")
        self.mac.save(f"{save_dir}/mac.pth")
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load all models."""
        self.hae.load_all(f"{load_dir}/agents")
        self.mac.load(f"{load_dir}/mac.pth")
        logger.info(f"Models loaded from {load_dir}")

