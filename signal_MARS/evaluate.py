"""
Evaluation script for Signal-MARS framework.

Runs trained model in test mode without exploration noise.
"""

import numpy as np
import torch
import os
import sys
from typing import Dict, Any, List
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SignalMARSConfig, get_default_config
from signals import SignalFactory
from macro import MacroLoader
from env import PortfolioEnv, StateBuilder
from agents import DDPGAgent
from mac import MACController
from risk import RiskOverlay
from utils import setup_logger, set_seed, sharpe_ratio


class SignalMARSEvaluator:
    """
    Evaluator for Signal-MARS.
    
    Runs trained models in test mode and outputs:
    - Equity curve
    - Drawdown
    - Sharpe ratio
    - MAC weight dynamics
    """
    
    def __init__(self, config: SignalMARSConfig, checkpoint_dir: str):
        """
        Initialize evaluator.
        
        Args:
            config: Signal-MARS configuration
            checkpoint_dir: Directory containing trained checkpoints
        """
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Set seed
        set_seed(config.training.seed)
        
        # Setup logging
        self.logger = setup_logger("eval", config.eval.results_dir)
        self.logger.info("Initializing Signal-MARS evaluator...")
        
        # Create signal factory
        self.signal_factory = SignalFactory(
            enabled_signals=config.signal.enabled_signals
        )
        
        # Create macro loader
        self.macro_loader = MacroLoader(
            forward_fill_days=config.macro.forward_fill_days
        )
        # TODO: Load actual macro data
        
        # Create state builder
        self.state_builder = StateBuilder(
            config, self.signal_factory, self.macro_loader
        )
        
        # Create environment
        self.env = PortfolioEnv(config, self.state_builder)
        
        # Load agents
        self.agents: List[DDPGAgent] = []
        state_dim = self.state_builder.get_state_dim()
        action_dim = config.env.num_assets
        
        for i, agent_type in enumerate(config.agent.agent_types):
            agent = DDPGAgent(
                state_dim, action_dim, agent_type, config, config.training.device
            )
            
            # Load checkpoint
            agent_path = os.path.join(checkpoint_dir, f"agent_{i}_{agent_type}.pth")
            if os.path.exists(agent_path):
                agent.load(agent_path)
                self.logger.info(f"Loaded agent {i} ({agent_type}) from {agent_path}")
            else:
                self.logger.warning(f"Agent checkpoint not found: {agent_path}")
            
            agent.eval()  # Set to evaluation mode
            self.agents.append(agent)
        
        # Load MAC controller
        if config.mac.use_mac:
            self.mac_controller = MACController(
                config.mac.mac_input_dim,
                config.agent.num_agents,
                config.mac.mac_hidden_dims
            ).to(config.training.device)
            
            mac_path = os.path.join(checkpoint_dir, "mac_controller.pth")
            if os.path.exists(mac_path):
                self.mac_controller.load_state_dict(torch.load(mac_path, map_location=config.training.device))
                self.logger.info(f"Loaded MAC controller from {mac_path}")
            else:
                self.logger.warning(f"MAC checkpoint not found: {mac_path}")
            
            self.mac_controller.eval()
        else:
            self.mac_controller = None
        
        # Create risk overlay
        self.risk_overlay = RiskOverlay(config)
        
        # Results storage
        self.equity_curve = []
        self.drawdown_curve = []
        self.mac_weight_history = []
        self.returns_history = []
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        
        # Reset environment
        initial_prices = np.ones(self.config.env.num_assets)  # Placeholder
        state = self.env.reset(initial_prices)
        
        done = False
        step = 0
        
        while not done:
            # Get actions from all agents (no exploration)
            agent_actions = []
            for agent in self.agents:
                action = agent.act(state, explore=False)  # No exploration in eval
                agent_actions.append(action)
            
            # MAC weighting
            if self.mac_controller is not None:
                mac_state = self.state_builder.build_mac_state(
                    self.env._get_market_data()
                )
                mac_weights = self.mac_controller.get_weights(mac_state)
                self.mac_weight_history.append(mac_weights.copy())
                
                # Aggregate actions
                aggregated_action = np.sum(
                    [w * a for w, a in zip(mac_weights, agent_actions)], axis=0
                )
            else:
                # Uniform weighting
                mac_weights = np.ones(len(self.agents)) / len(self.agents)
                aggregated_action = np.mean(agent_actions, axis=0)
            
            # Apply risk overlay
            portfolio_state = self.env._get_portfolio_state()
            portfolio_state["prices"] = self.env.current_prices
            constrained_action = self.risk_overlay.apply_constraints(
                aggregated_action, portfolio_state
            )
            
            # Step environment
            # TODO: Get actual new prices from test data
            new_prices = self.env.current_prices * (1 + np.random.randn(self.config.env.num_assets) * 0.01)
            next_state, reward, done, info = self.env.step(constrained_action, new_prices)
            
            # Store results
            self.equity_curve.append(self.env.get_portfolio_value())
            self.returns_history.append(reward)
            
            # Compute drawdown
            if len(self.equity_curve) > 1:
                peak = max(self.equity_curve)
                current = self.equity_curve[-1]
                drawdown = (current - peak) / peak
                self.drawdown_curve.append(drawdown)
            
            state = next_state
            step += 1
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save results
        if self.config.eval.save_results:
            self._save_results(metrics)
        
        self.logger.info("Evaluation completed!")
        return metrics
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        returns = np.array(self.returns_history)
        equity = np.array(self.equity_curve)
        
        # Sharpe ratio
        sharpe = sharpe_ratio(returns)
        
        # Maximum drawdown
        if len(self.drawdown_curve) > 0:
            max_drawdown = min(self.drawdown_curve)
        else:
            max_drawdown = 0.0
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0] if len(equity) > 0 else 0.0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # MAC weight statistics
        mac_stats = {}
        if len(self.mac_weight_history) > 0:
            mac_weights_array = np.array(self.mac_weight_history)
            for i, agent_type in enumerate(self.config.agent.agent_types):
                mac_stats[f"mac_weight_mean_{agent_type}"] = float(np.mean(mac_weights_array[:, i]))
                mac_stats[f"mac_weight_std_{agent_type}"] = float(np.std(mac_weights_array[:, i]))
        
        metrics = {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "volatility": volatility,
            "final_portfolio_value": float(equity[-1]) if len(equity) > 0 else 0.0,
            "num_steps": len(returns),
            **mac_stats
        }
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save evaluation results."""
        os.makedirs(self.config.eval.results_dir, exist_ok=True)
        
        # Save metrics
        import json
        metrics_path = os.path.join(self.config.eval.results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save equity curve
        equity_path = os.path.join(self.config.eval.results_dir, "equity_curve.npy")
        np.save(equity_path, np.array(self.equity_curve))
        
        # Save MAC weight history
        if len(self.mac_weight_history) > 0:
            mac_path = os.path.join(self.config.eval.results_dir, "mac_weights.npy")
            np.save(mac_path, np.array(self.mac_weight_history))
        
        # Plot equity curve
        if self.config.eval.render and HAS_MATPLOTLIB:
            self._plot_results()
        elif self.config.eval.render and not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, skipping plots")
        
        self.logger.info(f"Results saved to {self.config.eval.results_dir}")
    
    def _plot_results(self):
        """Plot evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Equity curve
        axes[0, 0].plot(self.equity_curve)
        axes[0, 0].set_title("Equity Curve")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Portfolio Value")
        axes[0, 0].grid(True)
        
        # Drawdown
        if len(self.drawdown_curve) > 0:
            axes[0, 1].plot(self.drawdown_curve)
            axes[0, 1].set_title("Drawdown")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Drawdown")
            axes[0, 1].grid(True)
        
        # Returns distribution
        axes[1, 0].hist(self.returns_history, bins=50)
        axes[1, 0].set_title("Returns Distribution")
        axes[1, 0].set_xlabel("Return")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)
        
        # MAC weights over time
        if len(self.mac_weight_history) > 0:
            mac_weights_array = np.array(self.mac_weight_history)
            for i, agent_type in enumerate(self.config.agent.agent_types):
                axes[1, 1].plot(mac_weights_array[:, i], label=agent_type, alpha=0.7)
            axes[1, 1].set_title("MAC Weights Over Time")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Weight")
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.eval.results_dir, "evaluation_plots.png")
        plt.savefig(plot_path)
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Signal-MARS model")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing trained checkpoints")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # TODO: Load from file
        config = get_default_config()
    else:
        config = get_default_config()
    
    # Create evaluator
    evaluator = SignalMARSEvaluator(config, args.checkpoint_dir)
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
