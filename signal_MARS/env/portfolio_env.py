"""
Portfolio trading environment for reinforcement learning.

Implements a gym-style environment for portfolio management.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from .state_builder import StateBuilder
from ..config import SignalMARSConfig


class PortfolioEnv:
    """
    Portfolio trading environment.
    
    Manages portfolio state, executes trades, and computes rewards.
    """
    
    def __init__(self, config: SignalMARSConfig, state_builder: StateBuilder):
        """
        Initialize portfolio environment.
        
        Args:
            config: Signal-MARS configuration
            state_builder: State builder instance
        """
        self.config = config
        self.state_builder = state_builder
        
        # Portfolio state
        self.num_assets = config.env.num_assets
        self.initial_cash = config.env.initial_cash
        self.transaction_cost = config.env.transaction_cost
        
        # Reset state
        self.reset()
    
    def reset(self, initial_prices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Args:
            initial_prices: Initial asset prices (num_assets,)
        
        Returns:
            Initial state vector
        """
        # Portfolio state
        self.positions = np.zeros(self.num_assets)  # Number of shares per asset
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        
        # Price history (for computing returns)
        if initial_prices is None:
            initial_prices = np.ones(self.num_assets)  # Placeholder
        
        self.current_prices = initial_prices.copy()
        self.price_history = [initial_prices.copy()]
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = self.config.env.max_steps
        
        # Return history (for reward computation)
        self.returns_history = []
        
        # Build initial state
        portfolio_state = self._get_portfolio_state()
        market_data = self._get_market_data()
        
        return self.state_builder.build_full_state(portfolio_state, market_data)
    
    def step(self, action: np.ndarray, new_prices: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action vector of shape (num_assets,)
                   Represents target portfolio weights
            new_prices: New asset prices of shape (num_assets,)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Clip action to valid range [0, 1] (no shorting by default)
        if not self.config.risk.allow_shorting:
            action = np.clip(action, 0.0, 1.0)
        
        # Normalize action to sum to 1 (portfolio weights)
        action = action / (np.sum(action) + 1e-8)
        
        # Execute trade
        old_value = self.portfolio_value
        self._execute_trade(action, new_prices)
        
        # Update prices
        self.current_prices = new_prices.copy()
        self.price_history.append(new_prices.copy())
        
        # Compute reward
        new_value = self.portfolio_value
        portfolio_return = (new_value - old_value) / old_value if old_value > 0 else 0.0
        reward = portfolio_return
        
        # Store return
        self.returns_history.append(portfolio_return)
        
        # Build next state
        portfolio_state = self._get_portfolio_state()
        market_data = self._get_market_data()
        next_state = self.state_builder.build_full_state(portfolio_state, market_data)
        
        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Info dictionary
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "returns": portfolio_return,
            "step": self.step_count
        }
        
        return next_state, reward, done, info
    
    def _execute_trade(self, target_weights: np.ndarray, prices: np.ndarray):
        """
        Execute trade to rebalance portfolio to target weights.
        
        Args:
            target_weights: Target portfolio weights (num_assets,)
            prices: Current asset prices (num_assets,)
        """
        # Current portfolio value
        current_value = self.portfolio_value
        
        # Target dollar amounts
        target_values = target_weights * current_value
        
        # Current dollar amounts
        current_values = self.positions * prices
        
        # Compute trades needed
        trades = target_values - current_values
        
        # Apply transaction costs
        trade_costs = np.abs(trades) * self.transaction_cost
        total_cost = np.sum(trade_costs)
        
        # Update cash (subtract trades and costs)
        self.cash -= np.sum(trades) + total_cost
        
        # Update positions
        new_positions = target_values / prices
        self.positions = new_positions
        
        # Update portfolio value
        self.portfolio_value = np.sum(self.positions * prices) + self.cash
    
    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state dictionary."""
        # Compute recent returns
        if len(self.price_history) >= 2:
            prev_prices = self.price_history[-2]
            returns = (self.current_prices - prev_prices) / prev_prices
        else:
            returns = np.zeros(self.num_assets)
        
        return {
            "positions": self.positions.copy(),
            "cash": self.cash,
            "returns": returns,
            "portfolio_value": self.portfolio_value
        }
    
    def _get_market_data(self) -> Dict[str, Any]:
        """Get market data dictionary for signal computation."""
        # Convert price history to array
        price_array = np.array(self.price_history)
        
        return {
            "prices": price_array,
            "current_prices": self.current_prices,
            "returns": self._compute_returns()
        }
    
    def _compute_returns(self) -> np.ndarray:
        """Compute returns from price history."""
        if len(self.price_history) < 2:
            return np.zeros((1, self.num_assets))
        
        price_array = np.array(self.price_history)
        returns = np.diff(price_array, axis=0) / price_array[:-1]
        return returns
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.portfolio_value
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Compute Sharpe ratio of episode returns.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return > 0:
            sharpe = np.sqrt(252) * mean_return / std_return  # Annualized
            return sharpe
        return 0.0
    
    def get_max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        if len(self.returns_history) == 0:
            return 0.0
        
        # Compute cumulative returns
        cumulative = np.cumprod(1 + np.array(self.returns_history))
        
        # Compute running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Compute drawdown
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)  # Most negative drawdown
