"""
Portfolio Management Environment (MDP) as specified in the paper.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioEnv:
    """
    Portfolio management environment with:
    - State: cash, holdings, features
    - Action: continuous allocation changes
    - Reward: return - transaction cost - risk penalty
    - Risk penalty: 0.5 * σ_30d + 2.0 * DD_30d
    """
    
    def __init__(self, 
                 features_dict: Dict[str, pd.DataFrame],
                 initial_cash: float = 1e6,
                 transaction_cost: float = 0.001,
                 max_trade_size: float = 0.1,
                 risk_window: int = 30):
        """
        Args:
            features_dict: Dict mapping ticker to feature DataFrame
            initial_cash: Initial cash balance
            transaction_cost: Transaction cost rate (0.1%)
            max_trade_size: Maximum trade size as fraction of portfolio
            risk_window: Window for computing risk metrics (30 days)
        """
        self.features_dict = features_dict
        self.tickers = sorted(list(features_dict.keys()))
        self.n_assets = len(self.tickers)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_trade_size = max_trade_size
        self.risk_window = risk_window
        
        # Align all features to common dates
        self._align_data()
        
        # State tracking
        self.current_step = 0
        self.cash = initial_cash
        self.holdings = np.zeros(self.n_assets)  # Number of shares per asset
        self.prices = None  # Current prices
        self.portfolio_value_history = []
        self.returns_history = []
        
        # Risk tracking
        self.volatility_history = []
        self.drawdown_history = []
        
    def _align_data(self):
        """Align all asset data to common date index."""
        all_dates = set()
        for df in self.features_dict.values():
            all_dates.update(df.index)
        self.dates = sorted(all_dates)
        
        # Extract prices (first feature)
        self.price_data = {}
        for ticker in self.tickers:
            df = self.features_dict[ticker]
            self.price_data[ticker] = df['price'].reindex(self.dates, method='ffill')
        
        # Extract all features
        self.feature_data = {}
        for ticker in self.tickers:
            df = self.features_dict[ticker]
            aligned = df.reindex(self.dates, method='ffill')
            self.feature_data[ticker] = aligned
        
        logger.info(f"Aligned {len(self.dates)} dates for {self.n_assets} assets")
    
    def reset(self, start_idx: int = 0) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = start_idx
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets)
        self.portfolio_value_history = [self.initial_cash]
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state vector:
        - Cash balance (normalized)
        - Holdings per asset (normalized)
        - Feature vector per asset (5 features)
        """
        state_parts = []
        
        # Normalized cash (log scale) - ensure it's a 1D array
        cash_norm = np.log1p(self.cash / self.initial_cash)
        cash_norm = np.atleast_1d(cash_norm)
        state_parts.append(cash_norm)
        
        # Current prices
        current_date = self.dates[self.current_step]
        self.prices = np.array([self.price_data[ticker].loc[current_date] 
                               for ticker in self.tickers])
        
        # Portfolio value
        portfolio_value = self.cash + np.sum(self.holdings * self.prices)
        
        # Normalized holdings (as fraction of portfolio) - ensure it's a 1D array
        if portfolio_value > 0:
            holdings_norm = (self.holdings * self.prices) / portfolio_value
        else:
            holdings_norm = np.zeros(self.n_assets)
        holdings_norm = np.atleast_1d(holdings_norm)
        state_parts.append(holdings_norm)
        
        # Feature vectors per asset (5 features each)
        for ticker in self.tickers:
            features = self.feature_data[ticker].loc[current_date]
            feature_vec = features[['price', 'macd', 'rsi', 'cci', 'adx']].values
            # Ensure feature_vec is always a 1D array
            feature_vec = np.atleast_1d(feature_vec)
            if feature_vec.ndim > 1:
                feature_vec = feature_vec.flatten()
            state_parts.append(feature_vec)
        
        # Ensure all parts are 1D before concatenation
        state_parts = [np.atleast_1d(part) for part in state_parts]
        state = np.concatenate(state_parts)
        return state.astype(np.float32)
    
    def _compute_portfolio_value(self) -> float:
        """Compute current portfolio value."""
        if self.prices is None:
            current_date = self.dates[self.current_step]
            self.prices = np.array([self.price_data[ticker].loc[current_date] 
                                   for ticker in self.tickers])
        return self.cash + np.sum(self.holdings * self.prices)
    
    def _compute_risk_penalty(self) -> float:
        """
        Compute risk penalty: ρ_t = 0.5 * σ_30d + 2.0 * DD_30d
        
        Where:
        - σ_30d: 30-day rolling volatility
        - DD_30d: 30-day maximum drawdown
        """
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # Compute returns
        returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        
        if len(returns) < self.risk_window:
            # Use available data
            window = len(returns)
        else:
            window = self.risk_window
        
        if window < 2:
            return 0.0
        
        recent_returns = returns[-window:]
        
        # Volatility (annualized)
        volatility = np.std(recent_returns) * np.sqrt(252)
        
        # Maximum drawdown over window
        recent_values = self.portfolio_value_history[-window:]
        peak = np.maximum.accumulate(recent_values)
        drawdown = (peak - recent_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Risk penalty
        risk_penalty = 0.5 * volatility + 2.0 * max_drawdown
        
        self.volatility_history.append(volatility)
        self.drawdown_history.append(max_drawdown)
        
        return risk_penalty
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).
        
        Args:
            action: Continuous vector in [-1, 1]^D representing target allocation change
            
        Returns:
            next_state: Next state vector
            reward: R_t = (V_{t+1} - V_t) / V_t - C_t - ρ_t
            done: Whether episode is complete
            info: Additional information
        """
        if self.current_step >= len(self.dates) - 1:
            return self._get_state(), 0.0, True, {}
        
        # Current portfolio value
        V_t = self._compute_portfolio_value()
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action by max trade size
        target_allocation_change = action * self.max_trade_size
        
        # Compute target holdings (as fraction of portfolio)
        current_allocation = (self.holdings * self.prices) / V_t if V_t > 0 else np.zeros(self.n_assets)
        target_allocation = current_allocation + target_allocation_change
        
        # Apply risk management overlay (will be done externally, but ensure no shorts)
        target_allocation = np.clip(target_allocation, 0.0, 1.0)
        
        # Normalize to sum to <= 1.0 (leave cash buffer)
        total_allocation = np.sum(target_allocation)
        if total_allocation > 0.95:  # Keep 5% cash buffer
            target_allocation = target_allocation * 0.95 / total_allocation
        
        # Compute target value per asset
        target_values = target_allocation * V_t
        
        # Compute trades
        current_values = self.holdings * self.prices
        trade_values = target_values - current_values
        
        # Compute transaction costs
        transaction_cost = np.sum(np.abs(trade_values)) * self.transaction_cost
        
        # Execute trades
        for i, ticker in enumerate(self.tickers):
            trade_value = trade_values[i]
            if abs(trade_value) > 1e-6:  # Avoid tiny trades
                shares_to_trade = trade_value / self.prices[i]
                self.holdings[i] += shares_to_trade
                self.cash -= trade_value
        
        # Deduct transaction cost
        self.cash -= transaction_cost
        
        # Safety check: ensure cash doesn't go negative
        if self.cash < 0:
            # Sell holdings to cover negative cash (emergency liquidation)
            for i in range(self.n_assets):
                if self.holdings[i] > 0 and self.cash < 0:
                    shares_to_sell = min(self.holdings[i], abs(self.cash) / self.prices[i] + 1e-6)
                    self.holdings[i] -= shares_to_sell
                    self.cash += shares_to_sell * self.prices[i]
                    if self.cash >= 0:
                        break
        
        # Move to next step
        self.current_step += 1
        
        # New portfolio value
        V_t1 = self._compute_portfolio_value()
        
        # Check for NaN in portfolio value
        if np.isnan(V_t1) or np.isinf(V_t1) or V_t1 < 0:
            V_t1 = self.initial_cash  # Reset to initial if invalid
            self.cash = self.initial_cash
            self.holdings = np.zeros(self.n_assets)
        
        self.portfolio_value_history.append(V_t1)
        
        # Compute reward
        if V_t > 1e-6:  # Avoid division by very small numbers
            return_rate = (V_t1 - V_t) / V_t
            transaction_cost_penalty = transaction_cost / V_t
        else:
            return_rate = 0.0
            transaction_cost_penalty = 0.0
        
        # Risk penalty
        risk_penalty = self._compute_risk_penalty()
        
        # Total reward (with NaN checks)
        reward = return_rate - transaction_cost_penalty - risk_penalty
        
        # Check for NaN and clip to reasonable range
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        reward = np.clip(reward, -1.0, 1.0)  # Clip to reasonable range
        
        self.returns_history.append(return_rate)
        
        # Check if done
        done = self.current_step >= len(self.dates) - 1
        
        # Info
        info = {
            'portfolio_value': V_t1,
            'return': return_rate,
            'transaction_cost': transaction_cost,
            'risk_penalty': risk_penalty,
            'cash': self.cash,
            'holdings': self.holdings.copy()
        }
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Get state dimension."""
        # 1 (cash) + n_assets (holdings) + n_assets * 5 (features)
        return 1 + self.n_assets + self.n_assets * 5
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.n_assets

