"""
Risk Management Overlay: Hard constraints applied after action aggregation.
"""
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskOverlay:
    """
    Rule-based risk management overlay enforcing:
    - Max position concentration: 20%
    - No short-selling
    - Maintain cash buffer
    - Trades clipped to feasibility
    """
    
    def __init__(self,
                 max_position_concentration: float = 0.20,
                 min_cash_buffer: float = 0.05,
                 max_leverage: float = 1.0):
        """
        Args:
            max_position_concentration: Maximum allocation to single asset (20%)
            min_cash_buffer: Minimum cash buffer to maintain (5%)
            max_leverage: Maximum leverage (1.0 = no leverage)
        """
        self.max_position_concentration = max_position_concentration
        self.min_cash_buffer = min_cash_buffer
        self.max_leverage = max_leverage
    
    def apply(self, 
              action: np.ndarray,
              current_holdings: np.ndarray,
              current_prices: np.ndarray,
              cash: float,
              portfolio_value: float) -> Tuple[np.ndarray, float]:
        """
        Apply risk management constraints to action.
        
        Args:
            action: Target allocation vector (sums to <= 1.0)
            current_holdings: Current number of shares per asset
            current_prices: Current prices per asset
            cash: Current cash balance
            portfolio_value: Total portfolio value
            
        Returns:
            (constrained_action, new_cash): Constrained allocation and remaining cash
        """
        # Ensure no negative allocations (no short-selling)
        action = np.clip(action, 0.0, 1.0)
        
        # Enforce max position concentration
        action = np.clip(action, 0.0, self.max_position_concentration)
        
        # Ensure total allocation leaves cash buffer
        total_allocation = np.sum(action)
        max_total_allocation = 1.0 - self.min_cash_buffer
        
        if total_allocation > max_total_allocation:
            # Scale down proportionally
            action = action * (max_total_allocation / total_allocation)
        
        # Compute target values
        target_values = action * portfolio_value
        
        # Compute trades
        current_values = current_holdings * current_prices
        trade_values = target_values - current_values
        
        # Clip trades to available cash
        total_buy = np.sum(np.clip(trade_values, 0, np.inf))
        total_sell = np.sum(np.clip(-trade_values, 0, np.inf))
        
        # Available cash for buying
        available_cash = cash + total_sell
        
        if total_buy > available_cash:
            # Scale down buys proportionally
            scale = available_cash / total_buy if total_buy > 0 else 0.0
            buy_mask = trade_values > 0
            trade_values[buy_mask] = trade_values[buy_mask] * scale
        
        # Recompute allocation from constrained trades
        new_values = current_values + trade_values
        new_portfolio_value = np.sum(new_values) + (cash - np.sum(np.clip(trade_values, 0, np.inf)) + total_sell)
        
        if new_portfolio_value > 0:
            constrained_action = new_values / new_portfolio_value
        else:
            constrained_action = np.zeros_like(action)
        
        # Update cash
        new_cash = cash - np.sum(np.clip(trade_values, 0, np.inf)) + total_sell
        
        return constrained_action, new_cash
    
    def compute_risk_metrics(self,
                           holdings: np.ndarray,
                           prices: np.ndarray,
                           cash: float) -> dict:
        """
        Compute risk metrics for monitoring.
        
        Returns:
            Dictionary with risk metrics:
            - concentration: Herfindahl-Hirschman Index
            - max_position: Maximum position size
            - leverage: Current leverage
            - cash_ratio: Cash as fraction of portfolio
        """
        portfolio_value = cash + np.sum(holdings * prices)
        
        if portfolio_value <= 0:
            return {
                'concentration': 0.0,
                'max_position': 0.0,
                'leverage': 0.0,
                'cash_ratio': 1.0
            }
        
        # Position values
        position_values = holdings * prices
        allocations = position_values / portfolio_value
        
        # Herfindahl-Hirschman Index (concentration)
        hhi = np.sum(allocations ** 2)
        
        # Maximum position
        max_position = np.max(allocations) if len(allocations) > 0 else 0.0
        
        # Leverage (should be 1.0 for long-only)
        leverage = np.sum(np.abs(position_values)) / portfolio_value
        
        # Cash ratio
        cash_ratio = cash / portfolio_value
        
        return {
            'concentration': hhi,
            'max_position': max_position,
            'leverage': leverage,
            'cash_ratio': cash_ratio
        }

