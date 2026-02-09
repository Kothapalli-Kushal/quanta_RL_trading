"""
Risk overlay for enforcing hard constraints before trade execution.

Final gate that applies risk constraints to actions.
"""

from typing import Dict, Any, Tuple
import numpy as np

from ..config import SignalMARSConfig


class RiskOverlay:
    """
    Applies hard risk constraints to actions.
    
    Enforces:
    - Max position size
    - No shorting (if configured)
    - Cash buffer
    - Leverage limits
    """
    
    def __init__(self, config: SignalMARSConfig):
        """
        Initialize risk overlay.
        
        Args:
            config: Signal-MARS configuration
        """
        self.config = config
        self.risk_config = config.risk
    
    def apply_constraints(self, action: np.ndarray, 
                         portfolio_state: Dict[str, Any]) -> np.ndarray:
        """
        Apply risk constraints to action.
        
        Args:
            action: Proposed action (portfolio weights)
            portfolio_state: Current portfolio state
        
        Returns:
            Constrained action
        """
        constrained_action = action.copy()
        
        # 1. No shorting constraint
        if not self.risk_config.allow_shorting:
            constrained_action = np.clip(constrained_action, 0.0, 1.0)
        
        # 2. Max position size constraint
        max_position = self.risk_config.max_position_size
        constrained_action = np.clip(constrained_action, 0.0, max_position)
        
        # 3. Cash buffer constraint
        # Ensure we don't allocate 100% of capital
        max_allocation = 1.0 - self.risk_config.cash_buffer
        if np.sum(constrained_action) > max_allocation:
            constrained_action = constrained_action / np.sum(constrained_action) * max_allocation
        
        # 4. Leverage constraint
        # Check if action would exceed leverage limit
        portfolio_value = portfolio_state.get("portfolio_value", 1.0)
        positions = portfolio_state.get("positions", np.zeros(len(action)))
        prices = portfolio_state.get("prices", np.ones(len(action)))
        
        # Compute current leverage
        current_leverage = self._compute_leverage(
            positions, prices, portfolio_state.get("cash", 0.0), portfolio_value
        )
        
        # If action would increase leverage beyond limit, scale down
        if current_leverage > self.risk_config.max_leverage:
            # Scale down to maintain leverage limit
            scale_factor = self.risk_config.max_leverage / current_leverage
            constrained_action = constrained_action * scale_factor
        
        # 5. Concentration limit
        concentration = np.sum(constrained_action ** 2)
        if concentration > self.risk_config.concentration_limit:
            # Redistribute to reduce concentration
            constrained_action = self._reduce_concentration(
                constrained_action, self.risk_config.concentration_limit
            )
        
        # Renormalize to ensure valid portfolio weights
        constrained_action = constrained_action / (np.sum(constrained_action) + 1e-8)
        
        return constrained_action
    
    def _compute_leverage(self, positions: np.ndarray, prices: np.ndarray,
                         cash: float, portfolio_value: float) -> float:
        """Compute current leverage."""
        if portfolio_value == 0:
            return 1.0
        
        gross_exposure = np.sum(np.abs(positions * prices))
        return gross_exposure / portfolio_value
    
    def _reduce_concentration(self, weights: np.ndarray, max_concentration: float) -> np.ndarray:
        """
        Reduce portfolio concentration.
        
        Args:
            weights: Portfolio weights
            max_concentration: Maximum allowed concentration
        
        Returns:
            Less concentrated weights
        """
        # Simple approach: flatten large positions
        sorted_indices = np.argsort(weights)[::-1]
        reduced_weights = weights.copy()
        
        # Redistribute from largest positions
        excess = np.sum(weights ** 2) - max_concentration
        if excess > 0:
            # Reduce largest positions and redistribute
            for idx in sorted_indices[:len(weights)//2]:
                reduction = reduced_weights[idx] * 0.1  # Reduce by 10%
                reduced_weights[idx] -= reduction
            
            # Renormalize
            reduced_weights = reduced_weights / np.sum(reduced_weights)
        
        return reduced_weights
    
    def validate_action(self, action: np.ndarray, 
                       portfolio_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that action satisfies all constraints.
        
        Args:
            action: Action to validate
            portfolio_state: Portfolio state
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check no shorting
        if not self.risk_config.allow_shorting and np.any(action < 0):
            return False, "Action contains negative weights (shorting not allowed)"
        
        # Check max position size
        if np.any(action > self.risk_config.max_position_size):
            return False, f"Action exceeds max position size ({self.risk_config.max_position_size})"
        
        # Check cash buffer
        if np.sum(action) > 1.0 - self.risk_config.cash_buffer:
            return False, f"Action exceeds cash buffer limit"
        
        # Check concentration
        concentration = np.sum(action ** 2)
        if concentration > self.risk_config.concentration_limit:
            return False, f"Action exceeds concentration limit ({self.risk_config.concentration_limit})"
        
        return True, ""
