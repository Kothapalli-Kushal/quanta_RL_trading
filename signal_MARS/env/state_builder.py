"""
State builder for constructing RL agent states.

Handles portfolio state, signal features, and macro features.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from ..signals import SignalFactory
from ..macro import MacroLoader
from ..config import SignalMARSConfig


class StateBuilder:
    """
    Constructs state representations for RL agents.
    
    Combines:
    - Portfolio state (positions, cash, returns)
    - Signal features (from SignalFactory)
    - Macro features (from MacroLoader)
    """
    
    def __init__(self, config: SignalMARSConfig, 
                 signal_factory: Optional[SignalFactory] = None,
                 macro_loader: Optional[MacroLoader] = None):
        """
        Initialize state builder.
        
        Args:
            config: Signal-MARS configuration
            signal_factory: Signal factory instance (optional)
            macro_loader: Macro loader instance (optional)
        """
        self.config = config
        self.signal_factory = signal_factory
        self.macro_loader = macro_loader
        
        # Compute state dimensions
        self._compute_dimensions()
    
    def _compute_dimensions(self):
        """Compute state dimensions based on config."""
        self.portfolio_dim = 0
        self.signal_dim = 0
        self.macro_dim = 0
        
        if self.config.env.include_portfolio_state:
            # Positions + cash + returns
            self.portfolio_dim = self.config.env.num_assets + 1 + self.config.env.num_assets
        
        if self.config.env.use_signals and self.signal_factory is not None:
            self.signal_dim = self.signal_factory.get_signal_dim()
        
        if self.config.env.use_macro and self.macro_loader is not None:
            self.macro_dim = self.macro_loader.get_feature_dim()
        
        self.total_dim = self.portfolio_dim + self.signal_dim + self.macro_dim
    
    def build_full_state(self, portfolio_state: Dict[str, Any],
                        market_data: Dict[str, Any],
                        current_date: Any = None) -> np.ndarray:
        """
        Build full state vector for RL agents.
        
        Args:
            portfolio_state: Dictionary with keys:
                - 'positions': (num_assets,) array of current positions
                - 'cash': Current cash balance
                - 'returns': (num_assets,) array of recent returns (optional)
            market_data: Raw market data for signal computation
            current_date: Current date for macro feature alignment
        
        Returns:
            Full state vector of shape (total_dim,)
        """
        state_parts = []
        
        # Portfolio state
        if self.config.env.include_portfolio_state:
            portfolio_vec = self._build_portfolio_state(portfolio_state)
            state_parts.append(portfolio_vec)
        
        # Signal features
        if self.config.env.use_signals and self.signal_factory is not None:
            signal_vec = self.signal_factory.compute(market_data)
            state_parts.append(signal_vec)
        
        # Macro features
        if self.config.env.use_macro and self.macro_loader is not None:
            macro_vec = self._get_macro_features(current_date)
            state_parts.append(macro_vec)
        
        if len(state_parts) == 0:
            return np.zeros(self.total_dim)
        
        return np.concatenate(state_parts)
    
    def build_mac_state(self, market_data: Dict[str, Any],
                       current_date: Any = None) -> np.ndarray:
        """
        Build reduced state for MAC controller.
        
        MAC only needs macro features and summarized signals (not full state).
        
        Args:
            market_data: Raw market data
            current_date: Current date
        
        Returns:
            MAC state vector
        """
        mac_parts = []
        
        # Macro features
        if self.config.env.use_macro and self.macro_loader is not None:
            macro_vec = self._get_macro_features(current_date)
            mac_parts.append(macro_vec)
        
        # Summarized signals (mean, std, etc.)
        if self.config.env.use_signals and self.signal_factory is not None:
            signal_vec = self.signal_factory.compute(market_data)
            # Summarize: mean, std, min, max, etc.
            signal_summary = self._summarize_signals(signal_vec)
            mac_parts.append(signal_summary)
        
        if len(mac_parts) == 0:
            return np.zeros(self.config.mac.mac_input_dim)
        
        return np.concatenate(mac_parts)
    
    def _build_portfolio_state(self, portfolio_state: Dict[str, Any]) -> np.ndarray:
        """Build portfolio state vector."""
        parts = []
        
        # Positions (normalized by portfolio value)
        positions = portfolio_state.get('positions', np.zeros(self.config.env.num_assets))
        parts.append(positions)
        
        # Cash (as fraction of initial capital)
        cash = portfolio_state.get('cash', 0.0)
        cash_frac = cash / self.config.env.initial_cash
        parts.append(np.array([cash_frac]))
        
        # Returns (recent returns for each asset)
        returns = portfolio_state.get('returns', np.zeros(self.config.env.num_assets))
        parts.append(returns)
        
        return np.concatenate(parts)
    
    def _get_macro_features(self, current_date: Any) -> np.ndarray:
        """Get macro features for current date."""
        if current_date is None:
            return np.zeros(self.macro_dim)
        
        try:
            return self.macro_loader.get_features(current_date)
        except Exception as e:
            print(f"Warning: Could not get macro features: {e}")
            return np.zeros(self.macro_dim)
    
    def _summarize_signals(self, signal_vec: np.ndarray) -> np.ndarray:
        """
        Summarize signal vector for MAC.
        
        Computes statistics: mean, std, min, max, etc.
        """
        summary = [
            np.mean(signal_vec),
            np.std(signal_vec),
            np.min(signal_vec),
            np.max(signal_vec),
            np.median(signal_vec),
            np.percentile(signal_vec, 25),
            np.percentile(signal_vec, 75),
            np.sum(signal_vec > 0) / len(signal_vec),  # Positive fraction
            np.sum(signal_vec < 0) / len(signal_vec),  # Negative fraction
            np.sum(np.abs(signal_vec)) / len(signal_vec)  # Mean absolute value
        ]
        return np.array(summary)
    
    def build_state_tensor(self, portfolio_state: Dict[str, Any],
                          market_data: Dict[str, Any],
                          current_date: Any = None,
                          device: str = "cpu") -> torch.Tensor:
        """
        Build state as PyTorch tensor.
        
        Args:
            portfolio_state: Portfolio state dictionary
            market_data: Market data dictionary
            current_date: Current date
            device: Device to place tensor on
        
        Returns:
            State tensor of shape (total_dim,)
        """
        state_array = self.build_full_state(portfolio_state, market_data, current_date)
        return torch.from_numpy(state_array).float().to(device)
    
    def get_state_dim(self) -> int:
        """Get total state dimension."""
        return self.total_dim
