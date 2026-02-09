"""
Risk metrics computation for portfolio risk assessment.

Computes various risk measures: concentration, leverage, volatility, etc.
"""

from typing import Dict, Any
import numpy as np


class RiskMetrics:
    """
    Computes various portfolio risk metrics.
    """
    
    @staticmethod
    def compute_concentration(weights: np.ndarray) -> float:
        """
        Compute portfolio concentration (Herfindahl index).
        
        Args:
            weights: Portfolio weights (num_assets,)
        
        Returns:
            Concentration score in [0, 1] (higher = more concentrated)
        """
        return np.sum(weights ** 2)
    
    @staticmethod
    def compute_leverage(positions: np.ndarray, prices: np.ndarray, 
                        cash: float, portfolio_value: float) -> float:
        """
        Compute portfolio leverage.
        
        Args:
            positions: Number of shares per asset (num_assets,)
            prices: Asset prices (num_assets,)
            cash: Cash balance
            portfolio_value: Total portfolio value
        
        Returns:
            Leverage ratio (1.0 = no leverage, >1.0 = leveraged)
        """
        if portfolio_value == 0:
            return 1.0
        
        gross_exposure = np.sum(np.abs(positions * prices))
        return gross_exposure / portfolio_value
    
    @staticmethod
    def compute_volatility_proxy(returns: np.ndarray, window: int = 20) -> float:
        """
        Compute volatility proxy from returns.
        
        Args:
            returns: Array of portfolio returns
            window: Lookback window
        
        Returns:
            Volatility estimate (standard deviation)
        """
        if len(returns) < 2:
            return 0.0
        
        recent_returns = returns[-window:] if len(returns) > window else returns
        return np.std(recent_returns)
    
    @staticmethod
    def compute_signal_disagreement(signal_vectors: np.ndarray) -> float:
        """
        Compute disagreement between signals.
        
        Placeholder for signal disagreement metric.
        
        Args:
            signal_vectors: Array of signal vectors (num_signals, signal_dim)
        
        Returns:
            Disagreement score in [0, 1] (higher = more disagreement)
        
        TODO: Implement actual signal disagreement computation.
        """
        if signal_vectors.shape[0] < 2:
            return 0.0
        
        # Placeholder: compute variance across signals
        signal_variance = np.var(signal_vectors, axis=0)
        disagreement = np.mean(signal_variance)
        
        # Normalize to [0, 1]
        return np.clip(disagreement, 0.0, 1.0)
    
    @staticmethod
    def compute_max_drawdown(returns: np.ndarray) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            returns: Array of portfolio returns
        
        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    @staticmethod
    def compute_value_at_risk(returns: np.ndarray, confidence: float = 0.05) -> float:
        """
        Compute Value at Risk (VaR).
        
        Args:
            returns: Array of portfolio returns
            confidence: Confidence level (e.g., 0.05 for 95% VaR)
        
        Returns:
            VaR (negative value)
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def compute_all_metrics(weights: np.ndarray, positions: np.ndarray,
                           prices: np.ndarray, cash: float,
                           portfolio_value: float, returns: np.ndarray) -> Dict[str, float]:
        """
        Compute all risk metrics.
        
        Args:
            weights: Portfolio weights
            positions: Number of shares
            prices: Asset prices
            cash: Cash balance
            portfolio_value: Total portfolio value
            returns: Portfolio returns
        
        Returns:
            Dictionary of all risk metrics
        """
        return {
            "concentration": RiskMetrics.compute_concentration(weights),
            "leverage": RiskMetrics.compute_leverage(positions, prices, cash, portfolio_value),
            "volatility": RiskMetrics.compute_volatility_proxy(returns),
            "max_drawdown": RiskMetrics.compute_max_drawdown(returns),
            "var_05": RiskMetrics.compute_value_at_risk(returns, 0.05)
        }
