"""
Evaluation metrics: CR, AR, Sharpe Ratio, Annualized Volatility, Max Drawdown.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_returns(portfolio_values: List[float]) -> np.ndarray:
    """Compute returns from portfolio values."""
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    return sharpe


def compute_max_drawdown(portfolio_values: List[float]) -> float:
    """Compute maximum drawdown."""
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    return max_dd


def compute_annualized_volatility(returns: np.ndarray) -> float:
    """Compute annualized volatility."""
    if len(returns) == 0:
        return 0.0
    return np.std(returns) * np.sqrt(252)


def compute_cumulative_return(portfolio_values: List[float]) -> float:
    """Compute cumulative return (CR)."""
    if len(portfolio_values) < 2:
        return 0.0
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    return (final_value - initial_value) / initial_value


def compute_annualized_return(portfolio_values: List[float], n_days: int) -> float:
    """Compute annualized return (AR)."""
    cr = compute_cumulative_return(portfolio_values)
    n_years = n_days / 252.0
    if n_years <= 0:
        return 0.0
    ar = (1 + cr) ** (1 / n_years) - 1
    return ar


def compute_metrics(portfolio_values: List[float],
                   dates: List = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        portfolio_values: List of portfolio values over time
        dates: Optional list of dates (for computing number of days)
        
    Returns:
        Dictionary with metrics:
        - CR: Cumulative Return
        - AR: Annualized Return
        - Sharpe: Sharpe Ratio
        - Volatility: Annualized Volatility
        - MaxDD: Maximum Drawdown
    """
    if len(portfolio_values) < 2:
        return {
            'CR': 0.0,
            'AR': 0.0,
            'Sharpe': 0.0,
            'Volatility': 0.0,
            'MaxDD': 0.0
        }
    
    returns = compute_returns(portfolio_values)
    
    # Cumulative Return
    cr = compute_cumulative_return(portfolio_values)
    
    # Annualized Return
    if dates is not None and len(dates) > 1:
        n_days = (dates[-1] - dates[0]).days if hasattr(dates[0], '__sub__') else len(dates)
    else:
        n_days = len(portfolio_values)
    ar = compute_annualized_return(portfolio_values, n_days)
    
    # Sharpe Ratio
    sharpe = compute_sharpe_ratio(returns)
    
    # Annualized Volatility
    volatility = compute_annualized_volatility(returns)
    
    # Maximum Drawdown
    max_dd = compute_max_drawdown(portfolio_values)
    
    return {
        'CR': cr,
        'AR': ar,
        'Sharpe': sharpe,
        'Volatility': volatility,
        'MaxDD': max_dd
    }


def compute_baseline_metrics(baseline_values: List[float],
                            dates: List = None) -> Dict[str, float]:
    """Compute metrics for baseline strategies."""
    return compute_metrics(baseline_values, dates)

