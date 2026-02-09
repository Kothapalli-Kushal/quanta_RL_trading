"""
Mathematical utilities for Signal-MARS.
"""

import numpy as np
import torch
from typing import Union


def softmax(x: Union[np.ndarray, torch.Tensor], axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute softmax.
    
    Args:
        x: Input array/tensor
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax output
    """
    if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=axis)
    else:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def normalize(x: Union[np.ndarray, torch.Tensor], 
              method: str = "zscore") -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize array/tensor.
    
    Args:
        x: Input array/tensor
        method: Normalization method ('zscore' or 'minmax')
    
    Returns:
        Normalized array/tensor
    """
    if isinstance(x, torch.Tensor):
        if method == "zscore":
            mean = x.mean()
            std = x.std()
            if std > 0:
                return (x - mean) / std
            return x
        elif method == "minmax":
            min_val = x.min()
            max_val = x.max()
            if max_val > min_val:
                return 2.0 * (x - min_val) / (max_val - min_val) - 1.0
            return x
    else:
        if method == "zscore":
            mean = np.mean(x)
            std = np.std(x)
            if std > 0:
                return (x - mean) / std
            return x
        elif method == "minmax":
            min_val = np.min(x)
            max_val = np.max(x)
            if max_val > min_val:
                return 2.0 * (x - min_val) / (max_val - min_val) - 1.0
            return x
    
    raise ValueError(f"Unknown normalization method: {method}")


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return > 0:
        return np.sqrt(252) * mean_return / std_return
    return 0.0
