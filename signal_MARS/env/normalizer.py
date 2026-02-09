"""
Rolling normalization utilities for time series data.

Provides stateless normalization that doesn't leak future information.
"""

from typing import Optional, Union
import numpy as np
from collections import deque


class RollingNormalizer:
    """
    Rolling normalization that maintains a window of historical statistics.
    
    Uses rolling mean and std for z-score normalization, or rolling
    min/max for percentile-based normalization.
    """
    
    def __init__(self, window_size: int = 20, method: str = "zscore"):
        """
        Initialize rolling normalizer.
        
        Args:
            window_size: Size of rolling window for statistics
            method: Normalization method ('zscore' or 'percentile')
        """
        self.window_size = window_size
        self.method = method
        
        # Rolling buffers
        self._buffer: deque = deque(maxlen=window_size)
        self._initialized = False
    
    def normalize(self, value: Union[float, np.ndarray], 
                  update: bool = True) -> Union[float, np.ndarray]:
        """
        Normalize a value using rolling statistics.
        
        Args:
            value: Value(s) to normalize
            update: Whether to update internal statistics
        
        Returns:
            Normalized value(s)
        """
        if update:
            self._buffer.append(value)
        
        if not self._initialized and len(self._buffer) < self.window_size:
            # Not enough data yet, return as-is or use simple normalization
            if isinstance(value, np.ndarray):
                return np.zeros_like(value)
            return 0.0
        
        self._initialized = True
        
        if self.method == "zscore":
            return self._zscore_normalize(value)
        elif self.method == "percentile":
            return self._percentile_normalize(value)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _zscore_normalize(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Z-score normalization using rolling mean and std."""
        buffer_array = np.array(list(self._buffer))
        
        if isinstance(value, np.ndarray):
            mean = np.mean(buffer_array, axis=0)
            std = np.std(buffer_array, axis=0)
            std = np.where(std > 1e-8, std, 1.0)  # Avoid division by zero
            return (value - mean) / std
        else:
            mean = np.mean(buffer_array)
            std = np.std(buffer_array)
            if std > 1e-8:
                return (value - mean) / std
            return 0.0
    
    def _percentile_normalize(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percentile-based normalization using rolling min/max."""
        buffer_array = np.array(list(self._buffer))
        
        if isinstance(value, np.ndarray):
            min_val = np.min(buffer_array, axis=0)
            max_val = np.max(buffer_array, axis=0)
            diff = max_val - min_val
            diff = np.where(diff > 1e-8, diff, 1.0)
            # Scale to [-1, 1]
            return 2.0 * (value - min_val) / diff - 1.0
        else:
            min_val = np.min(buffer_array)
            max_val = np.max(buffer_array)
            if max_val > min_val:
                return 2.0 * (value - min_val) / (max_val - min_val) - 1.0
            return 0.0
    
    def reset(self):
        """Reset the normalizer (clear buffer)."""
        self._buffer.clear()
        self._initialized = False
    
    def get_stats(self) -> dict:
        """
        Get current rolling statistics.
        
        Returns:
            Dictionary with mean, std, min, max
        """
        if len(self._buffer) == 0:
            return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0}
        
        buffer_array = np.array(list(self._buffer))
        return {
            "mean": np.mean(buffer_array),
            "std": np.std(buffer_array),
            "min": np.min(buffer_array),
            "max": np.max(buffer_array)
        }


class MultiNormalizer:
    """
    Manages multiple rolling normalizers for different features.
    
    Useful for normalizing different dimensions of state vectors.
    """
    
    def __init__(self, num_features: int, window_size: int = 20, 
                 method: str = "zscore"):
        """
        Initialize multi-normalizer.
        
        Args:
            num_features: Number of features to normalize
            window_size: Rolling window size
            method: Normalization method
        """
        self.num_features = num_features
        self.normalizers = [
            RollingNormalizer(window_size, method) 
            for _ in range(num_features)
        ]
    
    def normalize(self, values: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize feature vector.
        
        Args:
            values: Array of shape (num_features,) or (T, num_features)
            update: Whether to update statistics
        
        Returns:
            Normalized array of same shape
        """
        if values.ndim == 1:
            # Single timestep
            normalized = np.array([
                norm.normalize(val, update=update) 
                for norm, val in zip(self.normalizers, values)
            ])
            return normalized
        else:
            # Multiple timesteps
            normalized = np.array([
                [norm.normalize(val, update=update) 
                 for norm, val in zip(self.normalizers, row)]
                for row in values
            ])
            return normalized
    
    def reset(self):
        """Reset all normalizers."""
        for norm in self.normalizers:
            norm.reset()
