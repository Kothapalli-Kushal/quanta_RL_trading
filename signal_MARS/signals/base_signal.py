"""
Base signal interface for abstract signal generation.

All signals must inherit from BaseSignal and implement the compute method.
Signals are designed to be swappable and composable.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseSignal(ABC):
    """
    Abstract base class for all signal generators.
    
    Signals transform raw market data into normalized feature vectors
    that can be used by RL agents. All signals must return values in [-1, 1].
    """
    
    def __init__(self, name: str, output_dim: int, window: int = 20):
        """
        Initialize signal.
        
        Args:
            name: Unique identifier for this signal type
            output_dim: Dimension of the output feature vector
            window: Lookback window for signal computation
        """
        self.name = name
        self.output_dim = output_dim
        self.window = window
        self.enabled = True
    
    @abstractmethod
    def compute(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute signal features from raw data.
        
        This method must be implemented by all signal subclasses.
        The implementation should NOT contain concrete signal formulas yet.
        Instead, it should define the interface and return placeholder values.
        
        Args:
            raw_data: Dictionary containing market data. Expected keys:
                - 'prices': (T, N) array of asset prices
                - 'volumes': (T, N) array of trading volumes (optional)
                - 'returns': (T, N) array of returns (optional)
                - 'high': (T, N) array of high prices (optional)
                - 'low': (T, N) array of low prices (optional)
                - 'open': (T, N) array of open prices (optional)
                - 'timestamp': (T,) array of timestamps (optional)
                - Other custom keys as needed
        
        Returns:
            Signal features as numpy array of shape (output_dim,).
            Values must be normalized to [-1, 1].
        """
        pass
    
    def validate_output(self, output: np.ndarray) -> bool:
        """
        Validate that output is in correct format.
        
        Args:
            output: Signal output to validate
        
        Returns:
            True if valid, False otherwise
        """
        if output.shape != (self.output_dim,):
            return False
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            return False
        if np.any(output < -1.0) or np.any(output > 1.0):
            return False
        return True
    
    def clip_output(self, output: np.ndarray) -> np.ndarray:
        """
        Clip output to [-1, 1] range.
        
        Args:
            output: Signal output to clip
        
        Returns:
            Clipped output
        """
        return np.clip(output, -1.0, 1.0)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, dim={self.output_dim}, window={self.window})"


class PlaceholderSignal(BaseSignal):
    """
    Placeholder signal implementation for testing and scaffolding.
    
    Returns random normalized values. Replace with actual signal logic.
    """
    
    def __init__(self, name: str, output_dim: int, window: int = 20):
        """Initialize placeholder signal."""
        super().__init__(name, output_dim, window)
        self._rng = np.random.RandomState(42)
    
    def compute(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute placeholder signal (returns random normalized values).
        
        TODO: Implement actual signal computation logic here.
        This is a placeholder that returns random values in [-1, 1].
        
        Args:
            raw_data: Market data dictionary (not used in placeholder)
        
        Returns:
            Random normalized signal vector
        """
        # Placeholder: return random values
        signal = self._rng.randn(self.output_dim)
        signal = np.tanh(signal)  # Normalize to [-1, 1]
        return signal
