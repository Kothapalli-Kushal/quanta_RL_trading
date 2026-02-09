"""
Signal factory for creating and managing signal ensembles.

Combines multiple signals into a unified feature tensor.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch

from .signal_registry import SignalRegistry
from .base_signal import BaseSignal


class SignalFactory:
    """
    Factory for creating and combining multiple signals.
    
    Manages a collection of signal generators and combines their outputs
    into a unified feature tensor suitable for RL agents.
    """
    
    def __init__(self, signal_names: Optional[List[str]] = None, 
                 enabled_signals: Optional[List[str]] = None):
        """
        Initialize signal factory.
        
        Args:
            signal_names: List of signal names to instantiate.
                         If None, uses all registered signals.
            enabled_signals: List of signal names to enable.
                           If None, all signals are enabled.
        """
        if signal_names is None:
            signal_names = SignalRegistry.list_signals()
        
        self.signals: Dict[str, BaseSignal] = {}
        self.signal_order: List[str] = []
        
        # Create signal instances
        for name in signal_names:
            try:
                signal = SignalRegistry.create(name)
                self.signals[name] = signal
                self.signal_order.append(name)
            except ValueError as e:
                print(f"Warning: Could not create signal '{name}': {e}")
        
        # Enable/disable signals
        if enabled_signals is not None:
            for name in self.signals:
                self.signals[name].enabled = name in enabled_signals
        
        # Compute total output dimension
        self.total_dim = sum(s.output_dim for s in self.signals.values() if s.enabled)
    
    def compute(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute all enabled signals and concatenate outputs.
        
        Args:
            raw_data: Market data dictionary (see BaseSignal.compute)
        
        Returns:
            Concatenated signal features of shape (total_dim,)
        """
        signal_outputs = []
        
        for name in self.signal_order:
            signal = self.signals[name]
            if signal.enabled:
                try:
                    output = signal.compute(raw_data)
                    
                    # Validate output
                    if not signal.validate_output(output):
                        output = signal.clip_output(output)
                        if not signal.validate_output(output):
                            print(f"Warning: Signal '{name}' produced invalid output, using zeros")
                            output = np.zeros(signal.output_dim)
                    
                    signal_outputs.append(output)
                except Exception as e:
                    print(f"Error computing signal '{name}': {e}")
                    # Use zeros as fallback
                    signal_outputs.append(np.zeros(signal.output_dim))
        
        if len(signal_outputs) == 0:
            return np.zeros(self.total_dim)
        
        return np.concatenate(signal_outputs)
    
    def compute_tensor(self, raw_data: Dict[str, Any], 
                      device: str = "cpu") -> torch.Tensor:
        """
        Compute signals and return as PyTorch tensor.
        
        Args:
            raw_data: Market data dictionary
            device: Device to place tensor on
        
        Returns:
            Signal tensor of shape (total_dim,)
        """
        signal_array = self.compute(raw_data)
        return torch.from_numpy(signal_array).float().to(device)
    
    def get_signal_dim(self) -> int:
        """
        Get total dimension of all enabled signals.
        
        Returns:
            Total signal dimension
        """
        return self.total_dim
    
    def get_signal_info(self) -> Dict[str, Dict]:
        """
        Get information about all signals.
        
        Returns:
            Dictionary mapping signal names to their info
        """
        info = {}
        for name, signal in self.signals.items():
            info[name] = {
                "enabled": signal.enabled,
                "output_dim": signal.output_dim,
                "window": signal.window
            }
        return info
    
    def enable_signal(self, name: str):
        """Enable a signal by name."""
        if name in self.signals:
            self.signals[name].enabled = True
            self._recompute_dim()
        else:
            raise ValueError(f"Signal '{name}' not found")
    
    def disable_signal(self, name: str):
        """Disable a signal by name."""
        if name in self.signals:
            self.signals[name].enabled = False
            self._recompute_dim()
        else:
            raise ValueError(f"Signal '{name}' not found")
    
    def _recompute_dim(self):
        """Recompute total dimension after enabling/disabling signals."""
        self.total_dim = sum(s.output_dim for s in self.signals.values() if s.enabled)
