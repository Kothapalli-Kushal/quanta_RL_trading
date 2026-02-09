"""
Signal registry for managing and discovering signal implementations.

Provides a centralized way to register, enable, and disable signals.
"""

from typing import Dict, Type, List, Optional
from .base_signal import BaseSignal, PlaceholderSignal


class SignalRegistry:
    """
    Registry for signal classes.
    
    Maintains a mapping from signal names to signal classes,
    allowing dynamic signal discovery and instantiation.
    """
    
    _registry: Dict[str, Type[BaseSignal]] = {}
    _default_configs: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, name: str, signal_class: Type[BaseSignal], 
                 default_config: Optional[Dict] = None):
        """
        Register a signal class.
        
        Args:
            name: Unique signal identifier
            signal_class: Signal class (must inherit from BaseSignal)
            default_config: Default configuration dict for this signal
        """
        if not issubclass(signal_class, BaseSignal):
            raise ValueError(f"Signal class must inherit from BaseSignal, got {signal_class}")
        
        cls._registry[name] = signal_class
        if default_config is None:
            default_config = {}
        cls._default_configs[name] = default_config
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseSignal]]:
        """
        Get signal class by name.
        
        Args:
            name: Signal identifier
        
        Returns:
            Signal class or None if not found
        """
        return cls._registry.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseSignal:
        """
        Create signal instance by name.
        
        Args:
            name: Signal identifier
            **kwargs: Additional arguments to pass to signal constructor
        
        Returns:
            Signal instance
        """
        signal_class = cls.get(name)
        if signal_class is None:
            raise ValueError(f"Signal '{name}' not found in registry. "
                           f"Available signals: {list(cls._registry.keys())}")
        
        # Merge default config with provided kwargs
        config = cls._default_configs.get(name, {}).copy()
        config.update(kwargs)
        
        return signal_class(**config)
    
    @classmethod
    def list_signals(cls) -> List[str]:
        """
        List all registered signal names.
        
        Returns:
            List of signal names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if signal is registered.
        
        Args:
            name: Signal identifier
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls):
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._default_configs.clear()


# Register placeholder signals by default
def register_default_signals():
    """Register default placeholder signals."""
    SignalRegistry.register("trend", PlaceholderSignal, 
                           {"name": "trend", "output_dim": 16, "window": 20})
    SignalRegistry.register("meanrev", PlaceholderSignal,
                           {"name": "meanrev", "output_dim": 16, "window": 20})
    SignalRegistry.register("volatility", PlaceholderSignal,
                           {"name": "volatility", "output_dim": 16, "window": 20})
    SignalRegistry.register("momentum", PlaceholderSignal,
                           {"name": "momentum", "output_dim": 16, "window": 20})


# Auto-register on import
register_default_signals()
