"""
Macro feature registry for managing macro data sources.

Similar pattern to signal registry, but for macro features.
"""

from typing import Dict, Type, List, Optional, Callable
from .macro_loader import MacroLoader


class MacroRegistry:
    """
    Registry for macro feature loaders.
    
    Maintains a mapping from macro feature names to loader functions
    or data sources.
    """
    
    _registry: Dict[str, Callable] = {}
    _default_configs: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, name: str, loader_func: Callable, 
                 default_config: Optional[Dict] = None):
        """
        Register a macro feature loader.
        
        Args:
            name: Unique macro feature identifier
            loader_func: Function that returns macro feature array
            default_config: Default configuration dict
        """
        cls._registry[name] = loader_func
        if default_config is None:
            default_config = {}
        cls._default_configs[name] = default_config
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """
        Get macro loader function by name.
        
        Args:
            name: Macro feature identifier
        
        Returns:
            Loader function or None if not found
        """
        return cls._registry.get(name)
    
    @classmethod
    def list_macros(cls) -> List[str]:
        """
        List all registered macro feature names.
        
        Returns:
            List of macro feature names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if macro feature is registered.
        
        Args:
            name: Macro feature identifier
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry
    
    @classmethod
    def clear(cls):
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._default_configs.clear()


def register_default_macros():
    """Register default macro features (placeholder)."""
    def placeholder_macro_loader():
        """Placeholder macro loader."""
        # TODO: Implement actual macro data loading
        return np.zeros(10)  # Placeholder
    
    MacroRegistry.register("default", placeholder_macro_loader)


# Auto-register on import
import numpy as np
register_default_macros()
