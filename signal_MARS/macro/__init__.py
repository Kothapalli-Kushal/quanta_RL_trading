"""
Macro feature module for Signal-MARS.

Provides interfaces for loading and managing macroeconomic features.
"""

from .macro_loader import MacroLoader
from .macro_registry import MacroRegistry

__all__ = [
    "MacroLoader",
    "MacroRegistry"
]
