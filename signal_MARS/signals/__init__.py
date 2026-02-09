"""
Signal generation module for Signal-MARS.

Provides abstract interfaces for signal computation and management.
"""

from .base_signal import BaseSignal, PlaceholderSignal
from .signal_registry import SignalRegistry
from .signal_factory import SignalFactory

__all__ = [
    "BaseSignal",
    "PlaceholderSignal",
    "SignalRegistry",
    "SignalFactory"
]
