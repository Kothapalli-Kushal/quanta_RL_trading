"""
Signal-MARS: Signal-driven Meta-Adaptive Reinforcement Learning for Portfolio Management

A research-grade reinforcement learning framework for multi-agent portfolio management
with signal-based state representations and meta-adaptive control.
"""

__version__ = "0.1.0"

from .config import SignalMARSConfig, get_default_config

__all__ = [
    "SignalMARSConfig",
    "get_default_config"
]
