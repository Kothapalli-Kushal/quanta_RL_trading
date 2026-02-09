"""
MAC (Meta-Adaptive Controller) module for Signal-MARS.

Provides neural network controller for agent weight allocation.
"""

from .mac_controller import MACController, MACTrainer
from .mac_buffer import MACBuffer

__all__ = [
    "MACController",
    "MACTrainer",
    "MACBuffer"
]
