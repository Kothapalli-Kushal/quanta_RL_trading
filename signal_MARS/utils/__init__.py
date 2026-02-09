"""
Utilities module for Signal-MARS.

Provides logging, seeding, and mathematical utilities.
"""

from .logging import setup_logger, log_metrics, save_metrics
from .seeding import set_seed
from .math_utils import softmax, normalize, sharpe_ratio

__all__ = [
    "setup_logger",
    "log_metrics",
    "save_metrics",
    "set_seed",
    "softmax",
    "normalize",
    "sharpe_ratio"
]
