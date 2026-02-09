"""
Logging utilities for Signal-MARS.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any
import json


def setup_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
    
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], step: int, prefix: str = ""):
    """
    Log metrics to logger.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        step: Current step/episode
        prefix: Prefix for log message
    """
    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                            for k, v in metrics.items()])
    logger.info(f"{prefix}Step {step}: {metrics_str}")


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
