"""
Visualization: Equity curves, drawdown plots, MAC weight time series.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os


def plot_equity_curve(portfolio_values: List[float],
                     dates: Optional[List] = None,
                     baseline_values: Optional[List[float]] = None,
                     save_path: Optional[str] = None,
                     title: str = "Equity Curve"):
    """Plot equity curve."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None:
        x = dates
    else:
        x = range(len(portfolio_values))
    
    ax.plot(x, portfolio_values, label='Portfolio', linewidth=2)
    
    if baseline_values is not None:
        ax.plot(x, baseline_values, label='Baseline', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Portfolio Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_drawdown(portfolio_values: List[float],
                 dates: Optional[List] = None,
                 save_path: Optional[str] = None,
                 title: str = "Drawdown"):
    """Plot drawdown curve."""
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak * 100  # Percentage
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if dates is not None:
        x = dates
    else:
        x = range(len(drawdown))
    
    ax.fill_between(x, drawdown, 0, alpha=0.3, color='red')
    ax.plot(x, drawdown, color='red', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mac_weights(mac_weights_history: List[np.ndarray],
                    dates: Optional[List] = None,
                    n_agents: int = 10,
                    save_path: Optional[str] = None,
                    title: str = "MAC Weights Over Time"):
    """Plot MAC weight time series."""
    if not mac_weights_history:
        return
    
    weights_array = np.array(mac_weights_history)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if dates is not None:
        x = dates[:len(mac_weights_history)]
    else:
        x = range(len(mac_weights_history))
    
    # Plot weights for each agent
    for i in range(min(n_agents, weights_array.shape[1])):
        ax.plot(x, weights_array[:, i], label=f'Agent {i}', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison(results_dict: Dict[str, Dict],
                   save_path: Optional[str] = None,
                   title: str = "Method Comparison"):
    """Plot comparison of multiple methods."""
    methods = list(results_dict.keys())
    metrics = ['CR', 'AR', 'Sharpe', 'Volatility', 'MaxDD']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[method].get(metric, 0) for method in methods]
        axes[i].bar(methods, values)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

