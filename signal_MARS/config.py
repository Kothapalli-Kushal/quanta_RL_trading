"""
Central configuration for Signal-MARS framework.

This module defines all hyperparameters, architecture specs, and system settings
in a single location to ensure reproducibility and easy experimentation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # Signal dimensions (placeholder - will be set based on registered signals)
    signal_dim: int = 64  # Total dimension across all signals
    num_signals: int = 4  # Number of different signal types
    
    # Signal-specific settings
    enabled_signals: List[str] = None  # None means all enabled
    signal_window: int = 20  # Lookback window for signal computation
    
    # Normalization
    signal_normalize: bool = True
    signal_clip_range: tuple = (-1.0, 1.0)


@dataclass
class MacroConfig:
    """Configuration for macro features."""
    use_macro: bool = True
    macro_dim: int = 10  # Dimension of macro feature vector
    
    # Macro feature settings
    enabled_macros: List[str] = None  # None means all enabled
    macro_freq: str = "daily"  # Frequency of macro updates
    
    # Forward fill settings
    forward_fill_days: int = 5  # Days to forward fill missing values


@dataclass
class AgentConfig:
    """Configuration for individual RL agents."""
    num_agents: int = 5
    agent_types: List[str] = None  # e.g., ["trend", "meanrev", "vol", "defensive", "aggressive"]
    
    # Network architecture
    actor_hidden_dims: List[int] = None  # Default: [256, 128]
    critic_hidden_dims: List[int] = None  # Default: [256, 128, 64]
    safety_critic_hidden_dims: List[int] = None  # Default: [128, 64]
    
    # DDPG hyperparameters
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    safety_critic_lr: float = 5e-4
    
    # Feature masking (for signal specialization)
    use_feature_masking: bool = True
    mask_probability: float = 0.3  # Probability of masking a feature
    
    # Exploration
    noise_std: float = 0.1
    noise_decay: float = 0.995
    
    def __post_init__(self):
        """Set default values if None."""
        if self.agent_types is None:
            self.agent_types = ["trend", "meanrev", "vol", "defensive", "aggressive"][:self.num_agents]
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 128, 64]
        if self.safety_critic_hidden_dims is None:
            self.safety_critic_hidden_dims = [128, 64]


@dataclass
class MACConfig:
    """Configuration for Meta-Adaptive Controller."""
    use_mac: bool = True
    
    # Network architecture
    mac_hidden_dims: List[int] = None  # Default: [128, 64]
    mac_lr: float = 1e-4
    
    # Input dimensions (will be set based on state builder)
    mac_input_dim: int = 50  # Macro + summarized signals
    
    # Update frequency
    mac_update_freq: int = 10  # Update MAC every N steps
    
    # Loss weights
    q_weight: float = 1.0
    risk_weight: float = 0.5
    sharpe_weight: float = 0.3
    
    def __post_init__(self):
        """Set default values if None."""
        if self.mac_hidden_dims is None:
            self.mac_hidden_dims = [128, 64]


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Risk thresholds per agent (theta_i, lambda_i)
    risk_thresholds: Dict[str, float] = None  # Agent type -> threshold
    risk_lambdas: Dict[str, float] = None  # Agent type -> lambda (risk aversion)
    
    # Hard constraints
    max_position_size: float = 0.15  # Max 15% per asset
    max_leverage: float = 1.0  # No leverage
    cash_buffer: float = 0.05  # Keep 5% cash buffer
    allow_shorting: bool = False
    
    # Risk metrics
    lookback_window: int = 20  # Window for volatility/risk calculations
    concentration_limit: float = 0.4  # Max portfolio concentration
    
    def __post_init__(self):
        """Set default values if None."""
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                "trend": 0.3,
                "meanrev": 0.25,
                "vol": 0.35,
                "defensive": 0.15,
                "aggressive": 0.4
            }
        if self.risk_lambdas is None:
            self.risk_lambdas = {
                "trend": 0.5,
                "meanrev": 0.6,
                "vol": 0.4,
                "defensive": 0.8,
                "aggressive": 0.3
            }


@dataclass
class EnvConfig:
    """Configuration for portfolio environment."""
    num_assets: int = 10
    initial_cash: float = 1000000.0
    transaction_cost: float = 0.001  # 0.1% transaction cost
    
    # State representation
    use_signals: bool = True
    use_macro: bool = True
    include_portfolio_state: bool = True  # Include current positions, cash, etc.
    
    # State dimensions (will be computed)
    state_dim: int = None  # Total state dimension
    
    # Episode settings
    max_steps: int = 252  # Trading days in a year
    reset_on_terminal: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training loop
    num_episodes: int = 1000
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Update frequencies
    agent_update_freq: int = 1  # Update agents every step
    target_update_freq: int = 10  # Update target networks every N steps
    target_update_tau: float = 0.005  # Soft update coefficient
    
    # Logging
    log_freq: int = 10
    save_freq: int = 100
    eval_freq: int = 50
    
    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seeding
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    num_episodes: int = 10
    render: bool = False
    save_results: bool = True
    results_dir: str = "./results"


@dataclass
class SignalMARSConfig:
    """Main configuration class combining all sub-configs."""
    signal: SignalConfig = None
    macro: MacroConfig = None
    agent: AgentConfig = None
    mac: MACConfig = None
    risk: RiskConfig = None
    env: EnvConfig = None
    training: TrainingConfig = None
    eval: EvalConfig = None
    
    def __post_init__(self):
        """Initialize all sub-configs with defaults if None."""
        if self.signal is None:
            self.signal = SignalConfig()
        if self.macro is None:
            self.macro = MacroConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.mac is None:
            self.mac = MACConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.env is None:
            self.env = EnvConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.eval is None:
            self.eval = EvalConfig()
        
        # Compute derived dimensions
        self._compute_dimensions()
    
    def _compute_dimensions(self):
        """Compute state dimensions based on configuration."""
        state_dim = 0
        
        # Portfolio state (positions, cash, returns, etc.)
        if self.env.include_portfolio_state:
            state_dim += self.env.num_assets + 1  # positions + cash
        
        # Signal features
        if self.env.use_signals:
            state_dim += self.signal.signal_dim
        
        # Macro features
        if self.env.use_macro:
            state_dim += self.macro.macro_dim
        
        self.env.state_dim = state_dim
        
        # MAC input dimension (macro + summarized signals)
        if self.mac.use_mac:
            mac_input = 0
            if self.env.use_macro:
                mac_input += self.macro.macro_dim
            if self.env.use_signals:
                # Summarized signals (mean, std, etc.)
                mac_input += 10  # Placeholder for signal summaries
            self.mac.mac_input_dim = mac_input


def get_default_config() -> SignalMARSConfig:
    """Get default configuration."""
    return SignalMARSConfig()
