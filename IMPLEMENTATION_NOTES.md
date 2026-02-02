# MARS Implementation Notes

## Architecture Overview

### 1. Data Pipeline
- **DataLoader**: Fetches DJI and HSI constituent stocks using yfinance
- **FeatureEngineer**: Computes exactly 5 features per asset:
  - Price (normalized close)
  - MACD
  - RSI
  - CCI
  - ADX

### 2. Environment (MDP)
- **State**: [cash_norm, holdings_norm (n_assets), features (n_assets * 5)]
- **Action**: Continuous vector in [-1, 1]^D (target allocation change)
- **Reward**: R_t = (V_{t+1} - V_t) / V_t - C_t - ρ_t
  - ρ_t = 0.5 * σ_30d + 2.0 * DD_30d

### 3. Risk Management Overlay
- Max position concentration: 20%
- No short-selling
- Maintain 5% cash buffer
- Trades clipped to feasibility

### 4. Heterogeneous Agent Ensemble (HAE)
- N = 10 DDPG agents
- Each agent has:
  - Actor network
  - Critic network (Q-function)
  - Safety-Critic network (C-function)
- Risk parameters vary:
  - Risk thresholds: 0.2 (conservative) to 0.8 (aggressive)
  - Risk penalties: 2.0 (conservative) to 0.5 (aggressive)

### 5. Conditional Safety Penalty (CSP)
- Actor update: ∇J = ∇Q - λ_i * ∇ReLU(C_ξ - θ_i)
- Penalty only activates when C > θ

### 6. Meta-Adaptive Controller (MAC)
- Neural network: state → agent logits → softmax → weights
- Objective: L(ω) = -(E[Q̄_t] / (Std(Q̄_t) + ε) - 0.5 * E[C̄_t])
- Updates every meta_train_freq episodes

### 7. Training Loop
- Two-tier training:
  - Agents update every step (DDPG + Safety-Critic)
  - MAC updates every meta_train_freq episodes
- Separate replay buffers for agents and MAC

## Key Implementation Details

### Safety-Critic Training
The Safety-Critic learns to predict environment risk score C_env(s, a) which combines:
- Portfolio concentration (HHI)
- Max position concentration
- Leverage
- Trade impact magnitude

### MAC Training
MAC uses its own replay buffer storing:
- State
- Predicted weights
- Weighted Q-values (Q̄)
- Weighted safety scores (C̄)
- Reward

### Evaluation Metrics
- CR: Cumulative Return
- AR: Annualized Return
- Sharpe: Sharpe Ratio
- Volatility: Annualized Volatility
- MaxDD: Maximum Drawdown

## Ablation Studies

1. **MARS-Static**: Uniform weights (no adaptation)
2. **MARS-Homogeneous**: All agents have same risk parameters
3. **MARS-Div5**: 5 agents instead of 10
4. **MARS-Div15**: 15 agents instead of 10

## Baselines

1. **Buy-and-Hold**: Equal allocation at start, hold
2. **Equal Weight**: Rebalance to equal weights periodically

## Reproducibility

- Fixed seed: 42
- Deterministic PyTorch operations
- All random operations seeded

## Time Splits

### Test 2022
- Train: 2016-2020
- Val: 2021
- Test: 2022

### Test 2024
- Train: 2018-2022
- Val: 2023
- Test: 2024

(Currently configured for Test 2022 in default.yaml)

