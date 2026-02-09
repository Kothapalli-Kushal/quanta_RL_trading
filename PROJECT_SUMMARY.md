# MARS Paper Replication - Project Summary

## Overview

This is a full paper-level replication of:
**"MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management"** (arXiv:2508.01173)

## Project Structure

```
paper_replication/
├── src/
│   ├── data/              # Data pipeline (yfinance, feature engineering)
│   ├── env/               # Portfolio environment + risk overlay
│   ├── agents/            # DDPG agents with Safety-Critic
│   ├── mac/               # Meta-Adaptive Controller
│   ├── training/          # Two-tier training loop
│   ├── evaluation/        # Metrics and visualization
│   ├── ablation/          # Ablation study implementations
│   └── baselines/         # Baseline strategies
├── configs/               # Configuration files
├── main.py                # Main experiment script
├── run_ablation.py        # Ablation study runner
├── run_all_experiments.py # Run all experiments
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Components Implemented

### ✅ Data Pipeline
- yfinance integration for DJI and HSI constituents
- Feature engineering: Price, MACD, RSI, CCI, ADX (exactly 5 features)
- Train/val/test splits

### ✅ Environment (MDP)
- State: cash + holdings + features
- Action: continuous allocation changes
- Reward: return - transaction cost - risk penalty
- Risk penalty: 0.5 * σ_30d + 2.0 * DD_30d

### ✅ Risk Management
- Max position: 20%
- No short-selling
- Cash buffer: 5%
- Trade feasibility constraints

### ✅ Heterogeneous Agent Ensemble (HAE)
- 10 DDPG agents with varying risk parameters
- Each agent: Actor + Critic + Safety-Critic
- Conditional Safety Penalty (CSP) in actor updates

### ✅ Meta-Adaptive Controller (MAC)
- Neural network for adaptive agent weighting
- Sharpe-like objective: -(E[Q̄] / Std(Q̄) - 0.5 * E[C̄])
- Updates every meta_train_freq episodes

### ✅ Training Loop
- Two-tier: agents update every step, MAC updates periodically
- Separate replay buffers
- Follows Algorithm 1 from paper

### ✅ Evaluation
- Metrics: CR, AR, Sharpe, Volatility, MaxDD
- Visualizations: equity curves, drawdown, MAC weights
- Results tables

### ✅ Ablation Studies
- MARS-Static (uniform weights)
- MARS-Homogeneous (same agent)
- MARS-Div5 (5 agents)
- MARS-Div15 (15 agents)

### ✅ Baselines
- Buy-and-Hold
- Equal Weight

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run MARS experiment:**
   ```bash
   python main.py --config configs/default.yaml --index DJI --experiment MARS
   ```

3. **Run all experiments:**
   ```bash
   python run_all_experiments.py --config configs/default.yaml --index both
   ```

## Key Features

- **Paper-level replication**: No simplifications or shortcuts
- **Full framework**: All components as described in paper
- **Reproducibility**: Fixed seeds, deterministic operations
- **Comprehensive evaluation**: All metrics and visualizations
- **Ablation studies**: All variants implemented
- **Baselines**: Standard comparison strategies

## Notes

- Data is automatically downloaded from yfinance on first run
- Training time depends on hardware (GPU recommended)
- Results are saved in `results/` directory
- All random operations use seed=42 for reproducibility

## Status

✅ All components implemented
✅ Ready for training and evaluation
✅ Documentation complete

