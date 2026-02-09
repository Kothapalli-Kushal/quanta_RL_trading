# Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup (optional):**
   ```bash
   python verify_setup.py
   ```

## Running Experiments

### Basic Usage

**Run MARS for DJI:**
```bash
python main.py --config configs/default.yaml --index DJI --experiment MARS
```

**Run MARS for HSI:**
```bash
python main.py --config configs/default.yaml --index HSI --experiment MARS
```

**Run all experiments (MARS + ablations + baselines):**
```bash
python run_all_experiments.py --config configs/default.yaml --index both
```

### Ablation Studies

```bash
# Static weights
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Static

# Homogeneous agents
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Homogeneous

# 5 agents
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Div5

# 15 agents
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Div15
```

## Output

Results are saved in:
- `results/{index}/{experiment}/`
  - `equity_curve.png`
  - `drawdown.png`
  - `mac_weights.png` (for MARS)
  - `metrics.csv`
- `results/{index}/comparison_table.csv` (when running all experiments)

## Configuration

Edit `configs/default.yaml` to adjust:
- Time periods
- Hyperparameters
- Number of agents
- Training settings

## Notes

- First run will download data from yfinance (may take time)
- Training takes several hours (GPU recommended)
- All results use fixed seed=42 for reproducibility

