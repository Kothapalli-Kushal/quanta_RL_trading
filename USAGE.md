# MARS Paper Replication - Usage Guide

## Setup

1. Install dependencies:
```bash
cd paper_replication
pip install -r requirements.txt
```

2. Set up the project structure (already created):
```
paper_replication/
├── data/
│   ├── raw/          # Raw OHLCV data (will be downloaded)
│   └── processed/    # Computed features
├── src/              # Source code
├── configs/          # Configuration files
├── results/           # Output results
└── logs/             # Training logs
```

## Running Experiments

### 1. Run Full MARS Experiment

For a single index:
```bash
python main.py --config configs/default.yaml --index DJI --experiment MARS
python main.py --config configs/default.yaml --index HSI --experiment MARS
```

### 2. Run Ablation Studies

```bash
# MARS-Static (uniform weights)
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Static

# MARS-Homogeneous (same agent)
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Homogeneous

# MARS-Div5 (5 agents)
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Div5

# MARS-Div15 (15 agents)
python run_ablation.py --config configs/default.yaml --index DJI --ablation MARS-Div15
```

### 3. Run All Experiments

Run MARS, all ablations, and baselines for both indices:
```bash
python run_all_experiments.py --config configs/default.yaml --index both
```

Or for a single index:
```bash
python run_all_experiments.py --config configs/default.yaml --index DJI
```

## Configuration

Edit `configs/default.yaml` to adjust:
- Data time splits (train/val/test)
- Environment parameters (initial cash, transaction costs)
- Risk management constraints
- Training hyperparameters (episodes, batch size, learning rates)
- Number of agents

## Output

Results are saved in `results/{index}/{experiment}/`:
- `equity_curve.png`: Portfolio value over time
- `drawdown.png`: Drawdown curve
- `mac_weights.png`: MAC weight time series (for MARS)
- `metrics.csv`: Performance metrics (CR, AR, Sharpe, Volatility, MaxDD)

Comparison tables are saved in `results/{index}/comparison_table.csv`

## Reproducibility

The code uses fixed seed (42) for reproducibility. All random operations (numpy, torch, python random) are seeded.

## Notes

- Data will be automatically downloaded from yfinance on first run
- Training may take several hours depending on hardware
- GPU is recommended but not required (set `device: "cpu"` in config if no GPU)

