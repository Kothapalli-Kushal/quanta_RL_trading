# MARS: Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management

Full paper replication of arXiv:2508.01173

## Project Structure

```
paper_replication/
├── data/
│   ├── raw/          # Raw OHLCV data
│   └── processed/    # Computed features
├── src/
│   ├── data/         # Data pipeline
│   ├── env/          # Portfolio environment
│   ├── agents/       # DDPG agents with Safety-Critic
│   ├── mac/          # Meta-Adaptive Controller
│   ├── training/     # Training loops
│   └── evaluation/   # Metrics and visualization
├── configs/          # Configuration files
├── results/          # Output plots and tables
└── logs/             # Training logs
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config configs/default.yaml
```

