"""
Main script for MARS paper replication.
"""
import argparse
import yaml
import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from src.data import DataLoader, FeatureEngineer
from src.env import PortfolioEnv, RiskOverlay
from src.agents import HeterogeneousAgentEnsemble
from src.mac import MetaAdaptiveController
from src.training import Trainer
from src.evaluation import compute_metrics, plot_equity_curve, plot_drawdown, plot_mac_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(index_name: str, config: dict):
    """Prepare data for training/evaluation."""
    logger.info(f"Preparing data for {index_name}")
    
    data_loader = DataLoader(data_dir=config['data']['raw_dir'])
    feature_engineer = FeatureEngineer()
    
    # Prepare datasets
    datasets = data_loader.prepare_datasets(
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end']
    )
    
    if index_name not in datasets:
        logger.error(f"No data found for {index_name}")
        return None
    
    # Process features
    processed_datasets = {}
    for split in ['train', 'val', 'test']:
        raw_data = datasets[index_name][split]
        features = feature_engineer.process_all_assets(raw_data)
        processed_datasets[split] = features
    
    return processed_datasets


def create_environment(features_dict: dict, config: dict):
    """Create portfolio environment."""
    env = PortfolioEnv(
        features_dict=features_dict,
        initial_cash=config['env']['initial_cash'],
        transaction_cost=config['env']['transaction_cost'],
        max_trade_size=config['env']['max_trade_size'],
        risk_window=config['env']['risk_window']
    )
    return env


def create_models(env: PortfolioEnv, config: dict, device: str):
    """Create HAE and MAC models."""
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    n_agents = config['hae']['n_agents']
    
    # Create HAE
    hae = HeterogeneousAgentEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        device=device
    )
    
    # Create MAC
    mac = MetaAdaptiveController(
        state_dim=state_dim,
        n_agents=n_agents,
        lr=config['mac']['lr'],
        epsilon=config['mac']['epsilon'],
        device=device
    )
    
    # Create risk overlay
    risk_overlay = RiskOverlay(
        max_position_concentration=config['risk']['max_position_concentration'],
        min_cash_buffer=config['risk']['min_cash_buffer'],
        max_leverage=config['risk']['max_leverage']
    )
    
    return hae, mac, risk_overlay


def train_mars(env: PortfolioEnv, hae: HeterogeneousAgentEnsemble, mac: MetaAdaptiveController,
               risk_overlay: RiskOverlay, config: dict, save_dir: str):
    """Train MARS model."""
    logger.info("Starting MARS training")
    
    trainer = Trainer(
        env=env,
        hae=hae,
        mac=mac,
        risk_overlay=risk_overlay,
        meta_train_freq=config['training']['meta_train_freq'],
        batch_size=config['training']['batch_size'],
        device=config['training']['device']
    )
    
    training_stats = trainer.train(
        n_episodes=config['training']['n_episodes'],
        start_idx=0,
        log_dir=save_dir
    )
    
    trainer.save_models(f"{save_dir}/models")
    
    return trainer, training_stats


def evaluate_model(env: PortfolioEnv, hae: HeterogeneousAgentEnsemble, mac: MetaAdaptiveController,
                   risk_overlay: RiskOverlay, config: dict, split: str = 'test'):
    """Evaluate trained model."""
    logger.info(f"Evaluating on {split} set")
    
    state = env.reset(start_idx=0)
    done = False
    portfolio_values = [env.initial_cash]
    dates = []
    
    while not done:
        # Get actions from all agents (no noise during evaluation)
        agent_actions = hae.select_actions(state, add_noise=False)
        
        # Aggregate actions using MAC
        aggregated_action = mac.aggregate_actions(state, agent_actions)
        
        # Apply risk overlay
        current_holdings = env.holdings
        current_prices = env.prices
        cash = env.cash
        portfolio_value = env._compute_portfolio_value()
        
        current_allocation = (current_holdings * current_prices) / portfolio_value if portfolio_value > 0 else np.zeros(env.n_assets)
        constrained_allocation, new_cash = risk_overlay.apply(
            aggregated_action, current_holdings, current_prices, cash, portfolio_value
        )
        
        action_for_env = constrained_allocation - current_allocation
        action_for_env = np.clip(action_for_env / env.max_trade_size, -1.0, 1.0)
        
        next_state, reward, done, info = env.step(action_for_env)
        
        portfolio_values.append(info['portfolio_value'])
        if hasattr(env, 'dates') and env.current_step < len(env.dates):
            dates.append(env.dates[env.current_step])
        
        state = next_state
    
    # Compute metrics
    metrics = compute_metrics(portfolio_values, dates if dates else None)
    
    return portfolio_values, dates, metrics


def run_experiment(index_name: str, config: dict, experiment_name: str = "MARS"):
    """Run full experiment."""
    logger.info(f"Running {experiment_name} experiment for {index_name}")
    
    # Set seed
    set_seed(config['reproducibility']['seed'])
    
    # Auto-detect device (use GPU if available, otherwise CPU)
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"Using device: {device}")
    
    # Prepare data
    datasets = prepare_data(index_name, config)
    if datasets is None:
        return None
    
    # Create results directory
    results_dir = Path(config['paths']['results_dir']) / index_name.lower() / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    train_features = datasets['train']
    train_env = create_environment(train_features, config)
    hae, mac, risk_overlay = create_models(train_env, config, device)
    
    # Update config with detected device
    config['training']['device'] = device
    trainer, training_stats = train_mars(
        train_env, hae, mac, risk_overlay, config, str(results_dir)
    )
    
    # Evaluate on validation set
    val_features = datasets['val']
    val_env = create_environment(val_features, config)
    val_env.dates = sorted(set().union(*[df.index for df in val_features.values()]))
    val_portfolio_values, val_dates, val_metrics = evaluate_model(
        val_env, hae, mac, risk_overlay, config, split='val'
    )
    
    # Evaluate on test set
    test_features = datasets['test']
    test_env = create_environment(test_features, config)
    test_env.dates = sorted(set().union(*[df.index for df in test_features.values()]))
    test_portfolio_values, test_dates, test_metrics = evaluate_model(
        test_env, hae, mac, risk_overlay, config, split='test'
    )
    
    # Save results
    results = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_portfolio_values': val_portfolio_values,
        'test_portfolio_values': test_portfolio_values,
        'mac_weights_history': training_stats.get('mac_weights_history', [])
    }
    
    # Generate plots
    plot_equity_curve(
        test_portfolio_values, test_dates,
        save_path=f"{results_dir}/equity_curve.png",
        title=f"{experiment_name} - Equity Curve ({index_name})"
    )
    
    plot_drawdown(
        test_portfolio_values, test_dates,
        save_path=f"{results_dir}/drawdown.png",
        title=f"{experiment_name} - Drawdown ({index_name})"
    )
    
    if training_stats.get('mac_weights_history'):
        plot_mac_weights(
            training_stats['mac_weights_history'], test_dates,
            n_agents=config['hae']['n_agents'],
            save_path=f"{results_dir}/mac_weights.png",
            title=f"{experiment_name} - MAC Weights ({index_name})"
        )
    
    # Save metrics table
    metrics_df = pd.DataFrame({
        'Validation': val_metrics,
        'Test': test_metrics
    }).T
    metrics_df.to_csv(f"{results_dir}/metrics.csv")
    
    logger.info(f"Results saved to {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MARS Paper Replication')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--index', type=str, choices=['DJI', 'HSI', 'QQQ'], default='DJI',
                       help='Index to use')
    parser.add_argument('--experiment', type=str, default='MARS',
                       help='Experiment name (MARS, MARS-Static, etc.)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run experiment
    results = run_experiment(args.index, config, args.experiment)
    
    if results:
        logger.info("Experiment completed successfully")
        logger.info(f"Test Metrics: {results['test_metrics']}")
    else:
        logger.error("Experiment failed")


if __name__ == '__main__':
    main()

