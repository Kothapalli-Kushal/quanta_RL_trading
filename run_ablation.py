"""
Run ablation studies: MARS-Static, MARS-Homogeneous, MARS-Div5, MARS-Div15
"""
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import logging

from main import set_seed, prepare_data, create_environment, run_experiment
from src.ablation import MARSStatic, MARSHomogeneous, MARSDiv5, MARSDiv15

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ablation_study(index_name: str, config: dict, ablation_name: str):
    """Run a specific ablation study."""
    logger.info(f"Running {ablation_name} for {index_name}")
    
    # Set seed
    set_seed(config['reproducibility']['seed'])
    
    # Prepare data
    from src.data import DataLoader, FeatureEngineer
    
    data_loader = DataLoader(data_dir=config['data']['raw_dir'])
    feature_engineer = FeatureEngineer()
    
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
    
    # Create environment
    train_features = processed_datasets['train']
    train_env = create_environment(train_features, config)
    
    state_dim = train_env.get_state_dim()
    action_dim = train_env.get_action_dim()
    device = config['training']['device']
    
    # Create models based on ablation
    if ablation_name == 'MARS-Static':
        # Use standard HAE but with static MAC
        from src.agents import HeterogeneousAgentEnsemble
        from src.mac import MetaAdaptiveController
        from src.env import RiskOverlay
        from src.training import Trainer
        
        hae = HeterogeneousAgentEnsemble(state_dim, action_dim, n_agents=10, device=device)
        mac = MetaAdaptiveController(state_dim, n_agents=10, device=device)
        # Replace MAC with static weights
        mac.compute_weights = lambda state: np.ones(10) / 10
        risk_overlay = RiskOverlay()
        
        trainer = Trainer(train_env, hae, mac, risk_overlay, 
                         meta_train_freq=config['training']['meta_train_freq'],
                         batch_size=config['training']['batch_size'],
                         device=device)
        
        training_stats = trainer.train(n_episodes=config['training']['n_episodes'],
                                      start_idx=0,
                                      log_dir=f"logs/{index_name.lower()}/{ablation_name}")
        
    elif ablation_name == 'MARS-Homogeneous':
        # Use homogeneous ensemble
        hae = MARSHomogeneous.create_homogeneous_ensemble(
            state_dim, action_dim, n_agents=10, device=device
        )
        from src.mac import MetaAdaptiveController
        from src.env import RiskOverlay
        from src.training import Trainer
        
        mac = MetaAdaptiveController(state_dim, n_agents=10, device=device)
        risk_overlay = RiskOverlay()
        
        trainer = Trainer(train_env, hae, mac, risk_overlay,
                         meta_train_freq=config['training']['meta_train_freq'],
                         batch_size=config['training']['batch_size'],
                         device=device)
        
        training_stats = trainer.train(n_episodes=config['training']['n_episodes'],
                                      start_idx=0,
                                      log_dir=f"logs/{index_name.lower()}/{ablation_name}")
        
    elif ablation_name == 'MARS-Div5':
        # Use 5 agents
        hae = MARSDiv5.create_ensemble(state_dim, action_dim, n_agents=5, device=device)
        from src.mac import MetaAdaptiveController
        from src.env import RiskOverlay
        from src.training import Trainer
        
        mac = MetaAdaptiveController(state_dim, n_agents=5, device=device)
        risk_overlay = RiskOverlay()
        
        trainer = Trainer(train_env, hae, mac, risk_overlay,
                         meta_train_freq=config['training']['meta_train_freq'],
                         batch_size=config['training']['batch_size'],
                         device=device)
        
        training_stats = trainer.train(n_episodes=config['training']['n_episodes'],
                                      start_idx=0,
                                      log_dir=f"logs/{index_name.lower()}/{ablation_name}")
        
    elif ablation_name == 'MARS-Div15':
        # Use 15 agents
        hae = MARSDiv15.create_ensemble(state_dim, action_dim, n_agents=15, device=device)
        from src.mac import MetaAdaptiveController
        from src.env import RiskOverlay
        from src.training import Trainer
        
        mac = MetaAdaptiveController(state_dim, n_agents=15, device=device)
        risk_overlay = RiskOverlay()
        
        trainer = Trainer(train_env, hae, mac, risk_overlay,
                         meta_train_freq=config['training']['meta_train_freq'],
                         batch_size=config['training']['batch_size'],
                         device=device)
        
        training_stats = trainer.train(n_episodes=config['training']['n_episodes'],
                                      start_idx=0,
                                      log_dir=f"logs/{index_name.lower()}/{ablation_name}")
    
    # Evaluate
    test_features = processed_datasets['test']
    test_env = create_environment(test_features, config)
    test_env.dates = sorted(set().union(*[df.index for df in test_features.values()]))
    
    # Evaluation code similar to main.py
    from src.evaluation import compute_metrics, plot_equity_curve, plot_drawdown
    
    state = test_env.reset(start_idx=0)
    done = False
    portfolio_values = [test_env.initial_cash]
    dates = []
    
    while not done:
        agent_actions = hae.select_actions(state, add_noise=False)
        aggregated_action = mac.aggregate_actions(state, agent_actions)
        
        from src.env import RiskOverlay
        risk_overlay = RiskOverlay()
        current_holdings = test_env.holdings
        current_prices = test_env.prices
        cash = test_env.cash
        portfolio_value = test_env._compute_portfolio_value()
        
        current_allocation = (current_holdings * current_prices) / portfolio_value if portfolio_value > 0 else np.zeros(test_env.n_assets)
        constrained_allocation, new_cash = risk_overlay.apply(
            aggregated_action, current_holdings, current_prices, cash, portfolio_value
        )
        
        action_for_env = constrained_allocation - current_allocation
        action_for_env = np.clip(action_for_env / test_env.max_trade_size, -1.0, 1.0)
        
        next_state, reward, done, info = test_env.step(action_for_env)
        
        portfolio_values.append(info['portfolio_value'])
        if test_env.current_step < len(test_env.dates):
            dates.append(test_env.dates[test_env.current_step])
        
        state = next_state
    
    metrics = compute_metrics(portfolio_values, dates if dates else None)
    
    # Save results
    results_dir = Path(config['paths']['results_dir']) / index_name.lower() / ablation_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_equity_curve(portfolio_values, dates,
                     save_path=f"{results_dir}/equity_curve.png",
                     title=f"{ablation_name} - Equity Curve ({index_name})")
    
    plot_drawdown(portfolio_values, dates,
                 save_path=f"{results_dir}/drawdown.png",
                 title=f"{ablation_name} - Drawdown ({index_name})")
    
    import pandas as pd
    metrics_df = pd.DataFrame({'Test': metrics}).T
    metrics_df.to_csv(f"{results_dir}/metrics.csv")
    
    logger.info(f"{ablation_name} results saved to {results_dir}")
    logger.info(f"Metrics: {metrics}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run MARS Ablation Studies')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--index', type=str, choices=['DJI', 'HSI'], default='DJI',
                       help='Index to use')
    parser.add_argument('--ablation', type=str,
                       choices=['MARS-Static', 'MARS-Homogeneous', 'MARS-Div5', 'MARS-Div15'],
                       help='Ablation study to run')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_ablation_study(args.index, config, args.ablation)


if __name__ == '__main__':
    main()

