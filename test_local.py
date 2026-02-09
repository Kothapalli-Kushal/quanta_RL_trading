"""
Quick test script to verify the MARS pipeline works locally.
Runs for just 1-2 episodes to test the entire process.
"""
import yaml
import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path
import logging

from src.data import DataLoader, FeatureEngineer
from src.env import PortfolioEnv, RiskOverlay
from src.agents import HeterogeneousAgentEnsemble
from src.mac import MetaAdaptiveController
from src.training import Trainer
from src.evaluation import compute_metrics

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


def main():
    """Run quick test for 1-2 episodes."""
    logger.info("=" * 60)
    logger.info("MARS LOCAL TEST - Running 2 episodes for DJI")
    logger.info("=" * 60)
    
    # Load config
    config_path = Path('configs/default.yaml')
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Make sure you're running from the paper_replication directory")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['reproducibility']['seed'])
    
    # Auto-detect device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    elif device == 'cuda':
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"✅ Using device: {device}")
    
    # Prepare data for DJI only
    logger.info("\n📊 Preparing data for DJI...")
    data_loader = DataLoader(data_dir=config['data']['raw_dir'])
    feature_engineer = FeatureEngineer()
    
    # Prepare datasets (only DJI)
    logger.info("📥 Fetching data (this may take a few minutes on first run)...")
    datasets = data_loader.prepare_datasets(
        train_start=config['data']['train_start'],
        train_end=config['data']['train_end'],
        val_start=config['data']['val_start'],
        val_end=config['data']['val_end'],
        test_start=config['data']['test_start'],
        test_end=config['data']['test_end']
    )
    
    if 'DJI' not in datasets:
        logger.error("No data found for DJI")
        return
    
    if not datasets['DJI']['train']:
        logger.error("No training data found for DJI")
        return
    
    # Process features for training set only
    logger.info("🔧 Processing features...")
    train_features = feature_engineer.process_all_assets(datasets['DJI']['train'])
    logger.info(f"✅ Processed features for {len(train_features)} assets")
    
    # Create environment
    logger.info("\n🌍 Creating portfolio environment...")
    train_env = PortfolioEnv(
        features_dict=train_features,
        initial_cash=config['env']['initial_cash'],
        transaction_cost=config['env']['transaction_cost'],
        max_trade_size=config['env']['max_trade_size'],
        risk_window=config['env']['risk_window']
    )
    logger.info(f"✅ Environment created: {train_env.n_assets} assets, {len(train_env.dates)} dates")
    
    # Create models
    logger.info("\n🤖 Creating models...")
    state_dim = train_env.get_state_dim()
    action_dim = train_env.get_action_dim()
    n_agents = config['hae']['n_agents']
    
    hae = HeterogeneousAgentEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        device=device
    )
    
    mac = MetaAdaptiveController(
        state_dim=state_dim,
        n_agents=n_agents,
        lr=config['mac']['lr'],
        epsilon=config['mac']['epsilon'],
        device=device
    )
    
    risk_overlay = RiskOverlay(
        max_position_concentration=config['risk']['max_position_concentration'],
        min_cash_buffer=config['risk']['min_cash_buffer'],
        max_leverage=config['risk']['max_leverage']
    )
    logger.info("✅ All models created")
    
    # Create trainer
    logger.info("\n🏋️ Creating trainer...")
    trainer = Trainer(
        env=train_env,
        hae=hae,
        mac=mac,
        risk_overlay=risk_overlay,
        meta_train_freq=config['training']['meta_train_freq'],
        batch_size=config['training']['batch_size'],
        device=device
    )
    logger.info("✅ Trainer created")
    
    # Run test training (2 episodes)
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Starting test training (2 episodes)...")
    logger.info("=" * 60)
    
    n_test_episodes = 2
    training_stats = trainer.train(
        n_episodes=n_test_episodes,
        start_idx=0,
        log_dir="logs/test"
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ TEST COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Episodes completed: {n_test_episodes}")
    logger.info(f"Final portfolio value: ${training_stats['episode_values'][-1]:,.2f}")
    logger.info(f"Average reward: {np.mean(training_stats['episode_rewards']):.4f}")
    
    # Quick evaluation test
    logger.info("\n📈 Running quick evaluation test...")
    state = train_env.reset(start_idx=0)
    done = False
    portfolio_values = [train_env.initial_cash]
    steps = 0
    max_test_steps = 10  # Just test a few steps
    
    while not done and steps < max_test_steps:
        agent_actions = hae.select_actions(state, add_noise=False)
        aggregated_action = mac.aggregate_actions(state, agent_actions)
        
        current_holdings = train_env.holdings
        current_prices = train_env.prices
        cash = train_env.cash
        portfolio_value = train_env._compute_portfolio_value()
        
        current_allocation = (current_holdings * current_prices) / portfolio_value if portfolio_value > 0 else np.zeros(train_env.n_assets)
        constrained_allocation, new_cash = risk_overlay.apply(
            aggregated_action, current_holdings, current_prices, cash, portfolio_value
        )
        
        action_for_env = constrained_allocation - current_allocation
        action_for_env = np.clip(action_for_env / train_env.max_trade_size, -1.0, 1.0)
        
        next_state, reward, done, info = train_env.step(action_for_env)
        portfolio_values.append(info['portfolio_value'])
        state = next_state
        steps += 1
    
    logger.info(f"✅ Evaluation test passed: {steps} steps completed")
    logger.info(f"Portfolio value range: ${min(portfolio_values):,.2f} - ${max(portfolio_values):,.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL TESTS PASSED! Pipeline is working correctly.")
    logger.info("=" * 60)
    logger.info("\nYou can now run the full training with:")
    logger.info("  python main.py --config configs/default.yaml --index DJI --experiment MARS")


if __name__ == '__main__':
    main()
