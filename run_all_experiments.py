"""
Run all experiments: MARS, ablations, and baselines for both DJI and HSI.
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path
import logging

from main import run_experiment
from run_ablation import run_ablation_study
from src.baselines import BuyAndHold, EqualWeight
from src.data import DataLoader, FeatureEngineer
from src.evaluation import compute_metrics, plot_equity_curve, plot_drawdown
from main import set_seed, prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baselines(index_name: str, config: dict):
    """Run baseline strategies."""
    logger.info(f"Running baselines for {index_name}")
    
    set_seed(config['reproducibility']['seed'])
    
    # Prepare data
    datasets = prepare_data(index_name, config)
    if datasets is None:
        return None
    
    test_features = datasets['test']
    test_dates = sorted(set().union(*[df.index for df in test_features.values()]))
    
    results_dir = Path(config['paths']['results_dir']) / index_name.lower() / 'baselines'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Buy-and-Hold
    logger.info("Running Buy-and-Hold baseline")
    buy_hold = BuyAndHold(initial_cash=config['env']['initial_cash'])
    bh_values = buy_hold.evaluate(test_features, test_dates)
    bh_metrics = compute_metrics(bh_values, test_dates)
    
    plot_equity_curve(bh_values, test_dates,
                     save_path=f"{results_dir}/buy_hold_equity.png",
                     title=f"Buy-and-Hold - Equity Curve ({index_name})")
    
    # Equal Weight
    logger.info("Running Equal Weight baseline")
    equal_weight = EqualWeight(initial_cash=config['env']['initial_cash'])
    ew_values = equal_weight.evaluate(test_features, test_dates)
    ew_metrics = compute_metrics(ew_values, test_dates)
    
    plot_equity_curve(ew_values, test_dates,
                     save_path=f"{results_dir}/equal_weight_equity.png",
                     title=f"Equal Weight - Equity Curve ({index_name})")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Buy-and-Hold': bh_metrics,
        'Equal Weight': ew_metrics
    }).T
    metrics_df.to_csv(f"{results_dir}/metrics.csv")
    
    logger.info(f"Baseline results saved to {results_dir}")
    
    return {
        'buy_hold': bh_metrics,
        'equal_weight': ew_metrics
    }


def run_all_experiments(index_name: str, config: dict):
    """Run all experiments for an index."""
    logger.info(f"Running all experiments for {index_name}")
    
    results = {}
    
    # MARS
    logger.info("=" * 50)
    logger.info("Running MARS")
    logger.info("=" * 50)
    try:
        mars_results = run_experiment(index_name, config, "MARS")
        results['MARS'] = mars_results['test_metrics'] if mars_results else None
    except Exception as e:
        logger.error(f"MARS failed: {e}")
        results['MARS'] = None
    
    # Ablations
    ablations = ['MARS-Static', 'MARS-Homogeneous', 'MARS-Div5', 'MARS-Div15']
    for ablation in ablations:
        logger.info("=" * 50)
        logger.info(f"Running {ablation}")
        logger.info("=" * 50)
        try:
            ablation_metrics = run_ablation_study(index_name, config, ablation)
            results[ablation] = ablation_metrics
        except Exception as e:
            logger.error(f"{ablation} failed: {e}")
            results[ablation] = None
    
    # Baselines
    logger.info("=" * 50)
    logger.info("Running Baselines")
    logger.info("=" * 50)
    try:
        baseline_results = run_baselines(index_name, config)
        if baseline_results:
            results['Buy-and-Hold'] = baseline_results['buy_hold']
            results['Equal Weight'] = baseline_results['equal_weight']
    except Exception as e:
        logger.error(f"Baselines failed: {e}")
    
    # Create comparison table
    logger.info("=" * 50)
    logger.info("Creating comparison table")
    logger.info("=" * 50)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.dropna()
    
    results_dir = Path(config['paths']['results_dir']) / index_name.lower()
    comparison_df.to_csv(f"{results_dir}/comparison_table.csv")
    
    logger.info(f"\nComparison Table for {index_name}:")
    logger.info(f"\n{comparison_df.to_string()}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run All MARS Experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--index', type=str, choices=['DJI', 'HSI', 'QQQ', 'both'],
                       default='both', help='Index to use')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.index == 'both':
        logger.info("Running experiments for DJI, HSI, and QQQ")
        run_all_experiments('DJI', config)
        run_all_experiments('HSI', config)
        run_all_experiments('QQQ', config)
    else:
        run_all_experiments(args.index, config)
    
    logger.info("All experiments completed!")


if __name__ == '__main__':
    main()

