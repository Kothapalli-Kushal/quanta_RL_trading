"""
Smoke test: short training -> save models -> load models -> evaluate.
Verifies the full pipeline works without downloading data.
"""
import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import PortfolioEnv, RiskOverlay
from src.agents import HeterogeneousAgentEnsemble
from src.mac import MetaAdaptiveController
from src.training import Trainer


def make_minimal_features(n_dates: int = 100, n_assets: int = 3) -> dict:
    """Create minimal feature dict for testing (no network/data download)."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    features_dict = {}
    for i in range(n_assets):
        ticker = f"ASSET_{i}"
        df = pd.DataFrame(
            index=dates,
            data={
                "price": 100.0 * (1 + np.cumsum(np.random.randn(n_dates) * 0.01)),
                "macd": np.random.randn(n_dates).cumsum(),
                "rsi": 30 + 40 * np.random.rand(n_dates),
                "cci": np.random.randn(n_dates) * 50,
                "adx": 20 + 30 * np.random.rand(n_dates),
            },
        )
        df["price"] = df["price"].clip(lower=1.0)
        features_dict[ticker] = df
    return features_dict


def test_training_save_load_eval():
    """Run 2 episodes, save, load, run one eval step."""
    device = "cuda" if __import__("torch").torch.cuda.is_available() else "cpu"
    n_episodes = 2
    n_assets = 3
    n_dates = 80  # enough for risk_window=30 and a few steps

    features_dict = make_minimal_features(n_dates=n_dates, n_assets=n_assets)
    env = PortfolioEnv(
        features_dict=features_dict,
        initial_cash=1e6,
        transaction_cost=0.001,
        max_trade_size=0.1,
        risk_window=30,
    )

    hae = HeterogeneousAgentEnsemble(
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        n_agents=4,  # small for speed
        device=device,
    )
    mac = MetaAdaptiveController(
        state_dim=env.get_state_dim(),
        n_agents=4,
        lr=1e-4,
        epsilon=1e-6,
        device=device,
    )
    risk_overlay = RiskOverlay(
        max_position_concentration=0.2,
        min_cash_buffer=0.05,
        max_leverage=1.0,
    )

    trainer = Trainer(
        env=env,
        hae=hae,
        mac=mac,
        risk_overlay=risk_overlay,
        meta_train_freq=2,
        batch_size=32,
        agent_update_freq=1,
        device=device,
    )

    # Train a few episodes
    with tempfile.TemporaryDirectory() as logtmp:
        stats = trainer.train(n_episodes=n_episodes, start_idx=0, log_dir=logtmp)
    assert "episode_rewards" in stats
    assert len(stats["episode_rewards"]) == n_episodes

    # Save models (this used to fail with "Parent directory ... does not exist")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "models")
        trainer.save_models(save_dir)

        # Check files exist
        assert os.path.isdir(save_dir)
        agents_dir = os.path.join(save_dir, "agents")
        assert os.path.isdir(agents_dir)
        mac_path = os.path.join(save_dir, "mac.pth")
        assert os.path.isfile(mac_path)
        for i in range(4):
            assert os.path.isfile(os.path.join(agents_dir, f"agent_{i}.pth"))

        # Load into new models and run one eval step
        env2 = PortfolioEnv(
            features_dict=features_dict,
            initial_cash=1e6,
            transaction_cost=0.001,
            max_trade_size=0.1,
            risk_window=30,
        )
        hae2 = HeterogeneousAgentEnsemble(
            state_dim=env2.get_state_dim(),
            action_dim=env2.get_action_dim(),
            n_agents=4,
            device=device,
        )
        mac2 = MetaAdaptiveController(
            state_dim=env2.get_state_dim(),
            n_agents=4,
            lr=1e-4,
            epsilon=1e-6,
            device=device,
        )
        trainer2 = Trainer(
            env=env2,
            hae=hae2,
            mac=mac2,
            risk_overlay=risk_overlay,
            meta_train_freq=2,
            batch_size=32,
            agent_update_freq=1,
            device=device,
        )
        trainer2.load_models(save_dir)

        state = env2.reset(start_idx=0)
        agent_actions = hae2.select_actions(state, add_noise=False)
        aggregated = mac2.aggregate_actions(state, agent_actions)
        assert aggregated.shape == (n_assets,)
        assert not np.any(np.isnan(aggregated))

    print("test_training_save_load_eval passed.")


if __name__ == "__main__":
    test_training_save_load_eval()
