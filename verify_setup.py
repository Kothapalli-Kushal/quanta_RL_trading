"""
Quick verification script to test if all imports work correctly.
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        from src.data import DataLoader, FeatureEngineer
        print("✅ Data imports OK")
    except Exception as e:
        print(f"❌ Data imports failed: {e}")
        return False
    
    try:
        from src.env import PortfolioEnv, RiskOverlay
        print("✅ Environment imports OK")
    except Exception as e:
        print(f"❌ Environment imports failed: {e}")
        return False
    
    try:
        from src.agents import HeterogeneousAgentEnsemble, DDPGAgent
        print("✅ Agent imports OK")
    except Exception as e:
        print(f"❌ Agent imports failed: {e}")
        return False
    
    try:
        from src.mac import MetaAdaptiveController
        print("✅ MAC imports OK")
    except Exception as e:
        print(f"❌ MAC imports failed: {e}")
        return False
    
    try:
        from src.training import Trainer
        print("✅ Training imports OK")
    except Exception as e:
        print(f"❌ Training imports failed: {e}")
        return False
    
    try:
        from src.evaluation import compute_metrics, plot_equity_curve
        print("✅ Evaluation imports OK")
    except Exception as e:
        print(f"❌ Evaluation imports failed: {e}")
        return False
    
    try:
        from src.ablation import MARSStatic, MARSHomogeneous
        print("✅ Ablation imports OK")
    except Exception as e:
        print(f"❌ Ablation imports failed: {e}")
        return False
    
    try:
        from src.baselines import BuyAndHold, EqualWeight
        print("✅ Baseline imports OK")
    except Exception as e:
        print(f"❌ Baseline imports failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

