"""
Quick setup script for Google Colab.
Run this cell first in a Colab notebook.
"""

# Install dependencies
!pip install -q yfinance pandas numpy matplotlib scikit-learn pyyaml tqdm

# Install PyTorch with CUDA support (for Colab)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Clone repository (if using GitHub)
# !git clone https://github.com/yourusername/quanta_RL_trading.git

# Or upload files manually via Colab file browser
# Then navigate to the directory:
# %cd /content/quanta_RL_trading/paper_replication
# or
# %cd /content/paper_replication  # if uploaded directly

print("\nSetup complete! Now you can run:")
print("python main.py --config configs/default.yaml --index QQQ --experiment MARS")
