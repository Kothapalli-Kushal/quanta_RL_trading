"""
Setup script for MARS replication project.
"""
from setuptools import setup, find_packages

setup(
    name='mars-replication',
    version='1.0.0',
    description='Full replication of MARS paper (arXiv:2508.01173)',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'yfinance>=0.2.28',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.3.0',
        'ta>=0.11.0',
        'gymnasium>=0.29.0',
        'tensorboard>=2.14.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0'
    ],
)

