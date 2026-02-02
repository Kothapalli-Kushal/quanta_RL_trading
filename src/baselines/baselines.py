"""
Baseline strategies: Buy-and-Hold, Equal Weight, etc.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuyAndHold:
    """
    Buy-and-Hold baseline: Invest equally in all assets at the start.
    """
    
    def __init__(self, initial_cash: float = 1e6):
        self.initial_cash = initial_cash
    
    def evaluate(self, features_dict: Dict[str, pd.DataFrame],
                dates: List = None) -> List[float]:
        """
        Evaluate buy-and-hold strategy.
        
        Args:
            features_dict: Dictionary mapping ticker to feature DataFrame
            dates: List of dates (optional)
            
        Returns:
            List of portfolio values over time
        """
        tickers = sorted(list(features_dict.keys()))
        n_assets = len(tickers)
        
        if n_assets == 0:
            return [self.initial_cash]
        
        # Get common dates
        all_dates = set()
        for df in features_dict.values():
            all_dates.update(df.index)
        common_dates = sorted(all_dates)
        
        if dates is not None:
            common_dates = [d for d in common_dates if d in dates]
        
        if not common_dates:
            return [self.initial_cash]
        
        # Extract prices
        prices = {}
        for ticker in tickers:
            df = features_dict[ticker]
            prices[ticker] = df['price'].reindex(common_dates, method='ffill')
        
        # Initial allocation: equal weight
        allocation_per_asset = 1.0 / n_assets
        initial_value_per_asset = self.initial_cash * allocation_per_asset
        
        # Compute portfolio value over time
        portfolio_values = []
        for date in common_dates:
            portfolio_value = 0.0
            for ticker in tickers:
                if date in prices[ticker].index:
                    price = prices[ticker].loc[date]
                    # Number of shares = initial_value / initial_price
                    initial_price = prices[ticker].iloc[0]
                    shares = initial_value_per_asset / initial_price if initial_price > 0 else 0
                    portfolio_value += shares * price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values


class EqualWeight:
    """
    Equal Weight baseline: Rebalance to equal weights every period.
    """
    
    def __init__(self, initial_cash: float = 1e6, rebalance_freq: int = 1):
        """
        Args:
            initial_cash: Initial cash
            rebalance_freq: Rebalance frequency (1 = every day)
        """
        self.initial_cash = initial_cash
        self.rebalance_freq = rebalance_freq
    
    def evaluate(self, features_dict: Dict[str, pd.DataFrame],
                dates: List = None) -> List[float]:
        """
        Evaluate equal-weight strategy.
        
        Args:
            features_dict: Dictionary mapping ticker to feature DataFrame
            dates: List of dates (optional)
            
        Returns:
            List of portfolio values over time
        """
        tickers = sorted(list(features_dict.keys()))
        n_assets = len(tickers)
        
        if n_assets == 0:
            return [self.initial_cash]
        
        # Get common dates
        all_dates = set()
        for df in features_dict.values():
            all_dates.update(df.index)
        common_dates = sorted(all_dates)
        
        if dates is not None:
            common_dates = [d for d in common_dates if d in dates]
        
        if not common_dates:
            return [self.initial_cash]
        
        # Extract prices
        prices = {}
        for ticker in tickers:
            df = features_dict[ticker]
            prices[ticker] = df['price'].reindex(common_dates, method='ffill')
        
        # Track portfolio
        portfolio_value = self.initial_cash
        shares = {ticker: 0.0 for ticker in tickers}
        portfolio_values = [portfolio_value]
        
        for i, date in enumerate(common_dates[1:], 1):
            # Rebalance if needed
            if i % self.rebalance_freq == 0:
                # Compute current portfolio value
                current_value = 0.0
                for ticker in tickers:
                    if date in prices[ticker].index:
                        price = prices[ticker].loc[date]
                        current_value += shares[ticker] * price
                
                # Rebalance to equal weights
                target_value_per_asset = current_value / n_assets
                for ticker in tickers:
                    if date in prices[ticker].index:
                        price = prices[ticker].loc[date]
                        target_shares = target_value_per_asset / price if price > 0 else 0
                        shares[ticker] = target_shares
            
            # Compute portfolio value
            portfolio_value = 0.0
            for ticker in tickers:
                if date in prices[ticker].index:
                    price = prices[ticker].loc[date]
                    portfolio_value += shares[ticker] * price
            
            portfolio_values.append(portfolio_value)
        
        return portfolio_values

