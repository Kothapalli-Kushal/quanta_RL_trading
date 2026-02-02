"""
Data loader for fetching DJI and HSI constituent stocks using yfinance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads OHLCV data for DJI and HSI constituents."""
    
    # DJI constituents (as of 2024)
    DJI_TICKERS = [
        'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'CAT', 'MCD', 'AMGN', 'V', 'HON',
        'TRV', 'AXP', 'JPM', 'WMT', 'IBM', 'JNJ', 'PG', 'CVX', 'MRK', 'BA',
        'DIS', 'NKE', 'CSCO', 'DOW', 'VZ', 'INTC', 'AMZN', 'CRM', 'MMM', 'KO'
    ]
    
    # HSI constituents (as of 2024) - using common Hong Kong stocks
    HSI_TICKERS = [
        '0700.HK', '0941.HK', '1299.HK', '2318.HK', '1398.HK', '3988.HK',
        '0388.HK', '2628.HK', '0939.HK', '0005.HK', '9988.HK', '3690.HK',
        '1810.HK', '2269.HK', '1024.HK', '2382.HK', '1177.HK', '2015.HK',
        '1093.HK', '2020.HK', '0016.HK', '0011.HK', '0066.HK', '0002.HK',
        '0003.HK', '0012.HK', '0017.HK', '0027.HK', '0069.HK', '0083.HK',
        '0101.HK', '0144.HK', '0151.HK', '0175.HK', '0179.HK', '0200.HK',
        '0267.HK', '0288.HK', '0291.HK', '0293.HK', '0322.HK', '0386.HK',
        '0388.HK', '0669.HK', '0688.HK', '0753.HK', '0762.HK', '0823.HK',
        '0968.HK', '1109.HK'
    ]
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_index_data(self, index_name: str, tickers: List[str], 
                        start_date: str, end_date: str, 
                        max_stocks: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for index constituents.
        
        Args:
            index_name: 'DJI' or 'HSI'
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_stocks: Maximum number of stocks to fetch
            
        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV columns
        """
        logger.info(f"Fetching {index_name} data from {start_date} to {end_date}")
        
        data_dict = {}
        tickers_to_fetch = tickers[:max_stocks]
        
        for ticker in tickers_to_fetch:
            try:
                logger.info(f"Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Ensure OHLCV columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing columns for {ticker}")
                    continue
                
                # Rename to standard format
                df = df[required_cols].copy()
                df.columns = [c.lower() for c in df.columns]
                
                # Remove timezone if present
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                data_dict[ticker] = df
                logger.info(f"Successfully fetched {ticker}: {len(df)} days")
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(data_dict)} stocks for {index_name}")
        return data_dict
    
    def save_data(self, index_name: str, data_dict: Dict[str, pd.DataFrame]):
        """Save data to disk."""
        index_dir = self.data_dir / index_name.lower()
        index_dir.mkdir(parents=True, exist_ok=True)
        
        for ticker, df in data_dict.items():
            # Clean ticker name for filename
            clean_ticker = ticker.replace('.', '_').replace('-', '_')
            filepath = index_dir / f"{clean_ticker}.csv"
            df.to_csv(filepath)
            logger.info(f"Saved {ticker} to {filepath}")
    
    def load_data(self, index_name: str) -> Dict[str, pd.DataFrame]:
        """Load data from disk."""
        index_dir = self.data_dir / index_name.lower()
        data_dict = {}
        
        if not index_dir.exists():
            logger.warning(f"Data directory {index_dir} does not exist")
            return data_dict
        
        for filepath in index_dir.glob("*.csv"):
            ticker = filepath.stem.replace('_', '.')
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data_dict[ticker] = df
            logger.info(f"Loaded {ticker}: {len(df)} days")
        
        return data_dict
    
    def prepare_datasets(self, train_start: str, train_end: str,
                        val_start: str, val_end: str,
                        test_start: str, test_end: str):
        """
        Prepare train/val/test splits for both indices.
        
        Returns:
            Dictionary with structure:
            {
                'dji': {'train': {...}, 'val': {...}, 'test': {...}},
                'hsi': {'train': {...}, 'val': {...}, 'test': {...}}
            }
        """
        datasets = {}
        
        for index_name, tickers in [('DJI', self.DJI_TICKERS), ('HSI', self.HSI_TICKERS)]:
            logger.info(f"Preparing datasets for {index_name}")
            
            # Fetch all data
            all_data = self.fetch_index_data(
                index_name, tickers, train_start, test_end, max_stocks=50
            )
            
            if not all_data:
                logger.warning(f"No data fetched for {index_name}")
                continue
            
            # Split by date
            datasets[index_name] = {
                'train': {},
                'val': {},
                'test': {}
            }
            
            for ticker, df in all_data.items():
                train_df = df[(df.index >= train_start) & (df.index < val_start)]
                val_df = df[(df.index >= val_start) & (df.index < test_start)]
                test_df = df[(df.index >= test_start) & (df.index <= test_end)]
                
                if not train_df.empty:
                    datasets[index_name]['train'][ticker] = train_df
                if not val_df.empty:
                    datasets[index_name]['val'][ticker] = val_df
                if not test_df.empty:
                    datasets[index_name]['test'][ticker] = test_df
            
            # Save to disk
            self.save_data(index_name, all_data)
        
        return datasets

