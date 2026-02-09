"""
Feature engineering: Price, MACD, RSI, CCI, ADX (exactly 5 features per asset).
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes exactly 5 features: Price, MACD, RSI, CCI, ADX."""
    
    def __init__(self):
        pass
    
    def compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                    signal: int = 9) -> pd.Series:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compute_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 20) -> pd.Series:
        """Compute Commodity Channel Index."""
        tp = (high + low + close) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    def compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14) -> pd.Series:
        """Compute Average Directional Index."""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth TR and DM
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 5 features for a single asset.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with 5 feature columns: ['price', 'macd', 'rsi', 'cci', 'adx']
        """
        close = data['close']
        high = data['high']
        low = data['low']
        
        features = pd.DataFrame(index=data.index)
        
        # Feature 1: Price (normalized close)
        features['price'] = close / close.iloc[0]  # Normalize to start at 1.0
        
        # Feature 2: MACD
        features['macd'] = self.compute_macd(close)
        
        # Feature 3: RSI
        features['rsi'] = self.compute_rsi(close)
        
        # Feature 4: CCI
        features['cci'] = self.compute_cci(high, low, close)
        
        # Feature 5: ADX
        features['adx'] = self.compute_adx(high, low, close)
        
        # Fill NaN values (forward fill only - no lookahead bias)
        # Use forward fill to propagate last valid value forward, then fill remaining NaNs with 0
        features = features.ffill().fillna(0)
        
        return features
    
    def process_all_assets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all assets in a dataset.
        
        Args:
            data_dict: Dictionary mapping ticker to OHLCV DataFrame
            
        Returns:
            Dictionary mapping ticker to feature DataFrame
        """
        features_dict = {}
        
        for ticker, data in data_dict.items():
            try:
                features = self.compute_features(data)
                features_dict[ticker] = features
                logger.info(f"Computed features for {ticker}: {len(features)} days")
            except Exception as e:
                logger.error(f"Error computing features for {ticker}: {e}")
                continue
        
        return features_dict
    
    def align_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align all asset features to common date index.
        
        Returns:
            Multi-index DataFrame with (date, ticker) -> (price, macd, rsi, cci, adx)
        """
        if not features_dict:
            return pd.DataFrame()
        
        # Get common date index
        all_dates = set()
        for df in features_dict.values():
            all_dates.update(df.index)
        common_dates = sorted(all_dates)
        
        # Create aligned DataFrame
        aligned_data = []
        for date in common_dates:
            for ticker, df in features_dict.items():
                if date in df.index:
                    row = df.loc[date].copy()
                    row['ticker'] = ticker
                    row['date'] = date
                    aligned_data.append(row)
        
        if not aligned_data:
            return pd.DataFrame()
        
        result = pd.DataFrame(aligned_data)
        result = result.set_index(['date', 'ticker'])
        
        return result

