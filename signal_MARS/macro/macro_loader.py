"""
Macro feature loader for loading and aligning macroeconomic data.

Handles low-frequency macro features that are state-level (not asset-level).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime


class MacroLoader:
    """
    Loads and aligns macroeconomic features by date.
    
    Macro features are low-frequency (daily/weekly/monthly) and represent
    state-level information, not asset-specific information.
    """
    
    def __init__(self, forward_fill_days: int = 5):
        """
        Initialize macro loader.
        
        Args:
            forward_fill_days: Number of days to forward fill missing values
        """
        self.forward_fill_days = forward_fill_days
        self.macro_data: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
    
    def load_macro_data(self, data_source: Any) -> pd.DataFrame:
        """
        Load macro data from source.
        
        This is a placeholder method. In practice, this would load from:
        - CSV files
        - Databases
        - APIs (FRED, etc.)
        
        Args:
            data_source: Source of macro data (format depends on implementation)
        
        Returns:
            DataFrame with columns: [date, feature1, feature2, ...]
            Index should be date or datetime
        
        TODO: Implement actual data loading logic.
        """
        # Placeholder: return empty DataFrame with expected structure
        # In practice, load from actual data source
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_features = 10
        
        data = {
            'date': dates,
            **{f'macro_{i}': np.random.randn(len(dates)) for i in range(n_features)}
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        self.macro_data = df
        self.feature_names = [f'macro_{i}' for i in range(n_features)]
        
        return df
    
    def align_by_date(self, target_dates: np.ndarray) -> np.ndarray:
        """
        Align macro features to target dates with forward filling.
        
        Args:
            target_dates: Array of target dates (datetime or timestamps)
        
        Returns:
            Macro feature array of shape (len(target_dates), num_features)
        """
        if self.macro_data is None:
            raise ValueError("Macro data not loaded. Call load_macro_data() first.")
        
        # Convert target_dates to pandas DatetimeIndex if needed
        if not isinstance(target_dates, pd.DatetimeIndex):
            target_dates = pd.to_datetime(target_dates)
        
        # Reindex to target dates with forward fill
        aligned = self.macro_data.reindex(
            target_dates,
            method='ffill',
            limit=self.forward_fill_days
        )
        
        # Fill any remaining NaN with last valid value or zero
        aligned = aligned.fillna(method='ffill').fillna(0.0)
        
        return aligned[self.feature_names].values
    
    def get_features(self, date: datetime) -> np.ndarray:
        """
        Get macro features for a specific date.
        
        Args:
            date: Target date
        
        Returns:
            Macro feature vector of shape (num_features,)
        """
        if self.macro_data is None:
            raise ValueError("Macro data not loaded. Call load_macro_data() first.")
        
        # Find closest date (with forward fill logic)
        if date in self.macro_data.index:
            return self.macro_data.loc[date, self.feature_names].values
        
        # Forward fill: find last valid date before target
        valid_dates = self.macro_data.index[self.macro_data.index <= date]
        if len(valid_dates) > 0:
            last_valid = valid_dates[-1]
            days_diff = (date - last_valid).days
            if days_diff <= self.forward_fill_days:
                return self.macro_data.loc[last_valid, self.feature_names].values
        
        # Fallback: return zeros
        return np.zeros(len(self.feature_names))
    
    def get_feature_dim(self) -> int:
        """
        Get dimension of macro feature vector.
        
        Returns:
            Number of macro features
        """
        return len(self.feature_names)
    
    def normalize_features(self, features: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize macro features.
        
        Args:
            features: Feature array of shape (T, num_features) or (num_features,)
            method: Normalization method ('zscore' or 'minmax')
        
        Returns:
            Normalized features
        """
        if method == "zscore":
            if features.ndim == 1:
                mean = np.mean(features)
                std = np.std(features)
                if std > 0:
                    return (features - mean) / std
                return features
            else:
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True)
                std = np.where(std > 0, std, 1.0)
                return (features - mean) / std
        elif method == "minmax":
            if features.ndim == 1:
                min_val = np.min(features)
                max_val = np.max(features)
                if max_val > min_val:
                    return (features - min_val) / (max_val - min_val) * 2 - 1  # Scale to [-1, 1]
                return features
            else:
                min_val = np.min(features, axis=0, keepdims=True)
                max_val = np.max(features, axis=0, keepdims=True)
                diff = max_val - min_val
                diff = np.where(diff > 0, diff, 1.0)
                return (features - min_val) / diff * 2 - 1
        else:
            raise ValueError(f"Unknown normalization method: {method}")
