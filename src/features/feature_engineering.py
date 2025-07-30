"""Feature engineering and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Tuple, List, Optional


class FeatureScaler:
    """Scale features to standard normal distribution."""
    
    def __init__(self):
        """Initialize scaler."""
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on data."""
        self.feature_names = List[Any](X.columns)
        self.scaler.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        if self.feature_names is None:
            raise ValueError("Scaler not fitted yet")
        
        # Ensure columns match
        X_subset = X[self.feature_names]
        X_scaled = self.scaler.transform(X_subset)
        
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data."""
        X_unscaled = self.scaler.inverse_transform(X)
        return pd.DataFrame(X_unscaled, columns=X.columns, index=X.index)


class FeatureSelector:
    """Select top K features based on statistical tests."""
    
    def __init__(self, n_features: int = 50):
        """Initialize selector."""
        self.n_features = n_features
        self.selector = SelectKBest(score_func=f_regression, k=n_features)
        self.selected_features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit selector on data."""
        self.selector.fit(X, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = List[Any](X.columns[mask])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        if self.selected_features is None:
            raise ValueError("Selector not fitted yet")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get List[Any] of selected feature names."""
        if self.selected_features is None:
            raise ValueError("Selector not fitted yet")
        return self.selected_features


class FeaturePipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.scaler = FeatureScaler()
        self.selector = None
    
    def add_features(self, df: pd.DataFrame, indicators) -> pd.DataFrame:
        """Add technical indicators and engineered features."""
        # Add technical indicators
        df_with_indicators = indicators.add_all_indicators(df.copy())
        
        # Add custom features
        df_with_indicators["price_change"] = df_with_indicators["close"].pct_change()
        df_with_indicators["volume_change"] = df_with_indicators["volume"].pct_change()
        df_with_indicators["high_low_ratio"] = df_with_indicators["high"] / df_with_indicators["low"]
        df_with_indicators["close_to_high"] = df_with_indicators["close"] / df_with_indicators["high"]
        
        return df_with_indicators
    
    def prepare_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for model training."""
        # Drop non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Remove target if specified
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])
        
        # Remove any columns with all NaN
        numeric_df = numeric_df.dropna(axis=1, how="all")
        
        # Fill NaN with forward fill then backward fill
        numeric_df = numeric_df.ffill().bfill()
        
        # Convert to numpy array
        X = numeric_df.values
        feature_names = List[Any](numeric_df.columns)
        
        return X, feature_names