"""Feature selection utilities."""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureSelector:
    """Select relevant features for trading."""
    
    def __init__(self, method: str = "correlation", threshold: float = 0.95):
        """Initialize selector.
        
        Args:
            method: Selection method (correlation, variance, mutual_info)
            threshold: Threshold for feature removal
        """
        self.method = method
        self.threshold = threshold
        self.selected_features: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit selector on data.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Self
        """
        if self.method == "correlation":
            self._fit_correlation(X)
        elif self.method == "variance":
            self._fit_variance(X)
        elif self.method == "mutual_info":
            self._fit_mutual_info(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed data with selected features
        """
        if self.selected_features is None:
            raise ValueError("Selector not fitted yet")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def _fit_correlation(self, X: pd.DataFrame) -> None:
        """Fit using correlation threshold."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            correlated = List[Any](upper_tri.index[upper_tri[column] > self.threshold])
            to_drop.update(correlated)
        
        # Store selected features
        self.selected_features = [col for col in X.columns if col not in to_drop]
    
    def _fit_variance(self, X: pd.DataFrame) -> None:
        """Fit using variance threshold."""
        # Calculate variance
        variances = X.var()
        
        # Select features above threshold
        self.selected_features = List[Any](variances[variances > self.threshold].index)
    
    def _fit_mutual_info(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Fit using mutual information."""
        if y is None:
            raise ValueError("Target variable required for mutual info selection")
        
        # Mock implementation - would use sklearn.feature_selection.mutual_info_regression
        # For now, just select all features
        self.selected_features = List[Any](X.columns)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores.
        
        Returns:
            Series with feature importance
        """
        if self.selected_features is None:
            raise ValueError("Selector not fitted yet")
        
        # Mock importance scores
        importance = pd.Series(
            index=self.selected_features,
            data=np.random.random(len(self.selected_features))
        )
        return importance.sort_values(ascending=False)