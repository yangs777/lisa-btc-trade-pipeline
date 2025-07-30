"""Cross-validation utilities for time series."""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional


class TimeSeriesCV:
    """Time series cross-validation splitter."""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.8,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        """Initialize time series CV.
        
        Args:
            n_splits: Number of splits
            train_size: Training set size ratio
            gap: Gap between train and test sets
            max_train_size: Maximum training set size
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for time series CV.
        
        Args:
            data: DataFrame to split
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(data)
        
        # Calculate minimum training size
        min_train_size = int(n_samples * self.train_size / self.n_splits)
        
        # Calculate test size
        test_size = int(n_samples * (1 - self.train_size) / self.n_splits)
        
        for i in range(self.n_splits):
            # Calculate train end position
            train_end = min_train_size + i * test_size
            
            # Apply max train size constraint
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            # Calculate test positions
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            # Skip if not enough data for test
            if test_start >= n_samples:
                break
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, data: Optional[pd.DataFrame] = None) -> int:
        """Get number of splits.
        
        Args:
            data: Optional data (not used)
            
        Returns:
            Number of splits
        """
        return self.n_splits


class WalkForwardCV:
    """Walk-forward cross-validation for time series."""
    
    def __init__(
        self,
        train_period: int,
        test_period: int,
        step: Optional[int] = None
    ):
        """Initialize walk-forward CV.
        
        Args:
            train_period: Training period length
            test_period: Test period length
            step: Step size (defaults to test_period)
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step = step or test_period
    
    def split(self, data: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward CV.
        
        Args:
            data: DataFrame to split
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(data)
        
        # Start from first possible position
        test_start = self.train_period
        
        while test_start + self.test_period <= n_samples:
            # Train indices
            train_start = test_start - self.train_period
            train_indices = np.arange(train_start, test_start)
            
            # Test indices
            test_end = test_start + self.test_period
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
            
            # Move forward
            test_start += self.step