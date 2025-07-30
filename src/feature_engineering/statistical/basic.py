from typing import Dict, List, Any, Optional, Union, Tuple

"""Basic statistical indicators."""

import numpy as np
import pandas as pd

from ..base import PriceIndicator


class StdDev(PriceIndicator):
    """Standard Deviation."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize StdDev.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"STDDEV_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Standard Deviation."""
        price = self._get_price(df)
        stddev = price.rolling(window=self.window_size).std()
        return self._handle_nan(stddev)


class Variance(PriceIndicator):
    """Variance."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Variance.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"VAR_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Variance."""
        price = self._get_price(df)
        variance = price.rolling(window=self.window_size).var()
        return self._handle_nan(variance)


class SEM(PriceIndicator):
    """Standard Error of Mean."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize SEM.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"SEM_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SEM."""
        price = self._get_price(df)

        # SEM = std / sqrt(n)
        std = price.rolling(window=self.window_size).std()
        sem = std / np.sqrt(self.window_size)

        return self._handle_nan(sem)


class Skew(PriceIndicator):
    """Skewness."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Skew.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"SKEW_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Skewness."""
        price = self._get_price(df)
        skewness = price.rolling(window=self.window_size).skew()
        return self._handle_nan(skewness)


class Kurtosis(PriceIndicator):
    """Kurtosis."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Kurtosis.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"KURT_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Kurtosis."""
        price = self._get_price(df)
        kurt = price.rolling(window=self.window_size).kurt()
        return self._handle_nan(kurt)
