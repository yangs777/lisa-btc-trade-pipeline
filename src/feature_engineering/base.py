from typing import Dict, List, Any, Optional, Union, Tuple

"""Base classes for technical indicators."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseIndicator(ABC):
    """Base class for all technical indicators."""

    def __init__(
        self,
        window_size: int = 14,
        fillna: bool = True,
        fill_method: str = "zero",
    ):
        """Initialize base indicator.

        Args:
            window_size: Lookback period for calculations
            fillna: Whether to fill NaN values
            fill_method: Method to fill NaN ('zero', 'ffill', 'bfill')
        """
        self.window_size = window_size
        self.fillna = fillna
        self.fill_method = fill_method

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate indicator values.

        Args:
            df: Input DataFrame with OHLCV columns

        Returns:
            Series with indicator values
        """
        pass

    def _handle_nan(self, series: pd.Series) -> pd.Series:
        """Handle NaN values according to configuration."""
        if not self.fillna:
            return series

        if self.fill_method == "zero":
            return series.fillna(0)
        elif self.fill_method == "ffill":
            return series.fillna(method="ffill")
        elif self.fill_method == "bfill":
            return series.fillna(method="bfill")
        else:
            return series

    @property
    @abstractmethod
    def name(self) -> str:
        """Get indicator name."""
        pass


class PriceIndicator(BaseIndicator):
    """Base class for price-based indicators."""

    def __init__(
        self,
        price_col: str = "close",
        window_size: int = 14,
        fillna: bool = True,
        fill_method: str = "zero",
    ):
        """Initialize price indicator.

        Args:
            price_col: Column to use for price ('close', 'high', 'low', 'open')
            window_size: Lookback period
            fillna: Whether to fill NaN values
            fill_method: Method to fill NaN
        """
        super().__init__(window_size, fillna, fill_method)
        self.price_col = price_col

    def _get_price(self, df: pd.DataFrame) -> pd.Series:
        """Extract price series from DataFrame."""
        if self.price_col not in df.columns:
            raise ValueError(f"Column '{self.price_col}' not found in DataFrame")
        return df[self.price_col]


class VolumeIndicator(BaseIndicator):
    """Base class for volume-based indicators."""

    def _get_volume(self, df: pd.DataFrame) -> pd.Series:
        """Extract volume series from DataFrame."""
        if "volume" not in df.columns:
            raise ValueError("Column 'volume' not found in DataFrame")
        return df["volume"]


class OHLCVIndicator(BaseIndicator):
    """Base class for indicators using full OHLCV data."""

    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate required OHLCV columns exist."""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
