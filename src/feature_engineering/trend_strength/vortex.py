"""Vortex indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class VortexPlus(OHLCVIndicator):
    """Vortex Indicator Positive."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize VI+.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"VI_PLUS_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VI+."""
        self._validate_ohlcv(df)

        # Calculate Vortex Movements
        vm_plus = (df["high"] - df["low"].shift(1)).abs()

        # Calculate True Range
        true_range = self._calculate_true_range(df)

        # Sum over window
        vm_plus_sum = vm_plus.rolling(window=self.window_size).sum()
        tr_sum = true_range.rolling(window=self.window_size).sum()

        # Calculate VI+
        vi_plus = vm_plus_sum / tr_sum

        return self._handle_nan(vi_plus)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


class VortexMinus(OHLCVIndicator):
    """Vortex Indicator Negative."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize VI-.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"VI_MINUS_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VI-."""
        self._validate_ohlcv(df)

        # Calculate Vortex Movements
        vm_minus = (df["low"] - df["high"].shift(1)).abs()

        # Calculate True Range
        true_range = self._calculate_true_range(df)

        # Sum over window
        vm_minus_sum = vm_minus.rolling(window=self.window_size).sum()
        tr_sum = true_range.rolling(window=self.window_size).sum()

        # Calculate VI-
        vi_minus = vm_minus_sum / tr_sum

        return self._handle_nan(vi_minus)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
