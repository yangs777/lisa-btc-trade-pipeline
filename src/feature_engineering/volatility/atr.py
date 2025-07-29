"""ATR (Average True Range) indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class ATR(OHLCVIndicator):
    """Average True Range."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize ATR.

        Args:
            window: Period for ATR calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ATR_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR."""
        self._validate_ohlcv(df)

        # Calculate True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR as EMA of True Range
        atr = true_range.ewm(span=self.window_size, adjust=False).mean()

        return self._handle_nan(atr)


class NATR(OHLCVIndicator):
    """Normalized Average True Range."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize NATR.

        Args:
            window: Period for NATR calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"NATR_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate NATR."""
        # Calculate ATR
        atr_indicator = ATR(window=self.window_size, fillna=False)
        atr = atr_indicator.transform(df)

        # Normalize by close price
        natr = (atr / df["close"]) * 100

        return self._handle_nan(natr)
