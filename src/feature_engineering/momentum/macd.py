from typing import Dict, List, Any, Optional, Union, Tuple

"""MACD indicators."""

import pandas as pd

from ..base import PriceIndicator


class MACD(PriceIndicator):
    """MACD Line."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
        fillna: bool = True,
    ):
        """Initialize MACD.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=slow, fillna=fillna)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD line."""
        price = self._get_price(df)

        # Calculate EMAs
        ema_fast = price.ewm(span=self.fast, adjust=False).mean()
        ema_slow = price.ewm(span=self.slow, adjust=False).mean()

        # MACD line
        macd = ema_fast - ema_slow

        return self._handle_nan(macd)


class MACDSignal(PriceIndicator):
    """MACD Signal Line."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
        fillna: bool = True,
    ):
        """Initialize MACD Signal.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=slow, fillna=fillna)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD_SIGNAL_{self.fast}_{self.slow}_{self.signal}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD signal line."""
        # First calculate MACD line
        macd_indicator = MACD(
            fast=self.fast,
            slow=self.slow,
            signal=self.signal,
            price_col=self.price_col,
            fillna=False,
        )
        macd = macd_indicator.transform(df)

        # Signal line is EMA of MACD
        signal = macd.ewm(span=self.signal, adjust=False).mean()

        return self._handle_nan(signal)


class MACDHist(PriceIndicator):
    """MACD Histogram."""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
        fillna: bool = True,
    ):
        """Initialize MACD Histogram.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=slow, fillna=fillna)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD_HIST_{self.fast}_{self.slow}_{self.signal}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD histogram."""
        # Calculate MACD line
        macd_indicator = MACD(
            fast=self.fast,
            slow=self.slow,
            signal=self.signal,
            price_col=self.price_col,
            fillna=False,
        )
        macd = macd_indicator.transform(df)

        # Calculate signal line
        signal_indicator = MACDSignal(
            fast=self.fast,
            slow=self.slow,
            signal=self.signal,
            price_col=self.price_col,
            fillna=False,
        )
        signal = signal_indicator.transform(df)

        # Histogram is MACD - Signal
        histogram = macd - signal

        return self._handle_nan(histogram)
