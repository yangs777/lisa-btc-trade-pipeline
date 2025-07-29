"""Other volatility indicators."""

import numpy as np
import pandas as pd

from ..base import PriceIndicator


class UlcerIndex(PriceIndicator):
    """Ulcer Index."""

    def __init__(self, window: int = 14, price_col: str = "close", fillna: bool = True):
        """Initialize Ulcer Index.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ULCER_INDEX_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ulcer Index."""
        price = self._get_price(df)

        # Calculate rolling maximum
        rolling_max = price.rolling(window=self.window_size).max()

        # Calculate percentage drawdown
        drawdown = ((price - rolling_max) / rolling_max) * 100

        # Square the drawdowns
        squared_dd = drawdown ** 2

        # Calculate mean of squared drawdowns
        mean_squared_dd = squared_dd.rolling(window=self.window_size).mean()

        # Take square root
        ulcer_index = np.sqrt(mean_squared_dd)

        return self._handle_nan(ulcer_index)


class MassIndex(PriceIndicator):
    """Mass Index."""

    def __init__(self, window: int = 25, ema_period: int = 9, fillna: bool = True):
        """Initialize Mass Index.
        
        Args:
            window: Sum period
            ema_period: EMA period for range calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.ema_period = ema_period

    @property
    def name(self) -> str:
        return f"MASS_INDEX_{self.window_size}_{self.ema_period}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Mass Index."""
        # This indicator requires high and low prices
        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError("Mass Index requires 'high' and 'low' columns")

        # Calculate high-low range
        high_low = df['high'] - df['low']

        # Calculate single EMA of range
        ema1 = high_low.ewm(span=self.ema_period, adjust=False).mean()

        # Calculate double EMA of range
        ema2 = ema1.ewm(span=self.ema_period, adjust=False).mean()

        # Calculate ratio
        ratio = ema1 / ema2

        # Sum the ratios over the window period
        mass_index = ratio.rolling(window=self.window_size).sum()

        return self._handle_nan(mass_index)
