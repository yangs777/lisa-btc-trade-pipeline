"""Moving average indicators."""

import numpy as np
import pandas as pd

from ..base import PriceIndicator


class SMA(PriceIndicator):
    """Simple Moving Average."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize SMA.

        Args:
            window: Period for SMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"SMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SMA."""
        price = self._get_price(df)
        sma = price.rolling(window=self.window_size).mean()
        return self._handle_nan(sma)


class EMA(PriceIndicator):
    """Exponential Moving Average."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize EMA.

        Args:
            window: Period for EMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"EMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMA."""
        price = self._get_price(df)
        ema = price.ewm(span=self.window_size, adjust=False).mean()
        return self._handle_nan(ema)


class WMA(PriceIndicator):
    """Weighted Moving Average."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize WMA.

        Args:
            window: Period for WMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"WMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate WMA."""
        price = self._get_price(df)
        weights = np.arange(1, self.window_size + 1)

        def weighted_mean(x: np.ndarray) -> float:
            return float(np.sum(weights[-len(x) :] * x) / np.sum(weights[-len(x) :]))

        wma = price.rolling(window=self.window_size).apply(weighted_mean, raw=True)
        return self._handle_nan(wma)


class HMA(PriceIndicator):
    """Hull Moving Average."""

    def __init__(self, window: int = 9, price_col: str = "close", fillna: bool = True):
        """Initialize HMA.

        Args:
            window: Period for HMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"HMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate HMA."""
        price = self._get_price(df)

        # Calculate WMA with period n/2
        half_period = int(self.window_size / 2)
        wma_half = self._calculate_wma(price, half_period)

        # Calculate WMA with period n
        wma_full = self._calculate_wma(price, self.window_size)

        # Calculate WMA of (2 * WMA(n/2) - WMA(n)) with period sqrt(n)
        sqrt_period = int(np.sqrt(self.window_size))
        raw_hma = 2 * wma_half - wma_full
        hma = self._calculate_wma(raw_hma, sqrt_period)

        return self._handle_nan(hma)

    def _calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """Helper method to calculate WMA."""
        weights = np.arange(1, period + 1)

        def weighted_mean(x: np.ndarray) -> float:
            if len(x) < period:
                return float("nan")
            return float(np.sum(weights * x[-period:]) / np.sum(weights))

        return series.rolling(window=period).apply(weighted_mean, raw=True)


class TEMA(PriceIndicator):
    """Triple Exponential Moving Average."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize TEMA.

        Args:
            window: Period for TEMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"TEMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate TEMA."""
        price = self._get_price(df)

        # Calculate EMAs
        ema1 = price.ewm(span=self.window_size, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window_size, adjust=False).mean()
        ema3 = ema2.ewm(span=self.window_size, adjust=False).mean()

        # TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3

        return self._handle_nan(tema)


class DEMA(PriceIndicator):
    """Double Exponential Moving Average."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize DEMA.

        Args:
            window: Period for DEMA calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"DEMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate DEMA."""
        price = self._get_price(df)

        # Calculate EMAs
        ema1 = price.ewm(span=self.window_size, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window_size, adjust=False).mean()

        # DEMA = 2 * EMA1 - EMA2
        dema = 2 * ema1 - ema2

        return self._handle_nan(dema)


class KAMA(PriceIndicator):
    """Kaufman's Adaptive Moving Average."""

    def __init__(
        self,
        window: int = 10,
        fast: int = 2,
        slow: int = 30,
        price_col: str = "close",
        fillna: bool = True,
    ):
        """Initialize KAMA.

        Args:
            window: Period for efficiency ratio
            fast: Fast EMA period
            slow: Slow EMA period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"KAMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate KAMA."""
        price = self._get_price(df)

        # Calculate direction and volatility
        direction = (price - price.shift(self.window_size)).abs()
        volatility = (price.diff().abs()).rolling(window=self.window_size).sum()

        # Efficiency Ratio
        er = direction / volatility
        er = er.fillna(0)

        # Smoothing Constants
        fast_sc = 2 / (self.fast + 1)
        slow_sc = 2 / (self.slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Calculate KAMA
        kama = pd.Series(index=price.index, dtype=float)
        kama.iloc[self.window_size - 1] = price.iloc[self.window_size - 1]

        for i in range(self.window_size, len(price)):
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i - 1])

        return self._handle_nan(kama)
