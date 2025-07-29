"""ADX (Average Directional Index) indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class DIPlus(OHLCVIndicator):
    """Positive Directional Indicator (+DI)."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize +DI.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"DI_PLUS_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate +DI."""
        self._validate_ohlcv(df)

        # Calculate directional movements
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()

        # Positive directional movement
        pos_dm = pd.Series(0, index=df.index)
        pos_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff[
            (high_diff > low_diff) & (high_diff > 0)
        ]

        # Calculate True Range
        true_range = self._calculate_true_range(df)

        # Smooth with EMA
        pos_dm_smooth = pos_dm.ewm(span=self.window_size, adjust=False).mean()
        tr_smooth = true_range.ewm(span=self.window_size, adjust=False).mean()

        # Calculate +DI
        di_plus = 100 * pos_dm_smooth / tr_smooth

        return self._handle_nan(di_plus)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


class DIMinus(OHLCVIndicator):
    """Negative Directional Indicator (-DI)."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize -DI.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"DI_MINUS_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate -DI."""
        self._validate_ohlcv(df)

        # Calculate directional movements
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()

        # Negative directional movement
        neg_dm = pd.Series(0, index=df.index)
        neg_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff[
            (low_diff > high_diff) & (low_diff > 0)
        ]

        # Calculate True Range
        true_range = self._calculate_true_range(df)

        # Smooth with EMA
        neg_dm_smooth = neg_dm.ewm(span=self.window_size, adjust=False).mean()
        tr_smooth = true_range.ewm(span=self.window_size, adjust=False).mean()

        # Calculate -DI
        di_minus = 100 * neg_dm_smooth / tr_smooth

        return self._handle_nan(di_minus)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


class ADX(OHLCVIndicator):
    """Average Directional Index."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize ADX.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ADX_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX."""
        # Calculate +DI and -DI
        di_plus_indicator = DIPlus(window=self.window_size, fillna=False)
        di_minus_indicator = DIMinus(window=self.window_size, fillna=False)

        di_plus = di_plus_indicator.transform(df)
        di_minus = di_minus_indicator.transform(df)

        # Calculate DX
        di_sum = di_plus + di_minus
        di_diff = (di_plus - di_minus).abs()
        dx = 100 * di_diff / di_sum.replace(0, 1)  # Avoid division by zero

        # Calculate ADX as EMA of DX
        adx = dx.ewm(span=self.window_size, adjust=False).mean()

        return self._handle_nan(adx)
