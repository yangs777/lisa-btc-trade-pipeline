"""Pivot point indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class PivotHigh(OHLCVIndicator):
    """Pivot High detector."""

    def __init__(self, window: int = 5, fillna: bool = True):
        """Initialize Pivot High.

        Args:
            window: Number of bars on each side to check
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"PIVOT_HIGH_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Detect Pivot Highs."""
        self._validate_ohlcv(df)

        pivot_high = pd.Series(index=df.index, dtype=float)
        pivot_high[:] = 0.0

        # Look for pivot highs
        for i in range(self.window_size, len(df) - self.window_size):
            is_pivot = True
            current_high = df['high'].iloc[i]

            # Check left side
            for j in range(i - self.window_size, i):
                if df['high'].iloc[j] >= current_high:
                    is_pivot = False
                    break

            # Check right side
            if is_pivot:
                for j in range(i + 1, i + self.window_size + 1):
                    if df['high'].iloc[j] > current_high:
                        is_pivot = False
                        break

            if is_pivot:
                pivot_high.iloc[i] = current_high

        return self._handle_nan(pivot_high)


class PivotLow(OHLCVIndicator):
    """Pivot Low detector."""

    def __init__(self, window: int = 5, fillna: bool = True):
        """Initialize Pivot Low.

        Args:
            window: Number of bars on each side to check
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"PIVOT_LOW_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Detect Pivot Lows."""
        self._validate_ohlcv(df)

        pivot_low = pd.Series(index=df.index, dtype=float)
        pivot_low[:] = 0.0

        # Look for pivot lows
        for i in range(self.window_size, len(df) - self.window_size):
            is_pivot = True
            current_low = df['low'].iloc[i]

            # Check left side
            for j in range(i - self.window_size, i):
                if df['low'].iloc[j] <= current_low:
                    is_pivot = False
                    break

            # Check right side
            if is_pivot:
                for j in range(i + 1, i + self.window_size + 1):
                    if df['low'].iloc[j] < current_low:
                        is_pivot = False
                        break

            if is_pivot:
                pivot_low.iloc[i] = current_low

        return self._handle_nan(pivot_low)
