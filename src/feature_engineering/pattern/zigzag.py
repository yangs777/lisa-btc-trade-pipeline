from typing import Dict, List, Any, Optional, Union, Tuple

"""ZigZag indicator."""

import numpy as np
import pandas as pd

from ..base import OHLCVIndicator


class ZigZag(OHLCVIndicator):
    """ZigZag indicator."""

    def __init__(self, pct: float = 5.0, fillna: bool = True):
        """Initialize ZigZag.

        Args:
            pct: Minimum percentage change to form a new zig or zag
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)
        self.pct = pct / 100.0  # Convert percentage to decimal

    @property
    def name(self) -> str:
        return f"ZIG_ZAG_{int(self.pct * 100)}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ZigZag."""
        self._validate_ohlcv(df)

        high = df["high"].values
        low = df["low"].values

        zigzag = pd.Series(index=df.index, dtype=float)
        zigzag[:] = np.nan

        # Initialize
        trend = 0  # 0: undefined, 1: up, -1: down
        last_pivot_idx = 0
        last_pivot_price = high[0]

        zigzag.iloc[0] = last_pivot_price

        for i in range(1, len(df)):
            if trend == 0:
                # No trend defined yet
                if high[i] >= last_pivot_price * (1 + self.pct):
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = high[i]
                    zigzag.iloc[i] = high[i]
                elif low[i] <= last_pivot_price * (1 - self.pct):
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = low[i]
                    zigzag.iloc[i] = low[i]

            elif trend == 1:
                # Uptrend
                if high[i] > last_pivot_price:
                    # New high in uptrend
                    zigzag.iloc[last_pivot_idx] = np.nan  # Remove previous pivot
                    last_pivot_idx = i
                    last_pivot_price = high[i]
                    zigzag.iloc[i] = high[i]
                elif low[i] <= last_pivot_price * (1 - self.pct):
                    # Reversal to downtrend
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = low[i]
                    zigzag.iloc[i] = low[i]

            else:  # trend == -1
                # Downtrend
                if low[i] < last_pivot_price:
                    # New low in downtrend
                    zigzag.iloc[last_pivot_idx] = np.nan  # Remove previous pivot
                    last_pivot_idx = i
                    last_pivot_price = low[i]
                    zigzag.iloc[i] = low[i]
                elif high[i] >= last_pivot_price * (1 + self.pct):
                    # Reversal to uptrend
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = high[i]
                    zigzag.iloc[i] = high[i]

        # Forward fill to connect zigzag points
        zigzag = zigzag.fillna(method="ffill")

        return self._handle_nan(zigzag)
