from typing import Dict, List, Any, Optional, Union, Tuple

"""SuperTrend indicator."""

import pandas as pd

from ..base import OHLCVIndicator
from ..volatility.atr import ATR


class SuperTrend(OHLCVIndicator):
    """SuperTrend indicator."""

    def __init__(self, window: int = 10, mult: float = 3.0, fillna: bool = True):
        """Initialize SuperTrend.

        Args:
            window: ATR period
            mult: ATR multiplier
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.mult = mult

    @property
    def name(self) -> str:
        return f"SUPERTREND_{self.window_size}_{self.mult}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate SuperTrend."""
        self._validate_ohlcv(df)

        # Calculate ATR
        atr_indicator = ATR(window=self.window_size, fillna=False)
        atr = atr_indicator.transform(df)

        # Calculate basic bands
        hl_avg = (df["high"] + df["low"]) / 2
        basic_upper = hl_avg + self.mult * atr
        basic_lower = hl_avg - self.mult * atr

        # Initialize final bands
        final_upper = pd.Series(index=df.index, dtype=float)
        final_lower = pd.Series(index=df.index, dtype=float)
        supertrend = pd.Series(index=df.index, dtype=float)

        # Calculate final bands and SuperTrend
        for i in range(len(df)):
            if i == 0:
                final_upper.iloc[i] = basic_upper.iloc[i]
                final_lower.iloc[i] = basic_lower.iloc[i]
            else:
                # Upper band
                if (
                    basic_upper.iloc[i] < final_upper.iloc[i - 1]
                    or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
                ):
                    final_upper.iloc[i] = basic_upper.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i - 1]

                # Lower band
                if (
                    basic_lower.iloc[i] > final_lower.iloc[i - 1]
                    or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
                ):
                    final_lower.iloc[i] = basic_lower.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i - 1]

            # Determine SuperTrend
            if i == 0:
                if df["close"].iloc[i] <= final_upper.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                else:
                    supertrend.iloc[i] = final_lower.iloc[i]
            else:
                if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
                    if df["close"].iloc[i] <= final_upper.iloc[i]:
                        supertrend.iloc[i] = final_upper.iloc[i]
                    else:
                        supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    if df["close"].iloc[i] >= final_lower.iloc[i]:
                        supertrend.iloc[i] = final_lower.iloc[i]
                    else:
                        supertrend.iloc[i] = final_upper.iloc[i]

        return self._handle_nan(supertrend)
