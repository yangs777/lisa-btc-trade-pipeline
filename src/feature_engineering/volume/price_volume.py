"""Price-volume combination indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class MFI(OHLCVIndicator):
    """Money Flow Index."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize MFI.
        
        Args:
            window: Period for MFI calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"MFI_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MFI."""
        self._validate_ohlcv(df)

        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate raw money flow
        raw_money_flow = typical_price * df['volume']

        # Determine positive and negative money flow
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)

        # Compare typical prices to determine flow direction
        tp_diff = typical_price.diff()

        positive_mask = tp_diff > 0
        negative_mask = tp_diff < 0

        positive_flow[positive_mask] = raw_money_flow[positive_mask]
        negative_flow[negative_mask] = raw_money_flow[negative_mask]

        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=self.window_size).sum()
        negative_mf = negative_flow.rolling(window=self.window_size).sum()

        # Avoid division by zero
        mf_ratio = positive_mf / negative_mf.replace(0, 0.0001)

        # Calculate MFI
        mfi = 100 - (100 / (1 + mf_ratio))

        return self._handle_nan(mfi)


class VWAP(OHLCVIndicator):
    """Volume Weighted Average Price."""

    def __init__(self, fillna: bool = True):
        """Initialize VWAP.
        
        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "VWAP"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP.
        
        Note: This is a cumulative VWAP from the start of the data.
        For intraday VWAP that resets daily, additional date handling would be needed.
        """
        self._validate_ohlcv(df)

        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate cumulative values
        cumulative_pv = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()

        # Calculate VWAP
        vwap = cumulative_pv / cumulative_volume

        return self._handle_nan(vwap)


class VWMA(OHLCVIndicator):
    """Volume Weighted Moving Average."""

    def __init__(self, window: int = 20, fillna: bool = True):
        """Initialize VWMA.
        
        Args:
            window: Period for VWMA calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"VWMA_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWMA."""
        self._validate_ohlcv(df)

        # Calculate price * volume
        pv = df['close'] * df['volume']

        # Calculate rolling sums
        pv_sum = pv.rolling(window=self.window_size).sum()
        volume_sum = df['volume'].rolling(window=self.window_size).sum()

        # Calculate VWMA
        vwma = pv_sum / volume_sum

        return self._handle_nan(vwma)
