"""Parabolic SAR indicators."""

import pandas as pd
import numpy as np

from ..base import OHLCVIndicator


class PSAR(OHLCVIndicator):
    """Parabolic Stop and Reverse."""
    
    def __init__(self, af: float = 0.02, max_af: float = 0.2, fillna: bool = True):
        """Initialize PSAR.
        
        Args:
            af: Acceleration factor
            max_af: Maximum acceleration factor
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)
        self.af_init = af
        self.max_af = max_af
        
    @property
    def name(self) -> str:
        return f"PSAR_{self.af_init}_{self.max_af}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate PSAR."""
        self._validate_ohlcv(df)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        psar = pd.Series(index=df.index, dtype=float)
        
        # Initialize
        bull = True
        sar = low[0]
        ep = high[0]
        af = self.af_init
        
        psar.iloc[0] = sar
        
        for i in range(1, len(df)):
            if bull:
                sar = sar + af * (ep - sar)
                
                if low[i] < sar:
                    bull = False
                    sar = ep
                    ep = low[i]
                    af = self.af_init
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.af_init, self.max_af)
                    
                    # Make sure SAR is not above prior period's low
                    if i >= 2:
                        sar = min(sar, low[i-1], low[i-2])
            else:
                sar = sar + af * (ep - sar)
                
                if high[i] > sar:
                    bull = True
                    sar = ep
                    ep = high[i]
                    af = self.af_init
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.af_init, self.max_af)
                    
                    # Make sure SAR is not below prior period's high
                    if i >= 2:
                        sar = max(sar, high[i-1], high[i-2])
                        
            psar.iloc[i] = sar
            
        return self._handle_nan(psar)


class PSARTrend(OHLCVIndicator):
    """Parabolic SAR Trend (1 for uptrend, -1 for downtrend)."""
    
    def __init__(self, af: float = 0.02, max_af: float = 0.2, fillna: bool = True):
        """Initialize PSAR Trend.
        
        Args:
            af: Acceleration factor
            max_af: Maximum acceleration factor
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)
        self.af_init = af
        self.max_af = max_af
        
    @property
    def name(self) -> str:
        return f"PSAR_TREND_{self.af_init}_{self.max_af}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate PSAR Trend."""
        # Get PSAR values
        psar_indicator = PSAR(af=self.af_init, max_af=self.max_af, fillna=False)
        psar = psar_indicator.transform(df)
        
        # Determine trend
        trend = pd.Series(index=df.index, dtype=float)
        trend[df['close'] > psar] = 1.0
        trend[df['close'] <= psar] = -1.0
        
        return self._handle_nan(trend)