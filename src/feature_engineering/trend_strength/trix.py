"""TRIX indicator."""

import pandas as pd

from ..base import PriceIndicator


class TRIX(PriceIndicator):
    """TRIX (Triple Exponential Average)."""
    
    def __init__(self, window: int = 14, price_col: str = "close", fillna: bool = True):
        """Initialize TRIX.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        
    @property
    def name(self) -> str:
        return f"TRIX_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate TRIX."""
        price = self._get_price(df)
        
        # Calculate triple exponential moving average
        ema1 = price.ewm(span=self.window_size, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window_size, adjust=False).mean()
        ema3 = ema2.ewm(span=self.window_size, adjust=False).mean()
        
        # Calculate percentage rate of change
        trix = 10000 * ema3.pct_change()
        
        return self._handle_nan(trix)