"""Regression-based statistical indicators."""

import numpy as np
import pandas as pd
from scipy import stats

from ..base import PriceIndicator


class Correlation(PriceIndicator):
    """Rolling Correlation."""
    
    def __init__(self, window: int = 20, price_col: str = "close", 
                 benchmark_col: str = "volume", fillna: bool = True):
        """Initialize Correlation.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            benchmark_col: Column to correlate with
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.benchmark_col = benchmark_col
        
    @property
    def name(self) -> str:
        return f"CORREL_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Correlation."""
        price = self._get_price(df)
        
        if self.benchmark_col not in df.columns:
            raise ValueError(f"Column '{self.benchmark_col}' not found in DataFrame")
            
        benchmark = df[self.benchmark_col]
        correlation = price.rolling(window=self.window_size).corr(benchmark)
        
        return self._handle_nan(correlation)


class Beta(PriceIndicator):
    """Beta coefficient."""
    
    def __init__(self, window: int = 20, price_col: str = "close", 
                 market_col: str = "close", fillna: bool = True):
        """Initialize Beta.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            market_col: Market returns column
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.market_col = market_col
        
    @property
    def name(self) -> str:
        return f"BETA_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Beta."""
        price = self._get_price(df)
        
        # Calculate returns
        asset_returns = price.pct_change()
        market_returns = df[self.market_col].pct_change()
        
        # Calculate rolling beta
        covariance = asset_returns.rolling(window=self.window_size).cov(market_returns)
        market_variance = market_returns.rolling(window=self.window_size).var()
        
        beta = covariance / market_variance
        
        return self._handle_nan(beta)


class LinearReg(PriceIndicator):
    """Linear Regression."""
    
    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Linear Regression.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        
    @property
    def name(self) -> str:
        return f"LINEARREG_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Linear Regression (end point of regression line)."""
        price = self._get_price(df)
        
        def linear_regression_value(y: np.ndarray) -> float:
            if len(y) < 2:
                return float('nan')
            x = np.arange(len(y))
            slope, intercept, _, _, _ = stats.linregress(x, y)
            return float(slope * (len(y) - 1) + intercept)
            
        linreg = price.rolling(window=self.window_size).apply(linear_regression_value, raw=True)
        
        return self._handle_nan(linreg)


class LinearRegSlope(PriceIndicator):
    """Linear Regression Slope."""
    
    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Linear Regression Slope.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        
    @property
    def name(self) -> str:
        return f"LINEARREG_SLOPE_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Linear Regression Slope."""
        price = self._get_price(df)
        
        def calculate_slope(y: np.ndarray) -> float:
            if len(y) < 2:
                return float('nan')
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            return float(slope)
            
        slope = price.rolling(window=self.window_size).apply(calculate_slope, raw=True)
        
        return self._handle_nan(slope)


class LinearRegAngle(PriceIndicator):
    """Linear Regression Angle."""
    
    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Linear Regression Angle.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        
    @property
    def name(self) -> str:
        return f"LINEARREG_ANGLE_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Linear Regression Angle in degrees."""
        # First get the slope
        slope_indicator = LinearRegSlope(window=self.window_size, price_col=self.price_col, fillna=False)
        slope = slope_indicator.transform(df)
        
        # Convert slope to angle in degrees
        angle = np.arctan(slope) * 180 / np.pi
        
        return self._handle_nan(angle)


class TSF(PriceIndicator):
    """Time Series Forecast."""
    
    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize TSF.
        
        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        
    @property
    def name(self) -> str:
        return f"TSF_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Time Series Forecast (projects one period ahead)."""
        price = self._get_price(df)
        
        def forecast_next(y: np.ndarray) -> float:
            if len(y) < 2:
                return float('nan')
            x = np.arange(len(y))
            slope, intercept, _, _, _ = stats.linregress(x, y)
            # Project one period ahead
            return float(slope * len(y) + intercept)
            
        tsf = price.rolling(window=self.window_size).apply(forecast_next, raw=True)
        
        return self._handle_nan(tsf)