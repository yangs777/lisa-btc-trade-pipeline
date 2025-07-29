"""Ichimoku Cloud indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class IchimokuTenkan(OHLCVIndicator):
    """Ichimoku Tenkan-sen (Conversion Line)."""

    def __init__(self, window: int = 9, fillna: bool = True):
        """Initialize Ichimoku Tenkan.
        
        Args:
            window: Period for calculation (default 9)
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ICHIMOKU_TENKAN_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Tenkan-sen."""
        self._validate_ohlcv(df)

        # Tenkan-sen = (Highest High + Lowest Low) / 2
        high_max = df['high'].rolling(window=self.window_size).max()
        low_min = df['low'].rolling(window=self.window_size).min()
        tenkan = (high_max + low_min) / 2

        return self._handle_nan(tenkan)


class IchimokuKijun(OHLCVIndicator):
    """Ichimoku Kijun-sen (Base Line)."""

    def __init__(self, window: int = 26, fillna: bool = True):
        """Initialize Ichimoku Kijun.
        
        Args:
            window: Period for calculation (default 26)
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ICHIMOKU_KIJUN_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Kijun-sen."""
        self._validate_ohlcv(df)

        # Kijun-sen = (Highest High + Lowest Low) / 2
        high_max = df['high'].rolling(window=self.window_size).max()
        low_min = df['low'].rolling(window=self.window_size).min()
        kijun = (high_max + low_min) / 2

        return self._handle_nan(kijun)


class IchimokuSenkouA(OHLCVIndicator):
    """Ichimoku Senkou Span A (Leading Span A)."""

    def __init__(self, tenkan: int = 9, kijun: int = 26, fillna: bool = True):
        """Initialize Ichimoku Senkou A.
        
        Args:
            tenkan: Tenkan-sen period (default 9)
            kijun: Kijun-sen period (default 26)
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=kijun, fillna=fillna)
        self.tenkan_period = tenkan
        self.kijun_period = kijun

    @property
    def name(self) -> str:
        return f"ICHIMOKU_SENKOU_A_{self.tenkan_period}_{self.kijun_period}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Senkou Span A."""
        self._validate_ohlcv(df)

        # Calculate Tenkan-sen
        high_max_tenkan = df['high'].rolling(window=self.tenkan_period).max()
        low_min_tenkan = df['low'].rolling(window=self.tenkan_period).min()
        tenkan = (high_max_tenkan + low_min_tenkan) / 2

        # Calculate Kijun-sen
        high_max_kijun = df['high'].rolling(window=self.kijun_period).max()
        low_min_kijun = df['low'].rolling(window=self.kijun_period).min()
        kijun = (high_max_kijun + low_min_kijun) / 2

        # Senkou Span A = (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
        senkou_a = (tenkan + kijun) / 2
        senkou_a = senkou_a.shift(26)

        return self._handle_nan(senkou_a)


class IchimokuSenkouB(OHLCVIndicator):
    """Ichimoku Senkou Span B (Leading Span B)."""

    def __init__(self, window: int = 52, fillna: bool = True):
        """Initialize Ichimoku Senkou B.
        
        Args:
            window: Period for calculation (default 52)
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ICHIMOKU_SENKOU_B_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Senkou Span B."""
        self._validate_ohlcv(df)

        # Senkou Span B = (Highest High + Lowest Low) / 2, shifted 26 periods ahead
        high_max = df['high'].rolling(window=self.window_size).max()
        low_min = df['low'].rolling(window=self.window_size).min()
        senkou_b = (high_max + low_min) / 2
        senkou_b = senkou_b.shift(26)

        return self._handle_nan(senkou_b)
