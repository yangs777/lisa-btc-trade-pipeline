"""Momentum oscillator indicators."""

import numpy as np
import pandas as pd

from ..base import OHLCVIndicator, PriceIndicator


class RSI(PriceIndicator):
    """Relative Strength Index."""

    def __init__(self, window: int = 14, price_col: str = "close", fillna: bool = True):
        """Initialize RSI.
        
        Args:
            window: Period for RSI calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"RSI_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI."""
        price = self._get_price(df)

        # Calculate price changes
        delta = price.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=self.window_size).mean()
        avg_losses = losses.rolling(window=self.window_size).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return self._handle_nan(rsi)


class StochasticK(OHLCVIndicator):
    """Stochastic Oscillator %K."""

    def __init__(self, window: int = 14, smooth: int = 3, fillna: bool = True):
        """Initialize Stochastic K.
        
        Args:
            window: Period for calculation
            smooth: Smoothing period for %K
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.smooth = smooth

    @property
    def name(self) -> str:
        return f"STOCH_K_{self.window_size}_{self.smooth}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic %K."""
        self._validate_ohlcv(df)

        # Calculate highest high and lowest low
        high_max = df['high'].rolling(window=self.window_size).max()
        low_min = df['low'].rolling(window=self.window_size).min()

        # Calculate %K
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))

        # Smooth %K
        k_smooth = k_percent.rolling(window=self.smooth).mean()

        return self._handle_nan(k_smooth)


class StochasticD(OHLCVIndicator):
    """Stochastic Oscillator %D."""

    def __init__(self, window: int = 14, smooth_k: int = 3, smooth_d: int = 3, fillna: bool = True):
        """Initialize Stochastic D.
        
        Args:
            window: Period for calculation
            smooth_k: Smoothing period for %K
            smooth_d: Smoothing period for %D
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d

    @property
    def name(self) -> str:
        return f"STOCH_D_{self.window_size}_{self.smooth_k}_{self.smooth_d}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic %D."""
        # First calculate %K
        stoch_k = StochasticK(window=self.window_size, smooth=self.smooth_k, fillna=False)
        k_values = stoch_k.transform(df)

        # %D is smoothed %K
        d_values = k_values.rolling(window=self.smooth_d).mean()

        return self._handle_nan(d_values)


class StochRSIK(PriceIndicator):
    """Stochastic RSI %K."""

    def __init__(self, window: int = 14, smooth: int = 3, price_col: str = "close", fillna: bool = True):
        """Initialize StochRSI K.
        
        Args:
            window: Period for RSI and Stochastic
            smooth: Smoothing period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.smooth = smooth

    @property
    def name(self) -> str:
        return f"STOCHRSI_K_{self.window_size}_{self.smooth}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate StochRSI %K."""
        # First calculate RSI
        rsi_indicator = RSI(window=self.window_size, price_col=self.price_col, fillna=False)
        rsi = rsi_indicator.transform(df)

        # Apply Stochastic to RSI
        rsi_max = rsi.rolling(window=self.window_size).max()
        rsi_min = rsi.rolling(window=self.window_size).min()

        stoch_rsi_k = 100 * ((rsi - rsi_min) / (rsi_max - rsi_min))
        stoch_rsi_k_smooth = stoch_rsi_k.rolling(window=self.smooth).mean()

        return self._handle_nan(stoch_rsi_k_smooth)


class StochRSID(PriceIndicator):
    """Stochastic RSI %D."""

    def __init__(self, window: int = 14, smooth_k: int = 3, smooth_d: int = 3,
                 price_col: str = "close", fillna: bool = True):
        """Initialize StochRSI D.
        
        Args:
            window: Period for RSI and Stochastic
            smooth_k: Smoothing period for %K
            smooth_d: Smoothing period for %D
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d

    @property
    def name(self) -> str:
        return f"STOCHRSI_D_{self.window_size}_{self.smooth_k}_{self.smooth_d}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate StochRSI %D."""
        # First calculate StochRSI %K
        stoch_rsi_k = StochRSIK(window=self.window_size, smooth=self.smooth_k,
                                price_col=self.price_col, fillna=False)
        k_values = stoch_rsi_k.transform(df)

        # %D is smoothed %K
        d_values = k_values.rolling(window=self.smooth_d).mean()

        return self._handle_nan(d_values)


class CCI(OHLCVIndicator):
    """Commodity Channel Index."""

    def __init__(self, window: int = 20, fillna: bool = True):
        """Initialize CCI.
        
        Args:
            window: Period for CCI calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"CCI_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate CCI."""
        self._validate_ohlcv(df)

        # Calculate Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate SMA of Typical Price
        sma_tp = typical_price.rolling(window=self.window_size).mean()

        # Calculate Mean Deviation
        mad = typical_price.rolling(window=self.window_size).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        # Calculate CCI
        cci = (typical_price - sma_tp) / (0.015 * mad)

        return self._handle_nan(cci)


class WilliamsR(OHLCVIndicator):
    """Williams %R."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize Williams %R.
        
        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"WILLIAMS_R_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R."""
        self._validate_ohlcv(df)

        # Calculate highest high and lowest low
        high_max = df['high'].rolling(window=self.window_size).max()
        low_min = df['low'].rolling(window=self.window_size).min()

        # Calculate Williams %R
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))

        return self._handle_nan(williams_r)


class ROC(PriceIndicator):
    """Rate of Change."""

    def __init__(self, window: int = 10, price_col: str = "close", fillna: bool = True):
        """Initialize ROC.
        
        Args:
            window: Period for ROC calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"ROC_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ROC."""
        price = self._get_price(df)

        # ROC = ((Price - Price N periods ago) / Price N periods ago) * 100
        roc = ((price - price.shift(self.window_size)) / price.shift(self.window_size)) * 100

        return self._handle_nan(roc)


class Momentum(PriceIndicator):
    """Momentum indicator."""

    def __init__(self, window: int = 10, price_col: str = "close", fillna: bool = True):
        """Initialize Momentum.
        
        Args:
            window: Period for momentum calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"MOM_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Momentum."""
        price = self._get_price(df)

        # Momentum = Price - Price N periods ago
        momentum = price - price.shift(self.window_size)

        return self._handle_nan(momentum)


class TSI(PriceIndicator):
    """True Strength Index."""

    def __init__(self, slow: int = 25, fast: int = 13, price_col: str = "close", fillna: bool = True):
        """Initialize TSI.
        
        Args:
            slow: Slow EMA period
            fast: Fast EMA period
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=slow, fillna=fillna)
        self.slow = slow
        self.fast = fast

    @property
    def name(self) -> str:
        return f"TSI_{self.slow}_{self.fast}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate TSI."""
        price = self._get_price(df)

        # Calculate momentum
        momentum = price.diff()

        # Double smooth momentum
        momentum_slow = momentum.ewm(span=self.slow, adjust=False).mean()
        momentum_fast = momentum_slow.ewm(span=self.fast, adjust=False).mean()

        # Double smooth absolute momentum
        abs_momentum_slow = momentum.abs().ewm(span=self.slow, adjust=False).mean()
        abs_momentum_fast = abs_momentum_slow.ewm(span=self.fast, adjust=False).mean()

        # Calculate TSI
        tsi = 100 * (momentum_fast / abs_momentum_fast)

        return self._handle_nan(tsi)


class UltimateOscillator(OHLCVIndicator):
    """Ultimate Oscillator."""

    def __init__(self, fast: int = 7, medium: int = 14, slow: int = 28, fillna: bool = True):
        """Initialize Ultimate Oscillator.
        
        Args:
            fast: Fast period
            medium: Medium period
            slow: Slow period
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=slow, fillna=fillna)
        self.fast = fast
        self.medium = medium
        self.slow = slow

    @property
    def name(self) -> str:
        return f"UO_{self.fast}_{self.medium}_{self.slow}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator."""
        self._validate_ohlcv(df)

        # Calculate True Low and Buying Pressure
        true_low = df['low'].combine(df['close'].shift(1), min)
        buying_pressure = df['close'] - true_low

        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate averages for each period
        bp_fast = buying_pressure.rolling(window=self.fast).sum()
        tr_fast = true_range.rolling(window=self.fast).sum()
        avg_fast = bp_fast / tr_fast

        bp_medium = buying_pressure.rolling(window=self.medium).sum()
        tr_medium = true_range.rolling(window=self.medium).sum()
        avg_medium = bp_medium / tr_medium

        bp_slow = buying_pressure.rolling(window=self.slow).sum()
        tr_slow = true_range.rolling(window=self.slow).sum()
        avg_slow = bp_slow / tr_slow

        # Calculate Ultimate Oscillator
        uo = 100 * ((4 * avg_fast + 2 * avg_medium + avg_slow) / 7)

        return self._handle_nan(uo)


class AwesomeOscillator(OHLCVIndicator):
    """Awesome Oscillator."""

    def __init__(self, fast: int = 5, slow: int = 34, fillna: bool = True):
        """Initialize Awesome Oscillator.
        
        Args:
            fast: Fast SMA period
            slow: Slow SMA period
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=slow, fillna=fillna)
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"AO_{self.fast}_{self.slow}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Awesome Oscillator."""
        self._validate_ohlcv(df)

        # Calculate median price
        median_price = (df['high'] + df['low']) / 2

        # Calculate SMAs
        sma_fast = median_price.rolling(window=self.fast).mean()
        sma_slow = median_price.rolling(window=self.slow).mean()

        # Calculate AO
        ao = sma_fast - sma_slow

        return self._handle_nan(ao)
