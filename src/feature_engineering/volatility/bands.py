"""Band-based volatility indicators."""

import pandas as pd

from ..base import OHLCVIndicator, PriceIndicator
from .atr import ATR


class BollingerMiddle(PriceIndicator):
    """Bollinger Band Middle (SMA)."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Bollinger Middle.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"BB_MIDDLE_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Middle."""
        price = self._get_price(df)
        middle = price.rolling(window=self.window_size).mean()
        return self._handle_nan(middle)


class BollingerUpper(PriceIndicator):
    """Bollinger Band Upper."""

    def __init__(self, window: int = 20, std: float = 2.0, price_col: str = "close", fillna: bool = True):
        """Initialize Bollinger Upper.

        Args:
            window: Period for calculation
            std: Number of standard deviations
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.std = std

    @property
    def name(self) -> str:
        return f"BB_UPPER_{self.window_size}_{self.std}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Upper."""
        price = self._get_price(df)
        middle = price.rolling(window=self.window_size).mean()
        std_dev = price.rolling(window=self.window_size).std()
        upper = middle + (std_dev * self.std)
        return self._handle_nan(upper)


class BollingerLower(PriceIndicator):
    """Bollinger Band Lower."""

    def __init__(self, window: int = 20, std: float = 2.0, price_col: str = "close", fillna: bool = True):
        """Initialize Bollinger Lower.

        Args:
            window: Period for calculation
            std: Number of standard deviations
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.std = std

    @property
    def name(self) -> str:
        return f"BB_LOWER_{self.window_size}_{self.std}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Lower."""
        price = self._get_price(df)
        middle = price.rolling(window=self.window_size).mean()
        std_dev = price.rolling(window=self.window_size).std()
        lower = middle - (std_dev * self.std)
        return self._handle_nan(lower)


class BollingerWidth(PriceIndicator):
    """Bollinger Band Width."""

    def __init__(self, window: int = 20, std: float = 2.0, price_col: str = "close", fillna: bool = True):
        """Initialize Bollinger Width.

        Args:
            window: Period for calculation
            std: Number of standard deviations
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.std = std

    @property
    def name(self) -> str:
        return f"BB_WIDTH_{self.window_size}_{self.std}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Width."""
        # Calculate upper and lower bands
        upper_indicator = BollingerUpper(window=self.window_size, std=self.std,
                                       price_col=self.price_col, fillna=False)
        lower_indicator = BollingerLower(window=self.window_size, std=self.std,
                                       price_col=self.price_col, fillna=False)

        upper = upper_indicator.transform(df)
        lower = lower_indicator.transform(df)

        # Width = Upper - Lower
        width = upper - lower
        return self._handle_nan(width)


class BollingerPercent(PriceIndicator):
    """Bollinger Band %B."""

    def __init__(self, window: int = 20, std: float = 2.0, price_col: str = "close", fillna: bool = True):
        """Initialize Bollinger %B.

        Args:
            window: Period for calculation
            std: Number of standard deviations
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)
        self.std = std

    @property
    def name(self) -> str:
        return f"BB_PERCENT_{self.window_size}_{self.std}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger %B."""
        price = self._get_price(df)

        # Calculate bands
        upper_indicator = BollingerUpper(window=self.window_size, std=self.std,
                                       price_col=self.price_col, fillna=False)
        lower_indicator = BollingerLower(window=self.window_size, std=self.std,
                                       price_col=self.price_col, fillna=False)

        upper = upper_indicator.transform(df)
        lower = lower_indicator.transform(df)

        # %B = (Price - Lower) / (Upper - Lower)
        percent_b = (price - lower) / (upper - lower)
        return self._handle_nan(percent_b)


class KeltnerMiddle(PriceIndicator):
    """Keltner Channel Middle (EMA)."""

    def __init__(self, window: int = 20, price_col: str = "close", fillna: bool = True):
        """Initialize Keltner Middle.

        Args:
            window: Period for calculation
            price_col: Price column to use
            fillna: Whether to fill NaN values
        """
        super().__init__(price_col=price_col, window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"KC_MIDDLE_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Keltner Middle."""
        price = self._get_price(df)
        middle = price.ewm(span=self.window_size, adjust=False).mean()
        return self._handle_nan(middle)


class KeltnerUpper(OHLCVIndicator):
    """Keltner Channel Upper."""

    def __init__(self, window: int = 20, atr_window: int = 10, mult: float = 2.0, fillna: bool = True):
        """Initialize Keltner Upper.

        Args:
            window: Period for EMA calculation
            atr_window: Period for ATR calculation
            mult: ATR multiplier
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.atr_window = atr_window
        self.mult = mult

    @property
    def name(self) -> str:
        return f"KC_UPPER_{self.window_size}_{self.atr_window}_{self.mult}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Keltner Upper."""
        self._validate_ohlcv(df)

        # Calculate middle line (EMA)
        middle = df['close'].ewm(span=self.window_size, adjust=False).mean()

        # Calculate ATR
        atr_indicator = ATR(window=self.atr_window, fillna=False)
        atr = atr_indicator.transform(df)

        # Upper = Middle + (ATR * multiplier)
        upper = middle + (atr * self.mult)
        return self._handle_nan(upper)


class KeltnerLower(OHLCVIndicator):
    """Keltner Channel Lower."""

    def __init__(self, window: int = 20, atr_window: int = 10, mult: float = 2.0, fillna: bool = True):
        """Initialize Keltner Lower.

        Args:
            window: Period for EMA calculation
            atr_window: Period for ATR calculation
            mult: ATR multiplier
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)
        self.atr_window = atr_window
        self.mult = mult

    @property
    def name(self) -> str:
        return f"KC_LOWER_{self.window_size}_{self.atr_window}_{self.mult}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Keltner Lower."""
        self._validate_ohlcv(df)

        # Calculate middle line (EMA)
        middle = df['close'].ewm(span=self.window_size, adjust=False).mean()

        # Calculate ATR
        atr_indicator = ATR(window=self.atr_window, fillna=False)
        atr = atr_indicator.transform(df)

        # Lower = Middle - (ATR * multiplier)
        lower = middle - (atr * self.mult)
        return self._handle_nan(lower)


class DonchianUpper(OHLCVIndicator):
    """Donchian Channel Upper."""

    def __init__(self, window: int = 20, fillna: bool = True):
        """Initialize Donchian Upper.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"DC_UPPER_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Donchian Upper."""
        self._validate_ohlcv(df)
        upper = df['high'].rolling(window=self.window_size).max()
        return self._handle_nan(upper)


class DonchianLower(OHLCVIndicator):
    """Donchian Channel Lower."""

    def __init__(self, window: int = 20, fillna: bool = True):
        """Initialize Donchian Lower.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"DC_LOWER_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Donchian Lower."""
        self._validate_ohlcv(df)
        lower = df['low'].rolling(window=self.window_size).min()
        return self._handle_nan(lower)
