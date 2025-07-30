"""Aroon indicators."""

import pandas as pd

from ..base import OHLCVIndicator


class AroonUp(OHLCVIndicator):
    """Aroon Up indicator."""

    def __init__(self, window: int = 25, fillna: bool = True):
        """Initialize Aroon Up.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"AROON_UP_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Aroon Up."""
        self._validate_ohlcv(df)

        # Calculate periods since highest high
        aroon_up = pd.Series(index=df.index, dtype=float)

        for i in range(self.window_size - 1, len(df)):
            window_data = df["high"].iloc[i - self.window_size + 1 : i + 1]
            periods_since_high = self.window_size - 1 - window_data.values.argmax()
            aroon_up.iloc[i] = ((self.window_size - periods_since_high) / self.window_size) * 100

        return self._handle_nan(aroon_up)


class AroonDown(OHLCVIndicator):
    """Aroon Down indicator."""

    def __init__(self, window: int = 25, fillna: bool = True):
        """Initialize Aroon Down.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"AROON_DOWN_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Aroon Down."""
        self._validate_ohlcv(df)

        # Calculate periods since lowest low
        aroon_down = pd.Series(index=df.index, dtype=float)

        for i in range(self.window_size - 1, len(df)):
            window_data = df["low"].iloc[i - self.window_size + 1 : i + 1]
            periods_since_low = self.window_size - 1 - window_data.values.argmin()
            aroon_down.iloc[i] = ((self.window_size - periods_since_low) / self.window_size) * 100

        return self._handle_nan(aroon_down)


class AroonOsc(OHLCVIndicator):
    """Aroon Oscillator."""

    def __init__(self, window: int = 25, fillna: bool = True):
        """Initialize Aroon Oscillator.

        Args:
            window: Period for calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"AROON_OSC_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Aroon Oscillator."""
        # Calculate Aroon Up and Down
        aroon_up_indicator = AroonUp(window=self.window_size, fillna=False)
        aroon_down_indicator = AroonDown(window=self.window_size, fillna=False)

        aroon_up = aroon_up_indicator.transform(df)
        aroon_down = aroon_down_indicator.transform(df)

        # Aroon Oscillator = Aroon Up - Aroon Down
        aroon_osc = aroon_up - aroon_down

        return self._handle_nan(aroon_osc)
