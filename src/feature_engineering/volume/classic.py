"""Classic volume indicators."""

import pandas as pd

from ..base import OHLCVIndicator, VolumeIndicator


class OBV(VolumeIndicator):
    """On Balance Volume."""

    def __init__(self, fillna: bool = True):
        """Initialize OBV.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "OBV"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate OBV."""
        volume = self._get_volume(df)

        if "close" not in df.columns:
            raise ValueError("OBV requires 'close' column")

        # Calculate price direction
        price_diff = df["close"].diff()

        # Calculate OBV
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(df)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return self._handle_nan(obv)


class AD(OHLCVIndicator):
    """Accumulation/Distribution Index."""

    def __init__(self, fillna: bool = True):
        """Initialize A/D.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "AD"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate A/D."""
        self._validate_ohlcv(df)

        # Calculate CLV (Close Location Value)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        clv = clv.fillna(0)  # Handle division by zero

        # Calculate A/D
        ad = (clv * df["volume"]).cumsum()

        return self._handle_nan(ad)


class ADL(OHLCVIndicator):
    """Accumulation/Distribution Line (same as AD)."""

    def __init__(self, fillna: bool = True):
        """Initialize ADL.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "ADL"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADL (same as AD)."""
        # ADL is the same as AD
        ad_indicator = AD(fillna=self.fillna)
        return ad_indicator.transform(df)


class CMF(OHLCVIndicator):
    """Chaikin Money Flow."""

    def __init__(self, window: int = 20, fillna: bool = True):
        """Initialize CMF.

        Args:
            window: Period for CMF calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"CMF_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate CMF."""
        self._validate_ohlcv(df)

        # Calculate Money Flow Multiplier
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume
        mfv = mfm * df["volume"]

        # Calculate CMF
        cmf = (
            mfv.rolling(window=self.window_size).sum()
            / df["volume"].rolling(window=self.window_size).sum()
        )

        return self._handle_nan(cmf)


class EMV(OHLCVIndicator):
    """Ease of Movement."""

    def __init__(self, window: int = 14, fillna: bool = True):
        """Initialize EMV.

        Args:
            window: Period for EMV calculation
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"EMV_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate EMV."""
        self._validate_ohlcv(df)

        # Distance Moved
        dm = ((df["high"] + df["low"]) / 2) - ((df["high"].shift(1) + df["low"].shift(1)) / 2)

        # Box Ratio
        box_ratio = (df["volume"] / 1000000) / (df["high"] - df["low"])

        # EMV
        emv = dm / box_ratio

        # Smooth with SMA
        emv_smooth = emv.rolling(window=self.window_size).mean()

        return self._handle_nan(emv_smooth)


class ForceIndex(OHLCVIndicator):
    """Force Index."""

    def __init__(self, window: int = 13, fillna: bool = True):
        """Initialize Force Index.

        Args:
            window: Period for EMA smoothing
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=window, fillna=fillna)

    @property
    def name(self) -> str:
        return f"FI_{self.window_size}"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Force Index."""
        self._validate_ohlcv(df)

        # Raw Force Index = (Close - Previous Close) * Volume
        raw_fi = df["close"].diff() * df["volume"]

        # Smooth with EMA
        fi = raw_fi.ewm(span=self.window_size, adjust=False).mean()

        return self._handle_nan(fi)


class NVI(VolumeIndicator):
    """Negative Volume Index."""

    def __init__(self, fillna: bool = True):
        """Initialize NVI.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "NVI"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate NVI."""
        volume = self._get_volume(df)

        if "close" not in df.columns:
            raise ValueError("NVI requires 'close' column")

        # Calculate returns
        returns = df["close"].pct_change()

        # Initialize NVI
        nvi = pd.Series(index=df.index, dtype=float)
        nvi.iloc[0] = 1000  # Starting value

        for i in range(1, len(df)):
            if volume.iloc[i] < volume.iloc[i - 1]:
                # Volume decreased, update NVI
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + returns.iloc[i])
            else:
                # Volume increased or stayed same, NVI unchanged
                nvi.iloc[i] = nvi.iloc[i - 1]

        return self._handle_nan(nvi)


class PVI(VolumeIndicator):
    """Positive Volume Index."""

    def __init__(self, fillna: bool = True):
        """Initialize PVI.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "PVI"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate PVI."""
        volume = self._get_volume(df)

        if "close" not in df.columns:
            raise ValueError("PVI requires 'close' column")

        # Calculate returns
        returns = df["close"].pct_change()

        # Initialize PVI
        pvi = pd.Series(index=df.index, dtype=float)
        pvi.iloc[0] = 1000  # Starting value

        for i in range(1, len(df)):
            if volume.iloc[i] > volume.iloc[i - 1]:
                # Volume increased, update PVI
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + returns.iloc[i])
            else:
                # Volume decreased or stayed same, PVI unchanged
                pvi.iloc[i] = pvi.iloc[i - 1]

        return self._handle_nan(pvi)


class VPT(OHLCVIndicator):
    """Volume Price Trend."""

    def __init__(self, fillna: bool = True):
        """Initialize VPT.

        Args:
            fillna: Whether to fill NaN values
        """
        super().__init__(window_size=1, fillna=fillna)

    @property
    def name(self) -> str:
        return "VPT"

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VPT."""
        self._validate_ohlcv(df)

        # Calculate price change percentage
        price_change_pct = df["close"].pct_change()

        # VPT = Previous VPT + Volume * Price Change %
        vpt = (df["volume"] * price_change_pct).cumsum()

        return self._handle_nan(vpt)
