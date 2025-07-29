"""Main feature engineering class."""

import logging
from pathlib import Path

import pandas as pd

# Import all indicator categories to register them
from . import momentum, pattern, statistical, trend, trend_strength, volatility, volume
from .registry import registry

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main class for feature engineering."""

    def __init__(self, config_path: str | None = None):
        """Initialize Feature Engineer.

        Args:
            config_path: Path to indicators YAML config file
        """
        self.config_path = config_path or str(Path(__file__).parent.parent.parent / "indicators.yaml")
        self._register_all_indicators()

        # Load configuration
        registry.load_config(self.config_path)

        # Create all indicators
        self.indicators = registry.create_all_indicators()
        logger.info(f"Initialized {len(self.indicators)} indicators")

    def _register_all_indicators(self) -> None:
        """Register all indicator classes with the registry."""
        # Trend indicators
        registry.register("SMA", trend.SMA)
        registry.register("EMA", trend.EMA)
        registry.register("WMA", trend.WMA)
        registry.register("HMA", trend.HMA)
        registry.register("TEMA", trend.TEMA)
        registry.register("DEMA", trend.DEMA)
        registry.register("KAMA", trend.KAMA)
        registry.register("IchimokuTenkan", trend.IchimokuTenkan)
        registry.register("IchimokuKijun", trend.IchimokuKijun)
        registry.register("IchimokuSenkouA", trend.IchimokuSenkouA)
        registry.register("IchimokuSenkouB", trend.IchimokuSenkouB)

        # Momentum indicators
        registry.register("RSI", momentum.RSI)
        registry.register("StochasticK", momentum.StochasticK)
        registry.register("StochasticD", momentum.StochasticD)
        registry.register("StochRSIK", momentum.StochRSIK)
        registry.register("StochRSID", momentum.StochRSID)
        registry.register("MACD", momentum.MACD)
        registry.register("MACDSignal", momentum.MACDSignal)
        registry.register("MACDHist", momentum.MACDHist)
        registry.register("CCI", momentum.CCI)
        registry.register("WilliamsR", momentum.WilliamsR)
        registry.register("ROC", momentum.ROC)
        registry.register("Momentum", momentum.Momentum)
        registry.register("TSI", momentum.TSI)
        registry.register("UltimateOscillator", momentum.UltimateOscillator)
        registry.register("AwesomeOscillator", momentum.AwesomeOscillator)

        # Volatility indicators
        registry.register("ATR", volatility.ATR)
        registry.register("NATR", volatility.NATR)
        registry.register("BollingerUpper", volatility.BollingerUpper)
        registry.register("BollingerMiddle", volatility.BollingerMiddle)
        registry.register("BollingerLower", volatility.BollingerLower)
        registry.register("BollingerWidth", volatility.BollingerWidth)
        registry.register("BollingerPercent", volatility.BollingerPercent)
        registry.register("KeltnerUpper", volatility.KeltnerUpper)
        registry.register("KeltnerMiddle", volatility.KeltnerMiddle)
        registry.register("KeltnerLower", volatility.KeltnerLower)
        registry.register("DonchianUpper", volatility.DonchianUpper)
        registry.register("DonchianLower", volatility.DonchianLower)
        registry.register("UlcerIndex", volatility.UlcerIndex)
        registry.register("MassIndex", volatility.MassIndex)

        # Volume indicators
        registry.register("OBV", volume.OBV)
        registry.register("AD", volume.AD)
        registry.register("ADL", volume.ADL)
        registry.register("CMF", volume.CMF)
        registry.register("EMV", volume.EMV)
        registry.register("ForceIndex", volume.ForceIndex)
        registry.register("MFI", volume.MFI)
        registry.register("NVI", volume.NVI)
        registry.register("PVI", volume.PVI)
        registry.register("VWAP", volume.VWAP)
        registry.register("VPT", volume.VPT)
        registry.register("VWMA", volume.VWMA)

        # Trend strength indicators
        registry.register("ADX", trend_strength.ADX)
        registry.register("DIPlus", trend_strength.DIPlus)
        registry.register("DIMinus", trend_strength.DIMinus)
        registry.register("AroonUp", trend_strength.AroonUp)
        registry.register("AroonDown", trend_strength.AroonDown)
        registry.register("AroonOsc", trend_strength.AroonOsc)
        registry.register("VortexPlus", trend_strength.VortexPlus)
        registry.register("VortexMinus", trend_strength.VortexMinus)
        registry.register("TRIX", trend_strength.TRIX)

        # Pattern indicators
        registry.register("PSAR", pattern.PSAR)
        registry.register("PSARTrend", pattern.PSARTrend)
        registry.register("SuperTrend", pattern.SuperTrend)
        registry.register("ZigZag", pattern.ZigZag)
        registry.register("PivotHigh", pattern.PivotHigh)
        registry.register("PivotLow", pattern.PivotLow)

        # Statistical indicators
        registry.register("StdDev", statistical.StdDev)
        registry.register("Variance", statistical.Variance)
        registry.register("SEM", statistical.SEM)
        registry.register("Skew", statistical.Skew)
        registry.register("Kurtosis", statistical.Kurtosis)
        registry.register("Correlation", statistical.Correlation)
        registry.register("Beta", statistical.Beta)
        registry.register("LinearReg", statistical.LinearReg)
        registry.register("LinearRegSlope", statistical.LinearRegSlope)
        registry.register("LinearRegAngle", statistical.LinearRegAngle)
        registry.register("TSF", statistical.TSF)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all indicators to the dataframe.

        Args:
            df: Input dataframe with OHLCV columns

        Returns:
            DataFrame with all indicator features added
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Apply each indicator
        for name, indicator in self.indicators.items():
            try:
                logger.debug(f"Calculating {name}")
                result_df[name] = indicator.transform(df)
            except Exception as e:
                logger.error(f"Failed to calculate {name}: {e}")
                # Fill with NaN if calculation fails
                result_df[name] = pd.NA

        logger.info(f"Calculated {len(self.indicators)} indicators")
        return result_df

    def transform_selective(self, df: pd.DataFrame, indicator_names: list[str]) -> pd.DataFrame:
        """Apply only selected indicators.

        Args:
            df: Input dataframe with OHLCV columns
            indicator_names: List of indicator names to calculate

        Returns:
            DataFrame with selected indicator features added
        """
        result_df = df.copy()

        for name in indicator_names:
            if name not in self.indicators:
                logger.warning(f"Indicator {name} not found")
                continue

            try:
                logger.debug(f"Calculating {name}")
                result_df[name] = self.indicators[name].transform(df)
            except Exception as e:
                logger.error(f"Failed to calculate {name}: {e}")
                result_df[name] = pd.NA

        return result_df

    def get_indicator_info(self) -> dict[str, dict]:
        """Get information about all indicators.

        Returns:
            Dictionary with indicator information
        """
        info = {}

        for name, indicator in self.indicators.items():
            info[name] = {
                "class": indicator.__class__.__name__,
                "module": indicator.__class__.__module__,
                "window_size": getattr(indicator, "window_size", None),
            }

        return info
