"""Comprehensive tests for feature engineering modules to boost coverage."""

import pytest
import numpy as np
import pandas as pd

# Import all feature engineering modules
from src.feature_engineering.indicators import TechnicalIndicators
from src.feature_engineering.market_microstructure.order_flow import OrderFlowImbalance
from src.feature_engineering.market_microstructure.liquidity import LiquidityFeatures
from src.feature_engineering.market_microstructure.price_levels import PriceLevelFeatures
from src.feature_engineering.momentum.basic import BasicMomentum
from src.feature_engineering.momentum.rsi import RSI
from src.feature_engineering.momentum.stochastic import Stochastic
from src.feature_engineering.sentiment.market_sentiment import MarketSentiment
from src.feature_engineering.sentiment.news_sentiment import NewsSentiment
from src.feature_engineering.statistical.basic import BasicStatistics
from src.feature_engineering.statistical.regression import RegressionFeatures
from src.feature_engineering.trend.ichimoku import IchimokuCloud
from src.feature_engineering.trend.moving_averages import MovingAverages
from src.feature_engineering.trend_strength.adx import ADX
from src.feature_engineering.trend_strength.aroon import Aroon
from src.feature_engineering.trend_strength.trix import TRIX
from src.feature_engineering.trend_strength.vortex import VortexIndicator
from src.feature_engineering.volatility.atr import ATR
from src.feature_engineering.volatility.bands import BollingerBands, KeltnerChannel, DonchianChannel
from src.feature_engineering.volatility.other import HistoricalVolatility, ChoppinessIndex
from src.feature_engineering.volume.classic import OBV, VWMA, ADL, MFI, VPT, VolumeRSI, VolumeOscillator
from src.feature_engineering.volume.price_volume import PVT, EaseOfMovement


class TestFeatureEngineeringModules:
    """Test all feature engineering modules for coverage."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        np.random.seed(42)
        
        # Generate realistic price data
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        high_prices = close_prices + np.abs(np.random.randn(100) * 50)
        low_prices = close_prices - np.abs(np.random.randn(100) * 50)
        open_prices = close_prices + np.random.randn(100) * 30
        volume = np.abs(np.random.randn(100) * 1000000) + 100000
        
        return pd.DataFrame({
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
    
    def test_technical_indicators(self, sample_data):
        """Test TechnicalIndicators class."""
        indicators = TechnicalIndicators()
        
        # Test compute_all
        features = indicators.compute_all(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert len(features.columns) > 100  # Should have many features
        
        # Test compute_core
        core_features = indicators.compute_core(sample_data)
        assert isinstance(core_features, pd.DataFrame)
        assert len(core_features.columns) < len(features.columns)
    
    def test_order_flow_imbalance(self, sample_data):
        """Test OrderFlowImbalance features."""
        ofi = OrderFlowImbalance()
        
        # Test with default parameters
        features = ofi.compute(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert "ofi" in features.columns
        
        # Test with custom periods
        features_custom = ofi.compute(sample_data, periods=[5, 10])
        assert "ofi_5" in features_custom.columns
        assert "ofi_10" in features_custom.columns
    
    def test_liquidity_features(self, sample_data):
        """Test LiquidityFeatures."""
        lf = LiquidityFeatures()
        
        features = lf.compute(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert "bid_ask_spread" in features.columns
        assert "price_impact" in features.columns
    
    def test_price_level_features(self, sample_data):
        """Test PriceLevelFeatures."""
        plf = PriceLevelFeatures()
        
        features = plf.compute(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert "resistance_distance" in features.columns
        assert "support_distance" in features.columns
    
    def test_momentum_indicators(self, sample_data):
        """Test momentum indicators."""
        # BasicMomentum
        bm = BasicMomentum()
        features = bm.compute(sample_data)
        assert "roc_10" in features.columns
        
        # RSI
        rsi = RSI()
        features = rsi.compute(sample_data)
        assert "rsi_14" in features.columns
        
        # Stochastic
        stoch = Stochastic()
        features = stoch.compute(sample_data)
        assert "stoch_k" in features.columns
        assert "stoch_d" in features.columns
    
    def test_sentiment_features(self, sample_data):
        """Test sentiment features."""
        # MarketSentiment
        ms = MarketSentiment()
        features = ms.compute(sample_data)
        assert "put_call_ratio" in features.columns
        
        # NewsSentiment - test initialization at least
        ns = NewsSentiment()
        assert ns.window_size == 24
    
    def test_statistical_features(self, sample_data):
        """Test statistical features."""
        # BasicStatistics
        bs = BasicStatistics()
        features = bs.compute(sample_data)
        assert "mean_20" in features.columns
        assert "std_20" in features.columns
        assert "skew_20" in features.columns
        assert "kurt_20" in features.columns
        
        # RegressionFeatures
        rf = RegressionFeatures()
        features = rf.compute(sample_data)
        assert "linear_slope_20" in features.columns
        assert "r_squared_20" in features.columns
    
    def test_trend_indicators(self, sample_data):
        """Test trend indicators."""
        # MovingAverages
        ma = MovingAverages()
        features = ma.compute(sample_data)
        assert "sma_20" in features.columns
        assert "ema_20" in features.columns
        
        # IchimokuCloud
        ichimoku = IchimokuCloud()
        features = ichimoku.compute(sample_data)
        assert "tenkan_sen" in features.columns
        assert "kijun_sen" in features.columns
    
    def test_trend_strength_indicators(self, sample_data):
        """Test trend strength indicators."""
        # ADX
        adx = ADX()
        features = adx.compute(sample_data)
        assert "adx_14" in features.columns
        
        # Aroon
        aroon = Aroon()
        features = aroon.compute(sample_data)
        assert "aroon_up" in features.columns
        assert "aroon_down" in features.columns
        
        # TRIX
        trix = TRIX()
        features = trix.compute(sample_data)
        assert "trix" in features.columns
        
        # VortexIndicator
        vortex = VortexIndicator()
        features = vortex.compute(sample_data)
        assert "vi_positive" in features.columns
        assert "vi_negative" in features.columns
    
    def test_volatility_indicators(self, sample_data):
        """Test volatility indicators."""
        # ATR
        atr = ATR()
        features = atr.compute(sample_data)
        assert "atr_14" in features.columns
        
        # BollingerBands
        bb = BollingerBands()
        features = bb.compute(sample_data)
        assert "bb_upper" in features.columns
        assert "bb_lower" in features.columns
        assert "bb_width" in features.columns
        
        # KeltnerChannel
        kc = KeltnerChannel()
        features = kc.compute(sample_data)
        assert "kc_upper" in features.columns
        assert "kc_lower" in features.columns
        
        # DonchianChannel
        dc = DonchianChannel()
        features = dc.compute(sample_data)
        assert "dc_upper" in features.columns
        assert "dc_lower" in features.columns
        
        # HistoricalVolatility
        hv = HistoricalVolatility()
        features = hv.compute(sample_data)
        assert "hist_vol_20" in features.columns
        
        # ChoppinessIndex
        ci = ChoppinessIndex()
        features = ci.compute(sample_data)
        assert "choppiness" in features.columns
    
    def test_volume_indicators(self, sample_data):
        """Test volume indicators."""
        # OBV
        obv = OBV()
        features = obv.compute(sample_data)
        assert "obv" in features.columns
        
        # VWMA
        vwma = VWMA()
        features = vwma.compute(sample_data)
        assert "vwma_20" in features.columns
        
        # ADL
        adl = ADL()
        features = adl.compute(sample_data)
        assert "adl" in features.columns
        
        # MFI
        mfi = MFI()
        features = mfi.compute(sample_data)
        assert "mfi_14" in features.columns
        
        # VPT
        vpt = VPT()
        features = vpt.compute(sample_data)
        assert "vpt" in features.columns
        
        # VolumeRSI
        vrsi = VolumeRSI()
        features = vrsi.compute(sample_data)
        assert "volume_rsi" in features.columns
        
        # VolumeOscillator
        vo = VolumeOscillator()
        features = vo.compute(sample_data)
        assert "volume_oscillator" in features.columns
        
        # PVT
        pvt = PVT()
        features = pvt.compute(sample_data)
        assert "pvt" in features.columns
        
        # EaseOfMovement
        eom = EaseOfMovement()
        features = eom.compute(sample_data)
        assert "eom_14" in features.columns
    
    def test_edge_cases(self, sample_data):
        """Test edge cases and error handling."""
        # Test with insufficient data
        small_data = sample_data.head(5)
        
        # Should handle gracefully
        indicators = TechnicalIndicators()
        features = indicators.compute_all(small_data)
        assert len(features) == len(small_data)
        
        # Test with missing columns
        bad_data = sample_data.drop(columns=["volume"])
        
        # Volume indicators should handle missing volume
        obv = OBV()
        features = obv.compute(bad_data)
        assert len(features) == len(bad_data)
    
    def test_custom_parameters(self, sample_data):
        """Test indicators with custom parameters."""
        # RSI with custom period
        rsi = RSI()
        features = rsi.compute(sample_data, periods=[7, 21])
        assert "rsi_7" in features.columns
        assert "rsi_21" in features.columns
        
        # Moving averages with custom periods
        ma = MovingAverages()
        features = ma.compute(sample_data, periods=[10, 50])
        assert "sma_10" in features.columns
        assert "ema_50" in features.columns
        
        # Bollinger Bands with custom parameters
        bb = BollingerBands()
        features = bb.compute(sample_data, period=10, num_std=3)
        assert "bb_upper" in features.columns
        
    def test_all_features_integration(self, sample_data):
        """Test computing all features together."""
        indicators = TechnicalIndicators()
        
        # This should exercise most code paths
        all_features = indicators.compute_all(sample_data)
        
        # Check we have a reasonable number of features
        assert len(all_features.columns) > 100
        
        # Check no NaN in first row (after warmup)
        last_row = all_features.iloc[-1]
        non_nan_count = last_row.notna().sum()
        assert non_nan_count > 50  # Most features should be computed