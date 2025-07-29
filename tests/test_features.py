"""Tests for feature engineering modules."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestFeatureEngineering:
    """Test feature engineering."""

    def test_feature_scaler_initialization(self):
        """Test feature scaler initialization."""
        from src.features.feature_engineering import FeatureScaler

        scaler = FeatureScaler()
        assert hasattr(scaler, "fit")
        assert hasattr(scaler, "transform")
        assert hasattr(scaler, "fit_transform")

    def test_feature_scaler_fit_transform(self):
        """Test feature scaler fit and transform."""
        from src.features.feature_engineering import FeatureScaler

        # Create test data
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "feature3": [100, 200, 300, 400, 500]
        })

        scaler = FeatureScaler()
        
        # Fit and transform
        scaled = scaler.fit_transform(data)
        
        # Check output
        assert isinstance(scaled, pd.DataFrame)
        assert scaled.shape == data.shape
        assert list(scaled.columns) == list(data.columns)
        
        # Check scaling worked (values should be between -3 and 3 roughly)
        assert scaled.abs().max().max() < 4

    def test_feature_scaler_transform(self):
        """Test feature scaler transform."""
        from src.features.feature_engineering import FeatureScaler

        # Create train and test data
        train_data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })
        
        test_data = pd.DataFrame({
            "feature1": [6, 7],
            "feature2": [60, 70]
        })

        scaler = FeatureScaler()
        
        # Fit on train
        scaler.fit(train_data)
        
        # Transform test
        scaled_test = scaler.transform(test_data)
        
        # Check output
        assert isinstance(scaled_test, pd.DataFrame)
        assert scaled_test.shape == test_data.shape

    def test_feature_scaler_inverse_transform(self):
        """Test feature scaler inverse transform."""
        from src.features.feature_engineering import FeatureScaler

        # Create test data
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })

        scaler = FeatureScaler()
        
        # Fit and transform
        scaled = scaler.fit_transform(data)
        
        # Inverse transform
        unscaled = scaler.inverse_transform(scaled)
        
        # Check we get back close to original
        assert np.allclose(unscaled.values, data.values, rtol=1e-10)

    def test_feature_selector_initialization(self):
        """Test feature selector initialization."""
        from src.features.feature_engineering import FeatureSelector

        selector = FeatureSelector(n_features=10)
        assert selector.n_features == 10
        assert hasattr(selector, "fit")
        assert hasattr(selector, "transform")

    def test_feature_selector_fit_transform(self):
        """Test feature selector fit and transform."""
        from src.features.feature_engineering import FeatureSelector

        # Create test data with target
        np.random.seed(42)
        X = pd.DataFrame({
            f"feature_{i}": np.random.randn(100) for i in range(20)
        })
        # Create target correlated with first few features
        y = X["feature_0"] + 0.5 * X["feature_1"] + 0.3 * X["feature_2"] + np.random.randn(100) * 0.1

        selector = FeatureSelector(n_features=5)
        
        # Fit and transform
        selected = selector.fit_transform(X, y)
        
        # Check output
        assert isinstance(selected, pd.DataFrame)
        assert selected.shape[1] == 5
        assert selected.shape[0] == X.shape[0]

    def test_feature_selector_get_selected_features(self):
        """Test getting selected feature names."""
        from src.features.feature_engineering import FeatureSelector

        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            f"feature_{i}": np.random.randn(100) for i in range(10)
        })
        y = X["feature_0"] + X["feature_1"] + np.random.randn(100) * 0.1

        selector = FeatureSelector(n_features=3)
        selector.fit(X, y)
        
        # Get selected features
        selected_features = selector.get_selected_features()
        
        # Check output
        assert isinstance(selected_features, list)
        assert len(selected_features) == 3
        assert all(feature in X.columns for feature in selected_features)

    def test_feature_pipeline_initialization(self):
        """Test feature pipeline initialization."""
        from src.features.feature_engineering import FeaturePipeline

        pipeline = FeaturePipeline()
        assert hasattr(pipeline, "add_features")
        assert hasattr(pipeline, "prepare_features")

    def test_feature_pipeline_add_features(self):
        """Test adding features to pipeline."""
        from src.features.feature_engineering import FeaturePipeline
        from src.features.technical_indicators import TechnicalIndicators

        # Create test OHLCV data
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        df = pd.DataFrame({
            "open": np.random.uniform(49000, 51000, 200),
            "high": np.random.uniform(49500, 51500, 200),
            "low": np.random.uniform(48500, 50500, 200),
            "close": np.random.uniform(49000, 51000, 200),
            "volume": np.random.uniform(100, 1000, 200),
        }, index=dates)
        
        # Ensure OHLC relationships
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        pipeline = FeaturePipeline()
        indicators = TechnicalIndicators()
        
        # Add features
        df_with_features = pipeline.add_features(df, indicators)
        
        # Check output
        assert isinstance(df_with_features, pd.DataFrame)
        assert len(df_with_features.columns) > len(df.columns)
        assert all(col in df_with_features.columns for col in df.columns)

    def test_feature_pipeline_prepare_features(self):
        """Test preparing features for model."""
        from src.features.feature_engineering import FeaturePipeline

        # Create test data with features
        df = pd.DataFrame({
            "open": np.random.uniform(49000, 51000, 100),
            "high": np.random.uniform(49500, 51500, 100),
            "low": np.random.uniform(48500, 50500, 100),
            "close": np.random.uniform(49000, 51000, 100),
            "volume": np.random.uniform(100, 1000, 100),
            "sma_20": np.random.uniform(49000, 51000, 100),
            "rsi_14": np.random.uniform(30, 70, 100),
        })

        pipeline = FeaturePipeline()
        
        # Prepare features
        X, feature_names = pipeline.prepare_features(df)
        
        # Check output
        assert isinstance(X, np.ndarray)
        assert isinstance(feature_names, list)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(feature_names)
        assert len(feature_names) > 0


# Create feature engineering module if needed  
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))