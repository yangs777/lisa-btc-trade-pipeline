"""
Enhanced Feature Engineering with 104 Technical Indicators
Implements all features from the Master Blueprint v0.3
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from numba import jit, njit
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Calculates 104 technical indicators across 8 categories"""
    
    def __init__(self, feature_config: Optional[Dict] = None):
        self.feature_config = feature_config or self._get_default_config()
        self.feature_names = []
        self.feature_importance = {}
        
    def _get_default_config(self) -> Dict:
        """Default feature configuration"""
        return {
            'price_liquidity': True,
            'order_flow': True,
            'volatility': True,
            'momentum': True,
            'trend': True,
            'bands_channels': True,
            'volume_psych': True,
            'enable_all': True
        }
        
    def build_feature_matrix(self, 
                           df_orderbook: pd.DataFrame,
                           df_trades: Optional[pd.DataFrame] = None,
                           window: int = 3) -> pd.DataFrame:
        """
        Build complete feature matrix with 104 indicators
        
        Args:
            df_orderbook: Orderbook snapshot data with columns:
                         [timestamp, mid_price, best_bid, best_ask, spread, spread_pct,
                          imbalance_5, bid_0_price, bid_0_qty, ..., ask_19_price, ask_19_qty]
            df_trades: Trade data with columns:
                      [buy_volume_1s, sell_volume_1s, trade_imbalance, vwap_1s]
            window: Base window for calculations
        """
        features = pd.DataFrame(index=df_orderbook.index)
        
        # Merge trades data if available
        if df_trades is not None:
            df = pd.concat([df_orderbook, df_trades], axis=1)
        else:
            df = df_orderbook.copy()
            
        # 1. Price & Liquidity Features (18 features)
        if self.feature_config.get('price_liquidity', True):
            features = pd.concat([features, self._calculate_price_liquidity_features(df)], axis=1)
            
        # 2. Order Flow Features (22 features)
        if self.feature_config.get('order_flow', True):
            features = pd.concat([features, self._calculate_order_flow_features(df)], axis=1)
            
        # 3. Volatility Features (12 features)
        if self.feature_config.get('volatility', True):
            features = pd.concat([features, self._calculate_volatility_features(df)], axis=1)
            
        # 4. Momentum Features (14 features)
        if self.feature_config.get('momentum', True):
            features = pd.concat([features, self._calculate_momentum_features(df)], axis=1)
            
        # 5. Trend Features (13 features)
        if self.feature_config.get('trend', True):
            features = pd.concat([features, self._calculate_trend_features(df)], axis=1)
            
        # 6. Bands & Channels Features (8 features)
        if self.feature_config.get('bands_channels', True):
            features = pd.concat([features, self._calculate_bands_channels_features(df)], axis=1)
            
        # 7. Volume & Psychology Features (17 features)
        if self.feature_config.get('volume_psych', True):
            features = pd.concat([features, self._calculate_volume_psych_features(df)], axis=1)
            
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
        
    def _calculate_price_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price and liquidity features (18 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Basic price features
        features['mid_price_delta_1s'] = df['mid_price'].diff(1)
        features['mid_price_delta_3s'] = df['mid_price'].diff(3)
        features['mid_price_delta_5s'] = df['mid_price'].diff(5)
        
        # VWAP features
        if 'vwap_1s' in df.columns:
            features['vwap_mid_diff'] = df['vwap_1s'] - df['mid_price']
            features['vwap_mid_ratio'] = df['vwap_1s'] / df['mid_price']
        else:
            features['vwap_mid_diff'] = 0
            features['vwap_mid_ratio'] = 1
            
        # Spread features
        features['spread_bps'] = df['spread_pct'] * 100  # basis points
        features['spread_rolling_mean_10s'] = df['spread'].rolling(10).mean()
        features['spread_rolling_std_10s'] = df['spread'].rolling(10).std()
        features['spread_zscore'] = (df['spread'] - features['spread_rolling_mean_10s']) / features['spread_rolling_std_10s']
        
        # Micro-price (weighted by size)
        if all(col in df.columns for col in ['bid_0_price', 'bid_0_qty', 'ask_0_price', 'ask_0_qty']):
            features['micro_price'] = (
                df['bid_0_price'] * df['ask_0_qty'] + df['ask_0_price'] * df['bid_0_qty']
            ) / (df['bid_0_qty'] + df['ask_0_qty'])
            features['micro_price_diff'] = features['micro_price'] - df['mid_price']
        else:
            features['micro_price'] = df['mid_price']
            features['micro_price_diff'] = 0
            
        # Price position in range
        high_10s = df['mid_price'].rolling(10).max()
        low_10s = df['mid_price'].rolling(10).min()
        features['price_position_10s'] = (df['mid_price'] - low_10s) / (high_10s - low_10s)
        
        # Liquidity concentration
        total_bid_qty = sum(df[f'bid_{i}_qty'] for i in range(5) if f'bid_{i}_qty' in df.columns)
        total_ask_qty = sum(df[f'ask_{i}_qty'] for i in range(5) if f'ask_{i}_qty' in df.columns)
        
        if 'bid_0_qty' in df.columns:
            features['bid_concentration_l1'] = df['bid_0_qty'] / total_bid_qty
            features['ask_concentration_l1'] = df['ask_0_qty'] / total_ask_qty
        else:
            features['bid_concentration_l1'] = 0.2  # Default uniform
            features['ask_concentration_l1'] = 0.2
            
        # Effective spread
        features['effective_spread'] = 2 * np.abs(df['mid_price'].shift(-1) - df['mid_price'])
        
        # Log returns
        features['log_return_1s'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
        features['log_return_5s'] = np.log(df['mid_price'] / df['mid_price'].shift(5))
        features['log_return_10s'] = np.log(df['mid_price'] / df['mid_price'].shift(10))
        
        return features
        
    def _calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow features (22 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Order Flow Imbalance (OFI) at different levels
        for level in range(5):
            bid_col = f'bid_{level}_qty'
            ask_col = f'ask_{level}_qty'
            if bid_col in df.columns and ask_col in df.columns:
                # OFI = change in bid - change in ask
                bid_change = df[bid_col].diff()
                ask_change = df[ask_col].diff()
                features[f'ofi_level_{level}'] = bid_change - ask_change
            else:
                features[f'ofi_level_{level}'] = 0
                
        # Cumulative OFI
        features['ofi_total'] = sum(features[f'ofi_level_{i}'] for i in range(5))
        features['ofi_cumsum_10s'] = features['ofi_total'].rolling(10).sum()
        
        # Depth imbalance at different levels
        for level in [1, 3, 5, 10]:
            bid_depth = sum(df[f'bid_{i}_qty'] for i in range(min(level, 20)) if f'bid_{i}_qty' in df.columns)
            ask_depth = sum(df[f'ask_{i}_qty'] for i in range(min(level, 20)) if f'ask_{i}_qty' in df.columns)
            total_depth = bid_depth + ask_depth
            features[f'depth_imbalance_l{level}'] = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
        # Depth changes
        total_bid_depth = sum(df[f'bid_{i}_qty'] for i in range(5) if f'bid_{i}_qty' in df.columns)
        total_ask_depth = sum(df[f'ask_{i}_qty'] for i in range(5) if f'ask_{i}_qty' in df.columns)
        
        features['bid_depth_change_1s'] = total_bid_depth.diff(1)
        features['ask_depth_change_1s'] = total_ask_depth.diff(1)
        features['total_depth_change_1s'] = features['bid_depth_change_1s'] + features['ask_depth_change_1s']
        
        # Weighted depth imbalance
        if all(f'bid_{i}_price' in df.columns for i in range(5)):
            weighted_bid = sum(df[f'bid_{i}_qty'] * (1 - i/10) for i in range(5))
            weighted_ask = sum(df[f'ask_{i}_qty'] * (1 - i/10) for i in range(5))
            features['weighted_depth_imbalance'] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)
        else:
            features['weighted_depth_imbalance'] = df.get('imbalance_5', 0)
            
        # Queue intensity
        features['bid_intensity'] = total_bid_depth / df['best_bid'] if 'best_bid' in df.columns else 0
        features['ask_intensity'] = total_ask_depth / df['best_ask'] if 'best_ask' in df.columns else 0
        
        # Trade flow features
        if 'buy_volume_1s' in df.columns and 'sell_volume_1s' in df.columns:
            features['net_trade_flow'] = df['buy_volume_1s'] - df['sell_volume_1s']
            features['trade_imbalance'] = df.get('trade_imbalance', 
                (df['buy_volume_1s'] - df['sell_volume_1s']) / (df['buy_volume_1s'] + df['sell_volume_1s'] + 1e-8))
        else:
            features['net_trade_flow'] = 0
            features['trade_imbalance'] = 0
            
        return features
        
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features (12 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Rolling volatility at different time scales
        for window in [3, 10, 30, 60]:
            features[f'volatility_{window}s'] = df['mid_price'].pct_change().rolling(window).std()
            
        # ATR (Average True Range) adapted for seconds
        high = df['mid_price'].rolling(14).max()
        low = df['mid_price'].rolling(14).min()
        features['atr_14s'] = (high - low).rolling(14).mean()
        
        # Realized volatility
        returns = df['mid_price'].pct_change()
        features['realized_vol_30s'] = np.sqrt((returns ** 2).rolling(30).sum())
        
        # Volatility ratios
        features['vol_ratio_10_30'] = features['volatility_10s'] / features['volatility_30s']
        features['vol_ratio_30_60'] = features['volatility_30s'] / features['volatility_60s']
        
        # GARCH-like features (simplified)
        features['squared_returns'] = returns ** 2
        features['vol_momentum'] = features['squared_returns'].rolling(10).mean()
        
        # High-low spread volatility
        if all(col in df.columns for col in ['bid_0_price', 'ask_0_price']):
            hl_spread = df['ask_0_price'] - df['bid_0_price']
            features['spread_volatility_10s'] = hl_spread.rolling(10).std()
        else:
            features['spread_volatility_10s'] = df['spread'].rolling(10).std()
            
        # Volatility z-score
        vol_mean = features['volatility_30s'].rolling(300).mean()
        vol_std = features['volatility_30s'].rolling(300).std()
        features['volatility_zscore'] = (features['volatility_30s'] - vol_mean) / vol_std
        
        return features
        
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features (14 total)"""
        features = pd.DataFrame(index=df.index)
        
        # RSI at different periods
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = ta.rsi(df['mid_price'], length=period)
            
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}s'] = ((df['mid_price'] - df['mid_price'].shift(period)) / 
                                         df['mid_price'].shift(period)) * 100
                                         
        # Stochastic oscillator
        stoch = ta.stoch(df['mid_price'], df['mid_price'], df['mid_price'], k=14, d=3)
        if stoch is not None and len(stoch.columns) >= 2:
            features['stoch_k'] = stoch.iloc[:, 0]
            features['stoch_d'] = stoch.iloc[:, 1]
        else:
            features['stoch_k'] = 50  # Default neutral
            features['stoch_d'] = 50
            
        # Williams %R
        features['williams_r'] = ta.willr(df['mid_price'], df['mid_price'], df['mid_price'], length=14)
        
        # Commodity Channel Index (CCI)
        features['cci_20'] = ta.cci(df['mid_price'], df['mid_price'], df['mid_price'], length=20)
        
        # Money Flow Index (simplified without volume)
        if 'buy_volume_1s' in df.columns and 'sell_volume_1s' in df.columns:
            money_flow = df['mid_price'] * (df['buy_volume_1s'] + df['sell_volume_1s'])
            features['mfi_14'] = ta.mfi(df['mid_price'], df['mid_price'], df['mid_price'], 
                                       money_flow, length=14)
        else:
            features['mfi_14'] = 50  # Default neutral
            
        # Momentum
        features['momentum_10s'] = df['mid_price'] - df['mid_price'].shift(10)
        
        return features
        
    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features (13 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Exponential Moving Averages
        for period in [3, 9, 21, 50]:
            features[f'ema_{period}'] = ta.ema(df['mid_price'], length=period)
            
        # EMA distances
        features['price_ema3_dist'] = (df['mid_price'] - features['ema_3']) / df['mid_price']
        features['price_ema9_dist'] = (df['mid_price'] - features['ema_9']) / df['mid_price']
        features['ema3_ema9_dist'] = (features['ema_3'] - features['ema_9']) / features['ema_9']
        
        # MACD
        macd = ta.macd(df['mid_price'], fast=12, slow=26, signal=9)
        if macd is not None and len(macd.columns) >= 3:
            features['macd'] = macd.iloc[:, 0]
            features['macd_signal'] = macd.iloc[:, 1]
            features['macd_hist'] = macd.iloc[:, 2]
        else:
            features['macd'] = 0
            features['macd_signal'] = 0
            features['macd_hist'] = 0
            
        # ADX (Average Directional Index)
        adx = ta.adx(df['mid_price'], df['mid_price'], df['mid_price'], length=14)
        if adx is not None and len(adx.columns) >= 1:
            features['adx'] = adx.iloc[:, 0]
        else:
            features['adx'] = 25  # Default neutral trend strength
            
        # Linear regression slope
        for window in [10, 30]:
            features[f'linreg_slope_{window}s'] = df['mid_price'].rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
        return features
        
    def _calculate_bands_channels_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate bands and channels features (8 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Bollinger Bands
        bb = ta.bbands(df['mid_price'], length=20, std=2)
        if bb is not None and len(bb.columns) >= 3:
            features['bb_upper'] = bb.iloc[:, 0]
            features['bb_middle'] = bb.iloc[:, 1]
            features['bb_lower'] = bb.iloc[:, 2]
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (df['mid_price'] - features['bb_lower']) / features['bb_width']
        else:
            # Fallback calculation
            sma = df['mid_price'].rolling(20).mean()
            std = df['mid_price'].rolling(20).std()
            features['bb_upper'] = sma + 2 * std
            features['bb_middle'] = sma
            features['bb_lower'] = sma - 2 * std
            features['bb_width'] = 4 * std
            features['bb_position'] = 0.5
            
        # Keltner Channels
        kc = ta.kc(df['mid_price'], df['mid_price'], df['mid_price'], length=20)
        if kc is not None and len(kc.columns) >= 3:
            features['kc_upper'] = kc.iloc[:, 0]
            features['kc_lower'] = kc.iloc[:, 2]
            features['kc_position'] = (df['mid_price'] - features['kc_lower']) / (features['kc_upper'] - features['kc_lower'])
        else:
            # Simplified Keltner
            ema = ta.ema(df['mid_price'], length=20)
            atr = features.get('atr_14s', df['mid_price'].rolling(14).std() * 1.5)
            features['kc_upper'] = ema + 2 * atr
            features['kc_lower'] = ema - 2 * atr
            features['kc_position'] = 0.5
            
        # Donchian Channels
        features['donchian_upper'] = df['mid_price'].rolling(20).max()
        features['donchian_lower'] = df['mid_price'].rolling(20).min()
        features['donchian_position'] = (df['mid_price'] - features['donchian_lower']) / (
            features['donchian_upper'] - features['donchian_lower'])
            
        return features
        
    def _calculate_volume_psych_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume and psychology features (17 total)"""
        features = pd.DataFrame(index=df.index)
        
        # Volume features (if available)
        if 'buy_volume_1s' in df.columns and 'sell_volume_1s' in df.columns:
            total_volume = df['buy_volume_1s'] + df['sell_volume_1s']
            
            # On-Balance Volume (OBV) adapted
            features['obv'] = np.where(df['mid_price'].diff() > 0, total_volume, -total_volume).cumsum()
            
            # Volume-Weighted Moving Average
            features['vwma_10'] = (df['mid_price'] * total_volume).rolling(10).sum() / total_volume.rolling(10).sum()
            
            # Volume ratio
            features['volume_ratio_buy_sell'] = df['buy_volume_1s'] / (df['sell_volume_1s'] + 1e-8)
            
            # Volume momentum
            features['volume_momentum_10s'] = total_volume.rolling(10).mean() / total_volume.rolling(30).mean()
            
            # Large order detection
            volume_mean = total_volume.rolling(100).mean()
            volume_std = total_volume.rolling(100).std()
            features['large_order_indicator'] = (total_volume > volume_mean + 2 * volume_std).astype(int)
        else:
            # Default values when volume not available
            features['obv'] = 0
            features['vwma_10'] = df['mid_price'].rolling(10).mean()
            features['volume_ratio_buy_sell'] = 1
            features['volume_momentum_10s'] = 1
            features['large_order_indicator'] = 0
            
        # Order book features
        total_bid_qty = sum(df[f'bid_{i}_qty'] for i in range(10) if f'bid_{i}_qty' in df.columns)
        total_ask_qty = sum(df[f'ask_{i}_qty'] for i in range(10) if f'ask_{i}_qty' in df.columns)
        
        # Cancel rate proxy (changes in total quantity)
        features['bid_cancel_rate'] = total_bid_qty.diff().clip(upper=0).abs() / (total_bid_qty + 1e-8)
        features['ask_cancel_rate'] = total_ask_qty.diff().clip(upper=0).abs() / (total_ask_qty + 1e-8)
        
        # Psychological levels
        price = df['mid_price']
        features['distance_to_round_100'] = price % 100
        features['distance_to_round_1000'] = price % 1000
        features['at_psychological_level'] = ((price % 100 < 1) | (price % 1000 < 10)).astype(int)
        
        # Momentum divergence
        price_momentum = price.diff(10)
        if 'obv' in features.columns:
            obv_momentum = features['obv'].diff(10)
            features['momentum_divergence'] = np.sign(price_momentum) != np.sign(obv_momentum)
        else:
            features['momentum_divergence'] = 0
            
        # Time-based features
        if hasattr(df.index, 'hour'):
            features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            features['is_asian_session'] = df.index.hour.isin(range(0, 8)).astype(int)
            features['is_european_session'] = df.index.hour.isin(range(8, 16)).astype(int)
            features['is_us_session'] = df.index.hour.isin(range(16, 24)).astype(int)
        else:
            # Default time features
            features['hour_sin'] = 0
            features['hour_cos'] = 1
            features['is_asian_session'] = 0
            features['is_european_session'] = 0
            features['is_us_session'] = 1
            
        return features
        
    def select_features_shap(self, 
                           features: pd.DataFrame,
                           target: pd.Series,
                           n_features: int = 50) -> List[str]:
        """
        Select top features using SHAP importance
        
        Args:
            features: Feature matrix
            target: Target variable
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            import shap
            from lightgbm import LGBMRegressor
            
            # Train a simple model
            model = LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
                verbosity=-1
            )
            
            # Remove any NaN values
            mask = ~(features.isna().any(axis=1) | target.isna())
            X_clean = features[mask]
            y_clean = target[mask]
            
            if len(X_clean) < 1000:
                # Not enough data for SHAP, return all features
                return features.columns.tolist()[:n_features]
                
            # Fit model
            model.fit(X_clean, y_clean)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_clean[:1000])  # Use subset for speed
            
            # Get feature importance
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Store importance
            self.feature_importance = dict(zip(features.columns, shap_importance))
            
            # Select top features
            importance_df = pd.DataFrame({
                'feature': features.columns,
                'importance': shap_importance
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
            return selected_features
            
        except Exception as e:
            print(f"SHAP feature selection failed: {e}")
            # Fallback to variance-based selection
            variances = features.var()
            return variances.nlargest(n_features).index.tolist()

# Numba-optimized functions for performance
@njit
def calculate_ofi_fast(bid_changes: np.ndarray, ask_changes: np.ndarray) -> np.ndarray:
    """Fast OFI calculation using Numba"""
    return bid_changes - ask_changes

@njit
def calculate_rolling_stats_fast(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast rolling mean and std using Numba"""
    n = len(values)
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        window_vals = values[i - window + 1:i + 1]
        means[i] = np.mean(window_vals)
        stds[i] = np.std(window_vals)
        
    return means, stds

def test_feature_engineering():
    """Test feature engineering with sample data"""
    # Create sample data
    n_samples = 1000
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='S')
    
    # Generate synthetic orderbook data
    np.random.seed(42)
    base_price = 50000
    price = base_price + np.cumsum(np.random.randn(n_samples) * 10)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'mid_price': price,
        'best_bid': price - np.random.uniform(1, 5, n_samples),
        'best_ask': price + np.random.uniform(1, 5, n_samples),
        'spread': np.random.uniform(2, 10, n_samples),
        'spread_pct': np.random.uniform(0.004, 0.02, n_samples),
        'imbalance_5': np.random.uniform(-0.5, 0.5, n_samples),
        'buy_volume_1s': np.random.exponential(1, n_samples),
        'sell_volume_1s': np.random.exponential(1, n_samples),
        'vwap_1s': price + np.random.uniform(-2, 2, n_samples),
    })
    
    # Add orderbook levels
    for i in range(20):
        df[f'bid_{i}_price'] = df['best_bid'] - (i + 1) * 2
        df[f'bid_{i}_qty'] = np.random.exponential(0.5, n_samples)
        df[f'ask_{i}_price'] = df['best_ask'] + (i + 1) * 2
        df[f'ask_{i}_qty'] = np.random.exponential(0.5, n_samples)
        
    df.set_index('timestamp', inplace=True)
    
    # Create feature engineering instance
    fe = FeatureEngineering()
    
    # Build features
    features = fe.build_feature_matrix(df)
    
    print(f"Created {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")
    print(f"\nFirst few features:")
    print(features.columns[:10].tolist())
    
    # Test SHAP selection
    target = df['mid_price'].pct_change().shift(-1).fillna(0)
    selected = fe.select_features_shap(features, target, n_features=40)
    print(f"\nTop 10 selected features:")
    print(selected[:10])
    
    return features

if __name__ == "__main__":
    test_feature_engineering()