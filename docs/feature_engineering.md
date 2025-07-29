# Feature Engineering Module

This module implements 104 technical indicators for the BTC/USDT trading system.

## Overview

The feature engineering module provides a comprehensive set of technical indicators organized by category:

- **Trend Indicators** (23): SMA, EMA, WMA, HMA, TEMA, DEMA, KAMA, Ichimoku Cloud components
- **Momentum Indicators** (18): RSI, Stochastic, MACD, CCI, Williams %R, ROC, Momentum, TSI, UO, AO
- **Volatility Indicators** (16): ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Ulcer Index
- **Volume Indicators** (12): OBV, AD/ADL, CMF, EMV, Force Index, MFI, NVI/PVI, VWAP, VPT, VWMA
- **Trend Strength Indicators** (11): ADX, +DI/-DI, Aroon, Vortex, TRIX, Mass Index
- **Pattern Indicators** (7): Parabolic SAR, SuperTrend, ZigZag, Pivot Points
- **Statistical Indicators** (12): Standard Deviation, Variance, Skewness, Kurtosis, Correlation, Linear Regression

## Usage

### Basic Usage

```python
from src.feature_engineering import FeatureEngineer
import pandas as pd

# Load your OHLCV data
df = pd.read_parquet('data/btcusdt_1min.parquet')

# Initialize feature engineer
engineer = FeatureEngineer()

# Calculate all 104 indicators
features_df = engineer.transform(df)
```

### Selective Indicator Calculation

```python
# Calculate only specific indicators
selected_indicators = ['RSI_14', 'MACD', 'BB_UPPER', 'BB_LOWER', 'ATR_14']
features_df = engineer.transform_selective(df, selected_indicators)
```

### Individual Indicator Usage

```python
from src.feature_engineering.momentum.oscillators import RSI
from src.feature_engineering.trend.moving_averages import SMA

# Use indicators individually
rsi = RSI(window=14)
rsi_values = rsi.transform(df)

sma = SMA(window=20)
sma_values = sma.transform(df)
```

## Configuration

Indicators are configured in `indicators.yaml`. Each indicator has:
- `name`: Unique identifier (e.g., "RSI_14")
- `class`: Implementation class name
- `params`: Parameters for initialization

Example configuration:
```yaml
momentum:
  - name: RSI_14
    class: RSI
    params: {window: 14}
  - name: RSI_21
    class: RSI
    params: {window: 21}
```

## Architecture

### Base Classes

- `BaseIndicator`: Abstract base for all indicators
- `PriceIndicator`: Base for price-based indicators (default uses 'close')
- `VolumeIndicator`: Base for volume-based indicators
- `OHLCVIndicator`: Base for indicators using full OHLCV data

### Registry System

The `IndicatorRegistry` manages indicator registration and instantiation:
- Dynamically loads indicator configurations
- Creates indicator instances with specified parameters
- Handles errors gracefully with logging

### Feature Engineer

The `FeatureEngineer` class:
- Loads all configured indicators
- Applies indicators to dataframes
- Handles errors and fills NaN appropriately
- Provides performance monitoring

## Indicator Categories

### Trend Indicators
- Moving averages: SMA, EMA, WMA, HMA, TEMA, DEMA, KAMA
- Ichimoku Cloud: Tenkan-sen, Kijun-sen, Senkou Span A/B

### Momentum Indicators
- Oscillators: RSI, Stochastic (K/D), StochRSI (K/D)
- MACD: MACD line, Signal line, Histogram
- Others: CCI, Williams %R, ROC, Momentum, TSI, UO, AO

### Volatility Indicators
- Average True Range: ATR, NATR
- Bollinger Bands: Upper, Middle, Lower, Width, %B
- Keltner Channels: Upper, Middle, Lower
- Donchian Channels: Upper, Lower
- Others: Ulcer Index, Mass Index

### Volume Indicators
- Classic: OBV, A/D, ADL, VPT
- Money Flow: CMF, MFI, EMV, Force Index
- Volume Index: NVI, PVI
- Price-Volume: VWAP, VWMA

### Trend Strength Indicators
- ADX System: ADX, +DI, -DI
- Aroon: Aroon Up, Aroon Down, Aroon Oscillator
- Vortex: VI+, VI-
- TRIX

### Pattern Indicators
- Parabolic SAR: PSAR values and trend
- SuperTrend: Dynamic support/resistance
- ZigZag: Swing highs and lows
- Pivot Points: Pivot highs and lows

### Statistical Indicators
- Basic Stats: StdDev, Variance, SEM, Skew, Kurtosis
- Regression: Correlation, Beta, Linear Regression (value, slope, angle)
- Forecast: Time Series Forecast (TSF)

## Performance Considerations

1. **Batch Processing**: Use `transform()` for all indicators rather than calculating individually
2. **Data Requirements**: Some indicators need minimum data points (e.g., 200 for SMA_200)
3. **NaN Handling**: Initial values will be NaN due to lookback periods
4. **Memory Usage**: ~104 additional columns will be added to your dataframe

## Testing

Run the test script to verify functionality:
```bash
python scripts/test_feature_engineering.py
```

Run unit tests:
```bash
pytest tests/test_feature_engineering.py -v
pytest tests/test_indicators.py -v
```

## Adding New Indicators

1. Create indicator class inheriting from appropriate base class
2. Implement `name` property and `transform()` method
3. Register in `FeatureEngineer._register_all_indicators()`
4. Add configuration to `indicators.yaml`

Example:
```python
class MyIndicator(PriceIndicator):
    @property
    def name(self) -> str:
        return f"MY_IND_{self.window_size}"
        
    def transform(self, df: pd.DataFrame) -> pd.Series:
        price = self._get_price(df)
        # Your calculation logic
        return self._handle_nan(result)
```