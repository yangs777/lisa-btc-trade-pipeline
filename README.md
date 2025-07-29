# Bitcoin τ-SAC Trading System v0.3

[![CI Pipeline](https://github.com/unsuperior-ai/lisa-btc-trade-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/unsuperior-ai/lisa-btc-trade-pipeline/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-red.svg)](https://github.com/PyCQA/bandit)

Advanced reinforcement learning trading system for BTC/USDT futures with τ-selection mechanism.

## 🎯 Target Metrics

| Metric | Stage-1 (Pilot) | Stage-2 (Live) |
|--------|-----------------|----------------|
| Annual PnL | ≥ 38% | ≥ 52% |
| Sharpe Ratio | ≥ 1.3 | ≥ 1.45 |
| Max Drawdown | ≤ 8% | ≤ 10% |
| Hit Rate | ≥ 55% | ≥ 57% |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and add your keys
cp .env.example .env
```

### 2. Start Data Collection

```bash
# Set Google Cloud credentials (for GCS upload)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Start orderbook collector and GCS uploader
./scripts/start_collector.sh

# Monitor logs
tail -f logs/collector.log
tail -f logs/uploader.log
```

### 3. Daily Preprocessing (Scheduled at 04:05 KST)

```bash
# Run manually for specific date
python -m data_collection.daily_preprocessor --date 2024-01-27

# Or run on schedule
python -m data_collection.daily_preprocessor --schedule
```

### 4. Feature Engineering Test

```bash
# Test 104 indicators implementation
python -m features.feature_engineering
```

## 📊 System Architecture

```
lisa_btc_trade_pipeline_v1_1/
├── data_collection/          # Real-time data collection
│   ├── data_collector.py     # WebSocket orderbook collector (1s snapshots)
│   ├── gcs_uploader.py       # Auto-upload to Google Cloud Storage
│   └── daily_preprocessor.py # Daily feature generation (04:05 KST)
│
├── features/                 # Feature engineering
│   ├── feature_engineering.py # 104 technical indicators
│   ├── feature_selector.py    # SHAP-based selection
│   └── feature_validator.py   # Data quality checks
│
├── training/                 # RL training system
│   ├── tau_sac.py           # τ-SAC implementation
│   ├── rl_env.py            # Trading environment
│   └── vertex_train.py      # Vertex AI integration
│
├── trading/                  # Live trading system
│   ├── futures_trade_executor.py # Futures trading with risk controls
│   ├── prediction_server.py      # FastAPI server (<200ms)
│   └── main_trader.py           # Main trading loop
│
└── monitoring/              # System monitoring
    ├── prometheus_exporter.py
    └── grafana_dashboards/
```

## 📈 Feature Categories (104 Total)

1. **Price & Liquidity (18)**: Mid-price changes, VWAP, spreads, micro-price
2. **Order Flow (22)**: OFI, depth imbalance, queue intensity
3. **Volatility (12)**: Rolling σ, ATR, realized volatility
4. **Momentum (14)**: RSI, ROC, Stochastic, Williams %R
5. **Trend (13)**: EMAs, MACD, ADX, linear regression
6. **Bands/Channels (8)**: Bollinger, Keltner, Donchian
7. **Volume & Psychology (17)**: OBV, VWMA, cancel rates, time features

## 🤖 τ-SAC Configuration

- **Actions**: {Flat, Long, Short, Exit} × τ ∈ {3, 6, 9, 12} seconds
- **Observation**: 104 features + position state + timing info
- **Reward**: Risk-Balanced Sharpe Reward (RBSR)
- **Training**: Curriculum learning with progressive τ values

## ⚠️ Risk Management

- **Max Drawdown**: 8% → flatten all positions + policy rollback
- **Consecutive Losses**: 6 losses or -3% → 60 min pause
- **Time Fence**: τ+2s overflow → market exit
- **Position Size**: ≤ 1% equity (Kelly × 0.5)

## 🔧 Configuration

Edit `configs/trading_config.yaml` for:
- Trading parameters
- Risk limits
- Model hyperparameters
- Monitoring settings

## 📝 Environment Variables

Required in `.env`:
```bash
BINANCE_API_KEY=your_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Note: Binance secret key is entered at runtime for security.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test data collection
python -m data_collection.data_collector --test

# Test feature engineering
python -m features.feature_engineering
```

## 📊 Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **FastAPI Docs**: http://localhost:8000/docs

## 🚨 Important Notes

1. **Paper Trading**: Always run 4-week paper trading before live deployment
2. **Leverage**: Limited to 1-2x for testing, 5x for production
3. **Single Position**: Only one position allowed at a time
4. **Futures Only**: BTC/USDT perpetual futures, isolated margin

## 📞 Support

For issues or questions about this personal trading system, check logs in:
- `logs/collector.log` - Data collection logs
- `logs/uploader.log` - GCS upload logs
- `logs/trading.log` - Trading execution logs

---

⚠️ **Disclaimer**: This is a personal trading system. Use at your own risk. Not financial advice.