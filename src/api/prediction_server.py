"""FastAPI Prediction Server for Bitcoin Trading."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.features.technical_indicators import TechnicalIndicators
from src.risk_management.risk_manager import RiskManager
from src.risk_management.models.position_sizing import KellyPositionSizer
from src.risk_management.models.cost_model import BinanceCostModel
from src.risk_management.models.drawdown_guard import DrawdownGuard
from src.risk_management.models.api_throttler import BinanceAPIThrottler
from src.utils import setup_logging

logger = setup_logging(__name__)


class MarketData(BaseModel):
    """Market data for prediction."""
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., gt=0, description="Volume")
    
    @validator('high')
    def high_gte_open_close(cls, v, values):
        """Validate high >= max(open, close)."""
        if 'open' in values and 'close' in values:
            if v < max(values['open'], values['close']):
                raise ValueError('high must be >= max(open, close)')
        return v
    
    @validator('low')
    def low_lte_open_close(cls, v, values):
        """Validate low <= min(open, close)."""
        if 'open' in values and 'close' in values:
            if v > min(values['open'], values['close']):
                raise ValueError('low must be <= min(open, close)')
        return v


class PredictionRequest(BaseModel):
    """Single prediction request."""
    market_data: List[MarketData] = Field(..., min_items=200, description="Historical market data (min 200 candles)")
    portfolio_value: float = Field(10000.0, gt=0, description="Portfolio value in USD")
    confidence: float = Field(0.5, ge=0, le=1, description="Trading signal confidence")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    samples: List[PredictionRequest] = Field(..., min_items=1, max_items=100)


class RiskAnalysisRequest(BaseModel):
    """Risk analysis request."""
    portfolio_value: float = Field(..., gt=0)
    positions: Dict[str, float] = Field(default_factory=dict)
    recent_returns: List[float] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    """Prediction response."""
    action: str  # "buy", "sell", "hold"
    position_size: float  # Fraction of portfolio
    confidence: float
    risk_metrics: Dict[str, float]
    timestamp: str


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response."""
    current_drawdown: float
    max_drawdown: float
    risk_multiplier: float
    daily_pnl: float
    position_count: int
    risk_warnings: List[str]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    name: str
    version: str
    features: List[str]
    risk_parameters: Dict[str, float]
    last_updated: str


class PredictionServer:
    """FastAPI prediction server."""
    
    def __init__(self):
        """Initialize prediction server."""
        self.app = FastAPI(
            title="Bitcoin Trading Prediction API",
            description="τ-SAC based Bitcoin trading predictions with risk management",
            version="1.0.0"
        )
        
        # Initialize components
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager(
            position_sizer=KellyPositionSizer(min_edge=0.02),
            cost_model=BinanceCostModel(),
            drawdown_guard=DrawdownGuard(max_drawdown=0.1),
            api_throttler=BinanceAPIThrottler()
        )
        
        # Model placeholder (would load actual model in production)
        self.model = None
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Set up CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/")
        def root():
            """Root endpoint."""
            return {
                "message": "Bitcoin Trading Prediction API",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health")
        def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        def predict(request: PredictionRequest):
            """Make trading prediction."""
            try:
                # Convert market data to DataFrame
                df = pd.DataFrame([m.dict() for m in request.market_data])
                
                # Add technical indicators
                df_with_indicators = self.indicators.add_all_indicators(df)
                
                # Mock prediction (replace with actual model)
                action = self._mock_predict(df_with_indicators)
                
                # Calculate position size
                current_price = df['close'].iloc[-1]
                position_approved, position_size, reason = self.risk_manager.check_new_position(
                    symbol="BTCUSDT",
                    portfolio_value=request.portfolio_value,
                    current_price=current_price,
                    signal_confidence=request.confidence,
                    win_rate=0.55,  # Mock win rate
                    avg_win=0.03,   # Mock average win
                    avg_loss=0.01   # Mock average loss
                )
                
                if not position_approved:
                    action = "hold"
                    position_size = 0.0
                
                # Get risk metrics
                risk_report = self.risk_manager.get_risk_report()
                
                return PredictionResponse(
                    action=action,
                    position_size=position_size,
                    confidence=request.confidence,
                    risk_metrics={
                        "current_drawdown": risk_report["current_drawdown"],
                        "max_drawdown": risk_report["max_drawdown"],
                        "daily_pnl": risk_report["daily_pnl"],
                        "position_count": risk_report["active_positions"]
                    },
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e
        
        @self.app.post("/predict/batch")
        def batch_predict(request: BatchPredictionRequest):
            """Make batch predictions."""
            predictions = []
            for sample in request.samples:
                try:
                    pred = predict(sample)
                    predictions.append(pred)
                except Exception as e:
                    predictions.append({
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            return {"predictions": predictions}
        
        @self.app.post("/analyze/risk", response_model=RiskAnalysisResponse)
        def analyze_risk(request: RiskAnalysisRequest):
            """Analyze portfolio risk."""
            try:
                # Update risk manager state
                for symbol, position in request.positions.items():
                    self.risk_manager.active_positions[symbol] = {
                        "size": position,
                        "entry_price": 50000,  # Mock
                        "entry_time": datetime.now()
                    }
                
                # Update portfolio
                self.risk_manager.update_portfolio(
                    portfolio_value=request.portfolio_value,
                    current_prices={"BTCUSDT": 50000}  # Mock
                )
                
                # Get risk report
                report = self.risk_manager.get_risk_report()
                
                # Identify warnings
                warnings = []
                if report["current_drawdown"] > 0.05:
                    warnings.append(f"High drawdown: {report['current_drawdown']:.1%}")
                if report["active_positions"] >= 3:
                    warnings.append(f"Many open positions: {report['active_positions']}")
                if report["daily_pnl"] < -500:
                    warnings.append(f"Large daily loss: ${report['daily_pnl']:.2f}")
                
                return RiskAnalysisResponse(
                    current_drawdown=report["current_drawdown"],
                    max_drawdown=report["max_drawdown"],
                    risk_multiplier=self.risk_manager.drawdown_guard.get_risk_multiplier(),
                    daily_pnl=report["daily_pnl"],
                    position_count=report["active_positions"],
                    risk_warnings=warnings
                )
                
            except Exception as e:
                logger.error(f"Risk analysis error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        def model_info():
            """Get model information."""
            return ModelInfoResponse(
                name="TauSACTrader",
                version="1.0.0",
                features=[
                    "OHLCV data",
                    "200+ technical indicators",
                    "Risk-adjusted position sizing",
                    "Drawdown protection"
                ],
                risk_parameters={
                    "max_drawdown": 0.10,
                    "min_edge": 0.02,
                    "kelly_fraction": 0.25,
                    "max_positions": 5,
                    "daily_loss_limit": 1000
                },
                last_updated=datetime.now().isoformat()
            )
        
        @self.app.post("/backtest/run")
        def run_backtest(background_tasks: BackgroundTasks):
            """Run backtest (async)."""
            # Add backtest to background tasks
            background_tasks.add_task(self._run_backtest_task)
            return {
                "message": "Backtest started",
                "task_id": "mock-task-id",
                "status": "running"
            }
        
        @self.app.get("/metrics")
        def get_metrics():
            """Get system metrics."""
            api_metrics = self.risk_manager.api_throttler.get_metrics()
            return {
                "api_usage": api_metrics,
                "risk_metrics": self.risk_manager.get_risk_report(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _mock_predict(self, df: pd.DataFrame) -> str:
        """Mock prediction logic."""
        # Simple momentum strategy
        if len(df) < 2:
            return "hold"
        
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        if last_close > prev_close * 1.01:  # 1% increase
            return "buy"
        elif last_close < prev_close * 0.99:  # 1% decrease
            return "sell"
        else:
            return "hold"
    
    def _run_backtest_task(self):
        """Run backtest in background."""
        # Mock backtest implementation
        logger.info("Running backtest...")
        # Actual backtest logic would go here
        logger.info("Backtest completed")
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app instance."""
        return self.app


def create_app() -> FastAPI:
    """Create and return FastAPI app."""
    server = PredictionServer()
    return server.get_app()


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)