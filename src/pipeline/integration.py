"""Pipeline integration utilities."""

import asyncio
from typing import Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


async def run_live_trading(
    symbol: str = "BTCUSDT",
    duration_seconds: int = 3600,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run live trading pipeline.
    
    Args:
        symbol: Trading symbol
        duration_seconds: Duration to run
        model_path: Path to trained model
        
    Returns:
        Results dictionary
    """
    results = {
        "status": "completed",
        "predictions_made": 0,
        "data_collected": 0,
        "errors": []
    }
    
    try:
        # Import components
        from src.data_collection.binance_websocket import BinanceWebSocket
        from src.rl.models import TauSAC
        from src.api.prediction_server import PredictionServer
        
        # Initialize components
        ws = BinanceWebSocket(symbol.lower())
        
        # Mock model if not provided
        if model_path:
            model = TauSAC.load(model_path)
        else:
            model = TauSAC(observation_dim=10, action_dim=3, tau_values=[3, 6, 9])
        
        # Connect websocket
        await ws.connect()
        
        # Run collection loop
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                # Get data
                data = await ws.get_orderbook_update()
                if data:
                    results["data_collected"] += 1
                    
                    # Make prediction (mock)
                    # In real implementation would process data through model
                    results["predictions_made"] += 1
                    
            except Exception as e:
                results["errors"].append(str(e))
                await asyncio.sleep(1)
        
        # Disconnect
        await ws.disconnect()
        
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
    
    return results


class PipelineOrchestrator:
    """Orchestrate the full trading pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.components = {}
    
    def setup_data_collection(self) -> None:
        """Setup data collection components."""
        from src.data_collection.binance_websocket import BinanceWebSocket
        from src.data_collection.gcs_uploader import GCSUploader
        
        self.components["websocket"] = BinanceWebSocket(
            self.config.get("symbol", "btcusdt")
        )
        
        self.components["uploader"] = GCSUploader(
            self.config.get("bucket", "btc-orderbook-data")
        )
    
    def setup_feature_engineering(self) -> None:
        """Setup feature engineering components."""
        from src.feature_engineering.engineer import FeatureEngineer
        
        self.components["feature_engineer"] = FeatureEngineer()
    
    def setup_model(self) -> None:
        """Setup model components."""
        from src.rl.models import TauSAC
        
        model_config = self.config.get("model", {})
        self.components["model"] = TauSAC(
            observation_dim=model_config.get("observation_dim", 200),
            action_dim=model_config.get("action_dim", 3),
            tau_values=model_config.get("tau_values", [3, 6, 9, 12])
        )
    
    def setup_risk_management(self) -> None:
        """Setup risk management components."""
        from src.risk_management.risk_manager import RiskManager
        from src.risk_management.models.position_sizing import KellyPositionSizer
        
        self.components["risk_manager"] = RiskManager(
            position_sizer=KellyPositionSizer()
        )
    
    async def run(self) -> Dict[str, Any]:
        """Run the full pipeline.
        
        Returns:
            Pipeline results
        """
        # Setup all components
        self.setup_data_collection()
        self.setup_feature_engineering()
        self.setup_model()
        self.setup_risk_management()
        
        # Run pipeline
        results = {
            "trades_executed": 0,
            "total_pnl": 0,
            "errors": []
        }
        
        try:
            # Connect data source
            ws = self.components["websocket"]
            await ws.connect()
            
            # Main loop
            while True:
                # Get data
                data = await ws.get_orderbook_update()
                
                # Process features
                features = self.components["feature_engineer"].compute_features(data)
                
                # Get prediction
                action = self.components["model"].predict(features)
                
                # Check risk
                risk_check = self.components["risk_manager"].check_new_position(
                    symbol=self.config["symbol"],
                    portfolio_value=100000,
                    current_price=50000,
                    signal_confidence=0.8
                )
                
                if risk_check["approved"]:
                    # Execute trade (mock)
                    results["trades_executed"] += 1
                
                # Break after some iterations (mock)
                if results["trades_executed"] > 10:
                    break
            
        except Exception as e:
            results["errors"].append(str(e))
        finally:
            await ws.disconnect()
        
        return results