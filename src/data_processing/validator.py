"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing import Optional


class DataValidator:
    """Validate trading data integrity."""
    
    def __init__(self):
        """Initialize validator."""
        self.validation_rules = {
            "price_positive": lambda df: (df[["open", "high", "low", "close"]] > 0).all().all(),
            "high_low_consistency": lambda df: (df["high"] >= df["low"]).all(),
            "high_consistency": lambda df: (df["high"] >= df[["open", "close"]].max(axis=1)).all(),
            "low_consistency": lambda df: (df["low"] <= df[["open", "close"]].min(axis=1)).all(),
            "volume_positive": lambda df: (df["volume"] >= 0).all(),
        }
    
    def validate_ohlcv(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check data not empty
        if len(data) == 0:
            return False
        
        # Apply validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(data):
                    return False
            except Exception:
                return False
        
        return True
    
    def validate_timestamps(self, data: pd.DataFrame, timestamp_col: str = "timestamp") -> bool:
        """Validate timestamps are monotonic.
        
        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            True if valid, False otherwise
        """
        if timestamp_col not in data.columns:
            return False
        
        # Check monotonic increasing
        return data[timestamp_col].is_monotonic_increasing
    
    def get_validation_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed validation report.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Check OHLCV validity
        if not self.validate_ohlcv(data):
            report["is_valid"] = False
            report["errors"].append("OHLCV validation failed")
        
        # Collect statistics
        if "close" in data.columns:
            report["stats"]["price_range"] = [data["close"].min(), data["close"].max()]
            report["stats"]["volume_range"] = [data["volume"].min(), data["volume"].max()]
            report["stats"]["num_records"] = len(data)
        
        return report