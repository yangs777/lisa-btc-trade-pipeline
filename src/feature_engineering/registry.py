"""Indicator registry for dynamic loading and management."""

import logging
from typing import Dict, Type, Optional, List
import yaml
from pathlib import Path

from .base import BaseIndicator


logger = logging.getLogger(__name__)


class IndicatorRegistry:
    """Registry for managing and loading technical indicators."""
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._indicators: Dict[str, Type[BaseIndicator]] = {}
        self._configs: Dict[str, dict] = {}
        
    def register(self, name: str, indicator_class: Type[BaseIndicator]) -> None:
        """Register an indicator class.
        
        Args:
            name: Indicator name
            indicator_class: Indicator class
        """
        if name in self._indicators:
            logger.warning(f"Overwriting existing indicator: {name}")
        self._indicators[name] = indicator_class
        logger.debug(f"Registered indicator: {name}")
        
    def get(self, name: str) -> Optional[Type[BaseIndicator]]:
        """Get an indicator class by name.
        
        Args:
            name: Indicator name
            
        Returns:
            Indicator class or None if not found
        """
        return self._indicators.get(name)
        
    def list_indicators(self) -> List[str]:
        """Get list of registered indicator names."""
        return list(self._indicators.keys())
        
    def load_config(self, config_path: str) -> None:
        """Load indicator configurations from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Store configurations by indicator name
        for category, indicators in config.items():
            for indicator in indicators:
                self._configs[indicator['name']] = {
                    'class': indicator['class'],
                    'params': indicator.get('params', {}),
                    'category': category
                }
                
        logger.info(f"Loaded {len(self._configs)} indicator configurations")
        
    def create_indicator(self, name: str) -> Optional[BaseIndicator]:
        """Create an indicator instance from configuration.
        
        Args:
            name: Indicator name from config
            
        Returns:
            Indicator instance or None if not found
        """
        if name not in self._configs:
            logger.error(f"No configuration found for indicator: {name}")
            return None
            
        config = self._configs[name]
        class_name = config['class']
        params = config['params']
        
        indicator_class = self._indicators.get(class_name)
        if not indicator_class:
            logger.error(f"No class registered for: {class_name}")
            return None
            
        try:
            return indicator_class(**params)
        except Exception as e:
            logger.error(f"Failed to create indicator {name}: {e}")
            return None
            
    def create_all_indicators(self) -> Dict[str, BaseIndicator]:
        """Create all configured indicators.
        
        Returns:
            Dictionary of indicator instances by name
        """
        indicators = {}
        
        for name in self._configs:
            indicator = self.create_indicator(name)
            if indicator:
                indicators[name] = indicator
            else:
                logger.warning(f"Skipping indicator: {name}")
                
        logger.info(f"Created {len(indicators)} indicators")
        return indicators


# Global registry instance
registry = IndicatorRegistry()
