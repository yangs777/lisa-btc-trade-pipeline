"""Fallback management for graceful degradation."""

from typing import Any, Callable, Dict, List, Optional


class FallbackManager:
    """Manage fallback strategies for system resilience."""
    
    def __init__(self):
        """Initialize fallback manager."""
        self.strategies: Dict[str, Callable] = {}
        self.priority_order: List[str] = []
    
    def register(self, name: str, strategy: Callable, priority: Optional[int] = None) -> None:
        """Register a fallback strategy.
        
        Args:
            name: Strategy name
            strategy: Callable strategy
            priority: Priority (lower = higher priority)
        """
        self.strategies[name] = strategy
        
        if priority is not None:
            # Insert at specific position
            self.priority_order.insert(priority, name)
        else:
            # Append to end
            if name not in self.priority_order:
                self.priority_order.append(name)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute strategies in priority order until one succeeds.
        
        Args:
            *args: Positional arguments for strategies
            **kwargs: Keyword arguments for strategies
            
        Returns:
            Result from first successful strategy
            
        Raises:
            Exception: If all strategies fail
        """
        errors = []
        
        for strategy_name in self.priority_order:
            if strategy_name not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_name]
            
            try:
                result = strategy(*args, **kwargs)
                return result
            except Exception as e:
                errors.append(f"{strategy_name}: {str(e)}")
                continue
        
        # All strategies failed
        raise Exception(f"All strategies failed: {'; '.join(errors)}")
    
    def remove(self, name: str) -> None:
        """Remove a strategy.
        
        Args:
            name: Strategy name to remove
        """
        if name in self.strategies:
            del self.strategies[name]
        if name in self.priority_order:
            self.priority_order.remove(name)
    
    def clear(self) -> None:
        """Clear all strategies."""
        self.strategies.clear()
        self.priority_order.clear()
    
    def get_strategies(self) -> List[str]:
        """Get list of registered strategies in priority order.
        
        Returns:
            List of strategy names
        """
        return self.priority_order.copy()


class CacheFallback:
    """Fallback to cached values when primary source fails."""
    
    def __init__(self, cache_ttl: float = 300):
        """Initialize cache fallback.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
    
    def get_or_compute(
        self,
        key: str,
        compute_func: Callable,
        fallback_func: Optional[Callable] = None
    ) -> Any:
        """Get value from cache or compute it.
        
        Args:
            key: Cache key
            compute_func: Function to compute value
            fallback_func: Optional fallback function
            
        Returns:
            Cached or computed value
        """
        import time
        
        # Check cache
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["value"]
        
        # Try to compute
        try:
            value = compute_func()
            
            # Update cache
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            
            return value
            
        except Exception as e:
            # Try fallback
            if fallback_func:
                return fallback_func()
            
            # Return cached value if available (even if expired)
            if key in self.cache:
                return self.cache[key]["value"]
            
            # No fallback available
            raise e
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate cache.
        
        Args:
            key: Specific key to invalidate (None = clear all)
        """
        if key is None:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]