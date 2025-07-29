"""Error handling utilities."""

import time
import functools
from typing import Any, Callable, Optional, Type, Tuple


class ErrorHandler:
    """Handle errors with retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
            exceptions: Tuple of exceptions to catch
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions
    
    def with_retry(self, func: Callable) -> Callable:
        """Decorator to add retry logic to a function.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with retry logic
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt < self.max_retries - 1:
                        # Calculate backoff time
                        wait_time = self.backoff_factor ** attempt
                        time.sleep(wait_time)
                    else:
                        # Last attempt failed
                        raise last_exception
            
            # Should not reach here
            raise last_exception
        
        return wrapper
    
    def __call__(self, func: Callable) -> Callable:
        """Allow using as decorator directly."""
        return self.with_retry(func)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            # Check if we should try recovery
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset on half_open or reduce failure count
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            elif self.failure_count > 0:
                self.failure_count -= 1
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e
    
    def reset(self) -> None:
        """Reset circuit breaker state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"