"""
Circuit Breaker pattern implementation for IoT Edge Anomaly Detection.

This module provides circuit breaker functionality to prevent cascading failures
and improve system resilience by temporarily disabling failing components.
"""
import time
import logging
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout in seconds


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Monitors failures and prevents cascading failures by temporarily
    disabling failing services and allowing graceful recovery.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker for logging
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        
        logger.info(f"Circuit breaker '{name}' initialized: {self.config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            TimeoutError: If function times out
        """
        self.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.warning(f"Circuit breaker '{self.name}' is OPEN - failing fast")
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")
        
        # Execute function with timeout
        start_time = time.time()
        try:
            # Simple timeout implementation (note: this is basic, could be improved)
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > self.config.timeout:
                self.total_timeouts += 1
                self._record_failure()
                raise TimeoutError(f"Function execution timed out after {execution_time:.2f}s")
            
            self._record_success()
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Circuit breaker '{self.name}' - function failed after {execution_time:.2f}s: {e}")
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record a successful execution."""
        self.total_successes += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"Circuit breaker '{self.name}' - success in HALF_OPEN ({self.success_count}/{self.config.success_threshold})")
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _record_failure(self):
        """Record a failed execution."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Circuit breaker '{self.name}' - failure recorded ({self.failure_count}/{self.config.failure_threshold})")
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to_open()
        
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' -> CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit breaker '{self.name}' -> OPEN")
        self.state = CircuitState.OPEN
        self.success_count = 0
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        uptime = time.time() - (self.last_failure_time or time.time())
        
        failure_rate = (self.total_failures / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_timeouts": self.total_timeouts,
            "failure_rate_percent": round(failure_rate, 2),
            "current_failure_count": self.failure_count,
            "current_success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "uptime_seconds": round(uptime, 2) if self.last_failure_time else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}' manually reset")
        self._transition_to_closed()
    
    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        logger.warning(f"Circuit breaker '{self.name}' manually forced open")
        self._transition_to_open()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Name of the circuit breaker
        config: Optional configuration
        
    Returns:
        Decorated function
    """
    breaker = CircuitBreaker(name, config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to function for access to stats/control
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def register(self, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        self.breakers[breaker.name] = breaker
        logger.info(f"Registered circuit breaker: {breaker.name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered circuit breakers."""
        return {
            "circuit_breakers": {name: breaker.get_stats() 
                               for name, breaker in self.breakers.items()},
            "summary": {
                "total_breakers": len(self.breakers),
                "open_breakers": len([b for b in self.breakers.values() 
                                    if b.state == CircuitState.OPEN]),
                "half_open_breakers": len([b for b in self.breakers.values() 
                                         if b.state == CircuitState.HALF_OPEN])
            }
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()