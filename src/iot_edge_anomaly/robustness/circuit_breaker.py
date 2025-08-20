"""
Circuit Breaker Pattern for IoT Edge Anomaly Detection.

Implements circuit breaker pattern to prevent cascade failures and provide
graceful degradation when system components are failing.

Features:
- Configurable failure thresholds and recovery times
- Multiple circuit states (CLOSED, OPEN, HALF_OPEN)
- Automatic state transitions and recovery attempts
- Fallback mechanisms for failed operations
- Metrics tracking and health reporting
"""

import time
import threading
import logging
from typing import Any, Callable, Optional, Dict
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    state_change_time: float
    total_requests: int
    rejected_requests: int


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    The circuit breaker monitors calls to a protected function and can:
    - Allow calls when everything is working (CLOSED state)
    - Block calls when failures exceed threshold (OPEN state) 
    - Test recovery by allowing limited calls (HALF_OPEN state)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to recover (seconds)
            expected_exception: Exception type that counts as failure
            fallback_function: Function to call when circuit is open
            name: Name for logging and identification
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function
        self.name = name
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state_change_time = time.time()
        self.total_requests = 0
        self.rejected_requests = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open and no fallback
        """
        with self._lock:
            self.total_requests += 1
            
            # Check if we should allow the call
            if not self._should_allow_request():
                self.rejected_requests += 1
                if self.fallback_function:
                    logger.debug(f"Circuit breaker '{self.name}' using fallback")
                    return self.fallback_function(*args, **kwargs)
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is {self.state.value}"
                    )
            
            # Attempt the call
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on current state."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.recovery_timeout):
                self._transition_to_half_open()
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return True
        
        return False
    
    def _on_success(self):
        """Handle successful function call."""
        with self._lock:
            self.success_count += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on successful call
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed function call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        logger.warning(
            f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
        )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = time.time()
        self.failure_count = 0  # Reset for testing
        logger.info(f"Circuit breaker '{self.name}' entered half-open state")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.failure_count = 0
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def force_open(self):
        """Manually force circuit breaker to open state."""
        with self._lock:
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' manually opened")
    
    def force_close(self):
        """Manually force circuit breaker to closed state."""
        with self._lock:
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually closed")
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
            self.state_change_time = time.time()
            self.total_requests = 0
            self.rejected_requests = 0
            logger.info(f"Circuit breaker '{self.name}' reset")
    
    def get_metrics(self) -> CircuitMetrics:
        """Get current circuit breaker metrics."""
        with self._lock:
            return CircuitMetrics(
                state=self.state,
                failure_count=self.failure_count,
                success_count=self.success_count,
                last_failure_time=self.last_failure_time,
                last_success_time=self.last_success_time,
                state_change_time=self.state_change_time,
                total_requests=self.total_requests,
                rejected_requests=self.rejected_requests
            )
    
    def is_closed(self) -> bool:
        """Check if circuit breaker is in closed state."""
        return self.state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit breaker is in open state."""
        return self.state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is in half-open state."""
        return self.state == CircuitState.HALF_OPEN


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    Provides centralized management, monitoring, and configuration
    of multiple circuit breakers in the system.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        
    def create_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        fallback_function: Optional[Callable] = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                raise ValueError(f"Circuit breaker '{name}' already exists")
            
            circuit_breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                fallback_function=fallback_function,
                name=name
            )
            
            self.circuit_breakers[name] = circuit_breaker
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, CircuitMetrics]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: cb.get_metrics() 
                for name, cb in self.circuit_breakers.items()
            }
    
    def force_open_all(self):
        """Force all circuit breakers to open state."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.force_open()
    
    def force_close_all(self):
        """Force all circuit breakers to closed state."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.force_close()
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers."""
        metrics = self.get_all_metrics()
        
        total_breakers = len(metrics)
        open_breakers = sum(1 for m in metrics.values() if m.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for m in metrics.values() if m.state == CircuitState.HALF_OPEN)
        
        overall_health = "healthy"
        if open_breakers > 0:
            overall_health = "degraded" if open_breakers < total_breakers else "critical"
        elif half_open_breakers > 0:
            overall_health = "recovering"
        
        return {
            "overall_health": overall_health,
            "total_breakers": total_breakers,
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "closed_breakers": total_breakers - open_breakers - half_open_breakers,
            "details": {
                name: {
                    "state": metrics.state.value,
                    "failure_count": metrics.failure_count,
                    "success_count": metrics.success_count,
                    "total_requests": metrics.total_requests,
                    "rejected_requests": metrics.rejected_requests
                }
                for name, metrics in metrics.items()
            }
        }


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    fallback_function: Optional[Callable] = None
):
    """
    Decorator to add circuit breaker protection to a function.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before recovery attempt
        expected_exception: Exception type that triggers circuit breaker
        fallback_function: Function to call when circuit is open
    """
    def decorator(func: Callable) -> Callable:
        # Create or get existing circuit breaker
        cb = circuit_manager.get_circuit_breaker(name)
        if cb is None:
            cb = circuit_manager.create_circuit_breaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                fallback_function=fallback_function
            )
        
        return cb(func)
    
    return decorator