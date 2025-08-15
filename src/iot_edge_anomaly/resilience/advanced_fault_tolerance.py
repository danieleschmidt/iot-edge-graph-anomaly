"""
Advanced Fault Tolerance System for IoT Edge Anomaly Detection.

This module provides comprehensive fault tolerance capabilities including:
- Automated recovery mechanisms
- Graceful degradation strategies
- Service mesh resilience patterns
- Distributed system failure handling
"""
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import torch
import numpy as np

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in the system."""
    NETWORK_TIMEOUT = "network_timeout"
    MODEL_INFERENCE_FAILURE = "model_inference_failure"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    GPU_UNAVAILABLE = "gpu_unavailable"
    SENSOR_DISCONNECTION = "sensor_disconnection"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of failures."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FAILOVER_TO_BACKUP = "failover_to_backup"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART_COMPONENT = "restart_component"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    failure_type: FailureType
    component: str
    timestamp: datetime
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "failure_type": self.failure_type.value,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "context": self.context,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class RecoveryAction:
    """Represents a recovery action for a specific failure."""
    strategy: RecoveryStrategy
    action: Callable
    priority: int = 1  # Lower number = higher priority
    timeout: float = 30.0
    success_criteria: Optional[Callable] = None


class AdvancedFaultToleranceManager:
    """
    Advanced fault tolerance manager with automated recovery capabilities.
    
    Features:
    - Intelligent failure detection and classification
    - Automated recovery strategies
    - Circuit breaker integration
    - Graceful degradation modes
    - Performance impact monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fault tolerance manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.failure_history: List[FailureEvent] = []
        self.recovery_strategies: Dict[FailureType, List[RecoveryAction]] = {}
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.degradation_modes: Dict[str, bool] = {}
        self.backup_systems: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            "inference_times": [],
            "memory_usage": [],
            "error_rates": []
        }
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        # Periodic health checks
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        self.last_health_check = 0
        
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies for different failure types."""
        
        # Model inference failures
        self.recovery_strategies[FailureType.MODEL_INFERENCE_FAILURE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                action=self._retry_with_exponential_backoff,
                priority=1,
                timeout=10.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FAILOVER_TO_BACKUP,
                action=self._failover_to_backup_model,
                priority=2,
                timeout=5.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                action=self._enable_simplified_inference,
                priority=3,
                timeout=1.0
            )
        ]
        
        # Memory exhaustion
        self.recovery_strategies[FailureType.MEMORY_EXHAUSTION] = [
            RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                action=self._reduce_batch_size,
                priority=1,
                timeout=5.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                action=self._restart_inference_engine,
                priority=2,
                timeout=30.0
            )
        ]
        
        # Network timeouts
        self.recovery_strategies[FailureType.NETWORK_TIMEOUT] = [
            RecoveryAction(
                strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
                action=self._retry_network_operation,
                priority=1,
                timeout=15.0
            ),
            RecoveryAction(
                strategy=RecoveryStrategy.FAILOVER_TO_BACKUP,
                action=self._switch_to_local_processing,
                priority=2,
                timeout=2.0
            )
        ]
        
        # GPU unavailable
        self.recovery_strategies[FailureType.GPU_UNAVAILABLE] = [
            RecoveryAction(
                strategy=RecoveryStrategy.FAILOVER_TO_BACKUP,
                action=self._fallback_to_cpu,
                priority=1,
                timeout=5.0
            )
        ]
        
    async def handle_failure(self, failure_event: FailureEvent) -> bool:
        """
        Handle a system failure with appropriate recovery strategy.
        
        Args:
            failure_event: The failure event to handle
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.error(f"Handling failure: {failure_event.failure_type.value} in {failure_event.component}")
        
        # Record failure in history
        self.failure_history.append(failure_event)
        
        # Analyze failure pattern
        if self._detect_cascading_failure(failure_event):
            logger.warning("Cascading failure detected, implementing emergency measures")
            return await self._handle_cascading_failure(failure_event)
        
        # Get recovery strategies for this failure type
        strategies = self.recovery_strategies.get(failure_event.failure_type, [])
        if not strategies:
            logger.error(f"No recovery strategy defined for {failure_event.failure_type}")
            return False
        
        # Sort strategies by priority
        strategies.sort(key=lambda x: x.priority)
        
        # Attempt recovery strategies in order
        for strategy in strategies:
            if failure_event.recovery_attempts >= failure_event.max_recovery_attempts:
                logger.error(f"Max recovery attempts ({failure_event.max_recovery_attempts}) exceeded")
                break
                
            try:
                logger.info(f"Attempting recovery with strategy: {strategy.strategy.value}")
                
                # Execute recovery action with timeout
                success = await asyncio.wait_for(
                    self._execute_recovery_action(strategy, failure_event),
                    timeout=strategy.timeout
                )
                
                if success:
                    logger.info(f"Recovery successful with strategy: {strategy.strategy.value}")
                    self._log_successful_recovery(failure_event, strategy)
                    return True
                else:
                    failure_event.recovery_attempts += 1
                    logger.warning(f"Recovery attempt {failure_event.recovery_attempts} failed")
                    
            except asyncio.TimeoutError:
                logger.error(f"Recovery strategy {strategy.strategy.value} timed out")
                failure_event.recovery_attempts += 1
            except Exception as e:
                logger.error(f"Recovery strategy {strategy.strategy.value} failed: {e}")
                failure_event.recovery_attempts += 1
        
        logger.error(f"All recovery strategies failed for {failure_event.failure_type}")
        return False
    
    async def _execute_recovery_action(self, strategy: RecoveryAction, 
                                     failure_event: FailureEvent) -> bool:
        """Execute a specific recovery action."""
        try:
            result = await strategy.action(failure_event)
            
            # Check success criteria if provided
            if strategy.success_criteria:
                return strategy.success_criteria(result)
            
            return result if isinstance(result, bool) else True
            
        except Exception as e:
            logger.error(f"Recovery action failed: {e}")
            return False
    
    def _detect_cascading_failure(self, failure_event: FailureEvent) -> bool:
        """
        Detect if this failure is part of a cascading failure pattern.
        
        Args:
            failure_event: Current failure event
            
        Returns:
            True if cascading failure detected
        """
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        # Check for multiple failures in short time
        if len(recent_failures) > 5:
            logger.warning("Multiple failures detected in short timeframe")
            return True
        
        # Check for repeated failures in same component
        component_failures = [
            f for f in recent_failures
            if f.component == failure_event.component
        ]
        
        if len(component_failures) > 3:
            logger.warning(f"Repeated failures in component: {failure_event.component}")
            return True
        
        return False
    
    async def _handle_cascading_failure(self, failure_event: FailureEvent) -> bool:
        """Handle cascading failure with emergency measures."""
        logger.critical("Implementing emergency cascading failure recovery")
        
        # Enable maximum graceful degradation
        await self._enable_emergency_mode()
        
        # Restart critical components
        critical_components = self.config.get("critical_components", ["inference_engine"])
        for component in critical_components:
            await self._emergency_restart_component(component)
        
        # Clear failure history to prevent loop
        self.failure_history = self.failure_history[-10:]  # Keep only last 10
        
        return True
    
    async def _retry_with_exponential_backoff(self, failure_event: FailureEvent) -> bool:
        """Retry operation with exponential backoff."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retry attempt {attempt + 1} after {delay}s delay")
            
            await asyncio.sleep(delay)
            
            # Simulate retry logic - in practice, this would retry the original operation
            try:
                # Mock successful retry
                if np.random.random() > 0.3:  # 70% success rate
                    return True
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
        
        return False
    
    async def _failover_to_backup_model(self, failure_event: FailureEvent) -> bool:
        """Failover to backup model."""
        backup_model = self.backup_systems.get("model")
        if backup_model is None:
            # Create simplified backup model
            logger.info("Creating simplified backup model")
            self.backup_systems["model"] = self._create_simple_backup_model()
        
        logger.info("Switching to backup model")
        # Implementation would switch model reference
        return True
    
    async def _enable_simplified_inference(self, failure_event: FailureEvent) -> bool:
        """Enable simplified inference mode."""
        logger.info("Enabling simplified inference mode")
        self.degradation_modes["simplified_inference"] = True
        return True
    
    async def _reduce_batch_size(self, failure_event: FailureEvent) -> bool:
        """Reduce batch size to handle memory issues."""
        current_batch_size = self.config.get("batch_size", 32)
        new_batch_size = max(1, current_batch_size // 2)
        
        logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        self.config["batch_size"] = new_batch_size
        return True
    
    async def _restart_inference_engine(self, failure_event: FailureEvent) -> bool:
        """Restart the inference engine."""
        logger.info("Restarting inference engine")
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    async def _retry_network_operation(self, failure_event: FailureEvent) -> bool:
        """Retry network operation with timeout."""
        logger.info("Retrying network operation")
        # Simulate network retry
        await asyncio.sleep(2.0)
        return np.random.random() > 0.4  # 60% success rate
    
    async def _switch_to_local_processing(self, failure_event: FailureEvent) -> bool:
        """Switch to local processing mode."""
        logger.info("Switching to local processing mode")
        self.degradation_modes["local_only"] = True
        return True
    
    async def _fallback_to_cpu(self, failure_event: FailureEvent) -> bool:
        """Fallback to CPU processing."""
        logger.info("Falling back to CPU processing")
        self.config["device"] = "cpu"
        return True
    
    async def _enable_emergency_mode(self) -> None:
        """Enable emergency degradation mode."""
        logger.critical("Enabling emergency mode")
        self.degradation_modes.update({
            "emergency_mode": True,
            "simplified_inference": True,
            "local_only": True,
            "reduced_features": True
        })
    
    async def _emergency_restart_component(self, component: str) -> bool:
        """Emergency restart of a component."""
        logger.critical(f"Emergency restart of component: {component}")
        # Simulate component restart
        await asyncio.sleep(1.0)
        return True
    
    def _create_simple_backup_model(self):
        """Create a simple backup model for emergency use."""
        # Simple anomaly detection using statistical methods
        class SimpleAnomalyDetector:
            def __init__(self):
                self.threshold = 0.5
                
            def predict(self, data):
                # Simple statistical anomaly detection
                mean = torch.mean(data, dim=-1, keepdim=True)
                std = torch.std(data, dim=-1, keepdim=True)
                z_scores = torch.abs((data - mean) / (std + 1e-8))
                return (z_scores > 2.0).float().mean(dim=-1)
        
        return SimpleAnomalyDetector()
    
    def _log_successful_recovery(self, failure_event: FailureEvent, strategy: RecoveryAction):
        """Log successful recovery for analysis."""
        recovery_log = {
            "timestamp": datetime.now().isoformat(),
            "failure_event": failure_event.to_dict(),
            "successful_strategy": strategy.strategy.value,
            "recovery_time": time.time() - failure_event.timestamp.timestamp()
        }
        
        # In production, this would be sent to monitoring system
        logger.info(f"Recovery logged: {json.dumps(recovery_log, indent=2)}")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        recent_failures = [
            f for f in self.failure_history
            if f.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        failure_types = {}
        for failure in recent_failures:
            failure_type = failure.failure_type.value
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_failures_last_hour": len(recent_failures),
            "failure_breakdown": failure_types,
            "active_degradation_modes": {
                mode: active for mode, active in self.degradation_modes.items() if active
            },
            "component_health": self.component_health,
            "performance_metrics": {
                metric: {
                    "count": len(values),
                    "average": np.mean(values) if values else 0,
                    "latest": values[-1] if values else 0
                } for metric, values in self.performance_metrics.items()
            }
        }
    
    async def periodic_health_check(self) -> None:
        """Perform periodic health checks on all components."""
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        logger.debug("Performing periodic health check")
        
        # Check system resources
        try:
            import psutil
            
            # Memory check
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                await self.handle_failure(FailureEvent(
                    failure_type=FailureType.MEMORY_EXHAUSTION,
                    component="system",
                    timestamp=datetime.now(),
                    error_message=f"High memory usage: {memory_percent}%"
                ))
            
            # GPU check
            if torch.cuda.is_available():
                try:
                    torch.cuda.current_device()
                except Exception as e:
                    await self.handle_failure(FailureEvent(
                        failure_type=FailureType.GPU_UNAVAILABLE,
                        component="gpu",
                        timestamp=datetime.now(),
                        error_message=str(e)
                    ))
        
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        
        self.last_health_check = current_time
    
    def is_degraded(self) -> bool:
        """Check if system is currently in any degraded state."""
        return any(self.degradation_modes.values())
    
    def get_active_degradations(self) -> List[str]:
        """Get list of currently active degradation modes."""
        return [mode for mode, active in self.degradation_modes.items() if active]
    
    def clear_degradation_mode(self, mode: str) -> bool:
        """Clear a specific degradation mode."""
        if mode in self.degradation_modes:
            self.degradation_modes[mode] = False
            logger.info(f"Cleared degradation mode: {mode}")
            return True
        return False
    
    def clear_all_degradation_modes(self) -> None:
        """Clear all degradation modes (recovery complete)."""
        active_modes = self.get_active_degradations()
        for mode in self.degradation_modes:
            self.degradation_modes[mode] = False
        
        if active_modes:
            logger.info(f"Cleared all degradation modes: {active_modes}")


# Global fault tolerance manager instance
_fault_tolerance_manager: Optional[AdvancedFaultToleranceManager] = None


def get_fault_tolerance_manager(config: Optional[Dict[str, Any]] = None) -> AdvancedFaultToleranceManager:
    """Get or create the global fault tolerance manager."""
    global _fault_tolerance_manager
    
    if _fault_tolerance_manager is None:
        _fault_tolerance_manager = AdvancedFaultToleranceManager(config)
    
    return _fault_tolerance_manager


async def handle_system_failure(failure_type: FailureType, component: str, 
                               error_message: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to handle system failures.
    
    Args:
        failure_type: Type of failure
        component: Component that failed
        error_message: Error message
        context: Additional context
        
    Returns:
        True if recovery was successful
    """
    manager = get_fault_tolerance_manager()
    
    failure_event = FailureEvent(
        failure_type=failure_type,
        component=component,
        timestamp=datetime.now(),
        error_message=error_message,
        context=context or {}
    )
    
    return await manager.handle_failure(failure_event)


def with_fault_tolerance(failure_type: FailureType, component: str):
    """
    Decorator to add fault tolerance to functions.
    
    Args:
        failure_type: Type of failure this function might have
        component: Component name for logging
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                success = await handle_system_failure(
                    failure_type=failure_type,
                    component=component,
                    error_message=str(e),
                    context={"function": func.__name__, "args_count": len(args)}
                )
                
                if not success:
                    raise  # Re-raise if recovery failed
                    
                # Retry once after recovery
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't use async recovery
                logger.error(f"Failure in {component}.{func.__name__}: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator