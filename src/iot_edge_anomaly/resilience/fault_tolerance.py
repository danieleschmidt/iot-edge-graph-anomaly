"""
Fault tolerance and resilience framework for IoT edge anomaly detection.

This module provides comprehensive fault tolerance mechanisms including:
- Graceful degradation strategies
- Automatic error recovery
- State persistence and restoration
- Health monitoring integration
- Circuit breaker patterns
"""
import torch
import numpy as np
import time
import logging
import threading
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import deque
import json

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of system failures."""
    MODEL_INFERENCE_ERROR = "model_inference_error"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NETWORK_FAILURE = "network_failure"
    STORAGE_FAILURE = "storage_failure"
    HARDWARE_FAILURE = "hardware_failure"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_MODEL = "fallback_model"
    DEGRADED_MODE = "degraded_mode"
    SAFE_SHUTDOWN = "safe_shutdown"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class FailureEvent:
    """Container for failure event information."""
    timestamp: str
    failure_mode: FailureMode
    error_message: str
    context: Dict[str, Any]
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: Optional[bool] = None
    recovery_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StateManager:
    """Manages system state persistence and restoration."""
    
    def __init__(self, state_dir: str = "fault_tolerance_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_lock = threading.Lock()
        self.current_state = {}
        
        # Load existing state if available
        self._load_persistent_state()
    
    def save_state(self, key: str, value: Any, persistent: bool = True) -> bool:
        """Save state with optional persistence."""
        try:
            with self._state_lock:
                self.current_state[key] = value
                
                if persistent:
                    state_file = self.state_dir / f"{key}.pkl"
                    with open(state_file, 'wb') as f:
                        pickle.dump(value, f)
                    
                    # Also save as JSON if possible for readability
                    try:
                        json_file = self.state_dir / f"{key}.json"
                        with open(json_file, 'w') as f:
                            if isinstance(value, (dict, list, str, int, float, bool)):
                                json.dump(value, f, indent=2)
                            else:
                                json.dump(str(value), f, indent=2)
                    except Exception:
                        pass  # JSON serialization failed, but pickle succeeded
                
                return True
        except Exception as e:
            logger.error(f"Failed to save state for {key}: {e}")
            return False
    
    def load_state(self, key: str, default: Any = None) -> Any:
        """Load state with fallback to default."""
        try:
            with self._state_lock:
                if key in self.current_state:
                    return self.current_state[key]
                
                # Try to load from persistent storage
                state_file = self.state_dir / f"{key}.pkl"
                if state_file.exists():
                    with open(state_file, 'rb') as f:
                        value = pickle.load(f)
                        self.current_state[key] = value
                        return value
                
                return default
        except Exception as e:
            logger.warning(f"Failed to load state for {key}: {e}")
            return default
    
    def clear_state(self, key: str) -> bool:
        """Clear specific state."""
        try:
            with self._state_lock:
                if key in self.current_state:
                    del self.current_state[key]
                
                # Remove persistent files
                state_file = self.state_dir / f"{key}.pkl"
                if state_file.exists():
                    state_file.unlink()
                
                json_file = self.state_dir / f"{key}.json"
                if json_file.exists():
                    json_file.unlink()
                
                return True
        except Exception as e:
            logger.error(f"Failed to clear state for {key}: {e}")
            return False
    
    def _load_persistent_state(self):
        """Load all persistent state on startup."""
        try:
            for state_file in self.state_dir.glob("*.pkl"):
                key = state_file.stem
                try:
                    with open(state_file, 'rb') as f:
                        value = pickle.load(f)
                        self.current_state[key] = value
                    logger.debug(f"Loaded persistent state for {key}")
                except Exception as e:
                    logger.warning(f"Failed to load persistent state for {key}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load persistent states: {e}")


class FallbackModel:
    """Simple fallback model for degraded mode operations."""
    
    def __init__(self, input_size: int, threshold: float = 2.0):
        self.input_size = input_size
        self.threshold = threshold
        self.baseline_stats = None
        self.is_trained = False
    
    def train(self, normal_data: torch.Tensor):
        """Train simple statistical model on normal data."""
        try:
            # Compute baseline statistics
            self.baseline_stats = {
                'mean': torch.mean(normal_data, dim=(0, 1)),
                'std': torch.std(normal_data, dim=(0, 1)),
                'percentiles': {
                    'p95': torch.quantile(normal_data.flatten(), 0.95),
                    'p99': torch.quantile(normal_data.flatten(), 0.99)
                }
            }
            self.is_trained = True
            logger.info("Fallback model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to train fallback model: {e}")
            return False
    
    def predict_anomaly(self, data: torch.Tensor) -> Dict[str, Any]:
        """Simple anomaly detection using statistical thresholds."""
        if not self.is_trained:
            return {
                'is_anomaly': True,  # Conservative approach
                'confidence': 0.0,
                'method': 'untrained_fallback'
            }
        
        try:
            # Z-score based detection
            z_scores = torch.abs(data - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
            max_z_score = torch.max(z_scores)
            
            # Simple threshold-based decision
            is_anomaly = max_z_score > self.threshold
            confidence = min(max_z_score.item() / self.threshold, 1.0)
            
            return {
                'is_anomaly': is_anomaly.item(),
                'confidence': confidence,
                'max_z_score': max_z_score.item(),
                'method': 'statistical_fallback'
            }
        except Exception as e:
            logger.error(f"Fallback model prediction failed: {e}")
            return {
                'is_anomaly': True,
                'confidence': 0.0,
                'method': 'failed_fallback',
                'error': str(e)
            }


class ResilientAnomalyDetector:
    """Main resilient anomaly detection system with fault tolerance."""
    
    def __init__(self, primary_model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        self.primary_model = primary_model
        self.config = config or {}
        
        # Fault tolerance components
        self.state_manager = StateManager(
            self.config.get('state_dir', 'fault_tolerance_state')
        )
        
        # Fallback model
        input_size = self.config.get('input_size', 5)
        self.fallback_model = FallbackModel(
            input_size=input_size,
            threshold=self.config.get('fallback_threshold', 2.0)
        )
        
        # Failure tracking
        self.failure_history = deque(maxlen=self.config.get('max_failure_history', 1000))
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Performance monitoring
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_requests': 0,
            'avg_response_time_ms': 0.0,
            'last_reset_time': time.time()
        }
        
        # Circuit breaker for primary model
        from ..circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        self.primary_model_breaker = CircuitBreaker(
            "primary_model",
            CircuitBreakerConfig(
                failure_threshold=self.config.get('circuit_breaker_threshold', 5),
                recovery_timeout=self.config.get('circuit_breaker_timeout', 30.0),
                timeout=self.config.get('inference_timeout', 5.0)
            )
        )
        
        # Async executor for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("Resilient anomaly detector initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[FailureMode, RecoveryStrategy]:
        """Initialize recovery strategies for different failure modes."""
        return {
            FailureMode.MODEL_INFERENCE_ERROR: RecoveryStrategy.FALLBACK_MODEL,
            FailureMode.DATA_CORRUPTION: RecoveryStrategy.RETRY_WITH_BACKOFF,
            FailureMode.MEMORY_EXHAUSTION: RecoveryStrategy.DEGRADED_MODE,
            FailureMode.NETWORK_FAILURE: RecoveryStrategy.DEGRADED_MODE,
            FailureMode.STORAGE_FAILURE: RecoveryStrategy.DEGRADED_MODE,
            FailureMode.HARDWARE_FAILURE: RecoveryStrategy.SAFE_SHUTDOWN,
            FailureMode.UNKNOWN_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF
        }
    
    def initialize_fallback_model(self, normal_training_data: torch.Tensor) -> bool:
        """Initialize and train the fallback model."""
        try:
            success = self.fallback_model.train(normal_training_data)
            if success:
                # Save fallback model state
                self.state_manager.save_state('fallback_model_stats', 
                                             self.fallback_model.baseline_stats)
                logger.info("Fallback model initialized and saved")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize fallback model: {e}")
            return False
    
    def detect_anomaly(self, data: torch.Tensor, 
                      max_retries: int = 3,
                      timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """Main anomaly detection with fault tolerance."""
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # Input validation
        if not self._validate_input(data):
            return self._create_error_response("Invalid input data", start_time)
        
        # Try primary model first
        try:
            result = self._try_primary_model(data, max_retries, timeout_seconds)
            if result['success']:
                self.performance_stats['successful_requests'] += 1
                response_time = (time.time() - start_time) * 1000
                self._update_response_time_stats(response_time)
                return result
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            self._record_failure(FailureMode.MODEL_INFERENCE_ERROR, str(e), {'input_shape': data.shape})
        
        # Try fallback model
        try:
            result = self._try_fallback_model(data)
            if result['success']:
                self.performance_stats['fallback_requests'] += 1
                response_time = (time.time() - start_time) * 1000
                self._update_response_time_stats(response_time)
                return result
        except Exception as e:
            logger.error(f"Fallback model failed: {e}")
            self._record_failure(FailureMode.UNKNOWN_ERROR, str(e), {'fallback_failure': True})
        
        # Complete failure - return conservative result
        self.performance_stats['failed_requests'] += 1
        return self._create_conservative_response(start_time)
    
    def _validate_input(self, data: torch.Tensor) -> bool:
        """Validate input data."""
        try:
            if data.isnan().any() or data.isinf().any():
                return False
            if data.numel() == 0:
                return False
            return True
        except Exception:
            return False
    
    def _try_primary_model(self, data: torch.Tensor, 
                          max_retries: int, timeout_seconds: float) -> Dict[str, Any]:
        """Try primary model with circuit breaker protection."""
        
        def primary_inference():
            self.primary_model.eval()
            with torch.no_grad():
                if hasattr(self.primary_model, 'compute_reconstruction_error'):
                    error = self.primary_model.compute_reconstruction_error(data, reduction='mean')
                    return error.item()
                elif hasattr(self.primary_model, 'compute_hybrid_anomaly_score'):
                    score = self.primary_model.compute_hybrid_anomaly_score(data, reduction='mean')
                    return score.item()
                else:
                    # Generic approach
                    reconstruction = self.primary_model(data)
                    error = torch.mean((data - reconstruction) ** 2)
                    return error.item()
        
        try:
            # Use circuit breaker
            anomaly_score = self.primary_model_breaker.call(primary_inference)
            
            # Determine anomaly based on threshold
            threshold = self.config.get('anomaly_threshold', 0.5)
            is_anomaly = anomaly_score > threshold
            
            return {
                'success': True,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'confidence': min(anomaly_score / threshold, 1.0) if is_anomaly else 1.0 - (anomaly_score / threshold),
                'method': 'primary_model',
                'model_type': type(self.primary_model).__name__
            }
            
        except Exception as e:
            raise e  # Let caller handle the exception
    
    def _try_fallback_model(self, data: torch.Tensor) -> Dict[str, Any]:
        """Try fallback model."""
        try:
            result = self.fallback_model.predict_anomaly(data)
            return {
                'success': True,
                'is_anomaly': result['is_anomaly'],
                'anomaly_score': result.get('max_z_score', 0.0),
                'confidence': result['confidence'],
                'method': result['method'],
                'model_type': 'fallback_statistical'
            }
        except Exception as e:
            raise e
    
    def _create_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error response."""
        response_time = (time.time() - start_time) * 1000
        return {
            'success': False,
            'is_anomaly': True,  # Conservative approach
            'anomaly_score': float('inf'),
            'confidence': 0.0,
            'method': 'error_response',
            'error_message': error_message,
            'response_time_ms': response_time
        }
    
    def _create_conservative_response(self, start_time: float) -> Dict[str, Any]:
        """Create conservative response when all methods fail."""
        response_time = (time.time() - start_time) * 1000
        return {
            'success': False,
            'is_anomaly': True,  # Conservative - assume anomaly when uncertain
            'anomaly_score': float('inf'),
            'confidence': 0.0,
            'method': 'conservative_fallback',
            'message': 'All detection methods failed - assuming anomaly for safety',
            'response_time_ms': response_time
        }
    
    def _record_failure(self, failure_mode: FailureMode, error_message: str, context: Dict[str, Any]):
        """Record failure event for analysis."""
        failure_event = FailureEvent(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            failure_mode=failure_mode,
            error_message=error_message,
            context=context
        )
        
        self.failure_history.append(failure_event)
        
        # Save to persistent storage
        try:
            failures_list = self.state_manager.load_state('failure_history', [])
            failures_list.append(failure_event.to_dict())
            
            # Keep only recent failures
            max_history = self.config.get('max_persistent_failures', 100)
            if len(failures_list) > max_history:
                failures_list = failures_list[-max_history:]
            
            self.state_manager.save_state('failure_history', failures_list)
        except Exception as e:
            logger.warning(f"Failed to save failure history: {e}")
    
    def _update_response_time_stats(self, response_time_ms: float):
        """Update response time statistics."""
        # Simple moving average
        alpha = 0.1  # Smoothing factor
        self.performance_stats['avg_response_time_ms'] = (
            alpha * response_time_ms + 
            (1 - alpha) * self.performance_stats['avg_response_time_ms']
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        total_requests = self.performance_stats['total_requests']
        
        return {
            'status': 'healthy' if total_requests == 0 or 
                     (self.performance_stats['successful_requests'] / total_requests) > 0.8 
                     else 'degraded',
            'performance_stats': self.performance_stats.copy(),
            'circuit_breaker_status': {
                'state': self.primary_model_breaker.state.name,
                'failure_count': self.primary_model_breaker.failure_count,
                'last_failure_time': self.primary_model_breaker.last_failure_time
            },
            'failure_summary': {
                'recent_failures': len(self.failure_history),
                'failure_modes': list(set(f.failure_mode.value for f in self.failure_history))
            },
            'fallback_model_status': 'trained' if self.fallback_model.is_trained else 'untrained'
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_requests': 0,
            'avg_response_time_ms': 0.0,
            'last_reset_time': time.time()
        }
        logger.info("Performance statistics reset")
    
    def export_failure_analysis(self, output_path: str) -> bool:
        """Export failure analysis report."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Aggregate failure data
            failure_data = [f.to_dict() for f in self.failure_history]
            
            analysis = {
                'summary': {
                    'total_failures': len(failure_data),
                    'failure_modes': {},
                    'time_range': {
                        'start': failure_data[0]['timestamp'] if failure_data else None,
                        'end': failure_data[-1]['timestamp'] if failure_data else None
                    }
                },
                'performance_stats': self.performance_stats,
                'health_status': self.get_health_status(),
                'detailed_failures': failure_data
            }
            
            # Count failure modes
            for failure in failure_data:
                mode = failure['failure_mode']
                analysis['summary']['failure_modes'][mode] = analysis['summary']['failure_modes'].get(mode, 0) + 1
            
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Failure analysis exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export failure analysis: {e}")
            return False
    
    def shutdown(self):
        """Graceful shutdown."""
        try:
            # Save current state
            self.state_manager.save_state('performance_stats', self.performance_stats)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Resilient anomaly detector shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")