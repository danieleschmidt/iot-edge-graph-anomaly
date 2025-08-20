"""
Robust IoT Edge Anomaly Detection System.

This module integrates all robustness features into the main anomaly detection
system, providing production-grade reliability, monitoring, and fault tolerance.

Key Features:
- Health monitoring and alerting
- Circuit breaker protection
- Data validation and quality checks
- Error recovery and graceful degradation
- Performance monitoring and optimization
- Comprehensive logging and metrics
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np

from .models.advanced_hybrid_integration import (
    create_advanced_hybrid_system, AdvancedEnsembleModel
)
from .robustness.health_monitor import HealthMonitor
from .robustness.circuit_breaker import circuit_breaker, CircuitBreakerError
from .robustness.data_validator import DataValidator, ValidationStatus

logger = logging.getLogger(__name__)


class RobustAnomalyDetectionSystem:
    """
    Production-ready anomaly detection system with comprehensive robustness features.
    
    Integrates advanced ML models with production-grade reliability features:
    - Health monitoring and system diagnostics
    - Circuit breaker protection for critical components
    - Data validation and quality assurance
    - Error recovery and graceful degradation
    - Performance monitoring and alerting
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        robustness_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize robust anomaly detection system.
        
        Args:
            model_config: Configuration for the ensemble model
            robustness_config: Configuration for robustness features
        """
        self.model_config = model_config
        self.robustness_config = robustness_config or self._default_robustness_config()
        
        # Initialize core components
        self.ensemble_model = None
        self.health_monitor = None
        self.data_validator = None
        
        # System state
        self.is_initialized = False
        self.is_healthy = True
        self.degraded_mode = False
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.prediction_times = []
        
        # Initialize system
        self._initialize_system()
        
        logger.info("Robust anomaly detection system initialized")
    
    def _default_robustness_config(self) -> Dict[str, Any]:
        """Default robustness configuration."""
        return {
            "health_monitoring": {
                "enabled": True,
                "check_interval": 30.0,
                "alert_thresholds": {
                    "cpu_usage": 80.0,
                    "memory_usage": 85.0,
                    "error_rate": 0.05,
                    "model_accuracy": 0.8
                }
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout": 60.0
            },
            "data_validation": {
                "enabled": True,
                "quality_thresholds": {
                    "missing_ratio": 0.05,
                    "outlier_ratio": 0.10,
                    "drift_threshold": 0.15
                }
            },
            "performance_monitoring": {
                "enabled": True,
                "max_prediction_time": 5.0,
                "alert_on_degradation": True
            },
            "graceful_degradation": {
                "enabled": True,
                "fallback_strategies": ["simple_model", "statistical", "last_known_good"]
            }
        }
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            # Initialize ensemble model
            self.ensemble_model = create_advanced_hybrid_system(self.model_config)
            
            # Initialize health monitor
            if self.robustness_config["health_monitoring"]["enabled"]:
                self.health_monitor = HealthMonitor(
                    check_interval=self.robustness_config["health_monitoring"]["check_interval"],
                    alert_thresholds=self.robustness_config["health_monitoring"]["alert_thresholds"]
                )
                self.health_monitor.start_monitoring()
            
            # Initialize data validator
            if self.robustness_config["data_validation"]["enabled"]:
                self.data_validator = DataValidator(
                    quality_thresholds=self.robustness_config["data_validation"]["quality_thresholds"]
                )
            
            self.is_initialized = True
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.is_initialized = False
            self.is_healthy = False
            raise
    
    @circuit_breaker(
        name="ensemble_prediction",
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=Exception
    )
    def predict(
        self,
        sensor_data: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        sensor_metadata: Optional[Dict[str, torch.Tensor]] = None,
        return_explanations: bool = False,
        validate_input: bool = True
    ) -> Dict[str, Any]:
        """
        Make robust anomaly predictions with comprehensive error handling.
        
        Args:
            sensor_data: Input sensor data
            edge_index: Optional graph connectivity
            sensor_metadata: Optional sensor metadata
            return_explanations: Whether to return model explanations
            validate_input: Whether to validate input data
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Update health monitoring
            if self.health_monitor:
                self.health_monitor.record_request(success=True)
            
            # Input validation
            if validate_input and self.data_validator:
                validation_result = self._validate_input_data(
                    sensor_data, edge_index, sensor_metadata
                )
                
                if validation_result.status == ValidationStatus.FAILED:
                    raise ValueError(f"Input validation failed: {validation_result.issues}")
                
                # Log warnings but continue
                if validation_result.has_warnings():
                    logger.warning(f"Input validation warnings: {validation_result.issues}")
            
            # Make prediction
            if return_explanations:
                prediction, explanations = self.ensemble_model.predict(
                    sensor_data,
                    edge_index=edge_index,
                    sensor_metadata=sensor_metadata,
                    return_individual=True,
                    return_explanations=True
                )
            else:
                prediction = self.ensemble_model.predict(
                    sensor_data,
                    edge_index=edge_index,
                    sensor_metadata=sensor_metadata,
                    return_individual=True,
                    return_explanations=False
                )
                explanations = None
            
            # Performance monitoring
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # Keep only recent prediction times
            if len(self.prediction_times) > 100:
                self.prediction_times = self.prediction_times[-100:]
            
            # Check for performance degradation
            if self.robustness_config["performance_monitoring"]["enabled"]:
                max_time = self.robustness_config["performance_monitoring"]["max_prediction_time"]
                if prediction_time > max_time:
                    logger.warning(f"Slow prediction detected: {prediction_time:.2f}s > {max_time}s")
            
            # Update health monitoring with prediction
            if self.health_monitor:
                # Record prediction for accuracy tracking (if we had ground truth)
                self.health_monitor.record_prediction(
                    prediction.anomaly_score.mean().item()
                )
            
            # Prepare response
            response = {
                "anomaly_score": prediction.anomaly_score,
                "confidence": prediction.ensemble_confidence,
                "processing_time": prediction_time,
                "model_weights": prediction.model_weights,
                "individual_predictions": prediction.individual_predictions,
                "system_health": self.get_system_health_summary(),
                "metadata": {
                    "request_id": self.request_count,
                    "timestamp": time.time(),
                    "degraded_mode": self.degraded_mode,
                    "model_count": len(prediction.individual_predictions)
                }
            }
            
            if explanations:
                response["explanations"] = explanations
            
            if validate_input and self.data_validator:
                response["data_validation"] = {
                    "status": validation_result.status.value,
                    "quality_score": validation_result.quality_score,
                    "issues": [
                        {"field": issue.field, "severity": issue.severity.value, "message": issue.message}
                        for issue in validation_result.issues
                    ]
                }
            
            return response
            
        except CircuitBreakerError as e:
            # Circuit breaker is open - use fallback
            logger.warning(f"Circuit breaker open, using fallback: {e}")
            return self._fallback_prediction(sensor_data, start_time)
            
        except Exception as e:
            # Handle prediction errors
            self.error_count += 1
            if self.health_monitor:
                self.health_monitor.record_request(success=False)
            
            logger.error(f"Prediction error: {e}")
            
            # Try fallback if available
            if self.robustness_config["graceful_degradation"]["enabled"]:
                try:
                    return self._fallback_prediction(sensor_data, start_time)
                except Exception as fallback_error:
                    logger.error(f"Fallback prediction failed: {fallback_error}")
            
            # Re-raise if no fallback worked
            raise
    
    def _validate_input_data(
        self,
        sensor_data: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        sensor_metadata: Optional[Dict[str, torch.Tensor]]
    ):
        """Validate input data quality and format."""
        data_dict = {"sensor_data": sensor_data}
        
        if edge_index is not None:
            data_dict["edge_index"] = edge_index
        
        if sensor_metadata is not None:
            data_dict["metadata"] = sensor_metadata
        
        # Validate and update reference statistics periodically
        update_reference = (self.request_count % 100 == 0)
        
        validation_result = self.data_validator.validate(
            data_dict,
            update_reference=update_reference
        )
        
        # Record data quality for health monitoring
        if self.health_monitor:
            self.health_monitor.record_data_quality(validation_result.quality_score)
        
        return validation_result
    
    def _fallback_prediction(
        self,
        sensor_data: torch.Tensor,
        start_time: float
    ) -> Dict[str, Any]:
        """Provide fallback prediction when main system fails."""
        logger.info("Using fallback prediction strategy")
        self.degraded_mode = True
        
        strategies = self.robustness_config["graceful_degradation"]["fallback_strategies"]
        
        # Try fallback strategies in order
        for strategy in strategies:
            try:
                if strategy == "simple_model":
                    # Use simple statistical model
                    anomaly_score = self._simple_statistical_anomaly_detection(sensor_data)
                elif strategy == "statistical":
                    # Use basic statistical thresholds
                    anomaly_score = self._statistical_threshold_detection(sensor_data)
                elif strategy == "last_known_good":
                    # Return conservative prediction
                    anomaly_score = torch.tensor([0.3])  # Moderate anomaly score
                else:
                    continue
                
                processing_time = time.time() - start_time
                
                return {
                    "anomaly_score": anomaly_score,
                    "confidence": torch.tensor([0.5]),  # Low confidence for fallback
                    "processing_time": processing_time,
                    "model_weights": {},
                    "individual_predictions": [],
                    "system_health": self.get_system_health_summary(),
                    "metadata": {
                        "request_id": self.request_count,
                        "timestamp": time.time(),
                        "degraded_mode": True,
                        "fallback_strategy": strategy,
                        "model_count": 0
                    },
                    "fallback_used": True,
                    "fallback_strategy": strategy
                }
                
            except Exception as e:
                logger.warning(f"Fallback strategy '{strategy}' failed: {e}")
                continue
        
        # If all fallbacks fail, return error response
        raise RuntimeError("All fallback strategies failed")
    
    def _simple_statistical_anomaly_detection(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """Simple statistical anomaly detection as fallback."""
        # Z-score based detection
        mean = sensor_data.mean()
        std = sensor_data.std()
        
        if std < 1e-6:  # Avoid division by zero
            return torch.tensor([0.0])
        
        z_scores = torch.abs((sensor_data - mean) / std)
        max_z_score = z_scores.max()
        
        # Convert z-score to anomaly probability
        anomaly_score = torch.sigmoid((max_z_score - 2) / 2)  # Threshold at z=2
        
        return anomaly_score.unsqueeze(0)
    
    def _statistical_threshold_detection(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """Threshold-based anomaly detection as fallback."""
        # Simple range-based detection
        data_range = sensor_data.max() - sensor_data.min()
        
        # Anomaly if range is too large (indicating potential outliers)
        threshold = 10.0  # Configurable threshold
        anomaly_score = torch.clamp(data_range / threshold, 0, 1)
        
        return anomaly_score.unsqueeze(0)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        health_summary = {
            "overall_status": "healthy" if self.is_healthy else "unhealthy",
            "is_initialized": self.is_initialized,
            "degraded_mode": self.degraded_mode,
            "error_rate": self.error_count / max(1, self.request_count),
            "total_requests": self.request_count,
            "total_errors": self.error_count
        }
        
        # Add health monitor data
        if self.health_monitor:
            health_data = self.health_monitor.get_health_summary()
            health_summary.update(health_data)
        
        # Add performance metrics
        if self.prediction_times:
            health_summary["performance"] = {
                "avg_prediction_time": np.mean(self.prediction_times),
                "max_prediction_time": np.max(self.prediction_times),
                "min_prediction_time": np.min(self.prediction_times),
                "recent_predictions": len(self.prediction_times)
            }
        
        # Add data validation summary
        if self.data_validator:
            validation_summary = self.data_validator.get_validation_summary()
            health_summary["data_validation"] = validation_summary
        
        return health_summary
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_check = {
            "timestamp": time.time(),
            "system_status": "healthy" if self.is_healthy else "unhealthy",
            "components": {}
        }
        
        # Check ensemble model
        try:
            if self.ensemble_model:
                # Simple model check
                test_data = torch.randn(1, 10, 5)
                _ = self.ensemble_model.predict(test_data)
                health_check["components"]["ensemble_model"] = "healthy"
            else:
                health_check["components"]["ensemble_model"] = "not_initialized"
        except Exception as e:
            health_check["components"]["ensemble_model"] = f"error: {str(e)}"
        
        # Check health monitor
        if self.health_monitor:
            health_check["components"]["health_monitor"] = "healthy" if self.health_monitor.is_healthy() else "unhealthy"
        else:
            health_check["components"]["health_monitor"] = "disabled"
        
        # Check data validator
        if self.data_validator:
            health_check["components"]["data_validator"] = "healthy"
        else:
            health_check["components"]["data_validator"] = "disabled"
        
        # Add system metrics
        health_check.update(self.get_system_health_summary())
        
        return health_check
    
    def reset_system_state(self):
        """Reset system state and metrics."""
        self.request_count = 0
        self.error_count = 0
        self.prediction_times.clear()
        self.degraded_mode = False
        
        if self.health_monitor:
            self.health_monitor.reset_metrics()
        
        if self.data_validator:
            self.data_validator.reset_reference()
        
        logger.info("System state reset")
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down robust anomaly detection system")
        
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        # Clean up resources
        self.is_initialized = False
        self.is_healthy = False
        
        logger.info("System shutdown completed")


def create_robust_system(
    model_config: Dict[str, Any],
    robustness_config: Optional[Dict[str, Any]] = None
) -> RobustAnomalyDetectionSystem:
    """
    Factory function to create a robust anomaly detection system.
    
    Args:
        model_config: Configuration for the ensemble model
        robustness_config: Configuration for robustness features
        
    Returns:
        RobustAnomalyDetectionSystem instance
    """
    return RobustAnomalyDetectionSystem(model_config, robustness_config)


# Example usage and configuration
def get_production_config() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get production-ready configuration."""
    
    model_config = {
        'enable_transformer_vae': True,
        'enable_sparse_gat': True,
        'enable_physics_informed': True,
        'enable_self_supervised': False,  # Disable for production stability
        'ensemble_method': 'dynamic_weighting',
        'transformer_vae': {
            'input_size': 5,
            'd_model': 64,
            'num_layers': 2,
            'latent_dim': 32
        },
        'sparse_gat': {
            'input_dim': 32,
            'hidden_dim': 64,
            'output_dim': 32,
            'num_layers': 2
        },
        'physics_informed': {
            'input_size': 5,
            'hidden_size': 64,
            'latent_dim': 32
        },
        'uncertainty_quantification': {
            'mc_dropout_samples': 5,
            'temperature_scaling': True
        }
    }
    
    robustness_config = {
        "health_monitoring": {
            "enabled": True,
            "check_interval": 30.0,
            "alert_thresholds": {
                "cpu_usage": 85.0,
                "memory_usage": 90.0,
                "error_rate": 0.03,
                "model_accuracy": 0.85
            }
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 3,
            "recovery_timeout": 120.0
        },
        "data_validation": {
            "enabled": True,
            "quality_thresholds": {
                "missing_ratio": 0.03,
                "outlier_ratio": 0.08,
                "drift_threshold": 0.12
            }
        },
        "performance_monitoring": {
            "enabled": True,
            "max_prediction_time": 3.0,
            "alert_on_degradation": True
        },
        "graceful_degradation": {
            "enabled": True,
            "fallback_strategies": ["simple_model", "statistical"]
        }
    }
    
    return model_config, robustness_config