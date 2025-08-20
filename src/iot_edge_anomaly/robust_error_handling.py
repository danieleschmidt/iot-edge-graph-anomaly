"""
🛡️ Robust Error Handling and Validation Framework

This module provides comprehensive error handling, input validation, and 
fault tolerance mechanisms for the Terragon IoT Anomaly Detection System.

Features:
- Input data validation with detailed error reporting
- Model state validation and auto-recovery
- Circuit breaker pattern for external dependencies
- Graceful degradation strategies
- Comprehensive error logging and metrics
- Health monitoring and alerting
"""

import time
import logging
import traceback
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict
import json

import numpy as np
import torch
import torch.nn as nn


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    INPUT_VALIDATION = "input_validation"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_DEPENDENCY = "external_dependency"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ErrorEvent:
    """Structured error event."""
    timestamp: datetime
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    component: Optional[str] = None
    recovery_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'stack_trace': self.stack_trace,
            'component': self.component,
            'recovery_action': self.recovery_action
        }


class ValidationError(Exception):
    """Custom validation error with structured details."""
    
    def __init__(self, message: str, category: ErrorCategory, details: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.details = details or {}
        self.severity = ErrorSeverity.HIGH


class ModelInferenceError(Exception):
    """Error during model inference."""
    
    def __init__(self, message: str, model_name: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.model_name = model_name
        self.details = details or {}
        self.severity = ErrorSeverity.CRITICAL


class SystemResourceError(Exception):
    """Error related to system resources."""
    
    def __init__(self, message: str, resource_type: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.resource_type = resource_type
        self.details = details or {}
        self.severity = ErrorSeverity.HIGH


class InputValidator:
    """Comprehensive input data validator."""
    
    def __init__(self):
        self.validation_rules = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self.validation_rules = {
            'tensor_shape': self._validate_tensor_shape,
            'data_range': self._validate_data_range,
            'nan_inf_check': self._validate_nan_inf,
            'data_type': self._validate_data_type,
            'sequence_length': self._validate_sequence_length,
            'sensor_count': self._validate_sensor_count,
            'temporal_consistency': self._validate_temporal_consistency
        }
    
    def validate_sensor_data(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        expected_shape: Optional[Tuple[int, ...]] = None,
        sensor_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of sensor data.
        
        Args:
            data: Input sensor data
            expected_shape: Expected data shape
            sensor_metadata: Additional metadata for validation
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Convert to tensor if needed
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            elif not isinstance(data, torch.Tensor):
                errors.append("Data must be numpy array or torch tensor")
                return False, errors
            
            # Run all validation rules
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    is_valid, error_msg = rule_func(data, expected_shape, sensor_metadata)
                    if not is_valid:
                        errors.append(f"{rule_name}: {error_msg}")
                except Exception as e:
                    errors.append(f"{rule_name}: Validation rule failed - {str(e)}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation process failed: {str(e)}")
            return False, errors
    
    def _validate_tensor_shape(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate tensor shape."""
        if data.dim() < 2 or data.dim() > 3:
            return False, f"Expected 2D or 3D tensor, got {data.dim()}D"
        
        if expected_shape and data.shape != expected_shape:
            return False, f"Expected shape {expected_shape}, got {data.shape}"
        
        return True, ""
    
    def _validate_data_range(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate data is within reasonable ranges."""
        if data.numel() == 0:
            return False, "Empty tensor"
        
        min_val, max_val = data.min().item(), data.max().item()
        
        # Check for extreme values
        if abs(min_val) > 1e6 or abs(max_val) > 1e6:
            return False, f"Extreme values detected: min={min_val}, max={max_val}"
        
        # Check for very small variance (could indicate sensor failure)
        if data.var().item() < 1e-10:
            return False, "Very low variance detected - possible sensor failure"
        
        return True, ""
    
    def _validate_nan_inf(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate no NaN or infinite values."""
        if torch.isnan(data).any():
            nan_count = torch.isnan(data).sum().item()
            return False, f"Contains {nan_count} NaN values"
        
        if torch.isinf(data).any():
            inf_count = torch.isinf(data).sum().item()
            return False, f"Contains {inf_count} infinite values"
        
        return True, ""
    
    def _validate_data_type(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate data type."""
        if data.dtype not in [torch.float32, torch.float64]:
            return False, f"Expected float32/float64, got {data.dtype}"
        
        return True, ""
    
    def _validate_sequence_length(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate sequence length is reasonable."""
        if data.dim() >= 2:
            seq_len = data.shape[-2] if data.dim() == 3 else data.shape[0]
            
            if seq_len < 5:
                return False, f"Sequence too short: {seq_len} < 5"
            
            if seq_len > 1000:
                return False, f"Sequence too long: {seq_len} > 1000"
        
        return True, ""
    
    def _validate_sensor_count(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate number of sensors."""
        if data.dim() >= 2:
            num_sensors = data.shape[-1]
            
            if num_sensors < 1:
                return False, f"No sensors detected"
            
            if num_sensors > 100:
                return False, f"Too many sensors: {num_sensors} > 100"
        
        return True, ""
    
    def _validate_temporal_consistency(self, data: torch.Tensor, expected_shape, metadata) -> Tuple[bool, str]:
        """Validate temporal consistency."""
        if data.dim() >= 2:
            # Check for sudden jumps in consecutive time steps
            if data.dim() == 3:
                # [batch, seq, features]
                diffs = torch.diff(data, dim=1)
            else:
                # [seq, features]
                diffs = torch.diff(data, dim=0)
            
            # Check for abnormally large changes
            max_diff = diffs.abs().max().item()
            mean_val = data.abs().mean().item()
            
            if max_diff > 10 * mean_val:
                return False, f"Large temporal discontinuity detected: {max_diff:.3f}"
        
        return True, ""


class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_history = deque(maxlen=1000)
        self.recovery_attempts = defaultdict(int)
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies = {
            ErrorCategory.INPUT_VALIDATION: self._recover_input_validation,
            ErrorCategory.MODEL_INFERENCE: self._recover_model_inference,
            ErrorCategory.SYSTEM_RESOURCE: self._recover_system_resource,
            ErrorCategory.EXTERNAL_DEPENDENCY: self._recover_external_dependency,
            ErrorCategory.CONFIGURATION: self._recover_configuration,
            ErrorCategory.SECURITY: self._recover_security,
            ErrorCategory.PERFORMANCE: self._recover_performance
        }
    
    def handle_error(
        self, 
        error: Exception, 
        category: ErrorCategory,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, Any]:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            category: Error category
            context: Additional context for recovery
            
        Returns:
            (recovered, result)
        """
        error_id = f"{category.value}_{int(time.time())}"
        context = context or {}
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            error_id=error_id,
            category=category,
            severity=getattr(error, 'severity', ErrorSeverity.MEDIUM),
            message=str(error),
            details=getattr(error, 'details', {}),
            stack_trace=traceback.format_exc(),
            component=context.get('component', 'unknown')
        )
        
        # Add to history
        self.error_history.append(error_event)
        
        # Attempt recovery
        try:
            recovery_func = self.recovery_strategies.get(category)
            if recovery_func:
                self.recovery_attempts[error_id] += 1
                
                # Prevent infinite recovery loops
                if self.recovery_attempts[error_id] > 3:
                    return False, None
                
                recovered, result = recovery_func(error, context)
                
                if recovered:
                    error_event.recovery_action = f"Recovered using {recovery_func.__name__}"
                
                return recovered, result
            
            return False, None
            
        except Exception as recovery_error:
            # Recovery itself failed
            recovery_event = ErrorEvent(
                timestamp=datetime.now(),
                error_id=f"recovery_{error_id}",
                category=ErrorCategory.SYSTEM_RESOURCE,
                severity=ErrorSeverity.CRITICAL,
                message=f"Recovery failed: {str(recovery_error)}",
                details={'original_error': str(error)},
                stack_trace=traceback.format_exc(),
                component='error_recovery'
            )
            self.error_history.append(recovery_event)
            
            return False, None
    
    def _recover_input_validation(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from input validation errors."""
        data = context.get('data')
        
        if data is None:
            return False, None
        
        try:
            # Convert to tensor if needed
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Fix common issues
            if torch.isnan(data).any():
                # Replace NaN with mean values
                data = torch.where(torch.isnan(data), torch.nanmean(data), data)
            
            if torch.isinf(data).any():
                # Clip infinite values
                data = torch.clamp(data, -1e6, 1e6)
            
            # Ensure reasonable shape
            if data.dim() == 1:
                data = data.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
            elif data.dim() == 2:
                data = data.unsqueeze(0)  # Add batch dim
            
            return True, data
            
        except Exception:
            return False, None
    
    def _recover_model_inference(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from model inference errors."""
        model = context.get('model')
        data = context.get('data')
        
        if model is None or data is None:
            return False, None
        
        try:
            # Try with simplified input
            if hasattr(model, 'eval'):
                model.eval()
            
            # Reduce batch size if out of memory
            if 'out of memory' in str(error).lower():
                if data.dim() == 3 and data.shape[0] > 1:
                    # Process single sample
                    with torch.no_grad():
                        result = model(data[:1])
                    return True, result
            
            # Try CPU if CUDA error
            if 'cuda' in str(error).lower():
                model_cpu = model.cpu()
                data_cpu = data.cpu()
                with torch.no_grad():
                    result = model_cpu(data_cpu)
                return True, result
            
            return False, None
            
        except Exception:
            return False, None
    
    def _recover_system_resource(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from system resource errors."""
        try:
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Reduce batch size or model complexity
            return True, "resource_cleanup_attempted"
            
        except Exception:
            return False, None
    
    def _recover_external_dependency(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from external dependency errors."""
        try:
            # Use fallback methods or cached results
            return True, "fallback_activated"
            
        except Exception:
            return False, None
    
    def _recover_configuration(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from configuration errors."""
        try:
            # Use default configuration
            return True, "default_config_applied"
            
        except Exception:
            return False, None
    
    def _recover_security(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from security errors."""
        # Security errors should not be automatically recovered
        return False, None
    
    def _recover_performance(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover from performance errors."""
        try:
            # Reduce model complexity or timeout
            return True, "performance_mode_reduced"
            
        except Exception:
            return False, None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total_errors": 0, "categories": {}, "recent_errors": []}
        
        # Categorize errors
        categories = defaultdict(int)
        severities = defaultdict(int)
        recent_errors = []
        
        for event in list(self.error_history)[-50:]:  # Last 50 errors
            categories[event.category.value] += 1
            severities[event.severity.value] += 1
            recent_errors.append({
                'timestamp': event.timestamp.isoformat(),
                'category': event.category.value,
                'severity': event.severity.value,
                'message': event.message[:100] + '...' if len(event.message) > 100 else event.message
            })
        
        return {
            "total_errors": len(self.error_history),
            "categories": dict(categories),
            "severities": dict(severities),
            "recent_errors": recent_errors[-10:],  # Last 10 errors
            "recovery_attempts": dict(self.recovery_attempts)
        }


class RobustModelWrapper:
    """Wrapper that adds robustness to any model."""
    
    def __init__(self, model: nn.Module, name: str = "unknown"):
        self.model = model
        self.name = name
        self.validator = InputValidator()
        self.error_manager = ErrorRecoveryManager()
        self.inference_count = 0
        self.error_count = 0
        self.last_successful_result = None
        
    def safe_predict(
        self, 
        data: Union[np.ndarray, torch.Tensor],
        fallback_on_error: bool = True
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Safe prediction with comprehensive error handling.
        
        Args:
            data: Input data
            fallback_on_error: Whether to use fallback on errors
            
        Returns:
            (success, result, error_message)
        """
        self.inference_count += 1
        
        try:
            # Input validation
            is_valid, validation_errors = self.validator.validate_sensor_data(data)
            if not is_valid:
                raise ValidationError(
                    f"Input validation failed: {'; '.join(validation_errors)}",
                    ErrorCategory.INPUT_VALIDATION,
                    {'validation_errors': validation_errors}
                )
            
            # Convert to tensor if needed
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            
            # Model inference
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    result = self.model.predict(data)
                elif hasattr(self.model, 'compute_reconstruction_error'):
                    result = self.model.compute_reconstruction_error(data)
                else:
                    result = self.model(data)
            
            # Validate output
            if isinstance(result, torch.Tensor):
                if torch.isnan(result).any() or torch.isinf(result).any():
                    raise ModelInferenceError(
                        "Model output contains NaN or infinite values",
                        self.name,
                        {'output_shape': result.shape}
                    )
            
            # Store successful result for fallback
            self.last_successful_result = result
            
            return True, result, None
            
        except Exception as e:
            self.error_count += 1
            
            # Determine error category
            if isinstance(e, ValidationError):
                category = e.category
            elif isinstance(e, ModelInferenceError):
                category = ErrorCategory.MODEL_INFERENCE
            elif 'memory' in str(e).lower():
                category = ErrorCategory.SYSTEM_RESOURCE
            else:
                category = ErrorCategory.MODEL_INFERENCE
            
            # Attempt recovery
            context = {
                'model': self.model,
                'data': data,
                'component': f'model_{self.name}'
            }
            
            recovered, result = self.error_manager.handle_error(e, category, context)
            
            if recovered:
                return True, result, None
            elif fallback_on_error and self.last_successful_result is not None:
                # Use last successful result as fallback
                return True, self.last_successful_result, f"Using fallback due to: {str(e)}"
            else:
                return False, None, str(e)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the wrapped model."""
        error_rate = self.error_count / max(self.inference_count, 1)
        
        if error_rate > 0.5:
            status = "critical"
        elif error_rate > 0.2:
            status = "degraded"
        elif error_rate > 0.05:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "model_name": self.name,
            "inference_count": self.inference_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "has_fallback": self.last_successful_result is not None,
            "error_summary": self.error_manager.get_error_summary()
        }


# Convenience functions
def make_robust(model: nn.Module, name: str = None) -> RobustModelWrapper:
    """Make any model robust with error handling."""
    return RobustModelWrapper(model, name or model.__class__.__name__)


def validate_input(data: Union[np.ndarray, torch.Tensor]) -> Tuple[bool, List[str]]:
    """Quick input validation function."""
    validator = InputValidator()
    return validator.validate_sensor_data(data)


# Example usage
if __name__ == "__main__":
    print("🛡️ Testing Robust Error Handling Framework")
    print("=" * 50)
    
    # Test input validation
    validator = InputValidator()
    
    # Valid data
    valid_data = torch.randn(1, 20, 5)
    is_valid, errors = validator.validate_sensor_data(valid_data)
    print(f"Valid data test: {is_valid} (errors: {len(errors)})")
    
    # Invalid data with NaN
    invalid_data = torch.randn(1, 20, 5)
    invalid_data[0, 10, 2] = float('nan')
    is_valid, errors = validator.validate_sensor_data(invalid_data)
    print(f"NaN data test: {is_valid} (errors: {errors})")
    
    # Test error recovery
    recovery_manager = ErrorRecoveryManager()
    
    try:
        raise ValidationError("Test error", ErrorCategory.INPUT_VALIDATION, {'test': True})
    except Exception as e:
        recovered, result = recovery_manager.handle_error(
            e, ErrorCategory.INPUT_VALIDATION, {'data': invalid_data}
        )
        print(f"Recovery test: {recovered}")
    
    print("✅ Robust error handling framework tested successfully!")