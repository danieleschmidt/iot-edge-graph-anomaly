"""
Configuration validation for IoT Edge Anomaly Detection.

This module provides comprehensive validation for application configuration
to ensure robustness and prevent runtime errors from invalid configurations.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Configuration validation issue."""
    severity: ValidationSeverity
    field: str
    message: str
    suggested_value: Optional[Any] = None


class ConfigValidator:
    """
    Comprehensive configuration validator for IoT Edge Anomaly Detection.
    
    Validates all configuration sections and provides detailed feedback
    on issues with suggestions for fixes.
    """
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate complete configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation issues found
        """
        self.issues = []
        
        # Validate main sections
        self._validate_model_config(config.get('model', {}))
        self._validate_monitoring_config(config.get('monitoring', {}))
        self._validate_processing_config(config.get('processing', {}))
        self._validate_health_config(config.get('health', {}))
        self._validate_graph_config(config.get('graph', {}))
        self._validate_thresholds(config)
        
        return self.issues
    
    def _add_issue(self, severity: ValidationSeverity, field: str, message: str, suggested_value: Any = None):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(severity, field, message, suggested_value))
    
    def _validate_model_config(self, model_config: Dict[str, Any]):
        """Validate model configuration section."""
        # Input size validation
        input_size = model_config.get('input_size')
        if input_size is None:
            self._add_issue(ValidationSeverity.ERROR, 'model.input_size', 
                          'Input size is required', 5)
        elif not isinstance(input_size, int) or input_size <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'model.input_size',
                          'Input size must be a positive integer', 5)
        elif input_size > 100:
            self._add_issue(ValidationSeverity.WARNING, 'model.input_size',
                          'Large input size may impact performance')
        
        # Hidden size validation
        hidden_size = model_config.get('hidden_size', 64)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'model.hidden_size',
                          'Hidden size must be a positive integer', 64)
        elif hidden_size < 8:
            self._add_issue(ValidationSeverity.WARNING, 'model.hidden_size',
                          'Small hidden size may limit model capacity', 32)
        elif hidden_size > 512:
            self._add_issue(ValidationSeverity.WARNING, 'model.hidden_size',
                          'Large hidden size may impact edge device performance', 128)
        
        # Number of layers validation
        num_layers = model_config.get('num_layers', 2)
        if not isinstance(num_layers, int) or num_layers <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'model.num_layers',
                          'Number of layers must be a positive integer', 2)
        elif num_layers > 4:
            self._add_issue(ValidationSeverity.WARNING, 'model.num_layers',
                          'Many layers may impact inference speed on edge devices', 2)
        
        # Dropout validation
        dropout = model_config.get('dropout', 0.1)
        if not isinstance(dropout, (int, float)) or not (0 <= dropout < 1):
            self._add_issue(ValidationSeverity.ERROR, 'model.dropout',
                          'Dropout must be a number between 0 and 1', 0.1)
        
        # Sequence length validation
        seq_len = model_config.get('sequence_length', 20)
        if not isinstance(seq_len, int) or seq_len <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'model.sequence_length',
                          'Sequence length must be a positive integer', 20)
        elif seq_len < 5:
            self._add_issue(ValidationSeverity.WARNING, 'model.sequence_length',
                          'Short sequences may not capture temporal patterns well', 10)
        elif seq_len > 200:
            self._add_issue(ValidationSeverity.WARNING, 'model.sequence_length',
                          'Long sequences may impact memory usage', 50)
    
    def _validate_monitoring_config(self, monitoring_config: Dict[str, Any]):
        """Validate monitoring configuration section."""
        # OTLP endpoint validation
        otlp_endpoint = monitoring_config.get('otlp_endpoint')
        if otlp_endpoint and not isinstance(otlp_endpoint, str):
            self._add_issue(ValidationSeverity.ERROR, 'monitoring.otlp_endpoint',
                          'OTLP endpoint must be a string')
        elif otlp_endpoint and not (otlp_endpoint.startswith('http://') or 
                                  otlp_endpoint.startswith('https://')):
            self._add_issue(ValidationSeverity.WARNING, 'monitoring.otlp_endpoint',
                          'OTLP endpoint should start with http:// or https://')
        
        # Service name validation
        service_name = monitoring_config.get('service_name')
        if service_name and not isinstance(service_name, str):
            self._add_issue(ValidationSeverity.ERROR, 'monitoring.service_name',
                          'Service name must be a string', 'iot-edge-anomaly')
        elif service_name and len(service_name.strip()) == 0:
            self._add_issue(ValidationSeverity.WARNING, 'monitoring.service_name',
                          'Service name should not be empty', 'iot-edge-anomaly')
    
    def _validate_processing_config(self, processing_config: Dict[str, Any]):
        """Validate processing configuration section."""
        # Loop interval validation
        loop_interval = processing_config.get('loop_interval', 5.0)
        if not isinstance(loop_interval, (int, float)) or loop_interval <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'processing.loop_interval',
                          'Loop interval must be a positive number', 5.0)
        elif loop_interval < 0.1:
            self._add_issue(ValidationSeverity.WARNING, 'processing.loop_interval',
                          'Very short loop interval may overload the system', 1.0)
        elif loop_interval > 60:
            self._add_issue(ValidationSeverity.WARNING, 'processing.loop_interval',
                          'Long loop interval may miss time-sensitive anomalies', 10.0)
        
        # Max iterations validation (optional)
        max_iterations = processing_config.get('max_iterations')
        if max_iterations is not None:
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                self._add_issue(ValidationSeverity.ERROR, 'processing.max_iterations',
                              'Max iterations must be a positive integer or null')
    
    def _validate_health_config(self, health_config: Dict[str, Any]):
        """Validate health monitoring configuration section."""
        # Memory threshold validation
        memory_threshold = health_config.get('memory_threshold_mb', 100)
        if not isinstance(memory_threshold, (int, float)) or memory_threshold <= 0:
            self._add_issue(ValidationSeverity.ERROR, 'health.memory_threshold_mb',
                          'Memory threshold must be a positive number', 100)
        elif memory_threshold < 50:
            self._add_issue(ValidationSeverity.WARNING, 'health.memory_threshold_mb',
                          'Low memory threshold may cause frequent alerts', 100)
        elif memory_threshold > 1000:
            self._add_issue(ValidationSeverity.WARNING, 'health.memory_threshold_mb',
                          'High memory threshold may not catch memory issues', 500)
        
        # CPU threshold validation
        cpu_threshold = health_config.get('cpu_threshold_percent', 80)
        if not isinstance(cpu_threshold, (int, float)) or not (0 < cpu_threshold <= 100):
            self._add_issue(ValidationSeverity.ERROR, 'health.cpu_threshold_percent',
                          'CPU threshold must be between 0 and 100', 80)
    
    def _validate_graph_config(self, graph_config: Dict[str, Any]):
        """Validate graph configuration section."""
        # Graph method validation
        method = graph_config.get('method', 'correlation')
        valid_methods = ['correlation', 'fully_connected', 'distance']
        if method not in valid_methods:
            self._add_issue(ValidationSeverity.ERROR, 'graph.method',
                          f'Graph method must be one of {valid_methods}', 'correlation')
        
        # Threshold validation
        threshold = graph_config.get('threshold', 0.5)
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            self._add_issue(ValidationSeverity.ERROR, 'graph.threshold',
                          'Graph threshold must be between 0 and 1', 0.5)
    
    def _validate_thresholds(self, config: Dict[str, Any]):
        """Validate anomaly detection thresholds."""
        # Anomaly threshold validation
        anomaly_threshold = config.get('anomaly_threshold', 0.5)
        if not isinstance(anomaly_threshold, (int, float)) or anomaly_threshold < 0:
            self._add_issue(ValidationSeverity.ERROR, 'anomaly_threshold',
                          'Anomaly threshold must be a non-negative number', 0.5)
        elif anomaly_threshold == 0:
            self._add_issue(ValidationSeverity.WARNING, 'anomaly_threshold',
                          'Zero threshold will flag everything as anomaly', 0.1)
        elif anomaly_threshold > 100:
            self._add_issue(ValidationSeverity.WARNING, 'anomaly_threshold',
                          'Very high threshold may miss anomalies', 1.0)
    
    def get_error_count(self) -> int:
        """Get count of error-level issues."""
        return len([issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR])
    
    def get_warning_count(self) -> int:
        """Get count of warning-level issues."""
        return len([issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING])
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return self.get_error_count() > 0
    
    def print_issues(self):
        """Print all validation issues in a readable format."""
        if not self.issues:
            logger.info("✅ Configuration validation passed - no issues found")
            return
        
        logger.info(f"Configuration validation found {len(self.issues)} issue(s):")
        
        for issue in self.issues:
            severity_symbol = "❌" if issue.severity == ValidationSeverity.ERROR else \
                            "⚠️ " if issue.severity == ValidationSeverity.WARNING else "ℹ️ "
            
            message = f"{severity_symbol} {issue.severity.value.upper()}: {issue.field} - {issue.message}"
            if issue.suggested_value is not None:
                message += f" (suggested: {issue.suggested_value})"
            
            if issue.severity == ValidationSeverity.ERROR:
                logger.error(message)
            elif issue.severity == ValidationSeverity.WARNING:
                logger.warning(message)
            else:
                logger.info(message)


def validate_config(config: Dict[str, Any]) -> List[ValidationIssue]:
    """
    Validate configuration and return issues.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation issues
    """
    validator = ConfigValidator()
    issues = validator.validate(config)
    validator.print_issues()
    return issues


def apply_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to configuration for missing fields.
    
    Args:
        config: Original configuration
        
    Returns:
        Configuration with defaults applied
    """
    defaults = {
        'model': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'sequence_length': 20
        },
        'monitoring': {
            'otlp_endpoint': 'http://localhost:4317',
            'service_name': 'iot-edge-anomaly'
        },
        'processing': {
            'loop_interval': 5.0
        },
        'health': {
            'memory_threshold_mb': 100,
            'cpu_threshold_percent': 80
        },
        'graph': {
            'method': 'correlation',
            'threshold': 0.5
        },
        'anomaly_threshold': 0.5
    }
    
    # Deep merge defaults with provided config
    def deep_merge(default: Dict, provided: Dict) -> Dict:
        result = default.copy()
        for key, value in provided.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(defaults, config)