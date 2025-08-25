"""
Advanced Robustness Framework for Production IoT Anomaly Detection
Implements comprehensive error handling, validation, security, and resilience patterns.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import asynccontextmanager, contextmanager
import json
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
import uuid
import traceback
import psutil
import gc
import sys
import os
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecurityLevel(Enum):
    """Security levels for operations."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(Enum):
    """Risk assessment levels."""
    MINIMAL = auto()
    LOW = auto()
    MEDIUM = auto() 
    HIGH = auto()
    CRITICAL = auto()


class ValidationSeverity(Enum):
    """Validation issue severities."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    authenticated: bool = False
    encryption_required: bool = True
    audit_required: bool = True
    request_signature: Optional[str] = None


@dataclass 
class ValidationIssue:
    """Validation issue details."""
    field: str
    severity: ValidationSeverity
    message: str
    suggested_fix: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class RobustResult(Generic[T]):
    """Robust operation result with comprehensive error handling."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    retry_count: int = 0
    security_context: Optional[SecurityContext] = None
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)


class AdvancedValidator:
    """Advanced input validation with security and business rules."""
    
    def __init__(self):
        self.validation_rules = {}
        self.security_patterns = self._load_security_patterns()
        self.business_rules = self._load_business_rules()
    
    def _load_security_patterns(self) -> Dict[str, Any]:
        """Load security validation patterns."""
        return {
            'sql_injection_patterns': [
                r"(?i)(union\s+select)",
                r"(?i)(drop\s+table)",
                r"(?i)(insert\s+into)",
                r"(?i)(delete\s+from)",
                r"(?i)(update\s+.*set)",
                r"(?i)(\bor\s+1\s*=\s*1\b)",
                r"(?i)(\bor\s+'[^']*'\s*=\s*'[^']*')",
                r"(?i)(exec\s*\()",
                r"(?i)(script\s*>)"
            ],
            'xss_patterns': [
                r"(?i)(<script[^>]*>.*?</script>)",
                r"(?i)(javascript\s*:)",
                r"(?i)(on\w+\s*=)",
                r"(?i)(<iframe[^>]*>)",
                r"(?i)(<object[^>]*>)",
                r"(?i)(<embed[^>]*>)"
            ],
            'path_traversal_patterns': [
                r"\.\.[\\/]",
                r"[\\/]\.\.[\\/]",
                r"(?i)(etc[\\/]passwd)",
                r"(?i)(windows[\\/]system32)",
                r"(?i)(\.\.[\\\/]+)",
                r"(?i)(\.\.%[0-9a-f]{2})"
            ],
            'command_injection_patterns': [
                r"(?i)(;\s*(rm|del|format|shutdown|reboot))",
                r"(?i)(\|\s*(curl|wget|nc|netcat))",
                r"(?i)(&\s*(ping|nslookup|dig))",
                r"(?i)(>\s*[\\/])",
                r"(?i)(<\s*[\\/])"
            ]
        }
    
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business validation rules."""
        return {
            'sensor_data': {
                'min_values': {'temperature': -50, 'humidity': 0, 'pressure': 0},
                'max_values': {'temperature': 150, 'humidity': 100, 'pressure': 2000},
                'required_fields': ['timestamp', 'sensor_id', 'values'],
                'max_sequence_length': 1000,
                'max_batch_size': 10000
            },
            'model_config': {
                'min_hidden_size': 8,
                'max_hidden_size': 2048,
                'min_layers': 1,
                'max_layers': 10,
                'dropout_range': [0.0, 0.9],
                'learning_rate_range': [1e-6, 1e-1]
            },
            'performance': {
                'max_inference_time_ms': 10000,
                'max_memory_usage_mb': 1000,
                'max_cpu_usage_percent': 90,
                'min_accuracy': 0.7
            }
        }
    
    def validate_sensor_data(self, data: Dict[str, Any], context: SecurityContext) -> List[ValidationIssue]:
        """Validate IoT sensor data with security checks."""
        issues = []
        
        # Security validation first
        security_issues = self._validate_security(data, context)
        issues.extend(security_issues)
        
        # Business rule validation
        rules = self.business_rules['sensor_data']
        
        # Required fields check
        for field in rules['required_fields']:
            if field not in data:
                issues.append(ValidationIssue(
                    field=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing",
                    suggested_fix=f"Include '{field}' in the data payload",
                    error_code="MISSING_REQUIRED_FIELD"
                ))
        
        # Value range validation
        if 'values' in data and isinstance(data['values'], dict):
            for sensor_type, value in data['values'].items():
                if not isinstance(value, (int, float)):
                    issues.append(ValidationIssue(
                        field=f"values.{sensor_type}",
                        severity=ValidationSeverity.ERROR,
                        message=f"Sensor value must be numeric, got {type(value).__name__}",
                        suggested_fix="Ensure sensor values are numeric",
                        error_code="INVALID_VALUE_TYPE"
                    ))
                    continue
                
                # Range validation
                min_val = rules['min_values'].get(sensor_type)
                max_val = rules['max_values'].get(sensor_type)
                
                if min_val is not None and value < min_val:
                    issues.append(ValidationIssue(
                        field=f"values.{sensor_type}",
                        severity=ValidationSeverity.WARNING,
                        message=f"Value {value} below minimum threshold {min_val}",
                        suggested_fix=f"Verify sensor calibration - expected range: {min_val} to {max_val}",
                        error_code="VALUE_BELOW_MINIMUM"
                    ))
                
                if max_val is not None and value > max_val:
                    issues.append(ValidationIssue(
                        field=f"values.{sensor_type}",
                        severity=ValidationSeverity.WARNING, 
                        message=f"Value {value} above maximum threshold {max_val}",
                        suggested_fix=f"Verify sensor calibration - expected range: {min_val} to {max_val}",
                        error_code="VALUE_ABOVE_MAXIMUM"
                    ))
        
        # Timestamp validation
        if 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                
                # Check if timestamp is too far in the past or future
                if (now - timestamp).days > 1:
                    issues.append(ValidationIssue(
                        field="timestamp",
                        severity=ValidationSeverity.WARNING,
                        message="Timestamp is more than 1 day old",
                        suggested_fix="Ensure system clocks are synchronized",
                        error_code="TIMESTAMP_TOO_OLD"
                    ))
                
                if timestamp > now + timedelta(minutes=5):
                    issues.append(ValidationIssue(
                        field="timestamp", 
                        severity=ValidationSeverity.ERROR,
                        message="Timestamp is in the future",
                        suggested_fix="Check system clock synchronization",
                        error_code="TIMESTAMP_FUTURE"
                    ))
                    
            except (ValueError, TypeError) as e:
                issues.append(ValidationIssue(
                    field="timestamp",
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid timestamp format: {e}",
                    suggested_fix="Use ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ",
                    error_code="INVALID_TIMESTAMP_FORMAT"
                ))
        
        return issues
    
    def _validate_security(self, data: Dict[str, Any], context: SecurityContext) -> List[ValidationIssue]:
        """Validate security aspects of the data."""
        issues = []
        
        # Convert data to string for pattern matching
        data_str = json.dumps(data, default=str).lower()
        
        # SQL Injection check
        for pattern in self.security_patterns['sql_injection_patterns']:
            import re
            if re.search(pattern, data_str):
                issues.append(ValidationIssue(
                    field="__global__",
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential SQL injection detected",
                    suggested_fix="Remove SQL keywords and dangerous patterns",
                    error_code="SQL_INJECTION_DETECTED"
                ))
                break
        
        # XSS check
        for pattern in self.security_patterns['xss_patterns']:
            import re
            if re.search(pattern, data_str):
                issues.append(ValidationIssue(
                    field="__global__",
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential XSS attack detected",
                    suggested_fix="Remove script tags and JavaScript code",
                    error_code="XSS_DETECTED"
                ))
                break
        
        # Path traversal check
        for pattern in self.security_patterns['path_traversal_patterns']:
            import re
            if re.search(pattern, data_str):
                issues.append(ValidationIssue(
                    field="__global__",
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential path traversal attack detected",
                    suggested_fix="Remove directory traversal patterns",
                    error_code="PATH_TRAVERSAL_DETECTED"
                ))
                break
        
        # Command injection check
        for pattern in self.security_patterns['command_injection_patterns']:
            import re
            if re.search(pattern, data_str):
                issues.append(ValidationIssue(
                    field="__global__",
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential command injection detected",
                    suggested_fix="Remove shell command patterns",
                    error_code="COMMAND_INJECTION_DETECTED"
                ))
                break
        
        # Authentication check
        if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            if not context.authenticated:
                issues.append(ValidationIssue(
                    field="__security__",
                    severity=ValidationSeverity.ERROR,
                    message="Authentication required for this security level",
                    suggested_fix="Provide valid authentication credentials",
                    error_code="AUTHENTICATION_REQUIRED"
                ))
        
        return issues


class RobustnessOrchestrator:
    """Main orchestrator for robustness features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validator = AdvancedValidator()
        
        # Robustness settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_ms = config.get('retry_delay_ms', 1000)
        self.timeout_ms = config.get('timeout_ms', 30000)
        self.enable_circuit_breaker = config.get('enable_circuit_breaker', True)
        self.enable_rate_limiting = config.get('enable_rate_limiting', True)
        
        # Security settings
        self.security_level = SecurityLevel(config.get('security_level', 'medium'))
        self.enable_audit_logging = config.get('enable_audit_logging', True)
        self.enable_encryption = config.get('enable_encryption', True)
        
        # Performance monitoring
        self.performance_thresholds = config.get('performance_thresholds', {
            'memory_limit_mb': 500,
            'cpu_limit_percent': 80,
            'inference_time_limit_ms': 5000
        })
        
        # Circuit breaker state
        self._circuit_breaker_state = {}
        self._rate_limit_state = {}
        
        logger.info(f"Robustness orchestrator initialized with security level: {self.security_level}")
    
    @contextmanager
    def robust_operation(self, operation_name: str, context: Optional[SecurityContext] = None):
        """Context manager for robust operations with comprehensive error handling."""
        start_time = time.time()
        audit_trail = []
        
        if context is None:
            context = SecurityContext()
        
        try:
            # Pre-operation checks
            audit_trail.append(f"Starting operation: {operation_name}")
            
            # Performance monitoring setup
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            initial_cpu = psutil.cpu_percent()
            
            # Security checks
            if self.enable_audit_logging:
                audit_trail.append(f"Security context: level={context.security_level}, auth={context.authenticated}")
            
            yield {
                'context': context,
                'audit_trail': audit_trail,
                'start_time': start_time
            }
            
        except Exception as e:
            audit_trail.append(f"Operation failed: {str(e)}")
            logger.error(f"Robust operation '{operation_name}' failed: {e}")
            raise
        
        finally:
            # Post-operation monitoring
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = final_memory - initial_memory
            
            # Performance validation
            if memory_delta > self.performance_thresholds['memory_limit_mb']:
                logger.warning(f"Memory usage spike: {memory_delta:.2f} MB for operation '{operation_name}'")
            
            if execution_time_ms > self.performance_thresholds['inference_time_limit_ms']:
                logger.warning(f"Slow operation: {execution_time_ms:.2f} ms for '{operation_name}'")
            
            audit_trail.append(f"Operation completed in {execution_time_ms:.2f} ms")
            
            # Audit logging
            if self.enable_audit_logging:
                self._log_audit_trail(operation_name, audit_trail, context)
    
    async def robust_async_operation(self, operation_name: str, operation_func: Callable, 
                                   *args, context: Optional[SecurityContext] = None, **kwargs) -> RobustResult:
        """Execute an async operation with comprehensive robustness features."""
        if context is None:
            context = SecurityContext()
        
        result = RobustResult(success=False)
        result.security_context = context
        
        start_time = time.time()
        
        # Rate limiting check
        if self.enable_rate_limiting and not self._check_rate_limit(operation_name, context):
            result.error = "Rate limit exceeded"
            result.error_code = "RATE_LIMIT_EXCEEDED"
            return result
        
        # Circuit breaker check
        if self.enable_circuit_breaker and self._is_circuit_open(operation_name):
            result.error = "Circuit breaker is open"
            result.error_code = "CIRCUIT_BREAKER_OPEN"
            return result
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                result.audit_trail.append(f"Attempt {attempt + 1} starting")
                
                # Execute operation with timeout
                try:
                    data = await asyncio.wait_for(
                        operation_func(*args, **kwargs),
                        timeout=self.timeout_ms / 1000.0
                    )
                    
                    result.success = True
                    result.data = data
                    result.retry_count = attempt
                    
                    # Reset circuit breaker on success
                    if self.enable_circuit_breaker:
                        self._record_success(operation_name)
                    
                    break
                    
                except asyncio.TimeoutError:
                    error_msg = f"Operation timed out after {self.timeout_ms}ms"
                    result.warnings.append(error_msg)
                    result.audit_trail.append(error_msg)
                    
                    if attempt == self.max_retries:
                        result.error = error_msg
                        result.error_code = "TIMEOUT"
                        
                        # Record failure for circuit breaker
                        if self.enable_circuit_breaker:
                            self._record_failure(operation_name)
                    else:
                        # Exponential backoff
                        delay = self.retry_delay_ms * (2 ** attempt) / 1000.0
                        await asyncio.sleep(delay)
                        result.audit_trail.append(f"Retrying after {delay:.2f}s delay")
                    
            except Exception as e:
                error_msg = f"Operation failed: {str(e)}"
                result.warnings.append(error_msg)
                result.audit_trail.append(error_msg)
                
                if attempt == self.max_retries:
                    result.error = error_msg
                    result.error_code = "EXECUTION_ERROR"
                    
                    # Record failure for circuit breaker
                    if self.enable_circuit_breaker:
                        self._record_failure(operation_name)
                    
                    # Log full traceback for debugging
                    logger.error(f"Operation '{operation_name}' failed after {attempt + 1} attempts: {traceback.format_exc()}")
                else:
                    # Exponential backoff
                    delay = self.retry_delay_ms * (2 ** attempt) / 1000.0
                    await asyncio.sleep(delay)
                    result.audit_trail.append(f"Retrying after {delay:.2f}s delay")
        
        # Record execution metrics
        result.execution_time_ms = (time.time() - start_time) * 1000
        
        # Audit logging
        if self.enable_audit_logging:
            self._log_audit_trail(operation_name, result.audit_trail, context)
        
        return result
    
    def validate_with_security(self, data: Dict[str, Any], context: SecurityContext) -> RobustResult[Dict[str, Any]]:
        """Validate data with comprehensive security checks."""
        start_time = time.time()
        result = RobustResult(success=True, data=data)
        result.security_context = context
        
        try:
            # Security validation
            if data:
                validation_issues = self.validator.validate_sensor_data(data, context)
                result.validation_issues = validation_issues
                
                # Check for critical security issues
                critical_issues = [issue for issue in validation_issues 
                                 if issue.severity == ValidationSeverity.CRITICAL]
                
                if critical_issues:
                    result.success = False
                    result.error = f"Critical security violations detected: {[issue.message for issue in critical_issues]}"
                    result.error_code = "SECURITY_VIOLATION"
                
                # Add warnings for non-critical issues
                warning_issues = [issue for issue in validation_issues 
                                if issue.severity in [ValidationSeverity.WARNING, ValidationSeverity.ERROR]]
                result.warnings = [issue.message for issue in warning_issues]
            
            result.execution_time_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            result.success = False
            result.error = f"Validation failed: {str(e)}"
            result.error_code = "VALIDATION_ERROR"
            result.execution_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _check_rate_limit(self, operation_name: str, context: SecurityContext) -> bool:
        """Check if operation is within rate limits."""
        # Simple token bucket implementation
        now = time.time()
        key = f"{operation_name}_{context.user_id or 'anonymous'}"
        
        if key not in self._rate_limit_state:
            self._rate_limit_state[key] = {
                'tokens': 10,  # Initial tokens
                'last_refill': now,
                'max_tokens': 10,
                'refill_rate': 1.0  # tokens per second
            }
        
        state = self._rate_limit_state[key]
        
        # Refill tokens
        time_passed = now - state['last_refill']
        new_tokens = min(state['max_tokens'], 
                        state['tokens'] + time_passed * state['refill_rate'])
        
        state['tokens'] = new_tokens
        state['last_refill'] = now
        
        # Check if we have tokens
        if state['tokens'] >= 1:
            state['tokens'] -= 1
            return True
        else:
            logger.warning(f"Rate limit exceeded for operation '{operation_name}' by user '{context.user_id}'")
            return False
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for the operation."""
        if operation_name not in self._circuit_breaker_state:
            return False
        
        state = self._circuit_breaker_state[operation_name]
        
        if state['state'] == 'closed':
            return False
        elif state['state'] == 'open':
            # Check if we should try to close
            if time.time() - state['last_failure'] > 60:  # 60 second timeout
                state['state'] = 'half_open'
                return False
            return True
        elif state['state'] == 'half_open':
            return False
        
        return False
    
    def _record_success(self, operation_name: str):
        """Record successful operation for circuit breaker."""
        if operation_name in self._circuit_breaker_state:
            state = self._circuit_breaker_state[operation_name]
            if state['state'] == 'half_open':
                state['state'] = 'closed'
                state['failure_count'] = 0
    
    def _record_failure(self, operation_name: str):
        """Record failed operation for circuit breaker.""" 
        if operation_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[operation_name] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': 0
            }
        
        state = self._circuit_breaker_state[operation_name]
        state['failure_count'] += 1
        state['last_failure'] = time.time()
        
        # Open circuit if too many failures
        if state['failure_count'] >= 5:
            state['state'] = 'open'
            logger.warning(f"Circuit breaker opened for operation '{operation_name}' after {state['failure_count']} failures")
    
    def _log_audit_trail(self, operation_name: str, audit_trail: List[str], context: SecurityContext):
        """Log audit trail for compliance and security monitoring."""
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': operation_name,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'security_level': context.security_level.value,
            'authenticated': context.authenticated,
            'trail': audit_trail
        }
        
        # In production, this would go to a secure audit log system
        logger.info(f"AUDIT: {json.dumps(audit_entry)}")
    
    def get_robustness_status(self) -> Dict[str, Any]:
        """Get current robustness status and metrics."""
        return {
            'security_level': self.security_level.value,
            'circuit_breakers': {
                name: {
                    'state': state['state'],
                    'failure_count': state['failure_count'],
                    'last_failure': state.get('last_failure', 0)
                }
                for name, state in self._circuit_breaker_state.items()
            },
            'rate_limits': {
                name: {
                    'tokens': state['tokens'],
                    'max_tokens': state['max_tokens'],
                    'last_refill': state['last_refill']
                }
                for name, state in self._rate_limit_state.items()
            },
            'performance_thresholds': self.performance_thresholds,
            'configuration': {
                'max_retries': self.max_retries,
                'retry_delay_ms': self.retry_delay_ms,
                'timeout_ms': self.timeout_ms,
                'circuit_breaker_enabled': self.enable_circuit_breaker,
                'rate_limiting_enabled': self.enable_rate_limiting,
                'audit_logging_enabled': self.enable_audit_logging
            }
        }


def create_robustness_orchestrator(config: Optional[Dict[str, Any]] = None) -> RobustnessOrchestrator:
    """Factory function to create robustness orchestrator."""
    if config is None:
        config = {
            'max_retries': 3,
            'retry_delay_ms': 1000,
            'timeout_ms': 30000,
            'enable_circuit_breaker': True,
            'enable_rate_limiting': True,
            'security_level': 'high',
            'enable_audit_logging': True,
            'enable_encryption': True,
            'performance_thresholds': {
                'memory_limit_mb': 500,
                'cpu_limit_percent': 80,
                'inference_time_limit_ms': 5000
            }
        }
    
    return RobustnessOrchestrator(config)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_operation():
        """Example operation for testing robustness features."""
        robustness = create_robustness_orchestrator()
        
        # Create security context
        context = SecurityContext(
            user_id="test_user",
            session_id=str(uuid.uuid4()),
            permissions=["read", "write"],
            security_level=SecurityLevel.HIGH,
            authenticated=True
        )
        
        # Test data validation
        test_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sensor_id': 'sensor_001',
            'values': {
                'temperature': 25.6,
                'humidity': 60.2,
                'pressure': 1013.25
            }
        }
        
        # Validate data
        validation_result = robustness.validate_with_security(test_data, context)
        print(f"Validation result: {validation_result}")
        
        # Test robust operation
        async def sample_async_operation():
            await asyncio.sleep(0.1)  # Simulate some work
            return {"status": "success", "processed": len(test_data)}
        
        operation_result = await robustness.robust_async_operation(
            "sample_operation", sample_async_operation, context=context
        )
        
        print(f"Operation result: {operation_result}")
        print(f"Robustness status: {robustness.get_robustness_status()}")
    
    try:
        asyncio.run(example_operation())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")