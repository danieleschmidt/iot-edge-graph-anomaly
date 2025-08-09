"""
Security framework for IoT edge anomaly detection inference.

This module provides comprehensive security measures including:
- Input sanitization and validation
- Model protection and obfuscation
- Secure communication protocols
- Authentication and authorization
- Threat detection and mitigation
"""
import torch
import hashlib
import hmac
import time
import secrets
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import base64
import threading
from collections import defaultdict, deque
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecurityThreat(Enum):
    """Types of security threats."""
    MODEL_EXTRACTION_ATTEMPT = "model_extraction_attempt"
    ADVERSARIAL_INPUT = "adversarial_input"
    INJECTION_ATTACK = "injection_attack"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Container for security events."""
    timestamp: str
    threat_type: SecurityThreat
    severity: SecurityLevel
    client_id: Optional[str]
    details: Dict[str, Any]
    mitigated: bool = False


class InputSanitizer:
    """Sanitizes and validates input data for security."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.allowed_shapes = self.config.get('allowed_shapes', [(1, 20, 5), (1, 30, 8)])
        self.max_value_range = self.config.get('max_value_range', [-100, 100])
        self.max_tensor_size = self.config.get('max_tensor_size', 10000)
        
    def sanitize_tensor(self, data: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Sanitize input tensor and return warnings."""
        warnings = []
        original_data = data.clone()
        
        # Check tensor size
        if data.numel() > self.max_tensor_size:
            warnings.append(f"Tensor size {data.numel()} exceeds maximum {self.max_tensor_size}")
            # Truncate to max size
            flat_data = data.flatten()[:self.max_tensor_size]
            data = flat_data.reshape(-1, min(data.size(-1), self.max_tensor_size))
        
        # Shape validation
        if data.shape not in self.allowed_shapes:
            warnings.append(f"Unexpected tensor shape: {data.shape}")
        
        # Value range clamping
        data = torch.clamp(data, self.max_value_range[0], self.max_value_range[1])
        if not torch.equal(data, original_data):
            warnings.append("Values clamped to allowed range")
        
        # NaN/Inf sanitization
        if torch.isnan(data).any():
            data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
            warnings.append("NaN values replaced with zeros")
        
        if torch.isinf(data).any():
            data = torch.where(torch.isinf(data), 
                             torch.clamp(data, self.max_value_range[0], self.max_value_range[1]), 
                             data)
            warnings.append("Infinite values clamped")
        
        # Gradient sanitization (prevent gradient-based attacks)
        if data.requires_grad:
            data = data.detach()
            warnings.append("Gradients detached for security")
        
        return data, warnings
    
    def detect_adversarial_input(self, data: torch.Tensor) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect potential adversarial inputs."""
        details = {}
        
        # Statistical anomaly detection
        std_dev = torch.std(data)
        mean_val = torch.mean(data)
        
        # Check for unusual statistical properties
        suspicious_stats = False
        if std_dev < 1e-6:  # Extremely low variance
            suspicious_stats = True
            details['low_variance'] = std_dev.item()
        
        if abs(mean_val) > 50:  # Unusual mean
            suspicious_stats = True
            details['unusual_mean'] = mean_val.item()
        
        # Check for adversarial patterns
        # 1. High-frequency noise patterns
        if len(data.shape) >= 2:
            diff = torch.diff(data, dim=-2)
            high_freq_energy = torch.mean(diff ** 2)
            if high_freq_energy > 10:  # Threshold for high frequency content
                suspicious_stats = True
                details['high_frequency_energy'] = high_freq_energy.item()
        
        # 2. Repeating patterns (potential crafted input)
        flat_data = data.flatten()
        if len(flat_data) > 10:
            # Simple pattern detection
            pattern_length = min(10, len(flat_data) // 4)
            pattern = flat_data[:pattern_length]
            repeats = 0
            
            for i in range(pattern_length, len(flat_data) - pattern_length, pattern_length):
                chunk = flat_data[i:i+pattern_length]
                if torch.allclose(pattern, chunk, atol=1e-3):
                    repeats += 1
            
            if repeats > len(flat_data) // pattern_length * 0.8:  # >80% repetition
                suspicious_stats = True
                details['pattern_repetition'] = repeats / (len(flat_data) // pattern_length)
        
        # Calculate suspicion score
        suspicion_score = 0.0
        if suspicious_stats:
            suspicion_score = min(1.0, len(details) * 0.3)
        
        return suspicious_stats, suspicion_score, details


class ModelProtector:
    """Protects ML models from extraction and abuse."""
    
    def __init__(self, model: torch.nn.Module, protection_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.model = model
        self.protection_level = protection_level
        self.query_count = 0
        self.query_history = deque(maxlen=1000)
        self.model_hash = self._compute_model_hash()
        self._setup_protection()
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model parameters for integrity checking."""
        hasher = hashlib.sha256()
        for param in self.model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()
    
    def _setup_protection(self):
        """Setup model protection based on security level."""
        if self.protection_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            # Add noise to model outputs in high security mode
            self.output_noise_std = 0.001 if self.protection_level == SecurityLevel.HIGH else 0.0001
        else:
            self.output_noise_std = 0.0
        
        # Query limits based on security level
        self.max_queries_per_hour = {
            SecurityLevel.LOW: 10000,
            SecurityLevel.MEDIUM: 1000,
            SecurityLevel.HIGH: 100,
            SecurityLevel.CRITICAL: 50
        }[self.protection_level]
    
    def protected_inference(self, data: torch.Tensor, client_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform inference with protection measures."""
        self.query_count += 1
        query_info = {
            'timestamp': time.time(),
            'client_id': client_id,
            'input_shape': data.shape,
            'input_hash': hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()[:16]
        }
        self.query_history.append(query_info)
        
        # Check for model extraction attempts
        extraction_detected, extraction_details = self._detect_extraction_attempt()
        if extraction_detected:
            return {
                'success': False,
                'error': 'Security violation detected',
                'threat_type': SecurityThreat.MODEL_EXTRACTION_ATTEMPT.value,
                'details': extraction_details
            }
        
        # Perform inference
        try:
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'compute_reconstruction_error'):
                    output = self.model.compute_reconstruction_error(data, reduction='mean')
                elif hasattr(self.model, 'compute_hybrid_anomaly_score'):
                    output = self.model.compute_hybrid_anomaly_score(data, reduction='mean')
                else:
                    reconstruction = self.model(data)
                    output = torch.mean((data - reconstruction) ** 2)
                
                # Add protection noise
                if self.output_noise_std > 0:
                    noise = torch.normal(0, self.output_noise_std, output.shape)
                    output = output + noise
                
                return {
                    'success': True,
                    'output': output.item(),
                    'query_count': self.query_count
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_extraction_attempt(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect potential model extraction attempts."""
        if len(self.query_history) < 10:
            return False, {}
        
        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        
        # Check for systematic probing patterns
        input_hashes = [q['input_hash'] for q in recent_queries]
        unique_inputs = len(set(input_hashes))
        total_inputs = len(input_hashes)
        
        details = {
            'unique_inputs_ratio': unique_inputs / total_inputs,
            'query_rate': len([q for q in recent_queries if time.time() - q['timestamp'] < 3600]),  # Per hour
            'total_queries': self.query_count
        }
        
        # Detection criteria
        suspicious = False
        
        # 1. Too many unique inputs (systematic exploration)
        if unique_inputs / total_inputs > 0.95 and total_inputs > 50:
            suspicious = True
            details['reason'] = 'systematic_input_exploration'
        
        # 2. Exceeding query rate limits
        if details['query_rate'] > self.max_queries_per_hour:
            suspicious = True
            details['reason'] = 'query_rate_exceeded'
        
        # 3. Check for repeated similar inputs with small variations
        if len(recent_queries) > 20:
            shape_counts = defaultdict(int)
            for q in recent_queries:
                shape_counts[str(q['input_shape'])] += 1
            
            # If >80% queries have the same shape, potential systematic probing
            max_shape_ratio = max(shape_counts.values()) / len(recent_queries)
            if max_shape_ratio > 0.8:
                details['shape_repetition_ratio'] = max_shape_ratio
                if unique_inputs / total_inputs > 0.7:  # Many unique but same shape
                    suspicious = True
                    details['reason'] = 'systematic_shape_probing'
        
        return suspicious, details
    
    def verify_model_integrity(self) -> bool:
        """Verify model hasn't been tampered with."""
        current_hash = self._compute_model_hash()
        return current_hash == self.model_hash


class AuthenticationManager:
    """Manages authentication and authorization for secure access."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.active_tokens = {}
        self.failed_attempts = defaultdict(list)
        self.token_expiry = 3600  # 1 hour
        self._lock = threading.Lock()
    
    def generate_token(self, client_id: str, permissions: List[str]) -> str:
        """Generate authentication token."""
        timestamp = int(time.time())
        payload = {
            'client_id': client_id,
            'permissions': permissions,
            'issued_at': timestamp,
            'expires_at': timestamp + self.token_expiry
        }
        
        # Create HMAC signature
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = base64.b64encode(f"{payload_str}:{signature}".encode()).decode()
        
        with self._lock:
            self.active_tokens[token] = payload
        
        return token
    
    def validate_token(self, token: str, required_permission: str = 'inference') -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate authentication token."""
        details = {}
        
        try:
            # Decode token
            decoded = base64.b64decode(token.encode()).decode()
            payload_str, signature = decoded.rsplit(':', 1)
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                details['error'] = 'invalid_signature'
                return False, None, details
            
            # Parse payload
            payload = json.loads(payload_str)
            
            # Check expiration
            if time.time() > payload['expires_at']:
                details['error'] = 'token_expired'
                return False, None, details
            
            # Check permissions
            if required_permission not in payload.get('permissions', []):
                details['error'] = 'insufficient_permissions'
                return False, payload.get('client_id'), details
            
            # Token is valid
            return True, payload['client_id'], payload
            
        except Exception as e:
            details['error'] = f'token_validation_failed: {e}'
            return False, None, details
    
    def record_failed_attempt(self, client_identifier: str):
        """Record failed authentication attempt."""
        with self._lock:
            self.failed_attempts[client_identifier].append(time.time())
            # Keep only recent attempts (last hour)
            cutoff = time.time() - 3600
            self.failed_attempts[client_identifier] = [
                t for t in self.failed_attempts[client_identifier] if t > cutoff
            ]
    
    def is_rate_limited(self, client_identifier: str, max_attempts: int = 5) -> bool:
        """Check if client is rate limited due to failed attempts."""
        with self._lock:
            return len(self.failed_attempts.get(client_identifier, [])) >= max_attempts


class SecureInferenceEngine:
    """Main secure inference engine orchestrating all security measures."""
    
    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        # Security components
        self.sanitizer = InputSanitizer(self.config.get('input_sanitization', {}))
        self.protector = ModelProtector(
            model, 
            SecurityLevel(self.config.get('protection_level', 'medium'))
        )
        self.auth_manager = AuthenticationManager(self.config.get('secret_key'))
        
        # Security monitoring
        self.security_events = deque(maxlen=1000)
        self.threat_counts = defaultdict(int)
        
        # Configuration
        self.enable_authentication = self.config.get('enable_authentication', True)
        self.log_security_events = self.config.get('log_security_events', True)
        
        logger.info(f"Secure inference engine initialized with protection level: {self.protector.protection_level.value}")
    
    def secure_inference(self, 
                        data: torch.Tensor,
                        token: Optional[str] = None,
                        client_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform secure inference with comprehensive security checks."""
        
        # Authentication
        if self.enable_authentication:
            if not token:
                self._log_security_event(
                    SecurityThreat.UNAUTHORIZED_ACCESS,
                    SecurityLevel.HIGH,
                    client_id,
                    {'reason': 'missing_token'}
                )
                return {'success': False, 'error': 'Authentication required'}
            
            valid, validated_client_id, auth_details = self.auth_manager.validate_token(token, 'inference')
            if not valid:
                self.auth_manager.record_failed_attempt(client_id or 'unknown')
                self._log_security_event(
                    SecurityThreat.UNAUTHORIZED_ACCESS,
                    SecurityLevel.HIGH,
                    client_id,
                    auth_details
                )
                return {'success': False, 'error': 'Invalid authentication'}
            
            client_id = validated_client_id
        
        # Rate limiting check
        if self.auth_manager.is_rate_limited(client_id or 'unknown'):
            self._log_security_event(
                SecurityThreat.RATE_LIMIT_EXCEEDED,
                SecurityLevel.MEDIUM,
                client_id,
                {'reason': 'too_many_failed_attempts'}
            )
            return {'success': False, 'error': 'Rate limit exceeded'}
        
        # Input sanitization
        sanitized_data, sanitization_warnings = self.sanitizer.sanitize_tensor(data)
        
        # Adversarial input detection
        is_adversarial, suspicion_score, adversarial_details = self.sanitizer.detect_adversarial_input(sanitized_data)
        
        if is_adversarial and suspicion_score > 0.7:
            self._log_security_event(
                SecurityThreat.ADVERSARIAL_INPUT,
                SecurityLevel.HIGH,
                client_id,
                {
                    'suspicion_score': suspicion_score,
                    'details': adversarial_details
                }
            )
            return {
                'success': False,
                'error': 'Suspicious input detected',
                'suspicion_score': suspicion_score
            }
        
        # Protected inference
        result = self.protector.protected_inference(sanitized_data, client_id)
        
        # Add security metadata
        result.update({
            'security_level': self.protector.protection_level.value,
            'sanitization_warnings': sanitization_warnings,
            'adversarial_score': suspicion_score,
            'model_integrity_verified': self.protector.verify_model_integrity()
        })
        
        # Log successful inference
        if result.get('success') and self.log_security_events:
            logger.debug(f"Secure inference completed for client {client_id}")
        
        return result
    
    def _log_security_event(self, threat_type: SecurityThreat, severity: SecurityLevel, 
                           client_id: Optional[str], details: Dict[str, Any]):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            threat_type=threat_type,
            severity=severity,
            client_id=client_id,
            details=details
        )
        
        self.security_events.append(event)
        self.threat_counts[threat_type] += 1
        
        if self.log_security_events:
            logger.warning(f"Security event: {threat_type.value} - {severity.value} - "
                          f"Client: {client_id} - Details: {details}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'protection_level': self.protector.protection_level.value,
            'total_queries': self.protector.query_count,
            'security_events': {
                'total_events': len(self.security_events),
                'threat_counts': dict(self.threat_counts),
                'recent_events': [
                    {
                        'timestamp': event.timestamp,
                        'threat_type': event.threat_type.value,
                        'severity': event.severity.value,
                        'client_id': event.client_id
                    }
                    for event in list(self.security_events)[-10:]  # Last 10 events
                ]
            },
            'authentication': {
                'enabled': self.enable_authentication,
                'active_tokens': len(self.auth_manager.active_tokens),
                'failed_attempts_by_client': {
                    k: len(v) for k, v in self.auth_manager.failed_attempts.items()
                }
            },
            'model_integrity': self.protector.verify_model_integrity()
        }
    
    def export_security_report(self, output_path: str) -> bool:
        """Export comprehensive security report."""
        try:
            report = {
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'security_status': self.get_security_status(),
                'configuration': {
                    'protection_level': self.protector.protection_level.value,
                    'authentication_enabled': self.enable_authentication,
                    'max_queries_per_hour': self.protector.max_queries_per_hour
                },
                'detailed_events': [
                    {
                        'timestamp': event.timestamp,
                        'threat_type': event.threat_type.value,
                        'severity': event.severity.value,
                        'client_id': event.client_id,
                        'details': event.details,
                        'mitigated': event.mitigated
                    }
                    for event in self.security_events
                ],
                'recommendations': self._generate_security_recommendations()
            }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Security report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export security report: {e}")
            return False
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on observed events."""
        recommendations = []
        
        # Analyze threat patterns
        total_events = len(self.security_events)
        if total_events == 0:
            return ["Continue monitoring for security events"]
        
        # High threat activity
        if total_events > 100:
            recommendations.append("High security event activity detected - consider increasing security level")
        
        # Frequent adversarial inputs
        adversarial_count = self.threat_counts.get(SecurityThreat.ADVERSARIAL_INPUT, 0)
        if adversarial_count > total_events * 0.2:
            recommendations.append("Frequent adversarial inputs - implement stronger input validation")
        
        # Model extraction attempts
        extraction_count = self.threat_counts.get(SecurityThreat.MODEL_EXTRACTION_ATTEMPT, 0)
        if extraction_count > 0:
            recommendations.append("Model extraction attempts detected - consider query rate limiting")
        
        # Unauthorized access attempts
        unauthorized_count = self.threat_counts.get(SecurityThreat.UNAUTHORIZED_ACCESS, 0)
        if unauthorized_count > total_events * 0.3:
            recommendations.append("High unauthorized access attempts - strengthen authentication")
        
        # Model integrity
        if not self.protector.verify_model_integrity():
            recommendations.append("CRITICAL: Model integrity compromised - investigate immediately")
        
        return recommendations if recommendations else ["Security posture appears healthy"]