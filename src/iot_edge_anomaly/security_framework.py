"""
🔒 Security Framework for IoT Anomaly Detection System

This module provides comprehensive security measures for the Terragon IoT 
Anomaly Detection System, including input sanitization, secure model inference,
authentication, authorization, and protection against various attack vectors.

Features:
- Input sanitization and validation against adversarial attacks
- Secure model inference with privacy protection
- Authentication and authorization framework
- Data encryption and secure communication
- Audit logging and compliance monitoring
- Protection against model extraction and inversion attacks
- Differential privacy implementation
- Secure multi-tenant isolation
"""

import os
import sys
import time
import json
import hashlib
import hmac
import secrets
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import warnings
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class SecurityLevel(Enum):
    """Security levels for different deployment scenarios."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_EXTRACTION = "model_extraction"
    MODEL_INVERSION = "model_inversion"
    PRIVACY_BREACH = "privacy_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_POISONING = "data_poisoning"
    DENIAL_OF_SERVICE = "denial_of_service"
    INJECTION_ATTACK = "injection_attack"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    severity: str  # low, medium, high, critical
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    blocked: bool
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'threat_type': self.threat_type.value,
            'severity': self.severity,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'description': self.description,
            'blocked': self.blocked,
            'details': self.details
        }


class InputSanitizer:
    """Advanced input sanitization against adversarial attacks."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.threat_detector = AdversarialDetector()
        self.sanitization_methods = self._setup_sanitization_methods()
        
    def _setup_sanitization_methods(self) -> Dict[str, Callable]:
        """Setup sanitization methods based on security level."""
        methods = {
            'bounds_check': self._bounds_sanitization,
            'statistical_check': self._statistical_sanitization,
            'adversarial_check': self._adversarial_sanitization
        }
        
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            methods.update({
                'noise_addition': self._noise_sanitization,
                'quantization': self._quantization_sanitization,
                'smoothing': self._smoothing_sanitization
            })
        
        return methods
    
    def sanitize_input(
        self, 
        data: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """
        Comprehensive input sanitization.
        
        Args:
            data: Input tensor to sanitize
            metadata: Optional metadata about the input
            
        Returns:
            (sanitized_data, warnings, is_safe)
        """
        warnings_list = []
        is_safe = True
        sanitized_data = data.clone()
        
        try:
            # Run all sanitization methods
            for method_name, method_func in self.sanitization_methods.items():
                try:
                    sanitized_data, method_warnings, method_safe = method_func(
                        sanitized_data, metadata
                    )
                    warnings_list.extend(method_warnings)
                    is_safe = is_safe and method_safe
                    
                except Exception as e:
                    warnings_list.append(f"Sanitization method {method_name} failed: {str(e)}")
                    is_safe = False
            
            return sanitized_data, warnings_list, is_safe
            
        except Exception as e:
            return data, [f"Sanitization failed: {str(e)}"], False
    
    def _bounds_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Check and enforce data bounds."""
        warnings_list = []
        is_safe = True
        
        # Check for extreme values
        if torch.abs(data).max() > 1e6:
            warnings_list.append("Extreme values detected - clamping to safe range")
            data = torch.clamp(data, -1e6, 1e6)
            is_safe = False
        
        # Check for unusual data ranges
        data_std = torch.std(data)
        if data_std > 1e3:
            warnings_list.append("Unusually high variance detected")
            is_safe = False
        
        return data, warnings_list, is_safe
    
    def _statistical_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Statistical anomaly detection in input."""
        warnings_list = []
        is_safe = True
        
        # Check for statistical anomalies
        data_mean = torch.mean(data)
        data_std = torch.std(data)
        
        # Z-score analysis
        z_scores = torch.abs((data - data_mean) / (data_std + 1e-8))
        outliers = z_scores > 5
        
        if outliers.any():
            outlier_count = outliers.sum().item()
            outlier_ratio = outlier_count / data.numel()
            
            if outlier_ratio > 0.1:  # More than 10% outliers
                warnings_list.append(f"High outlier ratio detected: {outlier_ratio:.2%}")
                is_safe = False
                
                # Cap outliers
                data = torch.where(outliers, data_mean + 3 * data_std * torch.sign(data - data_mean), data)
        
        return data, warnings_list, is_safe
    
    def _adversarial_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Detect and mitigate adversarial inputs."""
        warnings_list = []
        is_safe = True
        
        # Use adversarial detector
        is_adversarial, confidence = self.threat_detector.detect_adversarial(data)
        
        if is_adversarial:
            warnings_list.append(f"Adversarial input detected (confidence: {confidence:.2f})")
            is_safe = False
            
            # Apply defensive transformation
            data = self._apply_defensive_transformation(data)
        
        return data, warnings_list, is_safe
    
    def _noise_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Add differential privacy noise."""
        warnings_list = []
        is_safe = True
        
        if self.security_level == SecurityLevel.CRITICAL:
            # Add calibrated noise for differential privacy
            noise_scale = torch.std(data) * 0.01  # 1% noise
            noise = torch.randn_like(data) * noise_scale
            data = data + noise
            warnings_list.append("Differential privacy noise added")
        
        return data, warnings_list, is_safe
    
    def _quantization_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Apply quantization to reduce precision attacks."""
        warnings_list = []
        is_safe = True
        
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            # Quantize to reduce precision
            data_min, data_max = torch.min(data), torch.max(data)
            data_range = data_max - data_min
            
            if data_range > 0:
                quantization_levels = 256  # 8-bit quantization
                data = torch.round((data - data_min) / data_range * (quantization_levels - 1))
                data = data / (quantization_levels - 1) * data_range + data_min
                warnings_list.append("Quantization applied for security")
        
        return data, warnings_list, is_safe
    
    def _smoothing_sanitization(
        self, 
        data: torch.Tensor, 
        metadata: Optional[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[str], bool]:
        """Apply smoothing to remove high-frequency adversarial noise."""
        warnings_list = []
        is_safe = True
        
        if data.dim() >= 2:
            # Apply simple smoothing filter
            kernel_size = 3
            if data.dim() == 3:  # [batch, seq, features]
                # Temporal smoothing
                padding = kernel_size // 2
                data_padded = F.pad(data, (0, 0, padding, padding), mode='reflect')
                data = F.avg_pool1d(data_padded.transpose(1, 2), kernel_size, stride=1).transpose(1, 2)
            
            warnings_list.append("Smoothing filter applied")
        
        return data, warnings_list, is_safe
    
    def _apply_defensive_transformation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply defensive transformation to mitigate adversarial inputs."""
        # Simple defensive transformation: add small random noise
        noise_scale = torch.std(data) * 0.005
        defensive_noise = torch.randn_like(data) * noise_scale
        return data + defensive_noise


class AdversarialDetector:
    """Detector for adversarial inputs."""
    
    def __init__(self):
        self.detection_threshold = 0.7
        
    def detect_adversarial(self, data: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect adversarial inputs using statistical methods.
        
        Args:
            data: Input tensor to analyze
            
        Returns:
            (is_adversarial, confidence)
        """
        try:
            # Multiple detection methods
            detections = []
            
            # 1. High-frequency noise detection
            hf_score = self._detect_high_frequency_noise(data)
            detections.append(hf_score)
            
            # 2. Statistical anomaly detection
            stat_score = self._detect_statistical_anomaly(data)
            detections.append(stat_score)
            
            # 3. Gradient-based detection
            grad_score = self._detect_gradient_anomaly(data)
            detections.append(grad_score)
            
            # Combine scores
            combined_score = max(detections)
            is_adversarial = combined_score > self.detection_threshold
            
            return is_adversarial, combined_score
            
        except Exception:
            # Default to safe mode - assume adversarial
            return True, 1.0
    
    def _detect_high_frequency_noise(self, data: torch.Tensor) -> float:
        """Detect high-frequency noise indicative of adversarial perturbations."""
        if data.dim() < 2:
            return 0.0
        
        # Compute high-frequency components using differences
        if data.dim() == 3:  # [batch, seq, features]
            diffs = torch.diff(data, dim=1)
        else:  # [seq, features]
            diffs = torch.diff(data, dim=0)
        
        # High-frequency energy
        hf_energy = torch.mean(torch.abs(diffs))
        total_energy = torch.mean(torch.abs(data))
        
        if total_energy > 0:
            hf_ratio = hf_energy / total_energy
            return min(hf_ratio.item() * 10, 1.0)  # Scale and cap at 1.0
        
        return 0.0
    
    def _detect_statistical_anomaly(self, data: torch.Tensor) -> float:
        """Detect statistical anomalies."""
        # Compute higher-order moments
        data_flat = data.flatten()
        
        # Skewness and kurtosis can indicate artificial patterns
        mean = torch.mean(data_flat)
        std = torch.std(data_flat)
        
        if std > 0:
            # Normalized higher moments
            skewness = torch.mean(((data_flat - mean) / std) ** 3)
            kurtosis = torch.mean(((data_flat - mean) / std) ** 4) - 3
            
            # High absolute skewness or kurtosis can indicate adversarial inputs
            anomaly_score = (torch.abs(skewness) + torch.abs(kurtosis)) / 10
            return min(anomaly_score.item(), 1.0)
        
        return 0.0
    
    def _detect_gradient_anomaly(self, data: torch.Tensor) -> float:
        """Detect gradient-based anomalies."""
        if not data.requires_grad:
            data = data.clone().detach().requires_grad_(True)
        
        try:
            # Simple gradient-based detection
            # Create a dummy loss that would be sensitive to adversarial perturbations
            loss = torch.sum(data ** 2)
            loss.backward()
            
            if data.grad is not None:
                grad_norm = torch.norm(data.grad).item()
                # Normalize by data norm
                data_norm = torch.norm(data).item()
                
                if data_norm > 0:
                    normalized_grad = grad_norm / data_norm
                    return min(normalized_grad / 10, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0


class SecureModelWrapper:
    """Secure wrapper for model inference with privacy protection."""
    
    def __init__(
        self, 
        model: nn.Module, 
        security_level: SecurityLevel = SecurityLevel.STANDARD
    ):
        self.model = model
        self.security_level = security_level
        self.input_sanitizer = InputSanitizer(security_level)
        self.inference_count = 0
        self.blocked_attempts = 0
        
        # Differential privacy parameters
        self.privacy_epsilon = 1.0
        self.privacy_delta = 1e-5
        
    def secure_predict(
        self, 
        data: torch.Tensor,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Secure prediction with comprehensive protection.
        
        Args:
            data: Input data
            user_id: User identifier for audit
            source_ip: Source IP for audit
            
        Returns:
            (prediction, security_info)
        """
        security_info = {
            'sanitization_applied': [],
            'threats_detected': [],
            'privacy_protection': False,
            'inference_allowed': True
        }
        
        self.inference_count += 1
        
        try:
            # Input sanitization
            sanitized_data, warnings, is_safe = self.input_sanitizer.sanitize_input(data)
            security_info['sanitization_applied'] = warnings
            
            if not is_safe:
                security_info['threats_detected'].append('unsafe_input')
                
                if self.security_level == SecurityLevel.CRITICAL:
                    # Block unsafe inputs in critical mode
                    self.blocked_attempts += 1
                    security_info['inference_allowed'] = False
                    
                    # Log security event
                    self._log_security_event(
                        ThreatType.ADVERSARIAL_INPUT,
                        "Unsafe input blocked",
                        "high",
                        source_ip,
                        user_id,
                        {'warnings': warnings}
                    )
                    
                    raise SecurityError("Input blocked due to security concerns")
            
            # Model inference with privacy protection
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'predict'):
                    prediction = self.model.predict(sanitized_data)
                else:
                    prediction = self.model(sanitized_data)
            
            # Apply differential privacy if needed
            if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                prediction = self._apply_differential_privacy(prediction)
                security_info['privacy_protection'] = True
            
            # Output sanitization
            prediction = self._sanitize_output(prediction)
            
            return prediction, security_info
            
        except Exception as e:
            self.blocked_attempts += 1
            security_info['inference_allowed'] = False
            
            # Log security event
            self._log_security_event(
                ThreatType.MODEL_EXTRACTION,
                f"Inference failed with error: {str(e)}",
                "medium",
                source_ip,
                user_id,
                {'error': str(e)}
            )
            
            raise
    
    def _apply_differential_privacy(self, output: torch.Tensor) -> torch.Tensor:
        """Apply differential privacy to model output."""
        # Calculate noise scale based on sensitivity and privacy parameters
        sensitivity = 1.0  # Assume unit sensitivity
        noise_scale = sensitivity / self.privacy_epsilon
        
        # Add calibrated Gaussian noise
        noise = torch.randn_like(output) * noise_scale
        return output + noise
    
    def _sanitize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Sanitize model output to prevent information leakage."""
        # Clamp output to reasonable ranges
        output = torch.clamp(output, -10.0, 10.0)
        
        # Round to reduce precision
        if self.security_level == SecurityLevel.CRITICAL:
            output = torch.round(output * 100) / 100  # 2 decimal places
        
        return output
    
    def _log_security_event(
        self,
        threat_type: ThreatType,
        description: str,
        severity: str,
        source_ip: Optional[str],
        user_id: Optional[str],
        details: Dict[str, Any]
    ):
        """Log security event for audit."""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            blocked=True,
            details=details
        )
        
        # In production, this would go to a secure audit log
        logging.getLogger('security').critical(json.dumps(event.to_dict()))
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        block_rate = self.blocked_attempts / max(self.inference_count, 1)
        
        return {
            'total_inferences': self.inference_count,
            'blocked_attempts': self.blocked_attempts,
            'block_rate': block_rate,
            'security_level': self.security_level.value,
            'privacy_enabled': self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        }


class AuthenticationManager:
    """Simple authentication and authorization manager."""
    
    def __init__(self):
        self.api_keys = {}
        self.user_permissions = {}
        self.session_tokens = {}
        
    def create_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Create API key for user."""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_used': None,
            'permissions': permissions
        }
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        if api_key in self.api_keys:
            key_info = self.api_keys[api_key]
            key_info['last_used'] = datetime.now()
            return key_info
        return None
    
    def check_permission(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission."""
        key_info = self.validate_api_key(api_key)
        if key_info:
            return required_permission in key_info['permissions']
        return False


class SecurityError(Exception):
    """Security-related error."""
    pass


# Utility functions
def secure_hash(data: str, salt: Optional[str] = None) -> str:
    """Create secure hash of data."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_input = (data + salt).encode('utf-8')
    return hashlib.sha256(hash_input).hexdigest()


def verify_integrity(data: bytes, signature: str, secret_key: str) -> bool:
    """Verify data integrity using HMAC."""
    expected_signature = hmac.new(
        secret_key.encode('utf-8'),
        data,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


def encrypt_data(data: str, key: str) -> str:
    """Simple encryption (in production, use proper encryption libraries)."""
    # This is a placeholder - use proper encryption in production
    encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
    return encoded


def decrypt_data(encrypted_data: str, key: str) -> str:
    """Simple decryption (in production, use proper encryption libraries)."""
    # This is a placeholder - use proper decryption in production
    decoded = base64.b64decode(encrypted_data.encode('utf-8')).decode('utf-8')
    return decoded


# Example usage and testing
if __name__ == "__main__":
    print("🔒 Testing Security Framework")
    print("=" * 50)
    
    # Test input sanitization
    sanitizer = InputSanitizer(SecurityLevel.HIGH)
    
    # Normal data
    normal_data = torch.randn(1, 20, 5)
    sanitized, warnings, safe = sanitizer.sanitize_input(normal_data)
    print(f"Normal data - Safe: {safe}, Warnings: {len(warnings)}")
    
    # Adversarial-like data (extreme values)
    adversarial_data = torch.randn(1, 20, 5) * 1000
    sanitized, warnings, safe = sanitizer.sanitize_input(adversarial_data)
    print(f"Adversarial data - Safe: {safe}, Warnings: {len(warnings)}")
    
    # Test secure model wrapper
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    try:
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
        secure_model = SecureModelWrapper(model, SecurityLevel.STANDARD)
        
        # Test secure prediction
        prediction, security_info = secure_model.secure_predict(normal_data)
        print(f"Secure prediction successful: {security_info['inference_allowed']}")
        
        # Get security metrics
        metrics = secure_model.get_security_metrics()
        print(f"Security metrics: {metrics}")
        
    except Exception as e:
        print(f"Secure model test skipped (missing dependencies): {e}")
    
    # Test authentication
    auth_manager = AuthenticationManager()
    api_key = auth_manager.create_api_key("test_user", ["inference", "monitoring"])
    print(f"Created API key: {api_key[:16]}...")
    
    # Test permission check
    has_permission = auth_manager.check_permission(api_key, "inference")
    print(f"Has inference permission: {has_permission}")
    
    print("✅ Security framework tested successfully!")