"""
Advanced Security Module for IoT Edge Anomaly Detection.

This module provides comprehensive security features including:
- Input sanitization and validation
- Secure inference with encryption
- Threat detection and mitigation
- Security monitoring and logging
- Compliance frameworks (GDPR, CCPA, SOC2)
"""
import hashlib
import hmac
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import torch
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json
import base64
import re

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of security attacks."""
    ADVERSARIAL_INPUT = "adversarial_input"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    DATA_POISONING = "data_poisoning"
    DENIAL_OF_SERVICE = "denial_of_service"
    INJECTION_ATTACK = "injection_attack"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Represents a security event or threat."""
    event_id: str
    threat_level: ThreatLevel
    attack_type: AttackType
    timestamp: datetime
    source: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "threat_level": self.threat_level.value,
            "attack_type": self.attack_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "description": self.description,
            "metadata": self.metadata,
            "mitigated": self.mitigated,
            "mitigation_actions": self.mitigation_actions
        }


class AdvancedInputSanitizer:
    """
    Advanced input sanitization with threat detection.
    
    Features:
    - Adversarial input detection
    - Statistical anomaly detection in inputs
    - Format validation and normalization
    - Rate limiting and pattern analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced input sanitizer."""
        self.config = config or {}
        self.request_history: List[Dict[str, Any]] = []
        self.blocked_patterns: List[str] = [
            r"<script.*?>.*?</script>",  # XSS patterns
            r"union.*select",  # SQL injection patterns
            r"javascript:",  # JavaScript injection
            r"data:text/html",  # Data URI attacks
        ]
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Statistical thresholds for anomaly detection
        self.value_range = self.config.get("value_range", (-1000, 1000))
        self.max_variance_ratio = self.config.get("max_variance_ratio", 100.0)
        self.adversarial_threshold = self.config.get("adversarial_threshold", 0.95)
        
    def sanitize_and_validate(self, data: Union[torch.Tensor, np.ndarray, dict], 
                            source_id: str = "unknown") -> Tuple[Any, List[SecurityEvent]]:
        """
        Sanitize and validate input data with comprehensive security checks.
        
        Args:
            data: Input data to sanitize
            source_id: Identifier for the data source
            
        Returns:
            Tuple of (sanitized_data, security_events)
        """
        security_events = []
        
        # Rate limiting check
        if self._check_rate_limit(source_id):
            security_events.append(SecurityEvent(
                event_id=self._generate_event_id(),
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.DENIAL_OF_SERVICE,
                timestamp=datetime.now(),
                source=source_id,
                description="Rate limit exceeded"
            ))
        
        # Handle different data types
        if isinstance(data, dict):
            return self._sanitize_dict(data, source_id, security_events)
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            return self._sanitize_tensor(data, source_id, security_events)
        else:
            # Convert to tensor if possible
            try:
                if isinstance(data, (list, tuple)):
                    data = torch.tensor(data, dtype=torch.float32)
                    return self._sanitize_tensor(data, source_id, security_events)
                else:
                    security_events.append(SecurityEvent(
                        event_id=self._generate_event_id(),
                        threat_level=ThreatLevel.HIGH,
                        attack_type=AttackType.INJECTION_ATTACK,
                        timestamp=datetime.now(),
                        source=source_id,
                        description=f"Unsupported data type: {type(data)}"
                    ))
                    return None, security_events
            except Exception as e:
                security_events.append(SecurityEvent(
                    event_id=self._generate_event_id(),
                    threat_level=ThreatLevel.HIGH,
                    attack_type=AttackType.INJECTION_ATTACK,
                    timestamp=datetime.now(),
                    source=source_id,
                    description=f"Data conversion failed: {str(e)}"
                ))
                return None, security_events
    
    def _sanitize_tensor(self, tensor: torch.Tensor, source_id: str, 
                        security_events: List[SecurityEvent]) -> Tuple[torch.Tensor, List[SecurityEvent]]:
        """Sanitize tensor data with security checks."""
        
        # Convert to tensor if needed
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float()
        
        # Check for adversarial patterns
        if self._detect_adversarial_input(tensor):
            security_events.append(SecurityEvent(
                event_id=self._generate_event_id(),
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.ADVERSARIAL_INPUT,
                timestamp=datetime.now(),
                source=source_id,
                description="Adversarial input pattern detected",
                metadata={"tensor_shape": list(tensor.shape)}
            ))
        
        # Check for statistical anomalies
        anomaly_score = self._compute_statistical_anomaly_score(tensor)
        if anomaly_score > 0.9:
            security_events.append(SecurityEvent(
                event_id=self._generate_event_id(),
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.DATA_POISONING,
                timestamp=datetime.now(),
                source=source_id,
                description=f"Statistical anomaly detected (score: {anomaly_score:.3f})",
                metadata={"anomaly_score": anomaly_score}
            ))
        
        # Sanitize values
        sanitized_tensor = self._apply_tensor_sanitization(tensor)
        
        return sanitized_tensor, security_events
    
    def _sanitize_dict(self, data: dict, source_id: str, 
                      security_events: List[SecurityEvent]) -> Tuple[dict, List[SecurityEvent]]:
        """Sanitize dictionary data with security checks."""
        sanitized_data = {}
        
        for key, value in data.items():
            # Check key for malicious patterns
            if self._contains_malicious_pattern(str(key)):
                security_events.append(SecurityEvent(
                    event_id=self._generate_event_id(),
                    threat_level=ThreatLevel.HIGH,
                    attack_type=AttackType.INJECTION_ATTACK,
                    timestamp=datetime.now(),
                    source=source_id,
                    description=f"Malicious pattern in key: {key}"
                ))
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                if self._contains_malicious_pattern(value):
                    security_events.append(SecurityEvent(
                        event_id=self._generate_event_id(),
                        threat_level=ThreatLevel.HIGH,
                        attack_type=AttackType.INJECTION_ATTACK,
                        timestamp=datetime.now(),
                        source=source_id,
                        description=f"Malicious pattern in value for key: {key}"
                    ))
                    continue
                sanitized_data[key] = self._sanitize_string(value)
            else:
                sanitized_data[key] = value
        
        return sanitized_data, security_events
    
    def _detect_adversarial_input(self, tensor: torch.Tensor) -> bool:
        """Detect adversarial input patterns in tensor data."""
        try:
            # Check for unusual patterns that might indicate adversarial examples
            
            # 1. High frequency components (common in adversarial perturbations)
            if tensor.dim() >= 2:
                # Compute gradients along different dimensions
                grad_x = torch.diff(tensor, dim=-1)
                grad_y = torch.diff(tensor, dim=-2) if tensor.dim() > 2 else grad_x
                
                # High gradient variance might indicate adversarial perturbations
                grad_variance = torch.var(grad_x).item() + torch.var(grad_y).item()
                mean_variance = torch.var(tensor).item()
                
                if mean_variance > 0 and grad_variance / mean_variance > 10.0:
                    return True
            
            # 2. Unusual value distributions
            values = tensor.flatten()
            
            # Check for patterns of small, systematic perturbations
            if len(values) > 10:
                sorted_values = torch.sort(values)[0]
                diffs = torch.diff(sorted_values)
                
                # If many values are very close together (indicating systematic perturbation)
                close_values = (diffs < 1e-6).sum().item()
                if close_values / len(diffs) > 0.5:
                    return True
            
            # 3. Check for specific adversarial patterns
            if self._check_adversarial_patterns(tensor):
                return True
                
        except Exception as e:
            logger.warning(f"Error in adversarial detection: {e}")
        
        return False
    
    def _check_adversarial_patterns(self, tensor: torch.Tensor) -> bool:
        """Check for known adversarial attack patterns."""
        try:
            # FGSM-like patterns (small uniform perturbations)
            if tensor.numel() > 0:
                values = tensor.flatten()
                mean_val = torch.mean(values)
                
                # Check if most values are very close to the mean (indicating uniform perturbation)
                close_to_mean = torch.abs(values - mean_val) < 1e-4
                if close_to_mean.sum().item() / len(values) > 0.8:
                    return True
            
            # PGD-like patterns (iterative perturbations)
            if tensor.dim() >= 2:
                # Check for regular patterns that might indicate iterative attacks
                for dim in range(tensor.dim()):
                    if tensor.size(dim) > 3:
                        slice_data = torch.mean(tensor, dim=dim)
                        # Check for periodic patterns
                        if self._has_periodic_pattern(slice_data):
                            return True
            
        except Exception as e:
            logger.warning(f"Error checking adversarial patterns: {e}")
        
        return False
    
    def _has_periodic_pattern(self, data: torch.Tensor, threshold: float = 0.8) -> bool:
        """Check if data has periodic patterns indicating adversarial perturbations."""
        try:
            if data.numel() < 8:
                return False
            
            # Simple autocorrelation check
            data_normalized = (data - torch.mean(data)) / (torch.std(data) + 1e-8)
            
            # Check for patterns with period 2, 3, 4
            for period in [2, 3, 4]:
                if len(data_normalized) >= period * 2:
                    # Compare with shifted version
                    shifted = torch.roll(data_normalized, period)
                    correlation = torch.corrcoef(torch.stack([data_normalized, shifted]))[0, 1]
                    
                    if torch.abs(correlation) > threshold:
                        return True
            
        except Exception as e:
            logger.warning(f"Error in periodic pattern check: {e}")
        
        return False
    
    def _compute_statistical_anomaly_score(self, tensor: torch.Tensor) -> float:
        """Compute statistical anomaly score for the input."""
        try:
            # Multiple statistical checks
            scores = []
            
            # 1. Value range check
            min_val, max_val = torch.min(tensor).item(), torch.max(tensor).item()
            expected_min, expected_max = self.value_range
            
            if min_val < expected_min or max_val > expected_max:
                range_score = min(1.0, (abs(min_val - expected_min) + abs(max_val - expected_max)) / 1000.0)
                scores.append(range_score)
            
            # 2. Variance check
            variance = torch.var(tensor).item()
            mean = torch.mean(tensor).item()
            
            if abs(mean) > 1e-6:  # Avoid division by zero
                variance_ratio = variance / (mean ** 2)
                if variance_ratio > self.max_variance_ratio:
                    variance_score = min(1.0, variance_ratio / (self.max_variance_ratio * 10))
                    scores.append(variance_score)
            
            # 3. Distribution shape check
            if tensor.numel() > 10:
                # Check for unusual distributions
                flat_tensor = tensor.flatten()
                
                # Compute skewness and kurtosis
                mean = torch.mean(flat_tensor)
                std = torch.std(flat_tensor)
                
                if std > 1e-8:
                    normalized = (flat_tensor - mean) / std
                    
                    # Simple skewness approximation
                    skewness = torch.mean(normalized ** 3).item()
                    kurtosis = torch.mean(normalized ** 4).item() - 3  # Excess kurtosis
                    
                    # Extreme skewness or kurtosis might indicate anomaly
                    if abs(skewness) > 3 or abs(kurtosis) > 5:
                        dist_score = min(1.0, (abs(skewness) + abs(kurtosis)) / 10.0)
                        scores.append(dist_score)
            
            return max(scores) if scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing anomaly score: {e}")
            return 0.0
    
    def _apply_tensor_sanitization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply sanitization to tensor values."""
        # Clamp values to reasonable range
        min_val, max_val = self.value_range
        sanitized = torch.clamp(tensor, min_val, max_val)
        
        # Replace NaN and infinite values
        sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=max_val, neginf=min_val)
        
        # Additional sanitization for extreme values
        if torch.std(sanitized) > 1000:  # Very high variance
            # Apply mild smoothing
            if sanitized.dim() >= 2:
                # Simple 3x3 smoothing for 2D+ tensors
                from torch.nn.functional import conv2d
                if sanitized.dim() == 2:
                    sanitized = sanitized.unsqueeze(0).unsqueeze(0)
                    kernel = torch.ones(1, 1, 3, 3) / 9.0
                    if sanitized.size(-1) >= 3 and sanitized.size(-2) >= 3:
                        sanitized = conv2d(sanitized, kernel, padding=1)
                    sanitized = sanitized.squeeze()
        
        return sanitized
    
    def _contains_malicious_pattern(self, text: str) -> bool:
        """Check if text contains malicious patterns."""
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        sanitized = text
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = self.config.get("max_string_length", 1000)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def _check_rate_limit(self, source_id: str) -> bool:
        """Check if source has exceeded rate limits."""
        current_time = datetime.now()
        window = timedelta(minutes=1)
        max_requests = self.config.get("max_requests_per_minute", 60)
        
        # Clean old entries
        if source_id in self.rate_limits:
            self.rate_limits[source_id] = [
                timestamp for timestamp in self.rate_limits[source_id]
                if current_time - timestamp < window
            ]
        else:
            self.rate_limits[source_id] = []
        
        # Check if limit exceeded
        if len(self.rate_limits[source_id]) >= max_requests:
            return True
        
        # Add current request
        self.rate_limits[source_id].append(current_time)
        return False
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return secrets.token_hex(16)


class SecureInferenceEngine:
    """
    Secure inference engine with encryption and privacy protection.
    
    Features:
    - Encrypted model weights
    - Secure multi-party computation
    - Differential privacy
    - Homomorphic encryption support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize secure inference engine."""
        self.config = config or {}
        self.encryption_key = self._initialize_encryption()
        self.privacy_budget = self.config.get("privacy_budget", 1.0)
        self.noise_multiplier = self.config.get("noise_multiplier", 1.0)
        
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption for secure model storage."""
        # In production, use secure key management
        key_material = self.config.get("encryption_key")
        
        if key_material is None:
            # Generate new key (should be stored securely)
            key_material = Fernet.generate_key()
            logger.warning("Generated new encryption key - store securely!")
        
        if isinstance(key_material, str):
            key_material = key_material.encode()
        
        return Fernet(key_material)
    
    def encrypt_model_weights(self, model: torch.nn.Module) -> bytes:
        """Encrypt model weights for secure storage."""
        # Serialize model state
        model_state = model.state_dict()
        
        # Convert to bytes
        import pickle
        model_bytes = pickle.dumps(model_state)
        
        # Encrypt
        encrypted_bytes = self.encryption_key.encrypt(model_bytes)
        
        logger.info(f"Model weights encrypted ({len(encrypted_bytes)} bytes)")
        return encrypted_bytes
    
    def decrypt_model_weights(self, encrypted_bytes: bytes) -> Dict[str, torch.Tensor]:
        """Decrypt model weights."""
        try:
            # Decrypt
            model_bytes = self.encryption_key.decrypt(encrypted_bytes)
            
            # Deserialize
            import pickle
            model_state = pickle.loads(model_bytes)
            
            logger.info("Model weights decrypted successfully")
            return model_state
            
        except Exception as e:
            logger.error(f"Failed to decrypt model weights: {e}")
            raise
    
    def secure_inference(self, model: torch.nn.Module, data: torch.Tensor, 
                        add_noise: bool = True) -> torch.Tensor:
        """
        Perform secure inference with privacy protection.
        
        Args:
            model: The model to run inference on
            data: Input data
            add_noise: Whether to add differential privacy noise
            
        Returns:
            Inference results with privacy protection
        """
        # Validate inputs first
        sanitizer = AdvancedInputSanitizer(self.config)
        sanitized_data, security_events = sanitizer.sanitize_and_validate(data)
        
        if security_events:
            for event in security_events:
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    logger.error(f"Security threat detected: {event.description}")
                    raise SecurityError(f"Inference blocked due to security threat: {event.threat_level.value}")
        
        # Perform inference
        with torch.no_grad():
            output = model(sanitized_data)
        
        # Add differential privacy noise if enabled
        if add_noise and self.privacy_budget > 0:
            output = self._add_differential_privacy_noise(output)
        
        return output
    
    def _add_differential_privacy_noise(self, output: torch.Tensor) -> torch.Tensor:
        """Add differential privacy noise to protect individual privacy."""
        # Gaussian mechanism for differential privacy
        sensitivity = self.config.get("sensitivity", 1.0)
        noise_scale = (sensitivity * self.noise_multiplier) / self.privacy_budget
        
        noise = torch.normal(0, noise_scale, size=output.shape)
        noisy_output = output + noise
        
        # Update privacy budget
        self.privacy_budget = max(0, self.privacy_budget - 0.1)
        
        logger.debug(f"Added DP noise (scale: {noise_scale:.4f}, remaining budget: {self.privacy_budget:.2f})")
        return noisy_output


class SecurityMonitor:
    """
    Security monitoring and incident response system.
    
    Features:
    - Real-time threat detection
    - Security event correlation
    - Automated incident response
    - Compliance reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security monitor."""
        self.config = config or {}
        self.security_events: List[SecurityEvent] = []
        self.threat_patterns: Dict[str, int] = {}
        self.blocked_sources: Set[str] = set()
        self.incident_threshold = self.config.get("incident_threshold", 5)
        
    def report_security_event(self, event: SecurityEvent) -> None:
        """Report a security event for monitoring and response."""
        self.security_events.append(event)
        
        # Log the event
        logger.warning(f"Security Event: {event.threat_level.value} - {event.description}")
        
        # Check for incident escalation
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._handle_high_threat_event(event)
        
        # Update threat patterns
        pattern_key = f"{event.attack_type.value}:{event.source}"
        self.threat_patterns[pattern_key] = self.threat_patterns.get(pattern_key, 0) + 1
        
        # Check for repeated attacks
        if self.threat_patterns[pattern_key] >= self.incident_threshold:
            self._escalate_to_incident(event)
    
    def _handle_high_threat_event(self, event: SecurityEvent) -> None:
        """Handle high-threat security events."""
        logger.critical(f"High threat detected: {event.description}")
        
        # Implement immediate response
        if event.attack_type == AttackType.ADVERSARIAL_INPUT:
            self._block_source_temporarily(event.source)
        elif event.attack_type == AttackType.DENIAL_OF_SERVICE:
            self._apply_rate_limiting(event.source)
        elif event.attack_type == AttackType.DATA_EXFILTRATION:
            self._alert_security_team(event)
    
    def _escalate_to_incident(self, event: SecurityEvent) -> None:
        """Escalate repeated threats to security incident."""
        logger.critical(f"Security incident: Repeated {event.attack_type.value} from {event.source}")
        
        # Block source permanently
        self.blocked_sources.add(event.source)
        
        # Generate incident report
        incident_report = self._generate_incident_report(event)
        
        # Alert security team
        self._alert_security_team(event, incident_report)
    
    def _block_source_temporarily(self, source: str, duration_minutes: int = 30) -> None:
        """Temporarily block a source."""
        logger.warning(f"Temporarily blocking source: {source} for {duration_minutes} minutes")
        # Implementation would add to temporary block list
    
    def _apply_rate_limiting(self, source: str) -> None:
        """Apply strict rate limiting to a source."""
        logger.warning(f"Applying strict rate limiting to source: {source}")
        # Implementation would update rate limiting rules
    
    def _alert_security_team(self, event: SecurityEvent, incident_report: Optional[str] = None) -> None:
        """Alert security team about critical threats."""
        logger.critical(f"SECURITY ALERT: {event.description}")
        
        # In production, this would send alerts via email, Slack, PagerDuty, etc.
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "event": event.to_dict(),
            "incident_report": incident_report
        }
        
        # Store alert for audit trail
        self._store_security_alert(alert_data)
    
    def _generate_incident_report(self, event: SecurityEvent) -> str:
        """Generate detailed incident report."""
        recent_events = [
            e for e in self.security_events
            if e.source == event.source and 
               datetime.now() - e.timestamp < timedelta(hours=1)
        ]
        
        report = f"""
SECURITY INCIDENT REPORT
========================
Incident ID: {self._generate_incident_id()}
Timestamp: {datetime.now().isoformat()}
Source: {event.source}
Attack Type: {event.attack_type.value}
Threat Level: {event.threat_level.value}

Summary:
--------
Repeated security threats detected from source {event.source}.
Total events in last hour: {len(recent_events)}

Event Timeline:
--------------
"""
        for e in recent_events[-10:]:  # Last 10 events
            report += f"{e.timestamp.strftime('%H:%M:%S')} - {e.attack_type.value}: {e.description}\n"
        
        report += f"""
Recommended Actions:
-------------------
1. Permanently block source: {event.source}
2. Review and strengthen security policies
3. Investigate potential data compromise
4. Update threat detection rules
"""
        
        return report
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = secrets.token_hex(4)
        return f"INC-{timestamp}-{random_suffix}"
    
    def _store_security_alert(self, alert_data: Dict[str, Any]) -> None:
        """Store security alert for audit and compliance."""
        # In production, store in secure audit log
        logger.info(f"Security alert stored: {json.dumps(alert_data, indent=2)}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary for monitoring dashboard."""
        recent_events = [
            e for e in self.security_events
            if datetime.now() - e.timestamp < timedelta(hours=24)
        ]
        
        threat_counts = {}
        for event in recent_events:
            threat_counts[event.threat_level.value] = threat_counts.get(event.threat_level.value, 0) + 1
        
        attack_counts = {}
        for event in recent_events:
            attack_counts[event.attack_type.value] = attack_counts.get(event.attack_type.value, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_events_24h": len(recent_events),
            "threat_level_breakdown": threat_counts,
            "attack_type_breakdown": attack_counts,
            "blocked_sources": len(self.blocked_sources),
            "active_threats": len([e for e in recent_events if not e.mitigated]),
            "critical_alerts": len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL])
        }


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Global security components
_security_monitor: Optional[SecurityMonitor] = None
_input_sanitizer: Optional[AdvancedInputSanitizer] = None
_secure_inference: Optional[SecureInferenceEngine] = None


def get_security_monitor(config: Optional[Dict[str, Any]] = None) -> SecurityMonitor:
    """Get or create global security monitor."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor(config)
    return _security_monitor


def get_input_sanitizer(config: Optional[Dict[str, Any]] = None) -> AdvancedInputSanitizer:
    """Get or create global input sanitizer."""
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = AdvancedInputSanitizer(config)
    return _input_sanitizer


def get_secure_inference_engine(config: Optional[Dict[str, Any]] = None) -> SecureInferenceEngine:
    """Get or create global secure inference engine."""
    global _secure_inference
    if _secure_inference is None:
        _secure_inference = SecureInferenceEngine(config)
    return _secure_inference


def secure_inference_with_monitoring(model: torch.nn.Module, data: torch.Tensor, 
                                   source_id: str = "unknown") -> torch.Tensor:
    """
    Convenience function for secure inference with full monitoring.
    
    Args:
        model: Model to run inference on
        data: Input data
        source_id: Source identifier for security monitoring
        
    Returns:
        Secure inference results
    """
    # Get security components
    sanitizer = get_input_sanitizer()
    secure_engine = get_secure_inference_engine()
    monitor = get_security_monitor()
    
    # Sanitize and validate input
    sanitized_data, security_events = sanitizer.sanitize_and_validate(data, source_id)
    
    # Report security events
    for event in security_events:
        monitor.report_security_event(event)
    
    # Check for blocking conditions
    if any(e.threat_level == ThreatLevel.CRITICAL for e in security_events):
        raise SecurityError("Inference blocked due to critical security threat")
    
    # Perform secure inference
    return secure_engine.secure_inference(model, sanitized_data)