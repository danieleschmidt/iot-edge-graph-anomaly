"""
Input validation and security measures for sentiment analysis.
"""
import re
import html
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import unicodedata

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_text: str
    warnings: List[str]
    blocked_content: List[str]
    risk_score: float  # 0.0 (safe) to 1.0 (high risk)


class InputValidator:
    """
    Comprehensive input validation and sanitization for sentiment analysis.
    
    Protects against malicious inputs, injection attacks, and inappropriate content.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STRICT):
        self.security_level = security_level
        
        # Malicious patterns
        self.malicious_patterns = [
            # Script injection
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'data:.*base64', re.IGNORECASE),
            
            # SQL injection patterns
            re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)', re.IGNORECASE),
            re.compile(r'(\b(UNION|OR|AND)\s+(1=1|TRUE|FALSE)\b)', re.IGNORECASE),
            re.compile(r'[\'";]', re.IGNORECASE),
            
            # Command injection
            re.compile(r'[;&|`$(){}[\]\\]'),
            re.compile(r'\b(rm|cp|mv|ls|cat|grep|awk|sed|wget|curl)\b'),
            
            # Path traversal
            re.compile(r'\.\./'),
            re.compile(r'\.\.\\\\'),
            
            # Email/URL extraction attempts
            re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
        ]
        
        # Suspicious patterns for higher risk scores
        self.suspicious_patterns = [
            re.compile(r'\b(password|secret|key|token|api)\b', re.IGNORECASE),
            re.compile(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'),  # Credit card pattern
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN pattern
            re.compile(r'[A-Z]{2,10}\s*[0-9]{6,}'),  # ID patterns
        ]
        
        # Content filters
        self.profanity_patterns = self._load_profanity_patterns()
        
        # Character limits by security level
        self.max_lengths = {
            SecurityLevel.BASIC: 10000,
            SecurityLevel.STRICT: 5000,
            SecurityLevel.PARANOID: 1000
        }
        
        logger.info(f"Initialized input validator with {security_level.value} security level")
    
    def _load_profanity_patterns(self) -> List[re.Pattern]:
        """Load profanity patterns (basic implementation)."""
        # Basic profanity list - in production, use a comprehensive filter
        basic_profanity = [
            r'\b(damn|hell|crap|stupid|idiot)\b',
            # Add more patterns as needed
        ]
        
        return [re.compile(pattern, re.IGNORECASE) for pattern in basic_profanity]
    
    def validate_text(self, text: str) -> ValidationResult:
        """
        Validate and sanitize input text.
        
        Args:
            text: Input text to validate
            
        Returns:
            ValidationResult with validation status and sanitized text
        """
        if not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                sanitized_text="",
                warnings=["Input must be a string"],
                blocked_content=["non-string input"],
                risk_score=1.0
            )
        
        warnings = []
        blocked_content = []
        risk_score = 0.0
        
        # Length validation
        max_length = self.max_lengths[self.security_level]
        if len(text) > max_length:
            warnings.append(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
            risk_score += 0.1
        
        # Unicode normalization
        try:
            text = unicodedata.normalize('NFKC', text)
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            warnings.append("Unicode normalization failed")
            risk_score += 0.2
        
        # HTML entity decoding and escaping
        text = html.unescape(text)
        original_text = text
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            matches = pattern.findall(text)
            if matches:
                blocked_content.extend(matches)
                text = pattern.sub('[BLOCKED]', text)
                risk_score += 0.3
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if pattern.search(text):
                warnings.append("Potentially sensitive content detected")
                risk_score += 0.2
                
                if self.security_level == SecurityLevel.PARANOID:
                    # Redact suspicious content in paranoid mode
                    text = pattern.sub('[REDACTED]', text)
        
        # Profanity filtering (optional)
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.profanity_patterns:
                if pattern.search(text):
                    warnings.append("Profanity detected and filtered")
                    text = pattern.sub('[FILTERED]', text)
                    risk_score += 0.1
        
        # Control character removal
        control_chars = ''.join(chr(i) for i in range(32) if i not in [9, 10, 13])  # Keep tab, newline, carriage return
        text = text.translate(str.maketrans('', '', control_chars))
        
        # Whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Final validation
        is_valid = len(text) > 0 and risk_score < 0.8
        
        if not is_valid and len(text) == 0:
            warnings.append("Text is empty after sanitization")
        
        if risk_score >= 0.8:
            warnings.append("High risk content detected")
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_text=text,
            warnings=warnings,
            blocked_content=blocked_content,
            risk_score=min(risk_score, 1.0)
        )
    
    def validate_batch(self, texts: List[str]) -> List[ValidationResult]:
        """Validate a batch of texts."""
        return [self.validate_text(text) for text in texts]
    
    def is_safe_for_processing(self, validation_result: ValidationResult) -> bool:
        """Check if validated text is safe for sentiment analysis."""
        return (
            validation_result.is_valid and
            validation_result.risk_score < 0.5 and
            len(validation_result.blocked_content) == 0
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security configuration report."""
        return {
            "security_level": self.security_level.value,
            "max_length": self.max_lengths[self.security_level],
            "malicious_patterns": len(self.malicious_patterns),
            "suspicious_patterns": len(self.suspicious_patterns),
            "profanity_patterns": len(self.profanity_patterns),
            "validation_features": [
                "malicious_pattern_detection",
                "suspicious_content_detection", 
                "length_limits",
                "unicode_normalization",
                "html_sanitization",
                "control_character_removal",
                "profanity_filtering" if self.security_level != SecurityLevel.BASIC else None
            ]
        }


class RateLimiter:
    """
    Rate limiting for API endpoints.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = {}  # client_id -> [(timestamp, count), ...]
        
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed for client.
        
        Returns:
            (allowed, info_dict)
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if timestamp > cutoff_time
            ]
        else:
            self.requests[client_id] = []
        
        # Count current requests
        current_count = sum(count for _, count in self.requests[client_id])
        
        # Check limit
        allowed = current_count < self.max_requests
        
        if allowed:
            # Add current request
            self.requests[client_id].append((current_time, 1))
        
        return allowed, {
            "current_count": current_count + (1 if allowed else 0),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "reset_time": current_time + self.time_window
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        active_clients = 0
        total_requests = 0
        
        for client_id, requests in self.requests.items():
            recent_requests = [
                count for timestamp, count in requests
                if timestamp > cutoff_time
            ]
            if recent_requests:
                active_clients += 1
                total_requests += sum(recent_requests)
        
        return {
            "active_clients": active_clients,
            "total_requests_in_window": total_requests,
            "max_requests_per_client": self.max_requests,
            "time_window_seconds": self.time_window
        }


# Global instances
input_validator = InputValidator(SecurityLevel.STRICT)
rate_limiter = RateLimiter(max_requests=1000, time_window=3600)  # 1000 requests per hour