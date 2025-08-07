"""
Tests for security and validation components.
"""
import pytest
import time
from sentiment_analyzer.security.validation import (
    InputValidator, SecurityLevel, ValidationResult, RateLimiter,
    input_validator, rate_limiter
)


class TestInputValidator:
    """Test input validation functionality."""
    
    def setup_method(self):
        self.validator = InputValidator(SecurityLevel.STRICT)
    
    def test_basic_text_validation(self):
        text = "This is a normal text message."
        result = self.validator.validate_text(text)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.sanitized_text == text
        assert len(result.warnings) == 0
        assert len(result.blocked_content) == 0
        assert result.risk_score == 0.0
    
    def test_malicious_script_detection(self):
        malicious_texts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "vbscript:msgbox('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for text in malicious_texts:
            result = self.validator.validate_text(text)
            assert not result.is_valid or len(result.blocked_content) > 0
            assert result.risk_score > 0
            assert "[BLOCKED]" in result.sanitized_text or not result.is_valid
    
    def test_sql_injection_detection(self):
        sql_injection_texts = [
            "'; DROP TABLE users; --",
            "1=1 OR 1=1",
            "SELECT * FROM users WHERE id=1",
            "UNION SELECT password FROM users"
        ]
        
        for text in sql_injection_texts:
            result = self.validator.validate_text(text)
            assert result.risk_score > 0
            if result.is_valid:
                assert len(result.blocked_content) > 0 or len(result.warnings) > 0
    
    def test_command_injection_detection(self):
        command_texts = [
            "test; rm -rf /",
            "$(cat /etc/passwd)",
            "|ls -la",
            "`whoami`"
        ]
        
        for text in command_texts:
            result = self.validator.validate_text(text)
            assert result.risk_score > 0
    
    def test_length_validation(self):
        # Test different security levels
        validators = {
            SecurityLevel.BASIC: InputValidator(SecurityLevel.BASIC),
            SecurityLevel.STRICT: InputValidator(SecurityLevel.STRICT),
            SecurityLevel.PARANOID: InputValidator(SecurityLevel.PARANOID)
        }
        
        long_text = "A" * 15000  # Very long text
        
        for level, validator in validators.items():
            result = validator.validate_text(long_text)
            max_length = validator.max_lengths[level]
            
            if len(long_text) > max_length:
                assert len(result.warnings) > 0
                assert len(result.sanitized_text) <= max_length
                assert result.risk_score > 0
    
    def test_suspicious_content_detection(self):
        suspicious_texts = [
            "My password is secret123",
            "API key: sk-1234567890abcdef",
            "Credit card: 1234-5678-9012-3456",
            "SSN: 123-45-6789"
        ]
        
        for text in suspicious_texts:
            result = self.validator.validate_text(text)
            assert result.risk_score > 0
            assert len(result.warnings) > 0
    
    def test_unicode_normalization(self):
        # Text with various unicode characters
        unicode_text = "café naïve résumé"
        result = self.validator.validate_text(unicode_text)
        
        assert result.is_valid
        assert isinstance(result.sanitized_text, str)
    
    def test_html_entity_handling(self):
        html_text = "&lt;script&gt;alert('test')&lt;/script&gt;"
        result = self.validator.validate_text(html_text)
        
        # Should decode HTML entities and then detect script
        assert result.risk_score > 0
        assert "[BLOCKED]" in result.sanitized_text or not result.is_valid
    
    def test_control_character_removal(self):
        # Text with control characters
        text_with_controls = "Normal text\x00\x01\x02with controls"
        result = self.validator.validate_text(text_with_controls)
        
        assert result.is_valid
        assert "\x00" not in result.sanitized_text
        assert "\x01" not in result.sanitized_text
        assert "\x02" not in result.sanitized_text
        assert "Normal text with controls" in result.sanitized_text
    
    def test_empty_and_none_input(self):
        # Empty string
        result = self.validator.validate_text("")
        assert not result.is_valid
        assert len(result.warnings) > 0
        
        # None input
        result = self.validator.validate_text(None)
        assert not result.is_valid
        assert len(result.warnings) > 0
        assert result.risk_score == 1.0
    
    def test_batch_validation(self):
        texts = [
            "Normal text",
            "<script>alert('xss')</script>",
            "Another normal text",
            ""
        ]
        
        results = self.validator.validate_batch(texts)
        assert len(results) == len(texts)
        
        for result in results:
            assert isinstance(result, ValidationResult)
        
        # First and third should be valid
        assert results[0].is_valid
        assert results[2].is_valid
        
        # Second should have high risk
        assert results[1].risk_score > 0
        
        # Fourth should be invalid (empty)
        assert not results[3].is_valid
    
    def test_safety_check(self):
        safe_text = "This is completely safe text."
        unsafe_text = "<script>alert('danger')</script>"
        
        safe_result = self.validator.validate_text(safe_text)
        unsafe_result = self.validator.validate_text(unsafe_text)
        
        assert self.validator.is_safe_for_processing(safe_result)
        assert not self.validator.is_safe_for_processing(unsafe_result)
    
    def test_security_levels(self):
        text_with_profanity = "This is damn good!"
        
        # Basic level - should allow profanity
        basic_validator = InputValidator(SecurityLevel.BASIC)
        basic_result = basic_validator.validate_text(text_with_profanity)
        
        # Strict level - should filter profanity
        strict_validator = InputValidator(SecurityLevel.STRICT)
        strict_result = strict_validator.validate_text(text_with_profanity)
        
        # Paranoid level - should be most restrictive
        paranoid_validator = InputValidator(SecurityLevel.PARANOID)
        paranoid_result = paranoid_validator.validate_text(text_with_profanity)
        
        # Basic should be most permissive
        assert basic_result.risk_score <= strict_result.risk_score
        assert strict_result.risk_score <= paranoid_result.risk_score
    
    def test_security_report(self):
        report = self.validator.get_security_report()
        
        assert isinstance(report, dict)
        assert 'security_level' in report
        assert 'max_length' in report
        assert 'malicious_patterns' in report
        assert 'validation_features' in report
        
        assert report['security_level'] == SecurityLevel.STRICT.value
        assert isinstance(report['validation_features'], list)


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        self.limiter = RateLimiter(max_requests=5, time_window=60)  # 5 requests per minute
    
    def test_basic_rate_limiting(self):
        client_id = "test_client"
        
        # Should allow first few requests
        for i in range(5):
            allowed, info = self.limiter.is_allowed(client_id)
            assert allowed
            assert info['current_count'] == i + 1
            assert info['max_requests'] == 5
        
        # Should block 6th request
        allowed, info = self.limiter.is_allowed(client_id)
        assert not allowed
        assert info['current_count'] == 5  # Doesn't increment when blocked
    
    def test_multiple_clients(self):
        client1 = "client1"
        client2 = "client2"
        
        # Each client should have separate limits
        for i in range(3):
            allowed1, _ = self.limiter.is_allowed(client1)
            allowed2, _ = self.limiter.is_allowed(client2)
            assert allowed1
            assert allowed2
    
    def test_time_window_reset(self):
        # Use short time window for testing
        short_limiter = RateLimiter(max_requests=2, time_window=1)  # 1 second window
        client_id = "test_client"
        
        # Use up limit
        for i in range(2):
            allowed, _ = short_limiter.is_allowed(client_id)
            assert allowed
        
        # Should be blocked
        allowed, _ = short_limiter.is_allowed(client_id)
        assert not allowed
        
        # Wait for window to reset
        time.sleep(1.1)
        
        # Should be allowed again
        allowed, _ = short_limiter.is_allowed(client_id)
        assert allowed
    
    def test_rate_limiter_stats(self):
        client1 = "client1"
        client2 = "client2"
        
        # Make some requests
        self.limiter.is_allowed(client1)
        self.limiter.is_allowed(client1)
        self.limiter.is_allowed(client2)
        
        stats = self.limiter.get_stats()
        
        assert isinstance(stats, dict)
        assert 'active_clients' in stats
        assert 'total_requests_in_window' in stats
        assert stats['active_clients'] >= 2
        assert stats['total_requests_in_window'] >= 3


class TestGlobalInstances:
    """Test global security instances."""
    
    def test_global_input_validator(self):
        # Test that global validator is properly initialized
        assert input_validator is not None
        assert isinstance(input_validator, InputValidator)
        assert input_validator.security_level == SecurityLevel.STRICT
    
    def test_global_rate_limiter(self):
        # Test that global rate limiter is properly initialized
        assert rate_limiter is not None
        assert isinstance(rate_limiter, RateLimiter)
        assert rate_limiter.max_requests == 1000
        assert rate_limiter.time_window == 3600
    
    def test_input_validator_integration(self):
        # Test using the global validator
        test_text = "This is a test message for global validation."
        result = input_validator.validate_text(test_text)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.sanitized_text == test_text
    
    def test_rate_limiter_integration(self):
        # Test using the global rate limiter
        client_id = "integration_test_client"
        allowed, info = rate_limiter.is_allowed(client_id)
        
        assert isinstance(allowed, bool)
        assert isinstance(info, dict)
        assert 'current_count' in info
        assert 'max_requests' in info


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_validation_and_rate_limiting_together(self):
        """Test validation and rate limiting working together."""
        client_id = "test_integration_client"
        
        # First check rate limiting
        allowed, rate_info = rate_limiter.is_allowed(client_id)
        
        if allowed:
            # Then validate input
            text = "This is a safe test message."
            validation_result = input_validator.validate_text(text)
            
            # Both should succeed for safe input
            assert validation_result.is_valid
            assert input_validator.is_safe_for_processing(validation_result)
        
        # Test with malicious input
        malicious_text = "<script>alert('xss')</script>"
        validation_result = input_validator.validate_text(malicious_text)
        
        # Should be caught by validation regardless of rate limiting
        assert not input_validator.is_safe_for_processing(validation_result)
    
    def test_security_configuration_consistency(self):
        """Test that security configurations are consistent."""
        validator_report = input_validator.get_security_report()
        limiter_stats = rate_limiter.get_stats()
        
        # Both should be properly configured
        assert validator_report['security_level'] == 'strict'
        assert limiter_stats['max_requests_per_client'] == 1000
        
        # Security features should be enabled
        features = validator_report['validation_features']
        assert 'malicious_pattern_detection' in features
        assert 'suspicious_content_detection' in features
        assert 'html_sanitization' in features


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_validator_with_extreme_inputs(self):
        validator = InputValidator(SecurityLevel.STRICT)
        
        # Very large input
        huge_text = "A" * 100000
        result = validator.validate_text(huge_text)
        assert len(result.sanitized_text) <= 5000  # Should be truncated
        
        # Binary data
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        result = validator.validate_text(binary_data.decode('latin1'))
        assert isinstance(result, ValidationResult)
        
        # Unicode edge cases
        unicode_edge = "🎉💯🚀" * 1000
        result = validator.validate_text(unicode_edge)
        assert isinstance(result, ValidationResult)
    
    def test_rate_limiter_edge_cases(self):
        # Zero limit
        zero_limiter = RateLimiter(max_requests=0, time_window=60)
        allowed, _ = zero_limiter.is_allowed("test")
        assert not allowed
        
        # Very short window
        short_limiter = RateLimiter(max_requests=10, time_window=0.1)
        allowed, _ = short_limiter.is_allowed("test")
        assert allowed  # Should work initially
        
        # Very long client ID
        long_client_id = "x" * 10000
        allowed, _ = rate_limiter.is_allowed(long_client_id)
        assert isinstance(allowed, bool)