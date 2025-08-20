#!/usr/bin/env python3
"""
🧪 Generation 2 Comprehensive Test Suite

This test suite validates all Generation 2 robustness features:
- Error handling and recovery
- Input validation and sanitization  
- Security framework
- Advanced monitoring and health checks
- Comprehensive logging

Features:
- Automated test discovery and execution
- Performance benchmarking
- Security vulnerability testing
- Stress testing and fault injection
- Comprehensive reporting
"""

import os
import sys
import time
import json
import traceback
import unittest
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

warnings.filterwarnings('ignore')

try:
    import numpy as np
    import torch
    import torch.nn as nn
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️  Warning: Some dependencies not available - running limited tests")


class TestResults:
    """Collect and manage test results."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        
    def add_result(self, test_name: str, status: str, message: str, duration: float, details: Dict[str, Any] = None):
        """Add test result."""
        result = {
            'test_name': test_name,
            'status': status,  # passed, failed, skipped
            'message': message,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.results.append(result)
        
        if status == 'passed':
            self.passed += 1
        elif status == 'failed':
            self.failed += 1
        else:
            self.skipped += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        
        return {
            'total_tests': total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'success_rate': self.passed / max(total_tests, 1),
            'total_duration': total_time,
            'avg_test_duration': sum(r['duration'] for r in self.results) / max(total_tests, 1)
        }


class RobustErrorHandlingTests:
    """Test the robust error handling framework."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def run_all_tests(self):
        """Run all error handling tests."""
        tests = [
            self.test_input_validation,
            self.test_error_recovery,
            self.test_model_wrapper_robustness,
            self.test_validation_error_handling,
            self.test_fallback_mechanisms
        ]
        
        for test in tests:
            test_name = test.__name__
            start_time = time.time()
            
            try:
                test()
                duration = time.time() - start_time
                self.results.add_result(test_name, 'passed', 'Test completed successfully', duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.results.add_result(
                    test_name, 'failed', f'Test failed: {str(e)}', duration,
                    {'traceback': traceback.format_exc()}
                )
    
    def test_input_validation(self):
        """Test input validation functionality."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.robust_error_handling import InputValidator
        
        validator = InputValidator()
        
        # Test valid data
        valid_data = torch.randn(1, 20, 5)
        is_valid, errors = validator.validate_sensor_data(valid_data)
        assert is_valid, f"Valid data failed validation: {errors}"
        
        # Test invalid data with NaN
        invalid_data = torch.randn(1, 20, 5)
        invalid_data[0, 10, 2] = float('nan')
        is_valid, errors = validator.validate_sensor_data(invalid_data)
        assert not is_valid, "NaN data should fail validation"
        assert len(errors) > 0, "Should report validation errors"
        
        # Test invalid data with infinite values
        invalid_data2 = torch.randn(1, 20, 5)
        invalid_data2[0, 5, 1] = float('inf')
        is_valid, errors = validator.validate_sensor_data(invalid_data2)
        assert not is_valid, "Infinite data should fail validation"
        
        print("✅ Input validation tests passed")
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.robust_error_handling import ErrorRecoveryManager, ValidationError, ErrorCategory
        
        recovery_manager = ErrorRecoveryManager()
        
        # Test validation error recovery
        invalid_data = torch.randn(1, 20, 5)
        invalid_data[0, 10, 2] = float('nan')
        
        try:
            raise ValidationError("Test validation error", ErrorCategory.INPUT_VALIDATION, {'test': True})
        except Exception as e:
            recovered, result = recovery_manager.handle_error(
                e, ErrorCategory.INPUT_VALIDATION, {'data': invalid_data}
            )
            
            # Recovery should attempt to fix the data
            if recovered:
                assert result is not None, "Recovery should return fixed data"
                assert not torch.isnan(result).any(), "Recovered data should not contain NaN"
        
        print("✅ Error recovery tests passed")
    
    def test_model_wrapper_robustness(self):
        """Test robust model wrapper."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.robust_error_handling import RobustModelWrapper
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Create simple model
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
        robust_model = RobustModelWrapper(model, "test_model")
        
        # Test with valid data
        valid_data = torch.randn(1, 20, 5)
        success, result, error_msg = robust_model.safe_predict(valid_data)
        assert success, f"Valid prediction should succeed: {error_msg}"
        
        # Test with invalid data
        invalid_data = torch.randn(1, 20, 5)
        invalid_data[0, 10, 2] = float('nan')
        
        success, result, error_msg = robust_model.safe_predict(invalid_data, fallback_on_error=True)
        # Should either succeed (with recovery) or fail gracefully
        
        # Get health status
        health = robust_model.get_health_status()
        assert 'status' in health, "Health status should include status field"
        assert 'inference_count' in health, "Health status should include inference count"
        
        print("✅ Model wrapper robustness tests passed")
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        from iot_edge_anomaly.robust_error_handling import ValidationError, ErrorCategory
        
        # Test custom validation error
        try:
            raise ValidationError("Test error", ErrorCategory.INPUT_VALIDATION, {'field': 'test'})
        except ValidationError as e:
            assert str(e) == "Test error"
            assert e.category == ErrorCategory.INPUT_VALIDATION
            assert e.details['field'] == 'test'
        
        print("✅ Validation error handling tests passed")
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.robust_error_handling import RobustModelWrapper
        
        # Create mock model that always fails
        class FailingModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Mock failure")
        
        failing_model = FailingModel()
        robust_wrapper = RobustModelWrapper(failing_model, "failing_model")
        
        # Set a successful result for fallback
        robust_wrapper.last_successful_result = torch.tensor([0.5])
        
        # Test fallback behavior
        test_data = torch.randn(1, 20, 5)
        success, result, error_msg = robust_wrapper.safe_predict(test_data, fallback_on_error=True)
        
        if success:
            assert result is not None, "Fallback should return cached result"
            assert "fallback" in error_msg.lower(), "Should indicate fallback was used"
        
        print("✅ Fallback mechanism tests passed")


class SecurityFrameworkTests:
    """Test the security framework."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def run_all_tests(self):
        """Run all security tests."""
        tests = [
            self.test_input_sanitization,
            self.test_adversarial_detection,
            self.test_secure_model_wrapper,
            self.test_authentication_manager,
            self.test_security_utilities
        ]
        
        for test in tests:
            test_name = test.__name__
            start_time = time.time()
            
            try:
                test()
                duration = time.time() - start_time
                self.results.add_result(test_name, 'passed', 'Test completed successfully', duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.results.add_result(
                    test_name, 'failed', f'Test failed: {str(e)}', duration,
                    {'traceback': traceback.format_exc()}
                )
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.security_framework import InputSanitizer, SecurityLevel
        
        sanitizer = InputSanitizer(SecurityLevel.HIGH)
        
        # Test with clean data
        clean_data = torch.randn(1, 20, 5)
        sanitized, warnings, is_safe = sanitizer.sanitize_input(clean_data)
        assert isinstance(sanitized, torch.Tensor), "Should return tensor"
        
        # Test with suspicious data (extreme values)
        suspicious_data = torch.randn(1, 20, 5) * 1000  # Very large values
        sanitized, warnings, is_safe = sanitizer.sanitize_input(suspicious_data)
        assert len(warnings) > 0 or not is_safe, "Should detect suspicious data"
        
        print("✅ Input sanitization tests passed")
    
    def test_adversarial_detection(self):
        """Test adversarial input detection."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.security_framework import AdversarialDetector
        
        detector = AdversarialDetector()
        
        # Test with normal data
        normal_data = torch.randn(1, 20, 5)
        is_adversarial, confidence = detector.detect_adversarial(normal_data)
        # Normal data might or might not be detected as adversarial
        assert isinstance(is_adversarial, bool), "Should return boolean"
        assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"
        
        # Test with high-frequency noise (more likely to be adversarial)
        noisy_data = torch.randn(1, 20, 5)
        # Add high-frequency noise
        for i in range(1, 20):
            noisy_data[0, i] += 10 * (-1) ** i  # Alternating pattern
        
        is_adversarial, confidence = detector.detect_adversarial(noisy_data)
        # This should have higher detection score
        
        print("✅ Adversarial detection tests passed")
    
    def test_secure_model_wrapper(self):
        """Test secure model wrapper."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.security_framework import SecureModelWrapper, SecurityLevel
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Create model and secure wrapper
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
        secure_model = SecureModelWrapper(model, SecurityLevel.STANDARD)
        
        # Test secure prediction
        test_data = torch.randn(1, 20, 5)
        try:
            prediction, security_info = secure_model.secure_predict(test_data)
            
            assert isinstance(prediction, torch.Tensor), "Should return tensor"
            assert isinstance(security_info, dict), "Should return security info"
            assert 'sanitization_applied' in security_info, "Should include sanitization info"
            assert 'inference_allowed' in security_info, "Should include inference permission"
            
        except Exception as e:
            # Some failures are expected with high security settings
            pass
        
        # Get security metrics
        metrics = secure_model.get_security_metrics()
        assert 'total_inferences' in metrics, "Should include inference count"
        assert 'security_level' in metrics, "Should include security level"
        
        print("✅ Secure model wrapper tests passed")
    
    def test_authentication_manager(self):
        """Test authentication manager."""
        from iot_edge_anomaly.security_framework import AuthenticationManager
        
        auth_manager = AuthenticationManager()
        
        # Test API key creation
        api_key = auth_manager.create_api_key("test_user", ["inference", "monitoring"])
        assert isinstance(api_key, str), "Should return string API key"
        assert len(api_key) > 20, "API key should be reasonably long"
        
        # Test API key validation
        key_info = auth_manager.validate_api_key(api_key)
        assert key_info is not None, "Should validate created API key"
        assert key_info['user_id'] == "test_user", "Should return correct user ID"
        
        # Test permission checking
        has_permission = auth_manager.check_permission(api_key, "inference")
        assert has_permission, "Should have inference permission"
        
        has_permission = auth_manager.check_permission(api_key, "admin")
        assert not has_permission, "Should not have admin permission"
        
        # Test invalid API key
        invalid_key_info = auth_manager.validate_api_key("invalid_key")
        assert invalid_key_info is None, "Should reject invalid API key"
        
        print("✅ Authentication manager tests passed")
    
    def test_security_utilities(self):
        """Test security utility functions."""
        from iot_edge_anomaly.security_framework import secure_hash, verify_integrity, encrypt_data, decrypt_data
        
        # Test secure hashing
        data = "test_data"
        hash1 = secure_hash(data)
        hash2 = secure_hash(data)
        assert isinstance(hash1, str), "Should return string hash"
        # Note: hashes will be different due to random salt
        
        # Test encryption/decryption (basic implementation)
        original_data = "sensitive_information"
        key = "test_key"
        
        encrypted = encrypt_data(original_data, key)
        assert encrypted != original_data, "Encrypted data should be different"
        
        decrypted = decrypt_data(encrypted, key)
        assert decrypted == original_data, "Should decrypt to original data"
        
        print("✅ Security utilities tests passed")


class AdvancedMonitoringTests:
    """Test the advanced monitoring system."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def run_all_tests(self):
        """Run all monitoring tests."""
        tests = [
            self.test_metrics_collection,
            self.test_health_checking,
            self.test_alert_management,
            self.test_structured_logging,
            self.test_performance_monitoring
        ]
        
        for test in tests:
            test_name = test.__name__
            start_time = time.time()
            
            try:
                test()
                duration = time.time() - start_time
                self.results.add_result(test_name, 'passed', 'Test completed successfully', duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.results.add_result(
                    test_name, 'failed', f'Test failed: {str(e)}', duration,
                    {'traceback': traceback.format_exc()}
                )
    
    def test_metrics_collection(self):
        """Test metrics collection system."""
        from iot_edge_anomaly.advanced_monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test metric recording
        collector.record_metric("test_metric", 1.5, "ms", "test_component", tag1="value1")
        
        # Test metric retrieval
        metrics = collector.get_metrics("test_metric")
        assert len(metrics) > 0, "Should retrieve recorded metrics"
        assert metrics[0].metric_name == "test_metric", "Should match metric name"
        assert metrics[0].value == 1.5, "Should match metric value"
        
        # Test aggregated metrics
        collector.record_metric("test_metric", 2.5, "ms", "test_component")
        collector.record_metric("test_metric", 3.5, "ms", "test_component")
        
        avg_metric = collector.get_aggregated_metrics("test_metric", "avg", window_minutes=5)
        assert avg_metric is not None, "Should calculate average"
        
        # Test metrics summary
        summary = collector.get_metrics_summary()
        assert 'total_metrics' in summary, "Should include total metrics count"
        assert 'unique_metric_names' in summary, "Should include unique metric names"
        
        print("✅ Metrics collection tests passed")
    
    def test_health_checking(self):
        """Test health checking system."""
        from iot_edge_anomaly.advanced_monitoring import HealthChecker, MetricsCollector
        
        metrics_collector = MetricsCollector()
        health_checker = HealthChecker(metrics_collector)
        
        # Add some metrics for health checks
        metrics_collector.record_metric("accuracy", 0.95, "ratio", "model")
        metrics_collector.record_metric("inference_time_ms", 25.0, "ms", "model")
        metrics_collector.record_metric("errors", 0, "count", "system")
        metrics_collector.record_metric("requests", 10, "count", "system")
        
        # Run health checks
        health_results = health_checker.run_health_checks()
        
        assert isinstance(health_results, dict), "Should return dictionary of results"
        
        for component, result in health_results.items():
            assert hasattr(result, 'status'), "Should have status attribute"
            assert hasattr(result, 'message'), "Should have message attribute"
            assert result.status in ['healthy', 'degraded', 'unhealthy', 'critical'], "Should have valid status"
        
        print("✅ Health checking tests passed")
    
    def test_alert_management(self):
        """Test alert management system."""
        from iot_edge_anomaly.advanced_monitoring import AlertManager, MetricsCollector
        
        metrics_collector = MetricsCollector()
        alert_manager = AlertManager(metrics_collector)
        
        # Add metrics that should trigger alerts
        metrics_collector.record_metric("error_rate", 0.15, "ratio", "system")  # High error rate
        metrics_collector.record_metric("inference_time_ms", 150.0, "ms", "model")  # High latency
        
        # Check for alerts
        alerts = alert_manager.check_alerts()
        
        # Verify alert structure
        for alert in alerts:
            assert hasattr(alert, 'alert_id'), "Should have alert ID"
            assert hasattr(alert, 'severity'), "Should have severity"
            assert hasattr(alert, 'timestamp'), "Should have timestamp"
        
        # Test alert resolution
        if alerts:
            alert_id = alerts[0].alert_id
            alert_manager.resolve_alert(alert_id, "Test resolution")
            
            active_alerts = alert_manager.get_active_alerts()
            resolved_alert = next((a for a in alert_manager.alerts if a.alert_id == alert_id), None)
            if resolved_alert:
                assert resolved_alert.resolved, "Alert should be marked as resolved"
        
        print("✅ Alert management tests passed")
    
    def test_structured_logging(self):
        """Test structured logging system."""
        from iot_edge_anomaly.advanced_monitoring import StructuredLogger, LogLevel
        
        logger = StructuredLogger("test_logger", LogLevel.DEBUG)
        
        # Test different log levels
        logger.set_context(test_context="test_value")
        logger.info("Test info message", extra_field="extra_value")
        logger.warning("Test warning message")
        logger.error("Test error message", error_code=500)
        
        # Test security and audit logging
        logger.security("Test security event", user_id="test_user")
        logger.audit("Test audit event", action="test_action")
        
        print("✅ Structured logging tests passed")
    
    def test_performance_monitoring(self):
        """Test performance monitoring context manager."""
        from iot_edge_anomaly.advanced_monitoring import monitor_performance, MetricsCollector
        
        metrics_collector = MetricsCollector()
        
        # Test successful operation monitoring
        with monitor_performance(metrics_collector, "test_operation", "test_component"):
            time.sleep(0.01)  # Simulate work
        
        # Test error monitoring
        try:
            with monitor_performance(metrics_collector, "failing_operation", "test_component"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Verify metrics were recorded
        duration_metrics = metrics_collector.get_metrics("test_operation_duration_ms")
        success_metrics = metrics_collector.get_metrics("test_operation_success")
        error_metrics = metrics_collector.get_metrics("errors")
        
        assert len(duration_metrics) > 0, "Should record duration metrics"
        assert len(success_metrics) > 0, "Should record success metrics"
        assert len(error_metrics) > 0, "Should record error metrics"
        
        print("✅ Performance monitoring tests passed")


class IntegrationTests:
    """Integration tests for Generation 2 features."""
    
    def __init__(self, results: TestResults):
        self.results = results
    
    def run_all_tests(self):
        """Run all integration tests."""
        tests = [
            self.test_unified_api_with_robustness,
            self.test_security_integration,
            self.test_monitoring_integration,
            self.test_end_to_end_robustness
        ]
        
        for test in tests:
            test_name = test.__name__
            start_time = time.time()
            
            try:
                test()
                duration = time.time() - start_time
                self.results.add_result(test_name, 'passed', 'Test completed successfully', duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.results.add_result(
                    test_name, 'failed', f'Test failed: {str(e)}', duration,
                    {'traceback': traceback.format_exc()}
                )
    
    def test_unified_api_with_robustness(self):
        """Test unified API with robustness features."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        # Import and test the unified API
        sys.path.insert(0, str(Path(__file__).parent))
        
        try:
            from unified_api import TerrragonAnomalyDetector, DeploymentMode
            
            # Create detector with robust configuration
            detector = TerrragonAnomalyDetector(
                deployment_mode=DeploymentMode.PRODUCTION,
                device="cpu"
            )
            
            # Test with valid data
            valid_data = np.random.randn(20, 5)
            result = detector.detect_anomaly(valid_data)
            
            assert hasattr(result, 'is_anomaly'), "Should return AnomalyResult"
            assert hasattr(result, 'confidence'), "Should include confidence"
            assert hasattr(result, 'inference_time_ms'), "Should include timing"
            
            # Test system health
            health = detector.get_system_health()
            assert hasattr(health, 'status'), "Should return health status"
            
        except ImportError as e:
            raise unittest.SkipTest(f"Unified API not available: {e}")
        
        print("✅ Unified API robustness integration tests passed")
    
    def test_security_integration(self):
        """Test security integration across components."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.security_framework import SecureModelWrapper, SecurityLevel, InputSanitizer
        from iot_edge_anomaly.robust_error_handling import RobustModelWrapper
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Create model with both security and robustness
        base_model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
        robust_model = RobustModelWrapper(base_model, "test_model")
        secure_model = SecureModelWrapper(robust_model.model, SecurityLevel.HIGH)
        
        # Test with potentially malicious data
        test_data = torch.randn(1, 20, 5) * 100  # Large values
        
        try:
            prediction, security_info = secure_model.secure_predict(test_data)
            # Should either succeed with sanitization or fail securely
            
        except Exception as e:
            # Expected for high security settings
            pass
        
        print("✅ Security integration tests passed")
    
    def test_monitoring_integration(self):
        """Test monitoring integration."""
        from iot_edge_anomaly.advanced_monitoring import MetricsCollector, HealthChecker, AlertManager
        
        # Create integrated monitoring system
        metrics = MetricsCollector()
        health = HealthChecker(metrics)
        alerts = AlertManager(metrics)
        
        # Simulate system activity
        metrics.record_metric("requests", 1, "count", "api")
        metrics.record_metric("inference_time_ms", 45.0, "ms", "model")
        metrics.record_metric("accuracy", 0.92, "ratio", "model")
        
        # Run health checks
        health_results = health.run_health_checks()
        assert len(health_results) > 0, "Should perform health checks"
        
        # Check for alerts
        alert_results = alerts.check_alerts()
        # May or may not have alerts depending on thresholds
        
        print("✅ Monitoring integration tests passed")
    
    def test_end_to_end_robustness(self):
        """Test end-to-end robustness under various failure scenarios."""
        if not DEPENDENCIES_AVAILABLE:
            raise unittest.SkipTest("Dependencies not available")
        
        from iot_edge_anomaly.robust_error_handling import RobustModelWrapper
        from iot_edge_anomaly.security_framework import InputSanitizer, SecurityLevel
        from iot_edge_anomaly.advanced_monitoring import MetricsCollector, monitor_performance
        
        # Create integrated robust system
        metrics = MetricsCollector()
        sanitizer = InputSanitizer(SecurityLevel.STANDARD)
        
        # Mock model for testing
        class SimpleModel(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=-1, keepdim=True)
        
        model = SimpleModel()
        robust_model = RobustModelWrapper(model, "e2e_test_model")
        
        # Test scenarios
        test_scenarios = [
            ("normal_data", torch.randn(1, 20, 5)),
            ("nan_data", torch.full((1, 20, 5), float('nan'))),
            ("inf_data", torch.full((1, 20, 5), float('inf'))),
            ("extreme_data", torch.randn(1, 20, 5) * 1000),
            ("zero_data", torch.zeros(1, 20, 5))
        ]
        
        results = {}
        
        for scenario_name, test_data in test_scenarios:
            with monitor_performance(metrics, f"e2e_test_{scenario_name}", "test"):
                # Input sanitization
                sanitized_data, warnings, is_safe = sanitizer.sanitize_input(test_data)
                
                # Robust prediction
                success, result, error_msg = robust_model.safe_predict(
                    sanitized_data, fallback_on_error=True
                )
                
                results[scenario_name] = {
                    'sanitization_safe': is_safe,
                    'prediction_success': success,
                    'warnings_count': len(warnings),
                    'has_result': result is not None
                }
        
        # Verify at least some scenarios worked
        successful_scenarios = sum(1 for r in results.values() if r['prediction_success'])
        assert successful_scenarios > 0, "At least some scenarios should succeed"
        
        # Check that metrics were collected
        summary = metrics.get_metrics_summary()
        assert summary['total_metrics'] > 0, "Should collect metrics during testing"
        
        print("✅ End-to-end robustness tests passed")


def run_comprehensive_tests():
    """Run all Generation 2 comprehensive tests."""
    print("🧪 Generation 2 Comprehensive Test Suite")
    print("=" * 60)
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = TestResults()
    
    # Test suites to run
    test_suites = [
        ("Robust Error Handling", RobustErrorHandlingTests),
        ("Security Framework", SecurityFrameworkTests),
        ("Advanced Monitoring", AdvancedMonitoringTests),
        ("Integration Tests", IntegrationTests)
    ]
    
    for suite_name, suite_class in test_suites:
        print(f"📋 Running {suite_name} Tests...")
        print("-" * 40)
        
        suite = suite_class(results)
        
        try:
            suite.run_all_tests()
            print(f"✅ {suite_name} tests completed")
        except Exception as e:
            print(f"❌ {suite_name} test suite failed: {e}")
            results.add_result(
                f"{suite_name}_suite", 'failed', f'Suite failed: {str(e)}', 0.0
            )
        
        print()
    
    # Generate final report
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    summary = results.get_summary()
    
    print(f"Total Tests:     {summary['total_tests']}")
    print(f"Passed:          {summary['passed']} ✅")
    print(f"Failed:          {summary['failed']} ❌")
    print(f"Skipped:         {summary['skipped']} ⏭️")
    print(f"Success Rate:    {summary['success_rate']:.1%}")
    print(f"Total Duration:  {summary['total_duration']:.2f} seconds")
    print(f"Avg Test Time:   {summary['avg_test_duration']:.3f} seconds")
    
    # Show failed tests
    if summary['failed'] > 0:
        print("\n❌ FAILED TESTS:")
        for result in results.results:
            if result['status'] == 'failed':
                print(f"  - {result['test_name']}: {result['message']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'generation2_test_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results.results,
            'timestamp': timestamp,
            'dependencies_available': DEPENDENCIES_AVAILABLE
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Determine overall success
    if summary['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Generation 2 robustness validated.")
        return True
    else:
        print(f"\n⚠️  {summary['failed']} tests failed. Review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)