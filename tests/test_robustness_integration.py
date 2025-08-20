"""
Integration tests for robustness features.

Tests the health monitoring, circuit breaker, and data validation
components to ensure they work correctly in production scenarios.
"""

import pytest
import torch
import time
import logging
from unittest.mock import patch, MagicMock

from src.iot_edge_anomaly.robustness.health_monitor import (
    HealthMonitor, HealthStatus, SystemHealth
)
from src.iot_edge_anomaly.robustness.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitBreakerError
)
from src.iot_edge_anomaly.robustness.data_validator import (
    DataValidator, ValidationStatus, ValidationSeverity
)

logger = logging.getLogger(__name__)


class TestHealthMonitor:
    """Test suite for health monitoring functionality."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create health monitor for testing."""
        return HealthMonitor(
            check_interval=0.1,  # Fast interval for testing
            history_size=10
        )
    
    def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initializes correctly."""
        assert health_monitor.check_interval == 0.1
        assert health_monitor.history_size == 10
        assert health_monitor.current_health is None
        assert len(health_monitor.health_history) == 0
    
    def test_health_monitoring_lifecycle(self, health_monitor):
        """Test starting and stopping health monitoring."""
        # Start monitoring
        health_monitor.start_monitoring()
        assert health_monitor._monitoring is True
        assert health_monitor._monitor_thread is not None
        
        # Give it time to collect some data
        time.sleep(0.3)
        
        # Stop monitoring
        health_monitor.stop_monitoring()
        assert health_monitor._monitoring is False
    
    def test_prediction_recording(self, health_monitor):
        """Test recording predictions for accuracy tracking."""
        # Record some predictions
        health_monitor.record_prediction(0.8, 1.0)
        health_monitor.record_prediction(0.3, 0.0)
        health_monitor.record_prediction(0.9, 1.0)
        
        assert len(health_monitor.model_predictions) == 3
        
        # Test accuracy computation
        accuracy = health_monitor._compute_model_accuracy()
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0
    
    def test_request_tracking(self, health_monitor):
        """Test request and error tracking."""
        # Record successful requests
        for _ in range(10):
            health_monitor.record_request(success=True)
        
        # Record some errors
        for _ in range(2):
            health_monitor.record_request(success=False)
        
        assert health_monitor.total_requests == 12
        assert health_monitor.error_count == 2
        
        error_rate = health_monitor._compute_error_rate()
        assert abs(error_rate - (2/12)) < 0.01
    
    def test_data_quality_tracking(self, health_monitor):
        """Test data quality score tracking."""
        # Record quality scores
        quality_scores = [0.9, 0.8, 0.95, 0.7, 0.85]
        for score in quality_scores:
            health_monitor.record_data_quality(score)
        
        assert len(health_monitor.data_quality_scores) == 5
        
        avg_quality = health_monitor._compute_data_quality()
        expected_avg = sum(quality_scores) / len(quality_scores)
        assert abs(avg_quality - expected_avg) < 0.01
    
    def test_health_summary(self, health_monitor):
        """Test health summary generation."""
        # Record some data
        health_monitor.record_request(success=True)
        health_monitor.record_prediction(0.8, 1.0)
        health_monitor.record_data_quality(0.9)
        
        # Manually check health (since monitoring thread isn't running)
        health_monitor._check_health()
        
        summary = health_monitor.get_health_summary()
        
        assert "status" in summary
        assert "overall_score" in summary
        assert "system_resources" in summary
        assert "performance" in summary


class TestCircuitBreaker:
    """Test suite for circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            name="test_breaker"
        )
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 1.0
        assert cb.name == "test_breaker"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_successful_calls(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def success_func():
            return "success"
        
        # Multiple successful calls should keep circuit closed
        for _ in range(5):
            result = cb.call(success_func)
            assert result == "success"
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == 0
    
    def test_circuit_opening(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise ValueError("Test failure")
        
        # First two failures should keep circuit closed
        for i in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
            assert cb.state == CircuitState.CLOSED
            assert cb.failure_count == i + 1
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3
    
    def test_circuit_breaker_error(self):
        """Test circuit breaker raises error when open."""
        cb = CircuitBreaker(failure_threshold=1)
        
        def failing_func():
            raise ValueError("Test failure")
        
        # Cause circuit to open
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerError
        def any_func():
            return "should not execute"
        
        with pytest.raises(CircuitBreakerError):
            cb.call(any_func)
    
    def test_fallback_function(self):
        """Test circuit breaker with fallback function."""
        def fallback():
            return "fallback_result"
        
        cb = CircuitBreaker(
            failure_threshold=1,
            fallback_function=fallback
        )
        
        def failing_func():
            raise ValueError("Test failure")
        
        # Cause circuit to open
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        # Next call should use fallback
        def any_func():
            return "should not execute"
        
        result = cb.call(any_func)
        assert result == "fallback_result"
    
    def test_circuit_recovery(self):
        """Test circuit breaker recovery from open to closed state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def failing_func():
            raise ValueError("Test failure")
        
        def success_func():
            return "success"
        
        # Open the circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should transition to half-open and succeed
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_decorator_interface(self):
        """Test circuit breaker as decorator."""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Successful calls
        assert test_function(5) == 10
        assert test_function(3) == 6
        
        # Failing calls
        with pytest.raises(ValueError):
            test_function(-1)
        with pytest.raises(ValueError):
            test_function(-2)
        
        # Circuit should now be open
        assert cb.state == CircuitState.OPEN
    
    def test_metrics_collection(self):
        """Test circuit breaker metrics collection."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def test_func(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "success"
        
        # Some successful calls
        cb.call(test_func, False)
        cb.call(test_func, False)
        
        # Some failures
        with pytest.raises(ValueError):
            cb.call(test_func, True)
        
        metrics = cb.get_metrics()
        assert metrics.state == CircuitState.CLOSED
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert metrics.total_requests == 3


class TestDataValidator:
    """Test suite for data validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create data validator for testing."""
        return DataValidator()
    
    @pytest.fixture
    def valid_sensor_data(self):
        """Create valid sensor data for testing."""
        return {
            "sensor_data": torch.randn(10, 5).float(),
            "edge_index": torch.randint(0, 10, (2, 15)).long(),
            "timestamps": torch.linspace(0, 9, 10).double()
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.schema is not None
        assert validator.quality_thresholds is not None
        assert validator.enable_drift_detection is True
    
    def test_valid_data_validation(self, validator, valid_sensor_data):
        """Test validation passes for valid data."""
        result = validator.validate(valid_sensor_data)
        
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.PASSED_WITH_WARNINGS]
        assert result.quality_score > 0.5
        assert isinstance(result.issues, list)
        assert isinstance(result.metadata, dict)
    
    def test_invalid_data_types(self, validator):
        """Test validation fails for invalid data types."""
        invalid_data = {
            "sensor_data": [1, 2, 3],  # Should be tensor
            "edge_index": "invalid"     # Should be tensor
        }
        
        result = validator.validate(invalid_data)
        
        assert result.status == ValidationStatus.FAILED
        assert result.has_errors()
        assert len(result.issues) > 0
        
        # Check for type errors
        type_errors = [issue for issue in result.issues 
                      if issue.severity == ValidationSeverity.ERROR]
        assert len(type_errors) > 0
    
    def test_missing_required_fields(self, validator):
        """Test validation fails for missing required fields."""
        incomplete_data = {
            "edge_index": torch.randint(0, 5, (2, 10)).long()
            # Missing required sensor_data
        }
        
        result = validator.validate(incomplete_data)
        
        assert result.status == ValidationStatus.FAILED
        assert result.has_errors()
        
        # Check for missing field error
        missing_errors = [issue for issue in result.issues 
                         if "missing" in issue.message.lower()]
        assert len(missing_errors) > 0
    
    def test_data_quality_checks(self, validator):
        """Test data quality validation."""
        # Create data with quality issues
        problematic_data = torch.zeros(10, 5)
        problematic_data[0, :] = float('nan')  # NaN values
        problematic_data[1, :] = 1000.0        # Potential outliers
        
        data = {"sensor_data": problematic_data}
        result = validator.validate(data)
        
        assert "missing_ratio" in result.metadata
        assert "outlier_ratio" in result.metadata
        
        # Should have warnings about data quality
        quality_warnings = [issue for issue in result.issues 
                          if issue.severity == ValidationSeverity.WARNING]
        assert len(quality_warnings) > 0
    
    def test_drift_detection(self, validator, valid_sensor_data):
        """Test data drift detection functionality."""
        # First validation to set reference
        result1 = validator.validate(valid_sensor_data, update_reference=True)
        assert "drift_detection" in result1.metadata
        
        # Second validation with similar data
        result2 = validator.validate(valid_sensor_data, update_reference=False)
        
        if "drift_score" in result2.metadata:
            assert result2.metadata["drift_score"] >= 0.0
        
        # Third validation with very different data
        different_data = {
            "sensor_data": torch.randn(10, 5) * 10 + 100  # Very different distribution
        }
        result3 = validator.validate(different_data, update_reference=False)
        
        # Should detect drift if reference is available
        if validator.reference_statistics:
            assert "drift_score" in result3.metadata
            # High drift should trigger warnings
            if result3.metadata["drift_score"] > validator.quality_thresholds["drift_threshold"]:
                drift_warnings = [issue for issue in result3.issues 
                                if "drift" in issue.message.lower()]
                assert len(drift_warnings) > 0
    
    def test_validation_history(self, validator, valid_sensor_data):
        """Test validation history tracking."""
        # Perform multiple validations
        for i in range(5):
            validator.validate(valid_sensor_data)
        
        assert len(validator.validation_history) == 5
        
        # Check history maintains order
        timestamps = [v.timestamp for v in validator.validation_history]
        assert timestamps == sorted(timestamps)
    
    def test_quality_score_calculation(self, validator):
        """Test quality score calculation for different scenarios."""
        # High quality data
        good_data = {"sensor_data": torch.randn(10, 5)}
        result_good = validator.validate(good_data)
        
        # Low quality data (with NaNs and extreme values)
        bad_data = {"sensor_data": torch.full((10, 5), float('nan'))}
        result_bad = validator.validate(bad_data)
        
        assert result_good.quality_score > result_bad.quality_score
        assert 0.0 <= result_good.quality_score <= 1.0
        assert 0.0 <= result_bad.quality_score <= 1.0
    
    def test_validation_summary(self, validator, valid_sensor_data):
        """Test validation summary generation."""
        # Perform some validations
        for _ in range(3):
            validator.validate(valid_sensor_data)
        
        summary = validator.get_validation_summary()
        
        assert "average_quality_score" in summary
        assert "total_validations" in summary
        assert "status_distribution" in summary
        assert "latest_validation" in summary
        assert summary["total_validations"] == 3


class TestIntegratedRobustness:
    """Integration tests for combined robustness features."""
    
    def test_health_monitor_with_circuit_breaker(self):
        """Test health monitor integration with circuit breaker."""
        health_monitor = HealthMonitor(check_interval=0.1)
        cb = CircuitBreaker(failure_threshold=2, name="test_integration")
        
        def monitored_function(should_fail=False):
            health_monitor.record_request(success=not should_fail)
            if should_fail:
                raise ValueError("Test failure")
            return "success"
        
        # Successful operations
        for _ in range(3):
            result = cb.call(monitored_function, False)
            assert result == "success"
        
        # Health monitor should show good metrics
        assert health_monitor.error_count == 0
        assert health_monitor.total_requests == 3
        
        # Failed operations
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(monitored_function, True)
        
        # Health monitor should track errors
        assert health_monitor.error_count == 2
        assert health_monitor.total_requests == 5
        
        # Circuit should be open
        assert cb.state == CircuitState.OPEN
    
    def test_validator_with_health_monitor(self):
        """Test data validator integration with health monitor."""
        validator = DataValidator()
        health_monitor = HealthMonitor()
        
        # Valid data should result in good health metrics
        valid_data = {"sensor_data": torch.randn(10, 5)}
        result = validator.validate(valid_data)
        
        health_monitor.record_data_quality(result.quality_score)
        
        # Invalid data should result in poor health metrics
        invalid_data = {"sensor_data": torch.full((10, 5), float('nan'))}
        result = validator.validate(invalid_data)
        
        health_monitor.record_data_quality(result.quality_score)
        
        # Check health monitor tracked quality scores
        assert len(health_monitor.data_quality_scores) == 2
        quality_avg = health_monitor._compute_data_quality()
        assert 0.0 <= quality_avg <= 1.0


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])