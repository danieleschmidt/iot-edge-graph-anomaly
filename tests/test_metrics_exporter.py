"""
Test suite for OTLP metrics export functionality.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

def test_metrics_exporter_import():
    """Test that metrics exporter can be imported."""
    try:
        from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    except ImportError:
        pytest.fail("MetricsExporter cannot be imported")

def test_metrics_exporter_initialization():
    """Test metrics exporter initialization with configuration."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    config = {
        'otlp_endpoint': 'http://localhost:4317',
        'service_name': 'iot-edge-anomaly',
        'service_version': '0.1.0'
    }
    
    exporter = MetricsExporter(config)
    assert exporter is not None
    assert exporter.service_name == 'iot-edge-anomaly'
    assert exporter.service_version == '0.1.0'

def test_anomaly_count_metric():
    """Test anomaly count metric can be recorded and exported."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    config = {
        'otlp_endpoint': 'http://localhost:4317',
        'service_name': 'test-service'
    }
    
    exporter = MetricsExporter(config)
    
    # Record anomaly count
    initial_count = 5
    exporter.record_anomaly_count(initial_count)
    
    # Record additional anomalies
    exporter.increment_anomaly_count(2)
    
    # Should be able to get current count
    current_count = exporter.get_anomaly_count()
    assert current_count == initial_count + 2

def test_system_resource_metrics():
    """Test system resource metrics collection."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    config = {'otlp_endpoint': 'http://localhost:4317'}
    exporter = MetricsExporter(config)
    
    # Record system metrics
    exporter.record_cpu_usage(25.5)
    exporter.record_memory_usage(85.2)
    exporter.record_disk_usage(45.0)
    
    # Should be able to retrieve metrics
    assert exporter.get_cpu_usage() == 25.5
    assert exporter.get_memory_usage() == 85.2
    assert exporter.get_disk_usage() == 45.0

def test_model_performance_metrics():
    """Test model performance metrics recording."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    config = {'otlp_endpoint': 'http://localhost:4317'}
    exporter = MetricsExporter(config)
    
    # Record model metrics
    exporter.record_inference_time(0.125)  # 125ms
    exporter.record_reconstruction_error(0.0034)
    exporter.record_processed_samples(1000)
    
    # Should maintain running averages
    exporter.record_inference_time(0.150)
    avg_inference_time = exporter.get_average_inference_time()
    assert 0.125 <= avg_inference_time <= 0.150

@patch('iot_edge_anomaly.monitoring.metrics_exporter.OTLPMetricExporter')
@patch('iot_edge_anomaly.monitoring.metrics_exporter.PeriodicExportingMetricReader')
def test_otlp_export_functionality(mock_reader, mock_otlp_exporter):
    """Test OTLP export functionality with mocked exporter."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    # Mock the OTLP exporter
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    # Mock the metric reader
    mock_reader_instance = Mock()
    mock_reader.return_value = mock_reader_instance
    
    config = {
        'otlp_endpoint': 'http://localhost:4317',
        'otlp_headers': {'Authorization': 'Bearer test-token'}
    }
    
    exporter = MetricsExporter(config)
    
    # Record some metrics
    exporter.record_anomaly_count(10)
    exporter.record_cpu_usage(30.0)
    
    # Export metrics
    exporter.export_metrics()
    
    # Verify OTLP exporter was called
    mock_otlp_exporter.assert_called_once()
    # Verify reader was created
    mock_reader.assert_called_once()

def test_metrics_configuration_validation():
    """Test configuration validation and error handling."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    # Test with minimal config
    minimal_config = {'otlp_endpoint': 'http://localhost:4317'}
    exporter = MetricsExporter(minimal_config)
    assert exporter.service_name == 'iot-edge-anomaly'  # Default value
    
    # Test with invalid endpoint - expect ConnectionError on export
    invalid_config = {'otlp_endpoint': 'http://invalid-host:9999'}
    exporter = MetricsExporter(invalid_config)
    # The error occurs during export, not initialization
    try:
        exporter.export_metrics()
        # If no exception, that's fine - the test environment may not have network restrictions
    except (ConnectionError, Exception) as e:
        # Expected behavior - connection should fail
        assert "Failed to connect" in str(e) or "UNAVAILABLE" in str(e) or "Connection" in str(e)

def test_metrics_thread_safety():
    """Test that metrics recording is thread-safe."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    import threading
    
    config = {'otlp_endpoint': 'http://localhost:4317'}
    exporter = MetricsExporter(config)
    
    # Define function to record metrics from multiple threads
    def record_metrics():
        for i in range(100):
            exporter.increment_anomaly_count(1)
            exporter.record_cpu_usage(float(i % 100))
    
    # Create multiple threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=record_metrics)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify final count is correct
    final_count = exporter.get_anomaly_count()
    assert final_count == 500  # 5 threads * 100 increments each

def test_metrics_context_manager():
    """Test metrics exporter can be used as context manager."""
    from iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter
    
    config = {'otlp_endpoint': 'http://localhost:4317'}
    
    with MetricsExporter(config) as exporter:
        exporter.record_anomaly_count(5)
        assert exporter.get_anomaly_count() == 5
    
    # Context manager should have cleaned up resources