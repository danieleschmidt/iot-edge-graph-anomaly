"""
Test suite for main application functionality.
"""
import pytest
import sys
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

def test_main_app_import():
    """Test that main application can be imported."""
    try:
        from iot_edge_anomaly.main import main, IoTAnomalyDetectionApp
    except ImportError:
        pytest.fail("Main application components cannot be imported")

def test_app_initialization():
    """Test IoT anomaly detection app initialization."""
    from iot_edge_anomaly.main import IoTAnomalyDetectionApp
    
    config = {
        'model': {
            'input_size': 5,
            'hidden_size': 32,
            'num_layers': 2
        },
        'monitoring': {
            'otlp_endpoint': 'http://localhost:4317'
        }
    }
    
    app = IoTAnomalyDetectionApp(config)
    assert app is not None
    assert app.config == config

@patch('iot_edge_anomaly.main.LSTMAutoencoder')
@patch('iot_edge_anomaly.main.MetricsExporter')
def test_app_components_initialization(mock_metrics, mock_model):
    """Test that app initializes model and metrics components."""
    from iot_edge_anomaly.main import IoTAnomalyDetectionApp
    
    # Mock the components
    mock_model_instance = Mock()
    mock_model.return_value = mock_model_instance
    mock_metrics_instance = Mock()
    mock_metrics.return_value = mock_metrics_instance
    
    config = {
        'model': {'input_size': 5, 'hidden_size': 32},
        'monitoring': {'otlp_endpoint': 'http://localhost:4317'}
    }
    
    app = IoTAnomalyDetectionApp(config)
    app.initialize_components()
    
    # Verify components were created
    mock_model.assert_called_once()
    mock_metrics.assert_called_once()
    assert app.model is not None
    assert app.metrics_exporter is not None

def test_app_data_processing():
    """Test data processing functionality."""
    from iot_edge_anomaly.main import IoTAnomalyDetectionApp
    import torch
    
    config = {
        'model': {'input_size': 3, 'hidden_size': 16},
        'monitoring': {'otlp_endpoint': 'http://localhost:4317'},
        'anomaly_threshold': 0.5
    }
    
    app = IoTAnomalyDetectionApp(config)
    app.initialize_components()
    
    # Mock sensor data (batch_size=1, seq_len=10, features=3)
    sensor_data = torch.randn(1, 10, 3)
    
    # Process data
    result = app.process_sensor_data(sensor_data)
    
    # Verify result structure
    assert 'reconstruction_error' in result
    assert 'is_anomaly' in result
    assert 'timestamp' in result
    assert isinstance(result['is_anomaly'], bool)

@patch('iot_edge_anomaly.main.time.sleep')
def test_app_main_loop(mock_sleep):
    """Test main processing loop."""
    from iot_edge_anomaly.main import IoTAnomalyDetectionApp
    
    config = {
        'model': {'input_size': 3, 'hidden_size': 16},
        'monitoring': {'otlp_endpoint': 'http://localhost:4317'},
        'processing': {'loop_interval': 1.0, 'max_iterations': 2}
    }
    
    app = IoTAnomalyDetectionApp(config)
    app.initialize_components()
    
    # Mock data source
    app.get_sensor_data = Mock(return_value=torch.randn(1, 10, 3))
    
    # Run limited loop
    app.run_processing_loop()
    
    # Verify loop ran
    assert app.get_sensor_data.call_count == 2
    assert mock_sleep.call_count == 2

def test_main_function_success():
    """Test main function successful execution."""
    from iot_edge_anomaly.main import main
    
    # Mock sys.argv for config
    with patch('sys.argv', ['main.py', '--config', 'test_config.yaml']):
        with patch('iot_edge_anomaly.main.IoTAnomalyDetectionApp') as mock_app:
            mock_app_instance = Mock()
            mock_app.return_value = mock_app_instance
            
            result = main()
            
            assert result == 0
            mock_app_instance.initialize_components.assert_called_once()
            mock_app_instance.run_processing_loop.assert_called_once()

def test_main_function_error_handling():
    """Test main function error handling."""
    from iot_edge_anomaly.main import main
    
    # Mock sys.argv to avoid argument parsing issues
    with patch('sys.argv', ['main.py']):
        with patch('iot_edge_anomaly.main.IoTAnomalyDetectionApp') as mock_app:
            mock_app.side_effect = Exception("Test error")
            
            result = main()
            
            assert result == 1  # Error exit code

def test_app_graceful_shutdown():
    """Test graceful shutdown handling."""
    from iot_edge_anomaly.main import IoTAnomalyDetectionApp
    
    config = {
        'model': {'input_size': 3, 'hidden_size': 16},
        'monitoring': {'otlp_endpoint': 'http://localhost:4317'}
    }
    
    app = IoTAnomalyDetectionApp(config)
    app.initialize_components()
    
    # Test shutdown
    app.shutdown()
    
    # Should not raise exceptions
    assert True

def test_configuration_loading():
    """Test configuration loading from file."""
    from iot_edge_anomaly.main import load_config
    import tempfile
    import yaml
    
    config_data = {
        'model': {'input_size': 5},
        'monitoring': {'otlp_endpoint': 'http://test:4317'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    
    try:
        loaded_config = load_config(config_file)
        assert loaded_config == config_data
    finally:
        Path(config_file).unlink(missing_ok=True)