"""Example integration tests for IoT Edge Graph Anomaly Detection."""

import pytest
import numpy as np
import torch
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from contextlib import asynccontextmanager

from tests.mocks.mock_services import (
    MockOTLPExporter,
    MockHealthChecker,
    MockAnomalyDetector,
    MockNetworkService,
    MockEdgeDevice,
    mock_environment_variables
)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model components working together."""
    
    def test_lstm_to_anomaly_detector_integration(self, sample_sensor_data, lstm_model_config):
        """Test LSTM model integration with anomaly detector."""
        data = sample_sensor_data
        config = lstm_model_config
        
        # Mock LSTM model output
        mock_lstm = Mock()
        mock_lstm.forward.return_value = torch.tensor(data, dtype=torch.float32)
        
        # Mock anomaly detector
        detector = MockAnomalyDetector()
        detector.threshold = 0.5
        
        # Simulate integration
        model_output = mock_lstm.forward(torch.tensor(data, dtype=torch.float32))
        reconstruction_error = torch.mean((model_output - torch.tensor(data, dtype=torch.float32)) ** 2).item()
        
        # Use anomaly detector
        result = detector.detect_anomaly(data[0])  # Test with first sample
        
        assert 'is_anomaly' in result
        assert 'reconstruction_error' in result
        assert 'confidence' in result
        assert detector.detection_calls == 1
    
    def test_data_pipeline_integration(self, swat_like_data):
        """Test complete data processing pipeline."""
        df = swat_like_data
        
        # Mock data preprocessing steps
        # 1. Data loading
        assert len(df) > 0
        assert 'timestamp' in df.columns
        
        # 2. Feature extraction (select sensor columns)
        sensor_cols = [col for col in df.columns if col not in ['timestamp', 'Normal/Attack']]
        sensor_data = df[sensor_cols].values
        
        # 3. Normalization
        normalized_data = (sensor_data - np.mean(sensor_data, axis=0)) / np.std(sensor_data, axis=0)
        
        # 4. Sequence generation
        sequence_length = 10
        sequences = []
        for i in range(len(normalized_data) - sequence_length + 1):
            sequences.append(normalized_data[i:i+sequence_length])
        
        sequences = np.array(sequences)
        
        # Verify pipeline results
        assert sequences.shape[0] > 0
        assert sequences.shape[1] == sequence_length
        assert sequences.shape[2] == len(sensor_cols)
        assert not np.isnan(sequences).any()


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring and observability components."""
    
    def test_metrics_export_integration(self, sample_metrics_data):
        """Test metrics export to multiple backends."""
        metrics = sample_metrics_data
        
        # Mock exporters
        otlp_exporter = MockOTLPExporter()
        prometheus_exporter = Mock()
        
        # Export to OTLP
        otlp_exporter.export([metrics])
        
        # Export to Prometheus (mock)
        prometheus_exporter.push(metrics)
        
        # Verify exports
        assert otlp_exporter.export_calls == 1
        assert len(otlp_exporter.exported_metrics) == 1
        prometheus_exporter.push.assert_called_once_with(metrics)
    
    def test_health_check_integration(self, mock_device_config):
        """Test health check system integration."""
        config = mock_device_config
        health_checker = MockHealthChecker()
        
        # Initial health check
        health_status = health_checker.check_health()
        assert health_status['status'] == 'healthy'
        
        # Simulate system stress and recheck
        health_checker.health_status['memory_usage'] = 95.0  # High memory usage
        health_checker.health_status['cpu_usage'] = 85.0    # High CPU usage
        
        # Health check should still pass (under limits)
        health_status = health_checker.check_health()
        assert health_status['status'] == 'healthy'
        
        # Simulate critical failure
        health_checker.set_unhealthy("Model inference failed")
        health_status = health_checker.check_health()
        assert health_status['status'] == 'unhealthy'
        assert 'reason' in health_status
    
    def test_alerting_integration(self):
        """Test alerting system integration."""
        # Mock alert manager
        alert_manager = Mock()
        alert_manager.send_alert = Mock()
        
        # Mock anomaly detector
        detector = MockAnomalyDetector()
        detector.threshold = 0.3
        
        # Simulate anomaly detection
        test_data = np.random.randn(10)
        detection_result = detector.detect_anomaly(test_data)
        
        # If anomaly detected, send alert
        if detection_result['is_anomaly']:
            alert_data = {
                'severity': 'warning',
                'message': 'Anomaly detected in sensor data',
                'details': detection_result,
                'timestamp': time.time(),
            }
            alert_manager.send_alert(alert_data)
            alert_manager.send_alert.assert_called_once_with(alert_data)


@pytest.mark.integration
class TestNetworkIntegration:
    """Integration tests for network communication."""
    
    def test_otlp_export_with_network_conditions(self, sample_metrics_data):
        """Test OTLP export under various network conditions."""
        metrics = sample_metrics_data
        network = MockNetworkService()
        otlp_exporter = MockOTLPExporter()
        
        # Test with good network conditions
        network.set_network_conditions(connected=True, latency=50, packet_loss=0.0)
        
        try:
            # Simulate network request for OTLP export
            response = network.send_request('http://otlp-endpoint:4317', metrics)
            otlp_exporter.export([metrics])
            
            assert response['status'] == 'success'
            assert otlp_exporter.export_calls == 1
        except Exception as e:
            pytest.fail(f"Export should succeed with good network: {e}")
        
        # Test with network issues
        network.set_network_conditions(connected=False)
        otlp_exporter.is_connected = False
        
        with pytest.raises(ConnectionError):
            network.send_request('http://otlp-endpoint:4317', metrics)
        
        with pytest.raises(ConnectionError):
            otlp_exporter.export([metrics])
    
    def test_retry_mechanism_integration(self):
        """Test retry mechanism with network failures."""
        network = MockNetworkService()
        max_retries = 3
        retry_count = 0
        
        # Simulate intermittent network issues
        network.set_network_conditions(connected=True, packet_loss=0.8)  # High packet loss
        
        for attempt in range(max_retries):
            try:
                network.send_request('http://test-endpoint', {'data': 'test'})
                break  # Success, exit retry loop
            except (TimeoutError, ConnectionError):
                retry_count += 1
                if retry_count >= max_retries:
                    break
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        # Should have attempted retries
        assert retry_count > 0
        assert retry_count <= max_retries


@pytest.mark.integration
class TestEdgeDeploymentIntegration:
    """Integration tests for edge deployment scenarios."""
    
    def test_container_deployment_integration(self, mock_device_config):
        """Test container deployment integration."""
        device = MockEdgeDevice()
        config = mock_device_config
        
        # Deploy container
        deployment_config = {
            'image': 'iot-edge-anomaly:latest',
            'port_mappings': {'8000': '8000', '9090': '9090'},
            'environment': {
                'LOG_LEVEL': 'INFO',
                'MODEL_PATH': '/app/models/model.pth',
            },
            'resource_limits': {
                'memory': f"{config['memory_limit']}m",
                'cpu': str(config['cpu_limit']),
            }
        }
        
        # Mock container deployment
        result = device.execute_command(f"docker run -d {deployment_config['image']}")
        assert result['status'] == 'success'
        
        # Verify deployment
        deployment_success = device.deploy_model({
            'model_id': 'lstm-autoencoder-v1',
            'version': '0.1.0',
            'config': deployment_config,
        })
        
        assert deployment_success
        assert len(device.deployed_models) > 0
    
    def test_resource_monitoring_integration(self, mock_device_config):
        """Test resource monitoring integration."""
        device = MockEdgeDevice()
        config = mock_device_config
        
        # Get current resource usage
        resources = device.get_resource_usage()
        
        # Check against limits
        memory_usage_percent = (resources['memory_used'] / resources['memory_total']) * 100
        cpu_usage_percent = resources['cpu_usage']
        storage_usage_percent = (resources['storage_used'] / resources['storage_total']) * 100
        
        # Verify within limits
        assert memory_usage_percent < 90  # 90% threshold
        assert cpu_usage_percent < config['cpu_limit'] * 100  # Convert to percentage
        assert storage_usage_percent < 80  # 80% threshold
        
        # Simulate resource stress
        device.resources['memory_used'] = int(resources['memory_total'] * 0.95)  # 95% memory usage
        
        updated_resources = device.get_resource_usage()
        memory_usage_percent = (updated_resources['memory_used'] / updated_resources['memory_total']) * 100
        
        # Should trigger resource alert
        assert memory_usage_percent > 90
    
    def test_model_update_integration(self):
        """Test model update and rollback integration."""
        device = MockEdgeDevice()
        
        # Deploy initial model
        initial_model = {
            'model_id': 'lstm-autoencoder-v1',
            'version': '0.1.0',
            'checksum': 'abc123',
        }
        
        success = device.deploy_model(initial_model)
        assert success
        assert len(device.deployed_models) == 1
        
        # Deploy updated model
        updated_model = {
            'model_id': 'lstm-autoencoder-v1',
            'version': '0.1.1',
            'checksum': 'def456',
        }
        
        success = device.deploy_model(updated_model)
        assert success
        assert len(device.deployed_models) == 2  # Keep history
        
        # Verify latest model
        latest_model = device.deployed_models[-1]
        assert latest_model['version'] == '0.1.1'


@pytest.mark.integration
class TestDataFlowIntegration:
    """Integration tests for end-to-end data flow."""
    
    async def test_streaming_data_pipeline(self, mock_sensor_generator):
        """Test streaming data processing pipeline."""
        from tests.mocks.mock_services import MockAsyncQueue
        
        # Setup components
        data_queue = MockAsyncQueue(maxsize=100)
        anomaly_detector = MockAnomalyDetector()
        sensor_generator = mock_sensor_generator
        
        # Simulate streaming data
        for _ in range(10):
            sample = sensor_generator.generate_sample()
            await data_queue.put(sample)
        
        # Process streaming data
        processed_samples = 0
        anomalies_detected = 0
        
        while not data_queue.empty():
            sample = await data_queue.get()
            result = anomaly_detector.detect_anomaly(sample)
            
            processed_samples += 1
            if result['is_anomaly']:
                anomalies_detected += 1
        
        # Verify processing
        assert processed_samples == 10
        assert anomaly_detector.detection_calls == 10
        assert data_queue.empty()
    
    def test_batch_processing_integration(self, sample_sensor_data):
        """Test batch processing integration."""
        data = sample_sensor_data
        batch_size = 16
        
        # Mock batch processor
        batch_processor = Mock()
        batch_processor.process_batch = Mock(return_value={'processed': True})
        
        # Process data in batches
        num_batches = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            result = batch_processor.process_batch(batch)
            num_batches += 1
            
            assert result['processed'] is True
        
        # Verify batch processing
        expected_batches = (len(data) + batch_size - 1) // batch_size  # Ceiling division
        assert num_batches == expected_batches
        assert batch_processor.process_batch.call_count == num_batches


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    def test_environment_variable_integration(self):
        """Test configuration loading from environment variables."""
        test_env = {
            'LOG_LEVEL': 'DEBUG',
            'MODEL_PATH': '/test/model.pth',
            'ANOMALY_THRESHOLD': '0.7',
            'BATCH_SIZE': '32',
        }
        
        with mock_environment_variables(test_env):
            import os
            
            # Verify environment variables are set
            assert os.environ['LOG_LEVEL'] == 'DEBUG'
            assert os.environ['MODEL_PATH'] == '/test/model.pth'
            assert float(os.environ['ANOMALY_THRESHOLD']) == 0.7
            assert int(os.environ['BATCH_SIZE']) == 32
    
    def test_configuration_override_integration(self):
        """Test configuration override chain."""
        # Default config
        default_config = {
            'log_level': 'INFO',
            'batch_size': 16,
            'threshold': 0.5,
        }
        
        # Environment overrides
        env_config = {
            'log_level': 'DEBUG',
            'batch_size': 32,
        }
        
        # CLI overrides
        cli_config = {
            'threshold': 0.8,
        }
        
        # Merge configurations (later overrides earlier)
        final_config = {**default_config, **env_config, **cli_config}
        
        # Verify final configuration
        assert final_config['log_level'] == 'DEBUG'  # From env
        assert final_config['batch_size'] == 32      # From env
        assert final_config['threshold'] == 0.8      # From CLI


# Test with different deployment scenarios
@pytest.mark.parametrize("device_type,memory_limit,cpu_limit", [
    ("raspberry-pi-4", 100, 0.25),
    ("nvidia-jetson", 500, 0.5),
    ("intel-nuc", 1000, 0.75),
])
@pytest.mark.integration
def test_multi_device_deployment(device_type, memory_limit, cpu_limit):
    """Test deployment across different device types."""
    device = MockEdgeDevice()
    device.device_id = f"test-{device_type}-001"
    
    # Adjust resources based on device type
    device.resources['memory_total'] = memory_limit * 10  # 10x for total memory
    
    # Deploy model with appropriate resource limits
    deployment_config = {
        'memory_limit': memory_limit,
        'cpu_limit': cpu_limit,
        'device_type': device_type,
    }
    
    success = device.deploy_model({
        'model_id': 'lstm-autoencoder',
        'version': '0.1.0',
        'config': deployment_config,
    })
    
    assert success
    assert len(device.deployed_models) > 0
    
    # Verify resource allocation
    deployed_model = device.deployed_models[0]
    assert deployed_model['config']['memory_limit'] == memory_limit
    assert deployed_model['config']['cpu_limit'] == cpu_limit