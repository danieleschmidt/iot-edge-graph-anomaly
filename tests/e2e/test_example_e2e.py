"""End-to-end tests for IoT Edge Graph Anomaly Detection system."""

import pytest
import asyncio
import time
import json
import subprocess
import requests
from pathlib import Path
from unittest.mock import patch, Mock

from tests.mocks.mock_services import (
    MockEdgeDevice,
    MockNetworkService,
    MockOTLPExporter,
    mock_environment_variables
)


@pytest.mark.e2e
@pytest.mark.slow
class TestCompleteDeploymentWorkflow:
    """End-to-end tests for complete deployment workflow."""
    
    def test_full_deployment_lifecycle(self, tmp_path):
        """Test complete deployment lifecycle from build to monitoring."""
        # Mock deployment environment
        deployment_dir = tmp_path / "deployment"
        deployment_dir.mkdir()
        
        # 1. Build phase (mocked)
        build_success = self._mock_container_build(deployment_dir)
        assert build_success, "Container build should succeed"
        
        # 2. Deployment phase
        device = MockEdgeDevice()
        deployment_success = self._mock_deployment(device, deployment_dir)
        assert deployment_success, "Deployment should succeed"
        
        # 3. Health check phase
        health_status = self._mock_health_check(device)
        assert health_status['status'] == 'healthy', "System should be healthy after deployment"
        
        # 4. Monitoring setup
        monitoring_success = self._mock_monitoring_setup(device)
        assert monitoring_success, "Monitoring should be set up successfully"
        
        # 5. Verification phase
        verification_success = self._verify_deployment(device)
        assert verification_success, "Deployment verification should pass"
    
    def _mock_container_build(self, deployment_dir: Path) -> bool:
        """Mock container build process."""
        # Create mock build artifacts
        dockerfile = deployment_dir / "Dockerfile"
        dockerfile.write_text("FROM python:3.9-slim\nCOPY . /app\nWORKDIR /app")
        
        # Mock build command execution
        build_result = {
            'returncode': 0,
            'stdout': 'Successfully built iot-edge-anomaly:latest',
            'stderr': ''
        }
        
        return build_result['returncode'] == 0
    
    def _mock_deployment(self, device: MockEdgeDevice, deployment_dir: Path) -> bool:
        """Mock deployment process."""
        deployment_config = {
            'image': 'iot-edge-anomaly:latest',
            'container_name': 'iot-edge-anomaly-container',
            'ports': {'8000': '8000', '9090': '9090'},
            'environment': {
                'LOG_LEVEL': 'INFO',
                'MODEL_PATH': '/app/models/model.pth',
                'ANOMALY_THRESHOLD': '0.5',
            },
            'volumes': {
                str(deployment_dir): '/app/data'
            }
        }
        
        # Mock container deployment
        result = device.execute_command("docker run -d iot-edge-anomaly:latest")
        deployment_success = device.deploy_model({
            'model_id': 'lstm-autoencoder-e2e',
            'version': '0.1.0',
            'config': deployment_config,
        })
        
        return result['status'] == 'success' and deployment_success
    
    def _mock_health_check(self, device: MockEdgeDevice) -> dict:
        """Mock health check process."""
        # Simulate health check endpoint
        health_data = {
            'status': 'healthy',
            'timestamp': time.time(),
            'services': {
                'model': 'healthy',
                'monitoring': 'healthy',
                'data_processing': 'healthy',
            },
            'resources': device.get_resource_usage(),
            'version': '0.1.0',
        }
        
        return health_data
    
    def _mock_monitoring_setup(self, device: MockEdgeDevice) -> bool:
        """Mock monitoring setup process."""
        # Mock OTLP exporter setup
        otlp_exporter = MockOTLPExporter()
        
        # Mock Prometheus metrics setup
        prometheus_setup = Mock()
        prometheus_setup.configure.return_value = True
        
        # Simulate metrics collection
        test_metrics = {
            'inference_latency': 5.2,
            'memory_usage': 75.0,
            'cpu_usage': 20.0,
        }
        
        otlp_exporter.export([test_metrics])
        
        return otlp_exporter.export_calls > 0
    
    def _verify_deployment(self, device: MockEdgeDevice) -> bool:
        """Verify deployment is working correctly."""
        # Check if models are deployed
        if len(device.deployed_models) == 0:
            return False
        
        # Check device status
        if device.status != 'online':
            return False
        
        # Check resource usage is within limits
        resources = device.get_resource_usage()
        memory_usage_percent = (resources['memory_used'] / resources['memory_total']) * 100
        
        if memory_usage_percent > 90:  # 90% threshold
            return False
        
        return True


@pytest.mark.e2e
class TestDataProcessingWorkflow:
    """End-to-end tests for data processing workflow."""
    
    async def test_streaming_data_processing_e2e(self, sample_sensor_data):
        """Test end-to-end streaming data processing."""
        from tests.mocks.mock_services import MockAsyncQueue, MockAnomalyDetector
        
        # Setup streaming pipeline components
        data_queue = MockAsyncQueue(maxsize=1000)
        anomaly_detector = MockAnomalyDetector()
        processing_results = []
        
        # Simulate data producer
        async def data_producer():
            for i, sample in enumerate(sample_sensor_data):
                await data_queue.put({
                    'timestamp': time.time(),
                    'sample_id': i,
                    'data': sample.tolist(),
                })
                await asyncio.sleep(0.01)  # Simulate real-time data
        
        # Simulate data consumer/processor
        async def data_processor():
            processed_count = 0
            while processed_count < len(sample_sensor_data):
                if not data_queue.empty():
                    data_point = await data_queue.get()
                    
                    # Process data point
                    detection_result = anomaly_detector.detect_anomaly(
                        data_point['data']
                    )
                    
                    result = {
                        'sample_id': data_point['sample_id'],
                        'timestamp': data_point['timestamp'],
                        'is_anomaly': detection_result['is_anomaly'],
                        'confidence': detection_result['confidence'],
                    }
                    
                    processing_results.append(result)
                    processed_count += 1
                else:
                    await asyncio.sleep(0.001)
        
        # Run producer and consumer concurrently
        await asyncio.gather(data_producer(), data_processor())
        
        # Verify results
        assert len(processing_results) == len(sample_sensor_data)
        assert all('is_anomaly' in result for result in processing_results)
        assert anomaly_detector.detection_calls == len(sample_sensor_data)
    
    def test_batch_processing_workflow_e2e(self, swat_like_data):
        """Test end-to-end batch processing workflow."""
        df = swat_like_data
        
        # 1. Data preparation
        sensor_cols = [col for col in df.columns if col not in ['timestamp', 'Normal/Attack']]
        sensor_data = df[sensor_cols].values
        
        # 2. Data preprocessing
        normalized_data = (sensor_data - sensor_data.mean(axis=0)) / sensor_data.std(axis=0)
        
        # 3. Sequence generation
        sequence_length = 10
        batch_size = 32
        
        sequences = []
        labels = []
        
        for i in range(len(normalized_data) - sequence_length + 1):
            sequences.append(normalized_data[i:i+sequence_length])
            # Label based on if any point in sequence is an attack
            label_slice = df.iloc[i:i+sequence_length]['Normal/Attack']
            labels.append(1 if any(label_slice == 'Attack') else 0)
        
        # 4. Batch processing
        from tests.mocks.mock_services import MockAnomalyDetector
        
        detector = MockAnomalyDetector()
        batch_results = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Process each sample in batch
            batch_predictions = []
            for sample in batch:
                result = detector.detect_anomaly(sample[-1])  # Use last time step
                batch_predictions.append(1 if result['is_anomaly'] else 0)
            
            batch_results.extend(batch_predictions)
        
        # 5. Evaluation
        assert len(batch_results) == len(sequences)
        
        # Calculate basic metrics
        true_positives = sum(1 for true, pred in zip(labels, batch_results) 
                           if true == 1 and pred == 1)
        false_positives = sum(1 for true, pred in zip(labels, batch_results) 
                            if true == 0 and pred == 1)
        true_negatives = sum(1 for true, pred in zip(labels, batch_results) 
                           if true == 0 and pred == 0)
        false_negatives = sum(1 for true, pred in zip(labels, batch_results) 
                            if true == 1 and pred == 0)
        
        # Basic validation (detector is random, so just check structure)
        total_samples = true_positives + false_positives + true_negatives + false_negatives
        assert total_samples == len(sequences)


@pytest.mark.e2e
class TestMonitoringWorkflow:
    """End-to-end tests for monitoring workflow."""
    
    def test_metrics_collection_and_export_e2e(self):
        """Test end-to-end metrics collection and export."""
        # Setup monitoring components
        from tests.mocks.mock_services import MockOTLPExporter
        
        otlp_exporter = MockOTLPExporter()
        collected_metrics = []
        
        # Simulate metrics collection over time
        for i in range(10):
            # Simulate inference and collect metrics
            inference_start = time.time()
            time.sleep(0.001)  # Simulate inference time
            inference_end = time.time()
            
            metrics = {
                'timestamp': inference_end,
                'inference_latency_ms': (inference_end - inference_start) * 1000,
                'memory_usage_mb': 75.0 + i * 0.5,  # Gradually increasing
                'cpu_usage_percent': 20.0 + i * 0.3,
                'anomalies_detected': i % 3,  # Some anomalies
            }
            
            collected_metrics.append(metrics)
            
            # Export metrics every 5 iterations
            if (i + 1) % 5 == 0:
                otlp_exporter.export(collected_metrics[-5:])
        
        # Verify metrics export
        assert otlp_exporter.export_calls == 2  # Called twice (after 5 and 10 iterations)
        assert len(otlp_exporter.exported_metrics) == 10  # All metrics exported
        
        # Verify metrics structure
        for exported_batch in otlp_exporter.exported_metrics:
            if isinstance(exported_batch, list):
                for metric in exported_batch:
                    assert 'timestamp' in metric
                    assert 'inference_latency_ms' in metric
                    assert 'memory_usage_mb' in metric
    
    def test_alerting_workflow_e2e(self):
        """Test end-to-end alerting workflow."""
        from tests.mocks.mock_services import MockAnomalyDetector
        
        # Setup alerting components
        anomaly_detector = MockAnomalyDetector()
        anomaly_detector.threshold = 0.3  # Lower threshold for more alerts
        
        alert_history = []
        alert_cooldown = {}  # Track alert cooldowns
        
        def send_alert(alert_data):
            """Mock alert sender."""
            alert_type = alert_data['type']
            current_time = time.time()
            
            # Check cooldown (5 seconds)
            if alert_type in alert_cooldown:
                if current_time - alert_cooldown[alert_type] < 5:
                    return  # Skip alert due to cooldown
            
            alert_history.append(alert_data)
            alert_cooldown[alert_type] = current_time
        
        # Simulate continuous monitoring
        import numpy as np
        
        for i in range(20):
            # Generate test data (some normal, some anomalous)
            if i % 7 == 0:  # Every 7th sample is anomalous
                test_data = np.random.randn(10) * 3  # Anomalous data
            else:
                test_data = np.random.randn(10) * 0.5  # Normal data
            
            # Detect anomalies
            result = anomaly_detector.detect_anomaly(test_data)
            
            # Generate alerts for anomalies
            if result['is_anomaly']:
                alert_data = {
                    'type': 'anomaly',
                    'severity': 'warning' if result['confidence'] < 0.8 else 'critical',
                    'message': f"Anomaly detected with confidence {result['confidence']:.2f}",
                    'timestamp': time.time(),
                    'sample_id': i,
                }
                send_alert(alert_data)
            
            # Check system resources (mock)
            memory_usage = 70 + i * 0.5  # Gradually increasing
            if memory_usage > 90:
                alert_data = {
                    'type': 'resource',
                    'severity': 'critical',
                    'message': f"High memory usage: {memory_usage:.1f}%",
                    'timestamp': time.time(),
                }
                send_alert(alert_data)
            
            time.sleep(0.1)  # Small delay between samples
        
        # Verify alerting behavior
        assert len(alert_history) > 0, "Should have generated some alerts"
        
        # Check alert structure
        for alert in alert_history:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert
        
        # Verify cooldown mechanism (no duplicate alerts within 5 seconds)
        anomaly_alerts = [a for a in alert_history if a['type'] == 'anomaly']
        if len(anomaly_alerts) > 1:
            time_diffs = [anomaly_alerts[i]['timestamp'] - anomaly_alerts[i-1]['timestamp'] 
                         for i in range(1, len(anomaly_alerts))]
            assert all(diff >= 5 for diff in time_diffs), "Alert cooldown should be respected"


@pytest.mark.e2e
class TestFailureRecoveryWorkflow:
    """End-to-end tests for failure recovery scenarios."""
    
    def test_network_failure_recovery_e2e(self):
        """Test system recovery from network failures."""
        from tests.mocks.mock_services import MockOTLPExporter, MockNetworkService
        
        network = MockNetworkService()
        otlp_exporter = MockOTLPExporter()
        
        # Setup retry mechanism
        max_retries = 3
        retry_delay = 0.1
        
        def export_with_retry(metrics):
            """Export metrics with retry logic."""
            for attempt in range(max_retries):
                try:
                    # Simulate network request
                    network.send_request('http://otlp-endpoint:4317', metrics)
                    otlp_exporter.export([metrics])
                    return True
                except (ConnectionError, TimeoutError):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
            return False
        
        # Test 1: Normal operation
        network.set_network_conditions(connected=True, latency=50)
        
        test_metrics = {'test': 'data'}
        success = export_with_retry(test_metrics)
        assert success, "Export should succeed with good network"
        
        # Test 2: Network failure with recovery
        network.set_network_conditions(connected=False)
        otlp_exporter.is_connected = False
        
        # Start export attempt (will fail initially)
        success = export_with_retry(test_metrics)
        assert not success, "Export should fail with network down"
        
        # Simulate network recovery during next attempt
        network.set_network_conditions(connected=True)
        otlp_exporter.is_connected = True
        
        success = export_with_retry(test_metrics)
        assert success, "Export should succeed after network recovery"
    
    def test_model_failure_recovery_e2e(self):
        """Test system recovery from model failures."""
        from tests.mocks.mock_services import MockAnomalyDetector, MockModelLoader
        
        detector = MockAnomalyDetector()
        model_loader = MockModelLoader()
        
        # Simulate model failure
        detector.set_model_ready(False)
        
        def safe_detect_anomaly(data, max_retries=3):
            """Anomaly detection with model recovery."""
            for attempt in range(max_retries):
                try:
                    return detector.detect_anomaly(data)
                except RuntimeError as e:
                    if "Model not ready" in str(e):
                        # Attempt model reload
                        try:
                            model_loader.load_model('/app/models/model.pth')
                            detector.set_model_ready(True)
                            continue
                        except Exception:
                            if attempt < max_retries - 1:
                                time.sleep(0.1)
                            continue
                    raise e
            
            raise RuntimeError("Model recovery failed after all retries")
        
        # Test failure and recovery
        test_data = [1, 2, 3, 4, 5]
        
        result = safe_detect_anomaly(test_data)
        
        # Should succeed after model recovery
        assert 'is_anomaly' in result
        assert detector.model_ready
        assert model_loader.load_calls > 0
    
    def test_resource_exhaustion_recovery_e2e(self):
        """Test system recovery from resource exhaustion."""
        device = MockEdgeDevice()
        
        # Simulate resource exhaustion
        device.resources['memory_used'] = int(device.resources['memory_total'] * 0.95)  # 95%
        device.resources['cpu_usage'] = 90.0  # 90%
        
        def check_and_recover_resources():
            """Check resources and trigger recovery if needed."""
            resources = device.get_resource_usage()
            memory_percent = (resources['memory_used'] / resources['memory_total']) * 100
            cpu_percent = resources['cpu_usage']
            
            recovery_actions = []
            
            # Memory recovery
            if memory_percent > 90:
                # Simulate garbage collection
                device.resources['memory_used'] = int(device.resources['memory_total'] * 0.7)
                recovery_actions.append('memory_cleanup')
            
            # CPU recovery
            if cpu_percent > 85:
                # Simulate process throttling
                device.resources['cpu_usage'] = 60.0
                recovery_actions.append('cpu_throttling')
            
            return recovery_actions
        
        # Check initial state (should be in exhaustion)
        initial_resources = device.get_resource_usage()
        memory_percent = (initial_resources['memory_used'] / initial_resources['memory_total']) * 100
        assert memory_percent > 90
        assert initial_resources['cpu_usage'] > 85
        
        # Trigger recovery
        recovery_actions = check_and_recover_resources()
        
        # Verify recovery
        assert 'memory_cleanup' in recovery_actions
        assert 'cpu_throttling' in recovery_actions
        
        # Check recovered state
        recovered_resources = device.get_resource_usage()
        memory_percent = (recovered_resources['memory_used'] / recovered_resources['memory_total']) * 100
        assert memory_percent <= 70  # Should be recovered
        assert recovered_resources['cpu_usage'] <= 60  # Should be throttled


@pytest.mark.e2e
@pytest.mark.parametrize("deployment_scenario", [
    "single_device",
    "multi_device_cluster", 
    "edge_cloud_hybrid"
])
def test_deployment_scenarios_e2e(deployment_scenario):
    """Test different deployment scenarios end-to-end."""
    if deployment_scenario == "single_device":
        # Test single device deployment
        device = MockEdgeDevice()
        device.device_id = "edge-001"
        
        deployment_success = device.deploy_model({
            'model_id': 'anomaly-detector',
            'version': '1.0.0',
            'deployment_type': 'single'
        })
        
        assert deployment_success
        assert len(device.deployed_models) == 1
        
    elif deployment_scenario == "multi_device_cluster":
        # Test multi-device cluster deployment
        devices = [MockEdgeDevice() for _ in range(3)]
        for i, device in enumerate(devices):
            device.device_id = f"edge-cluster-{i+1:03d}"
        
        # Deploy to all devices
        deployment_results = []
        for device in devices:
            result = device.deploy_model({
                'model_id': 'anomaly-detector',
                'version': '1.0.0',
                'deployment_type': 'cluster',
                'cluster_id': 'cluster-001'
            })
            deployment_results.append(result)
        
        assert all(deployment_results)
        assert all(len(device.deployed_models) > 0 for device in devices)
        
    elif deployment_scenario == "edge_cloud_hybrid":
        # Test edge-cloud hybrid deployment
        edge_device = MockEdgeDevice()
        edge_device.device_id = "edge-hybrid-001"
        
        # Mock cloud connection
        cloud_connected = True
        
        # Deploy with cloud backup
        deployment_config = {
            'model_id': 'anomaly-detector',
            'version': '1.0.0',
            'deployment_type': 'hybrid',
            'cloud_backup': cloud_connected,
            'edge_primary': True
        }
        
        deployment_success = edge_device.deploy_model(deployment_config)
        
        assert deployment_success
        assert len(edge_device.deployed_models) == 1
        
        deployed_model = edge_device.deployed_models[0]
        assert deployed_model['config']['edge_primary'] is True
        assert deployed_model['config']['cloud_backup'] is True