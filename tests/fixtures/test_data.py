"""Test data fixtures for IoT Edge Graph Anomaly Detection tests."""

import numpy as np
import pandas as pd
import pytest
import torch
from typing import Dict, List, Tuple, Any


@pytest.fixture
def sample_sensor_data() -> np.ndarray:
    """Generate sample sensor data for testing.
    
    Returns:
        np.ndarray: Sample sensor data with shape (100, 10) representing
                   100 time steps and 10 sensors.
    """
    np.random.seed(42)
    time_steps = 100
    num_sensors = 10
    
    # Generate normal operating data with some correlation between sensors
    base_signal = np.sin(np.linspace(0, 4 * np.pi, time_steps))
    data = np.zeros((time_steps, num_sensors))
    
    for i in range(num_sensors):
        # Each sensor has the base signal with some noise and phase shift
        phase_shift = i * 0.1
        noise = np.random.normal(0, 0.1, time_steps)
        data[:, i] = base_signal + np.sin(np.linspace(phase_shift, 4 * np.pi + phase_shift, time_steps)) * 0.3 + noise
    
    return data


@pytest.fixture
def anomalous_sensor_data() -> np.ndarray:
    """Generate sensor data with anomalies for testing.
    
    Returns:
        np.ndarray: Sensor data with injected anomalies.
    """
    np.random.seed(42)
    normal_data = sample_sensor_data()
    
    # Inject anomalies at specific time points
    anomaly_points = [25, 50, 75]
    for point in anomaly_points:
        # Sudden spike in multiple sensors
        normal_data[point:point+3, [2, 5, 8]] *= 3.0
        # Sudden drop in other sensors
        normal_data[point:point+2, [1, 4, 7]] *= -2.0
    
    return normal_data


@pytest.fixture
def swat_like_data() -> pd.DataFrame:
    """Generate SWaT-like dataset for testing.
    
    Returns:
        pd.DataFrame: SWaT-like dataset with timestamp and sensor columns.
    """
    np.random.seed(42)
    
    # Generate timestamps
    timestamps = pd.date_range('2023-01-01 00:00:00', periods=1000, freq='1S')
    
    # Generate sensor data similar to SWaT dataset structure
    sensors = {
        'FIT101': np.random.normal(2.5, 0.2, 1000),  # Flow sensor
        'LIT101': np.random.normal(800, 50, 1000),   # Level sensor
        'PIT101': np.random.normal(2.0, 0.1, 1000),  # Pressure sensor
        'AIT101': np.random.normal(20.5, 1.0, 1000), # Analyzer sensor
        'FIT201': np.random.normal(1.8, 0.15, 1000), # Flow sensor 2
        'LIT201': np.random.normal(650, 40, 1000),   # Level sensor 2
        'PIT201': np.random.normal(1.5, 0.08, 1000), # Pressure sensor 2
        'AIT201': np.random.normal(18.2, 0.8, 1000), # Analyzer sensor 2
        'MV101': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),  # Motor valve
        'P101': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),   # Pump
    }
    
    df = pd.DataFrame(sensors)
    df['timestamp'] = timestamps
    df['Normal/Attack'] = 'Normal'
    
    # Add some attack periods
    attack_periods = [(200, 250), (500, 520), (800, 830)]
    for start, end in attack_periods:
        df.loc[start:end, 'Normal/Attack'] = 'Attack'
        # Modify sensor values during attacks
        df.loc[start:end, 'FIT101'] *= 1.5
        df.loc[start:end, 'LIT101'] += 100
    
    return df


@pytest.fixture
def lstm_model_config() -> Dict[str, Any]:
    """LSTM model configuration for testing.
    
    Returns:
        Dict[str, Any]: Model configuration parameters.
    """
    return {
        'input_size': 10,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 10,
        'batch_size': 16,
        'learning_rate': 0.001,
    }


@pytest.fixture
def gnn_model_config() -> Dict[str, Any]:
    """GNN model configuration for testing.
    
    Returns:
        Dict[str, Any]: GNN model configuration parameters.
    """
    return {
        'input_dim': 10,
        'hidden_dim': 32,
        'output_dim': 16,
        'num_layers': 2,
        'dropout': 0.2,
        'aggregation': 'mean',
    }


@pytest.fixture
def sample_graph_data() -> Dict[str, torch.Tensor]:
    """Generate sample graph data for testing GNN components.
    
    Returns:
        Dict[str, torch.Tensor]: Graph data including node features and edge indices.
    """
    torch.manual_seed(42)
    
    num_nodes = 10
    num_features = 5
    
    # Node features
    node_features = torch.randn(num_nodes, num_features)
    
    # Edge indices (creating a simple ring topology)
    edge_indices = []
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        edge_indices.extend([[i, next_node], [next_node, i]])  # Bidirectional edges
    
    edge_index = torch.tensor(edge_indices).t().contiguous()
    
    return {
        'x': node_features,
        'edge_index': edge_index,
        'num_nodes': num_nodes,
    }


@pytest.fixture
def training_sequences() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training sequences for LSTM testing.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input sequences and targets.
    """
    torch.manual_seed(42)
    
    batch_size = 16
    sequence_length = 10
    num_features = 8
    
    # Generate input sequences
    inputs = torch.randn(batch_size, sequence_length, num_features)
    
    # For autoencoder, targets are the same as inputs
    targets = inputs.clone()
    
    return inputs, targets


@pytest.fixture
def mock_device_config() -> Dict[str, Any]:
    """Mock edge device configuration for testing.
    
    Returns:
        Dict[str, Any]: Device configuration parameters.
    """
    return {
        'device_id': 'test-device-001',
        'device_type': 'raspberry-pi-4',
        'location': 'test-facility',
        'memory_limit': 100,  # MB
        'cpu_limit': 0.25,    # 25%
        'storage_limit': 1024,  # MB
        'network_timeout': 30,
        'health_check_interval': 30,
        'metrics_export_interval': 60,
    }


@pytest.fixture
def mock_monitoring_config() -> Dict[str, Any]:
    """Mock monitoring configuration for testing.
    
    Returns:
        Dict[str, Any]: Monitoring configuration parameters.
    """
    return {
        'otlp_endpoint': 'http://localhost:4317',
        'service_name': 'iot-edge-anomaly-test',
        'service_version': '0.1.0-test',
        'metrics_enabled': True,
        'metrics_interval': 10,
        'health_checks_enabled': True,
        'logging_level': 'DEBUG',
    }


@pytest.fixture
def sample_metrics_data() -> Dict[str, float]:
    """Sample metrics data for testing monitoring components.
    
    Returns:
        Dict[str, float]: Sample metrics values.
    """
    return {
        'inference_latency_ms': 5.2,
        'memory_usage_mb': 75.4,
        'cpu_usage_percent': 18.6,
        'anomaly_detection_rate': 0.02,
        'model_accuracy': 0.94,
        'reconstruction_error_mean': 0.15,
        'reconstruction_error_std': 0.08,
        'throughput_samples_per_second': 95.3,
    }


@pytest.fixture
def test_model_artifacts() -> Dict[str, str]:
    """Test model artifacts paths and metadata.
    
    Returns:
        Dict[str, str]: Model artifact information.
    """
    return {
        'model_path': '/tmp/test_model.pth',
        'config_path': '/tmp/test_config.json',
        'scaler_path': '/tmp/test_scaler.pkl',
        'metrics_path': '/tmp/test_metrics.json',
        'model_version': '0.1.0-test',
        'training_timestamp': '2023-01-01T00:00:00Z',
    }


class MockSensorDataGenerator:
    """Mock sensor data generator for testing streaming scenarios."""
    
    def __init__(self, num_sensors: int = 10, anomaly_rate: float = 0.05):
        self.num_sensors = num_sensors
        self.anomaly_rate = anomaly_rate
        self.time_step = 0
        np.random.seed(42)
    
    def generate_sample(self) -> np.ndarray:
        """Generate a single sensor data sample.
        
        Returns:
            np.ndarray: Single time step of sensor data.
        """
        # Base signal with time progression
        base_signal = np.sin(self.time_step * 0.1)
        sample = np.zeros(self.num_sensors)
        
        for i in range(self.num_sensors):
            # Each sensor has slightly different characteristics
            phase_shift = i * 0.2
            amplitude = 1.0 + i * 0.1
            noise = np.random.normal(0, 0.05)
            
            sample[i] = amplitude * np.sin(self.time_step * 0.1 + phase_shift) + noise
            
            # Inject anomalies based on anomaly rate
            if np.random.random() < self.anomaly_rate:
                sample[i] *= 3.0  # Anomalous spike
        
        self.time_step += 1
        return sample


@pytest.fixture
def mock_sensor_generator() -> MockSensorDataGenerator:
    """Mock sensor data generator fixture.
    
    Returns:
        MockSensorDataGenerator: Generator instance for streaming data.
    """
    return MockSensorDataGenerator()