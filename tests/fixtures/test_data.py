"""Test data fixtures and generators for IoT Edge Anomaly Detection tests."""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any
import pytest


def generate_synthetic_sensor_data(
    num_sensors: int = 10,
    sequence_length: int = 100,
    anomaly_rate: float = 0.05,
    noise_level: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for testing.
    
    Args:
        num_sensors: Number of sensors to simulate
        sequence_length: Length of time series
        anomaly_rate: Fraction of data points that are anomalous
        noise_level: Gaussian noise standard deviation
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sensor readings and anomaly labels
    """
    np.random.seed(random_seed)
    
    # Generate timestamps
    timestamps = pd.date_range(
        start='2025-01-01', 
        periods=sequence_length, 
        freq='1min'
    )
    
    # Generate normal sensor patterns
    data = {}
    for sensor_id in range(num_sensors):
        # Base sinusoidal pattern with sensor-specific frequency
        freq = 0.1 + sensor_id * 0.02
        base_pattern = np.sin(2 * np.pi * freq * np.arange(sequence_length))
        
        # Add noise
        noise = np.random.normal(0, noise_level, sequence_length)
        sensor_data = base_pattern + noise
        
        # Inject anomalies
        num_anomalies = int(sequence_length * anomaly_rate)
        anomaly_indices = np.random.choice(
            sequence_length, 
            num_anomalies, 
            replace=False
        )
        
        for idx in anomaly_indices:
            # Create spike anomalies
            sensor_data[idx] += np.random.uniform(2, 5) * np.sign(sensor_data[idx])
        
        data[f'sensor_{sensor_id}'] = sensor_data
    
    # Create DataFrame
    df = pd.DataFrame(data, index=timestamps)
    
    # Add anomaly labels
    df['is_anomaly'] = False
    for sensor_id in range(num_sensors):
        anomaly_indices = np.where(
            np.abs(df[f'sensor_{sensor_id}']) > 2.0
        )[0]
        df.iloc[anomaly_indices, df.columns.get_loc('is_anomaly')] = True
    
    return df


def generate_lstm_input_batch(
    batch_size: int = 32,
    sequence_length: int = 10,
    num_features: int = 10,
    random_seed: int = 42
) -> torch.Tensor:
    """
    Generate batch of sequences for LSTM input testing.
    
    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        num_features: Number of features per timestep
        random_seed: Random seed for reproducibility
        
    Returns:
        Tensor of shape (batch_size, sequence_length, num_features)
    """
    torch.manual_seed(random_seed)
    return torch.randn(batch_size, sequence_length, num_features)


def create_mock_model_config() -> Dict[str, Any]:
    """Create mock model configuration for testing."""
    return {
        'model': {
            'input_size': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 10
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10
        },
        'anomaly_detection': {
            'threshold': 0.5,
            'window_size': 60,
            'cooldown_period': 300
        },
        'deployment': {
            'memory_limit': '100M',
            'cpu_limit': 0.25,
            'health_check_interval': 30
        }
    }


def create_sample_swat_data() -> pd.DataFrame:
    """Create sample SWaT-like dataset for testing."""
    np.random.seed(42)
    
    # SWaT has 51 sensors
    num_sensors = 51
    sequence_length = 1000
    
    # Generate sensor names like in SWaT dataset
    sensor_names = []
    for i in range(1, 7):  # 6 stages
        for j in range(1, 9):  # ~8 sensors per stage
            if len(sensor_names) < num_sensors:
                sensor_names.append(f'P{i}_{j:03d}')
    
    # Fill remaining sensors
    while len(sensor_names) < num_sensors:
        sensor_names.append(f'AIT_{len(sensor_names):03d}')
    
    # Generate realistic industrial sensor data
    data = {}
    timestamps = pd.date_range(
        start='2025-01-01', 
        periods=sequence_length, 
        freq='1S'
    )
    
    for sensor in sensor_names:
        if sensor.startswith('P'):  # Pressure sensors
            base_value = np.random.uniform(1.0, 5.0)
            noise = np.random.normal(0, 0.05, sequence_length)
            data[sensor] = base_value + noise
        elif sensor.startswith('AIT'):  # Temperature sensors
            base_value = np.random.uniform(20.0, 80.0)
            daily_cycle = 5 * np.sin(2 * np.pi * np.arange(sequence_length) / 86400)
            noise = np.random.normal(0, 2.0, sequence_length)
            data[sensor] = base_value + daily_cycle + noise
        else:  # Flow sensors
            base_value = np.random.uniform(0.1, 2.0)
            noise = np.random.normal(0, 0.02, sequence_length)
            data[sensor] = np.maximum(0, base_value + noise)
    
    # Add normal/attack label (SWaT format)
    data['Normal/Attack'] = ['Normal'] * (sequence_length - 100) + ['Attack'] * 100
    
    return pd.DataFrame(data, index=timestamps)


@pytest.fixture
def sample_sensor_data():
    """Pytest fixture for sample sensor data."""
    return generate_synthetic_sensor_data()


@pytest.fixture
def lstm_input_batch():
    """Pytest fixture for LSTM input batch."""
    return generate_lstm_input_batch()


@pytest.fixture
def mock_config():
    """Pytest fixture for mock configuration."""
    return create_mock_model_config()


@pytest.fixture
def sample_swat_data():
    """Pytest fixture for sample SWaT data."""
    return create_sample_swat_data()


@pytest.fixture
def anomaly_test_cases():
    """Pytest fixture for anomaly detection test cases."""
    return [
        {
            'input': torch.tensor([0.1, 0.2, 0.15]),
            'expected_anomaly': False,
            'description': 'Normal reconstruction error'
        },
        {
            'input': torch.tensor([0.8, 0.9, 0.85]),
            'expected_anomaly': True,
            'description': 'High reconstruction error'
        },
        {
            'input': torch.tensor([0.49, 0.51, 0.50]),
            'expected_anomaly': [False, True, False],  # Threshold = 0.5
            'description': 'Boundary cases around threshold'
        }
    ]


class MockLSTMAutoencoder:
    """Mock LSTM Autoencoder for testing without real model."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.training = False
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Mock forward pass that returns input with slight modification."""
        # Simulate reconstruction with small error
        noise = torch.randn_like(x) * 0.1
        return x + noise
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        return self
    
    def parameters(self):
        """Mock parameters for optimizer."""
        return [torch.randn(10, requires_grad=True)]


class MockDataLoader:
    """Mock data loader for testing without real data files."""
    
    def __init__(self, data: pd.DataFrame = None):
        self.data = data or generate_synthetic_sensor_data()
    
    def __iter__(self):
        """Iterate over batches."""
        batch_size = 32
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data.iloc[i:i+batch_size]
            yield self._to_tensor(batch_data)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.data) // 32
    
    def _to_tensor(self, data: pd.DataFrame) -> torch.Tensor:
        """Convert DataFrame to tensor."""
        numeric_data = data.select_dtypes(include=[np.number])
        return torch.tensor(numeric_data.values, dtype=torch.float32)


# Export commonly used test utilities
__all__ = [
    'generate_synthetic_sensor_data',
    'generate_lstm_input_batch', 
    'create_mock_model_config',
    'create_sample_swat_data',
    'MockLSTMAutoencoder',
    'MockDataLoader',
    # Fixtures
    'sample_sensor_data',
    'lstm_input_batch',
    'mock_config',
    'sample_swat_data',
    'anomaly_test_cases'
]