"""Test configuration and fixtures for IoT Edge Graph Anomaly Detection."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    timestamps = np.arange(100)
    values = np.sin(timestamps * 0.1) + np.random.normal(0, 0.1, 100)
    return timestamps, values.astype(np.float32)


@pytest.fixture
def sample_graph_structure():
    """Generate sample graph structure for testing."""
    # Simple 4-node graph with edge connections
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                               [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
    return edge_index


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        'input_size': 10,
        'hidden_size': 64,
        'num_layers': 2,
        'sequence_length': 10,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32
    }


@pytest.fixture
def mock_swat_dataset():
    """Mock SWaT dataset for testing."""
    np.random.seed(42)
    data = {
        'normal': np.random.randn(1000, 51).astype(np.float32),
        'attack': np.random.randn(200, 51).astype(np.float32),
        'labels': np.concatenate([
            np.zeros(1000, dtype=np.int32),
            np.ones(200, dtype=np.int32)
        ])
    }
    return data


@pytest.fixture
def mock_metrics_exporter():
    """Mock metrics exporter for testing."""
    with patch('iot_edge_anomaly.monitoring.metrics_exporter.MetricsExporter') as mock:
        exporter = Mock()
        mock.return_value = exporter
        yield exporter


@pytest.fixture
def mock_otlp_endpoint():
    """Mock OTLP endpoint for testing."""
    return "http://localhost:4317"


@pytest.fixture
def device():
    """Get appropriate device for testing (CPU or CUDA)."""
    return torch.device('cpu')  # Always use CPU for tests


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch('logging.getLogger') as mock_get_logger:
        logger = Mock()
        mock_get_logger.return_value = logger
        yield logger


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file for testing."""
    config_content = """
model:
  input_size: 10
  hidden_size: 64
  num_layers: 2
  sequence_length: 10
  dropout_rate: 0.2

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

monitoring:
  otlp_endpoint: "http://localhost:4317"
  metrics_interval: 30

edge:
  memory_limit: "100M"
  cpu_limit: 0.25
"""
    config_file = temp_dir / "config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        'LOG_LEVEL': 'INFO',
        'MODEL_PATH': './models/test_model.pth',
        'OTLP_ENDPOINT': 'http://localhost:4317',
        'ANOMALY_THRESHOLD': '0.5',
        'DEVICE_ID': 'test-device-001'
    }
    
    with patch.dict('os.environ', env_vars, clear=False):
        yield env_vars


class MockTorchDataset:
    """Mock PyTorch dataset for testing."""
    
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


@pytest.fixture
def mock_torch_dataset():
    """Mock PyTorch dataset fixture."""
    return MockTorchDataset


# Performance testing fixtures
@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return PerformanceBenchmark()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )