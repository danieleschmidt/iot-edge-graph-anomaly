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


# Docker and container testing fixtures
@pytest.fixture
def docker_client():
    """Docker client fixture for container testing."""
    import docker
    try:
        client = docker.from_env()
        yield client
    except Exception:
        pytest.skip("Docker not available")


@pytest.fixture
def docker_compose_file(temp_dir):
    """Create a test docker-compose file."""
    compose_content = """
version: '3.8'
services:
  test-app:
    build: .
    environment:
      - LOG_LEVEL=INFO
    ports:
      - "8080:8080"
"""
    compose_file = temp_dir / "docker-compose.test.yml"
    compose_file.write_text(compose_content)
    return compose_file


# ML Model testing fixtures
@pytest.fixture
def mock_trained_model():
    """Mock trained LSTM autoencoder model."""
    from unittest.mock import Mock
    
    model = Mock()
    model.eval.return_value = None
    model.forward.return_value = torch.randn(32, 10, 64)  # batch, seq, hidden
    model.state_dict.return_value = {}
    model.load_state_dict.return_value = None
    return model


@pytest.fixture
def sample_model_weights(temp_dir):
    """Create sample model weights file."""
    weights_file = temp_dir / "model_weights.pth"
    # Create dummy weights
    dummy_weights = {
        'encoder.weight': torch.randn(64, 10),
        'decoder.weight': torch.randn(10, 64),
        'encoder.bias': torch.randn(64),
        'decoder.bias': torch.randn(10)
    }
    torch.save(dummy_weights, weights_file)
    return weights_file


# Data loading and preprocessing fixtures
@pytest.fixture
def sample_swat_data():
    """Generate realistic SWaT-like industrial IoT data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 51
    
    # Generate realistic industrial sensor data
    base_values = np.random.uniform(0, 100, n_features)
    noise = np.random.normal(0, 1, (n_samples, n_features))
    trend = np.linspace(0, 1, n_samples).reshape(-1, 1)
    seasonal = np.sin(np.linspace(0, 4*np.pi, n_samples)).reshape(-1, 1)
    
    data = base_values + noise + trend * 5 + seasonal * 2
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    data[anomaly_indices] += np.random.normal(0, 10, (50, n_features))
    
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    return {
        'data': data.astype(np.float32),
        'labels': labels.astype(np.int32),
        'timestamps': np.arange(n_samples),
        'feature_names': [f'sensor_{i:02d}' for i in range(n_features)]
    }


@pytest.fixture
def mock_data_loader():
    """Mock data loader for ML training/testing."""
    from torch.utils.data import DataLoader
    from unittest.mock import Mock
    
    # Create mock data
    data = torch.randn(100, 10, 51)  # batch, seq, features
    labels = torch.randint(0, 2, (100,))
    
    mock_loader = Mock(spec=DataLoader)
    mock_loader.__iter__ = lambda x: iter([(data[i:i+1], labels[i:i+1]) for i in range(100)])
    mock_loader.__len__ = lambda x: 100
    
    return mock_loader


# Monitoring and observability fixtures
@pytest.fixture
def mock_prometheus_client():
    """Mock Prometheus client for metrics testing."""
    with patch('prometheus_client.Counter') as mock_counter, \
         patch('prometheus_client.Histogram') as mock_histogram, \
         patch('prometheus_client.Gauge') as mock_gauge:
        yield {
            'counter': mock_counter,
            'histogram': mock_histogram,
            'gauge': mock_gauge
        }


@pytest.fixture
def mock_otlp_exporter():
    """Mock OpenTelemetry OTLP exporter."""
    with patch('opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter') as mock_exporter:
        exporter = Mock()
        mock_exporter.return_value = exporter
        yield exporter


@pytest.fixture
def health_check_client():
    """HTTP client for health check testing."""
    import requests
    from unittest.mock import Mock
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'status': 'healthy',
        'timestamp': '2025-01-27T00:00:00Z',
        'version': '0.1.0',
        'checks': {
            'database': 'ok',
            'model': 'loaded',
            'memory': 'ok'
        }
    }
    
    with patch('requests.get', return_value=mock_response) as mock_get:
        yield mock_get


# Security testing fixtures
@pytest.fixture
def security_test_payloads():
    """Common security test payloads for input validation testing."""
    return {
        'sql_injection': [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --"
        ],
        'xss': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'><script>alert('xss')</script>"
        ],
        'command_injection': [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`"
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
    }


# Performance testing fixtures
@pytest.fixture
def memory_profiler():
    """Memory profiling fixture."""
    import psutil
    import os
    
    class MemoryProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.peak_memory = None
        
        def start(self):
            self.start_memory = self.process.memory_info().rss
            self.peak_memory = self.start_memory
        
        def update_peak(self):
            current_memory = self.process.memory_info().rss
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        
        @property
        def memory_usage_mb(self):
            if self.start_memory:
                return (self.peak_memory - self.start_memory) / 1024 / 1024
            return None
    
    return MemoryProfiler()


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


# Test data generators
@pytest.fixture
def anomaly_data_generator():
    """Generator for various types of anomaly data."""
    
    def generate_anomaly_data(anomaly_type='point', n_samples=1000, n_features=10):
        np.random.seed(42)
        
        # Generate base normal data
        normal_data = np.random.randn(n_samples, n_features)
        labels = np.zeros(n_samples)
        
        if anomaly_type == 'point':
            # Point anomalies
            anomaly_indices = np.random.choice(n_samples, size=n_samples//20, replace=False)
            normal_data[anomaly_indices] += np.random.uniform(-5, 5, (len(anomaly_indices), n_features))
            labels[anomaly_indices] = 1
            
        elif anomaly_type == 'collective':
            # Collective anomalies
            start_idx = n_samples // 2
            end_idx = start_idx + 50
            normal_data[start_idx:end_idx] += np.random.uniform(3, 5, (end_idx - start_idx, n_features))
            labels[start_idx:end_idx] = 1
            
        elif anomaly_type == 'contextual':
            # Contextual anomalies based on time
            for i in range(n_samples):
                if i % 100 < 10:  # Expected low values every 100 samples
                    if np.random.random() < 0.1:  # 10% chance of anomaly
                        normal_data[i] += np.random.uniform(2, 4, n_features)
                        labels[i] = 1
        
        return {
            'data': normal_data.astype(np.float32),
            'labels': labels.astype(np.int32),
            'anomaly_type': anomaly_type
        }
    
    return generate_anomaly_data


# Network and API testing fixtures
@pytest.fixture
def mock_api_server():
    """Mock API server for integration testing."""
    from unittest.mock import Mock
    import json
    
    class MockAPIServer:
        def __init__(self):
            self.routes = {}
        
        def add_route(self, method, path, response_data, status_code=200):
            self.routes[(method.upper(), path)] = (response_data, status_code)
        
        def request(self, method, path, **kwargs):
            response = Mock()
            key = (method.upper(), path)
            if key in self.routes:
                response_data, status_code = self.routes[key]
                response.status_code = status_code
                response.json.return_value = response_data
                response.text = json.dumps(response_data)
            else:
                response.status_code = 404
                response.json.return_value = {'error': 'Not Found'}
                response.text = '{"error": "Not Found"}'
            return response
    
    server = MockAPIServer()
    
    # Add default health check endpoint
    server.add_route('GET', '/health', {
        'status': 'healthy',
        'timestamp': '2025-01-27T00:00:00Z'
    })
    
    return server


# Configuration and environment fixtures
@pytest.fixture
def test_config():
    """Comprehensive test configuration."""
    return {
        'model': {
            'type': 'lstm_autoencoder',
            'input_size': 51,
            'hidden_size': 64,
            'num_layers': 2,
            'sequence_length': 10,
            'dropout_rate': 0.2,
            'bidirectional': False
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'optimizer': 'adam',
            'loss_function': 'mse'
        },
        'anomaly_detection': {
            'threshold_method': 'statistical',
            'threshold_percentile': 95,
            'contamination_rate': 0.1,
            'window_size': 100
        },
        'monitoring': {
            'otlp_endpoint': 'http://localhost:4317',
            'metrics_interval': 30,
            'log_level': 'INFO',
            'enable_prometheus': True
        },
        'edge': {
            'device_id': 'test-device-001',
            'memory_limit': '100M',
            'cpu_limit': 0.25,
            'inference_timeout': 5.0
        }
    }


# Cleanup and teardown fixtures
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Clean up any temporary files, processes, etc.
    import glob
    import os
    
    # Remove temporary model files
    for pattern in ['*.pth', '*.onnx', '*.pkl', 'test_*.json', 'test_*.yaml']:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    markers = [
        "unit: Unit tests - fast tests that don't require external dependencies",
        "integration: Integration tests - tests that involve multiple components", 
        "e2e: End-to-end tests - tests that exercise the full system",
        "slow: Slow running tests that may take several minutes",
        "gpu: Tests that require GPU acceleration",
        "network: Tests that require network access", 
        "docker: Tests that require Docker runtime",
        "security: Security-focused tests",
        "performance: Performance and benchmark tests",
        "smoke: Smoke tests for basic functionality",
        "regression: Regression tests for known issues",
        "flaky: Tests that may be flaky and require retry",
        "experimental: Tests for experimental features",
        "requires_data: Tests that require external data files",
        "ml_model: Tests specific to machine learning models"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)