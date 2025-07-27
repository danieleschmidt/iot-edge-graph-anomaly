"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_sensor_data() -> pd.DataFrame:
    """Generate sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_sensors = 10
    
    # Generate time series data with some anomalies
    data = {}
    for i in range(n_sensors):
        # Normal pattern with some noise
        normal_data = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples)
        
        # Inject some anomalies
        anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
        normal_data[anomaly_indices] += np.random.normal(0, 2, 20)
        
        data[f"sensor_{i}"] = normal_data
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_graph_topology() -> Dict[str, Any]:
    """Generate sample graph topology for testing."""
    return {
        "nodes": list(range(10)),
        "edges": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # Ring structure
            [5, 6], [6, 7], [7, 8], [8, 9], [9, 5],  # Another ring
            [0, 5], [2, 7]  # Connections between rings
        ],
        "node_features": {
            "sensor_type": ["temperature", "humidity", "pressure", "flow", "vibration"] * 2,
            "location": ["zone_a"] * 5 + ["zone_b"] * 5
        }
    }


@pytest.fixture
def mock_device() -> str:
    """Return device type for testing (CPU)."""
    return "cpu"


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Generate sample configuration for testing."""
    return {
        "model": {
            "lstm_hidden_size": 64,
            "lstm_num_layers": 2,
            "gnn_hidden_size": 32,
            "dropout": 0.1,
            "learning_rate": 0.001
        },
        "data": {
            "window_size": 50,
            "batch_size": 32,
            "normalize": True
        },
        "training": {
            "epochs": 10,
            "patience": 5,
            "threshold": 0.5
        }
    }


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock model for testing."""
    model = Mock()
    model.eval.return_value = None
    model.train.return_value = None
    model.forward.return_value = torch.randn(32, 10, 50)  # Mock output
    model.state_dict.return_value = {"dummy": torch.tensor([1.0])}
    return model


@pytest.fixture
def mock_metrics_exporter() -> Mock:
    """Create a mock metrics exporter for testing."""
    exporter = Mock()
    exporter.export_metrics.return_value = None
    exporter.increment_counter.return_value = None
    exporter.record_histogram.return_value = None
    return exporter


@pytest.fixture
def sample_tensor_data() -> torch.Tensor:
    """Generate sample tensor data for testing."""
    torch.manual_seed(42)
    return torch.randn(32, 10, 50)  # (batch_size, num_sensors, window_size)


@pytest.fixture
def sample_edge_index() -> torch.Tensor:
    """Generate sample edge index for graph testing."""
    return torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    ], dtype=torch.long)


@pytest.fixture(autouse=True)
def clean_environment() -> Generator[None, None, None]:
    """Clean environment variables before and after tests."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Clear test-related environment variables
    test_vars = [
        "MODEL_PATH", "DEVICE", "BATCH_SIZE", "WINDOW_SIZE",
        "THRESHOLD", "LOG_LEVEL", "METRICS_ENABLED"
    ]
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def disable_gpu() -> Generator[None, None, None]:
    """Disable GPU usage for testing."""
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    yield
    
    if original_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock external services for testing."""
    # Mock OpenTelemetry
    mock_tracer = Mock()
    mock_meter = Mock()
    monkeypatch.setattr("opentelemetry.trace.get_tracer", lambda _: mock_tracer)
    monkeypatch.setattr("opentelemetry.metrics.get_meter", lambda _: mock_meter)
    
    # Mock Prometheus client
    mock_registry = Mock()
    monkeypatch.setattr("prometheus_client.CollectorRegistry", lambda: mock_registry)