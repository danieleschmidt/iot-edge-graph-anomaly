"""Mock services and utilities for testing IoT Edge Graph Anomaly Detection."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, Mock
import numpy as np
import torch
from contextlib import contextmanager


class MockOTLPExporter:
    """Mock OpenTelemetry OTLP exporter for testing."""
    
    def __init__(self):
        self.exported_metrics = []
        self.exported_traces = []
        self.is_connected = True
        self.export_calls = 0
    
    def export(self, metrics: List[Any]) -> None:
        """Mock export method."""
        self.export_calls += 1
        self.exported_metrics.extend(metrics)
        
        if not self.is_connected:
            raise ConnectionError("OTLP endpoint unreachable")
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Mock force flush method."""
        return self.is_connected
    
    def shutdown(self) -> None:
        """Mock shutdown method."""
        self.exported_metrics.clear()
        self.exported_traces.clear()


class MockPrometheusGateway:
    """Mock Prometheus push gateway for testing."""
    
    def __init__(self):
        self.pushed_metrics = {}
        self.push_calls = 0
        self.is_available = True
    
    def push(self, job: str, registry: Any, gateway: str) -> None:
        """Mock push method."""
        self.push_calls += 1
        
        if not self.is_available:
            raise ConnectionError("Prometheus gateway unavailable")
        
        # Store metrics for verification
        self.pushed_metrics[job] = {
            'timestamp': time.time(),
            'gateway': gateway,
            'registry': registry
        }
    
    def delete(self, job: str, gateway: str) -> None:
        """Mock delete method."""
        if job in self.pushed_metrics:
            del self.pushed_metrics[job]


class MockHealthChecker:
    """Mock health checker for testing."""
    
    def __init__(self):
        self.health_status = {
            'status': 'healthy',
            'memory_usage': 75.0,
            'cpu_usage': 20.0,
            'disk_usage': 45.0,
            'model_loaded': True,
            'last_inference': time.time(),
        }
        self.check_calls = 0
    
    def check_health(self) -> Dict[str, Any]:
        """Mock health check method."""
        self.check_calls += 1
        return self.health_status.copy()
    
    def set_unhealthy(self, reason: str = "test failure") -> None:
        """Set health status to unhealthy for testing."""
        self.health_status['status'] = 'unhealthy'
        self.health_status['reason'] = reason
    
    def set_healthy(self) -> None:
        """Reset health status to healthy."""
        self.health_status['status'] = 'healthy'
        if 'reason' in self.health_status:
            del self.health_status['reason']


class MockModelLoader:
    """Mock model loader for testing."""
    
    def __init__(self):
        self.loaded_models = {}
        self.load_calls = 0
        self.save_calls = 0
        self.loading_time = 0.1  # Simulated loading time
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Mock model loading method."""
        self.load_calls += 1
        time.sleep(self.loading_time)  # Simulate loading time
        
        # Return a simple mock model
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.eval = MagicMock()
        mock_model.forward = MagicMock(return_value=torch.randn(1, 10))
        
        self.loaded_models[model_path] = mock_model
        return mock_model
    
    def save_model(self, model: torch.nn.Module, model_path: str) -> None:
        """Mock model saving method."""
        self.save_calls += 1
        self.loaded_models[model_path] = model
    
    def model_exists(self, model_path: str) -> bool:
        """Check if model exists in mock storage."""
        return model_path in self.loaded_models


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data: Optional[np.ndarray] = None):
        self.data = data or np.random.randn(100, 10)
        self.batch_size = 32
        self.current_index = 0
        self.load_calls = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        
        end_index = min(self.current_index + self.batch_size, len(self.data))
        batch = self.data[self.current_index:end_index]
        self.current_index = end_index
        
        return torch.tensor(batch, dtype=torch.float32)
    
    def reset(self):
        """Reset iterator to beginning."""
        self.current_index = 0
    
    def load_dataset(self, dataset_path: str) -> np.ndarray:
        """Mock dataset loading method."""
        self.load_calls += 1
        return self.data


class MockAnomalyDetector:
    """Mock anomaly detector for testing."""
    
    def __init__(self):
        self.detection_calls = 0
        self.threshold = 0.5
        self.model_ready = True
        self.detection_history = []
    
    def detect_anomaly(self, data: np.ndarray) -> Dict[str, Any]:
        """Mock anomaly detection method."""
        self.detection_calls += 1
        
        if not self.model_ready:
            raise RuntimeError("Model not ready")
        
        # Simulate detection logic
        reconstruction_error = np.random.random()
        is_anomaly = reconstruction_error > self.threshold
        confidence = abs(reconstruction_error - self.threshold) / self.threshold
        
        result = {
            'is_anomaly': is_anomaly,
            'reconstruction_error': reconstruction_error,
            'confidence': confidence,
            'timestamp': time.time(),
        }
        
        self.detection_history.append(result)
        return result
    
    def set_threshold(self, threshold: float) -> None:
        """Set detection threshold."""
        self.threshold = threshold
    
    def set_model_ready(self, ready: bool) -> None:
        """Set model readiness status."""
        self.model_ready = ready


class MockEdgeDevice:
    """Mock edge device for testing deployment scenarios."""
    
    def __init__(self):
        self.device_id = "test-device-001"
        self.status = "online"
        self.resources = {
            'memory_total': 4096,  # MB
            'memory_used': 1024,
            'cpu_cores': 4,
            'cpu_usage': 25.0,
            'storage_total': 32768,  # MB
            'storage_used': 8192,
            'temperature': 45.0,  # Celsius
        }
        self.deployed_models = []
        self.metrics = {}
        self.command_history = []
    
    def execute_command(self, command: str) -> Dict[str, Any]:
        """Mock command execution."""
        self.command_history.append({
            'command': command,
            'timestamp': time.time(),
        })
        
        if command.startswith("docker"):
            return {'status': 'success', 'output': 'Container started'}
        elif command.startswith("systemctl"):
            return {'status': 'success', 'output': 'Service restarted'}
        else:
            return {'status': 'success', 'output': 'Command executed'}
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return self.resources.copy()
    
    def deploy_model(self, model_info: Dict[str, Any]) -> bool:
        """Mock model deployment."""
        self.deployed_models.append(model_info)
        return True
    
    def set_offline(self) -> None:
        """Simulate device going offline."""
        self.status = "offline"
    
    def set_online(self) -> None:
        """Simulate device coming online."""
        self.status = "online"


class MockNetworkService:
    """Mock network service for testing connectivity scenarios."""
    
    def __init__(self):
        self.is_connected = True
        self.latency = 50  # ms
        self.bandwidth = 100  # Mbps
        self.packet_loss = 0.0  # percentage
        self.request_count = 0
        self.failed_requests = 0
    
    def send_request(self, endpoint: str, data: Any) -> Dict[str, Any]:
        """Mock network request."""
        self.request_count += 1
        
        if not self.is_connected:
            self.failed_requests += 1
            raise ConnectionError("Network unavailable")
        
        # Simulate network delay
        time.sleep(self.latency / 1000.0)
        
        # Simulate packet loss
        if np.random.random() < self.packet_loss:
            self.failed_requests += 1
            raise TimeoutError("Request timeout")
        
        return {
            'status': 'success',
            'response_time': self.latency,
            'data_sent': len(str(data)) if data else 0,
        }
    
    def set_network_conditions(self, connected: bool = True, latency: int = 50, 
                             packet_loss: float = 0.0) -> None:
        """Set network conditions for testing."""
        self.is_connected = connected
        self.latency = latency
        self.packet_loss = packet_loss


@contextmanager
def mock_environment_variables(env_vars: Dict[str, str]):
    """Context manager for mocking environment variables."""
    import os
    original_values = {}
    
    # Store original values and set new ones
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value


class MockAsyncQueue:
    """Mock async queue for testing streaming scenarios."""
    
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self.items = []
        self.get_calls = 0
        self.put_calls = 0
    
    async def put(self, item: Any) -> None:
        """Mock put method."""
        self.put_calls += 1
        if self.maxsize > 0 and len(self.items) >= self.maxsize:
            raise asyncio.QueueFull()
        self.items.append(item)
    
    async def get(self) -> Any:
        """Mock get method."""
        self.get_calls += 1
        if not self.items:
            raise asyncio.QueueEmpty()
        return self.items.pop(0)
    
    def qsize(self) -> int:
        """Get queue size."""
        return len(self.items)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self.maxsize > 0 and len(self.items) >= self.maxsize


def create_mock_model_response(input_shape: tuple, output_shape: tuple) -> torch.Tensor:
    """Create a mock model response for testing.
    
    Args:
        input_shape: Shape of input tensor
        output_shape: Shape of output tensor
        
    Returns:
        torch.Tensor: Mock model output
    """
    torch.manual_seed(42)
    return torch.randn(output_shape)


def create_mock_training_metrics() -> Dict[str, List[float]]:
    """Create mock training metrics for testing.
    
    Returns:
        Dict[str, List[float]]: Mock training metrics
    """
    epochs = 10
    return {
        'train_loss': [1.0 - i * 0.08 for i in range(epochs)],
        'val_loss': [1.1 - i * 0.07 for i in range(epochs)],
        'train_accuracy': [0.6 + i * 0.04 for i in range(epochs)],
        'val_accuracy': [0.55 + i * 0.04 for i in range(epochs)],
    }