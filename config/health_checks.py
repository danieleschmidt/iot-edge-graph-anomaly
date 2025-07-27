"""Health check endpoints for monitoring and observability."""

import time
import psutil
import torch
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class HealthStatus:
    """Health check status representation."""
    healthy: bool
    timestamp: float
    details: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking for the IoT edge anomaly detector."""

    def __init__(self, model=None, metrics_exporter=None):
        self.model = model
        self.metrics_exporter = metrics_exporter
        self.startup_time = time.time()

    def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        checks = {
            "model": self._check_model(),
            "memory": self._check_memory(),
            "cpu": self._check_cpu(),
            "metrics": self._check_metrics(),
            "dependencies": self._check_dependencies()
        }

        overall_healthy = all(check["healthy"] for check in checks.values())

        return HealthStatus(
            healthy=overall_healthy,
            timestamp=time.time(),
            details=checks
        )

    def check_readiness(self) -> HealthStatus:
        """Check if service is ready to handle requests."""
        checks = {
            "model_loaded": self._check_model_loaded(),
            "startup_complete": self._check_startup_complete()
        }

        ready = all(check["healthy"] for check in checks.values())

        return HealthStatus(
            healthy=ready,
            timestamp=time.time(),
            details=checks
        )

    def check_liveness(self) -> HealthStatus:
        """Basic liveness check."""
        return HealthStatus(
            healthy=True,
            timestamp=time.time(),
            details={"status": "alive", "uptime": time.time() - self.startup_time}
        )

    def _check_model(self) -> Dict[str, Any]:
        """Check model health."""
        if self.model is None:
            return {"healthy": False, "error": "Model not initialized"}

        try:
            # Quick inference test
            test_input = torch.randn(1, 10, 50)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

            self.model.eval()
            with torch.no_grad():
                start_time = time.time()
                _ = self.model(test_input, edge_index)
                inference_time = time.time() - start_time

            return {
                "healthy": True,
                "inference_time_ms": inference_time * 1000,
                "model_mode": "eval" if not self.model.training else "train"
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Edge device constraint: <100MB
            memory_mb = memory_info.rss / 1024 / 1024
            healthy = memory_mb < 100

            return {
                "healthy": healthy,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent,
                "threshold_mb": 100
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            # Edge device constraint: <25%
            healthy = cpu_percent < 25

            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "threshold_percent": 25
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_metrics(self) -> Dict[str, Any]:
        """Check metrics exporter health."""
        if self.metrics_exporter is None:
            return {"healthy": False, "error": "Metrics exporter not initialized"}

        try:
            # Test metrics export
            self.metrics_exporter.increment_inference_counter()
            return {"healthy": True, "exporter": "operational"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        try:
            import torch
            import numpy as np
            import pandas as pd

            return {
                "healthy": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available()
            }
        except ImportError as e:
            return {"healthy": False, "error": f"Missing dependency: {e}"}

    def _check_model_loaded(self) -> Dict[str, Any]:
        """Check if model is properly loaded."""
        if self.model is None:
            return {"healthy": False, "error": "Model not loaded"}

        return {"healthy": True, "model_state": "loaded"}

    def _check_startup_complete(self) -> Dict[str, Any]:
        """Check if startup phase is complete."""
        uptime = time.time() - self.startup_time
        # Consider ready after 30 seconds
        ready = uptime > 30

        return {
            "healthy": ready,
            "uptime_seconds": uptime,
            "ready_threshold": 30
        }