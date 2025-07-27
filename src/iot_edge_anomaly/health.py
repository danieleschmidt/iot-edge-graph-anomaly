"""Health check module for IoT Edge Anomaly Detection."""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Health check result."""
    status: HealthStatus
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None

class SystemHealthMonitor:
    """System health monitoring for edge deployment."""
    
    def __init__(self, memory_threshold_mb: float = 100, cpu_threshold_percent: float = 80):
        """Initialize health monitor.
        
        Args:
            memory_threshold_mb: Memory usage threshold in MB
            cpu_threshold_percent: CPU usage threshold percentage
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.start_time = time.time()
    
    def check_memory_usage(self) -> HealthCheck:
        """Check memory usage health."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.memory_threshold_mb:
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {memory_mb:.1f}MB",
                    timestamp=time.time(),
                    details={"memory_mb": memory_mb, "threshold_mb": self.memory_threshold_mb}
                )
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message=f"Memory usage normal: {memory_mb:.1f}MB",
                timestamp=time.time(),
                details={"memory_mb": memory_mb}
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > self.cpu_threshold_percent:
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    timestamp=time.time(),
                    details={"cpu_percent": cpu_percent, "threshold_percent": self.cpu_threshold_percent}
                )
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message=f"CPU usage normal: {cpu_percent:.1f}%",
                timestamp=time.time(),
                details={"cpu_percent": cpu_percent}
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_disk_usage(self) -> HealthCheck:
        """Check disk usage health."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > 90:
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message=f"High disk usage: {disk_percent:.1f}%",
                    timestamp=time.time(),
                    details={"disk_percent": disk_percent}
                )
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message=f"Disk usage normal: {disk_percent:.1f}%",
                timestamp=time.time(),
                details={"disk_percent": disk_percent}
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_torch_availability(self) -> HealthCheck:
        """Check PyTorch availability and CUDA status."""
        try:
            # Check PyTorch basic functionality
            test_tensor = torch.randn(2, 2)
            result = torch.mm(test_tensor, test_tensor)
            
            details = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available():
                details["cuda_version"] = torch.version.cuda
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message="PyTorch functioning normally",
                timestamp=time.time(),
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"PyTorch check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        checks = {
            "memory": self.check_memory_usage(),
            "cpu": self.check_cpu_usage(),
            "disk": self.check_disk_usage(),
            "torch": self.check_torch_availability()
        }
        
        # Determine overall status
        statuses = [check.status for check in checks.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": self.get_uptime(),
            "checks": {name: {
                "status": check.status.value,
                "message": check.message,
                "details": check.details or {}
            } for name, check in checks.items()}
        }

class ModelHealthMonitor:
    """Health monitoring for ML model operations."""
    
    def __init__(self):
        """Initialize model health monitor."""
        self.inference_times = []
        self.error_rates = []
        self.last_inference_time = None
    
    def record_inference(self, inference_time: float, error_occurred: bool = False):
        """Record inference metrics.
        
        Args:
            inference_time: Time taken for inference in seconds
            error_occurred: Whether an error occurred during inference
        """
        self.inference_times.append(inference_time)
        self.error_rates.append(1 if error_occurred else 0)
        self.last_inference_time = time.time()
        
        # Keep only last 100 records
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
            self.error_rates.pop(0)
    
    def check_inference_performance(self) -> HealthCheck:
        """Check inference performance health."""
        try:
            if not self.inference_times:
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message="No inference data available",
                    timestamp=time.time()
                )
            
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            max_inference_time = max(self.inference_times)
            error_rate = sum(self.error_rates) / len(self.error_rates)
            
            details = {
                "avg_inference_time_ms": avg_inference_time * 1000,
                "max_inference_time_ms": max_inference_time * 1000,
                "error_rate": error_rate,
                "sample_count": len(self.inference_times)
            }
            
            # Check thresholds
            if error_rate > 0.1:  # 10% error rate threshold
                return HealthCheck(
                    status=HealthStatus.UNHEALTHY,
                    message=f"High error rate: {error_rate:.2%}",
                    timestamp=time.time(),
                    details=details
                )
            
            if avg_inference_time > 0.1:  # 100ms threshold
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message=f"Slow inference: {avg_inference_time*1000:.1f}ms avg",
                    timestamp=time.time(),
                    details=details
                )
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message=f"Inference performance good: {avg_inference_time*1000:.1f}ms avg",
                timestamp=time.time(),
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"Inference performance check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_model_staleness(self) -> HealthCheck:
        """Check if model has been used recently."""
        try:
            if self.last_inference_time is None:
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message="Model never used",
                    timestamp=time.time()
                )
            
            staleness = time.time() - self.last_inference_time
            
            if staleness > 300:  # 5 minutes
                return HealthCheck(
                    status=HealthStatus.DEGRADED,
                    message=f"Model stale: {staleness:.0f}s since last inference",
                    timestamp=time.time(),
                    details={"staleness_seconds": staleness}
                )
            
            return HealthCheck(
                status=HealthStatus.HEALTHY,
                message=f"Model active: {staleness:.0f}s since last inference",
                timestamp=time.time(),
                details={"staleness_seconds": staleness}
            )
            
        except Exception as e:
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                message=f"Model staleness check failed: {str(e)}",
                timestamp=time.time()
            )