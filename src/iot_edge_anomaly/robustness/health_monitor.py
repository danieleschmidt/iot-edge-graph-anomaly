"""
System Health Monitoring for IoT Edge Anomaly Detection.

Provides comprehensive health monitoring capabilities including:
- System resource monitoring (CPU, memory, disk)
- Model performance tracking
- Data quality monitoring
- Network connectivity checks
- Automated health scoring and alerting
"""

import psutil
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import torch
import numpy as np

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemHealth:
    """System health information."""
    status: HealthStatus
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float]
    model_accuracy: Optional[float]
    data_quality_score: float
    network_latency: Optional[float]
    error_rate: float
    uptime: float
    last_updated: float
    alerts: List[str]
    
    def overall_score(self) -> float:
        """Compute overall health score (0-100)."""
        scores = []
        
        # Resource utilization (inverted - lower is better)
        scores.append(max(0, 100 - self.cpu_usage))
        scores.append(max(0, 100 - self.memory_usage))
        scores.append(max(0, 100 - self.disk_usage))
        
        if self.gpu_usage is not None:
            scores.append(max(0, 100 - self.gpu_usage))
        
        # Performance metrics
        if self.model_accuracy is not None:
            scores.append(self.model_accuracy * 100)
        
        scores.append(self.data_quality_score * 100)
        
        # Error rate (inverted)
        scores.append(max(0, 100 - self.error_rate * 100))
        
        # Network (inverted latency)
        if self.network_latency is not None:
            network_score = max(0, 100 - min(100, self.network_latency * 10))
            scores.append(network_score)
        
        return np.mean(scores)


class HealthMonitor:
    """
    Comprehensive system health monitoring.
    
    Monitors various aspects of system health including:
    - System resources (CPU, memory, disk, GPU)
    - Model performance metrics
    - Data quality and integrity
    - Network connectivity
    - Error rates and exceptions
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        history_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Health tracking
        self.current_health = None
        self.health_history = []
        self.alerts = []
        
        # Performance tracking
        self.model_predictions = []
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Data quality tracking
        self.data_samples = []
        self.data_quality_scores = []
        
        logger.info("Health monitor initialized")
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default alert thresholds."""
        return {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'gpu_usage': 85.0,
            'error_rate': 0.05,
            'model_accuracy': 0.8,
            'data_quality': 0.7,
            'network_latency': 1000.0  # ms
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_health(self):
        """Perform comprehensive health check."""
        # System resources
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_usage = memory.percent
        disk_usage = disk.percent
        
        # GPU usage
        gpu_usage = self._get_gpu_usage()
        
        # Model performance
        model_accuracy = self._compute_model_accuracy()
        
        # Data quality
        data_quality_score = self._compute_data_quality()
        
        # Network latency
        network_latency = self._check_network_latency()
        
        # Error rate
        error_rate = self._compute_error_rate()
        
        # Uptime
        uptime = time.time() - self.start_time
        
        # Generate alerts
        alerts = self._generate_alerts(
            cpu_usage, memory_usage, disk_usage, gpu_usage,
            model_accuracy, data_quality_score, network_latency, error_rate
        )
        
        # Determine overall status
        status = self._determine_status(alerts)
        
        # Create health object
        health = SystemHealth(
            status=status,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_usage=gpu_usage,
            model_accuracy=model_accuracy,
            data_quality_score=data_quality_score,
            network_latency=network_latency,
            error_rate=error_rate,
            uptime=uptime,
            last_updated=time.time(),
            alerts=alerts
        )
        
        # Update tracking
        self.current_health = health
        self.health_history.append(health)
        
        # Maintain history size
        if len(self.health_history) > self.history_size:
            self.health_history = self.health_history[-self.history_size:]
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"Health Alert: {alert}")
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                return float(gpu_usage)
        except Exception as e:
            logger.debug(f"Could not get GPU usage: {e}")
        return None
    
    def _compute_model_accuracy(self) -> Optional[float]:
        """Compute recent model accuracy."""
        if not self.model_predictions:
            return None
        
        # Use recent predictions (last 50)
        recent_predictions = self.model_predictions[-50:]
        if len(recent_predictions) < 5:
            return None
        
        # Compute accuracy for binary classification
        correct = sum(1 for pred, actual in recent_predictions if (pred > 0.5) == (actual > 0.5))
        accuracy = correct / len(recent_predictions)
        
        return accuracy
    
    def _compute_data_quality(self) -> float:
        """Compute data quality score."""
        if not self.data_quality_scores:
            return 1.0  # Default to good quality
        
        # Average recent scores
        recent_scores = self.data_quality_scores[-20:]
        return np.mean(recent_scores)
    
    def _check_network_latency(self) -> Optional[float]:
        """Check network latency (placeholder)."""
        # In a real implementation, this would ping a known server
        # For now, return a random value or None
        return None
    
    def _compute_error_rate(self) -> float:
        """Compute current error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests
    
    def _generate_alerts(
        self,
        cpu_usage: float,
        memory_usage: float,
        disk_usage: float,
        gpu_usage: Optional[float],
        model_accuracy: Optional[float],
        data_quality_score: float,
        network_latency: Optional[float],
        error_rate: float
    ) -> List[str]:
        """Generate alerts based on thresholds."""
        alerts = []
        
        if cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        if memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {memory_usage:.1f}%")
        
        if disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"High disk usage: {disk_usage:.1f}%")
        
        if gpu_usage and gpu_usage > self.alert_thresholds['gpu_usage']:
            alerts.append(f"High GPU usage: {gpu_usage:.1f}%")
        
        if model_accuracy and model_accuracy < self.alert_thresholds['model_accuracy']:
            alerts.append(f"Low model accuracy: {model_accuracy:.3f}")
        
        if data_quality_score < self.alert_thresholds['data_quality']:
            alerts.append(f"Poor data quality: {data_quality_score:.3f}")
        
        if network_latency and network_latency > self.alert_thresholds['network_latency']:
            alerts.append(f"High network latency: {network_latency:.1f}ms")
        
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {error_rate:.3f}")
        
        return alerts
    
    def _determine_status(self, alerts: List[str]) -> HealthStatus:
        """Determine overall health status."""
        if not alerts:
            return HealthStatus.HEALTHY
        
        # Check for critical alerts
        critical_keywords = ['disk usage', 'memory usage', 'error rate']
        for alert in alerts:
            for keyword in critical_keywords:
                if keyword in alert.lower():
                    return HealthStatus.CRITICAL
        
        # Otherwise warning
        return HealthStatus.WARNING
    
    def record_prediction(self, prediction: float, actual: Optional[float] = None):
        """Record a model prediction for accuracy tracking."""
        if actual is not None:
            self.model_predictions.append((prediction, actual))
            # Keep only recent predictions
            if len(self.model_predictions) > 200:
                self.model_predictions = self.model_predictions[-200:]
    
    def record_request(self, success: bool = True):
        """Record a request for error rate tracking."""
        self.total_requests += 1
        if not success:
            self.error_count += 1
    
    def record_data_quality(self, quality_score: float):
        """Record a data quality assessment."""
        self.data_quality_scores.append(quality_score)
        # Keep only recent scores
        if len(self.data_quality_scores) > 100:
            self.data_quality_scores = self.data_quality_scores[-100:]
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current health status."""
        return self.current_health
    
    def get_health_history(self) -> List[SystemHealth]:
        """Get health history."""
        return self.health_history.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        if not self.current_health:
            return {"status": "unknown", "message": "No health data available"}
        
        health = self.current_health
        summary = {
            "status": health.status.value,
            "overall_score": health.overall_score(),
            "system_resources": {
                "cpu_usage": health.cpu_usage,
                "memory_usage": health.memory_usage,
                "disk_usage": health.disk_usage,
                "gpu_usage": health.gpu_usage
            },
            "performance": {
                "model_accuracy": health.model_accuracy,
                "data_quality": health.data_quality_score,
                "error_rate": health.error_rate,
                "uptime": health.uptime
            },
            "alerts": health.alerts,
            "last_updated": health.last_updated
        }
        
        return summary
    
    def is_healthy(self) -> bool:
        """Check if system is currently healthy."""
        if not self.current_health:
            return False
        return self.current_health.status == HealthStatus.HEALTHY
    
    def reset_metrics(self):
        """Reset tracking metrics."""
        self.model_predictions.clear()
        self.error_count = 0
        self.total_requests = 0
        self.data_quality_scores.clear()
        logger.info("Health metrics reset")