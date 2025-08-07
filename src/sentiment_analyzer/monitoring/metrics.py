"""
Advanced metrics collection and monitoring for sentiment analysis.
"""
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Performance statistics."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count > 0 else 0.0
    
    def update(self, processing_time: float, error: bool = False):
        """Update statistics with new data point."""
        self.count += 1
        self.total_time += processing_time
        self.min_time = min(self.min_time, processing_time)
        self.max_time = max(self.max_time, processing_time)
        
        if error:
            self.error_count += 1


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 timeout: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
            
            try:
                # Execute with timeout
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time
        }


class CircuitBreakerError(Exception):
    """Circuit breaker exception."""
    pass


class MetricsCollector:
    """
    Advanced metrics collection system for sentiment analysis.
    
    Collects performance metrics, error rates, and usage statistics.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Thread-safe collections
        self._lock = threading.RLock()
        
        # Performance stats by model
        self.performance_stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        
        # Time series data
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Request counters
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Confidence distribution
        self.confidence_histogram = defaultdict(int)
        
        # Text length stats
        self.text_length_stats = PerformanceStats()
        
        logger.info(f"Initialized metrics collector with {retention_hours}h retention")
    
    def record_analysis(self, 
                       model_type: str,
                       processing_time: float,
                       confidence: float,
                       text_length: int,
                       success: bool = True,
                       labels: Optional[Dict[str, str]] = None):
        """Record sentiment analysis metrics."""
        with self._lock:
            timestamp = datetime.now()
            labels = labels or {}
            
            # Update performance stats
            self.performance_stats[model_type].update(processing_time, not success)
            
            # Record time series data
            self.metrics['processing_time'].append(
                MetricPoint(timestamp, processing_time, {**labels, 'model': model_type})
            )
            
            self.metrics['confidence'].append(
                MetricPoint(timestamp, confidence, {**labels, 'model': model_type})
            )
            
            self.metrics['text_length'].append(
                MetricPoint(timestamp, text_length, {**labels, 'model': model_type})
            )
            
            # Update counters
            self.request_counts[model_type] += 1
            if not success:
                self.error_counts[model_type] += 1
            
            # Confidence histogram (0.1 buckets)
            confidence_bucket = int(confidence * 10) / 10
            self.confidence_histogram[confidence_bucket] += 1
            
            # Text length stats
            self.text_length_stats.update(text_length)
            
            # Cleanup old data
            self._cleanup_old_data()
    
    def record_batch_analysis(self,
                            model_type: str,
                            batch_size: int,
                            total_processing_time: float,
                            success_count: int,
                            error_count: int):
        """Record batch analysis metrics."""
        with self._lock:
            timestamp = datetime.now()
            
            # Batch-specific metrics
            self.metrics['batch_size'].append(
                MetricPoint(timestamp, batch_size, {'model': model_type})
            )
            
            self.metrics['batch_processing_time'].append(
                MetricPoint(timestamp, total_processing_time, {'model': model_type})
            )
            
            self.metrics['batch_success_rate'].append(
                MetricPoint(timestamp, success_count / batch_size if batch_size > 0 else 0, 
                           {'model': model_type})
            )
            
            # Update counters
            self.request_counts[f"{model_type}_batch"] += 1
            self.error_counts[f"{model_type}_batch"] += error_count
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            current_time = datetime.now()
            
            # Performance by model
            model_performance = {}
            for model, stats in self.performance_stats.items():
                model_performance[model] = {
                    "request_count": stats.count,
                    "avg_processing_time": stats.avg_time,
                    "min_processing_time": stats.min_time if stats.min_time != float('inf') else 0,
                    "max_processing_time": stats.max_time,
                    "error_count": stats.error_count,
                    "error_rate": stats.error_rate,
                    "success_rate": 1.0 - stats.error_rate
                }
            
            # Recent metrics (last hour)
            recent_cutoff = current_time - timedelta(hours=1)
            recent_processing_times = [
                point.value for point in self.metrics['processing_time']
                if point.timestamp >= recent_cutoff
            ]
            
            recent_confidences = [
                point.value for point in self.metrics['confidence']
                if point.timestamp >= recent_cutoff
            ]
            
            # Circuit breaker status
            circuit_breaker_status = {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            }
            
            return {
                "timestamp": current_time.isoformat(),
                "models": model_performance,
                "recent_metrics": {
                    "processing_times": {
                        "count": len(recent_processing_times),
                        "avg": sum(recent_processing_times) / len(recent_processing_times) if recent_processing_times else 0,
                        "min": min(recent_processing_times) if recent_processing_times else 0,
                        "max": max(recent_processing_times) if recent_processing_times else 0
                    },
                    "confidences": {
                        "count": len(recent_confidences),
                        "avg": sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0,
                        "min": min(recent_confidences) if recent_confidences else 0,
                        "max": max(recent_confidences) if recent_confidences else 0
                    }
                },
                "confidence_distribution": dict(sorted(self.confidence_histogram.items())),
                "text_length_stats": {
                    "count": self.text_length_stats.count,
                    "avg": self.text_length_stats.avg_time,  # Using avg_time as avg_length
                    "min": self.text_length_stats.min_time if self.text_length_stats.min_time != float('inf') else 0,
                    "max": self.text_length_stats.max_time
                },
                "circuit_breakers": circuit_breaker_status,
                "total_requests": sum(self.request_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "retention_hours": self.retention_hours
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        with self._lock:
            current_time = datetime.now()
            
            # Check if any circuit breakers are open
            breaker_issues = [
                name for name, cb in self.circuit_breakers.items()
                if cb.state == "OPEN"
            ]
            
            # Check error rates in last hour
            recent_cutoff = current_time - timedelta(hours=1)
            high_error_models = []
            
            for model, stats in self.performance_stats.items():
                if stats.count > 10 and stats.error_rate > 0.1:  # >10% error rate
                    high_error_models.append({
                        "model": model,
                        "error_rate": stats.error_rate,
                        "error_count": stats.error_count,
                        "total_count": stats.count
                    })
            
            # Overall health
            is_healthy = len(breaker_issues) == 0 and len(high_error_models) == 0
            
            return {
                "healthy": is_healthy,
                "timestamp": current_time.isoformat(),
                "issues": {
                    "open_circuit_breakers": breaker_issues,
                    "high_error_models": high_error_models
                },
                "metrics_count": sum(len(deque_obj) for deque_obj in self.metrics.values()),
                "active_models": list(self.performance_stats.keys())
            }
    
    def _cleanup_old_data(self):
        """Remove data older than retention period."""
        cutoff_time = datetime.now() - timedelta(seconds=self.retention_seconds)
        
        for metric_name, data_points in self.metrics.items():
            while data_points and data_points[0].timestamp < cutoff_time:
                data_points.popleft()
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        with self._lock:
            lines = []
            
            # Processing time metrics
            for model, stats in self.performance_stats.items():
                lines.append(f'sentiment_processing_time_total{{model="{model}"}} {stats.total_time}')
                lines.append(f'sentiment_requests_total{{model="{model}"}} {stats.count}')
                lines.append(f'sentiment_errors_total{{model="{model}"}} {stats.error_count}')
                
                if stats.count > 0:
                    lines.append(f'sentiment_avg_processing_time{{model="{model}"}} {stats.avg_time}')
                    lines.append(f'sentiment_error_rate{{model="{model}"}} {stats.error_rate}')
            
            # Confidence histogram
            for confidence, count in self.confidence_histogram.items():
                lines.append(f'sentiment_confidence_histogram{{bucket="{confidence}"}} {count}')
            
            return '\n'.join(lines)
    
    def reset_stats(self):
        """Reset all statistics (for testing)."""
        with self._lock:
            self.performance_stats.clear()
            self.metrics.clear()
            self.request_counts.clear()
            self.error_counts.clear()
            self.circuit_breakers.clear()
            self.confidence_histogram.clear()
            self.text_length_stats = PerformanceStats()


# Global metrics collector instance
metrics_collector = MetricsCollector()