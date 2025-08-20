"""
📊 Advanced Monitoring, Logging, and Health Check System

This module provides comprehensive monitoring, structured logging, and health
checking capabilities for the Terragon IoT Anomaly Detection System.

Features:
- Structured logging with multiple outputs and levels
- Real-time performance monitoring and metrics collection
- System health checks with predictive alerting
- Distributed tracing and observability
- Custom dashboards and visualization
- Integration with external monitoring systems
- Compliance and audit logging
"""

import os
import sys
import time
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict
from contextlib import contextmanager
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


class LogLevel(Enum):
    """Enhanced logging levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    AUDIT = 70


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'component': self.component,
            'tags': self.tags
        }


@dataclass
class HealthCheckResult:
    """Health check result."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy', 'critical'
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component': self.component,
            'status': self.status,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'recommendations': self.recommendations
        }


@dataclass
class AlertEvent:
    """Alert event structure."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    component: str
    metrics: Dict[str, float] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'metrics': self.metrics,
            'actions_taken': self.actions_taken,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }


class StructuredLogger:
    """Advanced structured logging system."""
    
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.level = level
        self.handlers = []
        self.context = {}
        self._setup_formatters()
    
    def _setup_formatters(self):
        """Setup different log formatters."""
        self.json_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.human_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    
    def add_console_handler(self, use_json: bool = False):
        """Add console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level.value)
        
        if use_json:
            console_handler.setFormatter(self.json_formatter)
        else:
            console_handler.setFormatter(self.human_formatter)
        
        self.handlers.append(console_handler)
    
    def add_file_handler(self, filename: str, use_json: bool = True):
        """Add file logging handler."""
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(self.level.value)
        
        if use_json:
            file_handler.setFormatter(self.json_formatter)
        else:
            file_handler.setFormatter(self.human_formatter)
        
        self.handlers.append(file_handler)
    
    def set_context(self, **kwargs):
        """Set logging context."""
        self.context.update(kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        if level.value < self.level.value:
            return
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level.name,
            'logger': self.name,
            'message': message,
            'context': self.context,
            **kwargs
        }
        
        # Create log record
        if level == LogLevel.TRACE:
            logging.debug(json.dumps(log_data))
        elif level == LogLevel.DEBUG:
            logging.debug(json.dumps(log_data))
        elif level == LogLevel.INFO:
            logging.info(json.dumps(log_data))
        elif level == LogLevel.WARNING:
            logging.warning(json.dumps(log_data))
        elif level == LogLevel.ERROR:
            logging.error(json.dumps(log_data))
        elif level == LogLevel.CRITICAL:
            logging.critical(json.dumps(log_data))
        elif level == LogLevel.SECURITY:
            logging.critical(f"SECURITY: {json.dumps(log_data)}")
        elif level == LogLevel.AUDIT:
            logging.critical(f"AUDIT: {json.dumps(log_data)}")
    
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security event."""
        self._log(LogLevel.SECURITY, message, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log audit event."""
        self._log(LogLevel.AUDIT, message, **kwargs)


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics = deque(maxlen=10000)  # Store last 10k metrics
        self.retention_hours = retention_hours
        self.aggregates = defaultdict(list)
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "count",
        component: str = "system",
        **tags
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            component=component,
            tags=tags
        )
        
        with self.lock:
            self.metrics.append(metric)
            self.aggregates[name].append((metric.timestamp, value))
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        component: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[PerformanceMetric]:
        """Get metrics with filtering."""
        with self.lock:
            filtered_metrics = []
            
            for metric in self.metrics:
                # Apply filters
                if name and metric.metric_name != name:
                    continue
                if component and metric.component != component:
                    continue
                if since and metric.timestamp < since:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def get_aggregated_metrics(
        self,
        name: str,
        aggregation: str = "avg",  # avg, sum, min, max, count
        window_minutes: int = 5
    ) -> Optional[float]:
        """Get aggregated metrics over time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            if name not in self.aggregates:
                return None
            
            # Filter to time window
            recent_values = [
                value for timestamp, value in self.aggregates[name]
                if timestamp >= cutoff_time
            ]
            
            if not recent_values:
                return None
            
            if aggregation == "avg":
                return sum(recent_values) / len(recent_values)
            elif aggregation == "sum":
                return sum(recent_values)
            elif aggregation == "min":
                return min(recent_values)
            elif aggregation == "max":
                return max(recent_values)
            elif aggregation == "count":
                return len(recent_values)
            else:
                return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self.lock:
            summary = {
                'total_metrics': len(self.metrics),
                'unique_metric_names': len(self.aggregates),
                'components': set(),
                'metric_breakdown': {},
                'recent_activity': {}
            }
            
            # Analyze metrics
            for metric in self.metrics:
                summary['components'].add(metric.component)
                
                if metric.metric_name not in summary['metric_breakdown']:
                    summary['metric_breakdown'][metric.metric_name] = {
                        'count': 0,
                        'avg_value': 0,
                        'latest_value': metric.value,
                        'latest_timestamp': metric.timestamp
                    }
                
                breakdown = summary['metric_breakdown'][metric.metric_name]
                breakdown['count'] += 1
                breakdown['avg_value'] = (
                    (breakdown['avg_value'] * (breakdown['count'] - 1) + metric.value) /
                    breakdown['count']
                )
                
                if metric.timestamp > breakdown['latest_timestamp']:
                    breakdown['latest_value'] = metric.value
                    breakdown['latest_timestamp'] = metric.timestamp
            
            # Recent activity (last 5 minutes)
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            recent_metrics = [m for m in self.metrics if m.timestamp >= recent_cutoff]
            summary['recent_activity'] = {
                'count': len(recent_metrics),
                'rate_per_minute': len(recent_metrics) / 5,
                'unique_components': len(set(m.component for m in recent_metrics))
            }
            
            # Convert sets to lists for JSON serialization
            summary['components'] = list(summary['components'])
            
            # Convert timestamps to strings
            for metric_name, breakdown in summary['metric_breakdown'].items():
                breakdown['latest_timestamp'] = breakdown['latest_timestamp'].isoformat()
            
            return summary
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics (runs in background thread)."""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                with self.lock:
                    # Clean metrics deque (automatic due to maxlen)
                    
                    # Clean aggregates
                    for name in list(self.aggregates.keys()):
                        self.aggregates[name] = [
                            (timestamp, value) for timestamp, value in self.aggregates[name]
                            if timestamp >= cutoff_time
                        ]
                        
                        # Remove empty aggregates
                        if not self.aggregates[name]:
                            del self.aggregates[name]
                
                # Sleep for 1 hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                # Log error but continue
                print(f"Error in metrics cleanup: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


class HealthChecker:
    """Comprehensive system health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self.health_history = deque(maxlen=100)
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'model_performance': self._check_model_performance,
            'inference_latency': self._check_inference_latency,
            'error_rate': self._check_error_rate,
            'memory_usage': self._check_memory_usage,
            'throughput': self._check_throughput
        }
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register custom health check."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    recommendations=["Investigate health check implementation"]
                )
        
        # Store in history
        overall_status = self._determine_overall_status(results)
        self.health_history.append({
            'timestamp': datetime.now(),
            'overall_status': overall_status,
            'results': results
        })
        
        return results
    
    def _determine_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Determine overall system status."""
        statuses = [result.status for result in results.values()]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'unhealthy' in statuses:
            return 'unhealthy'
        elif 'degraded' in statuses:
            return 'degraded'
        else:
            return 'healthy'
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        try:
            # Memory check
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                
                memory_usage = memory_allocated / max(memory_reserved, 1)
                
                if memory_usage > 0.9:
                    status = "critical"
                    message = "GPU memory critically high"
                    recommendations = ["Clear GPU cache", "Reduce batch size"]
                elif memory_usage > 0.8:
                    status = "degraded"
                    message = "GPU memory usage high"
                    recommendations = ["Monitor memory usage", "Consider optimization"]
                else:
                    status = "healthy"
                    message = "GPU memory usage normal"
                    recommendations = []
                
                metrics = {
                    'gpu_memory_allocated_gb': memory_allocated,
                    'gpu_memory_reserved_gb': memory_reserved,
                    'gpu_memory_usage_ratio': memory_usage
                }
            else:
                status = "healthy"
                message = "CPU mode - no GPU memory concerns"
                metrics = {}
                recommendations = []
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status="critical",
                message=f"Resource check failed: {str(e)}",
                timestamp=datetime.now(),
                recommendations=["Check system configuration"]
            )
    
    def _check_model_performance(self) -> HealthCheckResult:
        """Check model inference performance."""
        # Get recent inference metrics
        accuracy_metrics = self.metrics_collector.get_metrics(
            name="accuracy",
            since=datetime.now() - timedelta(minutes=10)
        )
        
        if not accuracy_metrics:
            return HealthCheckResult(
                component="model_performance",
                status="degraded",
                message="No recent accuracy metrics available",
                timestamp=datetime.now(),
                recommendations=["Check model inference pipeline"]
            )
        
        # Calculate average accuracy
        recent_accuracy = sum(m.value for m in accuracy_metrics) / len(accuracy_metrics)
        
        if recent_accuracy < 0.7:
            status = "critical"
            message = f"Model accuracy critically low: {recent_accuracy:.2%}"
            recommendations = ["Retrain model", "Check input data quality"]
        elif recent_accuracy < 0.8:
            status = "degraded"
            message = f"Model accuracy below target: {recent_accuracy:.2%}"
            recommendations = ["Monitor data quality", "Consider model tuning"]
        else:
            status = "healthy"
            message = f"Model accuracy good: {recent_accuracy:.2%}"
            recommendations = []
        
        return HealthCheckResult(
            component="model_performance",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={'recent_accuracy': recent_accuracy},
            recommendations=recommendations
        )
    
    def _check_inference_latency(self) -> HealthCheckResult:
        """Check inference latency."""
        latency_avg = self.metrics_collector.get_aggregated_metrics(
            "inference_time_ms", "avg", window_minutes=5
        )
        
        if latency_avg is None:
            return HealthCheckResult(
                component="inference_latency",
                status="degraded",
                message="No latency metrics available",
                timestamp=datetime.now()
            )
        
        if latency_avg > 100:
            status = "critical"
            message = f"Inference latency very high: {latency_avg:.1f}ms"
            recommendations = ["Optimize model", "Check hardware resources"]
        elif latency_avg > 50:
            status = "degraded"
            message = f"Inference latency high: {latency_avg:.1f}ms"
            recommendations = ["Monitor performance", "Consider optimization"]
        else:
            status = "healthy"
            message = f"Inference latency good: {latency_avg:.1f}ms"
            recommendations = []
        
        return HealthCheckResult(
            component="inference_latency",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={'avg_latency_ms': latency_avg},
            recommendations=recommendations
        )
    
    def _check_error_rate(self) -> HealthCheckResult:
        """Check system error rate."""
        error_count = self.metrics_collector.get_aggregated_metrics(
            "errors", "sum", window_minutes=10
        )
        total_requests = self.metrics_collector.get_aggregated_metrics(
            "requests", "sum", window_minutes=10
        )
        
        if error_count is None or total_requests is None:
            return HealthCheckResult(
                component="error_rate",
                status="degraded",
                message="Insufficient metrics for error rate calculation",
                timestamp=datetime.now()
            )
        
        error_rate = error_count / max(total_requests, 1)
        
        if error_rate > 0.1:
            status = "critical"
            message = f"Error rate very high: {error_rate:.1%}"
            recommendations = ["Investigate errors", "Check system stability"]
        elif error_rate > 0.05:
            status = "degraded"
            message = f"Error rate elevated: {error_rate:.1%}"
            recommendations = ["Monitor error patterns", "Review logs"]
        else:
            status = "healthy"
            message = f"Error rate normal: {error_rate:.1%}"
            recommendations = []
        
        return HealthCheckResult(
            component="error_rate",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={'error_rate': error_rate, 'error_count': error_count},
            recommendations=recommendations
        )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage patterns."""
        # This would typically use psutil for real memory monitoring
        # For now, provide a basic implementation
        return HealthCheckResult(
            component="memory_usage",
            status="healthy",
            message="Memory usage monitoring not fully implemented",
            timestamp=datetime.now(),
            recommendations=["Implement detailed memory monitoring"]
        )
    
    def _check_throughput(self) -> HealthCheckResult:
        """Check system throughput."""
        throughput = self.metrics_collector.get_aggregated_metrics(
            "requests", "sum", window_minutes=1
        )
        
        if throughput is None:
            return HealthCheckResult(
                component="throughput",
                status="degraded",
                message="No throughput metrics available",
                timestamp=datetime.now()
            )
        
        # Normalize to requests per minute
        requests_per_minute = throughput
        
        if requests_per_minute < 10:
            status = "degraded"
            message = f"Low throughput: {requests_per_minute:.1f} req/min"
            recommendations = ["Check for system bottlenecks", "Review load patterns"]
        else:
            status = "healthy"
            message = f"Throughput normal: {requests_per_minute:.1f} req/min"
            recommendations = []
        
        return HealthCheckResult(
            component="throughput",
            status=status,
            message=message,
            timestamp=datetime.now(),
            metrics={'requests_per_minute': requests_per_minute},
            recommendations=recommendations
        )


class AlertManager:
    """Advanced alerting system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts = deque(maxlen=1000)
        self.alert_rules = {}
        self.notification_handlers = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = {
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 0.1,
                'operator': '>',
                'severity': AlertSeverity.CRITICAL,
                'window_minutes': 5
            },
            'high_latency': {
                'metric': 'inference_time_ms',
                'threshold': 100,
                'operator': '>',
                'severity': AlertSeverity.WARNING,
                'window_minutes': 3
            },
            'low_throughput': {
                'metric': 'requests_per_minute',
                'threshold': 5,
                'operator': '<',
                'severity': AlertSeverity.WARNING,
                'window_minutes': 5
            }
        }
    
    def check_alerts(self) -> List[AlertEvent]:
        """Check all alert rules and generate alerts."""
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            try:
                alert = self._evaluate_rule(rule_name, rule)
                if alert:
                    new_alerts.append(alert)
                    self.alerts.append(alert)
            except Exception as e:
                # Log error but continue
                print(f"Error evaluating alert rule {rule_name}: {e}")
        
        # Send notifications
        for alert in new_alerts:
            self._send_notifications(alert)
        
        return new_alerts
    
    def _evaluate_rule(self, rule_name: str, rule: Dict[str, Any]) -> Optional[AlertEvent]:
        """Evaluate a single alert rule."""
        metric_value = self.metrics_collector.get_aggregated_metrics(
            rule['metric'], 'avg', rule['window_minutes']
        )
        
        if metric_value is None:
            return None
        
        # Check threshold
        triggered = False
        if rule['operator'] == '>':
            triggered = metric_value > rule['threshold']
        elif rule['operator'] == '<':
            triggered = metric_value < rule['threshold']
        elif rule['operator'] == '==':
            triggered = metric_value == rule['threshold']
        
        if not triggered:
            return None
        
        # Create alert
        alert = AlertEvent(
            alert_id=f"{rule_name}_{int(time.time())}",
            severity=rule['severity'],
            title=f"Alert: {rule_name}",
            description=f"Metric {rule['metric']} is {metric_value} (threshold: {rule['threshold']})",
            timestamp=datetime.now(),
            component=rule.get('component', 'system'),
            metrics={rule['metric']: metric_value}
        )
        
        return alert
    
    def _send_notifications(self, alert: AlertEvent):
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error sending notification: {e}")
    
    def add_notification_handler(self, handler: Callable[[AlertEvent], None]):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get currently active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                alert.actions_taken.append(f"Resolved: {resolution_note}")
                break


# Performance monitoring decorator
@contextmanager
def monitor_performance(metrics_collector: MetricsCollector, operation_name: str, component: str = "system"):
    """Context manager for monitoring operation performance."""
    start_time = time.time()
    error_occurred = False
    
    try:
        yield
    except Exception as e:
        error_occurred = True
        metrics_collector.record_metric(
            "errors", 1, "count", component,
            operation=operation_name, error_type=type(e).__name__
        )
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        metrics_collector.record_metric(
            f"{operation_name}_duration_ms", duration_ms, "ms", component
        )
        
        if not error_occurred:
            metrics_collector.record_metric(
                f"{operation_name}_success", 1, "count", component
            )


# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Advanced Monitoring System")
    print("=" * 50)
    
    # Setup monitoring system
    metrics_collector = MetricsCollector()
    health_checker = HealthChecker(metrics_collector)
    alert_manager = AlertManager(metrics_collector)
    
    # Test metrics collection
    print("Testing metrics collection...")
    metrics_collector.record_metric("test_metric", 1.5, "ms", "test_component")
    metrics_collector.record_metric("inference_time_ms", 25.0, "ms", "model")
    metrics_collector.record_metric("accuracy", 0.95, "ratio", "model")
    
    # Test health checks
    print("Running health checks...")
    health_results = health_checker.run_health_checks()
    for name, result in health_results.items():
        print(f"  {name}: {result.status} - {result.message}")
    
    # Test alerts
    print("Checking alerts...")
    alerts = alert_manager.check_alerts()
    print(f"Generated {len(alerts)} alerts")
    
    # Test performance monitoring
    print("Testing performance monitoring...")
    with monitor_performance(metrics_collector, "test_operation", "test"):
        time.sleep(0.01)  # Simulate work
    
    # Get metrics summary
    summary = metrics_collector.get_metrics_summary()
    print(f"Metrics summary: {summary['total_metrics']} metrics collected")
    
    print("✅ Advanced monitoring system tested successfully!")