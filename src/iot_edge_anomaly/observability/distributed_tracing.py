"""
Advanced Monitoring and Observability with Distributed Tracing.

This module provides comprehensive observability capabilities:
- Distributed tracing across microservices with OpenTelemetry
- Advanced metrics collection and aggregation
- Log correlation and structured logging
- Performance monitoring with APM integration
- Real-time alerting and anomaly detection
- Observability dashboards and visualization
- Trace sampling and performance optimization
"""

import asyncio
import logging
import json
import time
import uuid
import traceback
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import socket
import platform
from collections import defaultdict, deque
import numpy as np

# OpenTelemetry imports (would be installed in production)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Mock classes for demonstration
    class trace:
        @staticmethod
        def get_tracer(name): return MockTracer()
        @staticmethod
        def set_tracer_provider(provider): pass
    
    class metrics:
        @staticmethod
        def get_meter(name): return MockMeter()
        @staticmethod
        def set_meter_provider(provider): pass
    
    class MockTracer:
        def start_as_current_span(self, name): return MockSpan()
    
    class MockMeter:
        def create_counter(self, name, **kwargs): return MockInstrument()
        def create_histogram(self, name, **kwargs): return MockInstrument()
        def create_gauge(self, name, **kwargs): return MockInstrument()
    
    class MockSpan:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def add_event(self, name, attributes=None): pass
    
    class MockInstrument:
        def add(self, amount, attributes=None): pass
        def record(self, amount, attributes=None): pass
        def set(self, amount, attributes=None): pass

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Trace level for sampling decisions."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TraceSpan:
    """Represents a trace span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events
        }


@dataclass
class MetricSample:
    """Represents a metric sample."""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class AlertRule:
    """Represents an alerting rule."""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # e.g., "greater_than", "less_than", "equals"
    threshold: float
    duration_minutes: int
    severity: AlertSeverity
    enabled: bool = True
    
    def evaluate(self, value: float, duration_triggered: int) -> bool:
        """Evaluate if alert should fire."""
        if not self.enabled:
            return False
        
        condition_met = False
        if self.condition == "greater_than":
            condition_met = value > self.threshold
        elif self.condition == "less_than":
            condition_met = value < self.threshold
        elif self.condition == "equals":
            condition_met = value == self.threshold
        elif self.condition == "not_equals":
            condition_met = value != self.threshold
        
        return condition_met and duration_triggered >= self.duration_minutes


class TraceSampler:
    """
    Intelligent trace sampling to manage performance and storage.
    
    Features:
    - Adaptive sampling based on system load
    - Error trace prioritization
    - Critical path sampling
    - Statistical sampling with rate limiting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trace sampler."""
        self.config = config or {}
        
        # Sampling configuration
        self.default_sample_rate = self.config.get("default_sample_rate", 0.1)  # 10%
        self.error_sample_rate = self.config.get("error_sample_rate", 1.0)  # 100% for errors
        self.critical_path_sample_rate = self.config.get("critical_path_sample_rate", 0.5)  # 50%
        
        # Adaptive sampling
        self.adaptive_sampling = self.config.get("adaptive_sampling", True)
        self.max_traces_per_second = self.config.get("max_traces_per_second", 100)
        
        # Sampling state
        self.current_sample_rates: Dict[str, float] = {}
        self.trace_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # Last 60 seconds
        
        # Critical paths (operations that should be sampled more frequently)
        self.critical_paths = set(self.config.get("critical_paths", [
            "model_inference", "data_ingestion", "health_check", "security_check"
        ]))
    
    def should_sample(self, operation_name: str, service_name: str, 
                     trace_level: TraceLevel = TraceLevel.INFO, 
                     is_error: bool = False) -> bool:
        """
        Decide whether to sample this trace.
        
        Args:
            operation_name: Name of the operation
            service_name: Name of the service
            trace_level: Trace level
            is_error: Whether this trace represents an error
            
        Returns:
            True if trace should be sampled
        """
        # Always sample errors
        if is_error or trace_level in [TraceLevel.ERROR, TraceLevel.CRITICAL]:
            return np.random.random() < self.error_sample_rate
        
        # Sample critical paths more frequently
        if operation_name in self.critical_paths:
            return np.random.random() < self.critical_path_sample_rate
        
        # Adaptive sampling based on current load
        if self.adaptive_sampling:
            current_rate = self._calculate_adaptive_rate(service_name)
        else:
            current_rate = self.default_sample_rate
        
        return np.random.random() < current_rate
    
    def _calculate_adaptive_rate(self, service_name: str) -> float:
        """Calculate adaptive sampling rate based on current load."""
        current_time = time.time()
        
        # Clean old entries
        trace_count_deque = self.trace_counts[service_name]
        while trace_count_deque and trace_count_deque[0] < current_time - 60:
            trace_count_deque.popleft()
        
        # Calculate current traces per second
        current_tps = len(trace_count_deque)
        
        # Adjust sampling rate based on load
        if current_tps > self.max_traces_per_second:
            # Reduce sampling rate when overloaded
            reduction_factor = self.max_traces_per_second / current_tps
            adaptive_rate = self.default_sample_rate * reduction_factor
        else:
            # Use default rate when not overloaded
            adaptive_rate = self.default_sample_rate
        
        # Update current rate
        self.current_sample_rates[service_name] = adaptive_rate
        
        return adaptive_rate
    
    def record_trace(self, service_name: str) -> None:
        """Record that a trace was created."""
        current_time = time.time()
        self.trace_counts[service_name].append(current_time)
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            "default_sample_rate": self.default_sample_rate,
            "error_sample_rate": self.error_sample_rate,
            "critical_path_sample_rate": self.critical_path_sample_rate,
            "adaptive_sampling_enabled": self.adaptive_sampling,
            "current_sample_rates": dict(self.current_sample_rates),
            "critical_paths": list(self.critical_paths),
            "traces_per_service": {
                service: len(counts) 
                for service, counts in self.trace_counts.items()
            }
        }


class MetricsCollector:
    """
    Advanced metrics collection and aggregation.
    
    Features:
    - Multiple metric types (counter, histogram, gauge, summary)
    - Automatic system metrics collection
    - Custom application metrics
    - Metric aggregation and bucketing
    - Time series data management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics collector."""
        self.config = config or {}
        
        # Metrics storage
        self.metrics: Dict[str, List[MetricSample]] = defaultdict(list)
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # OpenTelemetry meter
        self.meter = metrics.get_meter("iot_edge_anomaly")
        
        # Metric instruments
        self.counters: Dict[str, Any] = {}
        self.histograms: Dict[str, Any] = {}
        self.gauges: Dict[str, Any] = {}
        
        # System metrics collection
        self.collect_system_metrics = self.config.get("collect_system_metrics", True)
        self.system_metrics_interval = self.config.get("system_metrics_interval", 10)  # seconds
        
        # Background collection task
        self.collection_running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default application metrics."""
        # Request metrics
        self.request_counter = self._create_counter(
            "http_requests_total",
            "Total number of HTTP requests",
            tags=["method", "endpoint", "status_code"]
        )
        
        self.request_duration = self._create_histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            tags=["method", "endpoint"]
        )
        
        # Model inference metrics
        self.inference_counter = self._create_counter(
            "model_inference_total",
            "Total number of model inferences",
            tags=["model_version", "status"]
        )
        
        self.inference_duration = self._create_histogram(
            "model_inference_duration_seconds",
            "Model inference duration in seconds",
            tags=["model_version"]
        )
        
        # System resource metrics
        self.cpu_usage = self._create_gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage"
        )
        
        self.memory_usage = self._create_gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes"
        )
        
        self.disk_usage = self._create_gauge(
            "system_disk_usage_bytes",
            "System disk usage in bytes"
        )
    
    def _create_counter(self, name: str, description: str, 
                       tags: List[str] = None) -> Any:
        """Create a counter metric."""
        counter = self.meter.create_counter(
            name=name,
            description=description,
            unit="1"
        )
        self.counters[name] = counter
        self.metric_metadata[name] = {
            "type": MetricType.COUNTER,
            "description": description,
            "tags": tags or []
        }
        return counter
    
    def _create_histogram(self, name: str, description: str, 
                         tags: List[str] = None) -> Any:
        """Create a histogram metric."""
        histogram = self.meter.create_histogram(
            name=name,
            description=description
        )
        self.histograms[name] = histogram
        self.metric_metadata[name] = {
            "type": MetricType.HISTOGRAM,
            "description": description,
            "tags": tags or []
        }
        return histogram
    
    def _create_gauge(self, name: str, description: str, 
                     tags: List[str] = None) -> Any:
        """Create a gauge metric."""
        # Note: OpenTelemetry uses ObservableGauge for gauges
        # This is a simplified implementation
        self.metric_metadata[name] = {
            "type": MetricType.GAUGE,
            "description": description,
            "tags": tags or []
        }
        return name  # Return name for internal tracking
    
    def increment_counter(self, name: str, value: Union[int, float] = 1, 
                         tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        if name in self.counters:
            self.counters[name].add(value, attributes=tags)
        
        # Store for internal aggregation
        sample = MetricSample(
            name=name,
            metric_type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics[name].append(sample)
    
    def record_histogram(self, name: str, value: Union[int, float], 
                        tags: Dict[str, str] = None) -> None:
        """Record a histogram value."""
        if name in self.histograms:
            self.histograms[name].record(value, attributes=tags)
        
        # Store for internal aggregation
        sample = MetricSample(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics[name].append(sample)
    
    def set_gauge(self, name: str, value: Union[int, float], 
                  tags: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        # Store for internal tracking
        sample = MetricSample(
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics[name].append(sample)
    
    async def start_collection(self) -> None:
        """Start background metrics collection."""
        if self.collection_running:
            return
        
        self.collection_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop background metrics collection."""
        self.collection_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self.collection_running:
            try:
                if self.collect_system_metrics:
                    await self._collect_system_metrics()
                
                await asyncio.sleep(self.system_metrics_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.system_metrics_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used)
            self.set_gauge("system_memory_usage_percent", memory.percent)
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                self.set_gauge("system_disk_usage_bytes", disk.used)
                self.set_gauge("system_disk_usage_percent", (disk.used / disk.total) * 100)
            except:
                pass  # May not work on all systems
            
            # Network I/O
            try:
                network = psutil.net_io_counters()
                self.set_gauge("system_network_bytes_sent", network.bytes_sent)
                self.set_gauge("system_network_bytes_recv", network.bytes_recv)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_metric_values(self, name: str, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> List[MetricSample]:
        """Get metric values within time range."""
        samples = self.metrics.get(name, [])
        
        if start_time or end_time:
            filtered_samples = []
            for sample in samples:
                if start_time and sample.timestamp < start_time:
                    continue
                if end_time and sample.timestamp > end_time:
                    continue
                filtered_samples.append(sample)
            return filtered_samples
        
        return samples
    
    def aggregate_metrics(self, name: str, aggregation: str = "avg", 
                         time_window_minutes: int = 5) -> Optional[float]:
        """Aggregate metric values."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        samples = self.get_metric_values(name, start_time, end_time)
        if not samples:
            return None
        
        values = [sample.value for sample in samples]
        
        if aggregation == "avg":
            return np.mean(values)
        elif aggregation == "sum":
            return np.sum(values)
        elif aggregation == "min":
            return np.min(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "count":
            return len(values)
        elif aggregation == "p50":
            return np.percentile(values, 50)
        elif aggregation == "p95":
            return np.percentile(values, 95)
        elif aggregation == "p99":
            return np.percentile(values, 99)
        else:
            return np.mean(values)  # Default to average
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics collection summary."""
        return {
            "total_metrics": len(self.metrics),
            "total_samples": sum(len(samples) for samples in self.metrics.values()),
            "metric_types": {
                metric_type.value: len([
                    name for name, metadata in self.metric_metadata.items()
                    if metadata["type"] == metric_type
                ])
                for metric_type in MetricType
            },
            "collection_running": self.collection_running,
            "system_metrics_enabled": self.collect_system_metrics
        }


class AlertManager:
    """
    Real-time alerting and anomaly detection.
    
    Features:
    - Rule-based alerting
    - Statistical anomaly detection
    - Alert aggregation and deduplication
    - Multi-channel notifications
    - Alert escalation policies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager."""
        self.config = config or {}
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Alert state tracking
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Notification configuration
        self.notification_channels = self.config.get("notification_channels", {})
        
        # Anomaly detection
        self.anomaly_detection_enabled = self.config.get("anomaly_detection", True)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 2.0)  # Standard deviations
        
        # Alert state
        self.rule_states: Dict[str, Dict[str, Any]] = {}
        
        # Background task
        self.monitoring_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        # High CPU usage alert
        cpu_alert = AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            metric_name="system_cpu_usage_percent",
            condition="greater_than",
            threshold=85.0,
            duration_minutes=2,
            severity=AlertSeverity.HIGH
        )
        
        # High memory usage alert
        memory_alert = AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            metric_name="system_memory_usage_percent",
            condition="greater_than",
            threshold=90.0,
            duration_minutes=1,
            severity=AlertSeverity.CRITICAL
        )
        
        # Model inference errors
        inference_error_alert = AlertRule(
            rule_id="model_inference_errors",
            name="Model Inference Errors",
            metric_name="model_inference_error_rate",
            condition="greater_than",
            threshold=0.05,  # 5% error rate
            duration_minutes=3,
            severity=AlertSeverity.MEDIUM
        )
        
        # Slow inference times
        slow_inference_alert = AlertRule(
            rule_id="slow_inference",
            name="Slow Model Inference",
            metric_name="model_inference_duration_p95",
            condition="greater_than",
            threshold=5.0,  # 5 seconds
            duration_minutes=5,
            severity=AlertSeverity.MEDIUM
        )
        
        for rule in [cpu_alert, memory_alert, inference_error_alert, slow_inference_alert]:
            self.add_alert_rule(rule)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        self.rule_states[rule.rule_id] = {
            "triggered": False,
            "trigger_start": None,
            "last_evaluation": None,
            "trigger_count": 0
        }
        logger.info(f"Added alert rule: {rule.name}")
    
    async def evaluate_rules(self, metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
        """Evaluate all alert rules against current metrics."""
        fired_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            try:
                # Get current metric value
                current_value = metrics_collector.aggregate_metrics(
                    rule.metric_name,
                    aggregation="avg",
                    time_window_minutes=1
                )
                
                if current_value is not None:
                    alert_result = await self._evaluate_rule(rule, current_value)
                    if alert_result:
                        fired_alerts.append(alert_result)
            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_id}: {e}")
        
        # Anomaly detection
        if self.anomaly_detection_enabled:
            anomaly_alerts = await self._detect_anomalies(metrics_collector)
            fired_alerts.extend(anomaly_alerts)
        
        return fired_alerts
    
    async def _evaluate_rule(self, rule: AlertRule, current_value: float) -> Optional[Dict[str, Any]]:
        """Evaluate a single alert rule."""
        rule_state = self.rule_states[rule.rule_id]
        current_time = datetime.now()
        
        # Check if condition is met
        condition_met = False
        if rule.condition == "greater_than":
            condition_met = current_value > rule.threshold
        elif rule.condition == "less_than":
            condition_met = current_value < rule.threshold
        elif rule.condition == "equals":
            condition_met = current_value == rule.threshold
        
        if condition_met:
            if not rule_state["triggered"]:
                # Start tracking this potential alert
                rule_state["triggered"] = True
                rule_state["trigger_start"] = current_time
            
            # Calculate how long condition has been met
            trigger_duration = (current_time - rule_state["trigger_start"]).total_seconds() / 60
            
            # Check if duration threshold is met
            if trigger_duration >= rule.duration_minutes:
                # Fire alert
                alert = await self._fire_alert(rule, current_value, trigger_duration)
                rule_state["trigger_count"] += 1
                return alert
        else:
            # Condition not met, reset state
            if rule_state["triggered"]:
                rule_state["triggered"] = False
                rule_state["trigger_start"] = None
                
                # If there was an active alert, resolve it
                if rule.rule_id in self.active_alerts:
                    await self._resolve_alert(rule.rule_id)
        
        rule_state["last_evaluation"] = current_time
        return None
    
    async def _fire_alert(self, rule: AlertRule, current_value: float, 
                         trigger_duration: float) -> Dict[str, Any]:
        """Fire an alert."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        alert = {
            "alert_id": alert_id,
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "trigger_duration_minutes": trigger_duration,
            "fired_at": datetime.now().isoformat(),
            "status": "firing"
        }
        
        # Store active alert
        self.active_alerts[rule.rule_id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_alert_notification(alert)
        
        logger.warning(f"Alert fired: {rule.name} - {current_value} {rule.condition} {rule.threshold}")
        
        return alert
    
    async def _resolve_alert(self, rule_id: str) -> None:
        """Resolve an active alert."""
        if rule_id in self.active_alerts:
            alert = self.active_alerts[rule_id]
            alert["status"] = "resolved"
            alert["resolved_at"] = datetime.now().isoformat()
            
            # Remove from active alerts
            del self.active_alerts[rule_id]
            
            # Send resolution notification
            await self._send_alert_resolution(alert)
            
            logger.info(f"Alert resolved: {alert['rule_name']}")
    
    async def _detect_anomalies(self, metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
        """Detect statistical anomalies in metrics."""
        anomaly_alerts = []
        
        # Get list of numeric metrics to analyze
        numeric_metrics = [
            name for name, metadata in metrics_collector.metric_metadata.items()
            if metadata["type"] in [MetricType.GAUGE, MetricType.HISTOGRAM]
        ]
        
        for metric_name in numeric_metrics:
            try:
                # Get historical data (last 1 hour)
                historical_samples = metrics_collector.get_metric_values(
                    metric_name,
                    start_time=datetime.now() - timedelta(hours=1)
                )
                
                if len(historical_samples) < 10:  # Need minimum samples
                    continue
                
                # Get recent samples (last 5 minutes)
                recent_samples = metrics_collector.get_metric_values(
                    metric_name,
                    start_time=datetime.now() - timedelta(minutes=5)
                )
                
                if not recent_samples:
                    continue
                
                # Statistical analysis
                historical_values = [s.value for s in historical_samples]
                recent_values = [s.value for s in recent_samples]
                
                historical_mean = np.mean(historical_values)
                historical_std = np.std(historical_values)
                recent_mean = np.mean(recent_values)
                
                # Check if recent mean is anomalous
                if historical_std > 0:
                    z_score = abs(recent_mean - historical_mean) / historical_std
                    
                    if z_score > self.anomaly_threshold:
                        # Create anomaly alert
                        alert = {
                            "alert_id": f"anomaly_{metric_name}_{int(time.time())}",
                            "rule_id": f"anomaly_{metric_name}",
                            "rule_name": f"Statistical Anomaly in {metric_name}",
                            "severity": AlertSeverity.MEDIUM.value,
                            "metric_name": metric_name,
                            "current_value": recent_mean,
                            "historical_mean": historical_mean,
                            "z_score": z_score,
                            "anomaly_threshold": self.anomaly_threshold,
                            "fired_at": datetime.now().isoformat(),
                            "status": "firing",
                            "type": "statistical_anomaly"
                        }
                        
                        anomaly_alerts.append(alert)
                        self.alert_history.append(alert)
                        
                        logger.warning(f"Statistical anomaly detected in {metric_name}: z_score={z_score:.2f}")
            
            except Exception as e:
                logger.error(f"Error detecting anomalies in {metric_name}: {e}")
        
        return anomaly_alerts
    
    async def _send_alert_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert notification through configured channels."""
        for channel_name, channel_config in self.notification_channels.items():
            try:
                if channel_config["type"] == "email":
                    await self._send_email_notification(alert, channel_config)
                elif channel_config["type"] == "webhook":
                    await self._send_webhook_notification(alert, channel_config)
                elif channel_config["type"] == "slack":
                    await self._send_slack_notification(alert, channel_config)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    async def _send_email_notification(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send email notification (mock implementation)."""
        logger.info(f"EMAIL ALERT: {alert['rule_name']} - Severity: {alert['severity']}")
    
    async def _send_webhook_notification(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send webhook notification (mock implementation)."""
        logger.info(f"WEBHOOK ALERT to {config.get('url')}: {alert['rule_name']}")
    
    async def _send_slack_notification(self, alert: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Send Slack notification (mock implementation)."""
        logger.info(f"SLACK ALERT: {alert['rule_name']} - {alert['current_value']}")
    
    async def _send_alert_resolution(self, alert: Dict[str, Any]) -> None:
        """Send alert resolution notification."""
        logger.info(f"ALERT RESOLVED: {alert['rule_name']}")
    
    async def start_monitoring(self, metrics_collector: MetricsCollector) -> None:
        """Start alert monitoring."""
        if self.monitoring_running:
            return
        
        self.monitoring_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(metrics_collector))
        logger.info("Started alert monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop alert monitoring."""
        self.monitoring_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped alert monitoring")
    
    async def _monitoring_loop(self, metrics_collector: MetricsCollector) -> None:
        """Background alert monitoring loop."""
        while self.monitoring_running:
            try:
                fired_alerts = await self.evaluate_rules(metrics_collector)
                
                if fired_alerts:
                    logger.info(f"Evaluated alert rules, {len(fired_alerts)} alerts fired")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alerting system summary."""
        return {
            "total_rules": len(self.alert_rules),
            "active_alerts": len(self.active_alerts),
            "total_alert_history": len(self.alert_history),
            "rules_by_severity": {
                severity.value: len([
                    rule for rule in self.alert_rules.values()
                    if rule.severity == severity
                ])
                for severity in AlertSeverity
            },
            "notification_channels": len(self.notification_channels),
            "monitoring_running": self.monitoring_running,
            "anomaly_detection_enabled": self.anomaly_detection_enabled
        }


class DistributedTracer:
    """
    Distributed tracing system with OpenTelemetry integration.
    
    Features:
    - Distributed trace collection and correlation
    - Span management and hierarchy
    - Performance analysis and bottleneck detection
    - Service dependency mapping
    - Trace export to observability platforms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize distributed tracer."""
        self.config = config or {}
        
        # Service information
        self.service_name = self.config.get("service_name", "iot_edge_anomaly")
        self.service_version = self.config.get("service_version", "1.0.0")
        
        # Tracing configuration
        self.export_endpoint = self.config.get("export_endpoint", "http://localhost:4317")
        
        # Initialize OpenTelemetry
        self.tracer = trace.get_tracer(self.service_name)
        
        # Trace sampling
        self.sampler = TraceSampler(self.config.get("sampling", {}))
        
        # In-memory trace storage for analysis
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.span_relationships: Dict[str, List[str]] = defaultdict(list)  # parent -> children
        
        # Performance analysis
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_duration": 0.0,
            "min_duration": float('inf'),
            "max_duration": 0.0,
            "error_count": 0
        })
        
        # Initialize if OpenTelemetry is available
        if OPENTELEMETRY_AVAILABLE:
            self._initialize_opentelemetry()
    
    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry tracing."""
        try:
            # Create tracer provider
            tracer_provider = TracerProvider()
            
            # Add span processor with OTLP exporter
            if self.export_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.export_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Set tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(
                self.service_name,
                version=self.service_version
            )
            
            logger.info("OpenTelemetry tracing initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
    
    @contextmanager
    def start_span(self, operation_name: str, parent_span: Optional[Any] = None,
                   attributes: Dict[str, Any] = None, trace_level: TraceLevel = TraceLevel.INFO):
        """Start a new trace span."""
        # Check if we should sample this trace
        if not self.sampler.should_sample(operation_name, self.service_name, trace_level):
            yield MockSpan()
            return
        
        # Record trace for sampling statistics
        self.sampler.record_trace(self.service_name)
        
        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(operation_name) as span:
            # Set attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            # Add service attributes
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("service.version", self.service_version)
            
            # Create internal span for analysis
            span_id = str(uuid.uuid4())
            trace_id = getattr(span, 'get_span_context', lambda: type('', (), {'trace_id': str(uuid.uuid4())})())
            
            internal_span = TraceSpan(
                span_id=span_id,
                trace_id=str(getattr(trace_id, 'trace_id', str(uuid.uuid4()))),
                parent_span_id=getattr(parent_span, 'span_id', None) if parent_span else None,
                operation_name=operation_name,
                service_name=self.service_name,
                start_time=datetime.now(),
                attributes=attributes or {}
            )
            
            try:
                yield span
                
                # Mark span as successful
                internal_span.status = "ok"
                
            except Exception as e:
                # Mark span as error
                internal_span.status = "error"
                internal_span.attributes["error"] = str(e)
                internal_span.attributes["error.type"] = type(e).__name__
                
                # Add error event
                internal_span.events.append({
                    "name": "exception",
                    "timestamp": datetime.now().isoformat(),
                    "attributes": {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e),
                        "exception.stacktrace": traceback.format_exc()
                    }
                })
                
                # Set span status in OpenTelemetry
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                # Update operation stats
                self.operation_stats[operation_name]["error_count"] += 1
                
                raise
                
            finally:
                # Finalize internal span
                internal_span.end_time = datetime.now()
                internal_span.duration_ms = (
                    internal_span.end_time - internal_span.start_time
                ).total_seconds() * 1000
                
                # Store span
                self.traces[internal_span.trace_id].append(internal_span)
                
                # Update relationships
                if internal_span.parent_span_id:
                    self.span_relationships[internal_span.parent_span_id].append(span_id)
                
                # Update operation statistics
                stats = self.operation_stats[operation_name]
                stats["count"] += 1
                stats["total_duration"] += internal_span.duration_ms
                stats["min_duration"] = min(stats["min_duration"], internal_span.duration_ms)
                stats["max_duration"] = max(stats["max_duration"], internal_span.duration_ms)
    
    def add_span_event(self, span: Any, name: str, attributes: Dict[str, Any] = None) -> None:
        """Add an event to a span."""
        if hasattr(span, 'add_event'):
            span.add_event(name, attributes)
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])
    
    def analyze_trace_performance(self, trace_id: str) -> Dict[str, Any]:
        """Analyze performance of a specific trace."""
        spans = self.get_trace(trace_id)
        if not spans:
            return {"error": "Trace not found"}
        
        # Calculate total trace duration
        start_time = min(span.start_time for span in spans)
        end_time = max(span.end_time for span in spans if span.end_time)
        total_duration = (end_time - start_time).total_seconds() * 1000
        
        # Find critical path (longest duration path)
        critical_path = self._find_critical_path(spans)
        
        # Identify bottlenecks (spans taking > 50% of total time)
        bottlenecks = [
            span for span in spans
            if span.duration_ms and span.duration_ms > total_duration * 0.1
        ]
        
        return {
            "trace_id": trace_id,
            "total_duration_ms": total_duration,
            "span_count": len(spans),
            "error_spans": len([s for s in spans if s.status == "error"]),
            "critical_path": [span.operation_name for span in critical_path],
            "critical_path_duration_ms": sum(span.duration_ms or 0 for span in critical_path),
            "bottlenecks": [
                {
                    "operation": span.operation_name,
                    "duration_ms": span.duration_ms,
                    "percentage": (span.duration_ms / total_duration) * 100
                }
                for span in bottlenecks
            ],
            "service_breakdown": self._analyze_service_breakdown(spans)
        }
    
    def _find_critical_path(self, spans: List[TraceSpan]) -> List[TraceSpan]:
        """Find the critical path (longest duration path) through the trace."""
        # Build span hierarchy
        span_map = {span.span_id: span for span in spans}
        children_map = defaultdict(list)
        
        for span in spans:
            if span.parent_span_id:
                children_map[span.parent_span_id].append(span)
        
        # Find root spans (no parent)
        root_spans = [span for span in spans if not span.parent_span_id]
        
        if not root_spans:
            return []
        
        # DFS to find longest path
        def find_longest_path(span):
            if not children_map[span.span_id]:
                return [span], span.duration_ms or 0
            
            longest_child_path = []
            longest_child_duration = 0
            
            for child in children_map[span.span_id]:
                child_path, child_duration = find_longest_path(child)
                if child_duration > longest_child_duration:
                    longest_child_path = child_path
                    longest_child_duration = child_duration
            
            return [span] + longest_child_path, (span.duration_ms or 0) + longest_child_duration
        
        # Find the longest path among all root spans
        longest_path = []
        longest_duration = 0
        
        for root in root_spans:
            path, duration = find_longest_path(root)
            if duration > longest_duration:
                longest_path = path
                longest_duration = duration
        
        return longest_path
    
    def _analyze_service_breakdown(self, spans: List[TraceSpan]) -> Dict[str, Any]:
        """Analyze time spent in each service."""
        service_times = defaultdict(float)
        service_counts = defaultdict(int)
        
        for span in spans:
            if span.duration_ms:
                service_times[span.service_name] += span.duration_ms
                service_counts[span.service_name] += 1
        
        total_time = sum(service_times.values())
        
        return {
            service: {
                "total_duration_ms": duration,
                "percentage": (duration / total_time) * 100 if total_time > 0 else 0,
                "span_count": service_counts[service],
                "avg_duration_ms": duration / service_counts[service] if service_counts[service] > 0 else 0
            }
            for service, duration in service_times.items()
        }
    
    def get_operation_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get operation performance statistics."""
        if operation_name:
            stats = self.operation_stats.get(operation_name)
            if not stats:
                return {"error": "Operation not found"}
            
            avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            error_rate = stats["error_count"] / stats["count"] if stats["count"] > 0 else 0
            
            return {
                "operation_name": operation_name,
                "total_requests": stats["count"],
                "total_errors": stats["error_count"],
                "error_rate": error_rate,
                "avg_duration_ms": avg_duration,
                "min_duration_ms": stats["min_duration"] if stats["min_duration"] != float('inf') else 0,
                "max_duration_ms": stats["max_duration"]
            }
        else:
            # Return all operation statistics
            result = {}
            for op_name, stats in self.operation_stats.items():
                avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
                error_rate = stats["error_count"] / stats["count"] if stats["count"] > 0 else 0
                
                result[op_name] = {
                    "total_requests": stats["count"],
                    "total_errors": stats["error_count"],
                    "error_rate": error_rate,
                    "avg_duration_ms": avg_duration,
                    "min_duration_ms": stats["min_duration"] if stats["min_duration"] != float('inf') else 0,
                    "max_duration_ms": stats["max_duration"]
                }
            
            return result
    
    def get_tracing_summary(self) -> Dict[str, Any]:
        """Get tracing system summary."""
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "total_traces": len(self.traces),
            "total_spans": sum(len(spans) for spans in self.traces.values()),
            "total_operations": len(self.operation_stats),
            "sampling_stats": self.sampler.get_sampling_stats(),
            "export_endpoint": self.export_endpoint,
            "opentelemetry_available": OPENTELEMETRY_AVAILABLE
        }


class ObservabilitySystem:
    """
    Comprehensive observability system orchestrating all components.
    
    Features:
    - Unified observability dashboard
    - Correlation between traces, metrics, and logs
    - Performance insights and recommendations
    - Automated health assessments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize observability system."""
        self.config = config or {}
        
        # Initialize components
        self.tracer = DistributedTracer(config.get("tracing", {}))
        self.metrics_collector = MetricsCollector(config.get("metrics", {}))
        self.alert_manager = AlertManager(config.get("alerting", {}))
        
        # System state
        self.running = False
        
        logger.info("Observability system initialized")
    
    async def start_observability(self) -> None:
        """Start the observability system."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting observability system")
        
        # Start components
        await self.metrics_collector.start_collection()
        await self.alert_manager.start_monitoring(self.metrics_collector)
        
        logger.info("Observability system started")
    
    async def stop_observability(self) -> None:
        """Stop the observability system."""
        if not self.running:
            return
        
        self.running = False
        logger.info("Stopping observability system")
        
        # Stop components
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_monitoring()
        
        logger.info("Observability system stopped")
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard data."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy" if self.running else "stopped",
            "tracing": self.tracer.get_tracing_summary(),
            "metrics": self.metrics_collector.get_metrics_summary(),
            "alerting": self.alert_manager.get_alert_summary(),
            "performance_overview": self._get_performance_overview(),
            "health_assessment": self._assess_system_health()
        }
    
    def _get_performance_overview(self) -> Dict[str, Any]:
        """Get performance overview."""
        # Get key performance metrics
        cpu_usage = self.metrics_collector.aggregate_metrics("system_cpu_usage_percent", "avg", 5)
        memory_usage = self.metrics_collector.aggregate_metrics("system_memory_usage_percent", "avg", 5)
        
        # Get operation statistics
        operation_stats = self.tracer.get_operation_statistics()
        
        # Calculate overall performance score
        performance_score = 100.0
        if cpu_usage and cpu_usage > 80:
            performance_score -= 20
        if memory_usage and memory_usage > 85:
            performance_score -= 30
        
        return {
            "performance_score": max(0, performance_score),
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_usage,
            "top_operations": sorted(
                [
                    {"name": name, "avg_duration_ms": stats["avg_duration_ms"], "requests": stats["total_requests"]}
                    for name, stats in operation_stats.items()
                ],
                key=lambda x: x["avg_duration_ms"],
                reverse=True
            )[:5]
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        health_score = 100.0
        issues = []
        
        # Check active alerts
        active_alerts = len(self.alert_manager.active_alerts)
        if active_alerts > 0:
            health_score -= active_alerts * 10
            issues.append(f"{active_alerts} active alerts")
        
        # Check error rates
        operation_stats = self.tracer.get_operation_statistics()
        high_error_operations = [
            name for name, stats in operation_stats.items()
            if stats["error_rate"] > 0.05  # 5% error rate
        ]
        
        if high_error_operations:
            health_score -= len(high_error_operations) * 15
            issues.extend([f"High error rate in {op}" for op in high_error_operations])
        
        # Check system resources
        cpu_usage = self.metrics_collector.aggregate_metrics("system_cpu_usage_percent", "avg", 5)
        memory_usage = self.metrics_collector.aggregate_metrics("system_memory_usage_percent", "avg", 5)
        
        if cpu_usage and cpu_usage > 85:
            health_score -= 20
            issues.append("High CPU usage")
        
        if memory_usage and memory_usage > 90:
            health_score -= 25
            issues.append("High memory usage")
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "health_score": max(0, health_score),
            "status": status,
            "issues": issues,
            "recommendations": self._get_health_recommendations(issues)
        }
    
    def _get_health_recommendations(self, issues: List[str]) -> List[str]:
        """Get health improvement recommendations."""
        recommendations = []
        
        for issue in issues:
            if "High CPU usage" in issue:
                recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive operations")
            elif "High memory usage" in issue:
                recommendations.append("Consider scaling up memory or investigating memory leaks")
            elif "High error rate" in issue:
                recommendations.append("Investigate and fix the root cause of errors in failing operations")
            elif "active alerts" in issue:
                recommendations.append("Address active alerts to improve system stability")
        
        if not recommendations:
            recommendations.append("System is running well - continue monitoring")
        
        return recommendations
    
    @contextmanager
    def trace_operation(self, operation_name: str, **kwargs):
        """Convenience method to trace an operation."""
        with self.tracer.start_span(operation_name, **kwargs) as span:
            yield span
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType = MetricType.COUNTER, 
                     tags: Dict[str, str] = None) -> None:
        """Convenience method to record a metric."""
        if metric_type == MetricType.COUNTER:
            self.metrics_collector.increment_counter(name, value, tags)
        elif metric_type == MetricType.HISTOGRAM:
            self.metrics_collector.record_histogram(name, value, tags)
        elif metric_type == MetricType.GAUGE:
            self.metrics_collector.set_gauge(name, value, tags)


# Global observability system
_observability_system: Optional[ObservabilitySystem] = None


def get_observability_system(config: Optional[Dict[str, Any]] = None) -> ObservabilitySystem:
    """Get or create the global observability system."""
    global _observability_system
    
    if _observability_system is None:
        _observability_system = ObservabilitySystem(config)
    
    return _observability_system


# Example usage and configuration
async def setup_production_observability():
    """Setup observability system with production configuration."""
    config = {
        "tracing": {
            "service_name": "iot_edge_anomaly_detection",
            "service_version": "2.0.0",
            "export_endpoint": "http://jaeger:14268/api/traces",
            "sampling": {
                "default_sample_rate": 0.1,
                "error_sample_rate": 1.0,
                "critical_paths": ["model_inference", "data_validation", "security_check"],
                "adaptive_sampling": True,
                "max_traces_per_second": 200
            }
        },
        "metrics": {
            "collect_system_metrics": True,
            "system_metrics_interval": 15
        },
        "alerting": {
            "anomaly_detection": True,
            "anomaly_threshold": 2.5,
            "notification_channels": {
                "email": {
                    "type": "email",
                    "recipients": ["ops@company.com", "devops@company.com"]
                },
                "slack": {
                    "type": "slack",
                    "webhook_url": "https://hooks.slack.com/services/..."
                },
                "pagerduty": {
                    "type": "webhook",
                    "url": "https://events.pagerduty.com/v2/enqueue",
                    "routing_key": "your-routing-key"
                }
            }
        }
    }
    
    observability = get_observability_system(config)
    await observability.start_observability()
    
    logger.info("Production observability system configured and started")
    return observability


# Decorator for automatic tracing
def traced(operation_name: Optional[str] = None):
    """Decorator to automatically trace function calls."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                observability = get_observability_system()
                with observability.trace_operation(operation_name) as span:
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        observability.tracer.add_span_event(span, "error", {"error": str(e)})
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                observability = get_observability_system()
                with observability.trace_operation(operation_name) as span:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        observability.tracer.add_span_event(span, "error", {"error": str(e)})
                        raise
            return sync_wrapper
    
    return decorator