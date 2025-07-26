"""
OTLP Metrics Exporter for IoT Edge Anomaly Detection.

This module provides OpenTelemetry-based metrics export functionality
for monitoring anomaly detection performance, system resources, and
operational metrics on IoT edge devices.
"""
import time
import threading
import psutil
from typing import Dict, Any, Optional
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource


class MetricsExporter:
    """
    OTLP Metrics Exporter for IoT edge anomaly detection system.
    
    Provides functionality to:
    - Export anomaly counts and detection metrics
    - Monitor system resource usage (CPU, memory, disk)
    - Track model performance metrics
    - Send metrics to observability stack via OTLP
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - otlp_endpoint: OTLP receiver endpoint URL
            - service_name: Service name for telemetry
            - service_version: Service version (optional)
            - otlp_headers: Additional headers for OTLP export (optional)
            - export_interval: Metrics export interval in seconds (default: 30)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service_name = config.get('service_name', 'iot-edge-anomaly')
        self.service_version = config.get('service_version', '0.1.0')
        self.export_interval = config.get('export_interval', 30)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metric storage
        self._anomaly_count = 0
        self._cpu_usage = 0.0
        self._memory_usage = 0.0
        self._disk_usage = 0.0
        self._inference_times = []
        self._reconstruction_errors = []
        self._processed_samples = 0
        
        # Initialize OpenTelemetry
        self._setup_opentelemetry()
        
        # Create meters and instruments
        self._setup_instruments()
    
    def _setup_opentelemetry(self):
        """Initialize OpenTelemetry metrics provider and exporter."""
        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": self.service_version,
            "deployment.environment": "production"
        })
        
        # Configure OTLP exporter
        otlp_exporter = OTLPMetricExporter(
            endpoint=self.config['otlp_endpoint'],
            headers=self.config.get('otlp_headers', {}),
            insecure=True  # For edge devices, often using internal networks
        )
        
        # Create metric reader with periodic export
        reader = PeriodicExportingMetricReader(
            exporter=otlp_exporter,
            export_interval_millis=self.export_interval * 1000,
        )
        
        # Create and set meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader]
        )
        
        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)
        
        # Get meter for this service
        self.meter = metrics.get_meter(
            name=self.service_name,
            version=self.service_version
        )
    
    def _setup_instruments(self):
        """Create OpenTelemetry metric instruments."""
        # Anomaly detection metrics
        self.anomaly_counter = self.meter.create_counter(
            name="anomalies_detected_total",
            description="Total number of anomalies detected",
            unit="1"
        )
        
        self.anomaly_gauge = self.meter.create_up_down_counter(
            name="anomalies_current_count",
            description="Current anomaly count",
            unit="1"
        )
        
        # System resource metrics (using observable gauges)
        self.cpu_gauge = self.meter.create_observable_gauge(
            name="system_cpu_usage_percent",
            description="Current CPU usage percentage",
            unit="%"
        )
        
        self.memory_gauge = self.meter.create_observable_gauge(
            name="system_memory_usage_percent", 
            description="Current memory usage percentage",
            unit="%"
        )
        
        self.disk_gauge = self.meter.create_observable_gauge(
            name="system_disk_usage_percent",
            description="Current disk usage percentage", 
            unit="%"
        )
        
        # Model performance metrics
        self.inference_time_histogram = self.meter.create_histogram(
            name="model_inference_time_seconds",
            description="Model inference time distribution",
            unit="s"
        )
        
        self.reconstruction_error_histogram = self.meter.create_histogram(
            name="model_reconstruction_error",
            description="Model reconstruction error distribution",
            unit="1"
        )
        
        self.samples_counter = self.meter.create_counter(
            name="samples_processed_total",
            description="Total number of samples processed",
            unit="1"
        )
    
    def record_anomaly_count(self, count: int):
        """Record current anomaly count."""
        with self._lock:
            self._anomaly_count = count
            self.anomaly_gauge.add(count)
    
    def increment_anomaly_count(self, increment: int = 1):
        """Increment anomaly count by specified amount."""
        with self._lock:
            self._anomaly_count += increment
            self.anomaly_counter.add(increment)
            self.anomaly_gauge.add(increment)
    
    def get_anomaly_count(self) -> int:
        """Get current anomaly count."""
        with self._lock:
            return self._anomaly_count
    
    def record_cpu_usage(self, usage_percent: float):
        """Record current CPU usage percentage."""
        with self._lock:
            self._cpu_usage = usage_percent
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        with self._lock:
            return self._cpu_usage
    
    def record_memory_usage(self, usage_percent: float):
        """Record current memory usage percentage."""
        with self._lock:
            self._memory_usage = usage_percent
    
    def get_memory_usage(self) -> float:
        """Get current memory usage."""
        with self._lock:
            return self._memory_usage
    
    def record_disk_usage(self, usage_percent: float):
        """Record current disk usage percentage."""
        with self._lock:
            self._disk_usage = usage_percent
    
    def get_disk_usage(self) -> float:
        """Get current disk usage."""
        with self._lock:
            return self._disk_usage
    
    def record_inference_time(self, time_seconds: float):
        """Record model inference time."""
        with self._lock:
            self._inference_times.append(time_seconds)
            # Keep only recent measurements for memory efficiency
            if len(self._inference_times) > 1000:
                self._inference_times = self._inference_times[-500:]
            
            self.inference_time_histogram.record(time_seconds)
    
    def get_average_inference_time(self) -> float:
        """Get average inference time from recent measurements."""
        with self._lock:
            if not self._inference_times:
                return 0.0
            return sum(self._inference_times) / len(self._inference_times)
    
    def record_reconstruction_error(self, error: float):
        """Record model reconstruction error."""
        with self._lock:
            self._reconstruction_errors.append(error)
            # Keep only recent measurements
            if len(self._reconstruction_errors) > 1000:
                self._reconstruction_errors = self._reconstruction_errors[-500:]
                
            self.reconstruction_error_histogram.record(error)
    
    def record_processed_samples(self, count: int):
        """Record number of processed samples."""
        with self._lock:
            self._processed_samples += count
            self.samples_counter.add(count)
    
    def collect_system_metrics(self):
        """Automatically collect current system metrics."""
        try:
            # Get CPU usage (non-blocking)
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_cpu_usage(cpu_percent)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.record_memory_usage(memory.percent)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_disk_usage(disk_percent)
            
        except Exception as e:
            # Log but don't fail on metrics collection errors
            print(f"Warning: Failed to collect system metrics: {e}")
    
    def export_metrics(self):
        """Manually trigger metrics export."""
        try:
            # Collect current system metrics before export
            self.collect_system_metrics()
            
            # Force export through meter provider
            self.meter_provider.force_flush(timeout_millis=5000)
            
        except Exception as e:
            # Handle export failures gracefully
            if "invalid-url" in str(e) or "Connection" in str(e):
                raise ConnectionError(f"Failed to connect to OTLP endpoint: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure metrics are exported."""
        try:
            self.export_metrics()
        except Exception:
            pass  # Don't raise exceptions during cleanup
        
        # Shutdown meter provider
        try:
            self.meter_provider.shutdown()
        except Exception:
            pass