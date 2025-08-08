"""
Main entry point for IoT Edge Anomaly Detection application.
"""
import sys
import time
import logging
import argparse
import yaml
import torch
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .models.lstm_autoencoder import LSTMAutoencoder
from .monitoring.metrics_exporter import MetricsExporter
from .health import SystemHealthMonitor, ModelHealthMonitor
from .config_validator import validate_config, apply_config_defaults
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, circuit_breaker_registry
from .performance_optimizer import performance_monitor
from .async_processor import async_processor, stream_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


class IoTAnomalyDetectionApp:
    """
    Main IoT Edge Anomaly Detection Application.
    
    Orchestrates the LSTM autoencoder model, data processing, 
    and monitoring for real-time anomaly detection on IoT edge devices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.metrics_exporter = None
        self.running = False
        
        # Configuration defaults
        self.anomaly_threshold = config.get('anomaly_threshold', 0.5)
        self.loop_interval = config.get('processing', {}).get('loop_interval', 5.0)
        self.max_iterations = config.get('processing', {}).get('max_iterations', None)
        
        # Health monitoring
        self.system_health = SystemHealthMonitor(
            memory_threshold_mb=config.get('health', {}).get('memory_threshold_mb', 100),
            cpu_threshold_percent=config.get('health', {}).get('cpu_threshold_percent', 80)
        )
        self.model_health = ModelHealthMonitor()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize circuit breakers
        self.inference_breaker = CircuitBreaker(
            "model_inference",
            CircuitBreakerConfig(
                failure_threshold=config.get('circuit_breaker', {}).get('failure_threshold', 5),
                recovery_timeout=config.get('circuit_breaker', {}).get('recovery_timeout', 60.0),
                timeout=config.get('circuit_breaker', {}).get('timeout', 10.0)
            )
        )
        self.metrics_breaker = CircuitBreaker(
            "metrics_export", 
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        )
        
        # Register circuit breakers
        circuit_breaker_registry.register(self.inference_breaker)
        circuit_breaker_registry.register(self.metrics_breaker)
        
        logger.info(f"Initialized IoT Anomaly Detection App with threshold={self.anomaly_threshold}")
    
    def initialize_components(self):
        """Initialize ML model and monitoring components."""
        logger.info("Initializing components...")
        
        # Initialize LSTM autoencoder
        model_config = self.config.get('model', {})
        self.model = LSTMAutoencoder(
            input_size=model_config.get('input_size', 5),
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Optimize model for inference performance
        enable_optimization = self.config.get('performance', {}).get('enable_optimization', True)
        if enable_optimization:
            logger.info("Optimizing model for edge inference...")
            self.model = performance_monitor.optimize_model_for_inference(self.model)
        else:
            self.model.eval()  # Set to evaluation mode
        
        # Initialize metrics exporter
        monitoring_config = self.config.get('monitoring', {})
        self.metrics_exporter = MetricsExporter(monitoring_config)
        
        logger.info("Components initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def get_sensor_data(self) -> torch.Tensor:
        """
        Get sensor data from IoT devices.
        
        In production, this would connect to actual IoT sensors.
        For now, returns simulated data.
        """
        # Simulate sensor data - in production this would come from actual sensors
        batch_size = 1
        seq_len = self.config.get('model', {}).get('sequence_length', 20)
        input_size = self.config.get('model', {}).get('input_size', 5)
        
        # Generate realistic sensor-like data with some noise
        base_values = torch.tensor([20.0, 50.0, 1.0, 0.5, 100.0])  # Typical sensor readings
        noise = torch.randn(batch_size, seq_len, input_size) * 0.1
        data = base_values.unsqueeze(0).unsqueeze(0) + noise
        
        return data
    
    def process_sensor_data(self, sensor_data: torch.Tensor) -> Dict[str, Any]:
        """
        Process sensor data through the anomaly detection model.
        
        Args:
            sensor_data: Tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dict containing processing results
        """
        start_time = time.time()
        error_occurred = False
        
        try:
            # Validate input data
            if sensor_data.isnan().any():
                raise ValueError("Input data contains NaN values")
            if sensor_data.isinf().any():
                raise ValueError("Input data contains infinite values")
            
            # Use circuit breaker for model inference with performance monitoring
            @performance_monitor.monitor_operation("model_inference")
            def _model_inference():
                with torch.no_grad():
                    return self.model.compute_reconstruction_error(sensor_data, reduction='mean').item()
            
            reconstruction_error = self.inference_breaker.call(_model_inference)
            
            # Validate reconstruction error
            if not (0 <= reconstruction_error < float('inf')):
                raise ValueError(f"Invalid reconstruction error: {reconstruction_error}")
            
            # Determine if anomaly
            is_anomaly = reconstruction_error > self.anomaly_threshold
            
            # Record metrics with circuit breaker
            inference_time = time.time() - start_time
            
            def _record_metrics():
                self.metrics_exporter.record_inference_time(inference_time)
                self.metrics_exporter.record_reconstruction_error(reconstruction_error)
                if is_anomaly:
                    self.metrics_exporter.increment_anomaly_count(1)
                    
            try:
                self.metrics_breaker.call(_record_metrics)
            except CircuitBreakerError:
                logger.warning("Metrics recording circuit breaker is open")
            
            self.model_health.record_inference(inference_time, error_occurred)
            
            if is_anomaly:
                logger.warning(f"ANOMALY DETECTED! Error: {reconstruction_error:.4f}")
            else:
                logger.debug(f"Normal operation. Error: {reconstruction_error:.4f}")
            
            return {
                'reconstruction_error': reconstruction_error,
                'is_anomaly': is_anomaly,
                'timestamp': datetime.now().isoformat(),
                'inference_time': inference_time,
                'status': 'success'
            }
            
        except Exception as e:
            error_occurred = True
            inference_time = time.time() - start_time
            self.model_health.record_inference(inference_time, error_occurred)
            
            logger.error(f"Error processing sensor data: {e}")
            return {
                'reconstruction_error': float('inf'),
                'is_anomaly': True,
                'timestamp': datetime.now().isoformat(),
                'inference_time': inference_time,
                'status': 'error',
                'error_message': str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        try:
            system_health = self.system_health.get_comprehensive_health()
            model_performance = self.model_health.check_inference_performance()
            model_staleness = self.model_health.check_model_staleness()
            
            return {
                "system": system_health,
                "model": {
                    "performance": {
                        "status": model_performance.status.value,
                        "message": model_performance.message,
                        "details": model_performance.details or {}
                    },
                    "staleness": {
                        "status": model_staleness.status.value,
                        "message": model_staleness.message,
                        "details": model_staleness.details or {}
                    }
                },
                "circuit_breakers": circuit_breaker_registry.get_all_stats(),
                "performance": performance_monitor.get_performance_report()
            }
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def run_processing_loop(self):
        """Run the main processing loop."""
        logger.info("Starting processing loop...")
        self.running = True
        iteration_count = 0
        last_health_check = 0
        
        try:
            while self.running:
                # Check iteration limit (for testing)
                if self.max_iterations and iteration_count >= self.max_iterations:
                    logger.info(f"Reached max iterations: {self.max_iterations}")
                    break
                
                # Periodic health checks
                current_time = time.time()
                if current_time - last_health_check > 60:  # Every minute
                    health_status = self.get_health_status()
                    logger.info(f"Health status: {health_status['system']['status']}")
                    last_health_check = current_time
                
                # Get and process sensor data
                try:
                    sensor_data = self.get_sensor_data()
                    result = self.process_sensor_data(sensor_data)
                    
                    # Record processed samples only on success
                    if result.get('status') == 'success':
                        self.metrics_exporter.record_processed_samples(sensor_data.size(0))
                        
                except Exception as e:
                    logger.error(f"Failed to process sensor data: {e}")
                    # Continue processing - don't break the loop on single failures
                
                # Export metrics periodically
                if iteration_count % 10 == 0:
                    try:
                        self.metrics_exporter.export_metrics()
                    except Exception as e:
                        logger.warning(f"Failed to export metrics: {e}")
                
                iteration_count += 1
                
                # Sleep before next iteration
                time.sleep(self.loop_interval)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Critical error in processing loop: {e}")
            # Log health status on critical errors
            try:
                health_status = self.get_health_status()
                logger.error(f"Health status at error: {health_status}")
            except:
                pass
            raise
        finally:
            self.running = False
            logger.info("Processing loop stopped")
    
    def shutdown(self):
        """Gracefully shutdown the application."""
        logger.info("Shutting down application...")
        self.running = False
        
        # Final metrics export
        if self.metrics_exporter:
            try:
                self.metrics_exporter.export_metrics()
            except Exception as e:
                logger.warning(f"Failed final metrics export: {e}")
        
        logger.info("Application shutdown complete")


def main():
    """Main application entry point."""
    logger.info("Starting IoT Edge Anomaly Detection v0.1.0")
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='IoT Edge Anomaly Detection')
        parser.add_argument('--config', default='config/default.yaml',
                          help='Path to configuration file')
        args = parser.parse_args()
        
        # Load configuration (with defaults if file doesn't exist)
        try:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = {}
        
        # Apply defaults and validate configuration
        config = apply_config_defaults(config)
        validation_issues = validate_config(config)
        
        # Exit if there are configuration errors
        error_count = len([issue for issue in validation_issues 
                          if issue.severity.value == 'error'])
        if error_count > 0:
            logger.error(f"Configuration has {error_count} error(s). Cannot start application.")
            return 1
        
        # Initialize and run application
        app = IoTAnomalyDetectionApp(config)
        app.initialize_components()
        app.run_processing_loop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())