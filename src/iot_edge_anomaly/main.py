"""
Main entry point for IoT Edge Anomaly Detection application.
"""
import sys
import time
import logging
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .models.lstm_autoencoder import LSTMAutoencoder
from .monitoring.metrics_exporter import MetricsExporter

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
        self.model.eval()  # Set to evaluation mode
        
        # Initialize metrics exporter
        monitoring_config = self.config.get('monitoring', {})
        self.metrics_exporter = MetricsExporter(monitoring_config)
        
        logger.info("Components initialized successfully")
    
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
        
        with torch.no_grad():
            # Get reconstruction and error
            reconstruction_error = self.model.compute_reconstruction_error(
                sensor_data, reduction='mean'
            ).item()
            
            # Determine if anomaly
            is_anomaly = reconstruction_error > self.anomaly_threshold
            
            # Record metrics
            inference_time = time.time() - start_time
            self.metrics_exporter.record_inference_time(inference_time)
            self.metrics_exporter.record_reconstruction_error(reconstruction_error)
            
            if is_anomaly:
                self.metrics_exporter.increment_anomaly_count(1)
                logger.warning(f"ANOMALY DETECTED! Error: {reconstruction_error:.4f}")
            else:
                logger.debug(f"Normal operation. Error: {reconstruction_error:.4f}")
        
        return {
            'reconstruction_error': reconstruction_error,
            'is_anomaly': is_anomaly,
            'timestamp': datetime.now().isoformat(),
            'inference_time': inference_time
        }
    
    def run_processing_loop(self):
        """Run the main processing loop."""
        logger.info("Starting processing loop...")
        self.running = True
        iteration_count = 0
        
        try:
            while self.running:
                # Check iteration limit (for testing)
                if self.max_iterations and iteration_count >= self.max_iterations:
                    logger.info(f"Reached max iterations: {self.max_iterations}")
                    break
                
                # Get and process sensor data
                sensor_data = self.get_sensor_data()
                result = self.process_sensor_data(sensor_data)
                
                # Record processed samples
                self.metrics_exporter.record_processed_samples(sensor_data.size(0))
                
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
            logger.error(f"Error in processing loop: {e}")
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
        except FileNotFoundError:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = {
                'model': {
                    'input_size': 5,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'sequence_length': 20
                },
                'monitoring': {
                    'otlp_endpoint': 'http://localhost:4317',
                    'service_name': 'iot-edge-anomaly'
                },
                'anomaly_threshold': 0.5,
                'processing': {
                    'loop_interval': 5.0
                }
            }
        
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