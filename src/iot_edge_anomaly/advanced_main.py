"""
Advanced main entry point for IoT Edge Anomaly Detection with all 5 AI algorithms.
Supports the advanced ensemble system with breakthrough AI capabilities.
"""
import sys
import time
import logging
import argparse
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Enhanced imports for advanced features
from .main import IoTAnomalyDetectionApp, load_config
from .models.advanced_hybrid_integration import create_advanced_hybrid_system, AdvancedEnsemblePredictor
from .models.transformer_vae import TransformerVAE
from .models.sparse_graph_attention import SparseGraphAttentionNetwork
from .models.physics_informed_hybrid import PhysicsInformedHybrid
from .models.self_supervised_registration import SelfSupervisedRegistration
from .models.federated_learning import FederatedLearningClient
from .async_processor import AsyncBatchProcessor, StreamProcessor
from .monitoring.advanced_metrics import AdvancedMetricsCollector
from .security.secure_inference import SecureInferenceEngine
from .validation.model_validator import AdvancedModelValidator

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
)

logger = logging.getLogger(__name__)


class AdvancedIoTAnomalyDetectionApp(IoTAnomalyDetectionApp):
    """
    Advanced IoT Edge Anomaly Detection with 5 Breakthrough AI Algorithms.
    
    Features:
    - Transformer-VAE Temporal Modeling
    - Sparse Graph Attention Networks  
    - Physics-Informed Neural Networks
    - Self-Supervised Registration Learning
    - Privacy-Preserving Federated Learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Advanced configuration
        self.advanced_config = config.get('advanced_models', {})
        self.enable_all_models = self.advanced_config.get('enable_all_models', False)
        self.ensemble_method = self.advanced_config.get('ensemble_method', 'dynamic_weighting')
        self.uncertainty_quantification = self.advanced_config.get('uncertainty_quantification', True)
        
        # Advanced components
        self.ensemble_predictor: Optional[AdvancedEnsemblePredictor] = None
        self.async_processor: Optional[AsyncBatchProcessor] = None
        self.stream_processor: Optional[StreamProcessor] = None
        self.advanced_metrics: Optional[AdvancedMetricsCollector] = None
        self.secure_inference: Optional[SecureInferenceEngine] = None
        self.model_validator: Optional[AdvancedModelValidator] = None
        
        # Model registry for advanced algorithms
        self.model_registry = {}
        
        logger.info(f"Initialized Advanced IoT Anomaly Detection App with {self._count_enabled_models()} advanced models")
    
    def _count_enabled_models(self) -> int:
        """Count how many advanced models are enabled."""
        enabled_models = [
            self.advanced_config.get('enable_transformer_vae', False),
            self.advanced_config.get('enable_sparse_gat', False),
            self.advanced_config.get('enable_physics_informed', False),
            self.advanced_config.get('enable_self_supervised', False),
            self.advanced_config.get('enable_federated_learning', False),
        ]
        return sum(enabled_models)
    
    def initialize_advanced_components(self):
        """Initialize advanced AI models and ensemble system."""
        logger.info("Initializing advanced AI components...")
        
        # Initialize individual advanced models
        if self.advanced_config.get('enable_transformer_vae', False) or self.enable_all_models:
            logger.info("Initializing Transformer-VAE model...")
            self.model_registry['transformer_vae'] = TransformerVAE(
                input_dim=self.config.get('model', {}).get('input_size', 5),
                hidden_dim=self.config.get('model', {}).get('hidden_size', 64),
                latent_dim=32
            )
        
        if self.advanced_config.get('enable_sparse_gat', False) or self.enable_all_models:
            logger.info("Initializing Sparse Graph Attention Network...")
            self.model_registry['sparse_gat'] = SparseGraphAttentionNetwork(
                in_channels=self.config.get('model', {}).get('input_size', 5),
                hidden_channels=64,
                out_channels=32,
                num_heads=8
            )
        
        if self.advanced_config.get('enable_physics_informed', False) or self.enable_all_models:
            logger.info("Initializing Physics-Informed Hybrid model...")
            self.model_registry['physics_informed'] = PhysicsInformedHybrid(
                input_size=self.config.get('model', {}).get('input_size', 5),
                hidden_size=self.config.get('model', {}).get('hidden_size', 64)
            )
        
        if self.advanced_config.get('enable_self_supervised', False) or self.enable_all_models:
            logger.info("Initializing Self-Supervised Registration model...")
            self.model_registry['self_supervised'] = SelfSupervisedRegistration(
                input_dim=self.config.get('model', {}).get('input_size', 5),
                hidden_dim=self.config.get('model', {}).get('hidden_size', 64)
            )
        
        if self.advanced_config.get('enable_federated_learning', False):
            logger.info("Initializing Federated Learning client...")
            self.model_registry['federated'] = FederatedLearningClient(
                model_config=self.config.get('model', {}),
                client_id=self.config.get('federated', {}).get('client_id', 'default_client')
            )
        
        # Create advanced ensemble system
        if self.model_registry:
            logger.info(f"Creating advanced ensemble with {len(self.model_registry)} models...")
            self.ensemble_predictor = create_advanced_hybrid_system({
                'models': self.model_registry,
                'ensemble_method': self.ensemble_method,
                'uncertainty_quantification': self.uncertainty_quantification,
                **self.advanced_config
            })
        
        # Initialize advanced processors
        if self.config.get('processing', {}).get('enable_async_processing', False):
            self.async_processor = AsyncBatchProcessor(
                batch_size=self.config.get('processing', {}).get('batch_size', 32),
                max_latency_ms=self.config.get('processing', {}).get('max_latency_ms', 100)
            )
        
        if self.config.get('processing', {}).get('enable_stream_processing', False):
            self.stream_processor = StreamProcessor(
                window_size=self.config.get('processing', {}).get('window_size', 100),
                overlap=self.config.get('processing', {}).get('overlap', 0.5)
            )
        
        # Initialize advanced monitoring
        self.advanced_metrics = AdvancedMetricsCollector(
            enable_model_explanations=self.config.get('monitoring', {}).get('enable_model_explanations', False),
            enable_uncertainty_tracking=self.config.get('monitoring', {}).get('enable_uncertainty_tracking', False)
        )
        
        # Initialize secure inference engine
        self.secure_inference = SecureInferenceEngine(
            enable_differential_privacy=self.config.get('security', {}).get('enable_differential_privacy', False),
            privacy_epsilon=self.config.get('security', {}).get('privacy_epsilon', 1.0)
        )
        
        # Initialize model validator
        self.model_validator = AdvancedModelValidator(
            enable_drift_detection=self.config.get('validation', {}).get('enable_drift_detection', False),
            enable_adversarial_detection=self.config.get('validation', {}).get('enable_adversarial_detection', False)
        )
        
        logger.info("Advanced AI components initialized successfully")
    
    def initialize_components(self):
        """Initialize both standard and advanced components."""
        super().initialize_components()
        self.initialize_advanced_components()
    
    async def process_sensor_data_advanced(self, sensor_data, edge_index=None, sensor_metadata=None):
        """
        Advanced sensor data processing with ensemble prediction and uncertainty quantification.
        
        Args:
            sensor_data: Tensor of shape (batch_size, seq_len, input_size)
            edge_index: Graph edge indices for GNN models (optional)
            sensor_metadata: Additional sensor metadata (optional)
            
        Returns:
            Dict containing advanced processing results with explanations and uncertainty
        """
        start_time = time.time()
        
        try:
            if self.ensemble_predictor is None:
                # Fallback to basic processing
                return super().process_sensor_data(sensor_data)
            
            # Validate input data
            if self.model_validator:
                validation_result = await self.model_validator.validate_input_async(
                    sensor_data, edge_index, sensor_metadata
                )
                if not validation_result.is_valid:
                    raise ValueError(f"Input validation failed: {validation_result.error_message}")
            
            # Process with secure inference if enabled
            if self.secure_inference:
                sensor_data = await self.secure_inference.apply_privacy_protection(sensor_data)
            
            # Advanced ensemble prediction
            prediction_result = await self.ensemble_predictor.predict_async(
                sensor_data=sensor_data,
                edge_index=edge_index,
                sensor_metadata=sensor_metadata,
                return_explanations=True,
                return_uncertainty=self.uncertainty_quantification
            )
            
            # Extract results
            reconstruction_error = prediction_result['reconstruction_error']
            uncertainty = prediction_result.get('uncertainty', 0.0)
            explanations = prediction_result.get('explanations', {})
            model_contributions = prediction_result.get('model_contributions', {})
            
            # Multi-level anomaly classification
            threshold_levels = self.config.get('multi_threshold_levels', {
                'low_anomaly': 0.3,
                'medium_anomaly': 0.6, 
                'high_anomaly': 0.9
            })
            
            anomaly_level = self._classify_anomaly_level(reconstruction_error, threshold_levels)
            is_anomaly = anomaly_level != 'normal'
            
            # Record advanced metrics
            if self.advanced_metrics:
                await self.advanced_metrics.record_prediction_async(
                    reconstruction_error=reconstruction_error,
                    uncertainty=uncertainty,
                    anomaly_level=anomaly_level,
                    model_contributions=model_contributions,
                    explanations=explanations
                )
            
            inference_time = time.time() - start_time
            
            if is_anomaly:
                logger.warning(f"ADVANCED ANOMALY DETECTED! Level: {anomaly_level}, Error: {reconstruction_error:.4f}, Uncertainty: {uncertainty:.4f}")
            else:
                logger.debug(f"Normal operation. Error: {reconstruction_error:.4f}, Uncertainty: {uncertainty:.4f}")
            
            return {
                'reconstruction_error': reconstruction_error,
                'is_anomaly': is_anomaly,
                'anomaly_level': anomaly_level,
                'uncertainty': uncertainty,
                'explanations': explanations,
                'model_contributions': model_contributions,
                'timestamp': datetime.now().isoformat(),
                'inference_time': inference_time,
                'status': 'success',
                'processing_mode': 'advanced_ensemble'
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Advanced processing error: {e}")
            
            return {
                'reconstruction_error': float('inf'),
                'is_anomaly': True,
                'anomaly_level': 'critical_error',
                'uncertainty': 1.0,
                'timestamp': datetime.now().isoformat(),
                'inference_time': inference_time,
                'status': 'error',
                'error_message': str(e),
                'processing_mode': 'advanced_ensemble_error'
            }
    
    def _classify_anomaly_level(self, error: float, thresholds: Dict[str, float]) -> str:
        """Classify anomaly severity level based on reconstruction error."""
        if error >= thresholds.get('high_anomaly', 0.9):
            return 'high_anomaly'
        elif error >= thresholds.get('medium_anomaly', 0.6):
            return 'medium_anomaly'
        elif error >= thresholds.get('low_anomaly', 0.3):
            return 'low_anomaly'
        else:
            return 'normal'
    
    async def run_advanced_processing_loop(self):
        """Run advanced processing loop with async capabilities."""
        logger.info("Starting advanced processing loop...")
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
                if current_time - last_health_check > 60:
                    health_status = self.get_health_status()
                    logger.info(f"Advanced health status: {health_status.get('system', {}).get('status', 'unknown')}")
                    last_health_check = current_time
                
                # Get sensor data with optional graph structure
                try:
                    sensor_data = self.get_sensor_data()
                    edge_index = self._generate_sensor_graph() if self.model_registry.get('sparse_gat') else None
                    sensor_metadata = self._get_sensor_metadata()
                    
                    # Process with advanced ensemble
                    if self.async_processor:
                        result = await self.async_processor.process_batch([sensor_data])
                    else:
                        result = await self.process_sensor_data_advanced(
                            sensor_data, edge_index, sensor_metadata
                        )
                    
                    # Record processed samples on success
                    if isinstance(result, dict) and result.get('status') == 'success':
                        if self.metrics_exporter:
                            self.metrics_exporter.record_processed_samples(sensor_data.size(0))
                    elif isinstance(result, list) and len(result) > 0:
                        successful_results = [r for r in result if r.get('status') == 'success']
                        if successful_results and self.metrics_exporter:
                            self.metrics_exporter.record_processed_samples(len(successful_results))
                        
                except Exception as e:
                    logger.error(f"Failed to process advanced sensor data: {e}")
                
                # Export advanced metrics periodically
                if iteration_count % 10 == 0:
                    try:
                        if self.metrics_exporter:
                            self.metrics_exporter.export_metrics()
                        if self.advanced_metrics:
                            await self.advanced_metrics.export_advanced_metrics()
                    except Exception as e:
                        logger.warning(f"Failed to export advanced metrics: {e}")
                
                iteration_count += 1
                await asyncio.sleep(self.loop_interval)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Critical error in advanced processing loop: {e}")
            raise
        finally:
            self.running = False
            logger.info("Advanced processing loop stopped")
    
    def _generate_sensor_graph(self):
        """Generate a simple sensor graph topology for GNN models."""
        import torch
        # Simple ring topology for demonstration
        num_sensors = self.config.get('model', {}).get('input_size', 5)
        edge_list = []
        for i in range(num_sensors):
            # Connect to next sensor (ring topology)
            edge_list.append([i, (i + 1) % num_sensors])
            edge_list.append([(i + 1) % num_sensors, i])  # bidirectional
        
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _get_sensor_metadata(self) -> Dict[str, Any]:
        """Get sensor metadata for enhanced processing."""
        return {
            'timestamp': datetime.now().isoformat(),
            'sensor_count': self.config.get('model', {}).get('input_size', 5),
            'sampling_rate': self.config.get('processing', {}).get('sampling_rate', 1.0),
            'location': 'edge_device_01'
        }


def main():
    """Advanced main entry point with support for all 5 AI algorithms."""
    logger.info("Starting Advanced IoT Edge Anomaly Detection v4.0.0")
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Advanced IoT Edge Anomaly Detection')
        parser.add_argument('--config', default='config/advanced_ensemble.yaml',
                          help='Path to configuration file')
        parser.add_argument('--enable-all-models', action='store_true',
                          help='Enable all 5 advanced AI models')
        args = parser.parse_args()
        
        # Load configuration
        try:
            config = load_config(args.config)
            logger.info(f"Loaded advanced configuration from {args.config}")
        except FileNotFoundError:
            logger.warning(f"Config file {args.config} not found, using advanced defaults")
            config = {
                'advanced_models': {
                    'enable_all_models': args.enable_all_models,
                    'enable_transformer_vae': True,
                    'enable_sparse_gat': True,
                    'enable_physics_informed': True,
                    'enable_self_supervised': True,
                    'ensemble_method': 'dynamic_weighting',
                    'uncertainty_quantification': True
                }
            }
        
        # Override with command line options
        if args.enable_all_models:
            config.setdefault('advanced_models', {})['enable_all_models'] = True
        
        # Initialize and run advanced application
        app = AdvancedIoTAnomalyDetectionApp(config)
        app.initialize_components()
        
        # Run async processing loop
        asyncio.run(app.run_advanced_processing_loop())
        
        return 0
        
    except Exception as e:
        logger.error(f"Advanced application failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())