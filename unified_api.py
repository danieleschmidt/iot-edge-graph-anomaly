#!/usr/bin/env python3
"""
🔮 Terragon Unified API - Single Interface for All AI Models

This module provides a unified, production-ready API that seamlessly integrates
all 5 breakthrough AI algorithms into a single, easy-to-use interface.

Perfect for:
- Production deployments
- Research integration
- Real-time anomaly detection
- Edge device deployment
- Federated learning coordination

Usage:
    from unified_api import TerrragonAnomalyDetector
    
    detector = TerrragonAnomalyDetector(config='production')
    result = detector.detect_anomaly(sensor_data)
"""

import os
import sys
import time
import yaml
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available AI model types."""
    TRANSFORMER_VAE = "transformer_vae"
    SPARSE_GAT = "sparse_gat"
    PHYSICS_INFORMED = "physics_informed"
    SELF_SUPERVISED = "self_supervised"
    QUANTUM_CLASSICAL = "quantum_classical"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ENSEMBLE = "ensemble"


class DeploymentMode(Enum):
    """Deployment mode configurations."""
    RESEARCH = "research"          # Full models, maximum accuracy
    PRODUCTION = "production"      # Optimized for reliability
    EDGE = "edge"                 # Optimized for resource constraints
    REALTIME = "realtime"         # Optimized for latency
    FEDERATED = "federated"       # Distributed learning setup


@dataclass
class AnomalyResult:
    """Standardized anomaly detection result."""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    timestamp: str
    inference_time_ms: float
    model_used: str
    details: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    uncertainty: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SystemHealth:
    """System health status."""
    status: str  # 'healthy', 'degraded', 'critical'
    cpu_usage: float
    memory_usage: float
    inference_latency_ms: float
    error_rate: float
    last_update: str
    details: Dict[str, Any]


class TerrragonAnomalyDetector:
    """
    Unified API for Terragon's Advanced IoT Anomaly Detection System.
    
    This class provides a single interface to access all 5 breakthrough AI algorithms
    with automatic model selection, optimization, and deployment capabilities.
    """
    
    def __init__(
        self,
        config: Union[str, Dict[str, Any]] = "production",
        models: Optional[List[ModelType]] = None,
        deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION,
        device: Optional[str] = None
    ):
        """
        Initialize the unified anomaly detector.
        
        Args:
            config: Configuration name or dictionary
            models: List of models to use (None = auto-select)
            deployment_mode: Deployment optimization mode
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.deployment_mode = deployment_mode
        self.device = self._setup_device(device)
        self.models = models or self._auto_select_models()
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize components
        self.ensemble = None
        self.initialized = False
        self.performance_metrics = {
            'total_inferences': 0,
            'total_inference_time': 0.0,
            'anomalies_detected': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        logger.info(f"Initialized Terragon Anomaly Detector on {self.device}")
        logger.info(f"Deployment mode: {deployment_mode.value}")
        logger.info(f"Models: {[m.value for m in self.models]}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup compute device."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _auto_select_models(self) -> List[ModelType]:
        """Auto-select models based on deployment mode."""
        if self.deployment_mode == DeploymentMode.RESEARCH:
            return [
                ModelType.TRANSFORMER_VAE,
                ModelType.SPARSE_GAT,
                ModelType.PHYSICS_INFORMED,
                ModelType.SELF_SUPERVISED,
                ModelType.QUANTUM_CLASSICAL
            ]
        elif self.deployment_mode == DeploymentMode.EDGE:
            return [ModelType.LSTM_AUTOENCODER, ModelType.SPARSE_GAT]
        elif self.deployment_mode == DeploymentMode.REALTIME:
            return [ModelType.LSTM_AUTOENCODER]
        elif self.deployment_mode == DeploymentMode.FEDERATED:
            return [ModelType.SELF_SUPERVISED, ModelType.TRANSFORMER_VAE]
        else:  # PRODUCTION
            return [ModelType.ENSEMBLE]
    
    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration."""
        if isinstance(config, dict):
            return config
        
        # Predefined configurations
        configs = {
            "research": {
                "ensemble_method": "dynamic_weighting",
                "uncertainty_quantification": True,
                "explanation_enabled": True,
                "performance_optimization": False,
                "batch_size": 1,
                "anomaly_threshold": 0.5,
                "max_inference_time_ms": 100
            },
            "production": {
                "ensemble_method": "weighted_average",
                "uncertainty_quantification": True,
                "explanation_enabled": False,
                "performance_optimization": True,
                "batch_size": 8,
                "anomaly_threshold": 0.6,
                "max_inference_time_ms": 20
            },
            "edge": {
                "ensemble_method": "simple_average",
                "uncertainty_quantification": False,
                "explanation_enabled": False,
                "performance_optimization": True,
                "quantization": True,
                "pruning": True,
                "batch_size": 1,
                "anomaly_threshold": 0.7,
                "max_inference_time_ms": 10
            },
            "realtime": {
                "ensemble_method": "single_model",
                "uncertainty_quantification": False,
                "explanation_enabled": False,
                "performance_optimization": True,
                "quantization": True,
                "batch_size": 1,
                "anomaly_threshold": 0.8,
                "max_inference_time_ms": 5
            }
        }
        
        return configs.get(config, configs["production"])
    
    def initialize(self) -> bool:
        """
        Initialize the detection models.
        
        Returns:
            True if initialization successful
        """
        if self.initialized:
            return True
        
        try:
            logger.info("Initializing anomaly detection models...")
            
            # Import models (lazy loading)
            from iot_edge_anomaly.models.advanced_hybrid_integration import create_advanced_hybrid_system
            
            # Create ensemble configuration
            ensemble_config = {
                'deployment_mode': self.deployment_mode.value,
                'models': [m.value for m in self.models],
                **self.config
            }
            
            # Initialize ensemble
            self.ensemble = create_advanced_hybrid_system(ensemble_config)
            
            # Move to device
            if hasattr(self.ensemble, 'to'):
                self.ensemble = self.ensemble.to(self.device)
            
            self.initialized = True
            logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            
            # Fallback: Initialize basic LSTM autoencoder
            try:
                from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
                self.ensemble = LSTMAutoencoder(
                    input_size=5,
                    hidden_size=64,
                    num_layers=2
                ).to(self.device)
                self.ensemble.eval()
                self.initialized = True
                logger.info("Fallback model initialized")
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback initialization failed: {fallback_error}")
                return False
    
    def detect_anomaly(
        self,
        sensor_data: Union[np.ndarray, torch.Tensor, List],
        sensor_metadata: Optional[Dict[str, Any]] = None,
        return_explanation: bool = False
    ) -> AnomalyResult:
        """
        Detect anomalies in sensor data.
        
        Args:
            sensor_data: Sensor readings [batch_size, sequence_length, num_sensors]
                        or [sequence_length, num_sensors] for single sample
            sensor_metadata: Optional metadata about sensors
            return_explanation: Whether to include explanation in result
            
        Returns:
            AnomalyResult with detection information
        """
        start_time = time.time()
        
        # Ensure initialization
        if not self.initialize():
            return AnomalyResult(
                is_anomaly=True,  # Fail-safe: assume anomaly
                anomaly_score=1.0,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                inference_time_ms=0.0,
                model_used="error",
                details={"error": "Model initialization failed"}
            )
        
        try:
            # Preprocess input
            data_tensor = self._preprocess_input(sensor_data)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.ensemble, 'predict'):
                    # Advanced ensemble with full API
                    result = self.ensemble.predict(
                        sensor_data=data_tensor,
                        sensor_metadata=sensor_metadata,
                        return_explanations=return_explanation
                    )
                    
                    # Extract results
                    if isinstance(result, dict):
                        anomaly_score = float(result.get('anomaly_score', 0.5))
                        confidence = float(result.get('confidence', 0.0))
                        explanation = result.get('explanation') if return_explanation else None
                        uncertainty = result.get('uncertainty')
                        model_name = result.get('model_used', 'ensemble')
                    else:
                        anomaly_score = float(result)
                        confidence = 1.0 - abs(anomaly_score - 0.5) * 2
                        explanation = None
                        uncertainty = None
                        model_name = 'ensemble'
                
                else:
                    # Fallback: Basic model
                    if hasattr(self.ensemble, 'compute_reconstruction_error'):
                        error = self.ensemble.compute_reconstruction_error(data_tensor)
                        anomaly_score = float(torch.sigmoid(error * 2 - 1))  # Normalize to 0-1
                    else:
                        output = self.ensemble(data_tensor)
                        error = torch.mean((output - data_tensor) ** 2)
                        anomaly_score = float(torch.sigmoid(error * 2 - 1))
                    
                    confidence = 1.0 - abs(anomaly_score - 0.5) * 2
                    explanation = None
                    uncertainty = None
                    model_name = 'lstm_autoencoder'
            
            # Determine if anomaly
            threshold = self.config.get('anomaly_threshold', 0.5)
            is_anomaly = anomaly_score > threshold
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Update performance metrics
            self._update_metrics(inference_time, is_anomaly, error=False)
            
            # Create result
            result = AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                inference_time_ms=inference_time,
                model_used=model_name,
                explanation=explanation,
                uncertainty=uncertainty,
                details={
                    "threshold": threshold,
                    "input_shape": list(data_tensor.shape),
                    "device": str(self.device)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            
            # Update error metrics
            inference_time = (time.time() - start_time) * 1000
            self._update_metrics(inference_time, False, error=True)
            
            return AnomalyResult(
                is_anomaly=True,  # Fail-safe
                anomaly_score=1.0,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                inference_time_ms=inference_time,
                model_used="error",
                details={"error": str(e)}
            )
    
    def _preprocess_input(self, data: Union[np.ndarray, torch.Tensor, List]) -> torch.Tensor:
        """Preprocess input data to tensor format."""
        if isinstance(data, list):
            data = np.array(data)
        
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Ensure tensor is on correct device
        data = data.to(self.device)
        
        # Ensure 3D tensor: [batch_size, sequence_length, num_sensors]
        if data.dim() == 2:
            data = data.unsqueeze(0)  # Add batch dimension
        elif data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        
        return data
    
    def _update_metrics(self, inference_time: float, is_anomaly: bool, error: bool):
        """Update performance metrics."""
        self.performance_metrics['total_inferences'] += 1
        self.performance_metrics['total_inference_time'] += inference_time
        
        if is_anomaly:
            self.performance_metrics['anomalies_detected'] += 1
        
        if error:
            self.performance_metrics['errors'] += 1
    
    def batch_detect(
        self,
        sensor_data_batch: Union[np.ndarray, torch.Tensor, List[List]],
        sensor_metadata: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in a batch of sensor data.
        
        Args:
            sensor_data_batch: Batch of sensor readings
            sensor_metadata: Optional metadata about sensors
            
        Returns:
            List of AnomalyResult objects
        """
        if not isinstance(sensor_data_batch, (list, tuple)):
            # Single batch tensor - process as batch
            data_tensor = self._preprocess_input(sensor_data_batch)
            results = []
            
            for i in range(data_tensor.shape[0]):
                sample = data_tensor[i:i+1]  # Keep batch dimension
                result = self.detect_anomaly(sample, sensor_metadata)
                results.append(result)
            
            return results
        else:
            # List of samples
            results = []
            for sample in sensor_data_batch:
                result = self.detect_anomaly(sample, sensor_metadata)
                results.append(result)
            return results
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        total_time = time.time() - self.performance_metrics['start_time']
        total_inferences = self.performance_metrics['total_inferences']
        
        avg_latency = (
            self.performance_metrics['total_inference_time'] / max(total_inferences, 1)
        )
        
        error_rate = self.performance_metrics['errors'] / max(total_inferences, 1)
        
        # Simple health assessment
        if error_rate > 0.1 or avg_latency > self.config.get('max_inference_time_ms', 50):
            status = 'critical'
        elif error_rate > 0.05 or avg_latency > self.config.get('max_inference_time_ms', 50) * 0.8:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return SystemHealth(
            status=status,
            cpu_usage=0.0,  # Would need psutil for real monitoring
            memory_usage=0.0,  # Would need psutil for real monitoring
            inference_latency_ms=avg_latency,
            error_rate=error_rate,
            last_update=datetime.now().isoformat(),
            details={
                'total_inferences': total_inferences,
                'anomalies_detected': self.performance_metrics['anomalies_detected'],
                'uptime_seconds': total_time,
                'throughput_per_second': total_inferences / max(total_time, 1),
                'device': str(self.device),
                'models': [m.value for m in self.models]
            }
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        total_time = time.time() - self.performance_metrics['start_time']
        
        return {
            **self.performance_metrics,
            'uptime_seconds': total_time,
            'avg_inference_time_ms': (
                self.performance_metrics['total_inference_time'] / 
                max(self.performance_metrics['total_inferences'], 1)
            ),
            'throughput_per_second': (
                self.performance_metrics['total_inferences'] / max(total_time, 1)
            ),
            'error_rate': (
                self.performance_metrics['errors'] / 
                max(self.performance_metrics['total_inferences'], 1)
            ),
            'anomaly_rate': (
                self.performance_metrics['anomalies_detected'] / 
                max(self.performance_metrics['total_inferences'], 1)
            )
        }
    
    def save_model(self, path: str) -> bool:
        """Save the current model state."""
        try:
            if not self.initialized:
                return False
            
            save_dict = {
                'config': self.config,
                'deployment_mode': self.deployment_mode.value,
                'models': [m.value for m in self.models],
                'performance_metrics': self.performance_metrics
            }
            
            if hasattr(self.ensemble, 'state_dict'):
                save_dict['model_state'] = self.ensemble.state_dict()
            
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a saved model state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.config = checkpoint.get('config', self.config)
            self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
            
            # Reinitialize with loaded config
            if self.initialize():
                if 'model_state' in checkpoint and hasattr(self.ensemble, 'load_state_dict'):
                    self.ensemble.load_state_dict(checkpoint['model_state'])
                
                logger.info(f"Model loaded from {path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# Convenience functions for quick usage
def create_detector(
    mode: str = "production",
    device: str = "auto"
) -> TerrragonAnomalyDetector:
    """
    Create a pre-configured anomaly detector.
    
    Args:
        mode: 'research', 'production', 'edge', 'realtime', or 'federated'
        device: 'cpu', 'cuda', or 'auto'
    
    Returns:
        Configured TerrragonAnomalyDetector
    """
    deployment_mode = DeploymentMode(mode)
    return TerrragonAnomalyDetector(
        config=mode,
        deployment_mode=deployment_mode,
        device=device
    )


def quick_detect(sensor_data: Union[np.ndarray, List], mode: str = "production") -> bool:
    """
    Quick anomaly detection for single use.
    
    Args:
        sensor_data: Sensor readings
        mode: Detection mode
    
    Returns:
        True if anomaly detected
    """
    detector = create_detector(mode)
    result = detector.detect_anomaly(sensor_data)
    return result.is_anomaly


# Example usage and testing
if __name__ == "__main__":
    print("🔮 Terragon Unified API - Testing")
    print("=" * 50)
    
    # Create detector
    detector = create_detector("production")
    
    # Generate test data
    test_data = np.random.randn(20, 5)  # 20 timesteps, 5 sensors
    
    print("Testing single detection...")
    result = detector.detect_anomaly(test_data)
    print(f"Result: {result.is_anomaly} (score: {result.anomaly_score:.3f})")
    
    print("\nTesting batch detection...")
    batch_data = [np.random.randn(20, 5) for _ in range(5)]
    batch_results = detector.batch_detect(batch_data)
    print(f"Batch results: {[r.is_anomaly for r in batch_results]}")
    
    print("\nSystem health:")
    health = detector.get_system_health()
    print(f"Status: {health.status}")
    print(f"Avg latency: {health.inference_latency_ms:.2f} ms")
    
    print("\n✅ Unified API test completed!")