"""
Simple Ensemble System for Testing

A simplified version of the advanced ensemble system for quality gate testing.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimpleEnsemblePredictor:
    """Simple ensemble predictor for testing purposes."""
    
    def __init__(self, models: Dict[str, nn.Module], config: Dict[str, Any]):
        self.models = models
        self.config = config
        self.model_weights = {name: 1.0 / len(models) for name in models.keys()}
        
        logger.info(f"Initialized SimpleEnsemblePredictor with {len(models)} models")
    
    def predict(self, sensor_data: torch.Tensor, 
               edge_index: Optional[torch.Tensor] = None,
               sensor_metadata: Optional[Dict[str, Any]] = None,
               return_explanations: bool = False,
               return_uncertainty: bool = True) -> Dict[str, Any]:
        """Simple ensemble prediction."""
        
        model_predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'compute_reconstruction_error'):
                        prediction = model.compute_reconstruction_error(sensor_data).item()
                    else:
                        output = model(sensor_data)
                        prediction = torch.mean(torch.abs(output - sensor_data)).item()
                    
                    model_predictions[model_name] = prediction
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                model_predictions[model_name] = 1.0  # Default high error
        
        if not model_predictions:
            raise RuntimeError("All model predictions failed")
        
        # Simple average ensemble
        ensemble_method = self.config.get('ensemble_method', 'simple_average')
        
        if ensemble_method == 'simple_average':
            ensemble_prediction = np.mean(list(model_predictions.values()))
        else:
            # Weighted average
            weighted_sum = sum(pred * self.model_weights.get(name, 0) 
                             for name, pred in model_predictions.items())
            ensemble_prediction = weighted_sum
        
        # Simple uncertainty estimate
        uncertainty = np.std(list(model_predictions.values())) if len(model_predictions) > 1 else 0.0
        
        result = {
            'reconstruction_error': float(ensemble_prediction),
            'uncertainty': float(uncertainty),
            'confidence': 1.0 / (1.0 + uncertainty),
            'model_contributions': model_predictions,
            'model_weights': self.model_weights.copy(),
            'ensemble_method': ensemble_method,
            'processing_time': 0.01  # Placeholder
        }
        
        if return_explanations:
            result['explanations'] = {
                'method': 'simple_ensemble',
                'model_agreement': 1.0 - (uncertainty / (ensemble_prediction + 1e-6))
            }
        
        return result


def create_advanced_hybrid_system(config: Dict[str, Any]) -> SimpleEnsemblePredictor:
    """
    Create simple ensemble system for testing.
    
    Args:
        config: Configuration dictionary containing models and settings
        
    Returns:
        SimpleEnsemblePredictor instance
    """
    models = config.get('models', {})
    
    if not models:
        raise ValueError("No models provided in configuration")
    
    return SimpleEnsemblePredictor(models, config)