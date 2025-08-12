"""
Advanced Metrics Collection System for IoT Anomaly Detection

This module provides comprehensive metrics collection with advanced features:
- Model performance tracking
- Uncertainty quantification metrics
- Explanation quality metrics
- Real-time performance monitoring
- Custom metric dashboards
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPredictionMetrics:
    """Advanced metrics for individual predictions."""
    timestamp: str
    reconstruction_error: float
    uncertainty: float
    confidence: float
    anomaly_level: str
    processing_time: float
    model_contributions: Dict[str, float]
    explanation_quality: Optional[float] = None
    drift_score: Optional[float] = None


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for individual models."""
    model_name: str
    avg_inference_time: float
    std_inference_time: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    last_updated: str = ""


@dataclass
class EnsembleMetrics:
    """Ensemble-specific performance metrics."""
    ensemble_method: str
    model_agreement: float
    prediction_consistency: float
    uncertainty_calibration: float
    explanation_coverage: float
    avg_processing_time: float


class MetricsAggregator:
    """Aggregates and computes advanced metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history: List[AdvancedPredictionMetrics] = []
        self.model_performance_history: Dict[str, List[float]] = {}
        
    def add_prediction(self, metrics: AdvancedPredictionMetrics):
        """Add prediction metrics to history."""
        self.prediction_history.append(metrics)
        
        # Maintain window size
        if len(self.prediction_history) > self.window_size:
            self.prediction_history = self.prediction_history[-self.window_size:]
    
    def compute_performance_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive performance statistics."""
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        
        stats = {
            'prediction_count': len(self.prediction_history),
            'avg_processing_time': np.mean([p.processing_time for p in recent_predictions]),
            'std_processing_time': np.std([p.processing_time for p in recent_predictions]),
            'avg_uncertainty': np.mean([p.uncertainty for p in recent_predictions]),
            'avg_confidence': np.mean([p.confidence for p in recent_predictions]),
            'anomaly_rate': len([p for p in recent_predictions if p.anomaly_level != 'normal']) / len(recent_predictions),
            'uncertainty_distribution': self._compute_uncertainty_distribution(recent_predictions),
            'anomaly_level_distribution': self._compute_anomaly_distribution(recent_predictions),
            'model_contribution_stability': self._compute_contribution_stability(recent_predictions)
        }
        
        return stats
    
    def _compute_uncertainty_distribution(self, predictions: List[AdvancedPredictionMetrics]) -> Dict[str, float]:
        """Compute uncertainty distribution statistics."""
        uncertainties = [p.uncertainty for p in predictions]
        
        return {
            'min': float(np.min(uncertainties)),
            'max': float(np.max(uncertainties)),
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'percentile_25': float(np.percentile(uncertainties, 25)),
            'percentile_75': float(np.percentile(uncertainties, 75)),
            'percentile_95': float(np.percentile(uncertainties, 95))
        }
    
    def _compute_anomaly_distribution(self, predictions: List[AdvancedPredictionMetrics]) -> Dict[str, int]:
        """Compute distribution of anomaly levels."""
        distribution = {}
        
        for pred in predictions:
            level = pred.anomaly_level
            distribution[level] = distribution.get(level, 0) + 1
        
        return distribution
    
    def _compute_contribution_stability(self, predictions: List[AdvancedPredictionMetrics]) -> Dict[str, float]:
        """Compute stability of model contributions over time."""
        if len(predictions) < 2:
            return {}
        
        # Extract model contributions over time
        models = set()
        for pred in predictions:
            models.update(pred.model_contributions.keys())
        
        stability_scores = {}
        
        for model in models:
            contributions = []
            for pred in predictions:
                contributions.append(pred.model_contributions.get(model, 0.0))
            
            if len(contributions) > 1:
                # Coefficient of variation as stability measure
                mean_contrib = np.mean(contributions)
                std_contrib = np.std(contributions)
                
                if mean_contrib > 0:
                    stability_scores[model] = 1.0 - (std_contrib / mean_contrib)  # Higher is more stable
                else:
                    stability_scores[model] = 1.0 if std_contrib == 0 else 0.0
        
        return stability_scores


class AdvancedMetricsCollector:
    """
    Advanced metrics collection system with async capabilities.
    
    Features:
    - Real-time metrics aggregation
    - Model performance tracking
    - Uncertainty calibration monitoring
    - Explanation quality assessment
    - Custom dashboard generation
    """
    
    def __init__(self, 
                 enable_model_explanations: bool = True,
                 enable_uncertainty_tracking: bool = True,
                 metrics_window_size: int = 1000,
                 export_interval: int = 60):
        
        self.enable_model_explanations = enable_model_explanations
        self.enable_uncertainty_tracking = enable_uncertainty_tracking
        self.metrics_window_size = metrics_window_size
        self.export_interval = export_interval
        
        # Metrics storage
        self.aggregator = MetricsAggregator(window_size=metrics_window_size)
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.ensemble_metrics: Optional[EnsembleMetrics] = None
        
        # Async components
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_export_time = time.time()
        
        # Metrics cache for fast access
        self.metrics_cache = {}
        self.cache_update_interval = 10  # seconds
        self.last_cache_update = 0
        
        logger.info("AdvancedMetricsCollector initialized")
    
    async def record_prediction_async(self,
                                    reconstruction_error: float,
                                    uncertainty: float,
                                    anomaly_level: str,
                                    model_contributions: Dict[str, float],
                                    explanations: Optional[Dict[str, Any]] = None,
                                    processing_time: Optional[float] = None) -> None:
        """Record prediction metrics asynchronously."""
        
        loop = asyncio.get_event_loop()
        
        def _record_sync():
            # Compute confidence from uncertainty
            confidence = 1.0 / (1.0 + uncertainty) if uncertainty > 0 else 1.0
            
            # Compute explanation quality if available
            explanation_quality = None
            if explanations and self.enable_model_explanations:
                explanation_quality = self._compute_explanation_quality(explanations)
            
            # Create metrics object
            metrics = AdvancedPredictionMetrics(
                timestamp=datetime.now().isoformat(),
                reconstruction_error=reconstruction_error,
                uncertainty=uncertainty,
                confidence=confidence,
                anomaly_level=anomaly_level,
                processing_time=processing_time or 0.0,
                model_contributions=model_contributions,
                explanation_quality=explanation_quality
            )
            
            self.aggregator.add_prediction(metrics)
            
            # Update model performance tracking
            self._update_model_performance(model_contributions, processing_time)
        
        await loop.run_in_executor(self.executor, _record_sync)
    
    def _compute_explanation_quality(self, explanations: Dict[str, Any]) -> float:
        """Compute quality score for explanations."""
        if not explanations:
            return 0.0
        
        quality_score = 0.0
        total_components = 0
        
        # Check for individual model explanations
        individual_explanations = explanations.get('individual_explanations', {})
        if individual_explanations:
            valid_explanations = sum(1 for exp in individual_explanations.values() 
                                   if exp.get('status') != 'explanation_failed')
            quality_score += valid_explanations / len(individual_explanations) * 0.4
            total_components += 0.4
        
        # Check for ensemble summary
        ensemble_summary = explanations.get('ensemble_summary', {})
        if ensemble_summary:
            model_agreement = ensemble_summary.get('model_agreement', 0)
            if model_agreement > 0:
                quality_score += model_agreement * 0.3
            total_components += 0.3
        
        # Check for feature importance
        has_feature_importance = any(
            'feature_importance' in exp for exp in individual_explanations.values()
        )
        if has_feature_importance:
            quality_score += 0.3
        total_components += 0.3
        
        return quality_score / total_components if total_components > 0 else 0.0
    
    def _update_model_performance(self, model_contributions: Dict[str, float], 
                                processing_time: Optional[float]):
        """Update individual model performance metrics."""
        current_time = datetime.now().isoformat()
        
        for model_name, contribution in model_contributions.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ModelPerformanceMetrics(
                    model_name=model_name,
                    avg_inference_time=processing_time or 0.0,
                    std_inference_time=0.0,
                    last_updated=current_time
                )
            else:
                # Update inference time statistics
                existing = self.model_performance[model_name]
                if processing_time is not None:
                    # Simple moving average update
                    alpha = 0.1
                    existing.avg_inference_time = (1 - alpha) * existing.avg_inference_time + alpha * processing_time
                
                existing.last_updated = current_time
    
    def update_ensemble_metrics(self, 
                              ensemble_method: str,
                              model_agreement: float,
                              prediction_consistency: float,
                              avg_processing_time: float):
        """Update ensemble-level metrics."""
        
        self.ensemble_metrics = EnsembleMetrics(
            ensemble_method=ensemble_method,
            model_agreement=model_agreement,
            prediction_consistency=prediction_consistency,
            uncertainty_calibration=0.0,  # To be computed
            explanation_coverage=0.0,     # To be computed
            avg_processing_time=avg_processing_time
        )
    
    async def export_advanced_metrics(self) -> Dict[str, Any]:
        """Export comprehensive metrics for monitoring dashboards."""
        current_time = time.time()
        
        # Check if cache needs updating
        if current_time - self.last_cache_update > self.cache_update_interval:
            await self._update_metrics_cache()
            self.last_cache_update = current_time
        
        return self.metrics_cache
    
    async def _update_metrics_cache(self):
        """Update the metrics cache with latest statistics."""
        loop = asyncio.get_event_loop()
        
        def _compute_cache():
            performance_stats = self.aggregator.compute_performance_statistics()
            
            cache = {
                'timestamp': datetime.now().isoformat(),
                'performance_statistics': performance_stats,
                'model_performance': {
                    name: asdict(metrics) for name, metrics in self.model_performance.items()
                },
                'system_health': {
                    'prediction_rate': len(self.aggregator.prediction_history) / max(1, self.metrics_window_size),
                    'avg_processing_time': performance_stats.get('avg_processing_time', 0),
                    'uncertainty_calibration': self._compute_uncertainty_calibration(),
                    'model_availability': len(self.model_performance),
                },
                'advanced_analytics': {
                    'anomaly_trends': self._compute_anomaly_trends(),
                    'model_contribution_analysis': self._analyze_model_contributions(),
                    'uncertainty_insights': self._analyze_uncertainty_patterns(),
                }
            }
            
            if self.ensemble_metrics:
                cache['ensemble_metrics'] = asdict(self.ensemble_metrics)
            
            return cache
        
        self.metrics_cache = await loop.run_in_executor(self.executor, _compute_cache)
    
    def _compute_uncertainty_calibration(self) -> float:
        """Compute how well-calibrated the uncertainty estimates are."""
        if len(self.aggregator.prediction_history) < 10:
            return 0.0
        
        recent_predictions = self.aggregator.prediction_history[-100:]
        
        # Simple calibration check: correlation between confidence and accuracy
        # In a production system, this would need actual ground truth labels
        confidences = [p.confidence for p in recent_predictions]
        
        # Use prediction consistency as proxy for accuracy
        errors = [p.reconstruction_error for p in recent_predictions]
        normalized_errors = [(e - min(errors)) / (max(errors) - min(errors) + 1e-6) for e in errors]
        
        # Higher confidence should correlate with lower normalized error
        if len(confidences) > 1 and len(normalized_errors) > 1:
            correlation = np.corrcoef(confidences, normalized_errors)[0, 1]
            return max(0, -correlation)  # Negative correlation is good
        
        return 0.5  # Default neutral calibration
    
    def _compute_anomaly_trends(self) -> Dict[str, Any]:
        """Compute trends in anomaly detection over time."""
        if len(self.aggregator.prediction_history) < 10:
            return {'trend': 'insufficient_data'}
        
        recent_predictions = self.aggregator.prediction_history[-50:]  # Last 50 predictions
        older_predictions = self.aggregator.prediction_history[-100:-50] if len(self.aggregator.prediction_history) >= 100 else []
        
        # Compute anomaly rates
        recent_anomaly_rate = len([p for p in recent_predictions if p.anomaly_level != 'normal']) / len(recent_predictions)
        
        if older_predictions:
            older_anomaly_rate = len([p for p in older_predictions if p.anomaly_level != 'normal']) / len(older_predictions)
            trend = 'increasing' if recent_anomaly_rate > older_anomaly_rate * 1.1 else \
                   'decreasing' if recent_anomaly_rate < older_anomaly_rate * 0.9 else 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_anomaly_rate': recent_anomaly_rate,
            'anomaly_severity_distribution': self._compute_severity_distribution(recent_predictions)
        }
    
    def _compute_severity_distribution(self, predictions: List[AdvancedPredictionMetrics]) -> Dict[str, float]:
        """Compute distribution of anomaly severity levels."""
        total = len(predictions)
        if total == 0:
            return {}
        
        distribution = {}
        for pred in predictions:
            level = pred.anomaly_level
            distribution[level] = distribution.get(level, 0) + 1
        
        # Convert to percentages
        return {level: count / total for level, count in distribution.items()}
    
    def _analyze_model_contributions(self) -> Dict[str, Any]:
        """Analyze how different models contribute to predictions."""
        if not self.aggregator.prediction_history:
            return {}
        
        recent_predictions = self.aggregator.prediction_history[-100:]
        
        # Aggregate model contributions
        model_contribution_sums = {}
        for pred in recent_predictions:
            for model, contribution in pred.model_contributions.items():
                if model not in model_contribution_sums:
                    model_contribution_sums[model] = []
                model_contribution_sums[model].append(contribution)
        
        analysis = {}
        for model, contributions in model_contribution_sums.items():
            analysis[model] = {
                'avg_contribution': np.mean(contributions),
                'std_contribution': np.std(contributions),
                'max_contribution': np.max(contributions),
                'contribution_trend': self._compute_contribution_trend(contributions)
            }
        
        return analysis
    
    def _compute_contribution_trend(self, contributions: List[float]) -> str:
        """Compute trend in model contributions over time."""
        if len(contributions) < 5:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(contributions))
        slope, _ = np.polyfit(x, contributions, 1)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_uncertainty_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in uncertainty estimates."""
        if not self.aggregator.prediction_history:
            return {}
        
        recent_predictions = self.aggregator.prediction_history[-100:]
        uncertainties = [p.uncertainty for p in recent_predictions]
        reconstruction_errors = [p.reconstruction_error for p in recent_predictions]
        
        analysis = {
            'uncertainty_error_correlation': float(np.corrcoef(uncertainties, reconstruction_errors)[0, 1])
                if len(uncertainties) > 1 else 0.0,
            'high_uncertainty_predictions': len([u for u in uncertainties if u > 0.5]) / len(uncertainties),
            'uncertainty_trend': self._compute_contribution_trend(uncertainties),
            'calibration_score': self._compute_uncertainty_calibration()
        }
        
        return analysis
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get formatted data for monitoring dashboards."""
        if not self.metrics_cache:
            return {'status': 'no_data_available'}
        
        dashboard_data = {
            'timestamp': self.metrics_cache.get('timestamp'),
            'summary': {
                'total_predictions': len(self.aggregator.prediction_history),
                'avg_processing_time': self.metrics_cache.get('performance_statistics', {}).get('avg_processing_time', 0),
                'anomaly_rate': self.metrics_cache.get('performance_statistics', {}).get('anomaly_rate', 0),
                'system_health_score': self._compute_system_health_score()
            },
            'performance_charts': {
                'uncertainty_distribution': self.metrics_cache.get('performance_statistics', {}).get('uncertainty_distribution', {}),
                'anomaly_trends': self.metrics_cache.get('advanced_analytics', {}).get('anomaly_trends', {}),
                'model_contributions': self.metrics_cache.get('advanced_analytics', {}).get('model_contribution_analysis', {})
            },
            'alerts': self._generate_alerts()
        }
        
        return dashboard_data
    
    def _compute_system_health_score(self) -> float:
        """Compute overall system health score (0-1)."""
        if not self.metrics_cache:
            return 0.5
        
        factors = []
        
        # Processing time health (lower is better)
        avg_time = self.metrics_cache.get('performance_statistics', {}).get('avg_processing_time', 0)
        time_health = max(0, 1 - avg_time / 10.0)  # Assume 10s is maximum acceptable
        factors.append(time_health)
        
        # Uncertainty calibration health
        calibration = self.metrics_cache.get('system_health', {}).get('uncertainty_calibration', 0.5)
        factors.append(calibration)
        
        # Model availability health
        available_models = self.metrics_cache.get('system_health', {}).get('model_availability', 0)
        model_health = min(1.0, available_models / 3.0)  # Assume 3+ models is optimal
        factors.append(model_health)
        
        return np.mean(factors) if factors else 0.5
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics."""
        alerts = []
        
        if not self.metrics_cache:
            return alerts
        
        # High processing time alert
        avg_time = self.metrics_cache.get('performance_statistics', {}).get('avg_processing_time', 0)
        if avg_time > 5.0:  # 5 seconds threshold
            alerts.append({
                'severity': 'warning',
                'message': f'High average processing time: {avg_time:.2f}s',
                'timestamp': datetime.now().isoformat()
            })
        
        # High anomaly rate alert
        anomaly_rate = self.metrics_cache.get('performance_statistics', {}).get('anomaly_rate', 0)
        if anomaly_rate > 0.5:  # 50% threshold
            alerts.append({
                'severity': 'critical',
                'message': f'High anomaly rate detected: {anomaly_rate:.1%}',
                'timestamp': datetime.now().isoformat()
            })
        
        # Model performance alert
        model_performance = self.metrics_cache.get('model_performance', {})
        for model_name, metrics in model_performance.items():
            if metrics.get('avg_inference_time', 0) > 2.0:
                alerts.append({
                    'severity': 'warning',
                    'message': f'Model {model_name} showing high inference time',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts