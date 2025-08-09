"""
Comprehensive model validation and quality assurance framework.

This module provides extensive validation for ML models including:
- Input/output validation
- Model performance monitoring
- Data drift detection
- Model health checks
- Statistical validation
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import pickle
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import warnings

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Validation status types."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Container for validation results."""
    check_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details or {},
            'timestamp': self.timestamp or pd.Timestamp.now().isoformat()
        }


class DataValidator:
    """Validates input data quality and characteristics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.baseline_stats = {}
    
    def validate_input_tensor(self, data: torch.Tensor, 
                            expected_shape: Optional[Tuple[int, ...]] = None) -> List[ValidationResult]:
        """Validate input tensor characteristics."""
        results = []
        
        # Shape validation
        if expected_shape is not None:
            if data.shape != expected_shape:
                results.append(ValidationResult(
                    check_name="tensor_shape",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected shape {expected_shape}, got {data.shape}",
                    details={'expected': expected_shape, 'actual': data.shape}
                ))
            else:
                results.append(ValidationResult(
                    check_name="tensor_shape",
                    status=ValidationStatus.PASSED,
                    severity=ValidationSeverity.INFO,
                    message="Tensor shape validation passed"
                ))
        
        # Data type validation
        if not data.dtype.is_floating_point:
            results.append(ValidationResult(
                check_name="tensor_dtype",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Non-floating point tensor detected: {data.dtype}",
                details={'dtype': str(data.dtype)}
            ))
        
        # NaN/Inf validation
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        
        if nan_count > 0:
            results.append(ValidationResult(
                check_name="nan_values",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                message=f"Found {nan_count} NaN values in tensor",
                details={'nan_count': nan_count, 'total_elements': data.numel()}
            ))
        
        if inf_count > 0:
            results.append(ValidationResult(
                check_name="inf_values",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.ERROR,
                message=f"Found {inf_count} infinite values in tensor",
                details={'inf_count': inf_count, 'total_elements': data.numel()}
            ))
        
        # Range validation
        data_min = data.min().item()
        data_max = data.max().item()
        
        expected_min = self.config.get('min_value', -1000)
        expected_max = self.config.get('max_value', 1000)
        
        if data_min < expected_min or data_max > expected_max:
            results.append(ValidationResult(
                check_name="value_range",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Values outside expected range [{expected_min}, {expected_max}]",
                details={'min_value': data_min, 'max_value': data_max, 
                        'expected_min': expected_min, 'expected_max': expected_max}
            ))
        
        # Statistical validation
        mean_val = data.mean().item()
        std_val = data.std().item()
        
        if std_val < 1e-8:
            results.append(ValidationResult(
                check_name="data_variance",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message="Very low data variance detected - may indicate constant values",
                details={'std': std_val, 'mean': mean_val}
            ))
        
        return results
    
    def validate_data_drift(self, current_data: torch.Tensor, 
                           baseline_data: Optional[torch.Tensor] = None) -> List[ValidationResult]:
        """Detect data drift using statistical tests."""
        results = []
        
        if baseline_data is None and not self.baseline_stats:
            results.append(ValidationResult(
                check_name="data_drift",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.INFO,
                message="No baseline data available for drift detection"
            ))
            return results
        
        if baseline_data is not None:
            # Update baseline stats
            self.baseline_stats = {
                'mean': baseline_data.mean(dim=(0, 1)).numpy(),
                'std': baseline_data.std(dim=(0, 1)).numpy(),
                'distribution': baseline_data.flatten().numpy()
            }
        
        # Current data statistics
        current_mean = current_data.mean(dim=(0, 1)).numpy()
        current_std = current_data.std(dim=(0, 1)).numpy()
        
        # Mean shift detection
        mean_shift = np.abs(current_mean - self.baseline_stats['mean'])
        significant_shift = mean_shift > (2 * self.baseline_stats['std'])
        
        if np.any(significant_shift):
            drift_sensors = np.where(significant_shift)[0]
            results.append(ValidationResult(
                check_name="mean_shift_drift",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Significant mean shift detected in {len(drift_sensors)} sensors",
                details={'affected_sensors': drift_sensors.tolist(),
                        'shift_magnitude': mean_shift.tolist()}
            ))
        
        # Variance change detection
        variance_ratio = current_std / (self.baseline_stats['std'] + 1e-8)
        significant_variance_change = (variance_ratio < 0.5) | (variance_ratio > 2.0)
        
        if np.any(significant_variance_change):
            affected_sensors = np.where(significant_variance_change)[0]
            results.append(ValidationResult(
                check_name="variance_change_drift",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Significant variance change in {len(affected_sensors)} sensors",
                details={'affected_sensors': affected_sensors.tolist(),
                        'variance_ratio': variance_ratio.tolist()}
            ))
        
        # Kolmogorov-Smirnov test for distribution drift
        try:
            current_flat = current_data.flatten().numpy()
            baseline_flat = self.baseline_stats['distribution']
            
            # Sample for computational efficiency
            if len(current_flat) > 10000:
                current_sample = np.random.choice(current_flat, 10000, replace=False)
            else:
                current_sample = current_flat
                
            if len(baseline_flat) > 10000:
                baseline_sample = np.random.choice(baseline_flat, 10000, replace=False)
            else:
                baseline_sample = baseline_flat
            
            ks_statistic, p_value = stats.ks_2samp(baseline_sample, current_sample)
            
            if p_value < 0.05:
                results.append(ValidationResult(
                    check_name="distribution_drift",
                    status=ValidationStatus.WARNING,
                    severity=ValidationSeverity.WARNING,
                    message=f"Significant distribution drift detected (p={p_value:.4f})",
                    details={'ks_statistic': ks_statistic, 'p_value': p_value}
                ))
        except Exception as e:
            results.append(ValidationResult(
                check_name="distribution_drift",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.WARNING,
                message=f"Could not perform distribution drift test: {e}"
            ))
        
        return results


class ModelValidator:
    """Validates ML model performance and health."""
    
    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        self.baseline_metrics = {}
    
    def validate_model_structure(self) -> List[ValidationResult]:
        """Validate model architecture and parameters."""
        results = []
        
        # Parameter count validation
        param_count = sum(p.numel() for p in self.model.parameters())
        max_params = self.config.get('max_parameters', 10_000_000)
        
        if param_count > max_params:
            results.append(ValidationResult(
                check_name="parameter_count",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Model has {param_count:,} parameters (max: {max_params:,})",
                details={'parameter_count': param_count, 'max_allowed': max_params}
            ))
        else:
            results.append(ValidationResult(
                check_name="parameter_count",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"Parameter count within limits: {param_count:,}",
                details={'parameter_count': param_count}
            ))
        
        # Gradient flow validation
        try:
            total_norm = 0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                if total_norm < 1e-8:
                    results.append(ValidationResult(
                        check_name="gradient_flow",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.WARNING,
                        message="Very small gradient norm - possible vanishing gradients",
                        details={'gradient_norm': total_norm}
                    ))
                elif total_norm > 100:
                    results.append(ValidationResult(
                        check_name="gradient_flow",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.WARNING,
                        message="Large gradient norm - possible exploding gradients",
                        details={'gradient_norm': total_norm}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="gradient_flow",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message="Gradient flow is healthy",
                        details={'gradient_norm': total_norm}
                    ))
            else:
                results.append(ValidationResult(
                    check_name="gradient_flow",
                    status=ValidationStatus.SKIPPED,
                    severity=ValidationSeverity.INFO,
                    message="No gradients available for validation"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="gradient_flow",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.WARNING,
                message=f"Could not validate gradient flow: {e}"
            ))
        
        return results
    
    def validate_inference_performance(self, test_data: torch.Tensor, 
                                     test_labels: torch.Tensor) -> List[ValidationResult]:
        """Validate model inference performance."""
        results = []
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Compute predictions/scores
                if hasattr(self.model, 'compute_reconstruction_error'):
                    scores = []
                    for i in range(len(test_data)):
                        sample = test_data[i:i+1]
                        score = self.model.compute_reconstruction_error(sample, reduction='mean')
                        scores.append(score.item())
                    scores = np.array(scores)
                elif hasattr(self.model, 'compute_hybrid_anomaly_score'):
                    scores = []
                    for i in range(len(test_data)):
                        sample = test_data[i:i+1]
                        score = self.model.compute_hybrid_anomaly_score(sample, reduction='mean')
                        scores.append(score.item())
                    scores = np.array(scores)
                else:
                    # Generic approach - use reconstruction error
                    predictions = self.model(test_data)
                    errors = torch.mean((test_data - predictions) ** 2, dim=(1, 2))
                    scores = errors.numpy()
            
            # Compute metrics
            try:
                roc_auc = roc_auc_score(test_labels.numpy(), scores)
                
                # Find optimal threshold
                from sklearn.metrics import roc_curve
                fpr, tpr, thresholds = roc_curve(test_labels.numpy(), scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                predictions = scores > optimal_threshold
                f1 = f1_score(test_labels.numpy(), predictions)
                precision = precision_score(test_labels.numpy(), predictions, zero_division=0)
                recall = recall_score(test_labels.numpy(), predictions, zero_division=0)
                
                metrics = {
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'optimal_threshold': optimal_threshold
                }
                
                # Performance validation
                min_roc_auc = self.config.get('min_roc_auc', 0.6)
                min_f1 = self.config.get('min_f1_score', 0.5)
                
                if roc_auc < min_roc_auc:
                    results.append(ValidationResult(
                        check_name="performance_roc_auc",
                        status=ValidationStatus.FAILED,
                        severity=ValidationSeverity.ERROR,
                        message=f"ROC AUC {roc_auc:.3f} below minimum {min_roc_auc}",
                        details=metrics
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="performance_roc_auc",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"ROC AUC {roc_auc:.3f} meets requirements",
                        details=metrics
                    ))
                
                if f1 < min_f1:
                    results.append(ValidationResult(
                        check_name="performance_f1_score",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.WARNING,
                        message=f"F1-score {f1:.3f} below minimum {min_f1}",
                        details=metrics
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="performance_f1_score",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"F1-score {f1:.3f} meets requirements",
                        details=metrics
                    ))
                
                # Store baseline metrics
                self.baseline_metrics = metrics
                
            except Exception as e:
                results.append(ValidationResult(
                    check_name="performance_metrics",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.ERROR,
                    message=f"Could not compute performance metrics: {e}"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="inference_performance",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Model inference failed: {e}"
            ))
        
        return results
    
    def validate_performance_stability(self, test_data: torch.Tensor, 
                                     test_labels: torch.Tensor,
                                     n_runs: int = 5) -> List[ValidationResult]:
        """Validate model performance stability across multiple runs."""
        results = []
        
        try:
            performance_runs = []
            
            for run in range(n_runs):
                # Add small random noise to test stability
                noisy_data = test_data + 0.01 * torch.randn_like(test_data)
                
                self.model.eval()
                with torch.no_grad():
                    if hasattr(self.model, 'compute_reconstruction_error'):
                        scores = []
                        for i in range(len(noisy_data)):
                            sample = noisy_data[i:i+1]
                            score = self.model.compute_reconstruction_error(sample, reduction='mean')
                            scores.append(score.item())
                        scores = np.array(scores)
                    else:
                        predictions = self.model(noisy_data)
                        errors = torch.mean((noisy_data - predictions) ** 2, dim=(1, 2))
                        scores = errors.numpy()
                
                # Compute ROC AUC for this run
                try:
                    roc_auc = roc_auc_score(test_labels.numpy(), scores)
                    performance_runs.append(roc_auc)
                except Exception:
                    continue
            
            if len(performance_runs) >= 2:
                mean_performance = np.mean(performance_runs)
                std_performance = np.std(performance_runs)
                cv_performance = std_performance / mean_performance if mean_performance > 0 else float('inf')
                
                max_cv = self.config.get('max_performance_cv', 0.1)
                
                if cv_performance > max_cv:
                    results.append(ValidationResult(
                        check_name="performance_stability",
                        status=ValidationStatus.WARNING,
                        severity=ValidationSeverity.WARNING,
                        message=f"High performance variability (CV={cv_performance:.3f})",
                        details={'mean_roc_auc': mean_performance,
                                'std_roc_auc': std_performance,
                                'coefficient_of_variation': cv_performance,
                                'runs': performance_runs}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="performance_stability",
                        status=ValidationStatus.PASSED,
                        severity=ValidationSeverity.INFO,
                        message=f"Performance stability validated (CV={cv_performance:.3f})",
                        details={'mean_roc_auc': mean_performance,
                                'std_roc_auc': std_performance,
                                'coefficient_of_variation': cv_performance}
                    ))
            else:
                results.append(ValidationResult(
                    check_name="performance_stability",
                    status=ValidationStatus.FAILED,
                    severity=ValidationSeverity.WARNING,
                    message="Could not complete stability validation"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                check_name="performance_stability",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.WARNING,
                message=f"Performance stability validation failed: {e}"
            ))
        
        return results


class ComprehensiveValidator:
    """Main validator orchestrating all validation checks."""
    
    def __init__(self, model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        
        self.data_validator = DataValidator(config.get('data_validation', {}))
        self.model_validator = ModelValidator(model, config.get('model_validation', {}))
        
        self.validation_history = []
    
    def run_comprehensive_validation(self, 
                                   train_data: torch.Tensor,
                                   test_data: torch.Tensor, 
                                   test_labels: torch.Tensor) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive report."""
        logger.info("Running comprehensive model validation")
        
        all_results = []
        
        # Data validation
        logger.info("Validating input data...")
        data_results = self.data_validator.validate_input_tensor(
            test_data, expected_shape=None
        )
        all_results.extend(data_results)
        
        # Data drift validation
        drift_results = self.data_validator.validate_data_drift(test_data, train_data)
        all_results.extend(drift_results)
        
        # Model structure validation
        logger.info("Validating model structure...")
        structure_results = self.model_validator.validate_model_structure()
        all_results.extend(structure_results)
        
        # Performance validation
        logger.info("Validating model performance...")
        performance_results = self.model_validator.validate_inference_performance(
            test_data, test_labels
        )
        all_results.extend(performance_results)
        
        # Stability validation
        logger.info("Validating performance stability...")
        stability_results = self.model_validator.validate_performance_stability(
            test_data, test_labels
        )
        all_results.extend(stability_results)
        
        # Summarize results
        report = self._create_validation_report(all_results)
        
        # Store in history
        self.validation_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': [r.to_dict() for r in all_results],
            'summary': report['summary']
        })
        
        logger.info(f"Validation complete: {report['summary']['total_checks']} checks, "
                   f"{report['summary']['passed']} passed, "
                   f"{report['summary']['failed']} failed")
        
        return report
    
    def _create_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        # Categorize results
        passed = [r for r in results if r.status == ValidationStatus.PASSED]
        failed = [r for r in results if r.status == ValidationStatus.FAILED]
        warnings = [r for r in results if r.status == ValidationStatus.WARNING]
        skipped = [r for r in results if r.status == ValidationStatus.SKIPPED]
        
        # Severity counts
        critical = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        warn_severity = [r for r in results if r.severity == ValidationSeverity.WARNING]
        
        # Overall status
        if critical or len(errors) > 0:
            overall_status = "FAILED"
        elif len(warnings) > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_checks': len(results),
                'passed': len(passed),
                'failed': len(failed),
                'warnings': len(warnings),
                'skipped': len(skipped),
                'critical_issues': len(critical),
                'errors': len(errors)
            },
            'results_by_category': {
                'data_validation': [r.to_dict() for r in results if 'tensor' in r.check_name or 'drift' in r.check_name],
                'model_validation': [r.to_dict() for r in results if 'parameter' in r.check_name or 'gradient' in r.check_name],
                'performance_validation': [r.to_dict() for r in results if 'performance' in r.check_name]
            },
            'failed_checks': [r.to_dict() for r in failed],
            'critical_issues': [r.to_dict() for r in critical],
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Check for specific issues and generate targeted advice
        for result in results:
            if result.status == ValidationStatus.FAILED or result.severity == ValidationSeverity.CRITICAL:
                if 'nan_values' in result.check_name:
                    recommendations.append("Clean input data to remove NaN values before model inference")
                elif 'inf_values' in result.check_name:
                    recommendations.append("Implement input sanitization to handle infinite values")
                elif 'performance' in result.check_name:
                    recommendations.append("Consider model retraining or hyperparameter tuning to improve performance")
                elif 'gradient_flow' in result.check_name:
                    recommendations.append("Review model architecture and learning rate to address gradient issues")
            
            elif result.status == ValidationStatus.WARNING:
                if 'drift' in result.check_name:
                    recommendations.append("Monitor data distribution changes and consider model retraining")
                elif 'stability' in result.check_name:
                    recommendations.append("Investigate model stability issues - consider ensemble methods")
                elif 'parameter_count' in result.check_name:
                    recommendations.append("Consider model pruning or quantization for edge deployment")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def save_validation_report(self, report: Dict[str, Any], output_path: str):
        """Save validation report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get historical validation results."""
        return self.validation_history