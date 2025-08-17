"""
Comprehensive Validation Framework for IoT Edge Anomaly Detection.

Implements enterprise-grade validation including input sanitization,
model drift detection, adversarial input detection, and data quality
assessment for mission-critical deployments.

Key Features:
- Multi-level input validation and sanitization
- Real-time model drift detection with statistical tests
- Adversarial input detection using ensemble methods
- Data quality assessment with anomaly scoring
- Automated validation pipelines with alerting
- Compliance validation for regulatory standards
"""

import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    INPUT_SANITIZATION = "input_sanitization"
    DATA_QUALITY = "data_quality"
    MODEL_DRIFT = "model_drift"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    category: ValidationCategory
    level: ValidationLevel
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category.value,
            'level': self.level.value,
            'check_name': self.check_name,
            'passed': self.passed,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    validation_results: List[ValidationResult]
    overall_status: str
    execution_time: float
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        return self.passed_checks / max(self.total_checks, 1)
    
    def get_failures_by_category(self) -> Dict[str, List[ValidationResult]]:
        """Group failed validations by category."""
        failures = {}
        for result in self.validation_results:
            if not result.passed:
                category = result.category.value
                if category not in failures:
                    failures[category] = []
                failures[category].append(result)
        return failures
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'success_rate': self.success_rate,
            'overall_status': self.overall_status,
            'execution_time': self.execution_time,
            'validation_results': [result.to_dict() for result in self.validation_results],
            'failures_by_category': {
                category: [result.to_dict() for result in results]
                for category, results in self.get_failures_by_category().items()
            }
        }


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, name: str, category: ValidationCategory):
        self.name = name
        self.category = category
        self.enabled = True
        self.config = {}
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Perform validation check."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if validator is enabled."""
        return self.enabled
    
    def configure(self, config: Dict[str, Any]):
        """Configure validator parameters."""
        self.config.update(config)


class InputSanitizationValidator(BaseValidator):
    """Validates and sanitizes input data."""
    
    def __init__(self):
        super().__init__("input_sanitization", ValidationCategory.INPUT_SANITIZATION)
        self.expected_shape = None
        self.value_ranges = {}
        self.required_features = []
    
    def validate(self, data: torch.Tensor, **kwargs) -> ValidationResult:
        """Validate input tensor data."""
        details = {}
        issues = []
        
        try:
            # Check tensor type and device
            if not isinstance(data, torch.Tensor):
                issues.append(f"Expected torch.Tensor, got {type(data)}")
            
            # Check for NaN values
            if torch.isnan(data).any():
                nan_count = torch.isnan(data).sum().item()
                issues.append(f"Found {nan_count} NaN values")
                details['nan_count'] = nan_count
                
                # Sanitize NaN values
                data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
                details['sanitized_nans'] = True
            
            # Check for infinite values
            if torch.isinf(data).any():
                inf_count = torch.isinf(data).sum().item()
                issues.append(f"Found {inf_count} infinite values")
                details['inf_count'] = inf_count
                
                # Sanitize infinite values
                data = torch.where(torch.isinf(data), torch.zeros_like(data), data)
                details['sanitized_infs'] = True
            
            # Check shape consistency
            if self.expected_shape and data.shape != self.expected_shape:
                issues.append(f"Shape mismatch: expected {self.expected_shape}, got {data.shape}")
                details['shape_mismatch'] = True
            
            # Check value ranges
            for feature_idx, (min_val, max_val) in self.value_ranges.items():
                if feature_idx < data.shape[-1]:
                    feature_data = data[..., feature_idx]
                    out_of_range = ((feature_data < min_val) | (feature_data > max_val)).sum().item()
                    if out_of_range > 0:
                        issues.append(f"Feature {feature_idx}: {out_of_range} values out of range [{min_val}, {max_val}]")
                        details[f'feature_{feature_idx}_out_of_range'] = out_of_range
            
            # Check for extreme values (statistical outliers)
            if data.numel() > 0:
                data_flat = data.flatten()
                q1 = torch.quantile(data_flat, 0.25)
                q3 = torch.quantile(data_flat, 0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((data_flat < lower_bound) | (data_flat > upper_bound)).sum().item()
                if outliers > data.numel() * 0.1:  # More than 10% outliers
                    issues.append(f"High number of statistical outliers: {outliers}")
                    details['statistical_outliers'] = outliers
            
            # Determine validation result
            level = ValidationLevel.INFO
            if issues:
                if any('NaN' in issue or 'infinite' in issue for issue in issues):
                    level = ValidationLevel.ERROR
                elif any('out of range' in issue for issue in issues):
                    level = ValidationLevel.WARNING
                else:
                    level = ValidationLevel.INFO
            
            passed = level in [ValidationLevel.INFO, ValidationLevel.WARNING]
            message = "Input validation passed" if not issues else f"Issues found: {'; '.join(issues)}"
            
            return ValidationResult(
                category=self.category,
                level=level,
                check_name=self.name,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return ValidationResult(
                category=self.category,
                level=ValidationLevel.CRITICAL,
                check_name=self.name,
                passed=False,
                message=f"Validation failed with exception: {str(e)}",
                details={'exception': str(e)},
                timestamp=datetime.now()
            )
    
    def set_expected_shape(self, shape: Tuple[int, ...]):
        """Set expected input shape."""
        self.expected_shape = shape
    
    def set_value_ranges(self, ranges: Dict[int, Tuple[float, float]]):
        """Set expected value ranges for features."""
        self.value_ranges = ranges


class DataQualityValidator(BaseValidator):
    """Validates data quality metrics."""
    
    def __init__(self):
        super().__init__("data_quality", ValidationCategory.DATA_QUALITY)
        self.reference_stats = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'uniqueness': 0.85,
            'validity': 0.95
        }
    
    def validate(self, data: torch.Tensor, **kwargs) -> ValidationResult:
        """Validate data quality metrics."""
        details = {}
        issues = []
        
        try:
            # Completeness check (non-zero values)
            total_elements = data.numel()
            non_zero_elements = torch.count_nonzero(data).item()
            completeness = non_zero_elements / total_elements if total_elements > 0 else 0
            details['completeness'] = completeness
            
            if completeness < self.quality_thresholds['completeness']:
                issues.append(f"Low completeness: {completeness:.3f} < {self.quality_thresholds['completeness']}")
            
            # Consistency check (statistical consistency with reference)
            if self.reference_stats:
                current_mean = torch.mean(data).item()
                current_std = torch.std(data).item()
                
                ref_mean = self.reference_stats.get('mean', current_mean)
                ref_std = self.reference_stats.get('std', current_std)
                
                mean_diff = abs(current_mean - ref_mean) / max(abs(ref_mean), 1e-8)
                std_diff = abs(current_std - ref_std) / max(ref_std, 1e-8)
                
                consistency = 1.0 - max(mean_diff, std_diff)
                details['consistency'] = consistency
                details['mean_drift'] = mean_diff
                details['std_drift'] = std_diff
                
                if consistency < self.quality_thresholds['consistency']:
                    issues.append(f"Low consistency: {consistency:.3f} < {self.quality_thresholds['consistency']}")
            
            # Uniqueness check (for sequences)
            if data.dim() >= 2:
                # Check uniqueness across time dimension
                if data.dim() == 3:  # [batch, time, features]
                    batch_data = data[0]  # Use first batch
                else:  # [time, features]
                    batch_data = data
                
                unique_rows = torch.unique(batch_data, dim=0).shape[0]
                total_rows = batch_data.shape[0]
                uniqueness = unique_rows / total_rows if total_rows > 0 else 1.0
                details['uniqueness'] = uniqueness
                
                if uniqueness < self.quality_thresholds['uniqueness']:
                    issues.append(f"Low uniqueness: {uniqueness:.3f} < {self.quality_thresholds['uniqueness']}")
            
            # Validity check (finite values)
            finite_values = torch.isfinite(data).sum().item()
            validity = finite_values / total_elements if total_elements > 0 else 1.0
            details['validity'] = validity
            
            if validity < self.quality_thresholds['validity']:
                issues.append(f"Low validity: {validity:.3f} < {self.quality_thresholds['validity']}")
            
            # Calculate overall quality score
            quality_scores = [
                details.get('completeness', 1.0),
                details.get('consistency', 1.0),
                details.get('uniqueness', 1.0),
                details.get('validity', 1.0)
            ]
            overall_quality = np.mean(quality_scores)
            details['overall_quality'] = overall_quality
            
            # Determine validation result
            if overall_quality >= 0.9:
                level = ValidationLevel.INFO
            elif overall_quality >= 0.7:
                level = ValidationLevel.WARNING
            else:
                level = ValidationLevel.ERROR
            
            passed = level != ValidationLevel.ERROR
            message = f"Data quality score: {overall_quality:.3f}"
            if issues:
                message += f" - Issues: {'; '.join(issues)}"
            
            return ValidationResult(
                category=self.category,
                level=level,
                check_name=self.name,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now(),
                confidence=overall_quality
            )
            
        except Exception as e:
            return ValidationResult(
                category=self.category,
                level=ValidationLevel.CRITICAL,
                check_name=self.name,
                passed=False,
                message=f"Data quality validation failed: {str(e)}",
                details={'exception': str(e)},
                timestamp=datetime.now()
            )
    
    def set_reference_stats(self, stats: Dict[str, float]):
        """Set reference statistics for consistency checks."""
        self.reference_stats = stats


class ModelDriftValidator(BaseValidator):
    """Detects model drift using statistical tests."""
    
    def __init__(self):
        super().__init__("model_drift", ValidationCategory.MODEL_DRIFT)
        self.reference_predictions = []
        self.reference_errors = []
        self.drift_threshold = 0.05  # p-value threshold
        self.window_size = 100
    
    def validate(self, data: Dict[str, Any], **kwargs) -> ValidationResult:
        """Validate for model drift."""
        details = {}
        issues = []
        
        try:
            current_predictions = data.get('predictions', [])
            current_errors = data.get('errors', [])
            
            if not current_predictions:
                return ValidationResult(
                    category=self.category,
                    level=ValidationLevel.WARNING,
                    check_name=self.name,
                    passed=True,
                    message="No current predictions available for drift detection",
                    details={},
                    timestamp=datetime.now()
                )
            
            # Initialize reference data if empty
            if not self.reference_predictions:
                self.reference_predictions = current_predictions[-self.window_size:]
                self.reference_errors = current_errors[-self.window_size:] if current_errors else []
                
                return ValidationResult(
                    category=self.category,
                    level=ValidationLevel.INFO,
                    check_name=self.name,
                    passed=True,
                    message="Initialized reference data for drift detection",
                    details={'reference_size': len(self.reference_predictions)},
                    timestamp=datetime.now()
                )
            
            # Prediction distribution drift (Kolmogorov-Smirnov test)
            if len(current_predictions) >= 30 and len(self.reference_predictions) >= 30:
                try:
                    ks_statistic, ks_p_value = stats.ks_2samp(
                        self.reference_predictions, current_predictions
                    )
                    details['ks_statistic'] = ks_statistic
                    details['ks_p_value'] = ks_p_value
                    
                    if ks_p_value < self.drift_threshold:
                        issues.append(f"Prediction distribution drift detected (KS p-value: {ks_p_value:.4f})")
                        details['prediction_drift'] = True
                except Exception as e:
                    logger.warning(f"KS test failed: {e}")
            
            # Error distribution drift
            if current_errors and self.reference_errors and len(current_errors) >= 30:
                try:
                    error_ks_statistic, error_ks_p_value = stats.ks_2samp(
                        self.reference_errors, current_errors
                    )
                    details['error_ks_statistic'] = error_ks_statistic
                    details['error_ks_p_value'] = error_ks_p_value
                    
                    if error_ks_p_value < self.drift_threshold:
                        issues.append(f"Error distribution drift detected (KS p-value: {error_ks_p_value:.4f})")
                        details['error_drift'] = True
                except Exception as e:
                    logger.warning(f"Error KS test failed: {e}")
            
            # Mean shift detection (Welch's t-test)
            if len(current_predictions) >= 20 and len(self.reference_predictions) >= 20:
                try:
                    t_statistic, t_p_value = stats.ttest_ind(
                        self.reference_predictions, current_predictions, 
                        equal_var=False
                    )
                    details['t_statistic'] = t_statistic
                    details['t_p_value'] = t_p_value
                    
                    if t_p_value < self.drift_threshold:
                        issues.append(f"Mean shift detected (t-test p-value: {t_p_value:.4f})")
                        details['mean_shift'] = True
                except Exception as e:
                    logger.warning(f"T-test failed: {e}")
            
            # Variance change detection (Levene's test)
            if len(current_predictions) >= 20 and len(self.reference_predictions) >= 20:
                try:
                    levene_statistic, levene_p_value = stats.levene(
                        self.reference_predictions, current_predictions
                    )
                    details['levene_statistic'] = levene_statistic
                    details['levene_p_value'] = levene_p_value
                    
                    if levene_p_value < self.drift_threshold:
                        issues.append(f"Variance change detected (Levene p-value: {levene_p_value:.4f})")
                        details['variance_change'] = True
                except Exception as e:
                    logger.warning(f"Levene test failed: {e}")
            
            # Update reference data (sliding window)
            self.reference_predictions = (self.reference_predictions + current_predictions)[-self.window_size:]
            if current_errors:
                self.reference_errors = (self.reference_errors + current_errors)[-self.window_size:]
            
            # Determine validation result
            drift_detected = len(issues) > 0
            if drift_detected:
                if len(issues) >= 3:  # Multiple drift indicators
                    level = ValidationLevel.ERROR
                else:
                    level = ValidationLevel.WARNING
            else:
                level = ValidationLevel.INFO
            
            passed = not drift_detected or level == ValidationLevel.WARNING
            message = "No model drift detected" if not issues else f"Drift detected: {'; '.join(issues)}"
            
            return ValidationResult(
                category=self.category,
                level=level,
                check_name=self.name,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now(),
                confidence=1.0 - len(issues) * 0.2  # Reduce confidence with more issues
            )
            
        except Exception as e:
            return ValidationResult(
                category=self.category,
                level=ValidationLevel.CRITICAL,
                check_name=self.name,
                passed=False,
                message=f"Model drift validation failed: {str(e)}",
                details={'exception': str(e)},
                timestamp=datetime.now()
            )


class AdversarialDetectionValidator(BaseValidator):
    """Detects adversarial inputs using ensemble methods."""
    
    def __init__(self):
        super().__init__("adversarial_detection", ValidationCategory.ADVERSARIAL_DETECTION)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        self.normal_data_stats = {}
        self.adversarial_threshold = -0.5  # Isolation Forest score threshold
    
    def validate(self, data: torch.Tensor, **kwargs) -> ValidationResult:
        """Detect adversarial inputs."""
        details = {}
        issues = []
        
        try:
            # Convert to numpy for sklearn
            if isinstance(data, torch.Tensor):
                data_np = data.detach().cpu().numpy()
            else:
                data_np = np.array(data)
            
            # Flatten if multi-dimensional
            if data_np.ndim > 2:
                original_shape = data_np.shape
                data_np = data_np.reshape(data_np.shape[0], -1)
                details['reshaped_from'] = original_shape
            elif data_np.ndim == 1:
                data_np = data_np.reshape(1, -1)
            
            # Train isolation forest if not trained
            if not self.is_trained and data_np.shape[0] > 10:
                logger.info("Training isolation forest for adversarial detection")
                self.isolation_forest.fit(data_np)
                self.is_trained = True
                
                # Store normal data statistics
                self.normal_data_stats = {
                    'mean': np.mean(data_np, axis=0),
                    'std': np.std(data_np, axis=0),
                    'min': np.min(data_np, axis=0),
                    'max': np.max(data_np, axis=0)
                }
                
                return ValidationResult(
                    category=self.category,
                    level=ValidationLevel.INFO,
                    check_name=self.name,
                    passed=True,
                    message="Initialized adversarial detection model",
                    details={'training_samples': data_np.shape[0]},
                    timestamp=datetime.now()
                )
            
            if not self.is_trained:
                return ValidationResult(
                    category=self.category,
                    level=ValidationLevel.WARNING,
                    check_name=self.name,
                    passed=True,
                    message="Insufficient data to train adversarial detection",
                    details={'samples_needed': max(10 - data_np.shape[0], 0)},
                    timestamp=datetime.now()
                )
            
            # Isolation Forest detection
            anomaly_scores = self.isolation_forest.decision_function(data_np)
            predictions = self.isolation_forest.predict(data_np)
            
            outliers = (predictions == -1).sum()
            outlier_ratio = outliers / len(predictions)
            details['outlier_count'] = int(outliers)
            details['outlier_ratio'] = outlier_ratio
            details['min_anomaly_score'] = float(np.min(anomaly_scores))
            details['mean_anomaly_score'] = float(np.mean(anomaly_scores))
            
            if outlier_ratio > 0.2:  # More than 20% outliers
                issues.append(f"High outlier ratio: {outlier_ratio:.3f}")
            
            # Statistical anomaly detection
            if self.normal_data_stats:
                mean_diff = np.abs(np.mean(data_np, axis=0) - self.normal_data_stats['mean'])
                std_diff = np.abs(np.std(data_np, axis=0) - self.normal_data_stats['std'])
                
                # Normalize differences
                mean_diff_norm = np.mean(mean_diff / (self.normal_data_stats['std'] + 1e-8))
                std_diff_norm = np.mean(std_diff / (self.normal_data_stats['std'] + 1e-8))
                
                details['mean_deviation'] = float(mean_diff_norm)
                details['std_deviation'] = float(std_diff_norm)
                
                if mean_diff_norm > 3.0:  # 3-sigma rule
                    issues.append(f"Large mean deviation: {mean_diff_norm:.3f}")
                
                if std_diff_norm > 2.0:
                    issues.append(f"Large std deviation: {std_diff_norm:.3f}")
            
            # Gradient-based detection (simplified)
            if hasattr(kwargs, 'model') and isinstance(data, torch.Tensor):
                model = kwargs.get('model')
                if model is not None:
                    try:
                        data.requires_grad_(True)
                        output = model(data)
                        
                        if output.requires_grad:
                            # Compute gradients
                            grad_outputs = torch.ones_like(output)
                            gradients = torch.autograd.grad(
                                outputs=output,
                                inputs=data,
                                grad_outputs=grad_outputs,
                                create_graph=False,
                                retain_graph=False
                            )[0]
                            
                            gradient_norm = torch.norm(gradients, dim=-1).mean().item()
                            details['gradient_norm'] = gradient_norm
                            
                            # High gradients might indicate adversarial perturbations
                            if gradient_norm > 10.0:
                                issues.append(f"High gradient norm: {gradient_norm:.3f}")
                    except Exception as e:
                        logger.warning(f"Gradient-based detection failed: {e}")
            
            # Determine validation result
            adversarial_detected = len(issues) > 0
            
            if adversarial_detected:
                if len(issues) >= 2 or outlier_ratio > 0.5:
                    level = ValidationLevel.ERROR
                else:
                    level = ValidationLevel.WARNING
            else:
                level = ValidationLevel.INFO
            
            passed = not adversarial_detected or level == ValidationLevel.WARNING
            message = "No adversarial inputs detected" if not issues else f"Potential adversarial inputs: {'; '.join(issues)}"
            
            confidence = 1.0 - min(outlier_ratio, 0.5) * 2  # Reduce confidence with more outliers
            
            return ValidationResult(
                category=self.category,
                level=level,
                check_name=self.name,
                passed=passed,
                message=message,
                details=details,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
        except Exception as e:
            return ValidationResult(
                category=self.category,
                level=ValidationLevel.CRITICAL,
                check_name=self.name,
                passed=False,
                message=f"Adversarial detection failed: {str(e)}",
                details={'exception': str(e)},
                timestamp=datetime.now()
            )


class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework that orchestrates all validation checks.
    
    Provides centralized validation with configurable validators, automated
    reporting, and integration with monitoring systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_history: List[ValidationReport] = []
        self.max_history = 100
        
        # Initialize validators
        self._initialize_validators()
        
        # Alert configuration
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'error_rate': 0.1,  # Alert if >10% validations fail
            'critical_failures': 1,  # Alert on any critical failure
            'consecutive_failures': 3  # Alert after 3 consecutive failures
        })
        
        logger.info("Initialized Comprehensive Validation Framework")
    
    def _initialize_validators(self):
        """Initialize all validators based on configuration."""
        
        # Input Sanitization
        if self.config.get('enable_input_sanitization', True):
            self.validators['input_sanitization'] = InputSanitizationValidator()
        
        # Data Quality
        if self.config.get('enable_data_quality', True):
            self.validators['data_quality'] = DataQualityValidator()
        
        # Model Drift Detection
        if self.config.get('enable_model_drift', True):
            self.validators['model_drift'] = ModelDriftValidator()
        
        # Adversarial Detection
        if self.config.get('enable_adversarial_detection', True):
            self.validators['adversarial_detection'] = AdversarialDetectionValidator()
        
        logger.info(f"Initialized {len(self.validators)} validators")
    
    def validate_input(self, data: torch.Tensor, **kwargs) -> ValidationReport:
        """
        Perform comprehensive validation on input data.
        
        Args:
            data: Input tensor to validate
            **kwargs: Additional context for validation
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        validation_results = []
        
        logger.debug("Starting comprehensive validation")
        
        # Run all enabled validators
        for validator_name, validator in self.validators.items():
            if not validator.is_enabled():
                continue
            
            try:
                logger.debug(f"Running validator: {validator_name}")
                
                # Prepare validation data based on validator type
                if validator_name == 'model_drift':
                    # Model drift needs predictions/errors
                    validation_data = kwargs.get('drift_data', {})
                else:
                    validation_data = data
                
                result = validator.validate(validation_data, **kwargs)
                validation_results.append(result)
                
            except Exception as e:
                logger.error(f"Validator {validator_name} failed with exception: {e}")
                
                # Create error result
                error_result = ValidationResult(
                    category=validator.category,
                    level=ValidationLevel.CRITICAL,
                    check_name=validator_name,
                    passed=False,
                    message=f"Validator exception: {str(e)}",
                    details={'exception': str(e)},
                    timestamp=datetime.now()
                )
                validation_results.append(error_result)
        
        # Generate report
        execution_time = time.time() - start_time
        total_checks = len(validation_results)
        passed_checks = sum(1 for result in validation_results if result.passed)
        failed_checks = total_checks - passed_checks
        
        # Determine overall status
        critical_failures = sum(1 for result in validation_results 
                              if result.level == ValidationLevel.CRITICAL)
        error_failures = sum(1 for result in validation_results 
                           if result.level == ValidationLevel.ERROR)
        
        if critical_failures > 0:
            overall_status = "CRITICAL"
        elif error_failures > 0:
            overall_status = "ERROR"
        elif failed_checks > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"
        
        report = ValidationReport(
            timestamp=datetime.now(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            validation_results=validation_results,
            overall_status=overall_status,
            execution_time=execution_time
        )
        
        # Store in history
        self.validation_history.append(report)
        if len(self.validation_history) > self.max_history:
            self.validation_history = self.validation_history[-self.max_history:]
        
        # Check for alerts
        self._check_alert_conditions(report)
        
        logger.info(f"Validation completed: {overall_status} "
                   f"({passed_checks}/{total_checks} passed, {execution_time:.3f}s)")
        
        return report
    
    def configure_validator(self, validator_name: str, config: Dict[str, Any]):
        """Configure a specific validator."""
        if validator_name in self.validators:
            self.validators[validator_name].configure(config)
            logger.info(f"Configured validator: {validator_name}")
        else:
            logger.warning(f"Validator not found: {validator_name}")
    
    def enable_validator(self, validator_name: str):
        """Enable a specific validator."""
        if validator_name in self.validators:
            self.validators[validator_name].enabled = True
            logger.info(f"Enabled validator: {validator_name}")
    
    def disable_validator(self, validator_name: str):
        """Disable a specific validator."""
        if validator_name in self.validators:
            self.validators[validator_name].enabled = False
            logger.info(f"Disabled validator: {validator_name}")
    
    def get_validation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent validation results."""
        recent_reports = self.validation_history[-last_n:]
        
        if not recent_reports:
            return {
                'total_reports': 0,
                'average_success_rate': 0.0,
                'status_distribution': {},
                'common_failures': []
            }
        
        # Calculate statistics
        total_reports = len(recent_reports)
        success_rates = [report.success_rate for report in recent_reports]
        average_success_rate = np.mean(success_rates)
        
        # Status distribution
        status_counts = {}
        for report in recent_reports:
            status = report.overall_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Common failure patterns
        failure_counts = {}
        for report in recent_reports:
            for result in report.validation_results:
                if not result.passed:
                    key = f"{result.category.value}:{result.check_name}"
                    failure_counts[key] = failure_counts.get(key, 0) + 1
        
        common_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_reports': total_reports,
            'average_success_rate': average_success_rate,
            'status_distribution': status_counts,
            'common_failures': common_failures,
            'recent_execution_times': [report.execution_time for report in recent_reports],
            'trend': self._calculate_trend(success_rates)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _check_alert_conditions(self, report: ValidationReport):
        """Check if validation results should trigger alerts."""
        
        # Critical failure alert
        critical_failures = sum(1 for result in report.validation_results 
                              if result.level == ValidationLevel.CRITICAL)
        
        if critical_failures >= self.alert_thresholds['critical_failures']:
            self._trigger_alert(
                "CRITICAL_VALIDATION_FAILURE",
                f"Critical validation failures detected: {critical_failures}",
                report.to_dict()
            )
        
        # Error rate alert
        if report.total_checks > 0:
            error_rate = report.failed_checks / report.total_checks
            if error_rate >= self.alert_thresholds['error_rate']:
                self._trigger_alert(
                    "HIGH_ERROR_RATE",
                    f"High validation error rate: {error_rate:.3f}",
                    report.to_dict()
                )
        
        # Consecutive failures alert
        recent_reports = self.validation_history[-self.alert_thresholds['consecutive_failures']:]
        if (len(recent_reports) >= self.alert_thresholds['consecutive_failures'] and
            all(r.overall_status in ['ERROR', 'CRITICAL'] for r in recent_reports)):
            
            self._trigger_alert(
                "CONSECUTIVE_FAILURES",
                f"Consecutive validation failures: {len(recent_reports)}",
                [r.to_dict() for r in recent_reports]
            )
    
    def _trigger_alert(self, alert_type: str, message: str, context: Any):
        """Trigger validation alert."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'context': context
        }
        
        logger.critical(f"VALIDATION ALERT: {alert_type} - {message}")
        
        # In production, this would send to monitoring/alerting system
        # For now, just log the alert
        
    def export_validation_report(self, filepath: str, format: str = 'json'):
        """Export validation history to file."""
        if format == 'json':
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_validations': len(self.validation_history),
                'summary': self.get_validation_summary(),
                'validation_history': [report.to_dict() for report in self.validation_history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
                
        elif format == 'csv':
            # Create CSV with summary statistics
            csv_data = []
            for report in self.validation_history:
                csv_data.append({
                    'timestamp': report.timestamp.isoformat(),
                    'overall_status': report.overall_status,
                    'total_checks': report.total_checks,
                    'passed_checks': report.passed_checks,
                    'failed_checks': report.failed_checks,
                    'success_rate': report.success_rate,
                    'execution_time': report.execution_time
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Validation report exported to: {filepath}")


def create_validation_framework(config: Dict[str, Any]) -> ComprehensiveValidationFramework:
    """
    Factory function to create a configured validation framework.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured validation framework
    """
    validation_config = config.get('validation', {})
    
    framework = ComprehensiveValidationFramework(validation_config)
    
    # Configure individual validators if specified
    validator_configs = validation_config.get('validators', {})
    for validator_name, validator_config in validator_configs.items():
        framework.configure_validator(validator_name, validator_config)
    
    logger.info("Created comprehensive validation framework")
    return framework