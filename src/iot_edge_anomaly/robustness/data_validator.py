"""
Data Validation and Quality Assessment for IoT Edge Anomaly Detection.

Provides comprehensive data validation and quality checking including:
- Schema validation for input data
- Range and boundary checks
- Data type validation
- Missing value detection
- Outlier detection and handling
- Data drift detection
- Quality scoring and reporting
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    field: str
    severity: ValidationSeverity
    message: str
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    status: ValidationStatus
    quality_score: float
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


class DataValidator:
    """
    Comprehensive data validation and quality assessment.
    
    Validates input data against defined schemas and quality criteria,
    detecting issues that could impact model performance or system stability.
    """
    
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        enable_drift_detection: bool = True
    ):
        """
        Initialize data validator.
        
        Args:
            schema: Data schema definition
            quality_thresholds: Quality threshold configuration
            enable_drift_detection: Whether to track data drift
        """
        self.schema = schema or self._default_schema()
        self.quality_thresholds = quality_thresholds or self._default_quality_thresholds()
        self.enable_drift_detection = enable_drift_detection
        
        # Data drift tracking
        self.reference_statistics = {}
        self.validation_history = []
        
        logger.info("Data validator initialized")
    
    def _default_schema(self) -> Dict[str, Any]:
        """Default schema for IoT sensor data."""
        return {
            "sensor_data": {
                "type": "tensor",
                "shape": {"min_dims": 2, "max_dims": 3},
                "dtype": "float32",
                "value_range": {"min": -1000.0, "max": 1000.0}
            },
            "edge_index": {
                "type": "tensor",
                "shape": {"dims": 2},
                "dtype": "long",
                "optional": True
            },
            "timestamps": {
                "type": "tensor",
                "dtype": "float64",
                "optional": True
            },
            "metadata": {
                "type": "dict",
                "optional": True
            }
        }
    
    def _default_quality_thresholds(self) -> Dict[str, float]:
        """Default data quality thresholds."""
        return {
            "missing_ratio": 0.05,      # Max 5% missing values
            "outlier_ratio": 0.10,      # Max 10% outliers
            "drift_threshold": 0.15,    # Max 15% distribution drift
            "variance_threshold": 0.01, # Min variance for features
            "correlation_threshold": 0.95 # Max correlation between features
        }
    
    def validate(
        self,
        data: Dict[str, Any],
        update_reference: bool = False
    ) -> ValidationResult:
        """
        Perform comprehensive data validation.
        
        Args:
            data: Input data to validate
            update_reference: Whether to update reference statistics
            
        Returns:
            ValidationResult with status and details
        """
        issues = []
        metadata = {}
        
        # Schema validation
        schema_issues = self._validate_schema(data)
        issues.extend(schema_issues)
        
        # Quality checks
        quality_issues, quality_metadata = self._validate_quality(data)
        issues.extend(quality_issues)
        metadata.update(quality_metadata)
        
        # Data drift detection
        if self.enable_drift_detection:
            drift_issues, drift_metadata = self._detect_drift(data, update_reference)
            issues.extend(drift_issues)
            metadata.update(drift_metadata)
        
        # Compute overall quality score
        quality_score = self._compute_quality_score(issues, metadata)
        
        # Determine validation status
        status = self._determine_status(issues)
        
        # Create validation result
        result = ValidationResult(
            status=status,
            quality_score=quality_score,
            issues=issues,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        # Update history
        self.validation_history.append(result)
        if len(self.validation_history) > 100:  # Keep last 100 validations
            self.validation_history = self.validation_history[-100:]
        
        return result
    
    def _validate_schema(self, data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data against schema."""
        issues = []
        
        for field_name, field_schema in self.schema.items():
            if field_name not in data:
                if not field_schema.get("optional", False):
                    issues.append(ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Required field '{field_name}' is missing"
                    ))
                continue
            
            value = data[field_name]
            
            # Type validation
            if field_schema["type"] == "tensor":
                if not isinstance(value, torch.Tensor):
                    issues.append(ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' must be a tensor",
                        value=type(value).__name__,
                        expected="torch.Tensor"
                    ))
                    continue
                
                # Shape validation
                shape_config = field_schema.get("shape", {})
                if "dims" in shape_config:
                    if value.dim() != shape_config["dims"]:
                        issues.append(ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.ERROR,
                            message=f"Field '{field_name}' has wrong number of dimensions",
                            value=value.dim(),
                            expected=shape_config["dims"]
                        ))
                
                if "min_dims" in shape_config and value.dim() < shape_config["min_dims"]:
                    issues.append(ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' has too few dimensions",
                        value=value.dim(),
                        expected=f"at least {shape_config['min_dims']}"
                    ))
                
                if "max_dims" in shape_config and value.dim() > shape_config["max_dims"]:
                    issues.append(ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' has too many dimensions",
                        value=value.dim(),
                        expected=f"at most {shape_config['max_dims']}"
                    ))
                
                # Data type validation
                expected_dtype = field_schema.get("dtype")
                if expected_dtype:
                    if expected_dtype == "float32" and value.dtype != torch.float32:
                        issues.append(ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field '{field_name}' should be float32",
                            value=str(value.dtype),
                            expected="torch.float32"
                        ))
                    elif expected_dtype == "long" and value.dtype != torch.long:
                        issues.append(ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field '{field_name}' should be long",
                            value=str(value.dtype),
                            expected="torch.long"
                        ))
                
                # Value range validation
                value_range = field_schema.get("value_range", {})
                if "min" in value_range or "max" in value_range:
                    min_val = value.min().item()
                    max_val = value.max().item()
                    
                    if "min" in value_range and min_val < value_range["min"]:
                        issues.append(ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field '{field_name}' has values below minimum",
                            value=min_val,
                            expected=f">= {value_range['min']}"
                        ))
                    
                    if "max" in value_range and max_val > value_range["max"]:
                        issues.append(ValidationIssue(
                            field=field_name,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field '{field_name}' has values above maximum",
                            value=max_val,
                            expected=f"<= {value_range['max']}"
                        ))
            
            elif field_schema["type"] == "dict":
                if not isinstance(value, dict):
                    issues.append(ValidationIssue(
                        field=field_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field_name}' must be a dictionary",
                        value=type(value).__name__,
                        expected="dict"
                    ))
        
        return issues
    
    def _validate_quality(self, data: Dict[str, Any]) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Validate data quality metrics."""
        issues = []
        metadata = {}
        
        # Focus on main sensor data
        if "sensor_data" in data:
            sensor_data = data["sensor_data"]
            if isinstance(sensor_data, torch.Tensor):
                
                # Missing value detection
                if torch.isnan(sensor_data).any():
                    nan_ratio = torch.isnan(sensor_data).float().mean().item()
                    metadata["missing_ratio"] = nan_ratio
                    
                    if nan_ratio > self.quality_thresholds["missing_ratio"]:
                        issues.append(ValidationIssue(
                            field="sensor_data",
                            severity=ValidationSeverity.WARNING,
                            message=f"High missing value ratio: {nan_ratio:.3f}",
                            value=nan_ratio,
                            expected=f"<= {self.quality_thresholds['missing_ratio']}"
                        ))
                
                # Outlier detection using IQR method
                if sensor_data.numel() > 10:
                    outlier_ratio = self._detect_outliers(sensor_data)
                    metadata["outlier_ratio"] = outlier_ratio
                    
                    if outlier_ratio > self.quality_thresholds["outlier_ratio"]:
                        issues.append(ValidationIssue(
                            field="sensor_data",
                            severity=ValidationSeverity.WARNING,
                            message=f"High outlier ratio: {outlier_ratio:.3f}",
                            value=outlier_ratio,
                            expected=f"<= {self.quality_thresholds['outlier_ratio']}"
                        ))
                
                # Variance check
                if sensor_data.dim() >= 2:
                    feature_vars = sensor_data.var(dim=0)
                    low_var_features = (feature_vars < self.quality_thresholds["variance_threshold"]).sum().item()
                    metadata["low_variance_features"] = low_var_features
                    
                    if low_var_features > 0:
                        issues.append(ValidationIssue(
                            field="sensor_data",
                            severity=ValidationSeverity.INFO,
                            message=f"{low_var_features} features have low variance",
                            value=low_var_features
                        ))
                
                # Basic statistics
                metadata.update({
                    "data_shape": list(sensor_data.shape),
                    "mean": sensor_data.mean().item(),
                    "std": sensor_data.std().item(),
                    "min": sensor_data.min().item(),
                    "max": sensor_data.max().item()
                })
        
        return issues, metadata
    
    def _detect_outliers(self, data: torch.Tensor) -> float:
        """Detect outliers using IQR method."""
        # Flatten data for outlier detection
        flat_data = data.flatten()
        
        # Remove NaN values
        valid_data = flat_data[~torch.isnan(flat_data)]
        
        if len(valid_data) < 4:
            return 0.0
        
        # Compute quartiles
        q1 = torch.quantile(valid_data, 0.25)
        q3 = torch.quantile(valid_data, 0.75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Count outliers
        outliers = (valid_data < lower_bound) | (valid_data > upper_bound)
        outlier_ratio = outliers.float().mean().item()
        
        return outlier_ratio
    
    def _detect_drift(
        self,
        data: Dict[str, Any],
        update_reference: bool
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Detect data drift compared to reference distribution."""
        issues = []
        metadata = {}
        
        if "sensor_data" not in data:
            return issues, metadata
        
        sensor_data = data["sensor_data"]
        if not isinstance(sensor_data, torch.Tensor):
            return issues, metadata
        
        # Compute current statistics
        current_stats = self._compute_statistics(sensor_data)
        
        # Initialize reference if not available
        if not self.reference_statistics and update_reference:
            self.reference_statistics = current_stats
            metadata["drift_detection"] = "reference_initialized"
            return issues, metadata
        
        if not self.reference_statistics:
            metadata["drift_detection"] = "no_reference"
            return issues, metadata
        
        # Compare with reference
        drift_score = self._compute_drift_score(current_stats, self.reference_statistics)
        metadata["drift_score"] = drift_score
        
        if drift_score > self.quality_thresholds["drift_threshold"]:
            issues.append(ValidationIssue(
                field="sensor_data",
                severity=ValidationSeverity.WARNING,
                message=f"Data drift detected: {drift_score:.3f}",
                value=drift_score,
                expected=f"<= {self.quality_thresholds['drift_threshold']}"
            ))
        
        # Update reference if requested
        if update_reference:
            # Exponential moving average update
            alpha = 0.1
            for key in self.reference_statistics:
                if key in current_stats:
                    self.reference_statistics[key] = (
                        alpha * current_stats[key] + 
                        (1 - alpha) * self.reference_statistics[key]
                    )
        
        return issues, metadata
    
    def _compute_statistics(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute statistical features for drift detection."""
        # Remove NaN values
        valid_data = data[~torch.isnan(data)]
        
        if len(valid_data) == 0:
            return {}
        
        stats = {
            "mean": valid_data.mean().item(),
            "std": valid_data.std().item(),
            "min": valid_data.min().item(),
            "max": valid_data.max().item(),
            "q25": torch.quantile(valid_data, 0.25).item(),
            "q50": torch.quantile(valid_data, 0.50).item(),
            "q75": torch.quantile(valid_data, 0.75).item()
        }
        
        return stats
    
    def _compute_drift_score(
        self,
        current_stats: Dict[str, float],
        reference_stats: Dict[str, float]
    ) -> float:
        """Compute drift score between current and reference statistics."""
        if not current_stats or not reference_stats:
            return 0.0
        
        drift_scores = []
        
        for key in ["mean", "std", "q25", "q50", "q75"]:
            if key in current_stats and key in reference_stats:
                current_val = current_stats[key]
                reference_val = reference_stats[key]
                
                # Avoid division by zero
                if abs(reference_val) < 1e-8:
                    if abs(current_val) < 1e-8:
                        drift_scores.append(0.0)
                    else:
                        drift_scores.append(1.0)
                else:
                    relative_change = abs(current_val - reference_val) / abs(reference_val)
                    drift_scores.append(relative_change)
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _compute_quality_score(
        self,
        issues: List[ValidationIssue],
        metadata: Dict[str, Any]
    ) -> float:
        """Compute overall data quality score (0-1)."""
        # Start with perfect score
        score = 1.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.4
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.2
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.1
            elif issue.severity == ValidationSeverity.INFO:
                score -= 0.05
        
        # Additional deductions based on quality metrics
        if "missing_ratio" in metadata:
            score -= metadata["missing_ratio"] * 0.5
        
        if "outlier_ratio" in metadata:
            score -= metadata["outlier_ratio"] * 0.3
        
        if "drift_score" in metadata:
            score -= metadata["drift_score"] * 0.2
        
        return max(0.0, score)
    
    def _determine_status(self, issues: List[ValidationIssue]) -> ValidationStatus:
        """Determine overall validation status."""
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        has_warnings = any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        
        if has_critical or has_errors:
            return ValidationStatus.FAILED
        elif has_warnings:
            return ValidationStatus.PASSED_WITH_WARNINGS
        else:
            return ValidationStatus.PASSED
    
    def update_schema(self, schema: Dict[str, Any]):
        """Update validation schema."""
        self.schema = schema
        logger.info("Validation schema updated")
    
    def update_quality_thresholds(self, thresholds: Dict[str, float]):
        """Update quality thresholds."""
        self.quality_thresholds.update(thresholds)
        logger.info("Quality thresholds updated")
    
    def reset_reference(self):
        """Reset reference statistics for drift detection."""
        self.reference_statistics = {}
        logger.info("Reference statistics reset")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of recent validation results."""
        if not self.validation_history:
            return {"status": "no_validations", "message": "No validation history available"}
        
        recent_validations = self.validation_history[-10:]  # Last 10 validations
        
        avg_quality = np.mean([v.quality_score for v in recent_validations])
        
        status_counts = {}
        for status in ValidationStatus:
            status_counts[status.value] = sum(
                1 for v in recent_validations if v.status == status
            )
        
        # Count issue types
        issue_counts = {}
        for severity in ValidationSeverity:
            issue_counts[severity.value] = sum(
                sum(1 for issue in v.issues if issue.severity == severity)
                for v in recent_validations
            )
        
        return {
            "average_quality_score": avg_quality,
            "total_validations": len(recent_validations),
            "status_distribution": status_counts,
            "issue_distribution": issue_counts,
            "latest_validation": {
                "status": recent_validations[-1].status.value,
                "quality_score": recent_validations[-1].quality_score,
                "issue_count": len(recent_validations[-1].issues),
                "timestamp": recent_validations[-1].timestamp.isoformat()
            }
        }