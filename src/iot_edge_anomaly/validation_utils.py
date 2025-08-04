"""
Validation utilities for IoT Edge Anomaly Detection.

This module provides comprehensive input validation, data sanitization,
and error handling utilities to ensure robust operation.
"""
import torch
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result status."""
    VALID = "valid"
    CORRECTABLE = "correctable"
    INVALID = "invalid"


@dataclass
class ValidationResponse:
    """Response from data validation."""
    status: ValidationResult
    message: str
    corrected_data: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class TensorValidator:
    """Comprehensive tensor validation and sanitization."""
    
    def __init__(self, 
                 min_value: float = -1000.0,
                 max_value: float = 1000.0,
                 allow_zero_variance: bool = False):
        """
        Initialize tensor validator.
        
        Args:
            min_value: Minimum allowed value for tensor elements
            max_value: Maximum allowed value for tensor elements
            allow_zero_variance: Whether to allow tensors with zero variance
        """
        self.min_value = min_value
        self.max_value = max_value
        self.allow_zero_variance = allow_zero_variance
    
    def validate_tensor(self, tensor: torch.Tensor, 
                       expected_shape: Optional[Tuple[int, ...]] = None,
                       tensor_name: str = "tensor") -> ValidationResponse:
        """
        Comprehensive tensor validation.
        
        Args:
            tensor: Tensor to validate
            expected_shape: Expected tensor shape (None to skip shape check)
            tensor_name: Name of tensor for error messages
            
        Returns:
            ValidationResponse with status and details
        """
        if not isinstance(tensor, torch.Tensor):
            return ValidationResponse(
                ValidationResult.INVALID,
                f"{tensor_name} must be a torch.Tensor, got {type(tensor)}"
            )
        
        # Check for empty tensor
        if tensor.numel() == 0:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"{tensor_name} is empty"
            )
        
        # Check shape if specified
        if expected_shape is not None and tensor.shape != expected_shape:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"{tensor_name} shape {tensor.shape} doesn't match expected {expected_shape}"
            )
        
        # Check for NaN values
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total_elements = tensor.numel()
            
            if nan_count == total_elements:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"{tensor_name} contains only NaN values"
                )
            elif nan_count / total_elements > 0.5:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"{tensor_name} contains {nan_count}/{total_elements} NaN values (>50%)"
                )
            else:
                # Try to correct by replacing NaN with median
                corrected = tensor.clone()
                median_val = torch.nanmedian(tensor).item()
                corrected[torch.isnan(corrected)] = median_val
                
                return ValidationResponse(
                    ValidationResult.CORRECTABLE,
                    f"{tensor_name} contained {nan_count} NaN values, replaced with median ({median_val:.4f})",
                    corrected_data=corrected,
                    metadata={"nan_count": nan_count, "replacement_value": median_val}
                )
        
        # Check for infinite values
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            total_elements = tensor.numel()
            
            if inf_count / total_elements > 0.1:  # More than 10% infinite
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"{tensor_name} contains {inf_count}/{total_elements} infinite values (>10%)"
                )
            else:
                # Try to correct by clamping
                corrected = torch.clamp(tensor, self.min_value, self.max_value)
                
                return ValidationResponse(
                    ValidationResult.CORRECTABLE,
                    f"{tensor_name} contained {inf_count} infinite values, clamped to [{self.min_value}, {self.max_value}]",
                    corrected_data=corrected,
                    metadata={"inf_count": inf_count}
                )
        
        # Check value range
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if min_val < self.min_value or max_val > self.max_value:
            # Try to correct by clamping
            corrected = torch.clamp(tensor, self.min_value, self.max_value)
            
            return ValidationResponse(
                ValidationResult.CORRECTABLE,
                f"{tensor_name} values [{min_val:.4f}, {max_val:.4f}] outside allowed range [{self.min_value}, {self.max_value}], clamped",
                corrected_data=corrected,
                metadata={"original_min": min_val, "original_max": max_val}
            )
        
        # Check for zero variance (all same values)
        if not self.allow_zero_variance:
            if tensor.numel() > 1:
                variance = torch.var(tensor).item()
                if variance < 1e-12:  # Essentially zero variance
                    return ValidationResponse(
                        ValidationResult.CORRECTABLE,
                        f"{tensor_name} has zero variance (all values ≈ {tensor.mean().item():.6f})",
                        corrected_data=tensor,  # Don't modify, but flag as correctable
                        metadata={"variance": variance, "mean": tensor.mean().item()}
                    )
        
        return ValidationResponse(
            ValidationResult.VALID,
            f"{tensor_name} validation passed",
            metadata={
                "shape": list(tensor.shape),
                "min": min_val,
                "max": max_val,
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }
        )
    
    def validate_time_series(self, data: torch.Tensor) -> ValidationResponse:
        """
        Validate time series data specifically.
        
        Args:
            data: Time series data of shape (batch, seq_len, features)
            
        Returns:
            ValidationResponse
        """
        if data.dim() != 3:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Time series data must be 3D (batch, seq_len, features), got {data.dim()}D"
            )
        
        batch_size, seq_len, num_features = data.shape
        
        # Basic tensor validation
        basic_validation = self.validate_tensor(data, tensor_name="time_series_data")
        if basic_validation.status == ValidationResult.INVALID:
            return basic_validation
        
        # Check for reasonable dimensions
        if batch_size > 1000:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Batch size {batch_size} too large (>1000)"
            )
        
        if seq_len < 2:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Sequence length {seq_len} too short (<2)"
            )
        
        if num_features > 100:
            return ValidationResponse(
                ValidationResult.CORRECTABLE,
                f"Large number of features ({num_features}), may impact performance",
                corrected_data=data,
                metadata={"feature_count": num_features}
            )
        
        # Check for temporal consistency (no huge jumps)
        if seq_len > 1:
            diffs = torch.diff(data, dim=1)
            max_diff = torch.abs(diffs).max().item()
            mean_diff = torch.abs(diffs).mean().item()
            
            # If max difference is much larger than mean, might indicate data issues
            if max_diff > mean_diff * 100:  # More than 100x the average change
                return ValidationResponse(
                    ValidationResult.CORRECTABLE,
                    f"Large temporal jumps detected (max: {max_diff:.4f}, avg: {mean_diff:.4f})",
                    corrected_data=data,
                    metadata={"max_temporal_diff": max_diff, "mean_temporal_diff": mean_diff}
                )
        
        return ValidationResponse(
            ValidationResult.VALID,
            "Time series validation passed",
            metadata={
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "num_features": num_features,
                "temporal_stability": mean_diff if seq_len > 1 else None
            }
        )


class ModelInputValidator:
    """Validator for model inputs with auto-correction capabilities."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize model input validator.
        
        Args:
            model_config: Model configuration dictionary
        """
        self.expected_input_size = model_config.get('input_size', 5)
        self.max_sequence_length = model_config.get('max_sequence_length', 1000)
        self.tensor_validator = TensorValidator()
        
    def validate_lstm_input(self, data: torch.Tensor) -> ValidationResponse:
        """
        Validate input for LSTM autoencoder.
        
        Args:
            data: Input tensor for LSTM
            
        Returns:
            ValidationResponse with validation results
        """
        # Basic time series validation
        ts_validation = self.tensor_validator.validate_time_series(data)
        if ts_validation.status == ValidationResult.INVALID:
            return ts_validation
        
        batch_size, seq_len, num_features = data.shape
        
        # Check feature dimension matches expected
        if num_features != self.expected_input_size:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Input features {num_features} don't match expected {self.expected_input_size}"
            )
        
        # Check sequence length
        if seq_len > self.max_sequence_length:
            # Truncate to max length
            corrected = data[:, -self.max_sequence_length:, :]
            return ValidationResponse(
                ValidationResult.CORRECTABLE,
                f"Sequence length {seq_len} > max {self.max_sequence_length}, truncated",
                corrected_data=corrected,
                metadata={"original_length": seq_len, "truncated_length": self.max_sequence_length}
            )
        
        return ValidationResponse(
            ValidationResult.VALID,
            "LSTM input validation passed",
            metadata=ts_validation.metadata
        )
    
    def validate_gnn_input(self, node_features: torch.Tensor, 
                          edge_index: torch.Tensor) -> ValidationResponse:
        """
        Validate input for GNN layer.
        
        Args:
            node_features: Node feature tensor
            edge_index: Edge index tensor
            
        Returns:
            ValidationResponse
        """
        # Validate node features
        if node_features.dim() != 2:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Node features must be 2D (num_nodes, features), got {node_features.dim()}D"
            )
        
        num_nodes, feature_dim = node_features.shape
        
        # Basic tensor validation for node features
        node_validation = self.tensor_validator.validate_tensor(node_features, tensor_name="node_features")
        if node_validation.status == ValidationResult.INVALID:
            return node_validation
        
        # Validate edge index
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Edge index must be shape (2, num_edges), got {edge_index.shape}"
            )
        
        # Check edge indices are valid
        if edge_index.numel() > 0:
            max_node_idx = edge_index.max().item()
            min_node_idx = edge_index.min().item()
            
            if min_node_idx < 0:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Edge indices contain negative values (min: {min_node_idx})"
                )
            
            if max_node_idx >= num_nodes:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Edge indices ({max_node_idx}) exceed number of nodes ({num_nodes})"
                )
        
        return ValidationResponse(
            ValidationResult.VALID,
            "GNN input validation passed",
            metadata={
                "num_nodes": num_nodes,
                "feature_dim": feature_dim,
                "num_edges": edge_index.size(1) if edge_index.numel() > 0 else 0
            }
        )


def sanitize_sensor_data(data: Union[np.ndarray, torch.Tensor], 
                        config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Sanitize sensor data with robust error handling.
    
    Args:
        data: Input sensor data
        config: Optional configuration for sanitization
        
    Returns:
        Sanitized torch.Tensor
        
    Raises:
        ValueError: If data cannot be sanitized
    """
    if config is None:
        config = {}
    
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        try:
            data = torch.tensor(data).float()
        except Exception as e:
            raise ValueError(f"Cannot convert data to tensor: {e}")
    
    # Ensure float type
    if not data.dtype.is_floating_point:
        data = data.float()
    
    # Apply sanitization
    validator = TensorValidator(
        min_value=config.get('min_value', -1000.0),
        max_value=config.get('max_value', 1000.0)
    )
    
    validation = validator.validate_tensor(data, tensor_name="sensor_data")
    
    if validation.status == ValidationResult.INVALID:
        raise ValueError(f"Data validation failed: {validation.message}")
    elif validation.status == ValidationResult.CORRECTABLE:
        logger.warning(f"Data corrected: {validation.message}")
        return validation.corrected_data
    
    return data


def safe_tensor_operation(operation, *args, **kwargs) -> Tuple[bool, Union[torch.Tensor, str]]:
    """
    Safely execute tensor operations with error handling.
    
    Args:
        operation: Function to execute
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Tuple of (success: bool, result_or_error_message)
    """
    try:
        result = operation(*args, **kwargs)
        
        # Validate result if it's a tensor
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                return False, "Operation produced NaN values"
            if torch.isinf(result).any():
                return False, "Operation produced infinite values"
        
        return True, result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False, f"GPU out of memory: {e}"
        else:
            return False, f"Runtime error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"