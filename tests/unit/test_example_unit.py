"""Example unit tests demonstrating testing patterns."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from tests.fixtures.test_data import (
    MockLSTMAutoencoder, 
    generate_lstm_input_batch,
    create_mock_model_config
)


class TestLSTMAutoencoderUnit:
    """Unit tests for LSTM Autoencoder component."""
    
    def test_model_initialization(self):
        """Test that model initializes with correct parameters."""
        model = MockLSTMAutoencoder(input_size=10, hidden_size=64)
        
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert not model.training  # Should start in eval mode
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = MockLSTMAutoencoder(input_size=10, hidden_size=64)
        input_data = generate_lstm_input_batch(
            batch_size=32, 
            sequence_length=10, 
            num_features=10
        )
        
        output = model(input_data)
        
        assert output.shape == input_data.shape
        assert output.dtype == torch.float32
    
    def test_training_mode_toggle(self):
        """Test that training mode can be toggled correctly."""
        model = MockLSTMAutoencoder()
        
        # Test training mode
        model.train(True)
        assert model.training is True
        
        # Test eval mode
        model.eval()
        assert model.training is False
    
    @pytest.mark.parametrize("batch_size,seq_len,features", [
        (1, 10, 5),
        (16, 20, 10),
        (64, 5, 15),
    ])
    def test_various_input_shapes(self, batch_size, seq_len, features):
        """Test model handles various input shapes correctly."""
        model = MockLSTMAutoencoder(input_size=features)
        input_data = generate_lstm_input_batch(
            batch_size=batch_size,
            sequence_length=seq_len, 
            num_features=features
        )
        
        output = model(input_data)
        assert output.shape == (batch_size, seq_len, features)


class TestAnomalyDetectionUnit:
    """Unit tests for anomaly detection logic."""
    
    def test_threshold_based_detection(self):
        """Test basic threshold-based anomaly detection."""
        threshold = 0.5
        reconstruction_errors = torch.tensor([0.2, 0.8, 0.3, 0.9, 0.1])
        expected_anomalies = torch.tensor([False, True, False, True, False])
        
        # Mock anomaly detector
        anomalies = reconstruction_errors > threshold
        
        assert torch.equal(anomalies, expected_anomalies)
    
    def test_reconstruction_error_calculation(self):
        """Test MSE reconstruction error calculation."""
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        reconstructed = torch.tensor([[1.1, 2.1], [2.9, 4.1]])
        
        # Calculate MSE manually
        mse = torch.mean((original - reconstructed) ** 2, dim=1)
        expected_mse = torch.tensor([0.01, 0.01])  # (0.1^2 + 0.1^2) / 2
        
        assert torch.allclose(mse, expected_mse, atol=1e-6)
    
    @patch('time.time')
    def test_cooldown_period(self, mock_time):
        """Test anomaly detection cooldown period."""
        # Mock time progression
        mock_time.side_effect = [0, 100, 200, 400]  # 4 calls to time.time()
        
        cooldown_seconds = 300  # 5 minutes
        last_alert_time = 0
        
        def should_alert(current_time, last_time, cooldown):
            return (current_time - last_time) >= cooldown
        
        # Test alerts
        assert should_alert(100, 0, cooldown_seconds) is False  # Too soon
        assert should_alert(200, 0, cooldown_seconds) is False  # Still too soon  
        assert should_alert(400, 0, cooldown_seconds) is True   # Enough time passed


class TestDataProcessingUnit:
    """Unit tests for data processing utilities."""
    
    def test_normalize_data(self):
        """Test data normalization function."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        
        # Mock normalization (z-score)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / std
        
        # Verify properties of normalized data
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-6)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-6)
    
    def test_sequence_generation(self):
        """Test sliding window sequence generation."""
        data = np.arange(10)  # [0, 1, 2, ..., 9]
        sequence_length = 3
        
        # Mock sequence generation
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length].tolist())
        
        expected_sequences = [
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
            [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]
        ]
        
        assert sequences == expected_sequences
    
    def test_missing_data_handling(self):
        """Test handling of missing data points."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Mock missing data strategies
        # Strategy 1: Forward fill
        forward_filled = np.copy(data_with_nan)
        for i in range(1, len(forward_filled)):
            if np.isnan(forward_filled[i]):
                forward_filled[i] = forward_filled[i-1]
        
        expected_forward = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
        assert np.array_equal(forward_filled, expected_forward)
        
        # Strategy 2: Linear interpolation
        mask = ~np.isnan(data_with_nan)
        interpolated = np.copy(data_with_nan)
        interpolated[2] = (2.0 + 4.0) / 2  # Simple linear interpolation
        
        expected_interp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(interpolated, expected_interp)


class TestConfigurationUnit:
    """Unit tests for configuration management."""
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        config = create_mock_model_config()
        
        # Test required fields exist
        assert 'model' in config
        assert 'training' in config
        assert 'anomaly_detection' in config
        
        # Test model parameters
        model_config = config['model']
        assert model_config['input_size'] > 0
        assert model_config['hidden_size'] > 0
        assert 0 <= model_config['dropout'] <= 1
    
    def test_config_defaults(self):
        """Test configuration default values."""
        config = create_mock_model_config()
        
        # Test default values are sensible
        assert config['model']['sequence_length'] > 0
        assert config['training']['learning_rate'] > 0
        assert config['training']['batch_size'] > 0
        assert config['anomaly_detection']['threshold'] >= 0
    
    @pytest.mark.parametrize("invalid_config", [
        {'model': {'input_size': 0}},  # Invalid input size
        {'training': {'learning_rate': -0.01}},  # Negative learning rate
        {'anomaly_detection': {'threshold': -1}},  # Negative threshold
    ])
    def test_invalid_config_handling(self, invalid_config):
        """Test handling of invalid configuration values."""
        # Mock config validation function
        def validate_config(config_dict):
            if 'model' in config_dict:
                if config_dict['model'].get('input_size', 1) <= 0:
                    raise ValueError("input_size must be positive")
            if 'training' in config_dict:
                if config_dict['training'].get('learning_rate', 0.001) <= 0:
                    raise ValueError("learning_rate must be positive")
            if 'anomaly_detection' in config_dict:
                if config_dict['anomaly_detection'].get('threshold', 0.5) < 0:
                    raise ValueError("threshold must be non-negative")
        
        with pytest.raises(ValueError):
            validate_config(invalid_config)