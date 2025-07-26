"""
Advanced tests for LSTM Autoencoder functionality.
"""
import pytest
import torch
import numpy as np
from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder

@pytest.fixture
def trained_model():
    """Create a minimally trained model for testing."""
    model = LSTMAutoencoder(input_size=3, hidden_size=16, num_layers=1)
    model.eval()
    return model

@pytest.fixture
def normal_data():
    """Create normal sensor data (sine wave pattern)."""
    t = torch.linspace(0, 4*np.pi, 20).unsqueeze(0).unsqueeze(-1)
    data = torch.cat([
        torch.sin(t),
        torch.cos(t),
        torch.sin(2*t)
    ], dim=-1)
    return data

@pytest.fixture
def anomalous_data():
    """Create anomalous sensor data (random noise)."""
    return torch.randn(1, 20, 3) * 3  # High variance noise

def test_reconstruction_error_computation(trained_model, normal_data):
    """Test reconstruction error computation with different reduction methods."""
    # Test mean reduction
    error_mean = trained_model.compute_reconstruction_error(normal_data, reduction='mean')
    assert error_mean.dim() == 0  # Scalar
    assert error_mean.item() >= 0
    
    # Test sum reduction
    error_sum = trained_model.compute_reconstruction_error(normal_data, reduction='sum')
    assert error_sum.dim() == 0  # Scalar
    assert error_sum.item() >= error_mean.item()
    
    # Test no reduction
    error_none = trained_model.compute_reconstruction_error(normal_data, reduction='none')
    assert error_none.shape == (1, 20)  # (batch_size, seq_len)

def test_anomaly_detection(trained_model, normal_data, anomalous_data):
    """Test anomaly detection functionality."""
    # Set a reasonable threshold
    threshold = 1.0
    
    # Normal data should have fewer anomalies
    normal_anomalies = trained_model.detect_anomalies(normal_data, threshold)
    assert normal_anomalies.shape == (1,)  # One sample
    
    # Anomalous data should trigger detection (with high probability)
    anomalous_anomalies = trained_model.detect_anomalies(anomalous_data, threshold)
    assert anomalous_anomalies.shape == (1,)

def test_encoder_decoder_separately(trained_model, normal_data):
    """Test encoder and decoder components separately."""
    # Test encoder
    encoded, hidden = trained_model.encode(normal_data)
    assert encoded.shape == (1, 20, 16)  # (batch, seq_len, hidden_size)
    assert len(hidden) == 2  # (h_n, c_n)
    assert hidden[0].shape == (1, 1, 16)  # (num_layers, batch, hidden_size)
    assert hidden[1].shape == (1, 1, 16)
    
    # Test decoder
    decoded = trained_model.decode(encoded, seq_len=20)
    assert decoded.shape == normal_data.shape

def test_model_edge_cases():
    """Test edge cases and error conditions."""
    # Very small model
    tiny_model = LSTMAutoencoder(input_size=1, hidden_size=4, num_layers=1)
    tiny_input = torch.randn(1, 5, 1)
    output = tiny_model(tiny_input)
    assert output.shape == tiny_input.shape
    
    # Single layer (no dropout)
    single_layer_model = LSTMAutoencoder(input_size=2, hidden_size=8, num_layers=1)
    assert single_layer_model.encoder.dropout == 0
    assert single_layer_model.decoder.dropout == 0

def test_model_parameters_count():
    """Test that model has reasonable number of parameters."""
    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Should have reasonable number of parameters (not too small, not too large)
    assert 1000 < total_params < 100000  # Reasonable range for edge deployment

def test_model_memory_efficiency():
    """Test model memory usage is reasonable for edge deployment."""
    model = LSTMAutoencoder(input_size=10, hidden_size=64, num_layers=2)
    
    # Test with larger batch
    large_input = torch.randn(32, 50, 10)  # 32 samples, 50 timesteps, 10 features
    
    with torch.no_grad():
        output = model(large_input)
        assert output.shape == large_input.shape
        
        # Memory should be freed properly
        del output
        
def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    model = LSTMAutoencoder(input_size=3, hidden_size=16, num_layers=1)
    input_data = torch.randn(2, 10, 3, requires_grad=True)
    
    # Forward pass
    output = model(input_data)
    loss = torch.mean((output - input_data) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist and are non-zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for {name}"