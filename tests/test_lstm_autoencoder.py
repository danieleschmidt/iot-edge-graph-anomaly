"""
Test suite for LSTM Autoencoder core functionality.
"""
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture
def mock_sensor_data():
    """Create mock time-series sensor data for testing."""
    # Shape: (batch_size, sequence_length, features)
    batch_size, seq_len, n_features = 2, 10, 5
    return torch.randn(batch_size, seq_len, n_features)

def test_lstm_autoencoder_import():
    """Test that LSTM autoencoder can be imported."""
    try:
        from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    except ImportError:
        pytest.fail("LSTMAutoencoder cannot be imported")

def test_lstm_autoencoder_initialization():
    """Test LSTM autoencoder can be initialized with proper parameters."""
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    input_size = 5
    hidden_size = 64
    num_layers = 2
    
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    assert model is not None
    assert model.input_size == input_size
    assert model.hidden_size == hidden_size
    assert model.num_layers == num_layers

def test_lstm_autoencoder_forward_pass(mock_sensor_data):
    """Test LSTM autoencoder forward pass with mock data."""
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    input_size = mock_sensor_data.shape[-1]  # Number of features
    model = LSTMAutoencoder(input_size=input_size, hidden_size=32, num_layers=1)
    
    # Forward pass
    output = model(mock_sensor_data)
    
    # Output should have same shape as input
    assert output.shape == mock_sensor_data.shape
    assert output.dtype == torch.float32

def test_lstm_autoencoder_reconstruction_loss(mock_sensor_data):
    """Test that reconstruction loss can be calculated."""
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    input_size = mock_sensor_data.shape[-1]
    model = LSTMAutoencoder(input_size=input_size, hidden_size=32, num_layers=1)
    
    # Forward pass
    reconstruction = model(mock_sensor_data)
    
    # Calculate MSE loss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(reconstruction, mock_sensor_data)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0  # MSE loss should be non-negative

def test_lstm_autoencoder_training_mode():
    """Test that model can be set to training and evaluation modes."""
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
    
    # Test training mode
    model.train()
    assert model.training is True
    
    # Test evaluation mode
    model.eval()
    assert model.training is False

def test_lstm_autoencoder_save_load():
    """Test that model can be saved and loaded."""
    from iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
    
    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
    
    # Save model
    save_path = "/tmp/test_model.pth"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    loaded_model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=1)
    loaded_model.load_state_dict(torch.load(save_path, map_location='cpu'))
    
    # Models should have same parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
    
    # Clean up
    Path(save_path).unlink(missing_ok=True)