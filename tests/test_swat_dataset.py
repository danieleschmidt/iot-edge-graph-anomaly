"""
Test suite for SWaT dataset integration functionality.
"""
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

@pytest.fixture
def mock_swat_data():
    """Create mock SWaT dataset for testing."""
    # Create synthetic time-series data similar to SWaT format
    np.random.seed(42)
    n_samples = 1000
    n_features = 51  # SWaT has 51 sensor readings
    
    # Generate synthetic normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some anomalies (last 100 samples)
    anomaly_data = normal_data.copy()
    anomaly_data[-100:, :10] += np.random.normal(3, 0.5, (100, 10))  # Anomalies in first 10 features
    
    # Create labels (0 = normal, 1 = anomaly)
    labels = np.zeros(n_samples)
    labels[-100:] = 1
    
    # Create DataFrame similar to SWaT format
    columns = [f"sensor_{i:02d}" for i in range(n_features)] + ["label"]
    data = np.column_stack([anomaly_data, labels])
    
    return pd.DataFrame(data, columns=columns)

def test_swat_dataset_loader_import():
    """Test that SWaT dataset loader can be imported."""
    try:
        from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    except ImportError:
        pytest.fail("SWaTDataLoader cannot be imported")

def test_swat_dataset_initialization():
    """Test SWaT dataset loader initialization."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {
        'data_path': '/tmp/swat_data.csv',
        'sequence_length': 20,
        'test_split': 0.2,
        'validation_split': 0.1
    }
    
    loader = SWaTDataLoader(config)
    assert loader.sequence_length == 20
    assert loader.test_split == 0.2
    assert loader.validation_split == 0.1

def test_swat_data_preprocessing(mock_swat_data):
    """Test data preprocessing functionality."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    # Save mock data to temporary file
    temp_file = "/tmp/test_swat_data.csv"
    mock_swat_data.to_csv(temp_file, index=False)
    
    config = {
        'data_path': temp_file,
        'sequence_length': 10,
        'normalize': True
    }
    
    loader = SWaTDataLoader(config)
    processed_data = loader.preprocess_data(mock_swat_data)
    
    # Check normalization (features should be roughly zero-mean, unit variance)
    feature_data = processed_data.iloc[:, :-1]  # Exclude label column
    assert abs(feature_data.mean().mean()) < 0.1  # Close to zero mean
    assert abs(feature_data.std().mean() - 1.0) < 0.1  # Close to unit variance
    
    # Clean up
    Path(temp_file).unlink(missing_ok=True)

def test_sequence_generation(mock_swat_data):
    """Test time-series sequence generation."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {
        'sequence_length': 15,
        'step_size': 1
    }
    
    loader = SWaTDataLoader(config)
    sequences, labels = loader.create_sequences(mock_swat_data)
    
    # Check sequence shapes
    expected_num_sequences = len(mock_swat_data) - 15 + 1
    assert sequences.shape[0] == expected_num_sequences
    assert sequences.shape[1] == 15  # sequence_length
    assert sequences.shape[2] == 51  # n_features (excluding label)
    assert len(labels) == expected_num_sequences

def test_train_validation_test_split(mock_swat_data):
    """Test train/validation/test split functionality."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {
        'sequence_length': 10,
        'test_split': 0.2,
        'validation_split': 0.15
    }
    
    loader = SWaTDataLoader(config)
    sequences, labels = loader.create_sequences(mock_swat_data)
    
    train_data, val_data, test_data = loader.split_data(sequences, labels)
    
    # Check split sizes (allow some variance due to stratification)
    total_samples = len(sequences)
    expected_test_size = int(total_samples * 0.2)
    expected_val_size = int((total_samples - expected_test_size) * 0.15)
    expected_train_size = total_samples - expected_test_size - expected_val_size
    
    # Allow ±5% variance due to stratified sampling
    assert abs(len(train_data['sequences']) - expected_train_size) <= max(1, total_samples * 0.05)
    assert abs(len(val_data['sequences']) - expected_val_size) <= max(1, total_samples * 0.05)
    assert abs(len(test_data['sequences']) - expected_test_size) <= max(1, total_samples * 0.05)
    
    # Ensure total adds up
    total_split = len(train_data['sequences']) + len(val_data['sequences']) + len(test_data['sequences'])
    assert total_split == total_samples

def test_pytorch_dataset_creation(mock_swat_data):
    """Test PyTorch dataset creation."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {'sequence_length': 8}
    loader = SWaTDataLoader(config)
    
    sequences, labels = loader.create_sequences(mock_swat_data)
    dataset = loader.create_pytorch_dataset(sequences, labels)
    
    # Test dataset length and item access
    assert len(dataset) == len(sequences)
    
    # Test first item
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2  # (sequence, label)
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)
    assert sample[0].shape == (8, 51)  # (seq_len, n_features)

def test_dataloader_creation(mock_swat_data):
    """Test PyTorch DataLoader creation."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {
        'sequence_length': 10,
        'batch_size': 32,
        'shuffle': True
    }
    
    loader = SWaTDataLoader(config)
    sequences, labels = loader.create_sequences(mock_swat_data)
    
    dataloader = loader.create_dataloader(sequences, labels, batch_size=32, shuffle=True)
    
    # Test dataloader properties
    assert dataloader.batch_size == 32
    
    # Test batch retrieval
    batch = next(iter(dataloader))
    assert isinstance(batch, (tuple, list))  # Can be either tuple or list
    assert len(batch) == 2
    
    # Extract sequences and labels
    sequences, labels = batch[0], batch[1]
    assert sequences.shape[0] <= 32  # Batch size
    assert sequences.shape[1] == 10  # Sequence length
    assert sequences.shape[2] == 51  # Number of features

def test_anomaly_detection_validation(mock_swat_data):
    """Test anomaly detection metrics calculation."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    config = {'sequence_length': 5}
    loader = SWaTDataLoader(config)
    
    # Create sequences
    sequences, labels = loader.create_sequences(mock_swat_data)
    
    # Mock predictions (higher scores for anomalies)
    predictions = np.random.random(len(labels))
    predictions[labels == 1] += 0.5  # Make anomalies have higher scores
    
    # Calculate metrics
    metrics = loader.calculate_detection_metrics(labels, predictions > 0.5)
    
    # Verify metrics structure
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'accuracy' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_data_validation_and_error_handling():
    """Test data validation and error handling."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    # Test with invalid config
    with pytest.raises((ValueError, KeyError)):
        config = {}  # Missing required fields
        loader = SWaTDataLoader(config)
        loader.load_data()  # Should fail
    
    # Test with non-existent file
    config = {
        'data_path': '/nonexistent/path/data.csv',
        'sequence_length': 10
    }
    loader = SWaTDataLoader(config)
    
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_memory_efficiency():
    """Test memory efficiency for large datasets."""
    from iot_edge_anomaly.data.swat_loader import SWaTDataLoader
    
    # Create larger mock dataset
    large_data = pd.DataFrame(np.random.randn(10000, 52))  # 10k samples, 51 features + label
    large_data.columns = [f"sensor_{i:02d}" for i in range(51)] + ["label"]
    
    config = {
        'sequence_length': 20,
        'batch_size': 64,
        'chunk_size': 1000  # Process in chunks for memory efficiency
    }
    
    loader = SWaTDataLoader(config)
    
    # Should be able to create sequences without memory issues
    sequences, labels = loader.create_sequences(large_data)
    assert sequences.shape[0] > 0
    assert len(labels) == len(sequences)