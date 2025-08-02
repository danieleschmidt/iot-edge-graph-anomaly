"""Example integration tests demonstrating component interaction testing."""

import pytest
import torch
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from tests.fixtures.test_data import (
    MockLSTMAutoencoder,
    MockDataLoader,
    create_sample_swat_data,
    create_mock_model_config
)


class TestModelTrainingPipeline:
    """Integration tests for model training pipeline."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_training_data_flow(self, temp_model_dir):
        """Test complete training data flow from loader to model."""
        # Create sample training data
        train_data = create_sample_swat_data()
        data_loader = MockDataLoader(train_data)
        
        # Initialize model and training components
        model = MockLSTMAutoencoder(input_size=51)  # SWaT has 51 sensors
        optimizer = MagicMock()  # Mock optimizer
        
        # Simulate training loop
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in data_loader:
            if batch_count >= 3:  # Limit test to 3 batches
                break
                
            # Mock forward pass
            output = model(batch)
            
            # Mock loss calculation
            loss = torch.mean((batch - output) ** 2)
            total_loss += loss.item()
            batch_count += 1
            
            # Verify batch processing
            assert output.shape == batch.shape
            assert loss.item() >= 0
        
        # Verify training completed
        assert batch_count == 3
        assert total_loss > 0
        
        # Test model saving
        model_path = temp_model_dir / "test_model.pth"
        torch.save(model, model_path)
        assert model_path.exists()
    
    def test_model_persistence(self, temp_model_dir):
        """Test model saving and loading integration."""
        # Create and train model
        original_model = MockLSTMAutoencoder(input_size=10, hidden_size=32)
        test_input = torch.randn(1, 10, 10)
        original_output = original_model(test_input)
        
        # Save model
        model_path = temp_model_dir / "persistence_test.pth"
        torch.save(original_model, model_path)
        
        # Load model
        loaded_model = torch.load(model_path)
        loaded_output = loaded_model(test_input)
        
        # Verify loaded model produces same output
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        assert loaded_model.input_size == original_model.input_size
        assert loaded_model.hidden_size == original_model.hidden_size


class TestAnomalyDetectionWorkflow:
    """Integration tests for anomaly detection workflow."""
    
    def test_end_to_end_anomaly_detection(self):
        """Test complete anomaly detection from data to alerts."""
        # Setup components
        model = MockLSTMAutoencoder(input_size=10)
        test_data = torch.randn(5, 10, 10)  # 5 samples
        threshold = 0.5
        
        # Mock anomaly detection pipeline
        model.eval()
        detected_anomalies = []
        
        for sample in test_data:
            # Forward pass
            reconstruction = model(sample.unsqueeze(0))
            
            # Calculate reconstruction error
            error = torch.mean((sample.unsqueeze(0) - reconstruction) ** 2).item()
            
            # Anomaly detection
            is_anomaly = error > threshold
            detected_anomalies.append({
                'timestamp': pd.Timestamp.now(),
                'error': error,
                'is_anomaly': is_anomaly,
                'confidence': min(error / threshold, 2.0) if is_anomaly else 0.0
            })
        
        # Verify pipeline results
        assert len(detected_anomalies) == 5
        for result in detected_anomalies:
            assert 'timestamp' in result
            assert 'error' in result
            assert 'is_anomaly' in result
            assert result['error'] >= 0
            assert isinstance(result['is_anomaly'], bool)
    
    def test_batch_anomaly_processing(self):
        """Test batch processing of multiple sensor readings."""
        model = MockLSTMAutoencoder(input_size=5)
        batch_size = 10
        sequence_length = 8
        num_features = 5
        
        # Create test batch
        batch_data = torch.randn(batch_size, sequence_length, num_features)
        
        # Process batch
        model.eval()
        with torch.no_grad():
            reconstructions = model(batch_data)
        
        # Calculate batch errors
        batch_errors = torch.mean((batch_data - reconstructions) ** 2, dim=(1, 2))
        
        # Verify batch processing
        assert reconstructions.shape == batch_data.shape
        assert batch_errors.shape == (batch_size,)
        assert torch.all(batch_errors >= 0)


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_data_loader_preprocessing_integration(self):
        """Test integration between data loading and preprocessing."""
        # Create sample data with known properties
        sample_data = create_sample_swat_data()
        data_loader = MockDataLoader(sample_data)
        
        # Test preprocessing pipeline
        processed_batches = []
        for batch in data_loader:
            # Mock preprocessing steps
            # 1. Normalization
            batch_mean = torch.mean(batch, dim=0, keepdim=True)
            batch_std = torch.std(batch, dim=0, keepdim=True) + 1e-8
            normalized_batch = (batch - batch_mean) / batch_std
            
            # 2. Sequence creation (mock)
            if normalized_batch.shape[0] >= 10:  # Need at least 10 samples
                sequences = normalized_batch[:10].unsqueeze(0)  # Simplified
                processed_batches.append(sequences)
            
            if len(processed_batches) >= 2:  # Limit test
                break
        
        # Verify preprocessing integration
        assert len(processed_batches) >= 1
        for sequences in processed_batches:
            assert sequences.dim() == 3  # batch, seq, features
            assert not torch.isnan(sequences).any()
    
    @pytest.fixture
    def temp_data_file(self):
        """Create temporary data file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write sample CSV data
            f.write("timestamp,sensor_1,sensor_2,sensor_3,label\n")
            f.write("2025-01-01 00:00:00,1.0,2.0,3.0,Normal\n")
            f.write("2025-01-01 00:01:00,1.1,2.1,3.1,Normal\n")
            f.write("2025-01-01 00:02:00,5.0,8.0,9.0,Attack\n")
            f.write("2025-01-01 00:03:00,1.0,2.0,3.0,Normal\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_file_loading_integration(self, temp_data_file):
        """Test integration of file loading with data processing."""
        # Load data from file
        data = pd.read_csv(temp_data_file, parse_dates=['timestamp'])
        
        # Verify data loading
        assert len(data) == 4
        assert 'timestamp' in data.columns
        assert 'label' in data.columns
        
        # Test data processing integration
        numeric_columns = ['sensor_1', 'sensor_2', 'sensor_3']
        sensor_data = data[numeric_columns].values
        
        # Mock normalization
        mean = sensor_data.mean(axis=0)
        std = sensor_data.std(axis=0)
        normalized_data = (sensor_data - mean) / std
        
        # Verify processing results
        assert normalized_data.shape == (4, 3)
        assert not np.isnan(normalized_data).any()


class TestMonitoringIntegration:
    """Integration tests for monitoring and metrics collection."""
    
    @patch('time.time')
    def test_metrics_collection_integration(self, mock_time):
        """Test metrics collection during model inference."""
        mock_time.return_value = 1640995200  # Fixed timestamp
        
        # Setup monitoring components
        metrics = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'anomalies_detected': 0,
            'avg_reconstruction_error': 0.0
        }
        
        model = MockLSTMAutoencoder(input_size=5)
        test_samples = torch.randn(3, 1, 5, 5)  # 3 test samples
        
        # Process samples with metrics collection
        reconstruction_errors = []
        
        for i, sample in enumerate(test_samples):
            start_time = mock_time.return_value + i * 0.01  # Mock timing
            
            # Model inference
            model.eval()
            with torch.no_grad():
                reconstruction = model(sample)
            
            end_time = start_time + 0.005  # Mock 5ms inference time
            
            # Calculate metrics
            error = torch.mean((sample - reconstruction) ** 2).item()
            is_anomaly = error > 0.5
            
            # Update metrics
            metrics['inference_count'] += 1
            metrics['total_inference_time'] += (end_time - start_time)
            if is_anomaly:
                metrics['anomalies_detected'] += 1
            reconstruction_errors.append(error)
        
        # Calculate aggregate metrics
        metrics['avg_reconstruction_error'] = sum(reconstruction_errors) / len(reconstruction_errors)
        metrics['avg_inference_time'] = metrics['total_inference_time'] / metrics['inference_count']
        
        # Verify metrics collection
        assert metrics['inference_count'] == 3
        assert metrics['total_inference_time'] > 0
        assert metrics['avg_reconstruction_error'] >= 0
        assert metrics['avg_inference_time'] > 0
        assert 0 <= metrics['anomalies_detected'] <= 3
    
    def test_health_check_integration(self):
        """Test health check integration with model status."""
        model = MockLSTMAutoencoder(input_size=10)
        
        # Mock health check function
        def perform_health_check():
            health_status = {
                'status': 'healthy',
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_loaded': model is not None,
                'memory_usage': 45.2,  # Mock memory usage in MB
                'last_inference_time': 0.008,  # Mock inference time in seconds
                'error_count': 0
            }
            
            # Perform basic model test
            try:
                test_input = torch.randn(1, 10, 10)
                output = model(test_input)
                health_status['model_responsive'] = True
                health_status['last_inference_shape'] = list(output.shape)
            except Exception as e:
                health_status['status'] = 'unhealthy'
                health_status['error'] = str(e)
                health_status['model_responsive'] = False
            
            return health_status
        
        # Execute health check
        health_result = perform_health_check()
        
        # Verify health check results
        assert health_result['status'] == 'healthy'
        assert health_result['model_loaded'] is True
        assert health_result['model_responsive'] is True
        assert 'timestamp' in health_result
        assert health_result['memory_usage'] > 0
        assert health_result['last_inference_time'] > 0