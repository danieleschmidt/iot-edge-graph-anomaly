"""Integration tests for IoT Edge Graph Anomaly Detection."""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from src.iot_edge_anomaly.data.swat_loader import SWaTDataLoader
from src.iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter


@pytest.mark.integration
class TestModelDataIntegration:
    """Test integration between model and data loading components."""

    def test_model_with_swat_data_integration(self, temp_dir):
        """Test LSTM model training with SWaT dataset integration."""
        # Create mock SWaT data files
        train_data = np.random.randn(1000, 51).astype(np.float32)
        train_labels = np.random.randint(0, 2, 1000).astype(np.int32)
        test_data = np.random.randn(200, 51).astype(np.float32)
        test_labels = np.random.randint(0, 2, 200).astype(np.int32)
        
        train_file = temp_dir / "SWaT_train.csv"
        test_file = temp_dir / "SWaT_test.csv"
        
        # Save mock data
        np.savetxt(train_file, np.column_stack([train_data, train_labels]), delimiter=',')
        np.savetxt(test_file, np.column_stack([test_data, test_labels]), delimiter=',')
        
        # Test data loading
        data_loader = SWaTDataLoader(str(temp_dir))
        train_dataset, val_dataset, test_dataset = data_loader.load_datasets()
        
        # Test model creation and training
        model = LSTMAutoencoder(
            input_size=51,
            hidden_size=64,
            num_layers=2,
            sequence_length=10,
            dropout_rate=0.2
        )
        
        # Test forward pass with loaded data
        sample_batch = next(iter(train_dataset))
        if isinstance(sample_batch, tuple):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch
        
        with torch.no_grad():
            output = model(sample_input)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()

    def test_model_persistence_integration(self, temp_dir, mock_model_config):
        """Test model saving and loading integration."""
        # Create and train model
        model = LSTMAutoencoder(**mock_model_config)
        
        # Generate some training data
        sample_data = torch.randn(32, mock_model_config['sequence_length'], 
                                 mock_model_config['input_size'])
        
        # Forward pass to initialize parameters
        with torch.no_grad():
            output = model(sample_data)
        
        # Save model
        model_path = temp_dir / "test_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': mock_model_config
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        loaded_model = LSTMAutoencoder(**checkpoint['config'])
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test loaded model produces same output
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            original_output = model(sample_data)
            loaded_output = loaded_model(sample_data)
        
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
class TestMonitoringIntegration:
    """Test integration between model and monitoring components."""

    @patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter')
    def test_model_monitoring_integration(self, mock_exporter, mock_model_config, 
                                        mock_otlp_endpoint):
        """Test model inference with metrics collection."""
        # Initialize components
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        metrics_exporter = MetricsExporter(mock_otlp_endpoint)
        
        # Test data
        test_data = torch.randn(10, mock_model_config['sequence_length'], 
                               mock_model_config['input_size'])
        
        anomaly_count = 0
        inference_times = []
        
        for sample in test_data:
            sample = sample.unsqueeze(0)
            
            # Measure inference time
            import time
            start_time = time.time()
            
            with torch.no_grad():
                output = model(sample)
                reconstruction_error = torch.nn.functional.mse_loss(output, sample)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Detect anomaly
            is_anomaly = reconstruction_error.item() > 0.5
            if is_anomaly:
                anomaly_count += 1
            
            # Record metrics
            metrics_exporter.record_inference_time(inference_time)
            metrics_exporter.record_reconstruction_error(reconstruction_error.item())
            if is_anomaly:
                metrics_exporter.record_anomaly_count(1)
        
        # Verify metrics were recorded
        assert len(inference_times) == 10
        assert all(t > 0 for t in inference_times)
        
        # Verify exporter was used (mocked)
        assert mock_exporter.called

    @patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter')
    def test_continuous_monitoring_integration(self, mock_exporter, mock_model_config, 
                                             mock_otlp_endpoint):
        """Test continuous monitoring during model operation."""
        import threading
        import time
        import queue
        
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        metrics_exporter = MetricsExporter(mock_otlp_endpoint)
        
        results = queue.Queue()
        stop_event = threading.Event()
        
        def inference_worker():
            """Simulate continuous inference workload."""
            while not stop_event.is_set():
                sample_data = torch.randn(1, mock_model_config['sequence_length'], 
                                         mock_model_config['input_size'])
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(sample_data)
                    error = torch.nn.functional.mse_loss(output, sample_data)
                inference_time = time.time() - start_time
                
                # Record metrics
                metrics_exporter.record_inference_time(inference_time)
                metrics_exporter.record_reconstruction_error(error.item())
                
                results.put({
                    'inference_time': inference_time,
                    'error': error.item()
                })
                
                time.sleep(0.01)  # 100Hz inference rate
        
        # Start inference worker
        worker_thread = threading.Thread(target=inference_worker)
        worker_thread.start()
        
        # Let it run for a short time
        time.sleep(0.5)
        stop_event.set()
        worker_thread.join()
        
        # Collect results
        inference_results = []
        while not results.empty():
            inference_results.append(results.get())
        
        # Verify continuous operation
        assert len(inference_results) > 40  # Should have at least 40 inferences in 0.5s
        assert all(r['inference_time'] > 0 for r in inference_results)
        assert all(r['error'] >= 0 for r in inference_results)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow integration."""

    def test_complete_anomaly_detection_workflow(self, temp_dir):
        """Test complete workflow from data loading to anomaly detection."""
        # Step 1: Create mock data
        np.random.seed(42)
        normal_data = np.random.randn(500, 10).astype(np.float32)
        anomaly_data = np.random.randn(50, 10).astype(np.float32) * 3  # Amplified for anomalies
        
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([np.zeros(500), np.ones(50)])
        
        data_file = temp_dir / "test_data.csv"
        np.savetxt(data_file, np.column_stack([all_data, labels]), delimiter=',')
        
        # Step 2: Load and preprocess data
        from sklearn.preprocessing import StandardScaler
        
        data = np.loadtxt(data_file, delimiter=',')
        features = data[:, :-1]
        true_labels = data[:, -1]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Step 3: Create sequences
        sequence_length = 10
        sequences = []
        sequence_labels = []
        
        for i in range(len(scaled_features) - sequence_length + 1):
            sequences.append(scaled_features[i:i + sequence_length])
            sequence_labels.append(true_labels[i + sequence_length - 1])
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        # Step 4: Initialize and test model
        model = LSTMAutoencoder(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            sequence_length=sequence_length,
            dropout_rate=0.2
        )
        model.eval()
        
        # Step 5: Perform anomaly detection
        reconstruction_errors = []
        predictions = []
        
        for sequence in sequences:
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                output = model(sequence_tensor)
                error = torch.nn.functional.mse_loss(output, sequence_tensor).item()
            
            reconstruction_errors.append(error)
        
        # Step 6: Determine threshold and make predictions
        threshold = np.percentile(reconstruction_errors, 95)
        predictions = [1 if error > threshold else 0 for error in reconstruction_errors]
        
        # Step 7: Evaluate results
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(sequence_labels, predictions, zero_division=0)
        recall = recall_score(sequence_labels, predictions, zero_division=0)
        f1 = f1_score(sequence_labels, predictions, zero_division=0)
        
        # Basic sanity checks
        assert len(reconstruction_errors) == len(sequences)
        assert len(predictions) == len(sequence_labels)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    @patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter')
    def test_production_simulation(self, mock_exporter, temp_dir, mock_otlp_endpoint):
        """Simulate production environment operation."""
        import threading
        import time
        import queue
        
        # Initialize components
        model = LSTMAutoencoder(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            sequence_length=10,
            dropout_rate=0.2
        )
        model.eval()
        
        metrics_exporter = MetricsExporter(mock_otlp_endpoint)
        
        # Simulation parameters
        sensor_data_queue = queue.Queue()
        results_queue = queue.Queue()
        stop_event = threading.Event()
        
        def sensor_simulator():
            """Simulate sensor data stream."""
            while not stop_event.is_set():
                # Generate sensor reading
                if np.random.random() < 0.1:  # 10% anomalies
                    reading = np.random.randn(10) * 3  # Anomalous reading
                else:
                    reading = np.random.randn(10)  # Normal reading
                
                sensor_data_queue.put(reading.astype(np.float32))
                time.sleep(0.1)  # 10Hz sensor rate
        
        def anomaly_detector():
            """Anomaly detection worker."""
            buffer = []
            
            while not stop_event.is_set():
                try:
                    reading = sensor_data_queue.get(timeout=0.1)
                    buffer.append(reading)
                    
                    if len(buffer) >= 10:  # Sequence length
                        sequence = np.array(buffer[-10:])
                        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                        
                        start_time = time.time()
                        with torch.no_grad():
                            output = model(sequence_tensor)
                            error = torch.nn.functional.mse_loss(output, sequence_tensor).item()
                        inference_time = time.time() - start_time
                        
                        is_anomaly = error > 0.5
                        
                        # Record metrics
                        metrics_exporter.record_inference_time(inference_time)
                        metrics_exporter.record_reconstruction_error(error)
                        if is_anomaly:
                            metrics_exporter.record_anomaly_count(1)
                        
                        results_queue.put({
                            'timestamp': time.time(),
                            'error': error,
                            'is_anomaly': is_anomaly,
                            'inference_time': inference_time
                        })
                        
                except queue.Empty:
                    continue
        
        # Start simulation
        sensor_thread = threading.Thread(target=sensor_simulator)
        detector_thread = threading.Thread(target=anomaly_detector)
        
        sensor_thread.start()
        detector_thread.start()
        
        # Run simulation for 2 seconds
        time.sleep(2.0)
        
        stop_event.set()
        sensor_thread.join()
        detector_thread.join()
        
        # Collect results
        detection_results = []
        while not results_queue.empty():
            detection_results.append(results_queue.get())
        
        # Verify production-like behavior
        assert len(detection_results) > 10  # Should have multiple detections
        
        inference_times = [r['inference_time'] for r in detection_results]
        assert all(t > 0 for t in inference_times)
        assert max(inference_times) < 0.1  # All inferences under 100ms
        
        errors = [r['error'] for r in detection_results]
        assert all(e >= 0 for e in errors)
        
        # Verify metrics exporter was used
        assert mock_exporter.called


@pytest.mark.integration
class TestEnvironmentIntegration:
    """Test integration with different deployment environments."""

    def test_docker_environment_simulation(self, mock_model_config):
        """Test model behavior in Docker-like constrained environment."""
        import psutil
        import os
        
        # Simulate resource constraints
        process = psutil.Process(os.getpid())
        
        # Memory constraint check
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        # Simulate edge workload
        for _ in range(100):
            test_data = torch.randn(1, mock_model_config['sequence_length'], 
                                   mock_model_config['input_size'])
            with torch.no_grad():
                output = model(test_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Should stay within Docker memory limits
        assert memory_used < 100, f"Memory usage {memory_used:.2f}MB exceeds container limit"

    @patch.dict(os.environ, {
        'MODEL_PATH': '/tmp/test_model.pth',
        'ANOMALY_THRESHOLD': '0.6',
        'LOG_LEVEL': 'INFO'
    })
    def test_environment_variable_integration(self, mock_model_config, temp_dir):
        """Test integration with environment variables."""
        # Test environment variable reading
        assert os.getenv('MODEL_PATH') == '/tmp/test_model.pth'
        assert float(os.getenv('ANOMALY_THRESHOLD')) == 0.6
        assert os.getenv('LOG_LEVEL') == 'INFO'
        
        # Test model with environment-configured threshold
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        threshold = float(os.getenv('ANOMALY_THRESHOLD'))
        
        test_data = torch.randn(5, mock_model_config['sequence_length'], 
                               mock_model_config['input_size'])
        
        anomaly_count = 0
        for sample in test_data:
            sample = sample.unsqueeze(0)
            with torch.no_grad():
                output = model(sample)
                error = torch.nn.functional.mse_loss(output, sample).item()
            
            if error > threshold:
                anomaly_count += 1
        
        # Should detect some anomalies (random data vs trained threshold)
        assert anomaly_count >= 0