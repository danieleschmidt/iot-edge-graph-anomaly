"""Performance tests for IoT Edge Graph Anomaly Detection."""

import pytest
import numpy as np
import torch
import time
from unittest.mock import patch, Mock

from src.iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
from src.iot_edge_anomaly.monitoring.metrics_exporter import MetricsExporter


class TestModelPerformance:
    """Test model performance characteristics."""

    @pytest.mark.unit
    def test_lstm_inference_time(self, mock_model_config, performance_benchmark):
        """Test LSTM autoencoder inference time meets edge requirements."""
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        # Generate test data
        batch_size = 1  # Edge device single inference
        sequence_data = torch.randn(batch_size, mock_model_config['sequence_length'], 
                                   mock_model_config['input_size'])
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(sequence_data)
        
        # Benchmark inference time
        performance_benchmark.start()
        with torch.no_grad():
            for _ in range(100):
                output = model(sequence_data)
        performance_benchmark.stop()
        
        avg_inference_time = performance_benchmark.elapsed_time / 100
        
        # Edge requirement: <10ms inference time
        assert avg_inference_time < 0.01, f"Inference time {avg_inference_time:.4f}s exceeds 10ms limit"
        assert output.shape == sequence_data.shape

    @pytest.mark.unit
    def test_memory_usage_within_limits(self, mock_model_config):
        """Test model memory usage stays within edge device limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model and test data
        model = LSTMAutoencoder(**mock_model_config)
        test_data = torch.randn(32, mock_model_config['sequence_length'], 
                               mock_model_config['input_size'])
        
        # Forward pass
        with torch.no_grad():
            output = model(test_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Edge requirement: <50MB additional memory for model
        assert memory_used < 50, f"Model uses {memory_used:.2f}MB, exceeds 50MB limit"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_processing_performance(self, mock_model_config, sample_sensor_data):
        """Test batch processing performance for multiple sensor readings."""
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        # Create sequences from sensor data
        sequences = []
        seq_len = mock_model_config['sequence_length']
        for i in range(len(sample_sensor_data) - seq_len + 1):
            sequences.append(sample_sensor_data[i:i + seq_len])
        
        batch_data = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(batch_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(batch_data) / processing_time
        
        # Edge requirement: >100 sequences/second throughput
        assert throughput > 100, f"Throughput {throughput:.2f} seq/s below 100 seq/s requirement"

    @pytest.mark.unit
    def test_model_size_constraints(self, mock_model_config):
        """Test model size meets edge deployment constraints."""
        model = LSTMAutoencoder(**mock_model_config)
        
        # Calculate model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Edge requirement: <20MB model size
        assert model_size_mb < 20, f"Model size {model_size_mb:.2f}MB exceeds 20MB limit"


class TestMonitoringPerformance:
    """Test monitoring system performance."""

    @pytest.mark.unit
    @patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter')
    def test_metrics_export_performance(self, mock_exporter, mock_otlp_endpoint):
        """Test metrics export doesn't impact inference performance."""
        metrics_exporter = MetricsExporter(mock_otlp_endpoint)
        
        # Measure metrics collection time
        start_time = time.time()
        for _ in range(100):
            metrics_exporter.record_anomaly_count(1)
            metrics_exporter.record_inference_time(0.005)
            metrics_exporter.record_reconstruction_error(0.1)
        end_time = time.time()
        
        avg_metrics_time = (end_time - start_time) / 100
        
        # Requirement: <1ms for metrics collection
        assert avg_metrics_time < 0.001, f"Metrics collection {avg_metrics_time:.4f}s exceeds 1ms"

    @pytest.mark.integration
    @patch('opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter')
    def test_concurrent_monitoring_performance(self, mock_exporter, mock_otlp_endpoint):
        """Test monitoring performance under concurrent load."""
        import threading
        import queue
        
        metrics_exporter = MetricsExporter(mock_otlp_endpoint)
        results = queue.Queue()
        
        def monitor_worker():
            start = time.time()
            for _ in range(50):
                metrics_exporter.record_anomaly_count(1)
                time.sleep(0.001)  # Simulate processing
            end = time.time()
            results.put(end - start)
        
        # Start multiple monitoring threads
        threads = []
        for _ in range(4):
            thread = threading.Thread(target=monitor_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check all threads completed within reasonable time
        max_time = 0
        while not results.empty():
            thread_time = results.get()
            max_time = max(max_time, thread_time)
        
        # Should complete within 2x sequential time
        assert max_time < 0.2, f"Concurrent monitoring took {max_time:.3f}s, too slow"


class TestResourceConstraints:
    """Test resource constraint compliance for edge deployment."""

    @pytest.mark.integration
    def test_cpu_usage_constraint(self, mock_model_config, sample_sensor_data):
        """Test CPU usage stays within edge device limits."""
        import psutil
        
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        # Monitor CPU usage during inference
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Perform inference workload
        sequences = []
        seq_len = mock_model_config['sequence_length']
        for i in range(0, len(sample_sensor_data) - seq_len + 1, 10):
            sequences.append(sample_sensor_data[i:i + seq_len])
        
        batch_data = torch.tensor(np.array(sequences), dtype=torch.float32)
        
        start_time = time.time()
        with torch.no_grad():
            for batch in torch.split(batch_data, 8):  # Process in small batches
                _ = model(batch)
        
        cpu_percent_after = psutil.cpu_percent(interval=1)
        processing_time = time.time() - start_time
        
        # Edge requirement: <25% CPU usage on average
        avg_cpu_usage = (cpu_percent_after + cpu_percent_before) / 2
        assert avg_cpu_usage < 25, f"CPU usage {avg_cpu_usage:.1f}% exceeds 25% limit"

    @pytest.mark.unit
    def test_memory_leak_detection(self, mock_model_config):
        """Test for memory leaks during continuous operation."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate continuous operation
        for i in range(100):
            test_data = torch.randn(1, mock_model_config['sequence_length'], 
                                   mock_model_config['input_size'])
            with torch.no_grad():
                output = model(test_data)
                reconstruction_error = torch.nn.functional.mse_loss(output, test_data)
            
            # Force garbage collection periodically
            if i % 20 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (<5MB)
        assert memory_growth < 5, f"Memory growth {memory_growth:.2f}MB indicates potential leak"

    @pytest.mark.slow
    def test_long_running_stability(self, mock_model_config):
        """Test model stability during extended operation."""
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        reconstruction_errors = []
        inference_times = []
        
        # Simulate 1000 inferences
        for _ in range(1000):
            test_data = torch.randn(1, mock_model_config['sequence_length'], 
                                   mock_model_config['input_size'])
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_data)
                error = torch.nn.functional.mse_loss(output, test_data).item()
            inference_time = time.time() - start_time
            
            reconstruction_errors.append(error)
            inference_times.append(inference_time)
        
        # Check stability - variance should be low
        error_variance = np.var(reconstruction_errors)
        time_variance = np.var(inference_times)
        
        assert error_variance < 0.1, f"Reconstruction error variance {error_variance:.4f} too high"
        assert time_variance < 0.0001, f"Inference time variance {time_variance:.6f} too high"


@pytest.mark.e2e
class TestEndToEndPerformance:
    """End-to-end performance tests."""

    def test_complete_pipeline_performance(self, mock_model_config, sample_sensor_data, 
                                         mock_otlp_endpoint):
        """Test complete anomaly detection pipeline performance."""
        from src.iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder
        
        # Initialize components
        model = LSTMAutoencoder(**mock_model_config)
        model.eval()
        
        with patch('src.iot_edge_anomaly.monitoring.metrics_exporter.MetricsExporter'):
            # Simulate real-time processing
            total_start_time = time.time()
            
            for i in range(0, len(sample_sensor_data) - mock_model_config['sequence_length']):
                # Extract sequence
                sequence = sample_sensor_data[i:i + mock_model_config['sequence_length']]
                tensor_data = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                
                # Inference
                with torch.no_grad():
                    output = model(tensor_data)
                    reconstruction_error = torch.nn.functional.mse_loss(output, tensor_data)
                
                # Anomaly detection
                is_anomaly = reconstruction_error.item() > 0.5
                
                # Simulate monitoring (would be async in real system)
                time.sleep(0.001)  # Simulate monitoring overhead
            
            total_time = time.time() - total_start_time
            samples_processed = len(sample_sensor_data) - mock_model_config['sequence_length']
            throughput = samples_processed / total_time
            
            # End-to-end requirement: >50 samples/second
            assert throughput > 50, f"E2E throughput {throughput:.2f} samples/s below requirement"