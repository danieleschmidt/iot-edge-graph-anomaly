"""Performance benchmarks for the anomaly detection system."""

import time
import psutil
import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path

from iot_edge_anomaly.models.lstm_gnn_hybrid import LSTMGNNHybrid


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for edge deployment."""

    def test_memory_usage_benchmark(
        self,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test memory usage stays within edge device limits."""
        # Record initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        # Prepare batch data
        batch_size = 32
        test_input = torch.randn(batch_size, 10, 50)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        model.eval()
        
        # Run multiple inferences to measure peak memory
        peak_memory = initial_memory
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_input, edge_index)
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
        
        memory_increase = peak_memory - initial_memory
        
        # Edge device memory constraint: <100MB total usage
        assert memory_increase < 50, f"Memory increase {memory_increase:.2f}MB exceeds 50MB limit"
        print(f"Peak memory usage: {peak_memory:.2f}MB (increase: {memory_increase:.2f}MB)")

    def test_inference_latency_benchmark(
        self,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test inference latency meets edge device requirements."""
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Prepare test data
        test_input = torch.randn(1, 10, 50)  # Single sample for latency test
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input, edge_index)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(1000):
                start_time = time.perf_counter()
                _ = model(test_input, edge_index)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Edge device latency requirements
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert p95_latency < 150, f"P95 latency {p95_latency:.2f}ms exceeds 150ms"
        assert p99_latency < 200, f"P99 latency {p99_latency:.2f}ms exceeds 200ms"
        
        print(f"Latency stats - Avg: {avg_latency:.2f}ms, P50: {p50_latency:.2f}ms, "
              f"P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")

    def test_throughput_benchmark(
        self,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test inference throughput for batch processing."""
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 8, 16, 32, 64]
        throughput_results = {}
        
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 10, 50)
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input, edge_index)
            
            # Measure throughput
            start_time = time.perf_counter()
            num_iterations = 100
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(test_input, edge_index)
            
            end_time = time.perf_counter()
            
            total_samples = batch_size * num_iterations
            total_time = end_time - start_time
            throughput = total_samples / total_time
            
            throughput_results[batch_size] = throughput
            
            print(f"Batch size {batch_size}: {throughput:.2f} samples/second")
        
        # Verify minimum throughput requirement
        max_throughput = max(throughput_results.values())
        assert max_throughput > 100, f"Max throughput {max_throughput:.2f} samples/s below 100 samples/s"

    def test_cpu_utilization_benchmark(
        self,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test CPU utilization stays within edge device limits."""
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Prepare test data
        test_input = torch.randn(32, 10, 50)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        # Monitor CPU usage during inference
        cpu_percentages = []
        
        def monitor_cpu():
            """Monitor CPU usage in a separate thread."""
            while not stop_monitoring:
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)
        
        import threading
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Run inference for sustained period
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_input, edge_index)
                    time.sleep(0.01)  # Small delay to simulate real usage
        finally:
            stop_monitoring = True
            monitor_thread.join()
        
        if cpu_percentages:
            avg_cpu = np.mean(cpu_percentages)
            max_cpu = np.max(cpu_percentages)
            
            # Edge device CPU constraint: <25% on Raspberry Pi 4 (4 cores)
            assert avg_cpu < 25, f"Average CPU usage {avg_cpu:.2f}% exceeds 25%"
            assert max_cpu < 50, f"Peak CPU usage {max_cpu:.2f}% exceeds 50%"
            
            print(f"CPU usage - Average: {avg_cpu:.2f}%, Peak: {max_cpu:.2f}%")

    def test_model_size_benchmark(
        self,
        sample_config: Dict[str, Any],
        temp_dir: Path
    ) -> None:
        """Test model size meets edge deployment constraints."""
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        # Save model and measure size
        model_path = temp_dir / "model_benchmark.pth"
        torch.save(model.state_dict(), model_path)
        
        model_size_mb = model_path.stat().st_size / 1024 / 1024
        
        # Edge device storage constraint: <50MB model size
        assert model_size_mb < 50, f"Model size {model_size_mb:.2f}MB exceeds 50MB limit"
        
        print(f"Model size: {model_size_mb:.2f}MB")

    def test_concurrent_inference_benchmark(
        self,
        sample_config: Dict[str, Any]
    ) -> None:
        """Test performance under concurrent inference requests."""
        import threading
        import queue
        
        # Initialize model
        model = LSTMGNNHybrid(
            input_size=10,
            lstm_hidden_size=sample_config["model"]["lstm_hidden_size"],
            lstm_num_layers=sample_config["model"]["lstm_num_layers"],
            gnn_hidden_size=sample_config["model"]["gnn_hidden_size"],
            dropout=sample_config["model"]["dropout"]
        )
        
        model.eval()
        
        # Prepare test data
        test_input = torch.randn(1, 10, 50)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        # Results queue
        results_queue = queue.Queue()
        
        def worker():
            """Worker thread for concurrent inference."""
            latencies = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.perf_counter()
                    _ = model(test_input, edge_index)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            
            results_queue.put(latencies)
        
        # Test with multiple concurrent threads
        num_threads = 4
        threads = []
        
        start_time = time.perf_counter()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        # Collect results
        all_latencies = []
        while not results_queue.empty():
            thread_latencies = results_queue.get()
            all_latencies.extend(thread_latencies)
        
        avg_latency = np.mean(all_latencies)
        total_time = end_time - start_time
        total_inferences = len(all_latencies)
        throughput = total_inferences / total_time
        
        # Verify concurrent performance
        assert avg_latency < 200, f"Concurrent avg latency {avg_latency:.2f}ms exceeds 200ms"
        assert throughput > 20, f"Concurrent throughput {throughput:.2f} inferences/s below 20/s"
        
        print(f"Concurrent performance - {num_threads} threads, "
              f"Avg latency: {avg_latency:.2f}ms, Throughput: {throughput:.2f} inferences/s")