"""Performance benchmark tests for edge deployment constraints."""

import time
import pytest
import torch
import psutil
import os
from typing import List
from unittest.mock import patch, MagicMock

from src.iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder


class PerformanceBenchmark:
    """Performance benchmarking utilities for edge device constraints."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
        self.start_time = None
    
    def start_timer(self):
        """Start timing measurement."""
        self.start_time = time.perf_counter()
    
    def stop_timer(self) -> float:
        """Stop timing and return elapsed seconds."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return elapsed
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=1.0)
    
    def memory_delta(self) -> float:
        """Get memory increase since baseline."""
        return self.get_memory_usage() - self.baseline_memory


@pytest.fixture
def benchmark():
    """Provide performance benchmark utilities."""
    return PerformanceBenchmark()


@pytest.fixture
def test_model():
    """Provide test LSTM autoencoder model."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64, num_layers=2, dropout=0.2)
    model.eval()
    return model


@pytest.fixture
def test_data():
    """Provide test data for benchmarking."""
    return torch.randn(1000, 10, 51)


@pytest.mark.performance
class TestMemoryConstraints:
    """Test memory usage constraints for edge deployment."""
    
    def test_model_memory_footprint(self, benchmark, test_model):
        """Test model memory footprint stays within 100MB limit."""
        memory_before = benchmark.get_memory_usage()
        
        # Load model and run inference
        test_input = torch.randn(1, 10, 51)
        with torch.no_grad():
            _ = test_model(test_input)
        
        memory_after = benchmark.get_memory_usage()
        memory_used = memory_after - memory_before
        
        assert memory_after < 100, f"Total memory usage {memory_after:.1f}MB exceeds 100MB limit"
        assert memory_used < 50, f"Model memory footprint {memory_used:.1f}MB too high"
    
    def test_batch_processing_memory(self, benchmark, test_model, test_data):
        """Test memory usage during batch processing."""
        memory_before = benchmark.get_memory_usage()
        
        batch_size = 32
        with torch.no_grad():
            for batch in test_data.split(batch_size):
                output = test_model(batch)
                # Simulate processing
                del output
        
        memory_after = benchmark.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        assert memory_increase < 20, f"Batch processing increased memory by {memory_increase:.1f}MB"
    
    @pytest.mark.slow
    def test_memory_leak_detection(self, benchmark, test_model):
        """Test for memory leaks during extended operation."""
        memory_readings = []
        test_input = torch.randn(1, 10, 51)
        
        # Run inference 1000 times
        for i in range(1000):
            with torch.no_grad():
                output = test_model(test_input)
                del output
            
            if i % 100 == 0:
                memory_readings.append(benchmark.get_memory_usage())
        
        # Check for memory growth trend
        memory_growth = memory_readings[-1] - memory_readings[0]
        assert memory_growth < 5, f"Memory leak detected: {memory_growth:.1f}MB growth"


@pytest.mark.performance
class TestLatencyConstraints:
    """Test inference latency constraints for real-time processing."""
    
    def test_single_sample_latency(self, benchmark, test_model):
        """Test single sample inference latency under 10ms."""
        test_input = torch.randn(1, 10, 51)
        
        # Warm up the model
        for _ in range(10):
            with torch.no_grad():
                _ = test_model(test_input)
        
        # Measure latency over multiple runs
        latencies = []
        for _ in range(100):
            benchmark.start_timer()
            with torch.no_grad():
                _ = test_model(test_input)
            latency = benchmark.stop_timer()
            latencies.append(latency * 1000)  # Convert to ms
        
        # Statistical analysis
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        p99_latency = sorted(latencies)[99]
        
        assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms exceeds 10ms target"
        assert p95_latency < 20, f"P95 latency {p95_latency:.2f}ms exceeds 20ms threshold"
        assert p99_latency < 50, f"P99 latency {p99_latency:.2f}ms exceeds 50ms threshold"
    
    def test_batch_inference_latency(self, benchmark, test_model):
        """Test batch inference latency scales appropriately."""
        batch_sizes = [1, 8, 16, 32]
        latencies_per_sample = []
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 10, 51)
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = test_model(test_input)
            
            # Measure
            benchmark.start_timer()
            with torch.no_grad():
                _ = test_model(test_input)
            total_latency = benchmark.stop_timer()
            
            latency_per_sample = (total_latency * 1000) / batch_size
            latencies_per_sample.append(latency_per_sample)
        
        # Per-sample latency should improve with batching
        assert latencies_per_sample[-1] < latencies_per_sample[0], \
            "Batching should improve per-sample latency"


@pytest.mark.performance
class TestThroughputConstraints:
    """Test throughput constraints for sustained processing."""
    
    def test_sustained_throughput(self, benchmark, test_model, test_data):
        """Test sustained throughput exceeds 100 samples/second."""
        batch_size = 32
        
        benchmark.start_timer()
        samples_processed = 0
        
        with torch.no_grad():
            for batch in test_data.split(batch_size):
                _ = test_model(batch)
                samples_processed += len(batch)
        
        elapsed_time = benchmark.stop_timer()
        throughput = samples_processed / elapsed_time
        
        assert throughput > 100, f"Throughput {throughput:.1f} samples/sec below 100 target"
    
    @pytest.mark.slow
    def test_sustained_processing_stability(self, benchmark, test_model):
        """Test processing stability over extended periods."""
        test_input = torch.randn(32, 10, 51)
        throughput_readings = []
        
        # Run for 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            period_start = time.time()
            samples_in_period = 0
            
            # Process for 5 second intervals
            while time.time() - period_start < 5:
                with torch.no_grad():
                    _ = test_model(test_input)
                samples_in_period += len(test_input)
            
            period_throughput = samples_in_period / 5
            throughput_readings.append(period_throughput)
        
        # Check throughput stability
        avg_throughput = sum(throughput_readings) / len(throughput_readings)
        min_throughput = min(throughput_readings)
        
        assert avg_throughput > 100, f"Average throughput {avg_throughput:.1f} below target"
        assert min_throughput > 80, f"Minimum throughput {min_throughput:.1f} too low"


@pytest.mark.performance
class TestCPUConstraints:
    """Test CPU usage constraints for edge devices."""
    
    @pytest.mark.slow
    def test_cpu_usage_limits(self, benchmark, test_model):
        """Test CPU usage stays under 25% during sustained load."""
        test_input = torch.randn(16, 10, 51)
        cpu_readings = []
        
        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            with torch.no_grad():
                _ = test_model(test_input)
            
            cpu_usage = benchmark.get_cpu_usage()
            cpu_readings.append(cpu_usage)
            time.sleep(0.1)
        
        avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
        max_cpu = max(cpu_readings) if cpu_readings else 0
        
        assert avg_cpu < 25, f"Average CPU usage {avg_cpu:.1f}% exceeds 25% limit"
        assert max_cpu < 40, f"Peak CPU usage {max_cpu:.1f}% exceeds 40% threshold"
    
    def test_model_loading_cpu_impact(self, benchmark):
        """Test CPU impact during model loading."""
        cpu_before = benchmark.get_cpu_usage()
        
        # Load model
        model = LSTMAutoencoder(input_size=51, hidden_size=64, num_layers=2)
        
        cpu_after = benchmark.get_cpu_usage()
        cpu_spike = cpu_after - cpu_before
        
        # CPU spike during loading should be reasonable
        assert cpu_spike < 50, f"Model loading caused {cpu_spike:.1f}% CPU spike"


@pytest.mark.performance
class TestResourceConstraints:
    """Test overall resource constraints for edge deployment."""
    
    def test_model_file_size(self, test_model, tmp_path):
        """Test model file size is reasonable for edge storage."""
        model_path = tmp_path / "test_model.pth"
        torch.save(test_model.state_dict(), model_path)
        
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        assert model_size_mb < 50, f"Model file size {model_size_mb:.1f}MB too large"
    
    def test_startup_time(self, benchmark):
        """Test application startup time is acceptable."""
        benchmark.start_timer()
        
        # Simulate application startup
        model = LSTMAutoencoder(input_size=51, hidden_size=64, num_layers=2)
        model.eval()
        
        # Simulate initial inference
        test_input = torch.randn(1, 10, 51)
        with torch.no_grad():
            _ = model(test_input)
        
        startup_time = benchmark.stop_timer()
        
        assert startup_time < 30, f"Startup time {startup_time:.1f}s exceeds 30s limit"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, test_model):
        """Test GPU memory usage if CUDA is available."""
        device = torch.device("cuda")
        model = test_model.to(device)
        test_input = torch.randn(32, 10, 51).to(device)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            _ = model(test_input)
        
        memory_after = torch.cuda.memory_allocated()
        memory_used_mb = (memory_after - memory_before) / (1024 * 1024)
        
        assert memory_used_mb < 100, f"GPU memory usage {memory_used_mb:.1f}MB too high"


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_baseline_performance_metrics(self, benchmark, test_model):
        """Establish baseline performance metrics."""
        test_input = torch.randn(100, 10, 51)
        
        # Memory baseline
        memory_before = benchmark.get_memory_usage()
        
        # Latency baseline
        benchmark.start_timer()
        with torch.no_grad():
            for sample in test_input:
                _ = test_model(sample.unsqueeze(0))
        latency_total = benchmark.stop_timer()
        
        memory_after = benchmark.get_memory_usage()
        
        # Record baseline metrics
        baseline_metrics = {
            "memory_usage_mb": memory_after,
            "memory_increase_mb": memory_after - memory_before,
            "avg_latency_ms": (latency_total / len(test_input)) * 1000,
            "total_processing_time_s": latency_total
        }
        
        # These serve as regression test baselines
        assert baseline_metrics["memory_usage_mb"] < 100
        assert baseline_metrics["avg_latency_ms"] < 10
        
        # Could save these metrics for future regression testing
        print(f"Baseline metrics: {baseline_metrics}")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])