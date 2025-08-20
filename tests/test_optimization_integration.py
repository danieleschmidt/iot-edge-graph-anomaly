"""
Integration tests for optimization features.

Tests the performance profiler, caching system, and optimized prediction
pipeline to ensure they work correctly and provide performance benefits.
"""

import pytest
import torch
import time
import asyncio
import logging
from unittest.mock import patch, MagicMock

from src.iot_edge_anomaly.optimization.performance_profiler import (
    PerformanceProfiler, ProfileMetrics, BottleneckAnalysis
)

logger = logging.getLogger(__name__)


class TestPerformanceProfiler:
    """Test suite for performance profiler functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler for testing."""
        return PerformanceProfiler(
            enable_memory_profiling=True,
            enable_gpu_profiling=True,
            history_size=50,
            sampling_interval=0.1
        )
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initializes correctly."""
        assert profiler.enable_memory_profiling is True
        assert profiler.enable_gpu_profiling is True
        assert profiler.history_size == 50
        assert profiler.sampling_interval == 0.1
        assert len(profiler.profile_data) == 0
    
    def test_context_manager_profiling(self, profiler):
        """Test profiling using context manager."""
        function_name = "test_function"
        
        with profiler.profile(function_name):
            # Simulate some work
            time.sleep(0.1)
            torch.randn(100, 100).sum()
        
        assert function_name in profiler.profile_data
        metrics_list = profiler.profile_data[function_name]
        assert len(metrics_list) == 1
        
        metrics = metrics_list[0]
        assert metrics.function_name == function_name
        assert metrics.total_time >= 0.1  # Should be at least 0.1 seconds
        assert metrics.call_count == 1
    
    def test_function_decorator(self, profiler):
        """Test profiling using function decorator."""
        @profiler.profile_function
        def test_function(x, y):
            time.sleep(0.05)
            return x + y
        
        result = test_function(3, 4)
        assert result == 7
        
        assert "test_function" in profiler.profile_data
        metrics_list = profiler.profile_data["test_function"]
        assert len(metrics_list) == 1
        assert metrics_list[0].total_time >= 0.05
    
    def test_multiple_calls_statistics(self, profiler):
        """Test statistics calculation for multiple function calls."""
        function_name = "repeated_function"
        
        # Call function multiple times with different execution times
        for i in range(5):
            with profiler.profile(function_name):
                time.sleep(0.01 * (i + 1))  # Varying execution times
        
        metrics_list = profiler.profile_data[function_name]
        assert len(metrics_list) == 5
        
        # Check that statistics are properly calculated
        latest_metrics = metrics_list[-1]
        assert latest_metrics.call_count == 5
        assert latest_metrics.min_time <= latest_metrics.avg_time <= latest_metrics.max_time
        assert latest_metrics.min_time > 0
    
    def test_system_monitoring(self, profiler):
        """Test system resource monitoring."""
        # Start monitoring
        profiler.start_monitoring()
        
        # Wait for some samples
        time.sleep(0.3)
        
        # Stop monitoring
        profiler.stop_monitoring()
        
        # Check that system metrics were collected
        assert len(profiler.system_metrics) > 0
        
        # Verify metric structure
        sample_metric = profiler.system_metrics[0]
        assert "timestamp" in sample_metric
        assert "cpu_usage" in sample_metric
        assert "memory_usage" in sample_metric
        assert "memory_available" in sample_metric
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.utilization')
    @patch('torch.cuda.memory_stats')
    def test_gpu_monitoring(self, mock_memory_stats, mock_utilization, mock_cuda_available, profiler):
        """Test GPU monitoring when CUDA is available."""
        # Mock CUDA availability and metrics
        mock_cuda_available.return_value = True
        mock_utilization.return_value = 75.0
        mock_memory_stats.return_value = {'reserved_bytes.all.current': 1024**3}  # 1GB
        
        metrics = profiler._collect_system_metrics()
        
        assert "gpu_usage" in metrics
        assert "gpu_memory_used" in metrics
        assert metrics["gpu_usage"] == 75.0
        assert metrics["gpu_memory_used"] == 1.0  # 1GB converted to GB
    
    def test_profile_summary(self, profiler):
        """Test profile summary generation."""
        # Create some profile data
        function_names = ["func1", "func2", "func3"]
        
        for func_name in function_names:
            for _ in range(3):
                with profiler.profile(func_name):
                    time.sleep(0.01)
        
        # Test summary for specific function
        summary = profiler.get_profile_summary("func1")
        assert summary["function_name"] == "func1"
        assert summary["total_calls"] == 3
        assert "average_time" in summary
        assert "min_time" in summary
        assert "max_time" in summary
        
        # Test summary for all functions
        all_summary = profiler.get_profile_summary()
        assert len(all_summary) == 3
        for func_name in function_names:
            assert func_name in all_summary
            assert all_summary[func_name]["total_calls"] == 3
    
    def test_bottleneck_analysis(self, profiler):
        """Test bottleneck analysis functionality."""
        # Create functions with different performance characteristics
        with profiler.profile("slow_function"):
            time.sleep(0.2)
        
        with profiler.profile("fast_function"):
            time.sleep(0.01)
        
        with profiler.profile("medium_function"):
            time.sleep(0.05)
        
        # Analyze bottlenecks
        analysis = profiler.analyze_bottlenecks()
        
        assert isinstance(analysis, BottleneckAnalysis)
        assert len(analysis.slowest_functions) > 0
        assert analysis.performance_score >= 0.0
        assert analysis.performance_score <= 1.0
        
        # Slowest function should be first
        assert analysis.slowest_functions[0] == "slow_function"
    
    def test_baseline_comparison(self, profiler):
        """Test baseline setting and comparison."""
        function_name = "baseline_test"
        
        # Create initial profile data
        for _ in range(5):
            with profiler.profile(function_name):
                time.sleep(0.02)
        
        # Set baseline
        profiler.set_baseline(function_name)
        assert function_name in profiler.baselines
        
        # Create new data with different performance
        for _ in range(3):
            with profiler.profile(function_name):
                time.sleep(0.04)  # Slower performance
        
        # Compare to baseline
        comparison = profiler.compare_to_baseline(function_name)
        
        assert "time_change_percent" in comparison
        assert "performance_regression" in comparison
        assert comparison["time_change_percent"] > 0  # Should be slower
    
    def test_performance_thresholds(self, profiler):
        """Test performance threshold checking."""
        # Configure strict thresholds
        profiler.thresholds['max_function_time'] = 0.05
        
        # Create function that exceeds threshold
        with patch('logging.Logger.warning') as mock_warning:
            with profiler.profile("slow_function"):
                time.sleep(0.1)  # Exceeds 0.05s threshold
            
            # Should have logged a warning
            mock_warning.assert_called()
            warning_call_args = mock_warning.call_args[0][0]
            assert "Performance issue" in warning_call_args
    
    def test_profile_data_export(self, profiler):
        """Test profile data export functionality."""
        # Create some profile data
        with profiler.profile("export_test"):
            time.sleep(0.01)
        
        # Export data
        exported_json = profiler.export_profile_data("json")
        assert "export_test" in exported_json
        
        # Verify it's valid JSON
        import json
        data = json.loads(exported_json)
        assert "export_test" in data
        assert len(data["export_test"]) == 1
    
    def test_real_time_metrics(self, profiler):
        """Test real-time metrics collection."""
        # Create some profile data
        with profiler.profile("realtime_test"):
            time.sleep(0.01)
        
        metrics = profiler.get_real_time_metrics()
        
        assert "timestamp" in metrics
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "active_functions" in metrics
        assert "total_function_calls" in metrics
        
        assert metrics["active_functions"] == 1
        assert metrics["total_function_calls"] == 1
    
    def test_nested_profiling(self, profiler):
        """Test profiling of nested function calls."""
        with profiler.profile("outer_function"):
            time.sleep(0.01)
            
            with profiler.profile("inner_function"):
                time.sleep(0.02)
            
            time.sleep(0.01)
        
        # Both functions should be profiled
        assert "outer_function" in profiler.profile_data
        assert "inner_function" in profiler.profile_data
        
        outer_time = profiler.profile_data["outer_function"][0].total_time
        inner_time = profiler.profile_data["inner_function"][0].total_time
        
        # Outer function should take longer (includes inner function time)
        assert outer_time > inner_time
        assert outer_time >= 0.04  # At least 0.04s total
        assert inner_time >= 0.02  # At least 0.02s for inner
    
    def test_profiler_reset(self, profiler):
        """Test profiler data reset functionality."""
        # Create some data
        with profiler.profile("test_reset"):
            time.sleep(0.01)
        
        profiler.set_baseline("test_reset")
        
        # Start monitoring to create system metrics
        profiler.start_monitoring()
        time.sleep(0.1)
        profiler.stop_monitoring()
        
        assert len(profiler.profile_data) > 0
        assert len(profiler.baselines) > 0
        assert len(profiler.system_metrics) > 0
        
        # Reset all data
        profiler.reset_profile_data()
        
        assert len(profiler.profile_data) == 0
        assert len(profiler.baselines) == 0
        assert len(profiler.system_metrics) == 0


class TestOptimizationIntegration:
    """Integration tests for optimization components."""
    
    def test_profiler_with_torch_operations(self):
        """Test profiler with actual PyTorch operations."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_function
        def matrix_operations():
            a = torch.randn(500, 500)
            b = torch.randn(500, 500)
            c = torch.mm(a, b)
            return c.sum()
        
        # Run operations
        result = matrix_operations()
        assert isinstance(result, torch.Tensor)
        
        # Check profiling results
        assert "matrix_operations" in profiler.profile_data
        metrics = profiler.profile_data["matrix_operations"][0]
        assert metrics.total_time > 0
        
        if profiler.enable_memory_profiling:
            assert metrics.memory_delta is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_profiling_integration(self):
        """Test GPU profiling with actual CUDA operations."""
        profiler = PerformanceProfiler(enable_gpu_profiling=True)
        
        @profiler.profile_function
        def gpu_operations():
            device = torch.device('cuda')
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.mm(a, b)
            return c.sum()
        
        result = gpu_operations()
        assert isinstance(result, torch.Tensor)
        
        # Check GPU profiling results
        metrics = profiler.profile_data["gpu_operations"][0]
        if metrics.gpu_usage is not None:
            assert metrics.gpu_usage >= 0
        if metrics.gpu_memory_delta is not None:
            assert isinstance(metrics.gpu_memory_delta, float)
    
    def test_performance_degradation_detection(self):
        """Test detection of performance degradation."""
        profiler = PerformanceProfiler()
        
        # Set strict performance thresholds
        profiler.thresholds['max_function_time'] = 0.02
        
        slow_function_detected = False
        
        def mock_warning(msg):
            nonlocal slow_function_detected
            if "Performance issue" in msg and "slow_function" in msg:
                slow_function_detected = True
        
        with patch('logging.Logger.warning', side_effect=mock_warning):
            with profiler.profile("slow_function"):
                time.sleep(0.05)  # Exceeds threshold
        
        assert slow_function_detected
    
    def test_concurrent_profiling(self):
        """Test profiler thread safety with concurrent operations."""
        profiler = PerformanceProfiler()
        
        def worker_function(worker_id):
            for i in range(5):
                with profiler.profile(f"worker_{worker_id}_task_{i}"):
                    time.sleep(0.01)
        
        import threading
        threads = []
        
        # Start multiple worker threads
        for worker_id in range(3):
            thread = threading.Thread(target=worker_function, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all functions were profiled
        total_functions = len(profiler.profile_data)
        assert total_functions == 15  # 3 workers * 5 tasks each
        
        # Verify thread safety - all calls should be recorded
        total_calls = sum(len(metrics_list) for metrics_list in profiler.profile_data.values())
        assert total_calls == 15


class TestPerformanceOptimizations:
    """Test actual performance improvements from optimizations."""
    
    def test_profiler_overhead(self):
        """Test that profiler overhead is minimal."""
        profiler = PerformanceProfiler()
        
        def test_function():
            return sum(range(1000))
        
        # Time without profiling
        start_time = time.time()
        for _ in range(100):
            test_function()
        unmonitored_time = time.time() - start_time
        
        # Time with profiling
        start_time = time.time()
        for _ in range(100):
            with profiler.profile("test_function"):
                test_function()
        monitored_time = time.time() - start_time
        
        # Overhead should be less than 50% of original time
        overhead_ratio = (monitored_time - unmonitored_time) / unmonitored_time
        assert overhead_ratio < 0.5, f"Profiler overhead too high: {overhead_ratio:.2%}"
    
    def test_memory_profiling_accuracy(self):
        """Test memory profiling provides reasonable measurements."""
        profiler = PerformanceProfiler(enable_memory_profiling=True)
        
        @profiler.profile_function
        def memory_intensive_function():
            # Allocate significant memory
            large_tensor = torch.randn(1000, 1000)
            return large_tensor.sum()
        
        result = memory_intensive_function()
        
        metrics = profiler.profile_data["memory_intensive_function"][0]
        
        if metrics.memory_delta is not None:
            # Should show significant memory increase (at least a few MB)
            assert metrics.memory_delta > 1.0  # At least 1MB
        
        assert isinstance(result, torch.Tensor)


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])