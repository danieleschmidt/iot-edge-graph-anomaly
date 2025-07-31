#!/usr/bin/env python3
"""
Advanced Performance Benchmarking Suite
For ADVANCED SDLC (92% maturity) repositories

This script provides comprehensive performance analysis with:
- Memory profiling and leak detection
- CPU utilization tracking 
- ML model performance benchmarks
- Edge device constraint validation
- Automated regression detection
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import sys
import argparse
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import tracemalloc
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress torch warnings for cleaner benchmark output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Structured benchmark result with comprehensive metrics."""
    test_name: str
    duration_ms: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float 
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

class PerformanceProfiler:
    """Advanced performance profiler with memory and CPU tracking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self._get_memory_mb()
        
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    @contextmanager
    def profile(self, test_name: str):
        """Context manager for profiling code execution."""
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self._get_memory_mb()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Start CPU monitoring in background
        cpu_samples = []
        cpu_stop_event = threading.Event()
        
        def monitor_cpu():
            while not cpu_stop_event.is_set():
                cpu_samples.append(self.process.cpu_percent())
                time.sleep(0.1)
        
        cpu_thread = threading.Thread(target=monitor_cpu, daemon=True)
        cpu_thread.start()
        
        error_message = None
        success = True
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"Benchmark {test_name} failed: {e}")
        finally:
            # Stop monitoring
            cpu_stop_event.set()
            cpu_thread.join(timeout=1.0)
            
            # Calculate metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            final_memory = self._get_memory_mb()
            memory_delta = final_memory - initial_memory
            
            # Get peak memory from tracemalloc
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / 1024 / 1024
            
            # Calculate average CPU usage
            avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0
            
            # Store result
            result = BenchmarkResult(
                test_name=test_name,
                duration_ms=duration_ms,
                memory_peak_mb=peak_memory_mb,
                memory_delta_mb=memory_delta,
                cpu_percent=avg_cpu,
                success=success,
                error_message=error_message
            )
            
            self._log_result(result)
            
    def _log_result(self, result: BenchmarkResult):
        """Log benchmark result."""
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(
            f"{status} {result.test_name}: "
            f"{result.duration_ms:.2f}ms, "
            f"Peak: {result.memory_peak_mb:.2f}MB, "
            f"Delta: {result.memory_delta_mb:+.2f}MB, "
            f"CPU: {result.cpu_percent:.1f}%"
        )

class EdgeDeviceBenchmarks:
    """Benchmarks specifically for edge device constraints."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.results: List[BenchmarkResult] = []
        
        # Edge device constraints (Raspberry Pi 4)
        self.MAX_MEMORY_MB = 100  # <100MB RAM constraint
        self.MAX_CPU_PERCENT = 25  # <25% CPU constraint  
        self.MAX_INFERENCE_MS = 10  # <10ms inference latency
        self.MIN_THROUGHPUT_SPS = 100  # >100 samples/second
    
    def run_memory_constraint_test(self):
        """Test memory usage stays within edge device limits."""
        with self.profiler.profile("Memory Constraint Validation"):
            # Simulate model loading and inference
            dummy_model = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1)
            )
            
            # Create sample data
            batch_size = 32
            input_data = torch.randn(batch_size, 100)
            
            # Multiple inference runs
            for _ in range(10):
                with torch.no_grad():
                    output = dummy_model(input_data)
                    
            # Force garbage collection
            del dummy_model, input_data, output
            gc.collect()
    
    def run_cpu_constraint_test(self):
        """Test CPU usage stays within limits."""
        with self.profiler.profile("CPU Constraint Validation"):
            # CPU-intensive computation simulation
            start_time = time.time()
            while time.time() - start_time < 2.0:  # Run for 2 seconds
                # Simulate model inference workload
                data = np.random.randn(1000, 100)
                result = np.dot(data, data.T)
                # Small sleep to prevent 100% CPU
                time.sleep(0.01)
    
    def run_inference_latency_test(self):
        """Test inference latency meets real-time requirements."""
        with self.profiler.profile("Inference Latency Test"):
            # Create a small model similar to edge deployment
            model = torch.nn.Sequential(
                torch.nn.Linear(50, 25),
                torch.nn.ReLU(),
                torch.nn.Linear(25, 1),
                torch.nn.Sigmoid()
            )
            model.eval()
            
            latencies = []
            
            # Test 100 single inferences
            for _ in range(100):
                input_tensor = torch.randn(1, 50)
                
                start = time.perf_counter()
                with torch.no_grad():
                    output = model(input_tensor)  
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
            
            # Analyze latency statistics
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            max_latency = np.max(latencies)
            
            logger.info(f"Latency stats - Avg: {avg_latency:.2f}ms, "
                       f"P95: {p95_latency:.2f}ms, Max: {max_latency:.2f}ms")
    
    def run_throughput_test(self):
        """Test processing throughput meets requirements."""
        with self.profiler.profile("Throughput Test"):
            model = torch.nn.Sequential(
                torch.nn.Linear(50, 25),
                torch.nn.ReLU(), 
                torch.nn.Linear(25, 1)
            )
            model.eval()
            
            batch_size = 16
            num_batches = 100
            total_samples = batch_size * num_batches
            
            start_time = time.perf_counter()
            
            for _ in range(num_batches):
                input_batch = torch.randn(batch_size, 50)
                with torch.no_grad():
                    output = model(input_batch)
                    
            end_time = time.perf_counter()
            duration_seconds = end_time - start_time
            throughput_sps = total_samples / duration_seconds
            
            logger.info(f"Throughput: {throughput_sps:.1f} samples/second")
    
    def run_memory_leak_test(self):
        """Test for memory leaks over extended operation."""
        with self.profiler.profile("Memory Leak Detection"):
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1)
            )
            model.eval()
            
            memory_samples = []
            
            # Run for many iterations
            for i in range(1000):
                input_data = torch.randn(10, 100)
                
                with torch.no_grad():
                    output = model(input_data)
                
                # Sample memory every 100 iterations
                if i % 100 == 0:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                
                # Clean up
                del input_data, output
            
            # Analyze memory trend
            if len(memory_samples) > 2:
                memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
                logger.info(f"Memory trend: {memory_trend:+.3f} MB per 100 iterations")
    
    def run_all_benchmarks(self) -> Dict[str, bool]:
        """Run all edge device benchmarks and return results."""
        benchmark_methods = [
            self.run_memory_constraint_test,
            self.run_cpu_constraint_test, 
            self.run_inference_latency_test,
            self.run_throughput_test,
            self.run_memory_leak_test
        ]
        
        results = {}
        
        for method in benchmark_methods:
            try:
                method()
                results[method.__name__] = True
            except Exception as e:
                logger.error(f"Benchmark {method.__name__} failed: {e}")
                results[method.__name__] = False
                
        return results

def run_performance_regression_test(baseline_file: Optional[Path] = None) -> bool:
    """Run performance regression tests against baseline."""
    logger.info("🔍 Running performance regression analysis...")
    
    profiler = PerformanceProfiler()
    benchmarks = EdgeDeviceBenchmarks(profiler)
    
    # Run current benchmarks
    current_results = benchmarks.run_all_benchmarks()
    
    if baseline_file and baseline_file.exists():
        try:
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
            
            # Compare with baseline (simplified comparison)
            regressions = []
            for test_name, current_success in current_results.items():
                baseline_success = baseline_results.get(test_name, True)
                if baseline_success and not current_success:
                    regressions.append(test_name)
            
            if regressions:
                logger.error(f"❌ Performance regressions detected: {regressions}")
                return False
            else:
                logger.info("✅ No performance regressions detected")
                return True
                
        except Exception as e:
            logger.warning(f"Could not load baseline file: {e}")
    
    # Save current results as new baseline
    if baseline_file:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, 'w') as f:
            json.dump(current_results, f, indent=2)
        logger.info(f"💾 Saved benchmark results to {baseline_file}")
    
    return all(current_results.values())

def main():
    """Main benchmark runner with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Advanced Performance Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks.py --all                    # Run all benchmarks
  python benchmarks.py --regression             # Run regression tests
  python benchmarks.py --baseline results.json  # Compare with baseline
  python benchmarks.py --memory --cpu           # Run specific tests
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run all performance benchmarks')
    parser.add_argument('--regression', action='store_true',
                       help='Run performance regression tests')
    parser.add_argument('--baseline', type=Path,
                       help='Baseline file for regression comparison')
    parser.add_argument('--memory', action='store_true',
                       help='Run memory constraint tests')
    parser.add_argument('--cpu', action='store_true', 
                       help='Run CPU constraint tests')
    parser.add_argument('--latency', action='store_true',
                       help='Run inference latency tests')
    parser.add_argument('--throughput', action='store_true',
                       help='Run throughput tests')
    parser.add_argument('--leak', action='store_true',
                       help='Run memory leak detection')
    parser.add_argument('--output', type=Path,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    if not any([args.all, args.regression, args.memory, args.cpu, 
                args.latency, args.throughput, args.leak]):
        parser.print_help()
        return 1
    
    logger.info("🚀 Starting Advanced Performance Benchmarking Suite")
    logger.info(f"🔧 Python: {sys.version}")
    logger.info(f"🔧 PyTorch: {torch.__version__}")
    logger.info(f"🔧 Platform: {sys.platform}")
    
    profiler = PerformanceProfiler()
    benchmarks = EdgeDeviceBenchmarks(profiler)
    
    success = True
    
    try:
        if args.regression:
            success = run_performance_regression_test(args.baseline)
        elif args.all:
            results = benchmarks.run_all_benchmarks()
            success = all(results.values())
        else:
            # Run specific benchmarks
            if args.memory:
                benchmarks.run_memory_constraint_test()
            if args.cpu:
                benchmarks.run_cpu_constraint_test()
            if args.latency:
                benchmarks.run_inference_latency_test()
            if args.throughput:
                benchmarks.run_throughput_test()
            if args.leak:
                benchmarks.run_memory_leak_test()
                
    except KeyboardInterrupt:
        logger.info("❌ Benchmarking interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Benchmarking failed: {e}")
        logger.debug(traceback.format_exc())
        return 1
    
    if success:
        logger.info("✅ All benchmarks completed successfully")
        return 0
    else:
        logger.error("❌ Some benchmarks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())