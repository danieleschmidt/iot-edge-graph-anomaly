# Performance Benchmarks

This document defines performance benchmarks and testing procedures for the IoT Edge Graph Anomaly Detection system.

## Overview

Edge deployment requires strict performance constraints to ensure reliable operation on resource-limited devices like Raspberry Pi 4.

## Performance Targets

### Primary Constraints

| Metric | Target | Critical Threshold | Test Environment |
|--------|--------|-------------------|------------------|
| **Memory Usage** | <100MB | <128MB | Raspberry Pi 4 |
| **CPU Utilization** | <25% | <40% | Raspberry Pi 4 (ARM64) |
| **Inference Latency** | <10ms | <50ms | Per sample |
| **Throughput** | >100 samples/sec | >50 samples/sec | Sustained load |
| **Container Size** | <500MB | <1GB | Multi-arch image |
| **Startup Time** | <30 seconds | <60 seconds | Cold start |

### Secondary Metrics

| Metric | Target | Test Environment |
|--------|--------|------------------|
| **Model Loading Time** | <5 seconds | PyTorch model |
| **Memory Leak Rate** | 0MB/hour | 24-hour test |
| **Cache Hit Rate** | >90% | Inference cache |
| **Network Latency** | <100ms | Telemetry export |
| **Disk I/O** | <10MB/hour | Model + logs |
| **Battery Life Impact** | <5% | Mobile edge devices |

## Benchmarking Framework

### Test Infrastructure

```python
# tests/performance/test_benchmarks.py
import pytest
import time
import psutil
import torch
from memory_profiler import profile
from src.iot_edge_anomaly.models.lstm_autoencoder import LSTMAutoencoder

class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
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
```

### Benchmark Tests

#### Memory Usage Tests

```python
@pytest.mark.performance
@pytest.mark.slow
def test_memory_usage_limits(benchmark):
    """Test memory usage stays within limits."""
    # Initialize model
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    
    # Load test data
    test_data = torch.randn(1000, 10, 51)
    
    # Measure memory during inference
    memory_before = benchmark.get_memory_usage()
    
    with torch.no_grad():
        for batch in test_data.split(32):
            _ = model(batch)
    
    memory_after = benchmark.get_memory_usage()
    memory_used = memory_after - memory_before
    
    # Assert memory constraints
    assert memory_after < 100, f"Memory usage {memory_after:.1f}MB exceeds 100MB limit"
    assert memory_used < 50, f"Memory increase {memory_used:.1f}MB too high"

@pytest.mark.performance
def test_memory_leak_detection(benchmark):
    """Test for memory leaks during extended operation."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    test_data = torch.randn(100, 10, 51)
    
    memory_readings = []
    
    # Run inference 1000 times
    for i in range(1000):
        with torch.no_grad():
            _ = model(test_data[:1])
        
        if i % 100 == 0:
            memory_readings.append(benchmark.get_memory_usage())
    
    # Check for memory growth trend
    memory_growth = memory_readings[-1] - memory_readings[0]
    assert memory_growth < 5, f"Memory leak detected: {memory_growth:.1f}MB growth"
```

#### Performance Tests

```python
@pytest.mark.performance
def test_inference_latency(benchmark):
    """Test inference latency meets requirements."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    model.eval()
    
    test_sample = torch.randn(1, 10, 51)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_sample)
    
    # Measure latency over multiple runs
    latencies = []
    for _ in range(100):
        benchmark.start_timer()
        with torch.no_grad():
            _ = model(test_sample)
        latency = benchmark.stop_timer()
        latencies.append(latency * 1000)  # Convert to ms
    
    # Statistical analysis
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[95]
    p99_latency = sorted(latencies)[99]
    
    # Assert performance targets
    assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms exceeds 10ms"
    assert p95_latency < 20, f"P95 latency {p95_latency:.2f}ms exceeds 20ms"
    assert p99_latency < 50, f"P99 latency {p99_latency:.2f}ms exceeds 50ms"

@pytest.mark.performance
def test_throughput_capacity(benchmark):
    """Test system throughput under load."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    model.eval()
    
    batch_size = 32
    test_data = torch.randn(1000, 10, 51)
    
    benchmark.start_timer()
    samples_processed = 0
    
    with torch.no_grad():
        for batch in test_data.split(batch_size):
            _ = model(batch)
            samples_processed += len(batch)
    
    elapsed_time = benchmark.stop_timer()
    throughput = samples_processed / elapsed_time
    
    assert throughput > 100, f"Throughput {throughput:.1f} samples/sec below 100 target"
```

#### Resource Monitoring

```python
@pytest.mark.performance
@pytest.mark.slow
def test_sustained_cpu_usage(benchmark):
    """Test CPU usage under sustained load."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    test_data = torch.randn(100, 10, 51)
    
    cpu_readings = []
    
    # Run for 60 seconds
    start_time = time.time()
    while time.time() - start_time < 60:
        with torch.no_grad():
            _ = model(test_data)
        
        cpu_usage = benchmark.get_cpu_usage()
        cpu_readings.append(cpu_usage)
        time.sleep(0.1)
    
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    max_cpu = max(cpu_readings)
    
    assert avg_cpu < 25, f"Average CPU usage {avg_cpu:.1f}% exceeds 25%"
    assert max_cpu < 40, f"Peak CPU usage {max_cpu:.1f}% exceeds 40%"
```

## Continuous Benchmarking

### Automated Performance CI

```yaml
# .github/workflows/performance.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  performance:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform: [linux/amd64, linux/arm64]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Run performance tests
        run: |
          docker buildx build --platform ${{ matrix.platform }} \
            --target test \
            --output type=local,dest=./results .
          
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results-${{ matrix.platform }}
          path: ./results/performance/
```

### Performance Regression Detection

```python
# scripts/benchmark_regression.py
import json
import sys
from pathlib import Path
from typing import Dict, Any

class PerformanceRegression:
    """Detect performance regressions in benchmark results."""
    
    THRESHOLDS = {
        'memory_usage': 1.1,      # 10% increase
        'inference_latency': 1.2,  # 20% increase  
        'throughput': 0.9,         # 10% decrease
        'cpu_usage': 1.15,         # 15% increase
    }
    
    def __init__(self, baseline_file: str, current_file: str):
        self.baseline = self.load_results(baseline_file)
        self.current = self.load_results(current_file)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        with open(filename) as f:
            return json.load(f)
    
    def check_regressions(self) -> bool:
        """Check for performance regressions."""
        regressions = []
        
        for metric, threshold in self.THRESHOLDS.items():
            if metric not in self.baseline or metric not in self.current:
                continue
            
            baseline_value = self.baseline[metric]
            current_value = self.current[metric]
            
            if metric == 'throughput':
                # Higher is better for throughput
                ratio = current_value / baseline_value
                if ratio < threshold:
                    regressions.append(f"{metric}: {current_value:.2f} vs {baseline_value:.2f} (ratio: {ratio:.3f})")
            else:
                # Lower is better for other metrics
                ratio = current_value / baseline_value
                if ratio > threshold:
                    regressions.append(f"{metric}: {current_value:.2f} vs {baseline_value:.2f} (ratio: {ratio:.3f})")
        
        if regressions:
            print("Performance regressions detected:")
            for regression in regressions:
                print(f"  - {regression}")
            return False
        
        print("No performance regressions detected.")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmark_regression.py <baseline.json> <current.json>")
        sys.exit(1)
    
    checker = PerformanceRegression(sys.argv[1], sys.argv[2])
    if not checker.check_regressions():
        sys.exit(1)
```

## Hardware-Specific Testing

### Raspberry Pi 4 Validation

```bash
# scripts/rpi4_benchmark.sh
#!/bin/bash
set -e

echo "=== Raspberry Pi 4 Performance Validation ==="

# System info
echo "System Information:"
cat /proc/cpuinfo | grep "Model"
free -h
df -h

# Temperature monitoring
echo "Temperature monitoring enabled"
watch -n 1 'vcgencmd measure_temp' &
TEMP_PID=$!

# Run performance tests
echo "Running performance benchmarks..."
python -m pytest tests/performance/ -v --tb=short

# Stop temperature monitoring
kill $TEMP_PID

# Resource usage summary
echo "Resource Usage Summary:"
ps aux --sort=-%mem | head -10
```

### ARM64 Emulation Testing

```bash
# Test on ARM64 architecture
docker run --platform linux/arm64 --rm -v $(pwd):/app -w /app \
  python:3.11-slim bash -c "
    pip install -e .[test] &&
    python -m pytest tests/performance/ -v
  "
```

## Profiling and Analysis

### Memory Profiling

```python
# Run with memory profiler
@profile
def profile_model_inference():
    """Profile memory usage during model inference."""
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    test_data = torch.randn(1000, 10, 51)
    
    with torch.no_grad():
        for batch in test_data.split(32):
            output = model(batch)
            # Force garbage collection
            del output

# Usage: python -m memory_profiler profile_script.py
```

### CPU Profiling

```python
# Run with cProfile
import cProfile
import pstats

def profile_cpu_usage():
    """Profile CPU usage during inference."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your inference code here
    model = LSTMAutoencoder(input_size=51, hidden_size=64)
    test_data = torch.randn(100, 10, 51)
    
    with torch.no_grad():
        for batch in test_data.split(32):
            _ = model(batch)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

# Usage: python profile_script.py
```

## Reporting and Dashboards

### Performance Metrics Dashboard

Monitor key performance indicators:

- Real-time memory usage
- CPU utilization trends
- Inference latency distribution
- Throughput over time
- Temperature monitoring (edge devices)
- Battery usage (mobile devices)

### Alerting Rules

Set up alerts for:

- Memory usage >90MB (warning), >100MB (critical)
- CPU usage >20% (warning), >25% (critical)
- Inference latency >8ms (warning), >10ms (critical)
- Performance regression detection
- Container OOM kills
- Thermal throttling events

## Optimization Guidelines

### Memory Optimization

- Use `torch.no_grad()` during inference
- Clear intermediate tensors with `del`
- Enable garbage collection at checkpoints
- Use memory mapping for large datasets
- Implement tensor pooling for frequent allocations

### CPU Optimization

- Use optimized BLAS libraries (OpenBLAS, MKL)
- Enable PyTorch optimizations (`torch.jit.script`)
- Implement batch processing for multiple samples
- Use ARM-optimized containers
- Profile and optimize hot code paths

### Model Optimization

- Quantization (INT8, dynamic quantization)
- Model pruning for reduced size
- Knowledge distillation for smaller models
- ONNX runtime for faster inference
- TensorRT optimization (NVIDIA devices)

---

**Performance Benchmarks Version**: 1.0  
**Last Updated**: 2025-01-27  
**Next Review**: 2025-04-27