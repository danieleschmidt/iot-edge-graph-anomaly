# Advanced Performance Optimization Guide

## Overview

This guide covers advanced performance optimization techniques for the IoT edge anomaly detection system, focusing on resource-constrained edge devices.

## Edge Device Performance Targets

### Hardware Constraints (Raspberry Pi 4)
- **Memory**: < 100MB RAM usage
- **CPU**: < 25% utilization average
- **Inference**: < 10ms per sample
- **Throughput**: > 100 samples/second
- **Storage**: < 500MB total footprint

## Model Optimization Strategies

### 1. Model Quantization
```python
# Dynamic quantization for inference
import torch.quantization

# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)

# Static quantization for better performance
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
```

### 2. Model Pruning
```python
import torch.nn.utils.prune as prune

# Structured pruning for LSTM layers
prune.ln_structured(
    model.lstm, name="weight_ih_l0", 
    amount=0.3, n=2, dim=0
)

# Unstructured pruning for linear layers
prune.l1_unstructured(
    model.classifier, name="weight", amount=0.2
)
```

### 3. Knowledge Distillation
```python
# Teacher-student distillation
def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
```

## Memory Optimization

### 1. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# Use checkpointing for memory-intensive layers
def forward(self, x):
    x = checkpoint(self.lstm_layer, x)
    x = checkpoint(self.gnn_layer, x)
    return x
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    output = model(input_data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Memory Profiling
```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
```

## CPU Optimization

### 1. Threading Configuration
```python
import torch

# Optimize for edge devices
torch.set_num_threads(2)  # Limit threads for ARM64
torch.set_num_interop_threads(1)

# CPU-specific optimizations
if torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True
```

### 2. Vectorization
```python
import numpy as np

# Use NumPy vectorization for preprocessing
def vectorized_preprocess(data):
    # Replace loops with vectorized operations
    normalized = (data - data.mean(axis=0)) / data.std(axis=0)
    return normalized
```

### 3. Batch Processing Optimization
```python
# Optimal batch sizes for edge devices
EDGE_BATCH_SIZES = {
    'raspberry_pi_4': 8,
    'jetson_nano': 16,
    'x86_64_low_power': 32
}

def get_optimal_batch_size(device_type, available_memory_mb):
    base_size = EDGE_BATCH_SIZES.get(device_type, 8)
    if available_memory_mb < 512:
        return max(1, base_size // 2)
    return base_size
```

## Container Optimization

### 1. Multi-stage Docker Build
```dockerfile
# Optimized Dockerfile for edge deployment
FROM python:3.9-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY src/ /app/src/
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "src.iot_edge_anomaly.main"]
```

### 2. Alpine Linux Base
```dockerfile
FROM python:3.9-alpine AS production
RUN apk add --no-cache gcc musl-dev
# Minimal dependencies for edge deployment
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Monitoring and Profiling

### 1. Performance Monitoring
```python
import psutil
import time
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    cpu_percent: float
    memory_mb: float
    inference_time_ms: float
    throughput_samples_per_sec: float

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=end_memory,
            inference_time_ms=(end_time - start_time) * 1000,
            throughput_samples_per_sec=1 / (end_time - start_time)
        )
        
        # Log metrics to monitoring system
        export_metrics(metrics)
        
        return result
    return wrapper
```

### 2. Edge Device Benchmarking
```python
def benchmark_edge_performance(model, device_type, iterations=1000):
    """Comprehensive edge device performance benchmark"""
    
    results = {
        'memory_usage': [],
        'cpu_usage': [],
        'inference_times': [],
        'temperatures': []  # For thermal throttling detection
    }
    
    for i in range(iterations):
        # Memory before inference
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Inference timing
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(sample_input)
        end_time = time.perf_counter()
        
        # Collect metrics
        results['memory_usage'].append(mem_before)
        results['cpu_usage'].append(psutil.cpu_percent())
        results['inference_times'].append((end_time - start_time) * 1000)
        
        # Thermal monitoring (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                cpu_temp = temps['cpu_thermal'][0].current
                results['temperatures'].append(cpu_temp)
        except:
            pass
    
    return analyze_benchmark_results(results)
```

## Optimization Checklist

### Pre-deployment Optimization
- [ ] Model quantization applied (8-bit integer)
- [ ] Unnecessary layers pruned (20-30% reduction)
- [ ] Memory usage profiled and optimized
- [ ] CPU threading configured for target device
- [ ] Container image optimized (multi-stage build)
- [ ] Batch size tuned for device memory
- [ ] Thermal throttling considered

### Performance Validation
- [ ] Memory usage < 100MB validated
- [ ] CPU usage < 25% validated  
- [ ] Inference time < 10ms validated
- [ ] Throughput > 100 samples/sec validated
- [ ] Battery impact measured (mobile devices)
- [ ] Thermal performance tested
- [ ] Network bandwidth impact assessed

### Monitoring Setup
- [ ] Performance metrics collection enabled
- [ ] Alerting for performance regressions
- [ ] Continuous benchmarking in CI/CD
- [ ] Device-specific performance tracking
- [ ] Optimization recommendations automated

## Advanced Techniques

### 1. Neural Architecture Search (NAS)
```python
# Automated architecture optimization for edge constraints
from torchvision.models import mobilenet_v2

def edge_optimized_model(input_size, constraints):
    """Generate edge-optimized model architecture"""
    return mobilenet_v2(
        pretrained=False,
        width_mult=0.5,  # Reduce model width for edge deployment
        num_classes=2    # Binary anomaly classification
    )
```

### 2. Dynamic Inference
```python
def adaptive_inference(model, input_data, confidence_threshold=0.95):
    """Implement early exit for confident predictions"""
    with torch.no_grad():
        # Use simplified model for high-confidence cases
        quick_result = model.quick_classifier(input_data)
        confidence = torch.softmax(quick_result, dim=1).max()
        
        if confidence > confidence_threshold:
            return quick_result
        else:
            # Use full model for uncertain cases
            return model.full_forward(input_data)
```

### 3. Model Compilation
```python
# TorchScript optimization
scripted_model = torch.jit.script(model)
scripted_model = torch.jit.optimize_for_inference(scripted_model)

# Save optimized model
torch.jit.save(scripted_model, 'optimized_model.pt')
```

This optimization guide ensures maximum performance on resource-constrained edge devices while maintaining model accuracy and reliability.