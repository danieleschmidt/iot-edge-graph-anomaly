# Advanced Optimization Guide

This guide provides cutting-edge optimization strategies for the IoT Edge Graph Anomaly Detection system, appropriate for an ADVANCED SDLC maturity level (92%).

## Model Performance Optimization

### 1. Quantization for Edge Deployment

#### Post-Training Quantization
```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization for LSTM layers
model_int8 = quantize_dynamic(
    model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
)

# Reduces model size by ~75% with minimal accuracy loss
```

#### Quantization-Aware Training
```python
import torch.quantization as quant

# Prepare model for quantization-aware training
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model, inplace=False)

# Train with quantization simulation
for epoch in range(num_epochs):
    train_one_epoch(model_prepared, train_loader)
    
# Convert to quantized model
model_quantized = quant.convert(model_prepared, inplace=False)
```

### 2. Model Pruning

#### Structured Pruning
```python
import torch.nn.utils.prune as prune

# Prune 20% of connections in LSTM and Linear layers
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.LSTM)):
        prune.l1_unstructured(module, name='weight', amount=0.2)

# Remove pruning reparameterization
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.LSTM)):
        prune.remove(module, 'weight')
```

#### Magnitude-Based Pruning
```python
# Global magnitude pruning across all parameters
parameters_to_prune = [
    (module, 'weight') for name, module in model.named_modules()
    if isinstance(module, (torch.nn.Linear, torch.nn.LSTM))
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3  # Remove 30% of smallest magnitude weights
)
```

### 3. Knowledge Distillation

#### Teacher-Student Architecture
```python
class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature=4.0, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # Distillation loss
        kd_loss = self.kl_div(
            torch.log_softmax(student_logits / self.temperature, dim=1),
            torch.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Traditional loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss
```

### 4. Graph Optimization

#### GNN Layer Fusion
```python
class OptimizedGNNLayer(torch.nn.Module):
    """Fused GNN operations for better cache locality."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        # Fuse linear transformations
        self.fused_transform = torch.nn.Linear(in_features * 2, out_features)
        self.activation = torch.nn.ReLU(inplace=True)
    
    def forward(self, x, edge_index):
        # Fused neighbor aggregation and transformation
        row, col = edge_index
        neighbor_features = x[col]
        node_features = x[row]
        
        # Concatenate and transform in single operation
        combined = torch.cat([node_features, neighbor_features], dim=1)
        return self.activation(self.fused_transform(combined))
```

#### Sparse Tensor Optimization
```python
import torch.sparse

class SparseGNNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    
    def forward(self, x, adj_matrix_sparse):
        # Efficient sparse matrix multiplication
        aggregated = torch.sparse.mm(adj_matrix_sparse, x)
        return torch.mm(aggregated, self.weight)
```

## Memory Optimization

### 1. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLSTM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = torch.nn.LSTM(*args, **kwargs)
    
    def forward(self, x):
        # Trade compute for memory
        return checkpoint(self.lstm, x, use_reentrant=False)
```

### 2. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Memory-Efficient Attention
```python
class MemoryEfficientAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, chunk_size=1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.scale = (d_model // n_heads) ** -0.5
        
        self.qkv = torch.nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, C // self.n_heads)
        q, k, v = qkv.unbind(2)
        
        # Process attention in chunks to reduce memory
        output_chunks = []
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk_q = q[:, start:end]
            
            # Attention computation for chunk
            attn = torch.einsum('bthd,bshd->bhts', chunk_q, k) * self.scale
            attn = torch.softmax(attn, dim=-1)
            chunk_out = torch.einsum('bhts,bshd->bthd', attn, v)
            output_chunks.append(chunk_out)
        
        out = torch.cat(output_chunks, dim=1)
        return self.out_proj(out.reshape(B, T, C))
```

## Inference Optimization

### 1. TorchScript Compilation
```python
# Script the model for production deployment
model_scripted = torch.jit.script(model)

# Optimize for inference
model_optimized = torch.jit.optimize_for_inference(model_scripted)

# Save optimized model
torch.jit.save(model_optimized, 'model_optimized.pt')
```

### 2. ONNX Export and Optimization
```python
import onnx
import onnxruntime as ort
from onnxruntime.tools import optimizer

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Optimize ONNX model
optimized_model = optimizer.optimize_model(
    'model.onnx',
    model_type='bert',  # or appropriate model type
    opt_level=optimizer.OptLevel.ORT_ENABLE_ALL
)
optimized_model.save_model_to_file('model_optimized.onnx')

# Create optimized inference session
session = ort.InferenceSession(
    'model_optimized.onnx',
    providers=['CPUExecutionProvider']
)
```

### 3. Batch Processing Optimization
```python
class AdaptiveBatchProcessor:
    def __init__(self, model, max_batch_size=32, target_latency_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.current_batch_size = 1
        self.latency_history = []
    
    def process_batch(self, inputs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_history.append(latency_ms)
        
        # Adaptive batch size adjustment
        if len(self.latency_history) > 10:
            avg_latency = np.mean(self.latency_history[-10:])
            
            if avg_latency < self.target_latency_ms * 0.8:
                # Can increase batch size
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.max_batch_size
                )
            elif avg_latency > self.target_latency_ms:
                # Need to decrease batch size
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    1
                )
        
        return outputs
```

## System-Level Optimizations

### 1. CPU Affinity and Thread Optimization
```python
import os
import torch

# Set CPU affinity for consistent performance
os.sched_setaffinity(0, {0, 1})  # Use cores 0 and 1

# Optimize PyTorch threading
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# Enable JIT optimizations
torch.jit.set_optimization_level(3)
```

### 2. Memory Pool Optimization
```python
# Pre-allocate memory pools to reduce allocation overhead
class MemoryPool:
    def __init__(self, max_size_mb=50):
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.pool = {}
        self.current_size = 0
    
    def get_tensor(self, shape, dtype=torch.float32):
        key = (tuple(shape), dtype)
        
        if key not in self.pool:
            tensor = torch.empty(shape, dtype=dtype)
            tensor_size = tensor.numel() * tensor.element_size()
            
            if self.current_size + tensor_size <= self.max_size:
                self.pool[key] = tensor
                self.current_size += tensor_size
                return tensor.clone()
            else:
                return tensor
        
        return self.pool[key].clone()
    
    def clear(self):
        self.pool.clear()
        self.current_size = 0
```

### 3. Data Loading Optimization
```python
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=2):
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def __iter__(self):
        for batch in self.dataloader:
            # Non-blocking transfer to device
            batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            yield batch
```

## Monitoring and Profiling

### 1. Performance Profiler Integration
```python
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model_inference(model, input_data, warmup_steps=10, profile_steps=100):
    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(input_data)
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(profile_steps):
            with record_function("model_inference"):
                with torch.no_grad():
                    output = model(input_data)
    
    # Export results
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    return prof
```

### 2. Real-time Performance Monitoring
```python
import psutil
from collections import deque
import threading
import time

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        process = psutil.Process()
        
        while self.monitoring:
            # Monitor system resources
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_mb)
            
            time.sleep(0.1)  # Sample every 100ms
    
    def log_inference_latency(self, latency_ms):
        self.latencies.append(latency_ms)
    
    def get_statistics(self):
        if not self.latencies:
            return {}
        
        return {
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'samples': len(self.latencies)
        }
```

## Edge Deployment Optimizations

### 1. Container Optimization
```dockerfile
# Multi-stage build for minimal image size
FROM python:3.8-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.8-slim
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
USER app

# Copy only necessary files
COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/ ./models/

# Set environment variables for optimization
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV MKL_NUM_THREADS=2
ENV OMP_NUM_THREADS=2

EXPOSE 8080
CMD ["python", "-m", "src.iot_edge_anomaly.main"]
```

### 2. ARM64 Optimization
```python
import platform

def get_optimized_torch_config():
    """Get PyTorch configuration optimized for ARM64."""
    if platform.machine() == 'aarch64':
        # ARM64-specific optimizations
        torch.backends.mkldnn.enabled = False  # MKLDNN not optimal on ARM
        torch.set_num_threads(4)  # Raspberry Pi 4 has 4 cores
        
        # Use NEON instructions when available
        if hasattr(torch.backends, 'neon'):
            torch.backends.neon.enabled = True
    
    return torch.get_num_threads()
```

### 3. Dynamic Frequency Scaling
```python
class DynamicFrequencyScaler:
    """Adjust CPU frequency based on workload."""
    
    def __init__(self, min_freq=600, max_freq=1500):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.current_freq = max_freq
    
    def adjust_frequency(self, cpu_usage, target_usage=50):
        """Adjust CPU frequency based on usage."""
        try:
            if cpu_usage < target_usage * 0.7:
                # Scale down frequency
                new_freq = max(self.current_freq * 0.9, self.min_freq)
            elif cpu_usage > target_usage * 1.3:
                # Scale up frequency
                new_freq = min(self.current_freq * 1.1, self.max_freq)
            else:
                new_freq = self.current_freq
            
            if new_freq != self.current_freq:
                self._set_cpu_frequency(new_freq)
                self.current_freq = new_freq
                
        except Exception as e:
            # Fallback to default frequency
            pass
    
    def _set_cpu_frequency(self, freq_mhz):
        """Set CPU frequency (requires appropriate permissions)."""
        try:
            with open('/sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed', 'w') as f:
                f.write(str(int(freq_mhz * 1000)))
        except (PermissionError, FileNotFoundError):
            # Frequency scaling not available or no permissions
            pass
```

## Advanced Caching Strategies

### 1. Result Memoization
```python
from functools import lru_cache
import hashlib
import pickle

class InferenceCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _hash_input(self, input_tensor):
        """Create hash of input tensor for caching."""
        return hashlib.md5(pickle.dumps(input_tensor.cpu().numpy())).hexdigest()
    
    def get_or_compute(self, input_tensor, model_func):
        """Get cached result or compute and cache."""
        input_hash = self._hash_input(input_tensor)
        
        if input_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(input_hash)
            self.access_order.append(input_hash)
            return self.cache[input_hash]
        
        # Compute result
        result = model_func(input_tensor)
        
        # Add to cache
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[input_hash] = result
        self.access_order.append(input_hash)
        
        return result
```

### 2. Gradient Cache for Fine-tuning
```python
class GradientCache:
    """Cache gradients for efficient fine-tuning."""
    
    def __init__(self, cache_size=100):
        self.gradient_cache = {}
        self.cache_size = cache_size
        self.cache_keys = []
    
    def cache_gradients(self, model, input_hash):
        """Cache current model gradients."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        if len(self.cache_keys) >= self.cache_size:
            old_key = self.cache_keys.pop(0)
            del self.gradient_cache[old_key]
        
        self.gradient_cache[input_hash] = gradients
        self.cache_keys.append(input_hash)
    
    def apply_cached_gradients(self, model, input_hash, weight=1.0):
        """Apply cached gradients to model."""
        if input_hash not in self.gradient_cache:
            return False
        
        cached_grads = self.gradient_cache[input_hash]
        for name, param in model.named_parameters():
            if name in cached_grads and param.grad is not None:
                param.grad += weight * cached_grads[name]
        
        return True
```

This advanced optimization guide provides cutting-edge strategies for maximizing performance in production IoT edge deployments while maintaining the high code quality expected from a 92% SDLC maturity repository.