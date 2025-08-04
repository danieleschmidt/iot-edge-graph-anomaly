"""
Performance optimization utilities for IoT Edge Anomaly Detection.

This module provides performance monitoring, caching, batching, and optimization
features to ensure efficient operation on resource-constrained edge devices.
"""
import time
import threading
import queue
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=50))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=50))
    cache_hits: int = 0
    cache_misses: int = 0
    
    def record_execution(self, execution_time: float, memory_mb: float, cpu_percent: float):
        """Record execution metrics."""
        self.execution_times.append(execution_time)
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.execution_times:
            return {"operation": self.operation_name, "no_data": True}
        
        exec_times = list(self.execution_times)
        return {
            "operation": self.operation_name,
            "avg_execution_time_ms": sum(exec_times) / len(exec_times) * 1000,
            "min_execution_time_ms": min(exec_times) * 1000,
            "max_execution_time_ms": max(exec_times) * 1000,
            "p95_execution_time_ms": sorted(exec_times)[int(len(exec_times) * 0.95)] * 1000,
            "avg_memory_mb": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_executions": len(exec_times)
        }


class LRUCache:
    """Simple LRU cache implementation for tensor caching."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
                self.access_order.append(key)
                self.cache[key] = value
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used
                    lru_key = self.access_order.popleft()
                    del self.cache[lru_key]
                
                self.cache[key] = value
                self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "utilization": len(self.cache) / self.maxsize if self.maxsize > 0 else 0
            }


class TensorCache:
    """Specialized caching for tensor operations with smart key generation."""
    
    def __init__(self, maxsize: int = 64, ttl_seconds: int = 300):
        self.cache = LRUCache(maxsize)
        self.ttl_seconds = ttl_seconds
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def _generate_key(self, tensor: torch.Tensor, operation: str, **kwargs) -> str:
        """Generate cache key for tensor and operation."""
        # Use tensor shape, dtype, and hash of small sample for key
        sample_data = tensor.flatten()[:min(10, tensor.numel())]
        sample_tuple = tuple(sample_data.tolist())  # Convert to hashable tuple
        
        key_parts = [
            operation,
            str(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            str(hash(sample_tuple))
        ]
        
        # Add kwargs to key
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return "|".join(key_parts)
    
    def get(self, tensor: torch.Tensor, operation: str, **kwargs) -> Optional[torch.Tensor]:
        """Get cached result."""
        key = self._generate_key(tensor, operation, **kwargs)
        
        with self.lock:
            # Check TTL
            if key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    self._remove_key(key)
                    return None
            
            result = self.cache.get(key)
            return result.clone() if result is not None else None
    
    def put(self, tensor: torch.Tensor, operation: str, result: torch.Tensor, **kwargs):
        """Cache result."""
        key = self._generate_key(tensor, operation, **kwargs)
        
        with self.lock:
            # Store cloned result to avoid reference issues
            self.cache.put(key, result.clone())
            self.timestamps[key] = time.time()
    
    def _remove_key(self, key: str):
        """Remove key from cache and timestamps."""
        if key in self.timestamps:
            del self.timestamps[key]
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self.cache.get_stats()
        
        with self.lock:
            expired_count = sum(1 for ts in self.timestamps.values() 
                              if time.time() - ts > self.ttl_seconds)
            
            cache_stats.update({
                "ttl_seconds": self.ttl_seconds,
                "expired_entries": expired_count,
                "timestamp_entries": len(self.timestamps)
            })
        
        return cache_stats


class BatchProcessor:
    """Batch processing for improved throughput."""
    
    def __init__(self, batch_size: int = 8, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: List[Tuple[torch.Tensor, Future]] = []
        self.lock = threading.Lock()
        self.last_batch_time = time.time()
    
    def process_batch(self, processor_func: Callable, tensor: torch.Tensor) -> torch.Tensor:
        """
        Process tensor with batching for efficiency.
        
        Args:
            processor_func: Function to process batched tensors
            tensor: Input tensor
            
        Returns:
            Processed tensor result
        """
        future = Future()
        
        with self.lock:
            self.pending_requests.append((tensor, future))
            should_process = (len(self.pending_requests) >= self.batch_size or 
                            time.time() - self.last_batch_time > self.max_wait_time)
        
        if should_process:
            self._process_pending_batch(processor_func)
        
        return future.result(timeout=30.0)  # 30 second timeout
    
    def _process_pending_batch(self, processor_func: Callable):
        """Process pending batch of requests."""
        with self.lock:
            if not self.pending_requests:
                return
            
            batch_requests = self.pending_requests.copy()
            self.pending_requests.clear()
            self.last_batch_time = time.time()
        
        try:
            # Combine tensors into batch
            tensors = [req[0] for req in batch_requests]
            futures = [req[1] for req in batch_requests]
            
            # Stack tensors (assumes same shape)
            if len(tensors) == 1:
                batched_input = tensors[0]
            else:
                batched_input = torch.stack(tensors, dim=0)
            
            # Process batch
            batch_result = processor_func(batched_input)
            
            # Split results back to individual requests
            if len(futures) == 1:
                futures[0].set_result(batch_result)
            else:
                for i, future in enumerate(futures):
                    future.set_result(batch_result[i])
                    
        except Exception as e:
            # Set exception for all futures
            for _, future in batch_requests:
                future.set_exception(e)


class PerformanceMonitor:
    """Comprehensive performance monitoring and optimization."""
    
    def __init__(self, enable_caching: bool = True, enable_batching: bool = True):
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(lambda: PerformanceMetrics(""))
        
        # Caching
        self.tensor_cache = TensorCache(maxsize=32, ttl_seconds=300) if enable_caching else None
        self.result_cache = LRUCache(maxsize=128) if enable_caching else None
        
        # Batching
        self.batch_processor = BatchProcessor() if enable_batching else None
        
        # Background cleanup thread
        if enable_caching:
            self._start_cleanup_thread()
        
        logger.info(f"Performance monitor initialized (caching={enable_caching}, batching={enable_batching})")
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(30)  # Cleanup every 30 seconds
                    if self.tensor_cache:
                        self.tensor_cache.cleanup_expired()
                    gc.collect()  # Garbage collection
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def monitor_operation(self, operation_name: str):
        """Decorator for monitoring operation performance."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return self._execute_with_monitoring(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_monitoring(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with performance monitoring."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            
            # Record successful execution
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            # Initialize metrics if needed
            if operation_name not in self.metrics:
                self.metrics[operation_name] = PerformanceMetrics(operation_name)
            
            self.metrics[operation_name].record_execution(
                execution_time, 
                end_memory - start_memory,
                (start_cpu + end_cpu) / 2
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Performance monitoring - {operation_name} failed: {e}")
            raise
    
    def cached_tensor_operation(self, operation_name: str, tensor: torch.Tensor, func: Callable, **kwargs) -> torch.Tensor:
        """Execute tensor operation with caching."""
        if not self.enable_caching or self.tensor_cache is None:
            return func(tensor, **kwargs)
        
        # Check cache first
        cached_result = self.tensor_cache.get(tensor, operation_name, **kwargs)
        if cached_result is not None:
            self.metrics[operation_name].cache_hits += 1
            return cached_result
        
        # Cache miss - compute result
        self.metrics[operation_name].cache_misses += 1
        result = func(tensor, **kwargs)
        
        # Cache result
        self.tensor_cache.put(tensor, operation_name, result, **kwargs)
        
        return result
    
    def batched_operation(self, operation_name: str, tensor: torch.Tensor, func: Callable) -> torch.Tensor:
        """Execute operation with batching optimization."""
        if not self.enable_batching or self.batch_processor is None:
            return func(tensor)
        
        return self.batch_processor.process_batch(func, tensor)
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference performance."""
        logger.info("Optimizing model for inference...")
        
        # Set to evaluation mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Try to compile model if PyTorch 2.0+
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
        
        # Try to use TorchScript for optimization
        try:
            dummy_input = torch.randn(1, 20, 5)  # Adjust based on expected input
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Model optimized with TorchScript")
            return traced_model
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
            return model
    
    def suggest_optimizations(self) -> List[str]:
        """Analyze performance and suggest optimizations."""
        suggestions = []
        
        for op_name, metrics in self.metrics.items():
            stats = metrics.get_stats()
            
            if stats.get('no_data'):
                continue
            
            # Check average execution time
            avg_time_ms = stats['avg_execution_time_ms']
            if avg_time_ms > 100:  # >100ms
                suggestions.append(f"{op_name}: High execution time ({avg_time_ms:.1f}ms) - consider batching or caching")
            
            # Check cache hit rate
            hit_rate = stats.get('cache_hit_rate', 0)
            if hit_rate < 0.5 and metrics.cache_hits + metrics.cache_misses > 10:
                suggestions.append(f"{op_name}: Low cache hit rate ({hit_rate:.1%}) - review caching strategy")
            
            # Check memory usage
            avg_memory = stats.get('avg_memory_mb', 0)
            if avg_memory > 50:  # >50MB per operation
                suggestions.append(f"{op_name}: High memory usage ({avg_memory:.1f}MB) - consider reducing batch size")
            
            # Check CPU usage
            avg_cpu = stats.get('avg_cpu_percent', 0)
            if avg_cpu > 80:  # >80% CPU
                suggestions.append(f"{op_name}: High CPU usage ({avg_cpu:.1f}%) - consider optimization")
        
        return suggestions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "operations": {},
            "caching": {},
            "system": {
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(),
                "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024
            },
            "suggestions": self.suggest_optimizations()
        }
        
        # Add operation metrics
        for op_name, metrics in self.metrics.items():
            report["operations"][op_name] = metrics.get_stats()
        
        # Add caching stats
        if self.tensor_cache:
            report["caching"]["tensor_cache"] = self.tensor_cache.get_stats()
        if self.result_cache:
            report["caching"]["result_cache"] = self.result_cache.get_stats()
        
        return report
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics.clear()
        if self.tensor_cache:
            self.tensor_cache.cache.clear()
        if self.result_cache:
            self.result_cache.clear()
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()