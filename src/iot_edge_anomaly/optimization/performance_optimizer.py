"""
Advanced performance optimization framework for IoT edge anomaly detection.

This module provides comprehensive performance optimization including:
- Dynamic model optimization and quantization
- Intelligent caching strategies
- Adaptive batch processing
- Resource-aware scheduling
- Auto-scaling triggers
- Performance profiling and monitoring
"""
import torch
import torch.nn as nn
import numpy as np
import time
import threading
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, PriorityQueue, Empty
import psutil
import logging
import pickle
from pathlib import Path
import json
from collections import defaultdict, deque
import hashlib
import gc
import warnings

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for different deployment scenarios."""
    DEVELOPMENT = "development"
    EDGE_BASIC = "edge_basic"
    EDGE_AGGRESSIVE = "edge_aggressive"
    CLOUD_OPTIMIZED = "cloud_optimized"


class CacheStrategy(Enum):
    """Caching strategies for different data patterns."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    cache_hit_rate: float
    throughput_samples_per_sec: float
    model_size_mb: float
    energy_efficiency_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization': self.cpu_utilization,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'model_size_mb': self.model_size_mb,
            'energy_efficiency_score': self.energy_efficiency_score
        }


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""
    optimization_level: OptimizationLevel = OptimizationLevel.EDGE_BASIC
    enable_quantization: bool = True
    enable_caching: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_cache_size_mb: int = 50
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 100
    enable_async_processing: bool = True
    max_concurrent_requests: int = 10
    enable_model_pruning: bool = False
    pruning_sparsity: float = 0.3
    target_memory_mb: int = 100
    target_inference_time_ms: float = 10.0


class IntelligentCache:
    """Intelligent caching system with adaptive strategies."""
    
    def __init__(self, max_size_mb: int = 50, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size_mb = max_size_mb
        self.strategy = strategy
        self.cache = {}
        self.access_times = defaultdict(list)
        self.access_counts = defaultdict(int)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_mb': 0.0
        }
        self._lock = threading.Lock()
        
        # Adaptive cache parameters
        self.hit_rate_threshold = 0.7
        self.access_pattern_window = 1000
        self.strategy_adaptation_interval = 100
        self.access_history = deque(maxlen=self.access_pattern_window)
        
    def _compute_key_hash(self, data: torch.Tensor) -> str:
        """Compute hash key for tensor data."""
        return hashlib.md5(data.cpu().numpy().tobytes()).hexdigest()
    
    def _estimate_size_mb(self, obj: Any) -> float:
        """Estimate object size in MB."""
        try:
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.numel() / (1024 * 1024)
            else:
                # Rough estimation for other objects
                return len(pickle.dumps(obj)) / (1024 * 1024)
        except Exception:
            return 1.0  # Default estimate
    
    def _should_evict_lru(self) -> str:
        """Find key to evict using LRU strategy."""
        if not self.access_times:
            return list(self.cache.keys())[0]
        
        # Find least recently used key
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: max(self.access_times[k]) if self.access_times[k] else 0)
        return oldest_key
    
    def _should_evict_lfu(self) -> str:
        """Find key to evict using LFU strategy."""
        if not self.access_counts:
            return list(self.cache.keys())[0]
        
        # Find least frequently used key
        return min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
    
    def _adapt_strategy(self):
        """Adapt caching strategy based on access patterns."""
        if len(self.access_history) < self.strategy_adaptation_interval:
            return
        
        # Analyze access patterns
        recent_accesses = list(self.access_history)[-self.strategy_adaptation_interval:]
        unique_accesses = len(set(recent_accesses))
        repetition_rate = 1 - (unique_accesses / len(recent_accesses))
        
        current_hit_rate = self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses'])
        
        # Strategy adaptation logic
        if current_hit_rate < self.hit_rate_threshold:
            if repetition_rate > 0.5:  # High repetition, favor LRU
                self.strategy = CacheStrategy.LRU
            else:  # Low repetition, favor LFU
                self.strategy = CacheStrategy.LFU
        
        logger.debug(f"Cache strategy adapted to {self.strategy.value}, hit rate: {current_hit_rate:.3f}")
    
    def get(self, key: torch.Tensor) -> Optional[Any]:
        """Get cached result."""
        hash_key = self._compute_key_hash(key)
        
        with self._lock:
            if hash_key in self.cache:
                # Update access patterns
                self.access_times[hash_key].append(time.time())
                self.access_counts[hash_key] += 1
                self.access_history.append(hash_key)
                self.cache_stats['hits'] += 1
                
                return self.cache[hash_key]
            else:
                self.cache_stats['misses'] += 1
                self.access_history.append(hash_key)
                return None
    
    def put(self, key: torch.Tensor, value: Any):
        """Cache result with intelligent eviction."""
        hash_key = self._compute_key_hash(key)
        value_size = self._estimate_size_mb(value)
        
        with self._lock:
            # Check if we need to evict
            while (self.cache_stats['size_mb'] + value_size > self.max_size_mb and 
                   len(self.cache) > 0):
                
                # Choose eviction key based on strategy
                if self.strategy == CacheStrategy.LRU:
                    evict_key = self._should_evict_lru()
                elif self.strategy == CacheStrategy.LFU:
                    evict_key = self._should_evict_lfu()
                else:  # ADAPTIVE
                    self._adapt_strategy()
                    evict_key = (self._should_evict_lru() if self.strategy == CacheStrategy.LRU 
                               else self._should_evict_lfu())
                
                # Evict the chosen key
                if evict_key in self.cache:
                    evicted_size = self._estimate_size_mb(self.cache[evict_key])
                    del self.cache[evict_key]
                    self.cache_stats['size_mb'] -= evicted_size
                    self.cache_stats['evictions'] += 1
                    
                    # Clean up access tracking
                    if evict_key in self.access_times:
                        del self.access_times[evict_key]
                    if evict_key in self.access_counts:
                        del self.access_counts[evict_key]
                else:
                    break  # Prevent infinite loop
            
            # Add new entry
            self.cache[hash_key] = value
            self.cache_stats['size_mb'] += value_size
            self.access_times[hash_key].append(time.time())
            self.access_counts[hash_key] = 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / max(1, total_requests)
            
            return {
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.cache_stats['evictions'],
                'size_mb': self.cache_stats['size_mb'],
                'entries': len(self.cache),
                'strategy': self.strategy.value
            }


class BatchProcessor:
    """Intelligent batch processor for optimizing inference throughput."""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.batch_queue = Queue()
        self.result_futures = {}
        self.batch_stats = {
            'batches_processed': 0,
            'total_samples': 0,
            'avg_batch_size': 0.0,
            'timeout_batches': 0
        }
        self._processing = False
        self._processor_thread = None
    
    def start_processing(self, inference_func: Callable[[torch.Tensor], Any]):
        """Start batch processing thread."""
        self._processing = True
        self._processor_thread = threading.Thread(
            target=self._process_batches, 
            args=(inference_func,),
            daemon=True
        )
        self._processor_thread.start()
        logger.info("Batch processor started")
    
    def stop_processing(self):
        """Stop batch processing."""
        self._processing = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5)
        logger.info("Batch processor stopped")
    
    def submit_request(self, data: torch.Tensor, request_id: str) -> 'BatchFuture':
        """Submit inference request for batch processing."""
        future = BatchFuture(request_id)
        self.batch_queue.put((data, request_id, future))
        return future
    
    def _process_batches(self, inference_func: Callable[[torch.Tensor], Any]):
        """Main batch processing loop."""
        while self._processing:
            batch_data = []
            batch_requests = []
            batch_futures = []
            
            start_time = time.time()
            
            try:
                # Collect batch items with timeout
                while (len(batch_data) < self.max_batch_size and 
                       (time.time() - start_time) * 1000 < self.timeout_ms):
                    
                    try:
                        data, request_id, future = self.batch_queue.get(timeout=0.01)
                        batch_data.append(data)
                        batch_requests.append(request_id)
                        batch_futures.append(future)
                    except Empty:
                        if len(batch_data) > 0:
                            break  # Process partial batch
                        continue
                
                if not batch_data:
                    continue
                
                # Process batch
                try:
                    # Stack tensors into batch
                    batch_tensor = torch.stack(batch_data)
                    
                    # Run inference
                    start_inference = time.time()
                    batch_results = inference_func(batch_tensor)
                    inference_time = (time.time() - start_inference) * 1000
                    
                    # Distribute results to futures
                    if isinstance(batch_results, torch.Tensor):
                        # Handle tensor results (split by batch dimension)
                        for i, future in enumerate(batch_futures):
                            if i < len(batch_results):
                                future.set_result(batch_results[i])
                            else:
                                future.set_exception(IndexError("Batch result index out of range"))
                    else:
                        # Handle other result types
                        for future in batch_futures:
                            future.set_result(batch_results)
                    
                    # Update statistics
                    self.batch_stats['batches_processed'] += 1
                    self.batch_stats['total_samples'] += len(batch_data)
                    self.batch_stats['avg_batch_size'] = (
                        self.batch_stats['total_samples'] / self.batch_stats['batches_processed']
                    )
                    
                    if (time.time() - start_time) * 1000 >= self.timeout_ms:
                        self.batch_stats['timeout_batches'] += 1
                    
                    logger.debug(f"Processed batch of {len(batch_data)} samples in {inference_time:.2f}ms")
                    
                except Exception as e:
                    # Set exception for all futures in failed batch
                    for future in batch_futures:
                        future.set_exception(e)
                    logger.error(f"Batch processing failed: {e}")
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on persistent errors


class BatchFuture:
    """Future-like object for batch processing results."""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self._result = None
        self._exception = None
        self._done = False
        self._condition = threading.Condition()
    
    def set_result(self, result: Any):
        """Set the result."""
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()
    
    def set_exception(self, exception: Exception):
        """Set an exception."""
        with self._condition:
            self._exception = exception
            self._done = True
            self._condition.notify_all()
    
    def result(self, timeout: Optional[float] = None) -> Any:
        """Get the result, blocking if necessary."""
        with self._condition:
            if not self._done:
                self._condition.wait(timeout)
            
            if not self._done:
                raise TimeoutError("Future timed out")
            
            if self._exception:
                raise self._exception
            
            return self._result
    
    def done(self) -> bool:
        """Check if the future is done."""
        with self._condition:
            return self._done


class ModelOptimizer:
    """Advanced model optimization techniques."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []
    
    def optimize_model(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply comprehensive model optimizations."""
        optimization_report = {
            'original_size_mb': self._compute_model_size(model),
            'original_params': sum(p.numel() for p in model.parameters()),
            'optimizations_applied': []
        }
        
        optimized_model = model
        
        try:
            # 1. Quantization
            if self.config.enable_quantization:
                optimized_model, quant_report = self._apply_quantization(optimized_model)
                optimization_report['optimizations_applied'].append('quantization')
                optimization_report['quantization_report'] = quant_report
            
            # 2. Pruning
            if self.config.enable_model_pruning:
                optimized_model, prune_report = self._apply_pruning(optimized_model)
                optimization_report['optimizations_applied'].append('pruning')
                optimization_report['pruning_report'] = prune_report
            
            # 3. Graph optimization (JIT compilation)
            if self.config.optimization_level in [OptimizationLevel.EDGE_AGGRESSIVE, OptimizationLevel.CLOUD_OPTIMIZED]:
                optimized_model, jit_report = self._apply_jit_optimization(optimized_model)
                optimization_report['optimizations_applied'].append('jit_compilation')
                optimization_report['jit_report'] = jit_report
            
            # Final metrics
            optimization_report['optimized_size_mb'] = self._compute_model_size(optimized_model)
            optimization_report['optimized_params'] = sum(p.numel() for p in optimized_model.parameters() if p.requires_grad)
            optimization_report['size_reduction_ratio'] = (
                optimization_report['original_size_mb'] / max(optimization_report['optimized_size_mb'], 0.001)
            )
            
            self.optimization_history.append(optimization_report)
            logger.info(f"Model optimization complete. Size reduced by {optimization_report['size_reduction_ratio']:.2f}x")
            
            return optimized_model, optimization_report
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model, optimization_report
    
    def _compute_model_size(self, model: nn.Module) -> float:
        """Compute model size in MB."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 * 1024)
    
    def _apply_quantization(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply dynamic quantization."""
        try:
            # Dynamic quantization for inference
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.GRU}, 
                dtype=torch.qint8
            )
            
            original_size = self._compute_model_size(model)
            quantized_size = self._compute_model_size(quantized_model)
            
            report = {
                'method': 'dynamic_quantization',
                'dtype': 'qint8',
                'size_reduction': original_size / max(quantized_size, 0.001),
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size
            }
            
            return quantized_model, report
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model, {'error': str(e)}
    
    def _apply_pruning(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply structured pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply magnitude-based pruning
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.pruning_sparsity
                )
                
                # Remove pruning reparameterization
                for module, param_name in parameters_to_prune:
                    prune.remove(module, param_name)
            
            original_params = sum(p.numel() for p in model.parameters())
            remaining_params = sum((p != 0).sum().item() for p in model.parameters())
            
            report = {
                'method': 'l1_unstructured_pruning',
                'target_sparsity': self.config.pruning_sparsity,
                'actual_sparsity': 1 - (remaining_params / max(original_params, 1)),
                'original_params': original_params,
                'remaining_params': remaining_params
            }
            
            return model, report
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return model, {'error': str(e)}
    
    def _apply_jit_optimization(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply JIT compilation optimization."""
        try:
            model.eval()
            
            # Create example input for tracing
            if hasattr(model, 'input_size'):
                input_size = model.input_size
            else:
                input_size = 5  # Default
            
            example_input = torch.randn(1, 20, input_size)
            
            # TorchScript tracing
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            report = {
                'method': 'torchscript_tracing',
                'optimized_for_inference': True,
                'input_shape': list(example_input.shape)
            }
            
            return traced_model, report
            
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
            return model, {'error': str(e)}


class ResourceMonitor:
    """Monitors system resources and triggers optimization actions."""
    
    def __init__(self, target_memory_mb: int = 100, target_cpu_percent: float = 80.0):
        self.target_memory_mb = target_memory_mb
        self.target_cpu_percent = target_cpu_percent
        self.resource_history = deque(maxlen=60)  # 1 minute of history at 1s intervals
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = {
            'memory_pressure': [],
            'cpu_pressure': [],
            'resource_normal': []
        }
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def add_callback(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for resource events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect current metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                current_memory_mb = memory_info.used / (1024 * 1024)
                
                metrics = {
                    'timestamp': time.time(),
                    'memory_used_mb': current_memory_mb,
                    'memory_percent': memory_info.percent,
                    'cpu_percent': cpu_percent,
                    'memory_available_mb': memory_info.available / (1024 * 1024)
                }
                
                self.resource_history.append(metrics)
                
                # Check for resource pressure
                memory_pressure = current_memory_mb > self.target_memory_mb
                cpu_pressure = cpu_percent > self.target_cpu_percent
                
                if memory_pressure:
                    for callback in self.callbacks['memory_pressure']:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.warning(f"Memory pressure callback failed: {e}")
                
                if cpu_pressure:
                    for callback in self.callbacks['cpu_pressure']:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.warning(f"CPU pressure callback failed: {e}")
                
                if not memory_pressure and not cpu_pressure:
                    for callback in self.callbacks['resource_normal']:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.warning(f"Resource normal callback failed: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        if not self.resource_history:
            return {'status': 'no_data'}
        
        current = self.resource_history[-1]
        
        # Calculate trends
        if len(self.resource_history) > 10:
            recent = list(self.resource_history)[-10:]
            memory_trend = recent[-1]['memory_used_mb'] - recent[0]['memory_used_mb']
            cpu_trend = np.mean([m['cpu_percent'] for m in recent[-5:]])  # Recent average
        else:
            memory_trend = 0
            cpu_trend = current['cpu_percent']
        
        return {
            'current_memory_mb': current['memory_used_mb'],
            'memory_pressure': current['memory_used_mb'] > self.target_memory_mb,
            'current_cpu_percent': current['cpu_percent'],
            'cpu_pressure': current['cpu_percent'] > self.target_cpu_percent,
            'memory_trend_mb': memory_trend,
            'cpu_trend_percent': cpu_trend,
            'target_memory_mb': self.target_memory_mb,
            'target_cpu_percent': self.target_cpu_percent
        }


class OptimizedInferenceEngine:
    """Main optimized inference engine with comprehensive performance enhancements."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.original_model = model
        self.config = config
        
        # Optimization components
        self.model_optimizer = ModelOptimizer(config)
        self.cache = IntelligentCache(config.max_cache_size_mb, config.cache_strategy) if config.enable_caching else None
        self.batch_processor = BatchProcessor(config.max_batch_size, config.batch_timeout_ms) if config.enable_batching else None
        self.resource_monitor = ResourceMonitor(config.target_memory_mb, 80.0)
        
        # Optimized model
        self.optimized_model, self.optimization_report = self.model_optimizer.optimize_model(model)
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.total_requests = 0
        self.optimization_stats = {
            'cache_enabled': config.enable_caching,
            'batch_enabled': config.enable_batching,
            'model_optimized': True,
            'optimization_level': config.optimization_level.value
        }
        
        # Threading for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests) if config.enable_async_processing else None
        
        # Setup resource monitoring callbacks
        self._setup_resource_callbacks()
        
        # Start background services
        self._start_services()
        
        logger.info(f"Optimized inference engine initialized with level: {config.optimization_level.value}")
    
    def _setup_resource_callbacks(self):
        """Setup callbacks for resource monitoring."""
        def on_memory_pressure(metrics):
            logger.warning(f"Memory pressure detected: {metrics['memory_used_mb']:.1f}MB")
            if self.cache:
                # Reduce cache size under memory pressure
                old_max = self.cache.max_size_mb
                self.cache.max_size_mb = max(10, old_max * 0.7)  # Reduce by 30%
                logger.info(f"Cache size reduced from {old_max}MB to {self.cache.max_size_mb}MB")
            
            # Force garbage collection
            gc.collect()
        
        def on_cpu_pressure(metrics):
            logger.warning(f"CPU pressure detected: {metrics['cpu_percent']:.1f}%")
            # Could implement CPU-specific optimizations here
        
        def on_resource_normal(metrics):
            # Restore normal cache size when resources are available
            if self.cache and self.cache.max_size_mb < self.config.max_cache_size_mb:
                self.cache.max_size_mb = min(
                    self.config.max_cache_size_mb,
                    self.cache.max_size_mb * 1.1  # Gradual increase
                )
        
        self.resource_monitor.add_callback('memory_pressure', on_memory_pressure)
        self.resource_monitor.add_callback('cpu_pressure', on_cpu_pressure)
        self.resource_monitor.add_callback('resource_normal', on_resource_normal)
    
    def _start_services(self):
        """Start background services."""
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Start batch processor if enabled
        if self.batch_processor:
            self.batch_processor.start_processing(self._batch_inference)
    
    def _batch_inference(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Internal batch inference method."""
        self.optimized_model.eval()
        with torch.no_grad():
            if hasattr(self.optimized_model, 'compute_reconstruction_error'):
                # Handle reconstruction error computation for batches
                batch_errors = []
                for i in range(batch_data.size(0)):
                    sample = batch_data[i:i+1]
                    error = self.optimized_model.compute_reconstruction_error(sample, reduction='mean')
                    batch_errors.append(error)
                return torch.stack(batch_errors)
            else:
                # Generic inference
                return self.optimized_model(batch_data)
    
    def predict(self, data: torch.Tensor, use_cache: bool = True, use_batch: bool = True) -> Dict[str, Any]:
        """Optimized prediction with all performance enhancements."""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Check cache first
            if self.cache and use_cache:
                cached_result = self.cache.get(data)
                if cached_result is not None:
                    end_time = time.time()
                    self._record_performance_metrics(
                        inference_time_ms=(end_time - start_time) * 1000,
                        cache_hit=True,
                        batch_used=False
                    )
                    return {
                        'success': True,
                        'result': cached_result,
                        'source': 'cache',
                        'inference_time_ms': (end_time - start_time) * 1000
                    }
            
            # Batch processing
            if self.batch_processor and use_batch:
                request_id = f"req_{self.total_requests}_{int(time.time() * 1000)}"
                future = self.batch_processor.submit_request(data, request_id)
                
                try:
                    result = future.result(timeout=1.0)  # 1 second timeout
                    end_time = time.time()
                    
                    # Cache the result
                    if self.cache:
                        self.cache.put(data, result)
                    
                    self._record_performance_metrics(
                        inference_time_ms=(end_time - start_time) * 1000,
                        cache_hit=False,
                        batch_used=True
                    )
                    
                    return {
                        'success': True,
                        'result': result,
                        'source': 'batch_processing',
                        'inference_time_ms': (end_time - start_time) * 1000
                    }
                    
                except Exception as e:
                    logger.warning(f"Batch processing failed, falling back to direct inference: {e}")
            
            # Direct inference
            result = self._direct_inference(data)
            end_time = time.time()
            
            # Cache the result
            if self.cache:
                self.cache.put(data, result)
            
            self._record_performance_metrics(
                inference_time_ms=(end_time - start_time) * 1000,
                cache_hit=False,
                batch_used=False
            )
            
            return {
                'success': True,
                'result': result,
                'source': 'direct_inference',
                'inference_time_ms': (end_time - start_time) * 1000
            }
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Optimized prediction failed: {e}")
            
            self._record_performance_metrics(
                inference_time_ms=(end_time - start_time) * 1000,
                cache_hit=False,
                batch_used=False,
                error=True
            )
            
            return {
                'success': False,
                'error': str(e),
                'inference_time_ms': (end_time - start_time) * 1000
            }
    
    def _direct_inference(self, data: torch.Tensor) -> Any:
        """Direct inference without batching."""
        self.optimized_model.eval()
        with torch.no_grad():
            if hasattr(self.optimized_model, 'compute_reconstruction_error'):
                return self.optimized_model.compute_reconstruction_error(data, reduction='mean').item()
            elif hasattr(self.optimized_model, 'compute_hybrid_anomaly_score'):
                return self.optimized_model.compute_hybrid_anomaly_score(data, reduction='mean').item()
            else:
                reconstruction = self.optimized_model(data)
                error = torch.mean((data - reconstruction) ** 2)
                return error.item()
    
    def _record_performance_metrics(self, inference_time_ms: float, cache_hit: bool, 
                                   batch_used: bool, error: bool = False):
        """Record performance metrics."""
        # Get current resource usage
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_info.used / (1024 * 1024),
            cpu_utilization=cpu_percent,
            cache_hit_rate=1.0 if cache_hit else 0.0,
            throughput_samples_per_sec=1000.0 / max(inference_time_ms, 0.001),
            model_size_mb=self.optimization_report['optimized_size_mb']
        )
        
        self.performance_metrics.append(metrics)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.performance_metrics:
            return {'status': 'no_data'}
        
        # Aggregate metrics
        recent_metrics = list(self.performance_metrics)[-100:]  # Last 100 requests
        
        avg_inference_time = np.mean([m.inference_time_ms for m in recent_metrics])
        p95_inference_time = np.percentile([m.inference_time_ms for m in recent_metrics], 95)
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cpu_usage = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_samples_per_sec for m in recent_metrics])
        
        report = {
            'total_requests': self.total_requests,
            'performance_metrics': {
                'avg_inference_time_ms': avg_inference_time,
                'p95_inference_time_ms': p95_inference_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_cpu_usage_percent': avg_cpu_usage,
                'avg_throughput_samples_per_sec': avg_throughput,
                'model_size_mb': self.optimization_report['optimized_size_mb']
            },
            'optimization_report': self.optimization_report,
            'cache_stats': self.cache.get_stats() if self.cache else None,
            'batch_stats': self.batch_processor.batch_stats if self.batch_processor else None,
            'resource_status': self.resource_monitor.get_resource_status(),
            'optimization_stats': self.optimization_stats
        }
        
        # Performance targets analysis
        target_met = {
            'inference_time': avg_inference_time <= self.config.target_inference_time_ms,
            'memory_usage': avg_memory_usage <= self.config.target_memory_mb,
            'overall_performance': (
                avg_inference_time <= self.config.target_inference_time_ms and
                avg_memory_usage <= self.config.target_memory_mb
            )
        }
        
        report['targets_met'] = target_met
        
        return report
    
    def shutdown(self):
        """Graceful shutdown of all services."""
        logger.info("Shutting down optimized inference engine")
        
        # Stop services
        if self.batch_processor:
            self.batch_processor.stop_processing()
        
        self.resource_monitor.stop_monitoring()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Optimized inference engine shutdown complete")
    
    def export_performance_report(self, output_path: str) -> bool:
        """Export detailed performance report."""
        try:
            report = self.get_performance_report()
            
            # Add detailed metrics history
            report['detailed_metrics'] = [
                m.to_dict() for m in list(self.performance_metrics)[-1000:]
            ]
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            return False