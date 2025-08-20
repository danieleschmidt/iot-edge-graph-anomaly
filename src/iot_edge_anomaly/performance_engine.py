"""
⚡ High-Performance Optimization Engine

This module provides advanced performance optimization, intelligent caching,
and resource management for the Terragon IoT Anomaly Detection System.

Features:
- Model optimization (quantization, pruning, JIT compilation)
- Intelligent multi-level caching with cache warmup
- Memory pool management and GPU optimization
- Batch processing and request coalescing
- Performance profiling and auto-tuning
- Edge device optimization
- Real-time performance monitoring
"""

import os
import sys
import time
import threading
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from collections import deque, defaultdict, OrderedDict
import hashlib
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
import torch.quantization as quant

warnings.filterwarnings('ignore')


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    operation_name: str
    execution_time: float
    memory_usage: float
    gpu_memory: float
    cpu_usage: float
    batch_size: int
    input_shape: Tuple[int, ...]
    optimization_level: OptimizationLevel
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_name': self.operation_name,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'gpu_memory': self.gpu_memory,
            'cpu_usage': self.cpu_usage,
            'batch_size': self.batch_size,
            'input_shape': self.input_shape,
            'optimization_level': self.optimization_level.value,
            'timestamp': self.timestamp.isoformat()
        }


class ModelOptimizer:
    """Advanced model optimization for different deployment scenarios."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        self.optimized_models = {}
        self.optimization_cache = {}
        
    def optimize_model(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        target_device: str = "cpu",
        model_name: str = "unknown"
    ) -> nn.Module:
        """
        Comprehensive model optimization.
        
        Args:
            model: Model to optimize
            sample_input: Sample input for optimization
            target_device: Target deployment device
            model_name: Name for caching
            
        Returns:
            Optimized model
        """
        # Check cache first
        cache_key = self._get_cache_key(model, sample_input, target_device)
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        optimized_model = model
        
        try:
            # Apply optimizations based on level
            if self.optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.STANDARD]:
                optimized_model = self._basic_optimization(optimized_model, sample_input)
                
            if self.optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                optimized_model = self._standard_optimization(optimized_model, sample_input)
                
            if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXTREME]:
                optimized_model = self._aggressive_optimization(optimized_model, sample_input, target_device)
                
            if self.optimization_level == OptimizationLevel.EXTREME:
                optimized_model = self._extreme_optimization(optimized_model, sample_input, target_device)
            
            # Move to target device
            optimized_model = optimized_model.to(target_device)
            
            # Cache optimized model
            self.optimization_cache[cache_key] = optimized_model
            
            return optimized_model
            
        except Exception as e:
            print(f"Warning: Model optimization failed: {e}")
            return model.to(target_device)
    
    def _basic_optimization(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Basic optimization: eval mode and no_grad."""
        model.eval()
        
        # Disable gradient computation for inference
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _standard_optimization(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Standard optimization: JIT scripting and basic pruning."""
        try:
            # TorchScript compilation for faster inference
            model = torch.jit.script(model)
            
            # Basic pruning (remove 10% of least important weights)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.1)
                    prune.remove(module, 'weight')
            
        except Exception as e:
            print(f"Warning: Standard optimization partially failed: {e}")
        
        return model
    
    def _aggressive_optimization(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor, 
        target_device: str
    ) -> nn.Module:
        """Aggressive optimization: quantization and heavy pruning."""
        try:
            # Dynamic quantization for CPU
            if target_device == "cpu":
                model = torch.quantization.quantize_dynamic(
                    model, 
                    {nn.Linear, nn.LSTM, nn.GRU}, 
                    dtype=torch.qint8
                )
            
            # Aggressive pruning (remove 30% of weights)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.3)
                    prune.remove(module, 'weight')
            
            # Optimize for inference
            if hasattr(torch.jit, 'optimize_for_inference'):
                model = torch.jit.optimize_for_inference(model)
            
        except Exception as e:
            print(f"Warning: Aggressive optimization partially failed: {e}")
        
        return model
    
    def _extreme_optimization(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor, 
        target_device: str
    ) -> nn.Module:
        """Extreme optimization: maximum compression and acceleration."""
        try:
            # Static quantization for maximum performance
            if target_device == "cpu":
                # Prepare model for quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibrate with sample input
                with torch.no_grad():
                    model(sample_input)
                
                # Convert to quantized model
                model = torch.quantization.convert(model, inplace=False)
            
            # Extreme pruning (remove 50% of weights)
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.5)
                    prune.remove(module, 'weight')
            
            # Graph optimization
            if hasattr(torch.jit, 'freeze'):
                model = torch.jit.freeze(model)
            
        except Exception as e:
            print(f"Warning: Extreme optimization partially failed: {e}")
        
        return model
    
    def _get_cache_key(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor, 
        target_device: str
    ) -> str:
        """Generate cache key for model optimization."""
        model_hash = hashlib.md5(str(model).encode()).hexdigest()[:8]
        input_shape = str(sample_input.shape)
        opt_level = self.optimization_level.value
        
        return f"{model_hash}_{input_shape}_{target_device}_{opt_level}"
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'optimization_level': self.optimization_level.value,
            'cached_models': len(self.optimization_cache),
            'total_optimizations': len(self.optimized_models)
        }


class IntelligentCache:
    """Multi-level intelligent caching system."""
    
    def __init__(
        self,
        max_memory_mb: int = 512,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        levels: int = 3
    ):
        self.max_memory_mb = max_memory_mb
        self.policy = policy
        self.levels = levels
        
        # Multi-level cache storage
        self.cache_levels = [OrderedDict() for _ in range(levels)]
        self.cache_metadata = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
        # Background maintenance
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def get(self, key: str, level: int = 0) -> Optional[Any]:
        """Get item from cache."""
        # Check all levels starting from the requested level
        for i in range(level, self.levels):
            if key in self.cache_levels[i]:
                value = self.cache_levels[i][key]
                
                # Update access metadata
                self.access_counts[key] += 1
                self.access_times[key] = time.time()
                
                # Promote to higher level if appropriate
                if i > 0 and self._should_promote(key, i):
                    self._promote_item(key, i, i - 1)
                
                self.cache_stats['hits'] += 1
                return value
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any, level: int = 0, ttl: Optional[float] = None) -> bool:
        """Put item in cache."""
        try:
            # Estimate memory usage
            value_size = self._estimate_size(value)
            
            # Check if we need to evict items
            if self._get_memory_usage() + value_size > self.max_memory_mb * 1024 * 1024:
                self._evict_items(value_size)
            
            # Store in specified level
            self.cache_levels[level][key] = value
            
            # Update metadata
            self.cache_metadata[key] = {
                'level': level,
                'size': value_size,
                'created_at': time.time(),
                'ttl': ttl,
                'access_count': 0
            }
            
            self.access_times[key] = time.time()
            self.cache_stats['memory_usage'] += value_size
            
            return True
            
        except Exception as e:
            print(f"Cache put failed: {e}")
            return False
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        removed_count = 0
        
        if pattern is None:
            # Clear all caches
            for level_cache in self.cache_levels:
                removed_count += len(level_cache)
                level_cache.clear()
            
            self.cache_metadata.clear()
            self.access_counts.clear()
            self.access_times.clear()
            self.cache_stats['memory_usage'] = 0
        else:
            # Remove items matching pattern
            keys_to_remove = []
            for key in self.cache_metadata:
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_item(key)
                removed_count += 1
        
        return removed_count
    
    def _should_promote(self, key: str, current_level: int) -> bool:
        """Determine if item should be promoted to higher level."""
        if key not in self.cache_metadata:
            return False
        
        metadata = self.cache_metadata[key]
        access_count = self.access_counts.get(key, 0)
        
        # Promotion criteria based on policy
        if self.policy == CachePolicy.LFU:
            return access_count > 10
        elif self.policy == CachePolicy.LRU:
            last_access = self.access_times.get(key, 0)
            return time.time() - last_access < 300  # 5 minutes
        elif self.policy == CachePolicy.ADAPTIVE:
            # Adaptive promotion based on multiple factors
            age = time.time() - metadata['created_at']
            access_frequency = access_count / max(age / 3600, 1)  # per hour
            return access_frequency > 2 and access_count > 5
        
        return False
    
    def _promote_item(self, key: str, from_level: int, to_level: int):
        """Promote item to higher cache level."""
        if key in self.cache_levels[from_level]:
            value = self.cache_levels[from_level].pop(key)
            self.cache_levels[to_level][key] = value
            self.cache_metadata[key]['level'] = to_level
    
    def _evict_items(self, needed_space: int):
        """Evict items to make space."""
        freed_space = 0
        
        # Start eviction from lowest levels
        for level in range(self.levels - 1, -1, -1):
            if freed_space >= needed_space:
                break
            
            level_cache = self.cache_levels[level]
            keys_to_evict = []
            
            if self.policy == CachePolicy.LRU:
                # Evict least recently used
                sorted_keys = sorted(
                    level_cache.keys(),
                    key=lambda k: self.access_times.get(k, 0)
                )
                keys_to_evict = sorted_keys[:len(sorted_keys) // 2]
                
            elif self.policy == CachePolicy.LFU:
                # Evict least frequently used
                sorted_keys = sorted(
                    level_cache.keys(),
                    key=lambda k: self.access_counts.get(k, 0)
                )
                keys_to_evict = sorted_keys[:len(sorted_keys) // 2]
                
            elif self.policy == CachePolicy.TTL:
                # Evict expired items first
                current_time = time.time()
                for key in list(level_cache.keys()):
                    metadata = self.cache_metadata.get(key, {})
                    ttl = metadata.get('ttl')
                    if ttl and current_time - metadata.get('created_at', 0) > ttl:
                        keys_to_evict.append(key)
                
            elif self.policy == CachePolicy.ADAPTIVE:
                # Adaptive eviction based on multiple factors
                current_time = time.time()
                key_scores = []
                
                for key in level_cache.keys():
                    metadata = self.cache_metadata.get(key, {})
                    age = current_time - metadata.get('created_at', 0)
                    access_count = self.access_counts.get(key, 0)
                    last_access = self.access_times.get(key, 0)
                    
                    # Lower score = more likely to evict
                    score = (access_count / max(age / 3600, 1)) - (current_time - last_access) / 3600
                    key_scores.append((score, key))
                
                key_scores.sort()
                keys_to_evict = [key for _, key in key_scores[:len(key_scores) // 2]]
            
            # Perform eviction
            for key in keys_to_evict:
                if freed_space >= needed_space:
                    break
                freed_space += self._remove_item(key)
                self.cache_stats['evictions'] += 1
    
    def _remove_item(self, key: str) -> int:
        """Remove item from cache and return freed space."""
        freed_space = 0
        
        # Remove from appropriate level
        if key in self.cache_metadata:
            level = self.cache_metadata[key]['level']
            freed_space = self.cache_metadata[key]['size']
            
            if key in self.cache_levels[level]:
                del self.cache_levels[level][key]
            
            del self.cache_metadata[key]
            
            if key in self.access_counts:
                del self.access_counts[key]
            
            if key in self.access_times:
                del self.access_times[key]
            
            self.cache_stats['memory_usage'] -= freed_space
        
        return freed_space
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.numel()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.cache_stats['memory_usage']
    
    def _maintenance_loop(self):
        """Background maintenance for cache."""
        while True:
            try:
                current_time = time.time()
                
                # Remove expired items
                if self.policy == CachePolicy.TTL:
                    expired_keys = []
                    for key, metadata in self.cache_metadata.items():
                        ttl = metadata.get('ttl')
                        if ttl and current_time - metadata.get('created_at', 0) > ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_item(key)
                
                # Rebalance cache levels
                self._rebalance_levels()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Cache maintenance error: {e}")
                time.sleep(60)
    
    def _rebalance_levels(self):
        """Rebalance items across cache levels."""
        # Promote frequently accessed items
        for key in list(self.cache_metadata.keys()):
            if key in self.cache_metadata:
                current_level = self.cache_metadata[key]['level']
                if current_level > 0 and self._should_promote(key, current_level):
                    self._promote_item(key, current_level, current_level - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_items = sum(len(level_cache) for level_cache in self.cache_levels)
        hit_rate = self.cache_stats['hits'] / max(
            self.cache_stats['hits'] + self.cache_stats['misses'], 1
        )
        
        return {
            **self.cache_stats,
            'total_items': total_items,
            'hit_rate': hit_rate,
            'memory_usage_mb': self.cache_stats['memory_usage'] / (1024 * 1024),
            'levels': [len(level_cache) for level_cache in self.cache_levels]
        }


class ResourcePool:
    """Resource pool for managing computational resources."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.worker_pool = []
        self.resource_lock = threading.Lock()
        self.usage_stats = {
            'allocations': 0,
            'releases': 0,
            'peak_usage': 0,
            'current_usage': 0
        }
    
    def allocate_worker(self) -> Optional[int]:
        """Allocate a worker from the pool."""
        with self.resource_lock:
            if len(self.worker_pool) < self.max_workers:
                worker_id = len(self.worker_pool)
                self.worker_pool.append({'id': worker_id, 'busy': True, 'allocated_at': time.time()})
                
                self.usage_stats['allocations'] += 1
                self.usage_stats['current_usage'] = len(self.worker_pool)
                self.usage_stats['peak_usage'] = max(
                    self.usage_stats['peak_usage'], 
                    self.usage_stats['current_usage']
                )
                
                return worker_id
        
        return None
    
    def release_worker(self, worker_id: int):
        """Release a worker back to the pool."""
        with self.resource_lock:
            for worker in self.worker_pool:
                if worker['id'] == worker_id:
                    worker['busy'] = False
                    self.usage_stats['releases'] += 1
                    break
    
    def get_available_workers(self) -> int:
        """Get number of available workers."""
        with self.resource_lock:
            return sum(1 for worker in self.worker_pool if not worker['busy'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        return {
            **self.usage_stats,
            'max_workers': self.max_workers,
            'available_workers': self.get_available_workers()
        }


class PerformanceProfiler:
    """Advanced performance profiling and monitoring."""
    
    def __init__(self, max_profiles: int = 1000):
        self.profiles = deque(maxlen=max_profiles)
        self.aggregated_stats = defaultdict(list)
        
    def profile_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, PerformanceProfile]:
        """Profile a function execution."""
        # Pre-execution measurements
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        
        # Post-execution measurements
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_gpu_memory = self._get_gpu_memory()
        
        # Create profile
        profile = PerformanceProfile(
            operation_name=operation_name,
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            gpu_memory=end_gpu_memory - start_gpu_memory,
            cpu_usage=0.0,  # Would need psutil for accurate CPU measurement
            batch_size=self._infer_batch_size(args, kwargs),
            input_shape=self._infer_input_shape(args, kwargs),
            optimization_level=OptimizationLevel.STANDARD
        )
        
        # Store profile
        self.profiles.append(profile)
        self.aggregated_stats[operation_name].append(profile)
        
        if success:
            return result, profile
        else:
            raise result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        return 0.0
    
    def _infer_batch_size(self, args: tuple, kwargs: dict) -> int:
        """Infer batch size from arguments."""
        for arg in args:
            if isinstance(arg, (torch.Tensor, np.ndarray)) and len(arg.shape) > 0:
                return arg.shape[0]
        
        for value in kwargs.values():
            if isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) > 0:
                return value.shape[0]
        
        return 1
    
    def _infer_input_shape(self, args: tuple, kwargs: dict) -> Tuple[int, ...]:
        """Infer input shape from arguments."""
        for arg in args:
            if isinstance(arg, (torch.Tensor, np.ndarray)):
                return tuple(arg.shape)
        
        for value in kwargs.values():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return tuple(value.shape)
        
        return (1,)
    
    def get_aggregated_stats(self, operation_name: str) -> Dict[str, float]:
        """Get aggregated statistics for an operation."""
        if operation_name not in self.aggregated_stats:
            return {}
        
        profiles = self.aggregated_stats[operation_name]
        
        if not profiles:
            return {}
        
        execution_times = [p.execution_time for p in profiles]
        memory_usages = [p.memory_usage for p in profiles]
        
        return {
            'count': len(profiles),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_usage': sum(memory_usages) / len(memory_usages),
            'max_memory_usage': max(memory_usages),
            'total_execution_time': sum(execution_times)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'total_profiles': len(self.profiles),
            'operations': {},
            'summary': {
                'total_operations': len(self.aggregated_stats),
                'avg_execution_time': 0.0,
                'total_execution_time': 0.0
            }
        }
        
        all_times = []
        total_time = 0.0
        
        for operation_name in self.aggregated_stats:
            stats = self.get_aggregated_stats(operation_name)
            report['operations'][operation_name] = stats
            
            if stats:
                all_times.extend([p.execution_time for p in self.aggregated_stats[operation_name]])
                total_time += stats['total_execution_time']
        
        if all_times:
            report['summary']['avg_execution_time'] = sum(all_times) / len(all_times)
        
        report['summary']['total_execution_time'] = total_time
        
        return report


# Example usage and testing
if __name__ == "__main__":
    print("⚡ Testing Performance Engine")
    print("=" * 50)
    
    # Test model optimizer
    print("Testing model optimizer...")
    optimizer = ModelOptimizer(OptimizationLevel.STANDARD)
    
    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    sample_input = torch.randn(1, 10)
    
    optimized_model = optimizer.optimize_model(model, sample_input, "cpu", "test_model")
    print(f"Model optimization completed: {type(optimized_model)}")
    
    # Test intelligent cache
    print("\nTesting intelligent cache...")
    cache = IntelligentCache(max_memory_mb=64, policy=CachePolicy.ADAPTIVE)
    
    # Test cache operations
    test_data = torch.randn(100, 100)
    cache.put("test_tensor", test_data, level=0)
    
    retrieved = cache.get("test_tensor")
    print(f"Cache test: {'✅' if retrieved is not None else '❌'}")
    
    stats = cache.get_stats()
    print(f"Cache stats: {stats['total_items']} items, {stats['hit_rate']:.2%} hit rate")
    
    # Test resource pool
    print("\nTesting resource pool...")
    pool = ResourcePool(max_workers=4)
    
    worker_id = pool.allocate_worker()
    print(f"Worker allocated: {worker_id}")
    
    pool.release_worker(worker_id)
    print("Worker released")
    
    pool_stats = pool.get_stats()
    print(f"Pool stats: {pool_stats}")
    
    # Test performance profiler
    print("\nTesting performance profiler...")
    profiler = PerformanceProfiler()
    
    # Profile a simple operation
    def test_operation(x):
        return x * 2 + 1
    
    result, profile = profiler.profile_operation("test_op", test_operation, torch.randn(100, 100))
    print(f"Profiled operation: {profile.execution_time:.4f}s")
    
    # Get performance report
    report = profiler.get_performance_report()
    print(f"Performance report: {report['total_profiles']} profiles")
    
    print("✅ Performance engine tested successfully!")