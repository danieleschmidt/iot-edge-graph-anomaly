"""
Advanced Performance Optimization for IoT Edge Anomaly Detection.

This module provides comprehensive performance optimization including:
- Intelligent caching systems with adaptive policies
- Concurrent processing with resource pooling
- Load balancing and auto-scaling triggers
- Memory optimization and GPU acceleration
- Model optimization and quantization
"""
import asyncio
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, PriorityQueue
import multiprocessing
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict, OrderedDict
import pickle
import hashlib
import weakref

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class WorkloadType(Enum):
    """Types of computational workloads."""
    INFERENCE = "inference"
    TRAINING = "training"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    GPU_BOUND = "gpu_bound"


@dataclass
class CacheItem:
    """Represents an item in the cache."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache item has expired."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl
    
    def access(self) -> None:
        """Record cache access."""
        self.access_count += 1
        self.last_access = datetime.now()


@dataclass
class WorkloadRequest:
    """Represents a computational workload request."""
    request_id: str
    workload_type: WorkloadType
    priority: int  # Lower number = higher priority
    task: Callable
    args: tuple
    kwargs: dict
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """For priority queue sorting."""
        return self.priority < other.priority


class AdaptiveCache:
    """
    Adaptive caching system with intelligent replacement policies.
    
    Features:
    - Multiple cache policies (LRU, LFU, TTL, Adaptive)
    - Memory-aware cache sizing
    - Access pattern analysis
    - Automatic policy switching
    """
    
    def __init__(self, max_size_mb: float = 100.0, policy: CachePolicy = CachePolicy.ADAPTIVE):
        """
        Initialize adaptive cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            policy: Cache replacement policy
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.policy = policy
        self.cache: Dict[str, CacheItem] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.current_size_bytes = 0
        
        # Adaptive policy parameters
        self.hit_rates = {policy: [] for policy in CachePolicy}
        self.evaluation_window = 100  # Number of accesses per evaluation
        self.access_count = 0
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_accesses": 0
        }
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            self.stats["total_accesses"] += 1
            
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if item.is_expired():
                self._remove_item(key)
                self.stats["misses"] += 1
                return None
            
            # Update access patterns
            item.access()
            self.access_order.move_to_end(key)
            self.access_frequency[key] += 1
            
            self.stats["hits"] += 1
            
            # Adaptive policy evaluation
            if self.policy == CachePolicy.ADAPTIVE:
                self._update_adaptive_policy()
            
            return item.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate item size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default size if can't serialize
            
            # Check if item is too large for cache
            if size_bytes > self.max_size_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing item if present
            if key in self.cache:
                self._remove_item(key)
            
            # Make space if needed
            while (self.current_size_bytes + size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_item()
            
            # Add new item
            item = CacheItem(
                key=key,
                value=value,
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self.cache[key] = item
            self.access_order[key] = True
            self.current_size_bytes += size_bytes
            
            return True
    
    def _remove_item(self, key: str) -> None:
        """Remove item from cache."""
        if key in self.cache:
            self.current_size_bytes -= self.cache[key].size_bytes
            del self.cache[key]
            self.access_order.pop(key, None)
            self.access_frequency.pop(key, None)
    
    def _evict_item(self) -> None:
        """Evict item based on current policy."""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU or self.policy == CachePolicy.ADAPTIVE:
            # Remove least recently used
            key = next(iter(self.access_order))
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.access_frequency.keys(), key=lambda k: self.access_frequency[k])
        elif self.policy == CachePolicy.TTL:
            # Remove expired items first, then oldest
            expired_keys = [k for k, item in self.cache.items() if item.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        else:
            key = next(iter(self.cache))
        
        self._remove_item(key)
        self.stats["evictions"] += 1
    
    def _update_adaptive_policy(self) -> None:
        """Update adaptive policy based on performance."""
        self.access_count += 1
        
        if self.access_count % self.evaluation_window == 0:
            current_hit_rate = self.stats["hits"] / max(1, self.stats["total_accesses"])
            
            # Evaluate different policies
            for policy in [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.TTL]:
                self.hit_rates[policy].append(current_hit_rate)
                
                # Keep only recent history
                if len(self.hit_rates[policy]) > 10:
                    self.hit_rates[policy].pop(0)
            
            # Switch to best performing policy
            if len(self.hit_rates[CachePolicy.LRU]) >= 3:
                avg_rates = {
                    policy: np.mean(rates) 
                    for policy, rates in self.hit_rates.items() 
                    if rates
                }
                
                best_policy = max(avg_rates.keys(), key=lambda p: avg_rates[p])
                if best_policy != self.policy:
                    logger.info(f"Switching cache policy from {self.policy.value} to {best_policy.value}")
                    self.policy = best_policy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = self.stats["hits"] / max(1, self.stats["total_accesses"])
            
            return {
                "policy": self.policy.value,
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.current_size_bytes / max(1, self.max_size_bytes),
                "item_count": len(self.cache),
                "hit_rate": hit_rate,
                "total_accesses": self.stats["total_accesses"],
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"]
            }
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.current_size_bytes = 0


class ResourcePool:
    """
    Resource pool for managing computational resources.
    
    Features:
    - GPU memory pooling
    - Thread pool management
    - Process pool optimization
    - Dynamic resource allocation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resource pool."""
        self.config = config or {}
        
        # Thread pool configuration
        max_workers = self.config.get("max_workers", min(32, (psutil.cpu_count() or 1) + 4))
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Process pool for CPU-intensive tasks
        max_processes = self.config.get("max_processes", psutil.cpu_count() or 1)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        
        # GPU memory pool
        self.gpu_memory_pool = {}
        if torch.cuda.is_available():
            self._initialize_gpu_memory_pool()
        
        # Resource allocation tracking
        self.active_tasks = {}
        self.resource_usage = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_memory_percent": 0.0
        }
        
        # Performance metrics
        self.task_metrics = defaultdict(list)
        
        self._lock = threading.RLock()
    
    def _initialize_gpu_memory_pool(self) -> None:
        """Initialize GPU memory pool."""
        try:
            device_count = torch.cuda.device_count()
            
            for device_id in range(device_count):
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                # Reserve 80% for pooling, 20% for system
                pool_size = int(total_memory * 0.8)
                
                self.gpu_memory_pool[device_id] = {
                    "total_memory": total_memory,
                    "pool_size": pool_size,
                    "allocated": 0,
                    "cached_tensors": {}
                }
                
                logger.info(f"GPU {device_id} memory pool initialized: {pool_size / 1024**3:.2f} GB")
                
        except Exception as e:
            logger.warning(f"Failed to initialize GPU memory pool: {e}")
    
    async def submit_task(self, workload: WorkloadRequest) -> Any:
        """Submit task to appropriate resource pool."""
        with self._lock:
            task_id = workload.request_id
            self.active_tasks[task_id] = {
                "workload": workload,
                "start_time": time.time(),
                "resource_type": self._determine_resource_type(workload)
            }
        
        try:
            # Determine optimal execution strategy
            if workload.workload_type in [WorkloadType.CPU_BOUND, WorkloadType.PREPROCESSING]:
                result = await self._execute_in_process_pool(workload)
            elif workload.workload_type in [WorkloadType.IO_BOUND]:
                result = await self._execute_in_thread_pool(workload)
            elif workload.workload_type in [WorkloadType.GPU_BOUND, WorkloadType.INFERENCE]:
                result = await self._execute_with_gpu_optimization(workload)
            else:
                result = await self._execute_in_thread_pool(workload)
            
            # Record metrics
            self._record_task_completion(task_id, success=True)
            
            return result
            
        except Exception as e:
            self._record_task_completion(task_id, success=False, error=str(e))
            raise
        finally:
            with self._lock:
                self.active_tasks.pop(task_id, None)
    
    def _determine_resource_type(self, workload: WorkloadRequest) -> str:
        """Determine optimal resource type for workload."""
        if workload.workload_type == WorkloadType.GPU_BOUND:
            return "gpu"
        elif workload.workload_type == WorkloadType.CPU_BOUND:
            return "process"
        else:
            return "thread"
    
    async def _execute_in_thread_pool(self, workload: WorkloadRequest) -> Any:
        """Execute task in thread pool."""
        loop = asyncio.get_event_loop()
        
        future = self.thread_pool.submit(
            self._safe_task_execution,
            workload.task,
            *workload.args,
            **workload.kwargs
        )
        
        if workload.timeout:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=workload.timeout
                )
            except asyncio.TimeoutError:
                future.cancel()
                raise TimeoutError(f"Task {workload.request_id} timed out")
        else:
            result = await loop.run_in_executor(None, future.result)
        
        return result
    
    async def _execute_in_process_pool(self, workload: WorkloadRequest) -> Any:
        """Execute task in process pool."""
        loop = asyncio.get_event_loop()
        
        future = self.process_pool.submit(
            self._safe_task_execution,
            workload.task,
            *workload.args,
            **workload.kwargs
        )
        
        if workload.timeout:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result),
                    timeout=workload.timeout
                )
            except asyncio.TimeoutError:
                future.cancel()
                raise TimeoutError(f"Task {workload.request_id} timed out")
        else:
            result = await loop.run_in_executor(None, future.result)
        
        return result
    
    async def _execute_with_gpu_optimization(self, workload: WorkloadRequest) -> Any:
        """Execute task with GPU optimization."""
        # Get optimal GPU device
        device_id = self._get_optimal_gpu_device()
        
        if device_id is not None:
            # Execute with GPU acceleration
            return await self._execute_gpu_task(workload, device_id)
        else:
            # Fallback to CPU execution
            logger.warning("No GPU available, falling back to CPU")
            return await self._execute_in_thread_pool(workload)
    
    def _get_optimal_gpu_device(self) -> Optional[int]:
        """Get optimal GPU device based on memory availability."""
        if not torch.cuda.is_available():
            return None
        
        device_count = torch.cuda.device_count()
        best_device = None
        max_free_memory = 0
        
        for device_id in range(device_count):
            try:
                torch.cuda.set_device(device_id)
                free_memory = torch.cuda.get_device_properties(device_id).total_memory
                free_memory -= torch.cuda.memory_allocated(device_id)
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = device_id
                    
            except Exception as e:
                logger.warning(f"Error checking GPU {device_id}: {e}")
        
        return best_device
    
    async def _execute_gpu_task(self, workload: WorkloadRequest, device_id: int) -> Any:
        """Execute task on specific GPU device."""
        original_device = None
        
        try:
            if torch.cuda.is_available():
                original_device = torch.cuda.current_device()
                torch.cuda.set_device(device_id)
            
            # Execute task
            result = await self._execute_in_thread_pool(workload)
            
            return result
            
        finally:
            # Restore original device
            if original_device is not None:
                torch.cuda.set_device(original_device)
    
    def _safe_task_execution(self, task: Callable, *args, **kwargs) -> Any:
        """Safely execute task with error handling."""
        try:
            return task(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    def _record_task_completion(self, task_id: str, success: bool, error: Optional[str] = None) -> None:
        """Record task completion metrics."""
        with self._lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                execution_time = time.time() - task_info["start_time"]
                
                metric_key = f"{task_info['workload'].workload_type.value}_{task_info['resource_type']}"
                
                self.task_metrics[metric_key].append({
                    "execution_time": execution_time,
                    "success": success,
                    "error": error,
                    "timestamp": time.time()
                })
                
                # Keep only recent metrics
                if len(self.task_metrics[metric_key]) > 1000:
                    self.task_metrics[metric_key] = self.task_metrics[metric_key][-1000:]
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        with self._lock:
            # System resource usage
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            stats = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "active_tasks": len(self.active_tasks),
                "thread_pool_active": self.thread_pool._threads,
                "process_pool_active": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
            }
            
            # GPU stats
            if torch.cuda.is_available():
                gpu_stats = {}
                for device_id in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(device_id)
                        cached = torch.cuda.memory_reserved(device_id)
                        total = torch.cuda.get_device_properties(device_id).total_memory
                        
                        gpu_stats[f"gpu_{device_id}"] = {
                            "allocated_mb": allocated / 1024**2,
                            "cached_mb": cached / 1024**2,
                            "total_mb": total / 1024**2,
                            "utilization": allocated / total
                        }
                    except Exception as e:
                        logger.warning(f"Error getting GPU {device_id} stats: {e}")
                
                stats["gpu"] = gpu_stats
            
            # Task performance metrics
            performance_stats = {}
            for metric_key, metrics in self.task_metrics.items():
                if metrics:
                    recent_metrics = [m for m in metrics if time.time() - m["timestamp"] < 300]  # Last 5 minutes
                    if recent_metrics:
                        execution_times = [m["execution_time"] for m in recent_metrics]
                        success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
                        
                        performance_stats[metric_key] = {
                            "avg_execution_time": np.mean(execution_times),
                            "p95_execution_time": np.percentile(execution_times, 95),
                            "success_rate": success_rate,
                            "task_count": len(recent_metrics)
                        }
            
            stats["performance"] = performance_stats
            
            return stats
    
    def cleanup(self) -> None:
        """Clean up resource pools."""
        logger.info("Cleaning up resource pools...")
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LoadBalancer:
    """
    Load balancer for distributing workloads across multiple workers.
    
    Features:
    - Round-robin and weighted round-robin
    - Least connections algorithm
    - Health-based routing
    - Auto-scaling triggers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize load balancer."""
        self.config = config or {}
        self.workers: List[Dict[str, Any]] = []
        self.current_worker = 0
        self.worker_stats = defaultdict(lambda: {
            "active_connections": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "health_score": 1.0,
            "last_health_check": datetime.now()
        })
        
        # Load balancing algorithm
        self.algorithm = self.config.get("algorithm", "round_robin")
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = self.config.get("auto_scaling_enabled", True)
        self.min_workers = self.config.get("min_workers", 1)
        self.max_workers = self.config.get("max_workers", 10)
        self.scale_up_threshold = self.config.get("scale_up_threshold", 0.8)
        self.scale_down_threshold = self.config.get("scale_down_threshold", 0.3)
        
        self._lock = threading.RLock()
    
    def add_worker(self, worker_id: str, worker_instance: Any, weight: float = 1.0) -> None:
        """Add worker to the pool."""
        with self._lock:
            worker_info = {
                "id": worker_id,
                "instance": worker_instance,
                "weight": weight,
                "enabled": True,
                "added_time": datetime.now()
            }
            
            self.workers.append(worker_info)
            logger.info(f"Added worker {worker_id} with weight {weight}")
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from the pool."""
        with self._lock:
            for i, worker in enumerate(self.workers):
                if worker["id"] == worker_id:
                    self.workers.pop(i)
                    self.worker_stats.pop(worker_id, None)
                    logger.info(f"Removed worker {worker_id}")
                    return True
            return False
    
    def get_next_worker(self) -> Optional[Dict[str, Any]]:
        """Get next worker based on load balancing algorithm."""
        with self._lock:
            if not self.workers:
                return None
            
            # Filter healthy workers
            healthy_workers = [w for w in self.workers if w["enabled"] and self._is_worker_healthy(w["id"])]
            
            if not healthy_workers:
                logger.warning("No healthy workers available")
                return None
            
            if self.algorithm == "round_robin":
                worker = self._round_robin_selection(healthy_workers)
            elif self.algorithm == "weighted_round_robin":
                worker = self._weighted_round_robin_selection(healthy_workers)
            elif self.algorithm == "least_connections":
                worker = self._least_connections_selection(healthy_workers)
            else:
                # Default to round robin
                worker = self._round_robin_selection(healthy_workers)
            
            # Update connection count
            if worker:
                self.worker_stats[worker["id"]]["active_connections"] += 1
            
            return worker
    
    def _round_robin_selection(self, workers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round-robin worker selection."""
        worker = workers[self.current_worker % len(workers)]
        self.current_worker = (self.current_worker + 1) % len(workers)
        return worker
    
    def _weighted_round_robin_selection(self, workers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted round-robin worker selection."""
        # Simple weighted selection based on worker weights
        total_weight = sum(w["weight"] for w in workers)
        
        if total_weight == 0:
            return self._round_robin_selection(workers)
        
        # Select based on cumulative weights
        import random
        random_weight = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for worker in workers:
            cumulative_weight += worker["weight"]
            if random_weight <= cumulative_weight:
                return worker
        
        return workers[-1]  # Fallback
    
    def _least_connections_selection(self, workers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Least connections worker selection."""
        return min(workers, key=lambda w: self.worker_stats[w["id"]]["active_connections"])
    
    def _is_worker_healthy(self, worker_id: str) -> bool:
        """Check if worker is healthy."""
        stats = self.worker_stats[worker_id]
        
        # Health score based on error rate and response time
        error_rate = 0
        if stats["total_requests"] > 0:
            error_rate = stats["failed_requests"] / stats["total_requests"]
        
        # Consider worker unhealthy if error rate > 50% or avg response time > 30s
        return error_rate < 0.5 and stats["avg_response_time"] < 30.0
    
    def record_request_completion(self, worker_id: str, response_time: float, success: bool) -> None:
        """Record request completion for load balancing metrics."""
        with self._lock:
            stats = self.worker_stats[worker_id]
            
            # Update connection count
            stats["active_connections"] = max(0, stats["active_connections"] - 1)
            
            # Update request stats
            stats["total_requests"] += 1
            if not success:
                stats["failed_requests"] += 1
            
            # Update average response time (exponential moving average)
            alpha = 0.1
            stats["avg_response_time"] = (
                alpha * response_time + (1 - alpha) * stats["avg_response_time"]
            )
            
            # Update health score
            error_rate = stats["failed_requests"] / max(1, stats["total_requests"])
            response_time_factor = min(1.0, 10.0 / max(0.1, stats["avg_response_time"]))
            stats["health_score"] = (1 - error_rate) * response_time_factor
    
    def check_auto_scaling(self) -> Optional[str]:
        """Check if auto-scaling action is needed."""
        if not self.auto_scaling_enabled:
            return None
        
        with self._lock:
            if not self.workers:
                return "scale_up" if self.min_workers > 0 else None
            
            # Calculate average utilization
            total_connections = sum(stats["active_connections"] for stats in self.worker_stats.values())
            total_capacity = len(self.workers) * 10  # Assume 10 concurrent connections per worker
            
            if total_capacity > 0:
                utilization = total_connections / total_capacity
                
                # Scale up if utilization is high and we haven't reached max workers
                if utilization > self.scale_up_threshold and len(self.workers) < self.max_workers:
                    return "scale_up"
                
                # Scale down if utilization is low and we have more than min workers
                if utilization < self.scale_down_threshold and len(self.workers) > self.min_workers:
                    return "scale_down"
            
            return None
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            worker_stats = {}
            for worker in self.workers:
                worker_id = worker["id"]
                stats = self.worker_stats[worker_id]
                
                worker_stats[worker_id] = {
                    "enabled": worker["enabled"],
                    "weight": worker["weight"],
                    "active_connections": stats["active_connections"],
                    "total_requests": stats["total_requests"],
                    "error_rate": stats["failed_requests"] / max(1, stats["total_requests"]),
                    "avg_response_time": stats["avg_response_time"],
                    "health_score": stats["health_score"]
                }
            
            total_connections = sum(stats["active_connections"] for stats in self.worker_stats.values())
            
            return {
                "algorithm": self.algorithm,
                "worker_count": len(self.workers),
                "healthy_workers": len([w for w in self.workers if self._is_worker_healthy(w["id"])]),
                "total_active_connections": total_connections,
                "workers": worker_stats,
                "auto_scaling_enabled": self.auto_scaling_enabled
            }


class AdvancedPerformanceOptimizer:
    """
    Advanced performance optimizer integrating all optimization components.
    
    Features:
    - Adaptive caching with intelligent policies
    - Resource pooling and load balancing
    - Auto-scaling and performance monitoring
    - Model optimization and quantization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced performance optimizer."""
        self.config = config or {}
        
        # Initialize components
        self.cache = AdaptiveCache(
            max_size_mb=self.config.get("cache_size_mb", 100.0),
            policy=CachePolicy(self.config.get("cache_policy", "adaptive"))
        )
        
        self.resource_pool = ResourcePool(self.config.get("resource_pool", {}))
        self.load_balancer = LoadBalancer(self.config.get("load_balancer", {}))
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        self.optimization_history = []
        
        # Model optimization cache
        self.optimized_models = {}
        
        logger.info("Advanced Performance Optimizer initialized")
    
    async def optimized_inference(self, model: nn.Module, data: torch.Tensor, 
                                model_id: str = "default") -> torch.Tensor:
        """
        Perform optimized inference with caching and resource pooling.
        
        Args:
            model: PyTorch model
            data: Input data
            model_id: Unique identifier for the model
            
        Returns:
            Inference results
        """
        start_time = time.time()
        
        # Generate cache key
        data_hash = hashlib.md5(data.detach().cpu().numpy().tobytes()).hexdigest()
        cache_key = f"{model_id}_{data_hash}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached_result
        
        # Get optimized model
        optimized_model = await self._get_optimized_model(model, model_id)
        
        # Create inference workload
        workload = WorkloadRequest(
            request_id=f"inference_{int(time.time() * 1000000)}",
            workload_type=WorkloadType.GPU_BOUND if torch.cuda.is_available() else WorkloadType.CPU_BOUND,
            priority=1,
            task=self._inference_task,
            args=(optimized_model, data),
            kwargs={},
            timeout=30.0
        )
        
        # Execute with resource pooling
        result = await self.resource_pool.submit_task(workload)
        
        # Cache result
        self.cache.put(cache_key, result, ttl=300.0)  # 5 minute TTL
        
        # Record performance metrics
        inference_time = time.time() - start_time
        self.performance_metrics["inference_times"].append(inference_time)
        
        return result
    
    def _inference_task(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Execute inference task."""
        with torch.no_grad():
            return model(data)
    
    async def _get_optimized_model(self, model: nn.Module, model_id: str) -> nn.Module:
        """Get optimized version of model."""
        if model_id in self.optimized_models:
            return self.optimized_models[model_id]
        
        # Optimize model
        optimized_model = await self._optimize_model(model, model_id)
        self.optimized_models[model_id] = optimized_model
        
        return optimized_model
    
    async def _optimize_model(self, model: nn.Module, model_id: str) -> nn.Module:
        """Optimize model for inference."""
        logger.info(f"Optimizing model {model_id}")
        
        # Set to eval mode
        model.eval()
        
        # Try torch.jit optimization
        try:
            # Create example input for tracing
            example_input = torch.randn(1, 10, 5)  # Example shape
            traced_model = torch.jit.trace(model, example_input)
            logger.info(f"Model {model_id} optimized with TorchScript")
            return traced_model
        except Exception as e:
            logger.warning(f"TorchScript optimization failed for {model_id}: {e}")
        
        # Try quantization
        try:
            if hasattr(torch.quantization, 'quantize_dynamic'):
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
                )
                logger.info(f"Model {model_id} optimized with quantization")
                return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed for {model_id}: {e}")
        
        # Return original model if optimization fails
        return model
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cache": self.cache.get_stats(),
            "resource_pool": self.resource_pool.get_resource_stats(),
            "load_balancer": self.load_balancer.get_load_balancer_stats(),
            "performance_metrics": {
                "inference_times": {
                    "count": len(self.performance_metrics["inference_times"]),
                    "avg": np.mean(self.performance_metrics["inference_times"]) if self.performance_metrics["inference_times"] else 0,
                    "p95": np.percentile(self.performance_metrics["inference_times"], 95) if self.performance_metrics["inference_times"] else 0,
                    "latest": self.performance_metrics["inference_times"][-1] if self.performance_metrics["inference_times"] else 0
                }
            },
            "optimized_models": len(self.optimized_models)
        }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up optimizer resources."""
        logger.info("Cleaning up performance optimizer...")
        
        self.cache.clear()
        self.resource_pool.cleanup()
        self.optimized_models.clear()


# Global performance optimizer instance
_performance_optimizer: Optional[AdvancedPerformanceOptimizer] = None


def get_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> AdvancedPerformanceOptimizer:
    """Get or create global performance optimizer."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = AdvancedPerformanceOptimizer(config)
    
    return _performance_optimizer


async def optimized_model_inference(model: nn.Module, data: torch.Tensor, 
                                  model_id: str = "default") -> torch.Tensor:
    """
    Convenience function for optimized model inference.
    
    Args:
        model: PyTorch model
        data: Input data
        model_id: Model identifier for caching
        
    Returns:
        Optimized inference results
    """
    optimizer = get_performance_optimizer()
    return await optimizer.optimized_inference(model, data, model_id)