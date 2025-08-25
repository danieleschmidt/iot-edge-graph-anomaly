"""
HyperScale Optimization Engine for IoT Anomaly Detection
Advanced performance optimization, intelligent caching, concurrent processing, and auto-scaling.
"""

import logging
import asyncio
import threading
import multiprocessing
import concurrent.futures
import time
import gc
import weakref
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, OrderedDict, deque
import json
import hashlib
import pickle
import mmap
import psutil
import queue
import heapq
from datetime import datetime, timedelta, timezone
import uuid
import statistics
import numpy as np
from pathlib import Path
import os
import sys
import contextlib

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class ScalingTrigger(Enum):
    """Auto-scaling triggers."""
    CPU_USAGE = auto()
    MEMORY_USAGE = auto()
    QUEUE_DEPTH = auto()
    RESPONSE_TIME = auto()
    ERROR_RATE = auto()
    THROUGHPUT = auto()
    CUSTOM_METRIC = auto()


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on usage patterns


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    key: str
    value: V
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    queue_depth: int
    avg_response_time_ms: float
    requests_per_second: float
    error_rate_percent: float
    cache_hit_rate: float
    active_connections: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Auto-scaling decision with reasoning."""
    action: str  # "scale_up", "scale_down", "maintain"
    current_instances: int
    target_instances: int
    reason: str
    confidence_score: float
    estimated_impact: Dict[str, float]
    cooldown_until: float


class IntelligentCache(Generic[K, V]):
    """High-performance intelligent cache with adaptive strategies."""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[float] = None, max_memory_mb: float = 100):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache: Dict[K, CacheEntry[V]] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_counter = defaultdict(int)  # For LFU
        self._size_tracker = 0
        self._lock = threading.RLock()
        
        # Adaptive strategy learning
        self._access_patterns = deque(maxlen=1000)
        self._hit_rates = deque(maxlen=100)
        self._strategy_performance = {strategy: 0.0 for strategy in CacheStrategy}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"Initialized intelligent cache: max_size={max_size}, strategy={strategy.value}")
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                self._record_access_pattern(key, False)
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # Check TTL expiration
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                self._evict_entry(key)
                self.misses += 1
                self._record_access_pattern(key, False)
                return None
            
            # Update access metadata
            entry.last_accessed = current_time
            entry.access_count += 1
            self._frequency_counter[key] += 1
            
            # Update access order for LRU
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = current_time
            
            self.hits += 1
            self._record_access_pattern(key, True)
            
            return entry.value
    
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """Put value into cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()
            value_size = self._estimate_size(value)
            
            # Check if we need to make space
            if key not in self._cache and len(self._cache) >= self.max_size:
                if not self._make_space(value_size):
                    return False
            
            # Check memory limits
            if self._size_tracker + value_size > self.max_memory_bytes:
                if not self._free_memory(value_size):
                    return False
            
            # Remove old entry if updating
            if key in self._cache:
                old_entry = self._cache[key]
                self._size_tracker -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=str(key),
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                size_bytes=value_size,
                ttl=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self._access_order[key] = current_time
            self._frequency_counter[key] += 1
            self._size_tracker += value_size
            
            return True
    
    def _make_space(self, needed_bytes: int) -> bool:
        """Make space using current eviction strategy."""
        if self.strategy == CacheStrategy.LRU:
            return self._evict_lru(1)
        elif self.strategy == CacheStrategy.LFU:
            return self._evict_lfu(1)
        elif self.strategy == CacheStrategy.FIFO:
            return self._evict_fifo(1)
        elif self.strategy == CacheStrategy.TTL:
            return self._evict_expired()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            return self._evict_adaptive(needed_bytes)
        else:
            return self._evict_lru(1)  # Default fallback
    
    def _free_memory(self, needed_bytes: int) -> bool:
        """Free memory by evicting entries."""
        freed_bytes = 0
        entries_to_remove = []
        
        # Sort by eviction priority based on current strategy
        if self.strategy == CacheStrategy.LRU:
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        elif self.strategy == CacheStrategy.LFU:
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        else:
            sorted_entries = list(self._cache.items())
        
        for key, entry in sorted_entries:
            if freed_bytes >= needed_bytes:
                break
            entries_to_remove.append(key)
            freed_bytes += entry.size_bytes
        
        # Remove selected entries
        for key in entries_to_remove:
            self._evict_entry(key)
        
        return freed_bytes >= needed_bytes
    
    def _evict_lru(self, count: int) -> bool:
        """Evict least recently used entries."""
        if len(self._access_order) == 0:
            return False
        
        evicted = 0
        for _ in range(min(count, len(self._access_order))):
            if self._access_order:
                oldest_key = next(iter(self._access_order))
                self._evict_entry(oldest_key)
                evicted += 1
        
        return evicted > 0
    
    def _evict_lfu(self, count: int) -> bool:
        """Evict least frequently used entries."""
        if not self._frequency_counter:
            return False
        
        # Get entries sorted by frequency
        sorted_by_freq = sorted(self._frequency_counter.items(), key=lambda x: x[1])
        
        evicted = 0
        for key, _ in sorted_by_freq[:count]:
            if key in self._cache:
                self._evict_entry(key)
                evicted += 1
        
        return evicted > 0
    
    def _evict_fifo(self, count: int) -> bool:
        """Evict first-in entries."""
        if not self._cache:
            return False
        
        # Sort by creation time
        sorted_by_creation = sorted(self._cache.items(), key=lambda x: x[1].created_at)
        
        evicted = 0
        for key, _ in sorted_by_creation[:count]:
            self._evict_entry(key)
            evicted += 1
        
        return evicted > 0
    
    def _evict_expired(self) -> bool:
        """Evict expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._evict_entry(key)
        
        return len(expired_keys) > 0
    
    def _evict_adaptive(self, needed_bytes: int) -> bool:
        """Adaptive eviction based on learned patterns."""
        # Use the best performing strategy
        current_hit_rate = self.get_hit_rate()
        
        if current_hit_rate > 0.8:
            # High hit rate, use conservative LRU
            return self._evict_lru(1)
        elif current_hit_rate < 0.5:
            # Low hit rate, be more aggressive with LFU
            return self._evict_lfu(2)
        else:
            # Medium hit rate, use hybrid approach
            return self._evict_lru(1) or self._evict_lfu(1)
    
    def _evict_entry(self, key: K):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._size_tracker -= entry.size_bytes
            del self._cache[key]
            self.evictions += 1
        
        if key in self._access_order:
            del self._access_order[key]
        
        if key in self._frequency_counter:
            del self._frequency_counter[key]
    
    def _estimate_size(self, value: V) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(len(str(k)) + len(str(v)) for k, v in value.items()) * 2
            elif isinstance(value, list):
                return sum(len(str(item)) for item in value) * 2
            else:
                return 100  # Default estimate
    
    def _record_access_pattern(self, key: K, hit: bool):
        """Record access pattern for learning."""
        pattern = {
            'key_hash': hash(str(key)) % 1000,  # Anonymized key
            'hit': hit,
            'timestamp': time.time()
        }
        self._access_patterns.append(pattern)
        
        # Update hit rate tracking
        if len(self._hit_rates) >= 100:
            self._hit_rates.popleft()
        self._hit_rates.append(1.0 if hit else 0.0)
    
    def get_hit_rate(self) -> float:
        """Get current hit rate."""
        if self.hits + self.misses == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'memory_usage_bytes': self._size_tracker,
            'max_memory_bytes': self.max_memory_bytes,
            'hit_rate': self.get_hit_rate(),
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'strategy': self.strategy.value,
            'avg_access_count': statistics.mean([entry.access_count for entry in self._cache.values()]) if self._cache else 0
        }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._size_tracker = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0


class ConcurrentProcessor:
    """High-performance concurrent processor for IoT data streams."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = config.get('max_workers', multiprocessing.cpu_count())
        self.queue_size = config.get('queue_size', 10000)
        self.batch_size = config.get('batch_size', 100)
        self.batch_timeout_ms = config.get('batch_timeout_ms', 100)
        
        # Processing queues
        self._input_queue = queue.Queue(maxsize=self.queue_size)
        self._output_queue = queue.Queue(maxsize=self.queue_size)
        self._batch_queue = queue.Queue(maxsize=self.queue_size // self.batch_size)
        
        # Worker management
        self._workers = []
        self._batch_worker = None
        self._running = False
        
        # Performance tracking
        self._processed_count = 0
        self._error_count = 0
        self._processing_times = deque(maxlen=1000)
        self._start_time = time.time()
        
        logger.info(f"Concurrent processor initialized: max_workers={self.max_workers}, batch_size={self.batch_size}")
    
    def start(self):
        """Start concurrent processing."""
        if self._running:
            return
        
        self._running = True
        
        # Start batch collector
        self._batch_worker = threading.Thread(target=self._batch_collector, daemon=True)
        self._batch_worker.start()
        
        # Start processing workers
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._process_worker, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started concurrent processor with {self.max_workers} workers")
    
    def stop(self):
        """Stop concurrent processing."""
        self._running = False
        
        # Signal workers to stop
        for _ in range(self.max_workers):
            self._batch_queue.put(None)  # Sentinel value
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        if self._batch_worker:
            self._batch_worker.join(timeout=5.0)
        
        logger.info("Stopped concurrent processor")
    
    async def process_async(self, data: Any, processor_func: Callable) -> Any:
        """Process data asynchronously."""
        if not self._running:
            self.start()
        
        # Create future for result
        result_future = asyncio.get_event_loop().create_future()
        
        # Add to input queue with callback
        try:
            task = {
                'data': data,
                'processor': processor_func,
                'future': result_future,
                'timestamp': time.time()
            }
            self._input_queue.put(task, timeout=1.0)
            return await result_future
        except queue.Full:
            raise RuntimeError("Processing queue is full")
    
    def process_batch_sync(self, data_batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process batch of data synchronously."""
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(data_batch), self.max_workers)) as executor:
            future_to_data = {executor.submit(processor_func, data): data for data in data_batch}
            
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    self._error_count += 1
                    results.append(None)
        
        processing_time = (time.time() - start_time) * 1000
        self._processing_times.append(processing_time)
        self._processed_count += len(data_batch)
        
        return results
    
    def _batch_collector(self):
        """Collect individual items into batches."""
        current_batch = []
        last_batch_time = time.time()
        
        while self._running:
            try:
                # Try to get item with timeout
                try:
                    task = self._input_queue.get(timeout=0.1)
                    if task is None:  # Sentinel for shutdown
                        break
                    current_batch.append(task)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                batch_age_ms = (current_time - last_batch_time) * 1000
                
                # Send batch if it's full or timeout reached
                if (len(current_batch) >= self.batch_size or 
                    (current_batch and batch_age_ms >= self.batch_timeout_ms)):
                    
                    try:
                        self._batch_queue.put(current_batch, timeout=0.1)
                        current_batch = []
                        last_batch_time = current_time
                    except queue.Full:
                        logger.warning("Batch queue is full, dropping batch")
                        current_batch = []
                        last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Batch collector error: {e}")
        
        # Send final batch
        if current_batch:
            try:
                self._batch_queue.put(current_batch, timeout=1.0)
            except queue.Full:
                logger.warning("Final batch dropped due to full queue")
    
    def _process_worker(self, worker_id: int):
        """Worker thread for processing batches."""
        logger.info(f"Started processing worker {worker_id}")
        
        while self._running:
            try:
                # Get batch to process
                batch = self._batch_queue.get(timeout=1.0)
                if batch is None:  # Sentinel for shutdown
                    break
                
                self._process_batch(batch)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of tasks."""
        start_time = time.time()
        
        for task in batch:
            try:
                data = task['data']
                processor_func = task['processor']
                future = task['future']
                
                # Process the data
                result = processor_func(data)
                
                # Set result in future
                if not future.done():
                    future.get_loop().call_soon_threadsafe(future.set_result, result)
                
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                self._error_count += 1
                
                # Set exception in future
                future_obj = task.get('future')
                if future_obj and not future_obj.done():
                    future_obj.get_loop().call_soon_threadsafe(future_obj.set_exception, e)
        
        processing_time = (time.time() - start_time) * 1000
        self._processing_times.append(processing_time)
        self._processed_count += len(batch)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        uptime = time.time() - self._start_time
        throughput = self._processed_count / uptime if uptime > 0 else 0
        
        avg_processing_time = statistics.mean(self._processing_times) if self._processing_times else 0
        p95_processing_time = np.percentile(self._processing_times, 95) if self._processing_times else 0
        
        return {
            'processed_count': self._processed_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._processed_count, 1),
            'throughput_per_second': throughput,
            'avg_processing_time_ms': avg_processing_time,
            'p95_processing_time_ms': p95_processing_time,
            'queue_depth': self._input_queue.qsize(),
            'batch_queue_depth': self._batch_queue.qsize(),
            'active_workers': len([w for w in self._workers if w.is_alive()]),
            'uptime_seconds': uptime
        }


class AutoScalingOrchestrator:
    """Intelligent auto-scaling orchestrator with predictive capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Scaling configuration
        self.min_instances = config.get('min_instances', 1)
        self.max_instances = config.get('max_instances', 10)
        self.target_cpu_percent = config.get('target_cpu_percent', 70)
        self.target_memory_percent = config.get('target_memory_percent', 80)
        self.scale_up_cooldown = config.get('scale_up_cooldown_seconds', 300)
        self.scale_down_cooldown = config.get('scale_down_cooldown_seconds', 600)
        
        # Predictive scaling
        self.enable_predictive = config.get('enable_predictive_scaling', True)
        self.prediction_window_minutes = config.get('prediction_window_minutes', 15)
        
        # Current state
        self.current_instances = self.min_instances
        self.last_scale_time = 0
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        
        # Machine learning for predictions
        self._load_patterns = deque(maxlen=1440)  # 24 hours of minute data
        
        logger.info(f"Auto-scaling orchestrator initialized: {self.min_instances}-{self.max_instances} instances")
    
    def should_scale(self, metrics: PerformanceMetrics) -> Optional[ScalingDecision]:
        """Determine if scaling action is needed."""
        current_time = time.time()
        
        # Add metrics to history
        self.metrics_history.append(metrics)
        
        # Check cooldown period
        if current_time - self.last_scale_time < min(self.scale_up_cooldown, self.scale_down_cooldown):
            return None
        
        # Get recent metrics for analysis
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
        if len(recent_metrics) < 3:
            return None
        
        # Calculate averages
        avg_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage_percent for m in recent_metrics])
        avg_response_time = statistics.mean([m.avg_response_time_ms for m in recent_metrics])
        avg_error_rate = statistics.mean([m.error_rate_percent for m in recent_metrics])
        
        # Scaling decision logic
        scale_up_reasons = []
        scale_down_reasons = []
        
        # CPU-based scaling
        if avg_cpu > self.target_cpu_percent + 10:  # 10% buffer
            scale_up_reasons.append(f"High CPU usage: {avg_cpu:.1f}%")
        elif avg_cpu < self.target_cpu_percent - 20:  # Larger buffer for scale down
            scale_down_reasons.append(f"Low CPU usage: {avg_cpu:.1f}%")
        
        # Memory-based scaling
        if avg_memory > self.target_memory_percent + 10:
            scale_up_reasons.append(f"High memory usage: {avg_memory:.1f}%")
        elif avg_memory < self.target_memory_percent - 20:
            scale_down_reasons.append(f"Low memory usage: {avg_memory:.1f}%")
        
        # Response time-based scaling
        if avg_response_time > 5000:  # 5 seconds
            scale_up_reasons.append(f"High response time: {avg_response_time:.0f}ms")
        elif avg_response_time < 1000:  # 1 second
            scale_down_reasons.append(f"Low response time: {avg_response_time:.0f}ms")
        
        # Error rate-based scaling
        if avg_error_rate > 5.0:  # 5% error rate
            scale_up_reasons.append(f"High error rate: {avg_error_rate:.1f}%")
        
        # Queue depth-based scaling
        if metrics.queue_depth > 1000:
            scale_up_reasons.append(f"High queue depth: {metrics.queue_depth}")
        elif metrics.queue_depth < 10:
            scale_down_reasons.append(f"Low queue depth: {metrics.queue_depth}")
        
        # Predictive scaling
        if self.enable_predictive:
            predicted_load = self._predict_future_load()
            if predicted_load > 1.5:  # 50% increase predicted
                scale_up_reasons.append(f"Predicted load increase: {predicted_load:.2f}x")
            elif predicted_load < 0.7:  # 30% decrease predicted
                scale_down_reasons.append(f"Predicted load decrease: {predicted_load:.2f}x")
        
        # Make scaling decision
        if scale_up_reasons and self.current_instances < self.max_instances:
            confidence = min(1.0, len(scale_up_reasons) * 0.3)  # Max confidence with multiple reasons
            
            # Check cooldown for scale up
            if current_time - self.last_scale_time >= self.scale_up_cooldown:
                return ScalingDecision(
                    action="scale_up",
                    current_instances=self.current_instances,
                    target_instances=min(self.current_instances + 1, self.max_instances),
                    reason="; ".join(scale_up_reasons),
                    confidence_score=confidence,
                    estimated_impact={
                        'cpu_reduction': 30.0,
                        'memory_reduction': 25.0,
                        'response_time_improvement': 40.0
                    },
                    cooldown_until=current_time + self.scale_up_cooldown
                )
        
        elif scale_down_reasons and self.current_instances > self.min_instances:
            # Be more conservative with scale down
            if len(scale_down_reasons) >= 2:  # Need multiple reasons to scale down
                confidence = min(0.8, len(scale_down_reasons) * 0.2)  # Lower max confidence
                
                # Check cooldown for scale down
                if current_time - self.last_scale_time >= self.scale_down_cooldown:
                    return ScalingDecision(
                        action="scale_down",
                        current_instances=self.current_instances,
                        target_instances=max(self.current_instances - 1, self.min_instances),
                        reason="; ".join(scale_down_reasons),
                        confidence_score=confidence,
                        estimated_impact={
                            'cost_savings': 20.0,
                            'resource_efficiency': 15.0
                        },
                        cooldown_until=current_time + self.scale_down_cooldown
                    )
        
        return None
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        try:
            logger.info(f"Executing scaling decision: {decision.action} from {decision.current_instances} to {decision.target_instances}")
            logger.info(f"Reason: {decision.reason} (confidence: {decision.confidence_score:.2f})")
            
            # Update current instance count
            self.current_instances = decision.target_instances
            self.last_scale_time = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': decision.action,
                'from_instances': decision.current_instances,
                'to_instances': decision.target_instances,
                'reason': decision.reason,
                'confidence': decision.confidence_score
            })
            
            # In production, this would trigger actual infrastructure scaling
            # For now, we simulate the scaling action
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return False
    
    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns."""
        if len(self.metrics_history) < 60:  # Need at least 1 hour of data
            return 1.0  # No prediction, assume current load
        
        # Simple trend analysis (in production, would use more sophisticated ML)
        recent_load = statistics.mean([m.requests_per_second for m in list(self.metrics_history)[-15:]])
        historical_load = statistics.mean([m.requests_per_second for m in list(self.metrics_history)[-60:-15]])
        
        if historical_load > 0:
            trend = recent_load / historical_load
            
            # Time-based patterns (e.g., higher load during business hours)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                trend *= 1.2
            elif 22 <= current_hour or current_hour <= 6:  # Night time
                trend *= 0.8
            
            return trend
        
        return 1.0
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        recent_scaling = list(self.scaling_history)[-5:] if self.scaling_history else []
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'last_scale_time': self.last_scale_time,
            'scaling_cooldown_remaining': max(0, (self.last_scale_time + self.scale_up_cooldown) - time.time()),
            'recent_scaling_actions': recent_scaling,
            'total_scale_ups': len([s for s in self.scaling_history if s['action'] == 'scale_up']),
            'total_scale_downs': len([s for s in self.scaling_history if s['action'] == 'scale_down']),
            'predictive_scaling_enabled': self.enable_predictive,
            'metrics_history_length': len(self.metrics_history)
        }


class HyperScaleOptimizationEngine:
    """Main orchestrator for hyperscale optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        cache_config = config.get('cache', {})
        self.cache = IntelligentCache(
            max_size=cache_config.get('max_size', 10000),
            strategy=CacheStrategy(cache_config.get('strategy', 'adaptive')),
            default_ttl=cache_config.get('default_ttl'),
            max_memory_mb=cache_config.get('max_memory_mb', 100)
        )
        
        processor_config = config.get('processor', {})
        self.processor = ConcurrentProcessor(processor_config)
        
        scaling_config = config.get('scaling', {})
        self.scaler = AutoScalingOrchestrator(scaling_config)
        
        # Optimization settings
        self.optimization_level = OptimizationLevel(config.get('optimization_level', 'standard'))
        self.enable_auto_tuning = config.get('enable_auto_tuning', True)
        self.monitoring_interval = config.get('monitoring_interval_seconds', 60)
        
        # Performance monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        logger.info(f"HyperScale optimization engine initialized with level: {self.optimization_level}")
    
    def start_optimization(self):
        """Start optimization engine."""
        self.processor.start()
        
        if self.enable_auto_tuning:
            self._start_monitoring()
        
        logger.info("HyperScale optimization engine started")
    
    def stop_optimization(self):
        """Stop optimization engine."""
        self.processor.stop()
        self._stop_monitoring()
        
        logger.info("HyperScale optimization engine stopped")
    
    async def optimized_inference(self, data: Dict[str, Any], model_func: Callable) -> Any:
        """Perform optimized inference with caching and concurrent processing."""
        # Generate cache key
        cache_key = self._generate_cache_key(data)
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            return cached_result
        
        # Process with concurrent processing
        result = await self.processor.process_async(data, model_func)
        
        # Cache result
        self.cache.put(cache_key, result, ttl=self.config.get('inference_cache_ttl', 300))
        
        return result
    
    def optimize_batch(self, data_batch: List[Dict[str, Any]], model_func: Callable) -> List[Any]:
        """Optimize batch processing."""
        # Filter out items that are in cache
        cache_keys = [self._generate_cache_key(data) for data in data_batch]
        cached_results = {}
        uncached_data = []
        uncached_indices = []
        
        for i, (data, cache_key) in enumerate(zip(data_batch, cache_keys)):
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                cached_results[i] = cached_result
            else:
                uncached_data.append(data)
                uncached_indices.append(i)
        
        # Process uncached data
        if uncached_data:
            uncached_results = self.processor.process_batch_sync(uncached_data, model_func)
            
            # Cache new results
            for data, result, cache_key in zip(uncached_data, uncached_results, 
                                             [cache_keys[i] for i in uncached_indices]):
                if result is not None:
                    self.cache.put(cache_key, result, ttl=self.config.get('inference_cache_ttl', 300))
        else:
            uncached_results = []
        
        # Combine cached and uncached results
        final_results = [None] * len(data_batch)
        
        # Fill in cached results
        for i, result in cached_results.items():
            final_results[i] = result
        
        # Fill in uncached results
        for i, result in zip(uncached_indices, uncached_results):
            final_results[i] = result
        
        return final_results
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from data."""
        # Create deterministic hash of the data
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _start_monitoring(self):
        """Start performance monitoring thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Performance monitoring and auto-scaling loop."""
        logger.info("Started performance monitoring loop")
        
        while self._monitoring_active:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                # Check for scaling decisions
                scaling_decision = self.scaler.should_scale(metrics)
                if scaling_decision:
                    logger.info(f"Auto-scaling triggered: {scaling_decision.action} (confidence: {scaling_decision.confidence_score:.2f})")
                    self.scaler.execute_scaling(scaling_decision)
                
                # Auto-tune cache if needed
                if self.enable_auto_tuning:
                    self._auto_tune_cache(metrics)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Performance monitoring loop stopped")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Cache metrics
        cache_stats = self.cache.get_stats()
        
        # Processor metrics
        processor_stats = self.processor.get_performance_metrics()
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / 1024 / 1024,
            memory_usage_percent=memory.percent,
            queue_depth=processor_stats['queue_depth'],
            avg_response_time_ms=processor_stats['avg_processing_time_ms'],
            requests_per_second=processor_stats['throughput_per_second'],
            error_rate_percent=processor_stats['error_rate'] * 100,
            cache_hit_rate=cache_stats['hit_rate'],
            active_connections=processor_stats['active_workers']
        )
    
    def _auto_tune_cache(self, metrics: PerformanceMetrics):
        """Auto-tune cache parameters based on performance."""
        cache_stats = self.cache.get_stats()
        
        # If hit rate is low and memory usage is high, increase cache size
        if cache_stats['hit_rate'] < 0.6 and metrics.memory_usage_percent < 70:
            new_size = min(cache_stats['max_size'] * 1.2, 50000)  # 20% increase, max 50k
            logger.info(f"Auto-tuning: Increasing cache size from {cache_stats['max_size']} to {new_size}")
            # Would update cache size in production
        
        # If memory usage is very high, reduce cache size
        elif metrics.memory_usage_percent > 90:
            new_size = max(cache_stats['max_size'] * 0.8, 1000)  # 20% decrease, min 1k
            logger.info(f"Auto-tuning: Decreasing cache size from {cache_stats['max_size']} to {new_size}")
            # Would update cache size in production
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        return {
            'optimization_level': self.optimization_level.value,
            'auto_tuning_enabled': self.enable_auto_tuning,
            'monitoring_active': self._monitoring_active,
            'cache': self.cache.get_stats(),
            'processor': self.processor.get_performance_metrics(),
            'scaling': self.scaler.get_scaling_status(),
            'current_metrics': self._collect_performance_metrics().__dict__
        }


def create_hyperscale_engine(config: Optional[Dict[str, Any]] = None) -> HyperScaleOptimizationEngine:
    """Factory function to create hyperscale optimization engine."""
    if config is None:
        config = {
            'optimization_level': 'aggressive',
            'enable_auto_tuning': True,
            'monitoring_interval_seconds': 30,
            'cache': {
                'max_size': 10000,
                'strategy': 'adaptive',
                'max_memory_mb': 200,
                'default_ttl': 300
            },
            'processor': {
                'max_workers': multiprocessing.cpu_count() * 2,
                'queue_size': 10000,
                'batch_size': 100,
                'batch_timeout_ms': 100
            },
            'scaling': {
                'min_instances': 1,
                'max_instances': 20,
                'target_cpu_percent': 70,
                'target_memory_percent': 80,
                'enable_predictive_scaling': True,
                'scale_up_cooldown_seconds': 180,
                'scale_down_cooldown_seconds': 300
            },
            'inference_cache_ttl': 600
        }
    
    return HyperScaleOptimizationEngine(config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def example_hyperscale_optimization():
        """Example demonstrating hyperscale optimization."""
        engine = create_hyperscale_engine()
        engine.start_optimization()
        
        try:
            # Simulate model inference function
            def mock_model_inference(data):
                time.sleep(0.01)  # Simulate processing time
                return {
                    'prediction': data.get('sensor_id', 'unknown'),
                    'confidence': 0.95,
                    'processing_time': 10.5
                }
            
            # Test optimized inference
            test_data = {
                'sensor_id': 'sensor_001',
                'timestamp': datetime.now().isoformat(),
                'values': {'temperature': 25.6, 'humidity': 60.2}
            }
            
            # Single inference
            result1 = await engine.optimized_inference(test_data, mock_model_inference)
            print(f"First inference result: {result1}")
            
            # Second inference (should hit cache)
            result2 = await engine.optimized_inference(test_data, mock_model_inference)
            print(f"Second inference result (cached): {result2}")
            
            # Batch processing
            batch_data = [
                {**test_data, 'sensor_id': f'sensor_{i:03d}'} 
                for i in range(1, 101)
            ]
            
            batch_results = engine.optimize_batch(batch_data, mock_model_inference)
            print(f"Batch processed {len(batch_results)} items")
            
            # Wait a bit for monitoring
            await asyncio.sleep(5)
            
            # Get status
            status = engine.get_optimization_status()
            print(f"Optimization status: {json.dumps(status, indent=2, default=str)}")
            
        finally:
            engine.stop_optimization()
    
    try:
        asyncio.run(example_hyperscale_optimization())
    except KeyboardInterrupt:
        logger.info("Example stopped by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")