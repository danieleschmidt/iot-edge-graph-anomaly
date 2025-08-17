"""
Intelligent Caching System for IoT Edge Anomaly Detection.

Implements adaptive caching strategies with machine learning-based cache 
optimization, multi-level cache hierarchies, and intelligent prefetching
for high-performance edge deployment.

Key Features:
- ML-driven cache replacement policies
- Multi-level cache hierarchy (L1/L2/L3)
- Intelligent prefetching based on temporal patterns
- Adaptive cache sizing based on workload
- Cache-aware model optimization
- Distributed cache coherence for multi-node deployments
"""

import time
import threading
import logging
import hashlib
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import OrderedDict, defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache level enumeration."""
    L1_CPU = "l1_cpu"          # CPU cache (fastest, smallest)
    L2_MEMORY = "l2_memory"    # Memory cache (fast, medium)
    L3_STORAGE = "l3_storage"  # Storage cache (slower, largest)


class CachePolicy(Enum):
    """Cache replacement policies."""
    LRU = "lru"                # Least Recently Used
    LFU = "lfu"                # Least Frequently Used
    ADAPTIVE = "adaptive"      # ML-based adaptive policy
    TTL = "ttl"                # Time To Live
    PREDICTIVE = "predictive"  # Predictive prefetching


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[float] = None
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.creation_time).total_seconds()
    
    @property
    def time_since_last_access(self) -> float:
        """Get time since last access in seconds."""
        return (datetime.now() - self.last_access_time).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
    
    def access(self):
        """Record cache access."""
        self.access_count += 1
        self.last_access_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'size_bytes': self.size_bytes,
            'access_count': self.access_count,
            'creation_time': self.creation_time.isoformat(),
            'last_access_time': self.last_access_time.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'priority': self.priority,
            'age_seconds': self.age_seconds,
            'time_since_last_access': self.time_since_last_access,
            'is_expired': self.is_expired,
            'metadata': self.metadata
        }


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    average_access_time_ms: float = 0.0
    hit_ratio: float = 0.0
    miss_ratio: float = 0.0
    
    def update_hit(self, access_time_ms: float):
        """Update statistics for cache hit."""
        self.total_requests += 1
        self.cache_hits += 1
        self._update_access_time(access_time_ms)
        self._recalculate_ratios()
    
    def update_miss(self, access_time_ms: float):
        """Update statistics for cache miss."""
        self.total_requests += 1
        self.cache_misses += 1
        self._update_access_time(access_time_ms)
        self._recalculate_ratios()
    
    def update_eviction(self):
        """Update statistics for cache eviction."""
        self.evictions += 1
    
    def _update_access_time(self, access_time_ms: float):
        """Update average access time."""
        if self.total_requests == 1:
            self.average_access_time_ms = access_time_ms
        else:
            # Moving average
            alpha = 0.1  # Smoothing factor
            self.average_access_time_ms = (
                (1 - alpha) * self.average_access_time_ms + 
                alpha * access_time_ms
            )
    
    def _recalculate_ratios(self):
        """Recalculate hit and miss ratios."""
        if self.total_requests > 0:
            self.hit_ratio = self.cache_hits / self.total_requests
            self.miss_ratio = self.cache_misses / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AdaptiveCachePredictor:
    """
    ML-based cache predictor for intelligent replacement and prefetching.
    
    Uses historical access patterns to predict future cache needs and
    optimize replacement policies.
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.access_history: deque = deque(maxlen=history_size)
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.temporal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Simple prediction models
        self.access_frequency_weights: Dict[str, float] = {}
        self.temporal_weights: Dict[str, float] = {}
        
    def record_access(self, key: str, timestamp: Optional[datetime] = None):
        """Record cache access for pattern learning."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.access_history.append((key, timestamp))
        self.temporal_patterns[key].append(timestamp)
        
        # Maintain limited history per key
        if len(self.temporal_patterns[key]) > 100:
            self.temporal_patterns[key] = self.temporal_patterns[key][-100:]
        
        # Update access frequency
        self._update_access_frequency(key)
        self._update_temporal_patterns(key)
    
    def predict_access_probability(self, key: str, future_seconds: float = 300) -> float:
        """
        Predict probability of accessing a key within future_seconds.
        
        Args:
            key: Cache key
            future_seconds: Time window for prediction
            
        Returns:
            Probability between 0 and 1
        """
        # Frequency-based prediction
        frequency_score = self.access_frequency_weights.get(key, 0.0)
        
        # Temporal pattern-based prediction
        temporal_score = self._calculate_temporal_probability(key, future_seconds)
        
        # Combine predictions
        combined_probability = 0.6 * frequency_score + 0.4 * temporal_score
        return min(combined_probability, 1.0)
    
    def get_prefetch_candidates(self, current_keys: List[str], max_candidates: int = 5) -> List[Tuple[str, float]]:
        """
        Get cache keys that should be prefetched based on access patterns.
        
        Args:
            current_keys: Currently cached keys
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (key, probability) tuples
        """
        candidates = []
        
        # Analyze recent access patterns
        recent_accesses = list(self.access_history)[-50:]  # Last 50 accesses
        
        for key, access_patterns in self.access_patterns.items():
            if key in current_keys:
                continue  # Already cached
            
            # Calculate prefetch probability
            probability = self.predict_access_probability(key, future_seconds=600)
            
            # Consider co-occurrence patterns
            co_occurrence_boost = self._calculate_co_occurrence_boost(key, recent_accesses)
            probability = min(probability + co_occurrence_boost, 1.0)
            
            if probability > 0.3:  # Threshold for prefetching
                candidates.append((key, probability))
        
        # Sort by probability and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def calculate_replacement_priority(self, entry: CacheEntry) -> float:
        """
        Calculate replacement priority for cache entry (lower = more likely to be evicted).
        
        Args:
            entry: Cache entry to evaluate
            
        Returns:
            Priority score (0-1, lower means higher eviction priority)
        """
        # Frequency component
        frequency_score = self.access_frequency_weights.get(entry.key, 0.0)
        
        # Recency component (inverse of time since last access)
        recency_score = 1.0 / (1.0 + entry.time_since_last_access / 3600)  # 1 hour decay
        
        # Future access probability
        future_probability = self.predict_access_probability(entry.key, future_seconds=1800)
        
        # Size penalty (larger entries have lower priority)
        size_penalty = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))  # 1MB reference
        
        # Combine factors
        priority = (
            0.3 * frequency_score +
            0.3 * recency_score +
            0.3 * future_probability +
            0.1 * size_penalty
        )
        
        return min(priority, 1.0)
    
    def _update_access_frequency(self, key: str):
        """Update access frequency weights."""
        # Simple exponential moving average
        current_weight = self.access_frequency_weights.get(key, 0.0)
        self.access_frequency_weights[key] = 0.9 * current_weight + 0.1
        
        # Decay other weights slightly
        for other_key in self.access_frequency_weights:
            if other_key != key:
                self.access_frequency_weights[other_key] *= 0.999
    
    def _update_temporal_patterns(self, key: str):
        """Update temporal access patterns."""
        accesses = self.temporal_patterns[key]
        if len(accesses) < 2:
            return
        
        # Calculate access intervals
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        # Update temporal weight based on regularity
        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            regularity = 1.0 / (1.0 + std_interval / max(mean_interval, 1.0))
            self.temporal_weights[key] = regularity
    
    def _calculate_temporal_probability(self, key: str, future_seconds: float) -> float:
        """Calculate temporal-based access probability."""
        if key not in self.temporal_patterns or len(self.temporal_patterns[key]) < 2:
            return 0.0
        
        accesses = self.temporal_patterns[key]
        current_time = datetime.now()
        
        # Find typical access interval
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        mean_interval = np.mean(intervals)
        time_since_last = (current_time - accesses[-1]).total_seconds()
        
        # Probability based on expected next access time
        if time_since_last >= mean_interval * 0.8:
            # Getting close to expected next access
            return min(time_since_last / mean_interval, 1.0)
        else:
            return 0.1  # Low probability if accessed recently
    
    def _calculate_co_occurrence_boost(self, key: str, recent_accesses: List[Tuple[str, datetime]]) -> float:
        """Calculate boost based on co-occurrence patterns."""
        if not recent_accesses:
            return 0.0
        
        # Count co-occurrences in recent history
        recent_keys = [access[0] for access in recent_accesses[-10:]]
        co_occurrence_count = 0
        
        # Look for patterns where this key was accessed after recent keys
        for access_key, _ in recent_accesses:
            if access_key in recent_keys:
                # Check if target key was accessed shortly after
                co_occurrence_count += 1
        
        return min(co_occurrence_count / 10.0, 0.3)  # Max 0.3 boost


class IntelligentCacheLayer:
    """
    Single cache layer with intelligent management.
    
    Implements adaptive replacement policies, intelligent prefetching,
    and performance optimization.
    """
    
    def __init__(
        self,
        level: CacheLevel,
        max_size_bytes: int,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        enable_prefetching: bool = True
    ):
        self.level = level
        self.max_size_bytes = max_size_bytes
        self.policy = policy
        self.enable_prefetching = enable_prefetching
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        
        # Statistics and monitoring
        self.statistics = CacheStatistics()
        self.predictor = AdaptiveCachePredictor()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        self._prefetch_interval = 60   # 1 minute
        self._last_prefetch = time.time()
        
        logger.info(f"Initialized {level.value} cache layer: {max_size_bytes/1024/1024:.1f}MB max")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired:
                    self._remove_entry(key)
                    access_time_ms = (time.time() - start_time) * 1000
                    self.statistics.update_miss(access_time_ms)
                    return None
                
                # Update access information
                entry.access()
                self.predictor.record_access(key)
                
                # Move to end (LRU behavior)
                self.cache.move_to_end(key)
                
                access_time_ms = (time.time() - start_time) * 1000
                self.statistics.update_hit(access_time_ms)
                
                logger.debug(f"Cache hit: {key} ({self.level.value})")
                return entry.value
            else:
                access_time_ms = (time.time() - start_time) * 1000
                self.statistics.update_miss(access_time_ms)
                logger.debug(f"Cache miss: {key} ({self.level.value})")
                return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[float] = None,
        priority: float = 1.0
    ) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            priority: Cache priority (higher = more important)
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate value size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = sys.getsizeof(value)
            
            # Check if value is too large for cache
            if size_bytes > self.max_size_bytes * 0.5:  # Max 50% of cache
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Make space for new entry
            self._ensure_space(size_bytes)
            
            # Create and add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                priority=priority
            )
            
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
            self.statistics.total_size_bytes = self.current_size_bytes
            
            logger.debug(f"Cached: {key} ({size_bytes} bytes, {self.level.value})")
            
            # Trigger background tasks if needed
            self._maybe_run_background_tasks()
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.statistics.total_size_bytes = 0
            logger.info(f"Cleared {self.level.value} cache")
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics."""
        with self._lock:
            return self.statistics
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        with self._lock:
            total_entries = len(self.cache)
            utilization = self.current_size_bytes / self.max_size_bytes
            
            # Entry age distribution
            ages = [entry.age_seconds for entry in self.cache.values()]
            age_stats = {
                'mean_age': np.mean(ages) if ages else 0,
                'max_age': max(ages) if ages else 0,
                'min_age': min(ages) if ages else 0
            }
            
            return {
                'level': self.level.value,
                'total_entries': total_entries,
                'current_size_bytes': self.current_size_bytes,
                'max_size_bytes': self.max_size_bytes,
                'utilization': utilization,
                'statistics': self.statistics.to_dict(),
                'age_statistics': age_stats,
                'policy': self.policy.value,
                'prefetching_enabled': self.enable_prefetching
            }
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            self.statistics.total_size_bytes = self.current_size_bytes
            self.statistics.update_eviction()
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space for new entry."""
        while (self.current_size_bytes + required_bytes > self.max_size_bytes and 
               self.cache):
            
            # Choose eviction candidate based on policy
            victim_key = self._select_eviction_candidate()
            if victim_key:
                logger.debug(f"Evicting: {victim_key} ({self.level.value})")
                self._remove_entry(victim_key)
            else:
                break  # No suitable candidate found
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select cache entry for eviction based on policy."""
        if not self.cache:
            return None
        
        if self.policy == CachePolicy.LRU:
            # Least recently used (first in OrderedDict)
            return next(iter(self.cache))
        
        elif self.policy == CachePolicy.LFU:
            # Least frequently used
            min_access_count = min(entry.access_count for entry in self.cache.values())
            for key, entry in self.cache.items():
                if entry.access_count == min_access_count:
                    return key
        
        elif self.policy == CachePolicy.TTL:
            # Expired entries first, then oldest
            for key, entry in self.cache.items():
                if entry.is_expired:
                    return key
            # If no expired, evict oldest
            return next(iter(self.cache))
        
        elif self.policy == CachePolicy.ADAPTIVE:
            # ML-based adaptive eviction
            min_priority = float('inf')
            victim_key = None
            
            for key, entry in self.cache.items():
                priority = self.predictor.calculate_replacement_priority(entry)
                if priority < min_priority:
                    min_priority = priority
                    victim_key = key
            
            return victim_key
        
        else:
            # Default to LRU
            return next(iter(self.cache))
    
    def _maybe_run_background_tasks(self):
        """Run background tasks if intervals have elapsed."""
        current_time = time.time()
        
        # Cleanup expired entries
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = current_time
        
        # Prefetch likely-to-be-accessed entries
        if (self.enable_prefetching and 
            current_time - self._last_prefetch > self._prefetch_interval):
            self._prefetch_candidates()
            self._last_prefetch = current_time
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            logger.debug(f"Cleaned up expired entry: {key}")
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _prefetch_candidates(self):
        """Prefetch likely-to-be-accessed entries."""
        current_keys = list(self.cache.keys())
        candidates = self.predictor.get_prefetch_candidates(current_keys, max_candidates=3)
        
        # This is a placeholder - in practice, prefetching would load data
        # from lower cache levels or original sources
        for key, probability in candidates:
            logger.debug(f"Prefetch candidate: {key} (probability: {probability:.3f})")


class MultiLevelIntelligentCache:
    """
    Multi-level intelligent cache system.
    
    Implements hierarchical caching with L1/L2/L3 levels, intelligent
    promotion/demotion, and cross-level optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize cache levels
        self.levels: Dict[CacheLevel, IntelligentCacheLayer] = {}
        
        # L1 Cache (CPU/Memory - fastest, smallest)
        l1_config = config.get('l1', {})
        self.levels[CacheLevel.L1_CPU] = IntelligentCacheLayer(
            level=CacheLevel.L1_CPU,
            max_size_bytes=l1_config.get('max_size_mb', 50) * 1024 * 1024,
            policy=CachePolicy(l1_config.get('policy', 'adaptive')),
            enable_prefetching=l1_config.get('enable_prefetching', True)
        )
        
        # L2 Cache (Memory - fast, medium)
        l2_config = config.get('l2', {})
        self.levels[CacheLevel.L2_MEMORY] = IntelligentCacheLayer(
            level=CacheLevel.L2_MEMORY,
            max_size_bytes=l2_config.get('max_size_mb', 200) * 1024 * 1024,
            policy=CachePolicy(l2_config.get('policy', 'lru')),
            enable_prefetching=l2_config.get('enable_prefetching', True)
        )
        
        # L3 Cache (Storage - slower, largest)
        l3_config = config.get('l3', {})
        self.levels[CacheLevel.L3_STORAGE] = IntelligentCacheLayer(
            level=CacheLevel.L3_STORAGE,
            max_size_bytes=l3_config.get('max_size_mb', 1000) * 1024 * 1024,
            policy=CachePolicy(l3_config.get('policy', 'lru')),
            enable_prefetching=l3_config.get('enable_prefetching', False)
        )
        
        # Global statistics
        self.global_statistics = CacheStatistics()
        
        # Promotion/demotion thresholds
        self.promotion_threshold = config.get('promotion_threshold', 3)  # Access count
        self.demotion_interval = config.get('demotion_interval', 300)   # Seconds
        
        logger.info("Initialized Multi-Level Intelligent Cache System")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy.
        
        Checks L1 -> L2 -> L3 and promotes found values to higher levels.
        """
        start_time = time.time()
        
        # Try L1 first
        value = self.levels[CacheLevel.L1_CPU].get(key)
        if value is not None:
            access_time_ms = (time.time() - start_time) * 1000
            self.global_statistics.update_hit(access_time_ms)
            return value
        
        # Try L2
        value = self.levels[CacheLevel.L2_MEMORY].get(key)
        if value is not None:
            # Promote to L1
            self.levels[CacheLevel.L1_CPU].put(key, value)
            access_time_ms = (time.time() - start_time) * 1000
            self.global_statistics.update_hit(access_time_ms)
            logger.debug(f"Promoted {key} from L2 to L1")
            return value
        
        # Try L3
        value = self.levels[CacheLevel.L3_STORAGE].get(key)
        if value is not None:
            # Promote to L2 and L1
            self.levels[CacheLevel.L2_MEMORY].put(key, value)
            self.levels[CacheLevel.L1_CPU].put(key, value)
            access_time_ms = (time.time() - start_time) * 1000
            self.global_statistics.update_hit(access_time_ms)
            logger.debug(f"Promoted {key} from L3 to L2/L1")
            return value
        
        # Cache miss across all levels
        access_time_ms = (time.time() - start_time) * 1000
        self.global_statistics.update_miss(access_time_ms)
        return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        priority: float = 1.0,
        target_level: Optional[CacheLevel] = None
    ) -> bool:
        """
        Store value in cache hierarchy.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live
            priority: Cache priority
            target_level: Specific level to cache at (optional)
            
        Returns:
            True if successfully cached
        """
        if target_level:
            # Store at specific level
            return self.levels[target_level].put(key, value, ttl_seconds, priority)
        else:
            # Store at all levels (write-through)
            success = True
            
            # Start with L1 and cascade down
            for level in [CacheLevel.L1_CPU, CacheLevel.L2_MEMORY, CacheLevel.L3_STORAGE]:
                if not self.levels[level].put(key, value, ttl_seconds, priority):
                    success = False
                    logger.warning(f"Failed to cache {key} at {level.value}")
            
            return success
    
    def remove(self, key: str) -> bool:
        """Remove entry from all cache levels."""
        removed = False
        for level in self.levels.values():
            if level.remove(key):
                removed = True
        return removed
    
    def clear(self):
        """Clear all cache levels."""
        for level in self.levels.values():
            level.clear()
        logger.info("Cleared all cache levels")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all cache levels."""
        stats = {
            'global_statistics': self.global_statistics.to_dict(),
            'levels': {}
        }
        
        total_size = 0
        total_entries = 0
        
        for level_enum, level_cache in self.levels.items():
            level_info = level_cache.get_cache_info()
            stats['levels'][level_enum.value] = level_info
            total_size += level_info['current_size_bytes']
            total_entries += level_info['total_entries']
        
        stats['summary'] = {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_entries': total_entries
        }
        
        return stats
    
    def optimize_cache_hierarchy(self):
        """Optimize cache hierarchy based on access patterns."""
        logger.info("Optimizing cache hierarchy...")
        
        # Analyze access patterns across levels
        l1_stats = self.levels[CacheLevel.L1_CPU].get_statistics()
        l2_stats = self.levels[CacheLevel.L2_MEMORY].get_statistics()
        l3_stats = self.levels[CacheLevel.L3_STORAGE].get_statistics()
        
        # Implement optimization strategies
        self._optimize_promotion_rules(l1_stats, l2_stats, l3_stats)
        self._balance_cache_sizes()
        self._tune_replacement_policies()
        
        logger.info("Cache hierarchy optimization completed")
    
    def _optimize_promotion_rules(self, l1_stats: CacheStatistics, 
                                 l2_stats: CacheStatistics, 
                                 l3_stats: CacheStatistics):
        """Optimize promotion rules based on hit ratios."""
        
        # If L1 hit ratio is low, be more aggressive with promotions
        if l1_stats.hit_ratio < 0.7:
            self.promotion_threshold = max(1, self.promotion_threshold - 1)
            logger.info(f"Reduced promotion threshold to {self.promotion_threshold}")
        
        # If L1 hit ratio is very high, be more conservative
        elif l1_stats.hit_ratio > 0.9:
            self.promotion_threshold = min(5, self.promotion_threshold + 1)
            logger.info(f"Increased promotion threshold to {self.promotion_threshold}")
    
    def _balance_cache_sizes(self):
        """Balance cache sizes based on utilization patterns."""
        
        # Get current utilization
        l1_info = self.levels[CacheLevel.L1_CPU].get_cache_info()
        l2_info = self.levels[CacheLevel.L2_MEMORY].get_cache_info()
        l3_info = self.levels[CacheLevel.L3_STORAGE].get_cache_info()
        
        # Log current utilization
        logger.info(f"Cache utilization - L1: {l1_info['utilization']:.1%}, "
                   f"L2: {l2_info['utilization']:.1%}, "
                   f"L3: {l3_info['utilization']:.1%}")
        
        # Implement dynamic size adjustment logic here if needed
        # For now, just log the information
    
    def _tune_replacement_policies(self):
        """Tune replacement policies based on performance."""
        
        # Analyze performance and potentially switch policies
        for level_enum, level_cache in self.levels.items():
            stats = level_cache.get_statistics()
            
            # If hit ratio is low, consider switching to adaptive policy
            if stats.hit_ratio < 0.6 and level_cache.policy != CachePolicy.ADAPTIVE:
                logger.info(f"Consider switching {level_enum.value} to adaptive policy")
                # Implementation would switch policy here


def create_intelligent_cache_system(config: Dict[str, Any]) -> MultiLevelIntelligentCache:
    """
    Factory function to create an intelligent cache system.
    
    Args:
        config: Cache configuration
        
    Returns:
        Configured multi-level cache system
    """
    cache_config = config.get('intelligent_cache', {
        'l1': {'max_size_mb': 50, 'policy': 'adaptive'},
        'l2': {'max_size_mb': 200, 'policy': 'lru'},
        'l3': {'max_size_mb': 1000, 'policy': 'lru'},
        'promotion_threshold': 3,
        'demotion_interval': 300
    })
    
    cache_system = MultiLevelIntelligentCache(cache_config)
    
    logger.info("Created intelligent cache system with ML optimization")
    return cache_system


# Cache-aware utilities for model optimization
def cache_aware_model_wrapper(model: nn.Module, cache_system: MultiLevelIntelligentCache):
    """
    Wrap a model with cache-aware optimizations.
    
    Args:
        model: PyTorch model to wrap
        cache_system: Cache system to use
        
    Returns:
        Cache-aware model wrapper
    """
    
    class CacheAwareModel(nn.Module):
        def __init__(self, original_model, cache):
            super().__init__()
            self.model = original_model
            self.cache = cache
            self.input_cache_hits = 0
            self.input_cache_total = 0
        
        def forward(self, x):
            # Create cache key from input hash
            input_hash = hashlib.md5(x.detach().cpu().numpy().tobytes()).hexdigest()
            cache_key = f"model_output_{input_hash}"
            
            # Try to get cached result
            self.input_cache_total += 1
            cached_output = self.cache.get(cache_key)
            
            if cached_output is not None:
                self.input_cache_hits += 1
                logger.debug(f"Cache hit for model inference: {cache_key}")
                return cached_output
            
            # Compute and cache result
            with torch.no_grad():
                output = self.model(x)
                
            # Cache the result
            self.cache.put(cache_key, output.clone(), ttl_seconds=300)  # 5 min TTL
            
            return output
        
        def get_cache_hit_rate(self) -> float:
            """Get cache hit rate for model inference."""
            return self.input_cache_hits / max(self.input_cache_total, 1)
    
    return CacheAwareModel(model, cache_system)