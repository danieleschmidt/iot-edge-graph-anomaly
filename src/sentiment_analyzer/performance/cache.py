"""
Advanced caching system for sentiment analysis results.
"""
import time
import hashlib
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import OrderedDict
import logging
import pickle
import gzip

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - self.created_at) > ttl_seconds
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """
    Thread-safe LRU cache with TTL support and memory management.
    """
    
    def __init__(self, 
                 max_size: int = 10000,
                 ttl_seconds: float = 3600,
                 enable_compression: bool = True):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0,
            'total_size_bytes': 0
        }
        
        logger.info(f"Initialized LRU cache: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, text: str, model_type: str, **kwargs) -> str:
        """Generate cache key from input parameters."""
        # Create deterministic hash
        content = f"{text}:{model_type}:{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)
        if self.enable_compression:
            data = gzip.compress(data)
        return data
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.enable_compression:
            data = gzip.decompress(data)
        return pickle.loads(data)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            serialized = self._serialize_value(value)
            return len(serialized)
        except Exception:
            return 1024  # Default estimate
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired(self.ttl_seconds):
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache.pop(key)
            self._stats['total_size_bytes'] -= entry.size_bytes
            self._stats['expired_removals'] += 1
        
        logger.debug(f"Evicted {len(expired_keys)} expired entries")
    
    def _evict_lru(self):
        """Remove least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats['total_size_bytes'] -= entry.size_bytes
            self._stats['evictions'] += 1
            logger.debug(f"Evicted LRU entry: {key}")
    
    def get(self, text: str, model_type: str, **kwargs) -> Optional[Any]:
        """Get value from cache."""
        key = self._generate_key(text, model_type, **kwargs)
        
        with self._lock:
            # Clean expired entries periodically
            if self._stats['hits'] + self._stats['misses'] % 100 == 0:
                self._evict_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired(self.ttl_seconds):
                    self._cache.pop(key)
                    self._stats['total_size_bytes'] -= entry.size_bytes
                    self._stats['expired_removals'] += 1
                    self._stats['misses'] += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                
                self._stats['hits'] += 1
                
                try:
                    return self._deserialize_value(entry.value)
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached value: {e}")
                    self._cache.pop(key)
                    self._stats['total_size_bytes'] -= entry.size_bytes
                    self._stats['misses'] += 1
                    return None
            
            self._stats['misses'] += 1
            return None
    
    def put(self, text: str, model_type: str, value: Any, **kwargs):
        """Store value in cache."""
        key = self._generate_key(text, model_type, **kwargs)
        
        try:
            serialized_value = self._serialize_value(value)
            size_bytes = len(serialized_value)
            
            with self._lock:
                current_time = time.time()
                
                # Remove existing entry if present
                if key in self._cache:
                    old_entry = self._cache.pop(key)
                    self._stats['total_size_bytes'] -= old_entry.size_bytes
                
                # Ensure we have space
                while (len(self._cache) >= self.max_size or 
                       self._stats['total_size_bytes'] + size_bytes > self.max_size * 1024):
                    if not self._cache:
                        break
                    self._evict_lru()
                
                # Create and store new entry
                entry = CacheEntry(
                    value=serialized_value,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=0,
                    size_bytes=size_bytes
                )
                
                self._cache[key] = entry
                self._stats['total_size_bytes'] += size_bytes
                
                logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")
                
        except Exception as e:
            logger.warning(f"Failed to cache value: {e}")
    
    def invalidate(self, text: str = None, model_type: str = None, **kwargs):
        """Invalidate cache entries."""
        with self._lock:
            if text is not None and model_type is not None:
                # Invalidate specific entry
                key = self._generate_key(text, model_type, **kwargs)
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._stats['total_size_bytes'] -= entry.size_bytes
            else:
                # Clear all entries
                self._cache.clear()
                self._stats['total_size_bytes'] = 0
                logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'expired_removals': self._stats['expired_removals'],
                'total_size_bytes': self._stats['total_size_bytes'],
                'avg_entry_size': self._stats['total_size_bytes'] / len(self._cache) if self._cache else 0,
                'ttl_seconds': self.ttl_seconds,
                'compression_enabled': self.enable_compression
            }
    
    def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most accessed cache entries."""
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    'key': key[:8] + '...',
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes,
                    'age_seconds': time.time() - entry.created_at,
                    'last_accessed': time.time() - entry.accessed_at
                })
            
            entries.sort(key=lambda x: x['access_count'], reverse=True)
            return entries[:limit]


class AdaptiveCache:
    """
    Adaptive cache that adjusts parameters based on usage patterns.
    """
    
    def __init__(self, base_cache: LRUCache):
        self.base_cache = base_cache
        self._usage_patterns = {
            'avg_text_length': 0,
            'model_distribution': {},
            'access_patterns': [],
            'hit_rate_history': []
        }
        self._adaptation_interval = 300  # 5 minutes
        self._last_adaptation = time.time()
    
    def get(self, text: str, model_type: str, **kwargs) -> Optional[Any]:
        """Get with pattern tracking."""
        result = self.base_cache.get(text, model_type, **kwargs)
        
        # Track usage patterns
        self._track_usage(text, model_type, hit=result is not None)
        self._maybe_adapt()
        
        return result
    
    def put(self, text: str, model_type: str, value: Any, **kwargs):
        """Put with pattern tracking."""
        self.base_cache.put(text, model_type, value, **kwargs)
        self._track_usage(text, model_type, hit=False)
    
    def _track_usage(self, text: str, model_type: str, hit: bool):
        """Track usage patterns for adaptation."""
        patterns = self._usage_patterns
        
        # Track text length
        text_len = len(text)
        if patterns['avg_text_length'] == 0:
            patterns['avg_text_length'] = text_len
        else:
            patterns['avg_text_length'] = 0.9 * patterns['avg_text_length'] + 0.1 * text_len
        
        # Track model distribution
        patterns['model_distribution'][model_type] = patterns['model_distribution'].get(model_type, 0) + 1
        
        # Track access patterns
        patterns['access_patterns'].append({
            'timestamp': time.time(),
            'hit': hit,
            'model': model_type,
            'text_length': text_len
        })
        
        # Keep only recent patterns (last 1000 requests)
        if len(patterns['access_patterns']) > 1000:
            patterns['access_patterns'] = patterns['access_patterns'][-1000:]
    
    def _maybe_adapt(self):
        """Adapt cache parameters based on patterns."""
        current_time = time.time()
        
        if current_time - self._last_adaptation < self._adaptation_interval:
            return
        
        stats = self.base_cache.get_stats()
        hit_rate = stats['hit_rate']
        
        # Track hit rate history
        self._usage_patterns['hit_rate_history'].append(hit_rate)
        if len(self._usage_patterns['hit_rate_history']) > 10:
            self._usage_patterns['hit_rate_history'] = self._usage_patterns['hit_rate_history'][-10:]
        
        # Adaptive recommendations
        recommendations = []
        
        # If hit rate is declining, consider increasing cache size
        if len(self._usage_patterns['hit_rate_history']) >= 3:
            recent_rates = self._usage_patterns['hit_rate_history'][-3:]
            if all(recent_rates[i] > recent_rates[i+1] for i in range(len(recent_rates)-1)):
                if stats['size'] == stats['max_size']:
                    recommendations.append("Consider increasing cache size")
        
        # If average text length is much larger than expected, adjust TTL
        avg_length = self._usage_patterns['avg_text_length']
        if avg_length > 1000:  # Long texts might benefit from longer TTL
            recommendations.append("Consider increasing TTL for better hit rates")
        
        # Log recommendations
        if recommendations:
            logger.info(f"Cache adaptation recommendations: {'; '.join(recommendations)}")
        
        self._last_adaptation = current_time
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get adaptation information."""
        return {
            'usage_patterns': self._usage_patterns,
            'base_cache_stats': self.base_cache.get_stats(),
            'last_adaptation': self._last_adaptation
        }


# Global cache instances
result_cache = LRUCache(max_size=10000, ttl_seconds=3600)
adaptive_cache = AdaptiveCache(result_cache)