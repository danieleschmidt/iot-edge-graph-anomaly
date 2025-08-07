"""
Tests for sentiment analyzer performance components.
"""
import pytest
import time
from sentiment_analyzer.performance.cache import LRUCache


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def setup_method(self):
        self.cache = LRUCache(max_size=5, ttl_seconds=10)
    
    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        # Put and get
        self.cache.put("hello", "vader", "result1")
        result = self.cache.get("hello", "vader")
        assert result == "result1"
        
        # Miss
        result = self.cache.get("nonexistent", "vader")
        assert result is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(5):
            self.cache.put(f"text{i}", "vader", f"result{i}")
        
        # All should be present
        for i in range(5):
            assert self.cache.get(f"text{i}", "vader") == f"result{i}"
        
        # Add one more - should evict least recently used (text0)
        self.cache.put("text5", "vader", "result5")
        
        # text0 should be evicted
        assert self.cache.get("text0", "vader") is None
        assert self.cache.get("text5", "vader") == "result5"
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Make some hits and misses
        self.cache.put("hit1", "vader", "result1")
        self.cache.put("hit2", "vader", "result2")
        
        self.cache.get("hit1", "vader")  # hit
        self.cache.get("hit2", "vader")  # hit
        self.cache.get("miss1", "vader")  # miss
        self.cache.get("miss2", "vader")  # miss
        
        stats = self.cache.get_stats()
        assert stats['hits'] >= 2
        assert stats['misses'] >= 2
        assert stats['size'] >= 2
        assert 0 <= stats['hit_rate'] <= 1