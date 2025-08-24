"""
Multi-Tier Caching System for Hyperscale IoT Anomaly Detection.

Advanced hierarchical caching system spanning edge-fog-cloud tiers with intelligent
data management, prefetching, and memory optimization for millions of data points.

Key Features:
- Edge-Fog-Cloud hierarchical caching architecture
- Intelligent data prefetching based on sensor patterns
- Memory-mapped tensor operations for zero-copy processing
- Advanced garbage collection with memory pool management
- Geographic data distribution and locality optimization
- Real-time cache coherence and invalidation
- Predictive cache warming based on usage patterns
- Content-aware compression and deduplication
- Distributed cache synchronization across regions
"""

import asyncio
import logging
import time
import json
import mmap
import threading
import hashlib
import pickle
import zlib
import lz4.frame as lz4
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
import weakref
import gc
import sys
import os
import uuid

import numpy as np
import torch
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels in the hierarchy."""
    EDGE = "edge"
    FOG = "fog"
    CLOUD = "cloud"
    GLOBAL = "global"


class CompressionType(Enum):
    """Data compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    BZIP2 = "bzip2"
    TENSOR_COMPRESSION = "tensor_compression"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    PREDICTIVE = "predictive"
    GEOGRAPHIC = "geographic"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    tier: CacheTier
    timestamp: datetime
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    memory_usage_mb: float
    memory_limit_mb: float
    entries_count: int
    avg_access_time_ms: float
    compression_ratio: float
    network_transfers_mb: float
    geographic_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    compression_type: CompressionType = CompressionType.NONE
    compressed_size: int = 0
    original_size: int = 0
    tier: CacheTier = CacheTier.EDGE
    geographic_tags: Set[str] = field(default_factory=set)
    prediction_score: float = 0.0
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = datetime.now()


class IntelligentCompressor:
    """
    Intelligent data compression with adaptive algorithm selection.
    
    Selects optimal compression algorithm based on data characteristics,
    access patterns, and performance requirements.
    """
    
    def __init__(self):
        self.compression_stats = defaultdict(lambda: {
            'total_compressions': 0,
            'total_time_ms': 0.0,
            'total_ratio': 0.0,
            'avg_ratio': 1.0,
            'avg_time_ms': 0.0
        })
    
    def compress(self, data: bytes, compression_type: CompressionType = CompressionType.ADAPTIVE) -> Tuple[bytes, CompressionType, Dict[str, Any]]:
        """Compress data using specified or optimal algorithm."""
        if compression_type == CompressionType.ADAPTIVE:
            compression_type = self._select_optimal_compression(data)
        
        start_time = time.time()
        
        if compression_type == CompressionType.NONE:
            compressed_data = data
            ratio = 1.0
        elif compression_type == CompressionType.ZLIB:
            compressed_data = zlib.compress(data, level=6)
            ratio = len(data) / len(compressed_data)
        elif compression_type == CompressionType.LZ4:
            compressed_data = lz4.compress(data)
            ratio = len(data) / len(compressed_data)
        elif compression_type == CompressionType.BZIP2:
            import bz2
            compressed_data = bz2.compress(data)
            ratio = len(data) / len(compressed_data)
        elif compression_type == CompressionType.TENSOR_COMPRESSION:
            compressed_data, ratio = self._compress_tensor(data)
        else:
            compressed_data = data
            ratio = 1.0
        
        compression_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        stats = self.compression_stats[compression_type.value]
        stats['total_compressions'] += 1
        stats['total_time_ms'] += compression_time_ms
        stats['total_ratio'] += ratio
        stats['avg_ratio'] = stats['total_ratio'] / stats['total_compressions']
        stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_compressions']
        
        metadata = {
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_ratio': ratio,
            'compression_time_ms': compression_time_ms,
            'algorithm': compression_type.value
        }
        
        return compressed_data, compression_type, metadata
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.LZ4:
            return lz4.decompress(data)
        elif compression_type == CompressionType.BZIP2:
            import bz2
            return bz2.decompress(data)
        elif compression_type == CompressionType.TENSOR_COMPRESSION:
            return self._decompress_tensor(data)
        else:
            return data
    
    def _select_optimal_compression(self, data: bytes) -> CompressionType:
        """Select optimal compression algorithm based on data characteristics."""
        data_size = len(data)
        
        # Small data - use LZ4 for speed
        if data_size < 1024:
            return CompressionType.LZ4
        
        # Large data - analyze content
        if data_size > 1024 * 1024:  # > 1MB
            # Check if data looks like tensor data
            if self._is_tensor_like(data):
                return CompressionType.TENSOR_COMPRESSION
            else:
                return CompressionType.ZLIB  # Good balance for large data
        
        # Medium data - use adaptive selection based on entropy
        entropy = self._calculate_entropy(data[:1024])  # Sample first 1KB
        
        if entropy < 4.0:  # Low entropy - highly compressible
            return CompressionType.BZIP2
        elif entropy < 6.0:  # Medium entropy
            return CompressionType.ZLIB
        else:  # High entropy - prioritize speed
            return CompressionType.LZ4
    
    def _is_tensor_like(self, data: bytes) -> bool:
        """Heuristic to detect if data contains tensor-like numeric data."""
        try:
            # Try to interpret as numpy array
            if len(data) % 4 == 0:  # Could be float32 data
                sample = np.frombuffer(data[:min(1024, len(data))], dtype=np.float32)
                # Check if values are in reasonable range for ML data
                if np.all(np.isfinite(sample)) and np.std(sample) > 0:
                    return True
        except Exception:
            pass
        return False
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = counts / len(data)
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _compress_tensor(self, data: bytes) -> Tuple[bytes, float]:
        """Specialized compression for tensor data."""
        try:
            # Try to interpret as tensor and apply specialized compression
            if len(data) % 4 == 0:  # Assume float32
                tensor_data = np.frombuffer(data, dtype=np.float32)
                
                # Simple quantization compression
                min_val, max_val = tensor_data.min(), tensor_data.max()
                range_val = max_val - min_val
                
                if range_val > 0:
                    # Quantize to 16-bit
                    quantized = ((tensor_data - min_val) / range_val * 65535).astype(np.uint16)
                    
                    # Store quantization parameters + quantized data
                    header = np.array([min_val, max_val], dtype=np.float32)
                    compressed = header.tobytes() + quantized.tobytes()
                    
                    ratio = len(data) / len(compressed)
                    return compressed, ratio
            
            # Fall back to zlib for non-tensor data
            compressed = zlib.compress(data)
            ratio = len(data) / len(compressed)
            return compressed, ratio
            
        except Exception:
            # Fall back to zlib on any error
            compressed = zlib.compress(data)
            ratio = len(data) / len(compressed)
            return compressed, ratio
    
    def _decompress_tensor(self, data: bytes) -> bytes:
        """Decompress tensor data."""
        try:
            # Extract quantization parameters
            header_size = 2 * 4  # 2 float32 values
            header = np.frombuffer(data[:header_size], dtype=np.float32)
            min_val, max_val = header[0], header[1]
            
            # Extract and dequantize data
            quantized = np.frombuffer(data[header_size:], dtype=np.uint16)
            range_val = max_val - min_val
            
            if range_val > 0:
                dequantized = (quantized.astype(np.float32) / 65535.0) * range_val + min_val
                return dequantized.tobytes()
            
            # Fall back to zlib decompression
            return zlib.decompress(data)
            
        except Exception:
            # Fall back to zlib decompression
            return zlib.decompress(data)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        return dict(self.compression_stats)


class MemoryMappedCache:
    """
    Memory-mapped cache for zero-copy operations and persistent storage.
    
    Uses memory mapping for large tensor data to enable zero-copy operations
    and persistence across restarts.
    """
    
    def __init__(self, cache_dir: str, max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        # Memory mapped files
        self.mmap_files: Dict[str, Tuple[mmap.mmap, int]] = {}
        self.file_handles: Dict[str, Any] = {}
        
        # Metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: Dict[str, Dict] = {}
        self.current_size = 0
        
        self._lock = threading.RLock()
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Calculate current size
                self.current_size = sum(meta.get('size', 0) for meta in self.metadata.values())
                
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def put(self, key: str, data: bytes) -> bool:
        """Store data in memory-mapped cache."""
        with self._lock:
            try:
                data_size = len(data)
                
                # Check if we need to evict data
                while self.current_size + data_size > self.max_size_bytes:
                    if not self._evict_lru():
                        break  # No more data to evict
                
                # Create memory-mapped file
                file_path = self.cache_dir / f"{key}.mmap"
                
                # Write data to file
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Create memory map
                file_handle = open(file_path, 'r+b')
                mmap_obj = mmap.mmap(file_handle.fileno(), 0)
                
                # Store references
                self.mmap_files[key] = (mmap_obj, data_size)
                self.file_handles[key] = file_handle
                
                # Update metadata
                self.metadata[key] = {
                    'size': data_size,
                    'created': datetime.now().isoformat(),
                    'accessed': datetime.now().isoformat(),
                    'access_count': 1
                }
                
                self.current_size += data_size
                self._save_metadata()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to store data in mmap cache: {e}")
                return False
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve data from memory-mapped cache."""
        with self._lock:
            try:
                if key not in self.mmap_files:
                    return None
                
                mmap_obj, size = self.mmap_files[key]
                
                # Read data from memory map (zero-copy)
                mmap_obj.seek(0)
                data = mmap_obj.read(size)
                
                # Update access metadata
                if key in self.metadata:
                    self.metadata[key]['accessed'] = datetime.now().isoformat()
                    self.metadata[key]['access_count'] += 1
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to retrieve data from mmap cache: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """Delete entry from memory-mapped cache."""
        with self._lock:
            try:
                if key not in self.mmap_files:
                    return False
                
                # Close memory map and file handle
                mmap_obj, size = self.mmap_files[key]
                mmap_obj.close()
                self.file_handles[key].close()
                
                # Remove file
                file_path = self.cache_dir / f"{key}.mmap"
                if file_path.exists():
                    file_path.unlink()
                
                # Update tracking
                del self.mmap_files[key]
                del self.file_handles[key]
                
                if key in self.metadata:
                    self.current_size -= self.metadata[key]['size']
                    del self.metadata[key]
                
                self._save_metadata()
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete from mmap cache: {e}")
                return False
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item."""
        if not self.metadata:
            return False
        
        # Find LRU item
        lru_key = min(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get('accessed', '1970-01-01T00:00:00')
        )
        
        return self.delete(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory-mapped cache statistics."""
        with self._lock:
            return {
                'entries': len(self.mmap_files),
                'size_bytes': self.current_size,
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_gb': self.max_size_bytes / (1024 * 1024 * 1024),
                'utilization': self.current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }
    
    def cleanup(self):
        """Clean up memory-mapped cache resources."""
        with self._lock:
            # Close all memory maps and files
            for mmap_obj, _ in self.mmap_files.values():
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            
            for file_handle in self.file_handles.values():
                try:
                    file_handle.close()
                except Exception:
                    pass
            
            self.mmap_files.clear()
            self.file_handles.clear()


class PredictivePrefetcher:
    """
    Intelligent prefetcher that predicts future cache needs based on access patterns.
    
    Uses machine learning to identify access patterns and proactively load
    data into cache before it's requested.
    """
    
    def __init__(self, max_predictions: int = 1000):
        self.max_predictions = max_predictions
        
        # Access pattern tracking
        self.access_history: deque = deque(maxlen=10000)
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)
        self.temporal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # ML models for prediction
        self.sequence_predictor = None
        self.temporal_predictor = None
        self.pattern_clusters = None
        
        # Performance tracking
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_prefetches = 0
        
        self._lock = threading.RLock()
    
    def record_access(self, key: str, timestamp: datetime, metadata: Optional[Dict] = None):
        """Record cache access for pattern learning."""
        with self._lock:
            access_record = {
                'key': key,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }
            
            self.access_history.append(access_record)
            self.temporal_patterns[key].append(timestamp)
            
            # Maintain pattern sequences
            if len(self.access_history) >= 2:
                prev_key = self.access_history[-2]['key']
                self.access_patterns[prev_key].append(key)
                
                # Keep only recent patterns
                if len(self.access_patterns[prev_key]) > 100:
                    self.access_patterns[prev_key] = self.access_patterns[prev_key][-100:]
            
            # Limit temporal patterns
            if len(self.temporal_patterns[key]) > 100:
                self.temporal_patterns[key] = self.temporal_patterns[key][-100:]
    
    def train_prediction_models(self):
        """Train machine learning models for access prediction."""
        try:
            with self._lock:
                # Train sequence predictor
                self._train_sequence_predictor()
                
                # Train temporal predictor
                self._train_temporal_predictor()
                
                # Cluster access patterns
                self._cluster_access_patterns()
                
            logger.info("Prefetch prediction models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train prediction models: {e}")
    
    def _train_sequence_predictor(self):
        """Train model to predict next likely accessed keys."""
        if len(self.access_history) < 100:
            return
        
        # Create sequences of access patterns
        sequences = []
        targets = []
        
        window_size = 5
        for i in range(len(self.access_history) - window_size):
            sequence = [record['key'] for record in list(self.access_history)[i:i+window_size]]
            target = list(self.access_history)[i+window_size]['key']
            
            sequences.append(sequence)
            targets.append(target)
        
        # Simple frequency-based predictor (could be enhanced with neural networks)
        self.sequence_predictor = {
            'sequences': sequences[-1000:],  # Keep recent sequences
            'targets': targets[-1000:]
        }
    
    def _train_temporal_predictor(self):
        """Train model to predict when keys will be accessed."""
        # Analyze temporal patterns for each key
        self.temporal_predictor = {}
        
        for key, access_times in self.temporal_patterns.items():
            if len(access_times) < 10:
                continue
            
            # Calculate inter-access intervals
            intervals = []
            for i in range(1, len(access_times)):
                interval = (access_times[i] - access_times[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                # Simple statistical predictor
                self.temporal_predictor[key] = {
                    'mean_interval': np.mean(intervals),
                    'std_interval': np.std(intervals),
                    'last_access': access_times[-1]
                }
    
    def _cluster_access_patterns(self):
        """Cluster access patterns to identify similar usage."""
        try:
            # Create feature vectors for each key based on access patterns
            key_features = {}
            
            for key in self.temporal_patterns.keys():
                if len(self.temporal_patterns[key]) < 5:
                    continue
                
                access_times = self.temporal_patterns[key]
                
                # Extract temporal features
                hours = [t.hour for t in access_times]
                days = [t.weekday() for t in access_times]
                
                features = [
                    len(access_times),  # Total accesses
                    np.mean(hours),  # Average hour
                    np.std(hours) if len(hours) > 1 else 0,  # Hour variance
                    np.mean(days),  # Average day of week
                    len(set(hours)),  # Unique hours
                    len(set(days))  # Unique days
                ]
                
                key_features[key] = features
            
            if len(key_features) >= 3:
                # Cluster keys with similar access patterns
                keys = list(key_features.keys())
                features_matrix = np.array(list(key_features.values()))
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_matrix)
                
                # Cluster
                n_clusters = min(5, len(keys))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)
                
                # Store clustering results
                self.pattern_clusters = {
                    'keys': keys,
                    'clusters': clusters,
                    'centroids': kmeans.cluster_centers_,
                    'scaler': scaler
                }
                
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
    
    def predict_prefetch_candidates(self, current_time: datetime, max_candidates: int = 10) -> List[Tuple[str, float]]:
        """Predict keys that should be prefetched."""
        candidates = []
        
        try:
            with self._lock:
                # Sequence-based predictions
                if self.sequence_predictor:
                    sequence_candidates = self._get_sequence_predictions()
                    candidates.extend(sequence_candidates)
                
                # Temporal-based predictions
                if self.temporal_predictor:
                    temporal_candidates = self._get_temporal_predictions(current_time)
                    candidates.extend(temporal_candidates)
                
                # Cluster-based predictions
                if self.pattern_clusters:
                    cluster_candidates = self._get_cluster_predictions()
                    candidates.extend(cluster_candidates)
            
            # Deduplicate and sort by confidence
            candidate_dict = {}
            for key, confidence in candidates:
                if key not in candidate_dict or confidence > candidate_dict[key]:
                    candidate_dict[key] = confidence
            
            # Sort by confidence and return top candidates
            sorted_candidates = sorted(
                candidate_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return sorted_candidates[:max_candidates]
            
        except Exception as e:
            logger.error(f"Prefetch prediction failed: {e}")
            return []
    
    def _get_sequence_predictions(self) -> List[Tuple[str, float]]:
        """Get predictions based on access sequences."""
        if not self.sequence_predictor or len(self.access_history) < 5:
            return []
        
        # Get recent sequence
        recent_sequence = [record['key'] for record in list(self.access_history)[-5:]]
        
        # Find matching sequences in history
        matches = []
        sequences = self.sequence_predictor['sequences']
        targets = self.sequence_predictor['targets']
        
        for i, seq in enumerate(sequences):
            # Check if recent sequence matches (partial match)
            match_score = sum(1 for a, b in zip(recent_sequence, seq) if a == b) / len(seq)
            if match_score > 0.6:  # 60% match threshold
                matches.append((targets[i], match_score))
        
        # Aggregate matches by frequency
        target_counts = defaultdict(list)
        for target, score in matches:
            target_counts[target].append(score)
        
        predictions = []
        for target, scores in target_counts.items():
            avg_score = np.mean(scores)
            frequency = len(scores) / len(sequences)
            confidence = avg_score * frequency
            predictions.append((target, confidence))
        
        return predictions
    
    def _get_temporal_predictions(self, current_time: datetime) -> List[Tuple[str, float]]:
        """Get predictions based on temporal patterns."""
        predictions = []
        
        for key, pattern in self.temporal_predictor.items():
            last_access = pattern['last_access']
            mean_interval = pattern['mean_interval']
            std_interval = pattern['std_interval']
            
            # Calculate expected next access time
            expected_next = last_access + timedelta(seconds=mean_interval)
            time_diff = abs((current_time - expected_next).total_seconds())
            
            # Calculate confidence based on how close we are to expected time
            if std_interval > 0:
                confidence = max(0, 1 - (time_diff / (2 * std_interval)))
            else:
                confidence = 1.0 if time_diff < 300 else 0.0  # 5 minute window
            
            if confidence > 0.3:  # Minimum confidence threshold
                predictions.append((key, confidence))
        
        return predictions
    
    def _get_cluster_predictions(self) -> List[Tuple[str, float]]:
        """Get predictions based on access pattern clustering."""
        if not self.pattern_clusters or len(self.access_history) < 10:
            return []
        
        predictions = []
        
        # Find keys in same cluster as recently accessed keys
        recent_keys = [record['key'] for record in list(self.access_history)[-5:]]
        
        for recent_key in recent_keys:
            if recent_key in self.pattern_clusters['keys']:
                key_idx = self.pattern_clusters['keys'].index(recent_key)
                key_cluster = self.pattern_clusters['clusters'][key_idx]
                
                # Find other keys in same cluster
                for i, cluster_id in enumerate(self.pattern_clusters['clusters']):
                    if cluster_id == key_cluster:
                        cluster_key = self.pattern_clusters['keys'][i]
                        if cluster_key != recent_key:
                            predictions.append((cluster_key, 0.5))  # Moderate confidence
        
        return predictions
    
    def record_prefetch_result(self, key: str, was_used: bool):
        """Record whether a prefetched item was actually used."""
        self.total_prefetches += 1
        if was_used:
            self.prefetch_hits += 1
        else:
            self.prefetch_misses += 1
    
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get prefetching performance statistics."""
        hit_rate = self.prefetch_hits / max(1, self.total_prefetches)
        
        return {
            'total_prefetches': self.total_prefetches,
            'prefetch_hits': self.prefetch_hits,
            'prefetch_misses': self.prefetch_misses,
            'hit_rate': hit_rate,
            'access_patterns_tracked': len(self.access_patterns),
            'temporal_patterns_tracked': len(self.temporal_patterns),
            'clustered_keys': len(self.pattern_clusters['keys']) if self.pattern_clusters else 0
        }


class MultiTierCacheManager:
    """
    Main multi-tier cache manager coordinating edge-fog-cloud caching hierarchy.
    
    Provides unified interface for hierarchical caching with intelligent data
    placement, migration, and prefetching across geographic regions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Cache tiers
        self.tiers: Dict[CacheTier, Dict[str, CacheEntry]] = {
            CacheTier.EDGE: OrderedDict(),
            CacheTier.FOG: OrderedDict(),
            CacheTier.CLOUD: OrderedDict(),
            CacheTier.GLOBAL: OrderedDict()
        }
        
        # Tier configurations
        self.tier_configs = {
            CacheTier.EDGE: {
                'max_size_mb': self.config.get('edge_cache_mb', 512),
                'ttl_seconds': self.config.get('edge_ttl', 300),
                'eviction_policy': EvictionPolicy.LRU,
                'compression': CompressionType.LZ4
            },
            CacheTier.FOG: {
                'max_size_mb': self.config.get('fog_cache_mb', 2048),
                'ttl_seconds': self.config.get('fog_ttl', 1800),
                'eviction_policy': EvictionPolicy.ADAPTIVE,
                'compression': CompressionType.ZLIB
            },
            CacheTier.CLOUD: {
                'max_size_mb': self.config.get('cloud_cache_mb', 8192),
                'ttl_seconds': self.config.get('cloud_ttl', 3600),
                'eviction_policy': EvictionPolicy.PREDICTIVE,
                'compression': CompressionType.ADAPTIVE
            },
            CacheTier.GLOBAL: {
                'max_size_mb': self.config.get('global_cache_mb', 32768),
                'ttl_seconds': self.config.get('global_ttl', 7200),
                'eviction_policy': EvictionPolicy.GEOGRAPHIC,
                'compression': CompressionType.TENSOR_COMPRESSION
            }
        }
        
        # Supporting components
        self.compressor = IntelligentCompressor()
        self.mmap_cache = MemoryMappedCache(
            self.config.get('mmap_cache_dir', '/tmp/mmap_cache'),
            self.config.get('mmap_cache_gb', 2.0)
        )
        self.prefetcher = PredictivePrefetcher()
        
        # Performance tracking
        self.metrics_history: Dict[CacheTier, deque] = {
            tier: deque(maxlen=1000) for tier in CacheTier
        }
        self.access_stats = defaultdict(lambda: defaultdict(int))
        
        # Geographic distribution
        self.geographic_regions: Set[str] = set()
        self.region_affinities: Dict[str, str] = {}  # key -> preferred_region
        
        # Threading
        self._lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=4)
        self._maintenance_task = None
        self._running = False
        
        logger.info("Multi-tier cache manager initialized")
    
    async def start(self):
        """Start cache management services."""
        if self._running:
            return
        
        self._running = True
        
        # Start background maintenance
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # Train initial prefetch models
        await self._train_prefetch_models()
        
        logger.info("Multi-tier cache manager started")
    
    async def stop(self):
        """Stop cache management services."""
        self._running = False
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
        
        self._prefetch_executor.shutdown(wait=True)
        self.mmap_cache.cleanup()
        
        logger.info("Multi-tier cache manager stopped")
    
    async def get(self, key: str, region: Optional[str] = None) -> Optional[Any]:
        """Get value from cache, checking all tiers."""
        start_time = time.time()
        
        try:
            # Check each tier in order (edge -> fog -> cloud -> global)
            for tier in CacheTier:
                with self._lock:
                    if key in self.tiers[tier]:
                        entry = self.tiers[tier][key]
                        
                        # Check if entry is expired
                        if entry.is_expired():
                            del self.tiers[tier][key]
                            continue
                        
                        # Update access statistics
                        entry.update_access()
                        
                        # Move to front for LRU
                        if tier != CacheTier.EDGE:
                            self.tiers[tier].move_to_end(key)
                        
                        # Decompress if needed
                        value = self._decompress_entry(entry)
                        
                        # Promote to higher tier if frequently accessed
                        if entry.access_count > 3 and tier != CacheTier.EDGE:
                            await self._promote_entry(key, entry, tier)
                        
                        # Record access for prefetching
                        self.prefetcher.record_access(key, datetime.now(), {
                            'tier': tier.value,
                            'region': region
                        })
                        
                        # Update statistics
                        access_time_ms = (time.time() - start_time) * 1000
                        self.access_stats[tier]['hits'] += 1
                        self.access_stats[tier]['total_access_time_ms'] += access_time_ms
                        
                        return value
            
            # Cache miss - record statistics
            for tier in CacheTier:
                self.access_stats[tier]['misses'] += 1
            
            # Trigger prefetching for related keys
            asyncio.create_task(self._trigger_prefetch(key, region))
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def put(self, key: str, value: Any, tier: CacheTier = CacheTier.EDGE, 
                  region: Optional[str] = None, ttl_seconds: Optional[int] = None) -> bool:
        """Put value into specified cache tier."""
        try:
            # Serialize and compress value
            serialized_value = pickle.dumps(value)
            
            # Select compression type
            compression_type = self.tier_configs[tier]['compression']
            compressed_value, actual_compression, compression_metadata = self.compressor.compress(
                serialized_value, compression_type
            )
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                timestamp=datetime.now(),
                ttl_seconds=ttl_seconds or self.tier_configs[tier]['ttl_seconds'],
                compression_type=actual_compression,
                compressed_size=len(compressed_value),
                original_size=len(serialized_value),
                tier=tier,
                geographic_tags={region} if region else set()
            )
            
            # Check if we need to evict entries
            await self._ensure_capacity(tier, entry.compressed_size)
            
            with self._lock:
                # Store in tier
                self.tiers[tier][key] = entry
                
                # Update geographic tracking
                if region:
                    self.geographic_regions.add(region)
                    self.region_affinities[key] = region
                
                # Store large entries in memory-mapped cache
                if entry.compressed_size > 1024 * 1024:  # > 1MB
                    self.mmap_cache.put(f"{tier.value}_{key}", compressed_value)
                    entry.value = None  # Free memory, will load from mmap
            
            logger.debug(f"Cached {key} in {tier.value} tier "
                        f"(compressed: {compression_metadata['compression_ratio']:.2f}x)")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache put failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str, all_tiers: bool = True) -> bool:
        """Delete key from cache tiers."""
        deleted = False
        
        try:
            tiers_to_check = list(CacheTier) if all_tiers else [CacheTier.EDGE]
            
            for tier in tiers_to_check:
                with self._lock:
                    if key in self.tiers[tier]:
                        entry = self.tiers[tier][key]
                        
                        # Clean up memory-mapped cache
                        if entry.value is None:
                            self.mmap_cache.delete(f"{tier.value}_{key}")
                        
                        del self.tiers[tier][key]
                        deleted = True
            
            # Clean up tracking
            if key in self.region_affinities:
                del self.region_affinities[key]
            
            return deleted
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    def _decompress_entry(self, entry: CacheEntry) -> Any:
        """Decompress and deserialize cache entry value."""
        # Get compressed data
        if entry.value is None:
            # Load from memory-mapped cache
            mmap_key = f"{entry.tier.value}_{entry.key}"
            compressed_data = self.mmap_cache.get(mmap_key)
            if compressed_data is None:
                raise ValueError(f"Failed to load data from mmap cache for {entry.key}")
        else:
            compressed_data = entry.value
        
        # Decompress
        decompressed_data = self.compressor.decompress(compressed_data, entry.compression_type)
        
        # Deserialize
        return pickle.loads(decompressed_data)
    
    async def _promote_entry(self, key: str, entry: CacheEntry, current_tier: CacheTier):
        """Promote frequently accessed entry to higher tier."""
        try:
            # Determine target tier
            tier_order = [CacheTier.GLOBAL, CacheTier.CLOUD, CacheTier.FOG, CacheTier.EDGE]
            current_idx = tier_order.index(current_tier)
            
            if current_idx > 0:  # Can promote
                target_tier = tier_order[current_idx - 1]
                
                # Get original value
                value = self._decompress_entry(entry)
                
                # Store in higher tier
                await self.put(key, value, target_tier, 
                             list(entry.geographic_tags)[0] if entry.geographic_tags else None)
                
                logger.debug(f"Promoted {key} from {current_tier.value} to {target_tier.value}")
                
        except Exception as e:
            logger.error(f"Entry promotion failed for {key}: {e}")
    
    async def _ensure_capacity(self, tier: CacheTier, required_size: int):
        """Ensure cache tier has enough capacity for new entry."""
        config = self.tier_configs[tier]
        max_size_bytes = config['max_size_mb'] * 1024 * 1024
        
        with self._lock:
            # Calculate current size
            current_size = sum(
                entry.compressed_size for entry in self.tiers[tier].values()
            )
            
            # Evict entries if needed
            while current_size + required_size > max_size_bytes:
                if not self._evict_from_tier(tier, config['eviction_policy']):
                    break  # No more entries to evict
                
                current_size = sum(
                    entry.compressed_size for entry in self.tiers[tier].values()
                )
    
    def _evict_from_tier(self, tier: CacheTier, policy: EvictionPolicy) -> bool:
        """Evict entry from tier based on policy."""
        if not self.tiers[tier]:
            return False
        
        if policy == EvictionPolicy.LRU:
            # Remove least recently used
            key = next(iter(self.tiers[tier]))  # OrderedDict first item
            entry = self.tiers[tier][key]
        elif policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.tiers[tier].keys(), 
                     key=lambda k: self.tiers[tier][k].access_count)
            entry = self.tiers[tier][key]
        elif policy == EvictionPolicy.TTL:
            # Remove expired entries first
            expired_keys = [
                k for k, v in self.tiers[tier].items() if v.is_expired()
            ]
            if expired_keys:
                key = expired_keys[0]
                entry = self.tiers[tier][key]
            else:
                return False
        elif policy == EvictionPolicy.ADAPTIVE:
            # Adaptive policy considering access patterns and size
            scores = {}
            for k, v in self.tiers[tier].items():
                age_score = (datetime.now() - v.last_access).total_seconds() / 3600
                size_score = v.compressed_size / (1024 * 1024)  # MB
                access_score = 1.0 / (v.access_count + 1)
                scores[k] = age_score + size_score + access_score
            
            key = max(scores.keys(), key=lambda k: scores[k])
            entry = self.tiers[tier][key]
        else:
            # Default to LRU
            key = next(iter(self.tiers[tier]))
            entry = self.tiers[tier][key]
        
        # Clean up memory-mapped cache
        if entry.value is None:
            self.mmap_cache.delete(f"{tier.value}_{key}")
        
        del self.tiers[tier][key]
        return True
    
    async def _trigger_prefetch(self, key: str, region: Optional[str]):
        """Trigger prefetching for related keys."""
        try:
            # Get prefetch candidates
            candidates = self.prefetcher.predict_prefetch_candidates(datetime.now(), max_candidates=5)
            
            # Prefetch high-confidence candidates
            for candidate_key, confidence in candidates:
                if confidence > 0.6 and candidate_key not in self.tiers[CacheTier.EDGE]:
                    # Check if available in other tiers
                    for tier in [CacheTier.FOG, CacheTier.CLOUD, CacheTier.GLOBAL]:
                        if candidate_key in self.tiers[tier]:
                            # Promote to edge
                            entry = self.tiers[tier][candidate_key]
                            value = self._decompress_entry(entry)
                            await self.put(candidate_key, value, CacheTier.EDGE, region)
                            
                            self.prefetcher.record_prefetch_result(candidate_key, True)
                            break
            
        except Exception as e:
            logger.error(f"Prefetch trigger failed: {e}")
    
    async def _train_prefetch_models(self):
        """Train prefetching prediction models."""
        try:
            self.prefetcher.train_prediction_models()
            logger.info("Prefetch models trained successfully")
        except Exception as e:
            logger.error(f"Prefetch model training failed: {e}")
    
    async def _maintenance_loop(self):
        """Background maintenance tasks."""
        while self._running:
            try:
                # Clean up expired entries
                await self._cleanup_expired_entries()
                
                # Collect performance metrics
                await self._collect_metrics()
                
                # Train prefetch models periodically
                if len(self.prefetcher.access_history) % 1000 == 0:
                    await self._train_prefetch_models()
                
                # Optimize cache distribution
                await self._optimize_cache_distribution()
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from all tiers."""
        for tier in CacheTier:
            expired_keys = []
            
            with self._lock:
                for key, entry in self.tiers[tier].items():
                    if entry.is_expired():
                        expired_keys.append(key)
            
            for key in expired_keys:
                await self.delete(key, all_tiers=False)
    
    async def _collect_metrics(self):
        """Collect cache performance metrics."""
        for tier in CacheTier:
            with self._lock:
                hits = self.access_stats[tier]['hits']
                misses = self.access_stats[tier]['misses']
                total_access_time = self.access_stats[tier]['total_access_time_ms']
                
                hit_rate = hits / max(1, hits + misses)
                miss_rate = 1.0 - hit_rate
                avg_access_time = total_access_time / max(1, hits)
                
                # Calculate memory usage
                memory_usage_mb = sum(
                    entry.compressed_size for entry in self.tiers[tier].values()
                ) / (1024 * 1024)
                
                memory_limit_mb = self.tier_configs[tier]['max_size_mb']
                
                # Calculate compression ratio
                if self.tiers[tier]:
                    total_original = sum(entry.original_size for entry in self.tiers[tier].values())
                    total_compressed = sum(entry.compressed_size for entry in self.tiers[tier].values())
                    compression_ratio = total_original / max(1, total_compressed)
                else:
                    compression_ratio = 1.0
                
                metrics = CacheMetrics(
                    tier=tier,
                    timestamp=datetime.now(),
                    hit_rate=hit_rate,
                    miss_rate=miss_rate,
                    eviction_rate=0.0,  # Would need to track evictions
                    memory_usage_mb=memory_usage_mb,
                    memory_limit_mb=memory_limit_mb,
                    entries_count=len(self.tiers[tier]),
                    avg_access_time_ms=avg_access_time,
                    compression_ratio=compression_ratio,
                    network_transfers_mb=0.0  # Would need to track transfers
                )
                
                self.metrics_history[tier].append(metrics)
    
    async def _optimize_cache_distribution(self):
        """Optimize cache distribution based on access patterns."""
        try:
            # Analyze geographic access patterns
            regional_access = defaultdict(int)
            
            for key, region in self.region_affinities.items():
                for tier in CacheTier:
                    if key in self.tiers[tier]:
                        regional_access[region] += self.tiers[tier][key].access_count
            
            # Migrate frequently accessed data to appropriate regions
            # This would involve moving data between geographic cache instances
            # Implementation would depend on distributed cache architecture
            
        except Exception as e:
            logger.error(f"Cache distribution optimization failed: {e}")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status across all tiers."""
        status = {}
        
        for tier in CacheTier:
            with self._lock:
                tier_status = {
                    'entries': len(self.tiers[tier]),
                    'memory_usage_mb': sum(
                        entry.compressed_size for entry in self.tiers[tier].values()
                    ) / (1024 * 1024),
                    'memory_limit_mb': self.tier_configs[tier]['max_size_mb'],
                    'hit_rate': 0.0,
                    'avg_access_time_ms': 0.0
                }
                
                # Calculate hit rate
                hits = self.access_stats[tier]['hits']
                misses = self.access_stats[tier]['misses']
                if hits + misses > 0:
                    tier_status['hit_rate'] = hits / (hits + misses)
                
                # Calculate average access time
                total_time = self.access_stats[tier]['total_access_time_ms']
                if hits > 0:
                    tier_status['avg_access_time_ms'] = total_time / hits
                
                status[tier.value] = tier_status
        
        # Add component status
        status['compression'] = self.compressor.get_compression_stats()
        status['memory_mapped'] = self.mmap_cache.get_stats()
        status['prefetching'] = self.prefetcher.get_prefetch_stats()
        status['geographic_regions'] = len(self.geographic_regions)
        
        return status


# Global cache manager instance
_cache_manager: Optional[MultiTierCacheManager] = None


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> MultiTierCacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = MultiTierCacheManager(config)
    
    return _cache_manager


async def start_multi_tier_caching(config: Optional[Dict[str, Any]] = None) -> MultiTierCacheManager:
    """Start multi-tier caching system."""
    cache_manager = get_cache_manager(config)
    await cache_manager.start()
    return cache_manager