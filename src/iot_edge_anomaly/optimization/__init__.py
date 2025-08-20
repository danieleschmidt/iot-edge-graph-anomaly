"""
Performance Optimization Module for IoT Edge Anomaly Detection.

This module provides advanced optimization features for scalable deployment:
- Model quantization and compression
- Caching and memoization strategies
- Batch processing and pipeline optimization
- GPU acceleration and memory management
- Auto-scaling and load balancing
- Performance profiling and monitoring
"""

from .model_optimizer import ModelOptimizer, OptimizationStrategy
from .caching_system import CachingSystem, CacheStrategy
from .batch_processor import BatchProcessor, BatchingStrategy
from .gpu_accelerator import GPUAccelerator, DeviceManager
from .auto_scaler import AutoScaler, ScalingPolicy
from .performance_profiler import PerformanceProfiler, ProfileMetrics

__all__ = [
    'ModelOptimizer',
    'OptimizationStrategy',
    'CachingSystem', 
    'CacheStrategy',
    'BatchProcessor',
    'BatchingStrategy',
    'GPUAccelerator',
    'DeviceManager',
    'AutoScaler',
    'ScalingPolicy',
    'PerformanceProfiler',
    'ProfileMetrics'
]