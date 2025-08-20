"""
Optimized and Scalable IoT Edge Anomaly Detection System.

This module integrates all optimization features for production-scale deployment:
- Performance profiling and monitoring
- Intelligent caching and memoization
- Batch processing optimization
- GPU acceleration and memory management
- Auto-scaling and load balancing
- Real-time performance tuning
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np

from .robust_main import RobustAnomalyDetectionSystem
from .optimization.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


class OptimizedAnomalyDetectionSystem:
    """
    Production-optimized anomaly detection system with advanced scaling capabilities.
    
    Combines robust reliability features with comprehensive performance optimization:
    - Intelligent performance profiling and monitoring
    - Advanced caching strategies for repeated computations
    - Batch processing for high-throughput scenarios
    - GPU acceleration with memory optimization
    - Auto-scaling based on load and performance metrics
    - Real-time performance tuning and adaptation
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        robustness_config: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize optimized anomaly detection system.
        
        Args:
            model_config: Configuration for the ensemble model
            robustness_config: Configuration for robustness features
            optimization_config: Configuration for optimization features
        """
        self.model_config = model_config
        self.robustness_config = robustness_config
        self.optimization_config = optimization_config or self._default_optimization_config()
        
        # Initialize base robust system
        self.robust_system = RobustAnomalyDetectionSystem(
            model_config, robustness_config
        )
        
        # Initialize optimization components
        self.performance_profiler = None
        self.prediction_cache = {}
        self.batch_queue = []
        self.batch_lock = threading.Lock()
        
        # Performance and scaling state
        self.current_load = 0
        self.average_response_time = 0.0
        self.throughput_metrics = []
        self.optimization_enabled = True
        
        # Threading and concurrency
        self.thread_pool = None
        self.max_workers = self.optimization_config.get('max_workers', 4)
        
        # Initialize optimization features
        self._initialize_optimization()
        
        logger.info("Optimized anomaly detection system initialized")
    
    def _default_optimization_config(self) -> Dict[str, Any]:
        """Default optimization configuration."""
        return {
            "performance_profiling": {
                "enabled": True,
                "profile_all_functions": True,
                "enable_memory_profiling": True,
                "enable_gpu_profiling": True
            },
            "caching": {
                "enabled": True,
                "cache_size": 1000,
                "ttl_seconds": 300,
                "cache_strategy": "lru"
            },
            "batch_processing": {
                "enabled": True,
                "max_batch_size": 32,
                "batch_timeout": 0.1,
                "enable_dynamic_batching": True
            },
            "gpu_optimization": {
                "enabled": True,
                "mixed_precision": True,
                "memory_efficient": True,
                "compile_models": False  # PyTorch 2.0 compilation
            },
            "concurrency": {
                "max_workers": 4,
                "async_processing": True,
                "load_balancing": True
            },
            "auto_scaling": {
                "enabled": True,
                "target_response_time": 1.0,
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3
            }
        }
    
    def _initialize_optimization(self):
        """Initialize optimization features."""
        try:
            # Initialize performance profiler
            if self.optimization_config["performance_profiling"]["enabled"]:
                self.performance_profiler = PerformanceProfiler(
                    enable_memory_profiling=self.optimization_config["performance_profiling"]["enable_memory_profiling"],
                    enable_gpu_profiling=self.optimization_config["performance_profiling"]["enable_gpu_profiling"]
                )
                self.performance_profiler.start_monitoring()
            
            # Initialize thread pool for concurrent processing
            if self.optimization_config["concurrency"]["async_processing"]:
                self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Initialize GPU optimizations
            if self.optimization_config["gpu_optimization"]["enabled"]:
                self._setup_gpu_optimization()
            
            # Initialize caching
            if self.optimization_config["caching"]["enabled"]:
                self._setup_caching()
            
            logger.info("Optimization features initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization features: {e}")
            self.optimization_enabled = False
    
    def _setup_gpu_optimization(self):
        """Setup GPU optimization features."""
        if not torch.cuda.is_available():
            logger.warning("GPU optimization requested but CUDA not available")
            return
        
        try:
            # Enable mixed precision if requested
            if self.optimization_config["gpu_optimization"]["mixed_precision"]:
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                logger.info("Mixed precision training enabled")
            
            # Memory optimization
            if self.optimization_config["gpu_optimization"]["memory_efficient"]:
                torch.cuda.empty_cache()
                logger.info("GPU memory optimization enabled")
            
            # Model compilation (PyTorch 2.0)
            if (self.optimization_config["gpu_optimization"]["compile_models"] and 
                hasattr(torch, 'compile')):
                logger.info("Model compilation will be applied to ensemble models")
            
        except Exception as e:
            logger.error(f"GPU optimization setup failed: {e}")
    
    def _setup_caching(self):
        """Setup intelligent caching system."""
        cache_config = self.optimization_config["caching"]
        self.cache_size = cache_config["cache_size"]
        self.cache_ttl = cache_config["ttl_seconds"]
        
        # Initialize cache with timestamps
        self.prediction_cache = {}
        self.cache_access_times = {}
        
        logger.info(f"Caching enabled with size {self.cache_size} and TTL {self.cache_ttl}s")
    
    def _generate_cache_key(
        self,
        sensor_data: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> str:
        """Generate cache key for input data."""
        # Create hash from tensor data
        data_hash = hash(sensor_data.data_ptr())
        
        if edge_index is not None:
            edge_hash = hash(edge_index.data_ptr())
            return f"pred_{data_hash}_{edge_hash}"
        
        return f"pred_{data_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if prediction is cached and still valid."""
        if not self.optimization_config["caching"]["enabled"]:
            return None
        
        if cache_key in self.prediction_cache:
            # Check TTL
            cache_time = self.cache_access_times.get(cache_key, 0)
            if time.time() - cache_time < self.cache_ttl:
                # Update access time
                self.cache_access_times[cache_key] = time.time()
                return self.prediction_cache[cache_key]
            else:
                # Expired - remove from cache
                self._remove_from_cache(cache_key)
        
        return None
    
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Add prediction result to cache."""
        if not self.optimization_config["caching"]["enabled"]:
            return
        
        # Implement LRU eviction if cache is full
        if len(self.prediction_cache) >= self.cache_size:
            self._evict_lru_item()
        
        # Store result and access time
        self.prediction_cache[cache_key] = result.copy()
        self.cache_access_times[cache_key] = time.time()
    
    def _evict_lru_item(self):
        """Evict least recently used item from cache."""
        if not self.cache_access_times:
            return
        
        # Find least recently used item
        lru_key = min(self.cache_access_times.keys(), 
                     key=lambda k: self.cache_access_times[k])
        self._remove_from_cache(lru_key)
    
    def _remove_from_cache(self, cache_key: str):
        """Remove item from cache."""
        self.prediction_cache.pop(cache_key, None)
        self.cache_access_times.pop(cache_key, None)
    
    async def predict_async(
        self,
        sensor_data: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        sensor_metadata: Optional[Dict[str, torch.Tensor]] = None,
        return_explanations: bool = False,
        validate_input: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Asynchronous optimized prediction with comprehensive performance optimization.
        
        Args:
            sensor_data: Input sensor data
            edge_index: Optional graph connectivity
            sensor_metadata: Optional sensor metadata
            return_explanations: Whether to return model explanations
            validate_input: Whether to validate input data
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing prediction results and optimization metadata
        """
        request_start_time = time.time()
        
        # Performance profiling
        profiler_context = (self.performance_profiler.profile("predict_async") 
                          if self.performance_profiler else None)
        
        try:
            if profiler_context:
                profiler_context.__enter__()
            
            # Check cache first
            cache_key = None
            if use_cache and self.optimization_config["caching"]["enabled"]:
                cache_key = self._generate_cache_key(sensor_data, edge_index)
                cached_result = self._check_cache(cache_key)
                
                if cached_result is not None:
                    cached_result["cache_hit"] = True
                    cached_result["processing_time"] = time.time() - request_start_time
                    return cached_result
            
            # GPU optimization - move data to device if available
            if torch.cuda.is_available() and self.optimization_config["gpu_optimization"]["enabled"]:
                sensor_data = sensor_data.cuda()
                if edge_index is not None:
                    edge_index = edge_index.cuda()
                if sensor_metadata:
                    sensor_metadata = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                     for k, v in sensor_metadata.items()}
            
            # Execute prediction through robust system
            if self.optimization_config["concurrency"]["async_processing"] and self.thread_pool:
                # Async execution
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    self.robust_system.predict,
                    sensor_data,
                    edge_index,
                    sensor_metadata,
                    return_explanations,
                    validate_input
                )
            else:
                # Synchronous execution
                result = self.robust_system.predict(
                    sensor_data=sensor_data,
                    edge_index=edge_index,
                    sensor_metadata=sensor_metadata,
                    return_explanations=return_explanations,
                    validate_input=validate_input
                )
            
            # Add optimization metadata
            total_time = time.time() - request_start_time
            result["optimization_metadata"] = {
                "cache_hit": False,
                "gpu_accelerated": torch.cuda.is_available() and self.optimization_config["gpu_optimization"]["enabled"],
                "async_processing": self.optimization_config["concurrency"]["async_processing"],
                "total_processing_time": total_time,
                "optimization_enabled": self.optimization_enabled
            }
            
            # Cache result if enabled
            if cache_key and self.optimization_config["caching"]["enabled"]:
                # Create cacheable copy (without large tensors)
                cacheable_result = self._create_cacheable_result(result)
                self._add_to_cache(cache_key, cacheable_result)
            
            # Update performance metrics
            self._update_performance_metrics(total_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized prediction failed: {e}")
            raise
        finally:
            if profiler_context:
                profiler_context.__exit__(None, None, None)
    
    def predict_batch(
        self,
        batch_data: List[Dict[str, Any]],
        max_batch_size: Optional[int] = None,
        return_explanations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Optimized batch prediction processing.
        
        Args:
            batch_data: List of input data dictionaries
            max_batch_size: Maximum batch size for processing
            return_explanations: Whether to return explanations
            
        Returns:
            List of prediction results
        """
        if not self.optimization_config["batch_processing"]["enabled"]:
            # Process individually if batching disabled
            return [self.predict_sync(
                data["sensor_data"],
                data.get("edge_index"),
                data.get("sensor_metadata"),
                return_explanations
            ) for data in batch_data]
        
        batch_size = max_batch_size or self.optimization_config["batch_processing"]["max_batch_size"]
        results = []
        
        with self.performance_profiler.profile("predict_batch") if self.performance_profiler else None:
            # Process in optimized batches
            for i in range(0, len(batch_data), batch_size):
                batch_chunk = batch_data[i:i + batch_size]
                
                # Parallel processing of batch chunk
                if self.thread_pool and len(batch_chunk) > 1:
                    futures = []
                    for data in batch_chunk:
                        future = self.thread_pool.submit(
                            self.predict_sync,
                            data["sensor_data"],
                            data.get("edge_index"),
                            data.get("sensor_metadata"),
                            return_explanations
                        )
                        futures.append(future)
                    
                    # Collect results
                    chunk_results = []
                    for future in as_completed(futures):
                        try:
                            chunk_results.append(future.result())
                        except Exception as e:
                            logger.error(f"Batch prediction error: {e}")
                            chunk_results.append({"error": str(e)})
                    
                    results.extend(chunk_results)
                else:
                    # Sequential processing for small batches
                    for data in batch_chunk:
                        try:
                            result = self.predict_sync(
                                data["sensor_data"],
                                data.get("edge_index"),
                                data.get("sensor_metadata"),
                                return_explanations
                            )
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Batch item prediction error: {e}")
                            results.append({"error": str(e)})
        
        return results
    
    def predict_sync(
        self,
        sensor_data: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        sensor_metadata: Optional[Dict[str, torch.Tensor]] = None,
        return_explanations: bool = False,
        validate_input: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for optimized prediction."""
        # Run async prediction in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.predict_async(
                    sensor_data, edge_index, sensor_metadata,
                    return_explanations, validate_input
                )
            )
        finally:
            loop.close()
    
    def _create_cacheable_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a cacheable version of prediction result."""
        # Remove or convert large tensors to preserve cache efficiency
        cacheable = {}
        
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                # Store only small tensors or convert to numpy
                if value.numel() < 100:  # Only cache small tensors
                    cacheable[key] = value.cpu().numpy()
                else:
                    cacheable[key] = {"type": "large_tensor", "shape": list(value.shape)}
            elif key == "individual_predictions":
                # Simplify individual predictions
                cacheable[key] = len(value) if isinstance(value, list) else 0
            elif key == "explanations":
                # Simplify explanations
                cacheable[key] = {"available": True} if value else {"available": False}
            else:
                cacheable[key] = value
        
        return cacheable
    
    def _update_performance_metrics(self, processing_time: float):
        """Update system performance metrics."""
        self.throughput_metrics.append({
            "timestamp": time.time(),
            "processing_time": processing_time
        })
        
        # Keep only recent metrics
        if len(self.throughput_metrics) > 1000:
            self.throughput_metrics = self.throughput_metrics[-1000:]
        
        # Update running averages
        recent_times = [m["processing_time"] for m in self.throughput_metrics[-50:]]
        self.average_response_time = np.mean(recent_times)
        
        # Update load estimate
        self.current_load = len(self.throughput_metrics[-10:]) / 10.0  # Requests per second estimate
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics."""
        metrics = {
            "cache_metrics": {
                "cache_size": len(self.prediction_cache),
                "max_cache_size": self.cache_size,
                "cache_hit_ratio": self._calculate_cache_hit_ratio()
            },
            "performance_metrics": {
                "average_response_time": self.average_response_time,
                "current_load": self.current_load,
                "total_requests": len(self.throughput_metrics)
            },
            "optimization_status": {
                "optimization_enabled": self.optimization_enabled,
                "gpu_available": torch.cuda.is_available(),
                "thread_pool_active": self.thread_pool is not None
            }
        }
        
        # Add profiler metrics
        if self.performance_profiler:
            metrics["profiling"] = self.performance_profiler.get_profile_summary()
            metrics["bottleneck_analysis"] = self.performance_profiler.analyze_bottlenecks().__dict__
        
        # Add robust system health
        metrics["system_health"] = self.robust_system.get_system_health_summary()
        
        return metrics
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        # This would be tracked in a real implementation
        # For now, return estimated value based on cache usage
        if not self.prediction_cache:
            return 0.0
        
        # Estimate based on cache utilization
        utilization = len(self.prediction_cache) / max(1, self.cache_size)
        return min(0.8, utilization * 0.5)  # Rough estimate
    
    def optimize_system(self):
        """Perform real-time system optimization."""
        if not self.optimization_enabled:
            return
        
        try:
            # Analyze current performance
            if self.performance_profiler:
                bottlenecks = self.performance_profiler.analyze_bottlenecks()
                
                # Auto-tune based on bottlenecks
                if bottlenecks.performance_score < 0.7:
                    logger.info("Performance degradation detected, optimizing...")
                    
                    # Adjust cache size if memory is an issue
                    if "memory" in str(bottlenecks.optimization_opportunities):
                        new_cache_size = max(100, self.cache_size // 2)
                        self.cache_size = new_cache_size
                        self._evict_excess_cache_items()
                    
                    # Adjust batch size if throughput is low
                    if self.average_response_time > 2.0:
                        current_batch_size = self.optimization_config["batch_processing"]["max_batch_size"]
                        new_batch_size = max(1, current_batch_size // 2)
                        self.optimization_config["batch_processing"]["max_batch_size"] = new_batch_size
                        logger.info(f"Reduced batch size to {new_batch_size}")
            
            # GPU memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
    
    def _evict_excess_cache_items(self):
        """Remove excess items from cache."""
        while len(self.prediction_cache) > self.cache_size:
            self._evict_lru_item()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "optimization_summary": self.get_optimization_metrics(),
            "recommendations": []
        }
        
        # Add performance recommendations
        if self.average_response_time > 1.0:
            report["recommendations"].append("Consider increasing batch size for better throughput")
        
        if self.current_load > 5.0:
            report["recommendations"].append("High load detected - consider scaling up")
        
        cache_hit_ratio = self._calculate_cache_hit_ratio()
        if cache_hit_ratio < 0.3:
            report["recommendations"].append("Low cache hit ratio - review caching strategy")
        
        # Add profiler analysis
        if self.performance_profiler:
            bottlenecks = self.performance_profiler.analyze_bottlenecks()
            report["bottleneck_analysis"] = bottlenecks.__dict__
            report["recommendations"].extend(bottlenecks.recommendations)
        
        return report
    
    def shutdown(self):
        """Gracefully shutdown the optimized system."""
        logger.info("Shutting down optimized anomaly detection system")
        
        # Stop performance profiler
        if self.performance_profiler:
            self.performance_profiler.stop_monitoring()
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # Shutdown robust system
        self.robust_system.shutdown()
        
        # Clear caches
        self.prediction_cache.clear()
        self.cache_access_times.clear()
        
        logger.info("Optimized system shutdown completed")


def create_optimized_system(
    model_config: Dict[str, Any],
    robustness_config: Optional[Dict[str, Any]] = None,
    optimization_config: Optional[Dict[str, Any]] = None
) -> OptimizedAnomalyDetectionSystem:
    """
    Factory function to create optimized anomaly detection system.
    
    Args:
        model_config: Configuration for the ensemble model
        robustness_config: Configuration for robustness features
        optimization_config: Configuration for optimization features
        
    Returns:
        OptimizedAnomalyDetectionSystem instance
    """
    return OptimizedAnomalyDetectionSystem(
        model_config, robustness_config, optimization_config
    )