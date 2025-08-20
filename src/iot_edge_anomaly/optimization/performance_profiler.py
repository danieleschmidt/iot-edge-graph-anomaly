"""
Performance Profiling and Monitoring for IoT Edge Anomaly Detection.

Provides comprehensive performance analysis and optimization insights:
- Execution time profiling
- Memory usage tracking
- GPU utilization monitoring
- Bottleneck identification
- Performance regression detection
- Optimization recommendations
"""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import torch
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ProfileMetrics:
    """Performance profile metrics."""
    function_name: str
    total_time: float
    call_count: int
    avg_time: float
    min_time: float
    max_time: float
    memory_delta: float
    gpu_memory_delta: Optional[float]
    cpu_usage: float
    gpu_usage: Optional[float]
    timestamp: float


@dataclass 
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    slowest_functions: List[str]
    memory_hotspots: List[str]
    optimization_opportunities: List[str]
    performance_score: float
    recommendations: List[str]


class PerformanceProfiler:
    """
    Comprehensive performance profiler for production optimization.
    
    Features:
    - Function-level timing analysis
    - Memory usage tracking
    - GPU utilization monitoring
    - Bottleneck identification
    - Performance trend analysis
    - Automated optimization recommendations
    """
    
    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_gpu_profiling: bool = True,
        history_size: int = 1000,
        sampling_interval: float = 0.1
    ):
        """
        Initialize performance profiler.
        
        Args:
            enable_memory_profiling: Whether to track memory usage
            enable_gpu_profiling: Whether to track GPU metrics
            history_size: Number of profile records to keep
            sampling_interval: Sampling interval for system metrics
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_gpu_profiling = enable_gpu_profiling
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        
        # Profile data storage
        self.profile_data: Dict[str, List[ProfileMetrics]] = defaultdict(list)
        self.call_stack = []
        self.active_profiles = {}
        
        # System monitoring
        self.system_metrics = deque(maxlen=history_size)
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance baselines
        self.baselines = {}
        self.thresholds = self._default_thresholds()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Performance profiler initialized")
    
    def _default_thresholds(self) -> Dict[str, float]:
        """Default performance thresholds."""
        return {
            'max_function_time': 1.0,      # seconds
            'max_memory_delta': 100.0,     # MB
            'max_cpu_usage': 80.0,         # percent
            'max_gpu_usage': 85.0,         # percent
            'min_performance_score': 0.7   # 0-1 scale
        }
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for system metrics."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics.append(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        metrics = {
            'timestamp': time.time(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),  # GB
        }
        
        # GPU metrics if available
        if self.enable_gpu_profiling and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                metrics.update({
                    'gpu_usage': torch.cuda.utilization(),
                    'gpu_memory_used': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3),
                    'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3)
                })
            except Exception as e:
                logger.debug(f"Could not collect GPU metrics: {e}")
        
        return metrics
    
    @contextmanager
    def profile(self, function_name: str):
        """Context manager for profiling function execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        start_cpu = psutil.cpu_percent()
        start_gpu = self._get_gpu_usage()
        
        # Track nested calls
        self.call_stack.append(function_name)
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            end_memory = self._get_memory_usage()
            memory_delta = end_memory - start_memory if start_memory else 0
            
            end_gpu_memory = self._get_gpu_memory_usage()
            gpu_memory_delta = (end_gpu_memory - start_gpu_memory 
                              if start_gpu_memory and end_gpu_memory else None)
            
            end_cpu = psutil.cpu_percent()
            cpu_usage = (start_cpu + end_cpu) / 2 if start_cpu else end_cpu
            
            end_gpu = self._get_gpu_usage()
            gpu_usage = ((start_gpu + end_gpu) / 2 
                        if start_gpu and end_gpu else None)
            
            # Remove from call stack
            if self.call_stack and self.call_stack[-1] == function_name:
                self.call_stack.pop()
            
            # Record metrics
            self._record_metrics(
                function_name, execution_time, memory_delta, gpu_memory_delta,
                cpu_usage, gpu_usage
            )
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.enable_memory_profiling:
            return None
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)  # MB
        except Exception:
            return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self.enable_gpu_profiling or not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() / (1024**2)  # MB
        except Exception:
            return None
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not self.enable_gpu_profiling or not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.utilization()
        except Exception:
            return None
    
    def _record_metrics(
        self,
        function_name: str,
        execution_time: float,
        memory_delta: float,
        gpu_memory_delta: Optional[float],
        cpu_usage: float,
        gpu_usage: Optional[float]
    ):
        """Record performance metrics for a function."""
        with self._lock:
            # Get existing metrics for this function
            existing_metrics = self.profile_data[function_name]
            
            # Calculate statistics
            if existing_metrics:
                call_count = len(existing_metrics) + 1
                times = [m.total_time for m in existing_metrics] + [execution_time]
                avg_time = np.mean(times)
                min_time = min(times)
                max_time = max(times)
            else:
                call_count = 1
                avg_time = execution_time
                min_time = execution_time
                max_time = execution_time
            
            # Create metrics object
            metrics = ProfileMetrics(
                function_name=function_name,
                total_time=execution_time,
                call_count=call_count,
                avg_time=avg_time,
                min_time=min_time,
                max_time=max_time,
                memory_delta=memory_delta,
                gpu_memory_delta=gpu_memory_delta,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                timestamp=time.time()
            )
            
            # Store metrics
            self.profile_data[function_name].append(metrics)
            
            # Maintain history size
            if len(self.profile_data[function_name]) > self.history_size:
                self.profile_data[function_name] = self.profile_data[function_name][-self.history_size:]
            
            # Check for performance issues
            self._check_performance_thresholds(metrics)
    
    def _check_performance_thresholds(self, metrics: ProfileMetrics):
        """Check metrics against performance thresholds."""
        issues = []
        
        if metrics.total_time > self.thresholds['max_function_time']:
            issues.append(f"Slow execution: {metrics.function_name} took {metrics.total_time:.3f}s")
        
        if metrics.memory_delta > self.thresholds['max_memory_delta']:
            issues.append(f"High memory usage: {metrics.function_name} used {metrics.memory_delta:.1f}MB")
        
        if metrics.cpu_usage > self.thresholds['max_cpu_usage']:
            issues.append(f"High CPU usage: {metrics.function_name} used {metrics.cpu_usage:.1f}%")
        
        if (metrics.gpu_usage and 
            metrics.gpu_usage > self.thresholds['max_gpu_usage']):
            issues.append(f"High GPU usage: {metrics.function_name} used {metrics.gpu_usage:.1f}%")
        
        for issue in issues:
            logger.warning(f"Performance issue: {issue}")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        def wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def get_profile_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance profile summary."""
        with self._lock:
            if function_name:
                if function_name not in self.profile_data:
                    return {"error": f"No profile data for function '{function_name}'"}
                
                metrics_list = self.profile_data[function_name]
                latest_metrics = metrics_list[-1] if metrics_list else None
                
                return {
                    "function_name": function_name,
                    "total_calls": len(metrics_list),
                    "average_time": np.mean([m.total_time for m in metrics_list]),
                    "min_time": min(m.total_time for m in metrics_list),
                    "max_time": max(m.total_time for m in metrics_list),
                    "std_time": np.std([m.total_time for m in metrics_list]),
                    "average_memory_delta": np.mean([m.memory_delta for m in metrics_list if m.memory_delta]),
                    "latest_metrics": latest_metrics.__dict__ if latest_metrics else None
                }
            
            # Summary for all functions
            summary = {}
            for fname, metrics_list in self.profile_data.items():
                if metrics_list:
                    summary[fname] = {
                        "total_calls": len(metrics_list),
                        "average_time": np.mean([m.total_time for m in metrics_list]),
                        "total_time": sum(m.total_time for m in metrics_list),
                        "average_memory_delta": np.mean([m.memory_delta for m in metrics_list if m.memory_delta])
                    }
            
            return summary
    
    def analyze_bottlenecks(self) -> BottleneckAnalysis:
        """Analyze performance bottlenecks and provide recommendations."""
        with self._lock:
            if not self.profile_data:
                return BottleneckAnalysis([], [], [], 1.0, ["No profile data available"])
            
            # Find slowest functions
            function_times = {}
            function_memory = {}
            
            for function_name, metrics_list in self.profile_data.items():
                if metrics_list:
                    avg_time = np.mean([m.total_time for m in metrics_list])
                    total_time = sum(m.total_time for m in metrics_list)
                    avg_memory = np.mean([m.memory_delta for m in metrics_list if m.memory_delta])
                    
                    function_times[function_name] = total_time
                    function_memory[function_name] = avg_memory
            
            # Sort by performance impact
            slowest_functions = sorted(
                function_times.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            memory_hotspots = sorted(
                function_memory.items(),
                key=lambda x: x[1] if x[1] else 0,
                reverse=True
            )[:5]
            
            # Generate optimization opportunities
            optimization_opportunities = []
            recommendations = []
            
            for func_name, total_time in slowest_functions:
                if total_time > 10.0:  # More than 10 seconds total
                    optimization_opportunities.append(f"Optimize {func_name} (total: {total_time:.2f}s)")
                    recommendations.append(f"Consider caching or batching for {func_name}")
            
            for func_name, avg_memory in memory_hotspots:
                if avg_memory and avg_memory > 500:  # More than 500MB
                    optimization_opportunities.append(f"Reduce memory usage in {func_name} ({avg_memory:.1f}MB)")
                    recommendations.append(f"Consider memory optimization for {func_name}")
            
            # Calculate performance score
            performance_score = self._calculate_performance_score()
            
            return BottleneckAnalysis(
                slowest_functions=[f[0] for f in slowest_functions],
                memory_hotspots=[f[0] for f in memory_hotspots],
                optimization_opportunities=optimization_opportunities,
                performance_score=performance_score,
                recommendations=recommendations
            )
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-1)."""
        if not self.profile_data:
            return 1.0
        
        score_components = []
        
        # Time-based score
        all_times = []
        for metrics_list in self.profile_data.values():
            all_times.extend([m.total_time for m in metrics_list])
        
        if all_times:
            avg_time = np.mean(all_times)
            max_acceptable_time = self.thresholds['max_function_time']
            time_score = max(0, 1 - (avg_time / max_acceptable_time))
            score_components.append(time_score)
        
        # Memory-based score
        all_memory = []
        for metrics_list in self.profile_data.values():
            all_memory.extend([m.memory_delta for m in metrics_list if m.memory_delta])
        
        if all_memory:
            avg_memory = np.mean(all_memory)
            max_acceptable_memory = self.thresholds['max_memory_delta']
            memory_score = max(0, 1 - (avg_memory / max_acceptable_memory))
            score_components.append(memory_score)
        
        # System metrics score
        if self.system_metrics:
            recent_metrics = list(self.system_metrics)[-10:]  # Last 10 samples
            avg_cpu = np.mean([m['cpu_usage'] for m in recent_metrics])
            cpu_score = max(0, 1 - (avg_cpu / 100))
            score_components.append(cpu_score)
        
        return np.mean(score_components) if score_components else 1.0
    
    def set_baseline(self, function_name: str):
        """Set performance baseline for a function."""
        if function_name in self.profile_data:
            metrics_list = self.profile_data[function_name]
            if metrics_list:
                baseline = {
                    'avg_time': np.mean([m.total_time for m in metrics_list]),
                    'avg_memory': np.mean([m.memory_delta for m in metrics_list if m.memory_delta]),
                    'timestamp': time.time()
                }
                self.baselines[function_name] = baseline
                logger.info(f"Baseline set for {function_name}")
    
    def compare_to_baseline(self, function_name: str) -> Dict[str, Any]:
        """Compare current performance to baseline."""
        if function_name not in self.baselines:
            return {"error": "No baseline available"}
        
        if function_name not in self.profile_data:
            return {"error": "No current data available"}
        
        baseline = self.baselines[function_name]
        current_metrics = self.profile_data[function_name][-10:]  # Last 10 calls
        
        if not current_metrics:
            return {"error": "No recent metrics available"}
        
        current_avg_time = np.mean([m.total_time for m in current_metrics])
        current_avg_memory = np.mean([m.memory_delta for m in current_metrics if m.memory_delta])
        
        time_change = ((current_avg_time - baseline['avg_time']) / baseline['avg_time']) * 100
        memory_change = (((current_avg_memory or 0) - (baseline['avg_memory'] or 0)) / 
                        max(baseline['avg_memory'] or 1, 1)) * 100
        
        return {
            "function_name": function_name,
            "baseline_time": baseline['avg_time'],
            "current_time": current_avg_time,
            "time_change_percent": time_change,
            "baseline_memory": baseline['avg_memory'],
            "current_memory": current_avg_memory,
            "memory_change_percent": memory_change,
            "performance_regression": time_change > 20 or memory_change > 20
        }
    
    def export_profile_data(self, format: str = "json") -> str:
        """Export profile data for analysis."""
        import json
        
        with self._lock:
            export_data = {}
            
            for function_name, metrics_list in self.profile_data.items():
                export_data[function_name] = [
                    {
                        "timestamp": m.timestamp,
                        "execution_time": m.total_time,
                        "memory_delta": m.memory_delta,
                        "cpu_usage": m.cpu_usage,
                        "gpu_usage": m.gpu_usage
                    }
                    for m in metrics_list
                ]
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2)
            else:
                return str(export_data)
    
    def reset_profile_data(self):
        """Reset all profile data."""
        with self._lock:
            self.profile_data.clear()
            self.baselines.clear()
            self.system_metrics.clear()
            logger.info("Profile data reset")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics."""
        current_metrics = self._collect_system_metrics()
        
        # Add profile statistics
        if self.profile_data:
            active_functions = len(self.profile_data)
            total_calls = sum(len(metrics_list) for metrics_list in self.profile_data.values())
            
            current_metrics.update({
                "active_functions": active_functions,
                "total_function_calls": total_calls,
                "call_stack_depth": len(self.call_stack),
                "current_call_stack": self.call_stack.copy()
            })
        
        return current_metrics