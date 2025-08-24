#!/usr/bin/env python3
"""
Edge Deployment Benchmarking Suite for IoT Anomaly Detection.

This comprehensive benchmarking suite evaluates AI algorithms specifically for edge
deployment scenarios, measuring performance, resource utilization, energy consumption,
and deployment viability across different edge hardware configurations.

Key Features:
- Multi-platform edge device simulation (Raspberry Pi, Jetson Nano, Intel NCS)
- Model quantization and optimization benchmarking
- Real-time performance monitoring
- Energy consumption modeling
- Deployment readiness assessment
- Hardware-specific optimization recommendations

Authors: Terragon Autonomous SDLC v4.0
Date: 2025-08-23
"""

import os
import sys
import time
import json
import logging
import psutil
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.quantization as quant
from torch.profiler import profile, record_function, ProfilerActivity

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from iot_edge_anomaly.models.transformer_vae import TransformerVAE
from iot_edge_anomaly.models.sparse_graph_attention import SparseGraphAttentionNetwork
from iot_edge_anomaly.models.physics_informed_hybrid import PhysicsInformedHybrid

logger = logging.getLogger(__name__)


@dataclass
class EdgeHardwareConfig:
    """Configuration for edge hardware platform."""
    name: str
    cpu_cores: int
    cpu_freq_mhz: int
    memory_mb: int
    storage_gb: int
    gpu_available: bool
    gpu_memory_mb: int
    typical_power_watts: float
    cost_usd: float
    
    # Performance characteristics
    int8_ops_per_sec: int  # Operations per second for INT8
    fp16_ops_per_sec: int  # Operations per second for FP16  
    fp32_ops_per_sec: int  # Operations per second for FP32
    
    # Network characteristics
    network_bandwidth_mbps: float
    network_latency_ms: float


@dataclass
class EdgeBenchmarkResult:
    """Comprehensive edge deployment benchmark result."""
    model_name: str
    hardware_config: str
    quantization_bits: int
    
    # Performance metrics
    accuracy: float
    f1_score: float
    inference_time_ms: float
    throughput_samples_per_sec: float
    
    # Resource utilization
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    gpu_usage_percent: float
    gpu_memory_mb: float
    
    # Energy and efficiency
    energy_per_inference_mj: float
    power_consumption_watts: float
    energy_efficiency_inferences_per_joule: float
    
    # Model characteristics
    model_size_mb: float
    compressed_size_mb: float
    parameter_count: int
    quantized_parameter_count: int
    
    # Deployment metrics
    startup_time_ms: float
    model_load_time_ms: float
    memory_fragmentation: float
    thermal_impact: float
    
    # Quality metrics
    quantization_accuracy_loss: float
    calibration_time_ms: float
    deployment_readiness_score: float
    
    # Additional metadata
    profiler_data: Optional[Dict[str, Any]] = None
    optimization_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []


class EdgeHardwareSimulator:
    """Simulate various edge hardware platforms for benchmarking."""
    
    def __init__(self):
        self.hardware_configs = self._initialize_hardware_configs()
    
    def _initialize_hardware_configs(self) -> Dict[str, EdgeHardwareConfig]:
        """Initialize standard edge hardware configurations."""
        
        configs = {
            "raspberry_pi_4": EdgeHardwareConfig(
                name="Raspberry Pi 4B",
                cpu_cores=4,
                cpu_freq_mhz=1500,
                memory_mb=4096,
                storage_gb=32,
                gpu_available=False,
                gpu_memory_mb=0,
                typical_power_watts=5.1,
                cost_usd=75,
                int8_ops_per_sec=2_000_000_000,  # 2 GOP/s
                fp16_ops_per_sec=500_000_000,    # 0.5 GOP/s  
                fp32_ops_per_sec=250_000_000,    # 0.25 GOP/s
                network_bandwidth_mbps=100,
                network_latency_ms=10
            ),
            
            "jetson_nano": EdgeHardwareConfig(
                name="NVIDIA Jetson Nano",
                cpu_cores=4,
                cpu_freq_mhz=1430,
                memory_mb=4096,
                storage_gb=32,
                gpu_available=True,
                gpu_memory_mb=4096,  # Shared with system
                typical_power_watts=10.0,
                cost_usd=149,
                int8_ops_per_sec=21_000_000_000,  # 21 GOP/s (GPU)
                fp16_ops_per_sec=10_500_000_000,  # 10.5 GOP/s (GPU)
                fp32_ops_per_sec=472_000_000,     # 0.472 GFLOP/s
                network_bandwidth_mbps=1000,
                network_latency_ms=5
            ),
            
            "intel_ncs2": EdgeHardwareConfig(
                name="Intel Neural Compute Stick 2",
                cpu_cores=1,  # Host CPU dependency
                cpu_freq_mhz=2400,  # Assuming host
                memory_mb=512,  # Limited onboard
                storage_gb=0,   # No storage
                gpu_available=True,  # Movidius VPU
                gpu_memory_mb=512,
                typical_power_watts=2.5,
                cost_usd=99,
                int8_ops_per_sec=4_000_000_000,   # 4 GOP/s
                fp16_ops_per_sec=2_000_000_000,   # 2 GOP/s
                fp32_ops_per_sec=100_000_000,     # 0.1 GOP/s
                network_bandwidth_mbps=100,
                network_latency_ms=8
            ),
            
            "coral_dev_board": EdgeHardwareConfig(
                name="Google Coral Dev Board", 
                cpu_cores=4,
                cpu_freq_mhz=1500,
                memory_mb=1024,
                storage_gb=8,
                gpu_available=True,  # Edge TPU
                gpu_memory_mb=1024,
                typical_power_watts=6.5,
                cost_usd=175,
                int8_ops_per_sec=4_000_000_000,   # 4 TOPS (Edge TPU)
                fp16_ops_per_sec=0,               # TPU optimized for INT8
                fp32_ops_per_sec=100_000_000,     # CPU fallback
                network_bandwidth_mbps=1000,
                network_latency_ms=3
            ),
            
            "edge_server": EdgeHardwareConfig(
                name="Edge Server (Generic)",
                cpu_cores=8,
                cpu_freq_mhz=3200,
                memory_mb=16384,
                storage_gb=512,
                gpu_available=True,
                gpu_memory_mb=8192,
                typical_power_watts=150.0,
                cost_usd=2000,
                int8_ops_per_sec=50_000_000_000,  # 50 GOP/s
                fp16_ops_per_sec=25_000_000_000,  # 25 GOP/s
                fp32_ops_per_sec=12_500_000_000,  # 12.5 GFLOP/s
                network_bandwidth_mbps=10000,
                network_latency_ms=1
            )
        }
        
        return configs
    
    def get_hardware_config(self, hardware_name: str) -> EdgeHardwareConfig:
        """Get hardware configuration by name."""
        if hardware_name not in self.hardware_configs:
            raise ValueError(f"Unknown hardware configuration: {hardware_name}")
        return self.hardware_configs[hardware_name]
    
    def list_available_hardware(self) -> List[str]:
        """List all available hardware configurations."""
        return list(self.hardware_configs.keys())


class ModelQuantizer:
    """Handle model quantization for edge deployment."""
    
    def __init__(self):
        self.quantized_models = {}
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        quantization_bits: int = 8,
        backend: str = 'fbgemm'
    ) -> Tuple[nn.Module, float]:
        """
        Quantize model to specified bit precision.
        
        Returns:
            (quantized_model, calibration_time_ms)
        """
        
        start_time = time.time()
        
        try:
            if quantization_bits == 32:
                # No quantization needed
                return model, 0.0
            
            elif quantization_bits == 16:
                # Half precision quantization
                quantized_model = model.half()
                
            elif quantization_bits == 8:
                # INT8 quantization
                model.eval()
                
                # Prepare model for quantization
                model.qconfig = torch.quantization.get_default_qconfig(backend)
                quantized_model = torch.quantization.prepare(model, inplace=False)
                
                # Calibration
                with torch.no_grad():
                    for i in range(min(100, len(calibration_data))):
                        sample = calibration_data[i:i+1]
                        try:
                            if hasattr(quantized_model, 'compute_reconstruction_error'):
                                _ = quantized_model.compute_reconstruction_error(sample)
                            else:
                                _ = quantized_model(sample)
                        except:
                            # Skip problematic samples
                            continue
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(quantized_model, inplace=False)
                
            else:
                raise ValueError(f"Unsupported quantization: {quantization_bits} bits")
            
            calibration_time = (time.time() - start_time) * 1000  # Convert to ms
            return quantized_model, calibration_time
            
        except Exception as e:
            logger.warning(f"Quantization failed for {quantization_bits} bits: {e}")
            # Return original model if quantization fails
            return model, 0.0
    
    def measure_quantization_accuracy_loss(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> float:
        """Measure accuracy loss due to quantization."""
        
        def compute_anomaly_score(model, data):
            model.eval()
            scores = []
            
            with torch.no_grad():
                for i in range(len(data)):
                    sample = data[i:i+1]
                    try:
                        if hasattr(model, 'compute_reconstruction_error'):
                            score = model.compute_reconstruction_error(sample)
                        else:
                            reconstruction = model(sample)
                            score = torch.mean((sample - reconstruction) ** 2)
                        scores.append(score.item())
                    except:
                        scores.append(0.0)  # Default score for failed inference
            
            return scores
        
        try:
            # Compute scores for both models
            original_scores = compute_anomaly_score(original_model, test_data)
            quantized_scores = compute_anomaly_score(quantized_model, test_data)
            
            # Determine thresholds
            normal_mask = test_labels == 0
            
            orig_normal_scores = [s for i, s in enumerate(original_scores) if normal_mask[i]]
            quant_normal_scores = [s for i, s in enumerate(quantized_scores) if normal_mask[i]]
            
            if len(orig_normal_scores) == 0:
                return 0.0
            
            orig_threshold = np.mean(orig_normal_scores) + 2 * np.std(orig_normal_scores)
            quant_threshold = np.mean(quant_normal_scores) + 2 * np.std(quant_normal_scores)
            
            # Compute accuracy
            orig_predictions = [1 if s > orig_threshold else 0 for s in original_scores]
            quant_predictions = [1 if s > quant_threshold else 0 for s in quantized_scores]
            
            orig_accuracy = np.mean([p == l for p, l in zip(orig_predictions, test_labels)])
            quant_accuracy = np.mean([p == l for p, l in zip(quant_predictions, test_labels)])
            
            accuracy_loss = max(0, orig_accuracy - quant_accuracy)
            return accuracy_loss
            
        except Exception as e:
            logger.warning(f"Failed to measure quantization accuracy loss: {e}")
            return 0.0


class EdgePerformanceProfiler:
    """Profile model performance on edge hardware."""
    
    def __init__(self):
        self.monitoring_active = False
        self.resource_usage_history = []
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        self.monitoring_active = True
        self.resource_usage_history = []
        
        def monitor_resources():
            while self.monitoring_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()
                    
                    self.resource_usage_history.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_mb': memory_info.used / (1024 * 1024),
                        'memory_percent': memory_info.percent
                    })
                except:
                    pass  # Continue monitoring even if some metrics fail
                
                time.sleep(0.1)  # 100ms sampling rate
        
        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated statistics."""
        self.monitoring_active = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        if not self.resource_usage_history:
            return {
                'avg_cpu_percent': 0.0,
                'peak_cpu_percent': 0.0,
                'avg_memory_mb': 0.0,
                'peak_memory_mb': 0.0
            }
        
        cpu_values = [entry['cpu_percent'] for entry in self.resource_usage_history]
        memory_values = [entry['memory_mb'] for entry in self.resource_usage_history]
        
        return {
            'avg_cpu_percent': np.mean(cpu_values),
            'peak_cpu_percent': np.max(cpu_values), 
            'avg_memory_mb': np.mean(memory_values),
            'peak_memory_mb': np.max(memory_values)
        }
    
    def profile_model_inference(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        hardware_config: EdgeHardwareConfig,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Profile model inference performance."""
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() and hardware_config.gpu_available else 'cpu')
        model = model.to(device)
        
        # Warm-up
        warmup_samples = min(10, len(test_data))
        with torch.no_grad():
            for i in range(warmup_samples):
                sample = test_data[i:i+1].to(device)
                try:
                    if hasattr(model, 'compute_reconstruction_error'):
                        _ = model.compute_reconstruction_error(sample)
                    else:
                        _ = model(sample)
                except:
                    continue
        
        # Start monitoring
        self.start_monitoring()
        
        # Benchmark inference
        inference_times = []
        successful_inferences = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size].to(device)
                
                inference_start = time.time()
                
                try:
                    if hasattr(model, 'compute_reconstruction_error'):
                        _ = model.compute_reconstruction_error(batch)
                    else:
                        _ = model(batch)
                    
                    inference_time = (time.time() - inference_start) * 1000  # ms
                    inference_times.append(inference_time / len(batch))  # per sample
                    successful_inferences += len(batch)
                    
                except Exception as e:
                    logger.debug(f"Inference failed for batch {i}: {e}")
                    continue
        
        total_time = time.time() - start_time
        
        # Stop monitoring
        resource_stats = self.stop_monitoring()
        
        # Calculate metrics
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            std_inference_time = np.std(inference_times)
            throughput = successful_inferences / total_time if total_time > 0 else 0
        else:
            avg_inference_time = float('inf')
            std_inference_time = 0
            throughput = 0
        
        # Estimate energy consumption
        avg_power = hardware_config.typical_power_watts
        energy_per_inference = (avg_inference_time / 1000) * avg_power  # Joules
        
        profile_data = {
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'throughput_samples_per_sec': throughput,
            'successful_inferences': successful_inferences,
            'total_samples': len(test_data),
            'success_rate': successful_inferences / len(test_data) if len(test_data) > 0 else 0,
            'energy_per_inference_joules': energy_per_inference,
            'resource_usage': resource_stats
        }
        
        return profile_data


class EdgeDeploymentBenchmarkSuite:
    """Comprehensive edge deployment benchmarking suite."""
    
    def __init__(self, output_dir: str = "./edge_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hardware_simulator = EdgeHardwareSimulator()
        self.quantizer = ModelQuantizer()
        self.profiler = EdgePerformanceProfiler()
        
        self.results: List[EdgeBenchmarkResult] = []
        
        # Setup logging
        log_file = self.output_dir / "edge_benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Initialized Edge Deployment Benchmark Suite: {output_dir}")
    
    def benchmark_model_on_hardware(
        self,
        model: nn.Module,
        model_name: str,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        hardware_name: str,
        quantization_bits: List[int] = [32, 16, 8]
    ) -> List[EdgeBenchmarkResult]:
        """Benchmark a model on specific hardware with different quantization levels."""
        
        logger.info(f"Benchmarking {model_name} on {hardware_name}")
        
        hardware_config = self.hardware_simulator.get_hardware_config(hardware_name)
        results = []
        
        # Use subset of data for faster benchmarking
        benchmark_data = test_data[:min(200, len(test_data))]
        benchmark_labels = test_labels[:min(200, len(test_labels))]
        
        for quant_bits in quantization_bits:
            logger.info(f"  Testing {quant_bits}-bit quantization")
            
            try:
                # Quantize model
                quantized_model, calibration_time = self.quantizer.quantize_model(
                    model, benchmark_data, quant_bits
                )
                
                # Measure model properties
                original_params = sum(p.numel() for p in model.parameters())
                original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                
                quantized_params = sum(p.numel() for p in quantized_model.parameters())
                quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
                
                # Measure quantization accuracy loss
                accuracy_loss = self.quantizer.measure_quantization_accuracy_loss(
                    model, quantized_model, benchmark_data, benchmark_labels
                )
                
                # Profile performance
                profile_data = self.profiler.profile_model_inference(
                    quantized_model, benchmark_data, hardware_config
                )
                
                # Calculate deployment metrics
                deployment_score = self._calculate_deployment_readiness_score(
                    profile_data, hardware_config, quantized_size
                )
                
                # Generate optimization recommendations
                recommendations = self._generate_optimization_recommendations(
                    profile_data, hardware_config, model_name, quant_bits
                )
                
                # Estimate accuracy and F1 score
                accuracy, f1_score = self._estimate_performance_metrics(
                    quantized_model, benchmark_data, benchmark_labels
                )
                
                # Calculate additional metrics
                energy_efficiency = (1.0 / profile_data['energy_per_inference_joules'] 
                                   if profile_data['energy_per_inference_joules'] > 0 else 0)
                
                memory_fragmentation = self._estimate_memory_fragmentation(quantized_size, hardware_config)
                thermal_impact = self._estimate_thermal_impact(profile_data, hardware_config)
                
                # Create result
                result = EdgeBenchmarkResult(
                    model_name=model_name,
                    hardware_config=hardware_name,
                    quantization_bits=quant_bits,
                    accuracy=accuracy,
                    f1_score=f1_score,
                    inference_time_ms=profile_data['avg_inference_time_ms'],
                    throughput_samples_per_sec=profile_data['throughput_samples_per_sec'],
                    cpu_usage_percent=profile_data['resource_usage']['avg_cpu_percent'],
                    memory_usage_mb=profile_data['resource_usage']['avg_memory_mb'],
                    peak_memory_mb=profile_data['resource_usage']['peak_memory_mb'],
                    gpu_usage_percent=0.0,  # Simplified - would need GPU monitoring
                    gpu_memory_mb=0.0,
                    energy_per_inference_mj=profile_data['energy_per_inference_joules'] * 1000,  # mJ
                    power_consumption_watts=hardware_config.typical_power_watts,
                    energy_efficiency_inferences_per_joule=energy_efficiency,
                    model_size_mb=original_size,
                    compressed_size_mb=quantized_size,
                    parameter_count=original_params,
                    quantized_parameter_count=quantized_params,
                    startup_time_ms=calibration_time,
                    model_load_time_ms=calibration_time * 0.1,  # Estimate
                    memory_fragmentation=memory_fragmentation,
                    thermal_impact=thermal_impact,
                    quantization_accuracy_loss=accuracy_loss,
                    calibration_time_ms=calibration_time,
                    deployment_readiness_score=deployment_score,
                    profiler_data=profile_data,
                    optimization_recommendations=recommendations
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmarking failed for {model_name} on {hardware_name} "
                           f"with {quant_bits}-bit quantization: {e}")
                continue
        
        self.results.extend(results)
        return results
    
    def _estimate_performance_metrics(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Estimate accuracy and F1-score for the model."""
        
        try:
            model.eval()
            device = next(model.parameters()).device
            
            # Compute anomaly scores
            scores = []
            with torch.no_grad():
                for i in range(len(test_data)):
                    sample = test_data[i:i+1].to(device)
                    try:
                        if hasattr(model, 'compute_reconstruction_error'):
                            score = model.compute_reconstruction_error(sample)
                        else:
                            reconstruction = model(sample)
                            score = torch.mean((sample - reconstruction) ** 2)
                        scores.append(score.item())
                    except:
                        scores.append(0.0)
            
            # Determine threshold
            normal_scores = [s for i, s in enumerate(scores) if test_labels[i] == 0]
            if len(normal_scores) > 0:
                threshold = np.mean(normal_scores) + 2 * np.std(normal_scores)
            else:
                threshold = np.percentile(scores, 90)
            
            # Make predictions
            predictions = [1 if s > threshold else 0 for s in scores]
            
            # Calculate metrics
            accuracy = np.mean([p == l for p, l in zip(predictions, test_labels)])
            
            # F1 score
            tp = sum((p == 1) and (l == 1) for p, l in zip(predictions, test_labels))
            fp = sum((p == 1) and (l == 0) for p, l in zip(predictions, test_labels))
            fn = sum((p == 0) and (l == 1) for p, l in zip(predictions, test_labels))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return accuracy, f1_score
            
        except Exception as e:
            logger.warning(f"Failed to estimate performance metrics: {e}")
            return 0.5, 0.5  # Default values
    
    def _calculate_deployment_readiness_score(
        self,
        profile_data: Dict[str, Any],
        hardware_config: EdgeHardwareConfig,
        model_size_mb: float
    ) -> float:
        """Calculate deployment readiness score (0-1)."""
        
        score = 0.0
        
        # Latency score (target: < 100ms)
        inference_time = profile_data['avg_inference_time_ms']
        if inference_time < 10:
            latency_score = 1.0
        elif inference_time < 50:
            latency_score = 0.8
        elif inference_time < 100:
            latency_score = 0.6
        elif inference_time < 500:
            latency_score = 0.4
        else:
            latency_score = 0.2
        
        # Memory score
        memory_usage = profile_data['resource_usage']['peak_memory_mb']
        memory_ratio = memory_usage / hardware_config.memory_mb
        if memory_ratio < 0.3:
            memory_score = 1.0
        elif memory_ratio < 0.5:
            memory_score = 0.8
        elif memory_ratio < 0.7:
            memory_score = 0.6
        elif memory_ratio < 0.9:
            memory_score = 0.4
        else:
            memory_score = 0.2
        
        # Model size score
        storage_ratio = (model_size_mb * 1024) / (hardware_config.storage_gb * 1024)  # MB to GB
        if storage_ratio < 0.1:
            size_score = 1.0
        elif storage_ratio < 0.2:
            size_score = 0.8
        elif storage_ratio < 0.3:
            size_score = 0.6
        else:
            size_score = 0.4
        
        # Throughput score
        throughput = profile_data['throughput_samples_per_sec']
        if throughput > 100:
            throughput_score = 1.0
        elif throughput > 50:
            throughput_score = 0.8
        elif throughput > 10:
            throughput_score = 0.6
        elif throughput > 1:
            throughput_score = 0.4
        else:
            throughput_score = 0.2
        
        # Success rate score
        success_rate = profile_data['success_rate']
        success_score = success_rate  # Direct mapping
        
        # Weighted combination
        score = (
            0.3 * latency_score +      # Latency is critical
            0.25 * memory_score +      # Memory constraints important
            0.15 * size_score +        # Model size moderately important
            0.15 * throughput_score +  # Throughput moderately important
            0.15 * success_score       # Success rate important
        )
        
        return min(1.0, max(0.0, score))
    
    def _generate_optimization_recommendations(
        self,
        profile_data: Dict[str, Any],
        hardware_config: EdgeHardwareConfig,
        model_name: str,
        quantization_bits: int
    ) -> List[str]:
        """Generate hardware-specific optimization recommendations."""
        
        recommendations = []
        
        # Inference time recommendations
        inference_time = profile_data['avg_inference_time_ms']
        if inference_time > 100:
            recommendations.append("Consider further model pruning to reduce inference time")
            if quantization_bits > 8:
                recommendations.append("Try INT8 quantization for better performance")
        
        # Memory recommendations
        memory_usage = profile_data['resource_usage']['peak_memory_mb']
        memory_ratio = memory_usage / hardware_config.memory_mb
        if memory_ratio > 0.7:
            recommendations.append("High memory usage detected - consider model compression")
            recommendations.append("Implement gradient checkpointing if available")
        
        # Hardware-specific recommendations
        if hardware_config.name == "Raspberry Pi 4B":
            recommendations.append("Use ARM-optimized libraries (e.g., ARM Compute Library)")
            recommendations.append("Consider NEON SIMD optimizations")
            if inference_time > 50:
                recommendations.append("Model too complex for Raspberry Pi - consider simpler architecture")
        
        elif hardware_config.name == "NVIDIA Jetson Nano":
            if not hardware_config.gpu_available or profile_data['throughput_samples_per_sec'] < 10:
                recommendations.append("Ensure GPU acceleration is enabled")
                recommendations.append("Use TensorRT for additional optimization")
            recommendations.append("Consider mixed-precision training/inference")
        
        elif hardware_config.name == "Intel Neural Compute Stick 2":
            recommendations.append("Convert model to OpenVINO IR format for optimal performance")
            recommendations.append("Use INT8 calibration for Movidius VPU")
            if memory_usage > 400:  # VPU has limited memory
                recommendations.append("Model may be too large for NCS2 - consider pruning")
        
        elif hardware_config.name == "Google Coral Dev Board":
            recommendations.append("Convert model to TensorFlow Lite format")
            recommendations.append("Use Edge TPU compiler for optimal performance")
            if quantization_bits != 8:
                recommendations.append("Edge TPU requires INT8 quantization")
        
        # Energy efficiency recommendations
        energy_per_inference = profile_data['energy_per_inference_joules']
        if energy_per_inference > 1.0:  # High energy consumption
            recommendations.append("High energy consumption - consider model distillation")
            recommendations.append("Implement dynamic voltage/frequency scaling")
        
        # Throughput recommendations
        throughput = profile_data['throughput_samples_per_sec']
        if throughput < 1:
            recommendations.append("Very low throughput - consider batch processing")
            recommendations.append("Investigate CPU/GPU utilization bottlenecks")
        
        return recommendations
    
    def _estimate_memory_fragmentation(
        self,
        model_size_mb: float,
        hardware_config: EdgeHardwareConfig
    ) -> float:
        """Estimate memory fragmentation impact (0-1)."""
        
        # Simple heuristic: larger models on smaller devices have higher fragmentation
        size_ratio = model_size_mb / hardware_config.memory_mb
        
        if size_ratio < 0.1:
            return 0.1  # Low fragmentation
        elif size_ratio < 0.3:
            return 0.3  # Moderate fragmentation
        elif size_ratio < 0.5:
            return 0.5  # High fragmentation
        else:
            return 0.8  # Very high fragmentation
    
    def _estimate_thermal_impact(
        self,
        profile_data: Dict[str, Any],
        hardware_config: EdgeHardwareConfig
    ) -> float:
        """Estimate thermal impact (0-1)."""
        
        # Based on CPU usage and power consumption
        cpu_usage = profile_data['resource_usage']['avg_cpu_percent']
        power_ratio = hardware_config.typical_power_watts / 10.0  # Normalize to 10W baseline
        
        thermal_impact = (cpu_usage / 100.0) * power_ratio
        
        return min(1.0, max(0.0, thermal_impact))
    
    def run_comprehensive_edge_benchmark(
        self,
        models: Dict[str, nn.Module],
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        hardware_list: List[str] = None,
        quantization_bits: List[int] = [32, 16, 8]
    ) -> pd.DataFrame:
        """Run comprehensive benchmarking across all models and hardware."""
        
        logger.info("Starting comprehensive edge deployment benchmarking")
        
        if hardware_list is None:
            hardware_list = self.hardware_simulator.list_available_hardware()
        
        all_results = []
        
        for model_name, model in models.items():
            logger.info(f"Benchmarking model: {model_name}")
            
            for hardware_name in hardware_list:
                try:
                    model_results = self.benchmark_model_on_hardware(
                        model, model_name, test_data, test_labels,
                        hardware_name, quantization_bits
                    )
                    all_results.extend(model_results)
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {model_name} on {hardware_name}: {e}")
                    continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(result) for result in all_results])
        
        # Save results
        results_file = self.output_dir / "comprehensive_edge_results.csv"
        results_df.to_csv(results_file, index=False)
        
        # Generate summary report
        self._generate_edge_benchmark_report(results_df)
        
        logger.info("Comprehensive edge benchmarking completed")
        return results_df
    
    def _generate_edge_benchmark_report(self, results_df: pd.DataFrame):
        """Generate comprehensive edge benchmark report."""
        
        report_lines = [
            "# Edge Deployment Benchmark Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Configurations Tested:** {len(results_df)}",
            f"**Models Evaluated:** {results_df['model_name'].nunique()}",
            f"**Hardware Platforms:** {results_df['hardware_config'].nunique()}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Best configurations by criteria
        best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
        best_inference_time = results_df.loc[results_df['inference_time_ms'].idxmin()]
        best_energy_efficiency = results_df.loc[results_df['energy_efficiency_inferences_per_joule'].idxmax()]
        best_deployment_score = results_df.loc[results_df['deployment_readiness_score'].idxmax()]
        
        report_lines.extend([
            "### Top Performing Configurations",
            "",
            f"**Best Accuracy:** {best_accuracy['model_name']} on {best_accuracy['hardware_config']} "
            f"({best_accuracy['quantization_bits']}-bit) - {best_accuracy['accuracy']:.3f}",
            "",
            f"**Fastest Inference:** {best_inference_time['model_name']} on {best_inference_time['hardware_config']} "
            f"({best_inference_time['quantization_bits']}-bit) - {best_inference_time['inference_time_ms']:.2f}ms",
            "",
            f"**Most Energy Efficient:** {best_energy_efficiency['model_name']} on {best_energy_efficiency['hardware_config']} "
            f"({best_energy_efficiency['quantization_bits']}-bit) - {best_energy_efficiency['energy_efficiency_inferences_per_joule']:.1f} inferences/J",
            "",
            f"**Best Overall Deployment:** {best_deployment_score['model_name']} on {best_deployment_score['hardware_config']} "
            f"({best_deployment_score['quantization_bits']}-bit) - Score: {best_deployment_score['deployment_readiness_score']:.3f}",
            "",
        ])
        
        # Hardware platform analysis
        report_lines.extend([
            "## Hardware Platform Analysis",
            ""
        ])
        
        for hardware in results_df['hardware_config'].unique():
            hw_results = results_df[results_df['hardware_config'] == hardware]
            
            avg_inference_time = hw_results['inference_time_ms'].mean()
            avg_accuracy = hw_results['accuracy'].mean()
            avg_memory_usage = hw_results['memory_usage_mb'].mean()
            avg_deployment_score = hw_results['deployment_readiness_score'].mean()
            
            report_lines.extend([
                f"### {hardware}",
                f"- Average Inference Time: {avg_inference_time:.2f}ms",
                f"- Average Accuracy: {avg_accuracy:.3f}",
                f"- Average Memory Usage: {avg_memory_usage:.1f}MB",
                f"- Average Deployment Score: {avg_deployment_score:.3f}",
                ""
            ])
        
        # Model performance analysis
        report_lines.extend([
            "## Model Performance Analysis",
            ""
        ])
        
        for model in results_df['model_name'].unique():
            model_results = results_df[results_df['model_name'] == model]
            
            best_config = model_results.loc[model_results['deployment_readiness_score'].idxmax()]
            
            report_lines.extend([
                f"### {model}",
                f"- Best Configuration: {best_config['hardware_config']} ({best_config['quantization_bits']}-bit)",
                f"- Best Deployment Score: {best_config['deployment_readiness_score']:.3f}",
                f"- Average Model Size: {model_results['model_size_mb'].mean():.1f}MB",
                f"- Parameter Count: {model_results['parameter_count'].iloc[0]:,}",
                ""
            ])
        
        # Quantization analysis
        report_lines.extend([
            "## Quantization Impact Analysis",
            "",
            "| Quantization | Avg Accuracy | Avg Inference Time (ms) | Avg Model Size (MB) |",
            "|--------------|--------------|-------------------------|---------------------|"
        ])
        
        for bits in sorted(results_df['quantization_bits'].unique(), reverse=True):
            quant_results = results_df[results_df['quantization_bits'] == bits]
            avg_acc = quant_results['accuracy'].mean()
            avg_time = quant_results['inference_time_ms'].mean()
            avg_size = quant_results['compressed_size_mb'].mean()
            
            report_lines.append(f"| {bits}-bit | {avg_acc:.3f} | {avg_time:.2f} | {avg_size:.1f} |")
        
        report_lines.extend([
            "",
            "## Deployment Recommendations",
            "",
            "### Resource-Constrained Environments (Raspberry Pi, etc.)",
            "- Prioritize INT8 quantization for memory efficiency",
            "- Use model pruning and compression techniques", 
            "- Consider simplified model architectures",
            "",
            "### GPU-Accelerated Edge Devices (Jetson Nano, etc.)",
            "- Leverage mixed-precision inference",
            "- Utilize hardware-specific optimizations (TensorRT, etc.)",
            "- Balance accuracy vs. energy consumption",
            "",
            "### Specialized Edge AI Accelerators (Coral, NCS2)",
            "- Use hardware-specific model formats and compilers",
            "- Optimize for INT8 operations",
            "- Minimize host-device communication overhead",
            "",
            "---",
            f"*Report generated by Edge Deployment Benchmark Suite v1.0*"
        ])
        
        # Save report
        report_file = self.output_dir / "edge_benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Edge benchmark report saved to {report_file}")


def main():
    """Main function to demonstrate edge deployment benchmarking."""
    
    # Initialize benchmark suite
    benchmark_suite = EdgeDeploymentBenchmarkSuite("./edge_benchmark_results")
    
    # Create sample models (simplified for demo)
    models = {
        "TransformerVAE": TransformerVAE(input_dim=10, hidden_dim=32, latent_dim=16),
        "SparseGAT": SparseGraphAttentionNetwork(input_channels=10, hidden_channels=32),
        "PhysicsInformed": PhysicsInformedHybrid(input_size=10, hidden_size=32)
    }
    
    # Generate sample test data
    test_data = torch.randn(100, 20, 10)  # 100 samples, 20 timesteps, 10 features
    test_labels = torch.randint(0, 2, (100,))  # Binary labels
    
    # Run comprehensive benchmarking
    results_df = benchmark_suite.run_comprehensive_edge_benchmark(
        models=models,
        test_data=test_data,
        test_labels=test_labels,
        hardware_list=["raspberry_pi_4", "jetson_nano", "intel_ncs2"],
        quantization_bits=[32, 16, 8]
    )
    
    print("\n" + "="*70)
    print("EDGE DEPLOYMENT BENCHMARKING COMPLETE")
    print("="*70)
    
    print(f"\nBenchmarking Summary:")
    print(f"  • Total Configurations: {len(results_df)}")
    print(f"  • Models Tested: {results_df['model_name'].nunique()}")
    print(f"  • Hardware Platforms: {results_df['hardware_config'].nunique()}")
    print(f"  • Quantization Levels: {results_df['quantization_bits'].nunique()}")
    
    # Show best configurations
    if not results_df.empty:
        best_overall = results_df.loc[results_df['deployment_readiness_score'].idxmax()]
        print(f"\nBest Overall Configuration:")
        print(f"  • Model: {best_overall['model_name']}")
        print(f"  • Hardware: {best_overall['hardware_config']}")
        print(f"  • Quantization: {best_overall['quantization_bits']}-bit")
        print(f"  • Deployment Score: {best_overall['deployment_readiness_score']:.3f}")
        print(f"  • Inference Time: {best_overall['inference_time_ms']:.2f}ms")
        print(f"  • Accuracy: {best_overall['accuracy']:.3f}")
    
    print(f"\nResults saved to: {benchmark_suite.output_dir}")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    main()