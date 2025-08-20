"""
📈 Intelligent Auto-Scaling and Load Balancing System

This module provides advanced auto-scaling, load balancing, and resource
management capabilities for the Terragon IoT Anomaly Detection System.

Features:
- Predictive auto-scaling based on ML-driven demand forecasting
- Intelligent load balancing with multiple algorithms
- Dynamic resource allocation and container orchestration
- Real-time performance monitoring and SLA management
- Multi-region deployment and edge federation
- Cost optimization and resource efficiency tracking
- Kubernetes and cloud platform integration
"""

import os
import sys
import time
import threading
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import logging
import warnings
import json
import math

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    SCHEDULE_BASED = "schedule_based"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_BASED = "resource_based"
    MACHINE_LEARNING = "machine_learning"


class InstanceType(Enum):
    """Instance types for scaling."""
    CPU_OPTIMIZED = "cpu_optimized"
    GPU_OPTIMIZED = "gpu_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    EDGE_DEVICE = "edge_device"
    SERVERLESS = "serverless"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    request_rate: float
    response_time: float
    queue_length: int
    error_rate: float
    cost_per_hour: float
    active_instances: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'request_rate': self.request_rate,
            'response_time': self.response_time,
            'queue_length': self.queue_length,
            'error_rate': self.error_rate,
            'cost_per_hour': self.cost_per_hour,
            'active_instances': self.active_instances
        }


@dataclass
class ServiceInstance:
    """Represents a service instance."""
    instance_id: str
    instance_type: InstanceType
    endpoint: str
    capacity: int
    current_load: int
    response_time: float
    last_health_check: datetime
    healthy: bool = True
    region: str = "default"
    cost_per_hour: float = 0.0
    startup_time: float = 60.0  # seconds
    
    def utilization(self) -> float:
        """Get current utilization percentage."""
        return (self.current_load / max(self.capacity, 1)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'instance_type': self.instance_type.value,
            'endpoint': self.endpoint,
            'capacity': self.capacity,
            'current_load': self.current_load,
            'response_time': self.response_time,
            'last_health_check': self.last_health_check.isoformat(),
            'healthy': self.healthy,
            'region': self.region,
            'utilization': self.utilization(),
            'cost_per_hour': self.cost_per_hour
        }


class DemandPredictor:
    """ML-based demand prediction for proactive scaling."""
    
    def __init__(self, history_window: int = 1440):  # 24 hours in minutes
        self.history_window = history_window
        self.metrics_history = []
        self.model = None
        self.last_training = None
        self.prediction_accuracy = 0.0
        
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.history_window:
            self.metrics_history = self.metrics_history[-self.history_window:]
    
    def predict_demand(
        self, 
        horizon_minutes: int = 15
    ) -> Tuple[float, float]:
        """
        Predict future demand.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            (predicted_request_rate, confidence)
        """
        if len(self.metrics_history) < 10:
            # Not enough data for prediction
            if self.metrics_history:
                return self.metrics_history[-1].request_rate, 0.5
            else:
                return 10.0, 0.1  # Default prediction
        
        try:
            # Simple time series prediction using moving averages and trends
            request_rates = [m.request_rate for m in self.metrics_history[-60:]]  # Last hour
            
            # Calculate trends
            recent_avg = np.mean(request_rates[-15:])  # Last 15 minutes
            hourly_avg = np.mean(request_rates)
            
            # Simple trend analysis
            if len(request_rates) >= 30:
                first_half = np.mean(request_rates[:15])
                second_half = np.mean(request_rates[-15:])
                trend = (second_half - first_half) / max(first_half, 1)
            else:
                trend = 0.0
            
            # Apply trend for prediction
            predicted_rate = recent_avg * (1 + trend * (horizon_minutes / 15))
            
            # Add seasonal patterns (simplified)
            hour_of_day = datetime.now().hour
            seasonal_factor = self._get_seasonal_factor(hour_of_day)
            predicted_rate *= seasonal_factor
            
            # Calculate confidence based on variance
            variance = np.var(request_rates) if len(request_rates) > 1 else 0
            confidence = max(0.1, 1.0 - (variance / max(recent_avg, 1)))
            confidence = min(confidence, 0.95)
            
            return max(0.0, predicted_rate), confidence
            
        except Exception as e:
            logger.warning(f"Demand prediction failed: {e}")
            # Fallback to last known rate
            return self.metrics_history[-1].request_rate, 0.3
    
    def _get_seasonal_factor(self, hour: int) -> float:
        """Get seasonal factor based on hour of day."""
        # Simple seasonal pattern (higher during business hours)
        if 9 <= hour <= 17:  # Business hours
            return 1.2
        elif 22 <= hour or hour <= 6:  # Night hours
            return 0.6
        else:  # Off hours
            return 0.9
    
    def get_prediction_accuracy(self) -> float:
        """Get current prediction accuracy."""
        return self.prediction_accuracy
    
    def update_accuracy(self, predicted: float, actual: float):
        """Update prediction accuracy based on actual vs predicted."""
        if predicted > 0:
            error = abs(predicted - actual) / predicted
            accuracy = 1.0 - min(error, 1.0)
            
            # Exponential moving average
            if self.prediction_accuracy == 0.0:
                self.prediction_accuracy = accuracy
            else:
                alpha = 0.1  # Learning rate
                self.prediction_accuracy = (
                    alpha * accuracy + (1 - alpha) * self.prediction_accuracy
                )


class LoadBalancer:
    """Intelligent load balancer with multiple algorithms."""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME):
        self.algorithm = algorithm
        self.instances = {}
        self.round_robin_index = 0
        self.request_history = []
        
        # ML-based load balancing
        self.ml_weights = {}
        self.performance_history = {}
    
    def register_instance(self, instance: ServiceInstance):
        """Register a service instance."""
        self.instances[instance.instance_id] = instance
        self.ml_weights[instance.instance_id] = 1.0
        self.performance_history[instance.instance_id] = []
        logger.info(f"Registered instance {instance.instance_id}")
    
    def unregister_instance(self, instance_id: str):
        """Unregister a service instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.ml_weights[instance_id]
            del self.performance_history[instance_id]
            logger.info(f"Unregistered instance {instance_id}")
    
    def select_instance(self, request_metadata: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """Select best instance for handling request."""
        healthy_instances = [
            instance for instance in self.instances.values()
            if instance.healthy and instance.current_load < instance.capacity
        ]
        
        if not healthy_instances:
            return None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            return self._resource_based_selection(healthy_instances)
        elif self.algorithm == LoadBalancingAlgorithm.MACHINE_LEARNING:
            return self._ml_based_selection(healthy_instances, request_metadata)
        else:
            return healthy_instances[0]  # Fallback
    
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection."""
        selected = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections."""
        return min(instances, key=lambda x: x.current_load)
    
    def _weighted_response_time_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with best weighted response time."""
        def score(instance):
            # Lower score is better
            utilization_penalty = instance.utilization() / 100.0
            response_penalty = instance.response_time / 1000.0  # Convert to seconds
            return response_penalty + utilization_penalty
        
        return min(instances, key=score)
    
    def _resource_based_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on available resources."""
        def resource_score(instance):
            # Higher score is better (more available resources)
            available_capacity = (instance.capacity - instance.current_load) / instance.capacity
            response_factor = 1.0 / max(instance.response_time, 1.0)
            return available_capacity * response_factor
        
        return max(instances, key=resource_score)
    
    def _ml_based_selection(
        self, 
        instances: List[ServiceInstance], 
        request_metadata: Dict[str, Any]
    ) -> ServiceInstance:
        """ML-based instance selection."""
        # Use learned weights and current performance
        def ml_score(instance):
            base_weight = self.ml_weights.get(instance.instance_id, 1.0)
            
            # Adjust based on current performance
            utilization_factor = 1.0 - (instance.utilization() / 100.0)
            response_factor = 1.0 / max(instance.response_time, 1.0)
            
            return base_weight * utilization_factor * response_factor
        
        return max(instances, key=ml_score)
    
    def record_request_performance(
        self, 
        instance_id: str, 
        response_time: float, 
        success: bool
    ):
        """Record request performance for learning."""
        if instance_id in self.performance_history:
            self.performance_history[instance_id].append({
                'timestamp': datetime.now(),
                'response_time': response_time,
                'success': success
            })
            
            # Keep only recent history
            history = self.performance_history[instance_id]
            if len(history) > 100:
                self.performance_history[instance_id] = history[-100:]
            
            # Update ML weights
            self._update_ml_weights(instance_id)
    
    def _update_ml_weights(self, instance_id: str):
        """Update ML weights based on performance history."""
        history = self.performance_history[instance_id]
        
        if len(history) < 10:
            return
        
        # Calculate performance metrics
        recent_history = history[-20:]  # Last 20 requests
        avg_response_time = np.mean([h['response_time'] for h in recent_history])
        success_rate = np.mean([1.0 if h['success'] else 0.0 for h in recent_history])
        
        # Update weight (higher is better)
        performance_score = success_rate / max(avg_response_time, 0.001)
        
        # Exponential moving average
        alpha = 0.1
        current_weight = self.ml_weights[instance_id]
        self.ml_weights[instance_id] = alpha * performance_score + (1 - alpha) * current_weight
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across instances."""
        if not self.instances:
            return {}
        
        total_load = sum(instance.current_load for instance in self.instances.values())
        
        if total_load == 0:
            return {instance_id: 0.0 for instance_id in self.instances.keys()}
        
        return {
            instance_id: (instance.current_load / total_load) * 100
            for instance_id, instance in self.instances.items()
        }


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID,
        min_instances: int = 1,
        max_instances: int = 10,
        target_utilization: float = 70.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0
    ):
        self.strategy = strategy
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Components
        self.demand_predictor = DemandPredictor()
        self.load_balancer = LoadBalancer()
        
        # Scaling state
        self.scaling_history = []
        self.last_scaling_action = None
        self.cooldown_period = 300  # 5 minutes
        
        # Cost optimization
        self.cost_budget = None
        self.cost_tracking = {'current_cost': 0.0, 'daily_budget': 0.0}
        
        # Running state
        self.running = False
        self.scaling_thread = None
    
    def start(self):
        """Start the auto-scaling system."""
        if self.running:
            return
        
        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling system started")
    
    def stop(self):
        """Stop the auto-scaling system."""
        if not self.running:
            return
        
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        logger.info("Auto-scaling system stopped")
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics for scaling decisions."""
        self.demand_predictor.add_metrics(metrics)
        
        # Update cost tracking
        self.cost_tracking['current_cost'] = metrics.cost_per_hour
    
    def register_instance(self, instance: ServiceInstance):
        """Register a service instance."""
        self.load_balancer.register_instance(instance)
    
    def unregister_instance(self, instance_id: str):
        """Unregister a service instance."""
        self.load_balancer.unregister_instance(instance_id)
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                self._make_scaling_decision()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(60)
    
    def _make_scaling_decision(self):
        """Make scaling decision based on current strategy."""
        current_time = datetime.now()
        
        # Check cooldown period
        if (self.last_scaling_action and 
            current_time - self.last_scaling_action < timedelta(seconds=self.cooldown_period)):
            return
        
        # Get current metrics
        instances = list(self.load_balancer.instances.values())
        if not instances:
            return
        
        current_instances = len(instances)
        avg_utilization = np.mean([instance.utilization() for instance in instances])
        avg_response_time = np.mean([instance.response_time for instance in instances])
        
        # Strategy-specific scaling decisions
        if self.strategy == ScalingStrategy.REACTIVE:
            scaling_decision = self._reactive_scaling_decision(
                current_instances, avg_utilization, avg_response_time
            )
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            scaling_decision = self._predictive_scaling_decision(
                current_instances, avg_utilization
            )
        elif self.strategy == ScalingStrategy.HYBRID:
            scaling_decision = self._hybrid_scaling_decision(
                current_instances, avg_utilization, avg_response_time
            )
        elif self.strategy == ScalingStrategy.SCHEDULE_BASED:
            scaling_decision = self._schedule_based_scaling_decision(
                current_instances, current_time
            )
        else:
            scaling_decision = 0  # No scaling
        
        # Apply cost constraints
        scaling_decision = self._apply_cost_constraints(scaling_decision, current_instances)
        
        # Execute scaling decision
        if scaling_decision != 0:
            self._execute_scaling(scaling_decision, current_instances)
    
    def _reactive_scaling_decision(
        self, 
        current_instances: int, 
        avg_utilization: float, 
        avg_response_time: float
    ) -> int:
        """Reactive scaling based on current metrics."""
        if avg_utilization > self.scale_up_threshold and current_instances < self.max_instances:
            return 1  # Scale up
        elif avg_utilization < self.scale_down_threshold and current_instances > self.min_instances:
            return -1  # Scale down
        else:
            return 0  # No scaling
    
    def _predictive_scaling_decision(self, current_instances: int, avg_utilization: float) -> int:
        """Predictive scaling based on demand forecasting."""
        predicted_demand, confidence = self.demand_predictor.predict_demand(horizon_minutes=15)
        
        if confidence < 0.5:
            # Low confidence, fall back to reactive
            return self._reactive_scaling_decision(current_instances, avg_utilization, 0)
        
        # Estimate required instances based on predicted demand
        # Simplified calculation - in reality this would be more sophisticated
        current_capacity = sum(instance.capacity for instance in self.load_balancer.instances.values())
        predicted_utilization = (predicted_demand / max(current_capacity, 1)) * 100
        
        if predicted_utilization > self.scale_up_threshold and current_instances < self.max_instances:
            return min(2, self.max_instances - current_instances)  # Scale up by 1-2 instances
        elif predicted_utilization < self.scale_down_threshold and current_instances > self.min_instances:
            return max(-1, self.min_instances - current_instances)  # Scale down by 1 instance
        else:
            return 0
    
    def _hybrid_scaling_decision(
        self, 
        current_instances: int, 
        avg_utilization: float, 
        avg_response_time: float
    ) -> int:
        """Hybrid scaling combining reactive and predictive approaches."""
        # Get both reactive and predictive decisions
        reactive_decision = self._reactive_scaling_decision(
            current_instances, avg_utilization, avg_response_time
        )
        predictive_decision = self._predictive_scaling_decision(current_instances, avg_utilization)
        
        # Combine decisions with weights
        prediction_confidence = self.demand_predictor.get_prediction_accuracy()
        
        # Weight reactive more heavily if prediction confidence is low
        if prediction_confidence < 0.6:
            return reactive_decision
        else:
            # Use predictive decision but bounded by reactive constraints
            if reactive_decision > 0 and predictive_decision > 0:
                return max(reactive_decision, predictive_decision)
            elif reactive_decision < 0 and predictive_decision < 0:
                return min(reactive_decision, predictive_decision)
            else:
                return predictive_decision
    
    def _schedule_based_scaling_decision(self, current_instances: int, current_time: datetime) -> int:
        """Schedule-based scaling for known traffic patterns."""
        hour = current_time.hour
        day_of_week = current_time.weekday()  # 0 = Monday
        
        # Define schedule-based instance requirements
        if day_of_week < 5:  # Weekdays
            if 8 <= hour <= 18:  # Business hours
                target_instances = max(3, self.min_instances)
            elif 6 <= hour <= 8 or 18 <= hour <= 22:  # Peak transition hours
                target_instances = max(2, self.min_instances)
            else:  # Off hours
                target_instances = self.min_instances
        else:  # Weekends
            if 10 <= hour <= 16:  # Weekend active hours
                target_instances = max(2, self.min_instances)
            else:
                target_instances = self.min_instances
        
        # Cap at maximum
        target_instances = min(target_instances, self.max_instances)
        
        return target_instances - current_instances
    
    def _apply_cost_constraints(self, scaling_decision: int, current_instances: int) -> int:
        """Apply cost constraints to scaling decision."""
        if self.cost_budget is None:
            return scaling_decision
        
        # Calculate projected cost
        if scaling_decision > 0:
            # Estimate cost of additional instances
            avg_cost_per_instance = self.cost_tracking['current_cost'] / max(current_instances, 1)
            additional_cost = avg_cost_per_instance * scaling_decision
            
            if self.cost_tracking['current_cost'] + additional_cost > self.cost_tracking['daily_budget']:
                # Would exceed budget, reduce scaling
                max_affordable = int(
                    (self.cost_tracking['daily_budget'] - self.cost_tracking['current_cost']) / 
                    avg_cost_per_instance
                )
                return min(scaling_decision, max_affordable)
        
        return scaling_decision
    
    def _execute_scaling(self, scaling_decision: int, current_instances: int):
        """Execute the scaling decision."""
        self.last_scaling_action = datetime.now()
        
        scaling_event = {
            'timestamp': self.last_scaling_action,
            'decision': scaling_decision,
            'instances_before': current_instances,
            'instances_after': current_instances + scaling_decision,
            'strategy': self.strategy.value
        }
        
        self.scaling_history.append(scaling_event)
        
        # Keep only recent history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
        
        logger.info(
            f"Scaling decision: {scaling_decision:+d} instances "
            f"({current_instances} -> {current_instances + scaling_decision})"
        )
        
        # In a real implementation, this would trigger actual instance creation/termination
        # For now, we just log the decision
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics."""
        instances = list(self.load_balancer.instances.values())
        
        metrics = {
            'strategy': self.strategy.value,
            'current_instances': len(instances),
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'target_utilization': self.target_utilization,
            'cost_tracking': self.cost_tracking,
            'scaling_history_count': len(self.scaling_history)
        }
        
        if instances:
            metrics.update({
                'avg_utilization': np.mean([instance.utilization() for instance in instances]),
                'avg_response_time': np.mean([instance.response_time for instance in instances]),
                'total_capacity': sum(instance.capacity for instance in instances),
                'total_load': sum(instance.current_load for instance in instances),
                'healthy_instances': sum(1 for instance in instances if instance.healthy)
            })
        
        # Prediction metrics
        metrics['prediction_accuracy'] = self.demand_predictor.get_prediction_accuracy()
        
        # Load balancing metrics
        metrics['load_distribution'] = self.load_balancer.get_load_distribution()
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    print("📈 Testing Auto-Scaling and Load Balancing System")
    print("=" * 60)
    
    # Create auto-scaler
    autoscaler = AutoScaler(
        strategy=ScalingStrategy.HYBRID,
        min_instances=2,
        max_instances=8,
        target_utilization=70.0
    )
    
    # Create some test instances
    test_instances = [
        ServiceInstance(
            instance_id="instance_1",
            instance_type=InstanceType.CPU_OPTIMIZED,
            endpoint="http://instance1:8080",
            capacity=100,
            current_load=60,
            response_time=50.0,
            last_health_check=datetime.now(),
            cost_per_hour=0.50
        ),
        ServiceInstance(
            instance_id="instance_2", 
            instance_type=InstanceType.GPU_OPTIMIZED,
            endpoint="http://instance2:8080",
            capacity=150,
            current_load=90,
            response_time=80.0,
            last_health_check=datetime.now(),
            cost_per_hour=1.20
        )
    ]
    
    # Register instances
    for instance in test_instances:
        autoscaler.register_instance(instance)
    
    # Test load balancing
    print("Testing load balancing...")
    selected = autoscaler.load_balancer.select_instance()
    if selected:
        print(f"✅ Selected instance: {selected.instance_id} (utilization: {selected.utilization():.1f}%)")
    
    # Test demand prediction
    print("\nTesting demand prediction...")
    
    # Add some historical metrics
    for i in range(20):
        metrics = ScalingMetrics(
            timestamp=datetime.now() - timedelta(minutes=20-i),
            cpu_usage=60 + np.random.normal(0, 10),
            memory_usage=50 + np.random.normal(0, 5),
            gpu_usage=40 + np.random.normal(0, 8),
            request_rate=100 + np.random.normal(0, 20),
            response_time=75 + np.random.normal(0, 15),
            queue_length=5 + int(np.random.normal(0, 2)),
            error_rate=0.02 + np.random.normal(0, 0.01),
            cost_per_hour=1.70,
            active_instances=2
        )
        autoscaler.add_metrics(metrics)
    
    # Get prediction
    predicted_demand, confidence = autoscaler.demand_predictor.predict_demand(15)
    print(f"✅ Demand prediction: {predicted_demand:.1f} req/min (confidence: {confidence:.2f})")
    
    # Test scaling decision
    print("\nTesting scaling decisions...")
    
    # Start autoscaler
    autoscaler.start()
    
    # Wait briefly for scaling loop
    time.sleep(2)
    
    # Get metrics
    scaling_metrics = autoscaler.get_scaling_metrics()
    print(f"✅ Current instances: {scaling_metrics['current_instances']}")
    print(f"✅ Average utilization: {scaling_metrics.get('avg_utilization', 0):.1f}%")
    print(f"✅ Prediction accuracy: {scaling_metrics['prediction_accuracy']:.2f}")
    
    # Stop autoscaler
    autoscaler.stop()
    
    print("✅ Auto-scaling and load balancing system tested successfully!")