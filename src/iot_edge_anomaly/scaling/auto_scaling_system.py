"""
Auto-Scaling System for IoT Edge Anomaly Detection.

This module provides intelligent auto-scaling capabilities including:
- Dynamic resource allocation based on load
- Predictive scaling using machine learning
- Multi-tier scaling (vertical and horizontal)
- Edge-cloud hybrid scaling strategies
- Cost optimization and resource efficiency
"""
import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import psutil
import torch
import numpy as np
from collections import deque, defaultdict
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling - add instances
    SCALE_IN = "scale_in"    # Horizontal scaling - remove instances
    NO_ACTION = "no_action"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    INSTANCES = "instances"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    METRIC_THRESHOLD = "metric_threshold"
    PREDICTIVE = "predictive"
    SCHEDULE_BASED = "schedule_based"
    CUSTOM_RULE = "custom_rule"
    EMERGENCY = "emergency"


@dataclass
class ScalingMetric:
    """Represents a metric for scaling decisions."""
    name: str
    current_value: float
    threshold_scale_up: float
    threshold_scale_down: float
    weight: float = 1.0
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    history: List[float] = field(default_factory=list)
    
    def add_value(self, value: float) -> None:
        """Add new value and maintain history."""
        self.current_value = value
        self.history.append(value)
        
        # Keep only recent history (last 100 points)
        if len(self.history) > 100:
            self.history.pop(0)
        
        # Update trend
        if len(self.history) >= 5:
            recent_trend = np.polyfit(range(5), self.history[-5:], 1)[0]
            if recent_trend > 0.1:
                self.trend = "increasing"
            elif recent_trend < -0.1:
                self.trend = "decreasing"
            else:
                self.trend = "stable"
    
    def should_scale_up(self) -> bool:
        """Check if metric indicates scale up needed."""
        return self.current_value > self.threshold_scale_up
    
    def should_scale_down(self) -> bool:
        """Check if metric indicates scale down possible."""
        return self.current_value < self.threshold_scale_down


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    timestamp: datetime
    action: ScalingAction
    resource_type: ResourceType
    trigger: ScalingTrigger
    old_value: Any
    new_value: Any
    metrics: Dict[str, float]
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "trigger": self.trigger.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metrics": self.metrics,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time": self.execution_time
        }


class PredictiveScaler:
    """
    Predictive scaling using machine learning to forecast resource needs.
    
    Features:
    - Time series forecasting for resource utilization
    - Workload pattern recognition
    - Seasonal trend analysis
    - Proactive scaling based on predictions
    """
    
    def __init__(self, prediction_horizon: int = 300):  # 5 minutes ahead
        """Initialize predictive scaler."""
        self.prediction_horizon = prediction_horizon
        self.models = {}  # Model for each metric
        self.scalers = {}  # Scalers for normalization
        self.training_data = defaultdict(lambda: {"timestamps": [], "values": []})
        self.last_training = {}
        self.min_training_samples = 50
        
    def add_data_point(self, metric_name: str, timestamp: float, value: float) -> None:
        """Add new data point for training."""
        data = self.training_data[metric_name]
        data["timestamps"].append(timestamp)
        data["values"].append(value)
        
        # Keep only recent data (last 1000 points)
        if len(data["values"]) > 1000:
            data["timestamps"] = data["timestamps"][-1000:]
            data["values"] = data["values"][-1000:]
    
    def train_model(self, metric_name: str) -> bool:
        """Train prediction model for a specific metric."""
        data = self.training_data[metric_name]
        
        if len(data["values"]) < self.min_training_samples:
            return False
        
        try:
            # Prepare features
            X = self._extract_features(data["timestamps"], data["values"])
            y = np.array(data["values"])
            
            # Scale features
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            X_scaled = self.scalers[metric_name].fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            self.models[metric_name] = model
            self.last_training[metric_name] = time.time()
            
            logger.info(f"Trained prediction model for {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model for {metric_name}: {e}")
            return False
    
    def predict(self, metric_name: str, steps_ahead: int = 1) -> Optional[float]:
        """Predict future value for a metric."""
        if metric_name not in self.models:
            return None
        
        data = self.training_data[metric_name]
        if len(data["values"]) < 10:
            return None
        
        try:
            # Use recent data to predict
            recent_timestamps = data["timestamps"][-10:]
            recent_values = data["values"][-10:]
            
            # Predict next timestamp
            future_timestamp = recent_timestamps[-1] + steps_ahead * 60  # Assuming 1-minute intervals
            
            # Extract features for prediction
            X_pred = self._extract_features([future_timestamp], recent_values[-1:])
            X_pred_scaled = self.scalers[metric_name].transform(X_pred)
            
            prediction = self.models[metric_name].predict(X_pred_scaled)[0]
            
            # Ensure prediction is reasonable
            recent_avg = np.mean(recent_values)
            if abs(prediction - recent_avg) > 3 * np.std(recent_values):
                # Prediction too extreme, use recent average with trend
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                prediction = recent_avg + trend * steps_ahead
            
            return max(0, prediction)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Prediction failed for {metric_name}: {e}")
            return None
    
    def _extract_features(self, timestamps: List[float], values: List[float]) -> np.ndarray:
        """Extract features for machine learning."""
        features = []
        
        for i, timestamp in enumerate(timestamps):
            dt = datetime.fromtimestamp(timestamp)
            
            feature_vector = [
                dt.hour,  # Hour of day
                dt.weekday(),  # Day of week
                dt.minute,  # Minute of hour
                i,  # Sequence position
            ]
            
            # Add statistical features if we have enough historical data
            if i > 0 and len(values) >= i:
                recent_values = values[max(0, i-5):i]
                if recent_values:
                    feature_vector.extend([
                        np.mean(recent_values),
                        np.std(recent_values),
                        np.max(recent_values),
                        np.min(recent_values)
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0])
            else:
                feature_vector.extend([0, 0, 0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def should_retrain(self, metric_name: str) -> bool:
        """Check if model should be retrained."""
        if metric_name not in self.models:
            return True
        
        last_train = self.last_training.get(metric_name, 0)
        
        # Retrain every hour or if we have new data
        return (time.time() - last_train > 3600 or 
                len(self.training_data[metric_name]["values"]) % 50 == 0)


class AutoScalingController:
    """
    Main auto-scaling controller coordinating all scaling decisions.
    
    Features:
    - Multi-metric scaling decisions
    - Predictive and reactive scaling
    - Cost optimization
    - Scaling policy enforcement
    - Emergency scaling handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize auto-scaling controller."""
        self.config = config or {}
        
        # Scaling configuration
        self.min_instances = self.config.get("min_instances", 1)
        self.max_instances = self.config.get("max_instances", 10)
        self.scale_up_cooldown = self.config.get("scale_up_cooldown", 300)  # 5 minutes
        self.scale_down_cooldown = self.config.get("scale_down_cooldown", 600)  # 10 minutes
        
        # Current state
        self.current_instances = self.min_instances
        self.last_scale_action = {}
        
        # Metrics tracking
        self.metrics: Dict[str, ScalingMetric] = {}
        self.scaling_events: List[ScalingEvent] = []
        
        # Predictive scaling
        self.predictive_scaler = PredictiveScaler()
        self.predictive_scaling_enabled = self.config.get("predictive_scaling", True)
        
        # Resource handlers
        self.resource_handlers: Dict[ResourceType, Callable] = {}
        
        # Monitoring
        self.monitoring_interval = self.config.get("monitoring_interval", 60)  # 1 minute
        self.monitoring_task = None
        self.running = False
        
        self._lock = threading.RLock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default scaling metrics."""
        self.metrics = {
            "cpu_utilization": ScalingMetric(
                name="cpu_utilization",
                current_value=0.0,
                threshold_scale_up=70.0,
                threshold_scale_down=30.0,
                weight=1.0
            ),
            "memory_utilization": ScalingMetric(
                name="memory_utilization", 
                current_value=0.0,
                threshold_scale_up=80.0,
                threshold_scale_down=40.0,
                weight=1.0
            ),
            "request_rate": ScalingMetric(
                name="request_rate",
                current_value=0.0,
                threshold_scale_up=100.0,
                threshold_scale_down=20.0,
                weight=1.5  # Higher weight for request rate
            ),
            "response_time": ScalingMetric(
                name="response_time",
                current_value=0.0,
                threshold_scale_up=2000.0,  # 2 seconds
                threshold_scale_down=500.0,  # 0.5 seconds
                weight=1.2
            ),
            "error_rate": ScalingMetric(
                name="error_rate",
                current_value=0.0,
                threshold_scale_up=5.0,  # 5% error rate
                threshold_scale_down=1.0,
                weight=2.0  # High weight for error rate
            )
        }
    
    def register_resource_handler(self, resource_type: ResourceType, handler: Callable) -> None:
        """Register handler for specific resource type."""
        self.resource_handlers[resource_type] = handler
        logger.info(f"Registered handler for {resource_type.value}")
    
    def update_metric(self, metric_name: str, value: float) -> None:
        """Update metric value."""
        with self._lock:
            if metric_name in self.metrics:
                self.metrics[metric_name].add_value(value)
                
                # Add to predictive scaler
                if self.predictive_scaling_enabled:
                    self.predictive_scaler.add_data_point(metric_name, time.time(), value)
    
    def add_custom_metric(self, metric: ScalingMetric) -> None:
        """Add custom scaling metric."""
        with self._lock:
            self.metrics[metric.name] = metric
            logger.info(f"Added custom metric: {metric.name}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring and auto-scaling."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting auto-scaling monitoring")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped auto-scaling monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect current metrics
                await self._collect_system_metrics()
                
                # Make scaling decision
                scaling_decision = await self._make_scaling_decision()
                
                # Execute scaling action if needed
                if scaling_decision.action != ScalingAction.NO_ACTION:
                    await self._execute_scaling_action(scaling_decision)
                
                # Train predictive models if needed
                if self.predictive_scaling_enabled:
                    await self._update_predictive_models()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_metric("cpu_utilization", cpu_percent)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.update_metric("memory_utilization", memory.percent)
            
            # GPU utilization (if available)
            if torch.cuda.is_available():
                gpu_memory_used = 0
                gpu_memory_total = 0
                
                for i in range(torch.cuda.device_count()):
                    gpu_memory_used += torch.cuda.memory_allocated(i)
                    gpu_memory_total += torch.cuda.get_device_properties(i).total_memory
                
                if gpu_memory_total > 0:
                    gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                    self.update_metric("gpu_utilization", gpu_utilization)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _make_scaling_decision(self) -> 'ScalingDecision':
        """Make scaling decision based on current metrics and predictions."""
        with self._lock:
            # Check cooldown periods
            current_time = time.time()
            
            last_scale_up = self.last_scale_action.get(ScalingAction.SCALE_OUT, 0)
            last_scale_down = self.last_scale_action.get(ScalingAction.SCALE_IN, 0)
            
            can_scale_up = (current_time - last_scale_up) > self.scale_up_cooldown
            can_scale_down = (current_time - last_scale_down) > self.scale_down_cooldown
            
            # Calculate scaling scores
            scale_up_score = 0.0
            scale_down_score = 0.0
            
            current_metrics = {}
            
            for metric_name, metric in self.metrics.items():
                current_metrics[metric_name] = metric.current_value
                
                if metric.should_scale_up():
                    excess = (metric.current_value - metric.threshold_scale_up) / metric.threshold_scale_up
                    scale_up_score += excess * metric.weight
                
                if metric.should_scale_down():
                    under = (metric.threshold_scale_down - metric.current_value) / metric.threshold_scale_down
                    scale_down_score += under * metric.weight
            
            # Consider predictive scaling
            if self.predictive_scaling_enabled:
                prediction_boost = await self._get_predictive_scaling_boost()
                scale_up_score += prediction_boost
            
            # Consider trends
            for metric in self.metrics.values():
                if metric.trend == "increasing":
                    scale_up_score += 0.1 * metric.weight
                elif metric.trend == "decreasing":
                    scale_down_score += 0.1 * metric.weight
            
            # Make decision
            decision = ScalingDecision(
                timestamp=datetime.now(),
                current_instances=self.current_instances,
                scale_up_score=scale_up_score,
                scale_down_score=scale_down_score,
                can_scale_up=can_scale_up,
                can_scale_down=can_scale_down,
                metrics=current_metrics
            )
            
            # Emergency scaling (high priority)
            if any(m.current_value > m.threshold_scale_up * 1.5 for m in self.metrics.values()):
                decision.action = ScalingAction.SCALE_OUT
                decision.trigger = ScalingTrigger.EMERGENCY
                decision.target_instances = min(self.max_instances, self.current_instances + 2)
            
            # Normal scaling decisions
            elif scale_up_score > 1.0 and can_scale_up and self.current_instances < self.max_instances:
                decision.action = ScalingAction.SCALE_OUT
                decision.trigger = ScalingTrigger.METRIC_THRESHOLD
                decision.target_instances = min(self.max_instances, self.current_instances + 1)
            
            elif scale_down_score > 1.0 and can_scale_down and self.current_instances > self.min_instances:
                decision.action = ScalingAction.SCALE_IN
                decision.trigger = ScalingTrigger.METRIC_THRESHOLD
                decision.target_instances = max(self.min_instances, self.current_instances - 1)
            
            else:
                decision.action = ScalingAction.NO_ACTION
                decision.trigger = ScalingTrigger.METRIC_THRESHOLD
                decision.target_instances = self.current_instances
            
            return decision
    
    async def _get_predictive_scaling_boost(self) -> float:
        """Get scaling boost based on predictions."""
        boost = 0.0
        
        for metric_name in ["cpu_utilization", "memory_utilization", "request_rate"]:
            if metric_name in self.metrics:
                # Predict 5 minutes ahead
                prediction = self.predictive_scaler.predict(metric_name, steps_ahead=5)
                
                if prediction is not None:
                    metric = self.metrics[metric_name]
                    
                    # If prediction shows metric will exceed threshold, add boost
                    if prediction > metric.threshold_scale_up:
                        excess = (prediction - metric.threshold_scale_up) / metric.threshold_scale_up
                        boost += excess * metric.weight * 0.5  # 50% weight for predictions
        
        return boost
    
    async def _execute_scaling_action(self, decision: 'ScalingDecision') -> None:
        """Execute scaling action."""
        start_time = time.time()
        event_id = f"scale_{int(time.time() * 1000)}"
        
        scaling_event = ScalingEvent(
            event_id=event_id,
            timestamp=decision.timestamp,
            action=decision.action,
            resource_type=ResourceType.INSTANCES,
            trigger=decision.trigger,
            old_value=self.current_instances,
            new_value=decision.target_instances,
            metrics=decision.metrics.copy()
        )
        
        try:
            logger.info(f"Executing scaling action: {decision.action.value} from {self.current_instances} to {decision.target_instances}")
            
            # Execute scaling through resource handler
            if ResourceType.INSTANCES in self.resource_handlers:
                handler = self.resource_handlers[ResourceType.INSTANCES]
                success = await handler(self.current_instances, decision.target_instances)
                
                if success:
                    self.current_instances = decision.target_instances
                    self.last_scale_action[decision.action] = time.time()
                    scaling_event.success = True
                    logger.info(f"Scaling successful: now running {self.current_instances} instances")
                else:
                    scaling_event.success = False
                    scaling_event.error_message = "Resource handler returned failure"
                    logger.error("Scaling failed: resource handler returned failure")
            else:
                # Simulate scaling for testing
                self.current_instances = decision.target_instances
                self.last_scale_action[decision.action] = time.time()
                scaling_event.success = True
                logger.info(f"Scaling simulated: now running {self.current_instances} instances")
            
        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            logger.error(f"Scaling execution failed: {e}")
        
        finally:
            scaling_event.execution_time = time.time() - start_time
            self.scaling_events.append(scaling_event)
            
            # Keep only recent events
            if len(self.scaling_events) > 1000:
                self.scaling_events = self.scaling_events[-1000:]
    
    async def _update_predictive_models(self) -> None:
        """Update predictive models if needed."""
        for metric_name in self.metrics.keys():
            if self.predictive_scaler.should_retrain(metric_name):
                await asyncio.get_event_loop().run_in_executor(
                    None, self.predictive_scaler.train_model, metric_name
                )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        with self._lock:
            recent_events = [e for e in self.scaling_events 
                           if datetime.now() - e.timestamp < timedelta(hours=1)]
            
            successful_events = [e for e in recent_events if e.success]
            failed_events = [e for e in recent_events if not e.success]
            
            metrics_status = {}
            for name, metric in self.metrics.items():
                metrics_status[name] = {
                    "current_value": metric.current_value,
                    "threshold_scale_up": metric.threshold_scale_up,
                    "threshold_scale_down": metric.threshold_scale_down,
                    "trend": metric.trend,
                    "should_scale_up": metric.should_scale_up(),
                    "should_scale_down": metric.should_scale_down()
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "current_instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "running": self.running,
                "metrics": metrics_status,
                "recent_events": {
                    "total": len(recent_events),
                    "successful": len(successful_events),
                    "failed": len(failed_events),
                    "success_rate": len(successful_events) / max(1, len(recent_events))
                },
                "predictive_scaling": {
                    "enabled": self.predictive_scaling_enabled,
                    "trained_models": len(self.predictive_scaler.models)
                }
            }


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    timestamp: datetime
    current_instances: int
    scale_up_score: float
    scale_down_score: float
    can_scale_up: bool
    can_scale_down: bool
    metrics: Dict[str, float]
    action: ScalingAction = ScalingAction.NO_ACTION
    trigger: ScalingTrigger = ScalingTrigger.METRIC_THRESHOLD
    target_instances: int = 0


# Global auto-scaling controller
_auto_scaling_controller: Optional[AutoScalingController] = None


def get_auto_scaling_controller(config: Optional[Dict[str, Any]] = None) -> AutoScalingController:
    """Get or create global auto-scaling controller."""
    global _auto_scaling_controller
    
    if _auto_scaling_controller is None:
        _auto_scaling_controller = AutoScalingController(config)
    
    return _auto_scaling_controller


async def start_auto_scaling(config: Optional[Dict[str, Any]] = None) -> AutoScalingController:
    """Start auto-scaling system."""
    controller = get_auto_scaling_controller(config)
    await controller.start_monitoring()
    return controller


def update_scaling_metric(metric_name: str, value: float) -> None:
    """Update scaling metric value."""
    controller = get_auto_scaling_controller()
    controller.update_metric(metric_name, value)


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status."""
    controller = get_auto_scaling_controller()
    return controller.get_scaling_status()